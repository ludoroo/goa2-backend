"""Determinization for ISMCTS.

At a planning decision, both upgraded loadouts and simultaneous face-down
commitments can be hidden. ``determinize`` samples both from static hero data and
public facts, without consulting a non-owner's actual upgraded hand.
"""

from __future__ import annotations

import random
from typing import cast

from goa2.domain.card_knowledge import (
    LoadoutHypothesis,
    enumerate_static_loadout_hypotheses,
    has_standard_loadout_provenance,
)
from goa2.domain.models import CardColor, CardState, CardTier, GamePhase
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.phases import commit_card, uncommit_card

from .clone import clone_state


def determinize(state: GameState, perspective_hero_id: str, rng: random.Random) -> GameState:
    """Return a clone with commitments hidden from one hero resampled.

    Only valid during PLANNING (the phase with hidden simultaneous commits).
    Outside PLANNING it is just an independent clone.
    """
    clone = clone_state(state)
    if clone.phase != GamePhase.PLANNING:
        return clone

    perspective = clone.get_hero(HeroID(perspective_hero_id))
    if perspective is None:
        raise ValueError(f"unknown perspective hero {perspective_hero_id!r}")

    # Keep one planning slot temporarily open. Re-committing the last hidden
    # card must not run revelation on a clone that was captured with every
    # planner already represented in the pending buffers.
    perspective_hid = HeroID(perspective_hero_id)
    perspective_was_pending = perspective_hid in clone.pending_inputs
    perspective_pending = clone.pending_inputs.pop(perspective_hid, None)
    try:
        for team in clone.teams.values():
            for hero in team.heroes:
                if hero.id == perspective_hero_id:
                    continue
                _determinize_hero(clone, hero, rng)
    finally:
        if perspective_was_pending:
            clone.pending_inputs[perspective_hid] = perspective_pending
    return clone


def _determinize_hero(state: GameState, hero: Hero, rng: random.Random) -> None:
    """Sample one loadout, then replace facedown cards from that sampled hand."""
    hid = HeroID(hero.id)
    first = state.pending_inputs.get(hid)
    second = state.pending_second_cards.get(hid)
    committed = [card for card in (first, second) if card is not None]
    was_done = hid in state.planning_done

    # Use the engine's canonical LIFO take-back path so card lifecycle flags,
    # current_turn_card, pending buffers, and planning_done stay coherent.
    for _ in committed:
        uncommit_card(state, hid)

    revealed_ids = tuple(
        sorted(
            record.card_id for record in state.public_revealed_cards if record.hero_id == hero.id
        )
    )
    if has_standard_loadout_provenance(hero):
        hypotheses = enumerate_static_loadout_hypotheses(
            hero,
            hero.level,
            public_items=hero.items,
            revealed_card_ids=revealed_ids,
        )
        if hypotheses == ():
            # Aggregate items can include effect-derived extras. Prefer paths
            # whose ordinary card items fit within that public aggregate.
            hypotheses = enumerate_static_loadout_hypotheses(
                hero,
                hero.level,
                public_items=hero.items,
                revealed_card_ids=revealed_ids,
                allow_extra_public_items=True,
            )
        if hypotheses == ():
            hypotheses = enumerate_static_loadout_hypotheses(
                hero, hero.level, revealed_card_ids=revealed_ids
            )
    else:
        # These heroes can swap cards or derive items through nonstandard rules;
        # historical reveals and item provenance do not constrain current cards.
        hypotheses = enumerate_static_loadout_hypotheses(hero, hero.level)
    if hypotheses:
        _apply_loadout_hypothesis(hero, rng.choice(hypotheses))
    # Unsupported static structures or contradictory public reveals fail closed:
    # preserve the clone's lifecycle. Commitment sampling below then necessarily
    # uses that hero's existing hand because no sound static policy is available.

    for _ in committed:
        card = rng.choice(list(hero.hand))
        commit_card(state, hid, card)

    if was_done and hid not in state.planning_done:
        state.planning_done.append(hid)


_ORDINARY_COLORS = {CardColor.RED, CardColor.BLUE, CardColor.GREEN}
_TIER_NUMBER = {CardTier.I: 1, CardTier.II: 2, CardTier.III: 3}


def _apply_loadout_hypothesis(hero: Hero, hypothesis: LoadoutHypothesis) -> None:
    """Rebuild ordinary card references while retaining special-card lifecycle."""
    ordinary = [
        card for card in hero.deck if card.color in _ORDINARY_COLORS and card.tier in _TIER_NUMBER
    ]
    by_id = {str(card.id): card for card in ordinary}
    upgraded_ids = set(hypothesis.active_card_ids)
    item_ids = set(hypothesis.item_card_ids)
    upgraded_colors = {by_id[card_id].color for card_id in upgraded_ids}
    active_ids = upgraded_ids | {
        str(card.id)
        for card in ordinary
        if card.tier is CardTier.I and card.color not in upgraded_colors
    }
    active_tiers = {
        cast(CardColor, by_id[card_id].color): _TIER_NUMBER[by_id[card_id].tier]
        for card_id in active_ids
    }

    played_positions = {
        str(card.id): index
        for index, card in enumerate(hero.played_cards)
        if card is not None and str(card.id) in active_ids
    }
    discarded_ids = {str(card.id) for card in hero.discard_pile if str(card.id) in active_ids}

    ordinary_ids = set(by_id)
    hero.hand = [card for card in hero.hand if str(card.id) not in ordinary_ids]
    hero.discard_pile = [card for card in hero.discard_pile if str(card.id) not in ordinary_ids]
    hero.played_cards = [
        None if card is not None and str(card.id) in ordinary_ids else card
        for card in hero.played_cards
    ]
    if hero.current_turn_card is not None and str(hero.current_turn_card.id) in ordinary_ids:
        hero.current_turn_card = None
    if hero.extra_turn_card is not None and str(hero.extra_turn_card.id) in ordinary_ids:
        hero.extra_turn_card = None

    for card in ordinary:
        card_id = str(card.id)
        card.is_facedown = False
        card.played_this_round = False
        if card_id in item_ids:
            card.state = CardState.ITEM
        elif card_id in active_ids:
            if card_id in played_positions:
                card.state = CardState.RESOLVED
                card.played_this_round = True
                hero.played_cards[played_positions[card_id]] = card
            elif card_id in discarded_ids:
                card.state = CardState.DISCARD
                card.played_this_round = True
                hero.discard_pile.append(card)
            else:
                card.state = CardState.HAND
                hero.hand.append(card)
        elif _TIER_NUMBER[card.tier] < active_tiers[cast(CardColor, card.color)]:
            card.state = CardState.RETIRED
        else:
            card.state = CardState.DECK
