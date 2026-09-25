"""Conservative player-scoped knowledge about hero card loadouts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, SkipValidation

from goa2.data.heroes.registry import HeroRegistry
from goa2.domain.models import Card, CardColor, CardState, CardTier, Hero, StatType
from goa2.domain.state import GameState


class CardKnowledgeStatus(StrEnum):
    EXACT = "EXACT"
    INFERRED = "INFERRED"
    UNAVAILABLE = "UNAVAILABLE"
    INCONSISTENT = "INCONSISTENT"


class LoadoutHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    active_card_ids: tuple[str, ...] = ()
    item_card_ids: tuple[str, ...] = ()


class HeroCardKnowledge(BaseModel):
    model_config = ConfigDict(frozen=True)

    starting_card_ids: tuple[str, ...]
    revealed_card_ids: tuple[str, ...]
    committed_card_ids: tuple[str, ...] | None
    status: CardKnowledgeStatus
    active_upgraded_card_ids: tuple[str, ...] | None
    loadout_hypotheses: tuple[LoadoutHypothesis, ...] = ()


class PublicCardKnowledge(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    heroes: SkipValidation[Mapping[str, HeroCardKnowledge]]


_UPGRADE_COLORS = {CardColor.RED, CardColor.BLUE, CardColor.GREEN}
_TIER_NUMBER = {CardTier.I: 1, CardTier.II: 2, CardTier.III: 3}
_NUMBER_TIER = {2: CardTier.II, 3: CardTier.III}
_NONSTANDARD_HEROES = {"Min", "Dodger", "Snorri"}


def _definition(hero: Hero) -> Hero:
    definition = HeroRegistry.get(hero.name)
    return definition if definition is not None else hero


def _starting_ids(definition: Hero) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(card.id)
            for card in definition.deck
            if card.tier in {CardTier.UNTIERED, CardTier.I} and not card.starts_in_deck
        )
    )


def _item_counts(cards: tuple[Card, ...]) -> Counter[StatType]:
    return Counter(card.item for card in cards if card.item is not None)


def _ordinary_hypotheses(
    definition: Hero,
    level: int,
    public_items: Mapping[StatType, int] | None,
    allow_extra_public_items: bool = False,
) -> tuple[LoadoutHypothesis, ...] | None:
    """Enumerate legal ordinary upgrades from static same-color tier pairs."""
    if level < 1 or level > 8:
        return None
    upgrades = min(level - 1, 6)

    active = {
        CardColor(card.color): card
        for card in definition.deck
        if card.color in _UPGRADE_COLORS and card.tier == CardTier.I and not card.starts_in_deck
    }
    if len(active) != 3:
        return None

    states: list[tuple[dict[CardColor, Card], tuple[Card, ...]]] = [(active, ())]
    for _ in range(upgrades):
        next_states: list[tuple[dict[CardColor, Card], tuple[Card, ...]]] = []
        for current, items in states:
            minimum = min(_TIER_NUMBER[card.tier] for card in current.values())
            target_tier = _NUMBER_TIER.get(minimum + 1)
            if target_tier is None:
                continue
            for color, old_card in current.items():
                if _TIER_NUMBER[old_card.tier] != minimum:
                    continue
                pair = [
                    card
                    for card in definition.deck
                    if card.color == color and card.tier == target_tier
                ]
                if len(pair) != 2:
                    continue
                for chosen, item in ((pair[0], pair[1]), (pair[1], pair[0])):
                    updated = dict(current)
                    updated[color] = chosen
                    next_states.append((updated, (*items, item)))
        states = next_states
        if not states:
            return None

    wanted = (
        Counter({stat: count for stat, count in public_items.items() if count})
        if public_items is not None
        else None
    )
    results = {
        (
            tuple(sorted(str(card.id) for card in current.values() if card.tier != CardTier.I)),
            tuple(sorted(str(card.id) for card in items)),
        )
        for current, items in states
        if wanted is None
        or _item_counts(items) == wanted
        or (
            allow_extra_public_items
            and all(count <= wanted[stat] for stat, count in _item_counts(items).items())
        )
    }
    return tuple(
        LoadoutHypothesis(active_card_ids=active_ids, item_card_ids=item_ids)
        for active_ids, item_ids in sorted(results)
    )


def _matches_reveals(
    hypothesis: LoadoutHypothesis, definition: Hero, revealed_ids: tuple[str, ...]
) -> bool:
    """Check upgraded reveals against the upgrade path encoded by a loadout."""
    cards = {str(card.id): card for card in definition.deck}
    item_ids = set(hypothesis.item_card_ids)
    active_ids = set(hypothesis.active_card_ids)
    for card_id in revealed_ids:
        card = cards.get(card_id)
        if card is None or card.tier not in {CardTier.II, CardTier.III}:
            continue
        if card.tier == CardTier.III:
            if card_id not in active_ids:
                return False
            continue
        paired_ids = {
            str(candidate.id)
            for candidate in definition.deck
            if candidate.color == card.color and candidate.tier == CardTier.II
        }
        if not (paired_ids - {card_id}) <= item_ids:
            return False
    return True


def enumerate_static_loadout_hypotheses(
    hero: Hero,
    level: int,
    *,
    public_items: Mapping[StatType, int] | None = None,
    revealed_card_ids: tuple[str, ...] = (),
    allow_extra_public_items: bool = False,
) -> tuple[LoadoutHypothesis, ...] | None:
    """Enumerate ordinary level-legal loadouts using only static/public facts.

    ``None`` means the hero does not follow the ordinary upgrade structure.
    Public aggregates can legitimately contain effect-derived extras. Set
    ``allow_extra_public_items`` to accept card-item multisets that are subsets
    of that aggregate. Omitting ``public_items`` avoids item conditioning.
    """
    definition = _definition(hero)
    hypotheses = _ordinary_hypotheses(definition, level, public_items, allow_extra_public_items)
    if hypotheses is None:
        return None
    return tuple(
        hypothesis
        for hypothesis in hypotheses
        if _matches_reveals(hypothesis, definition, revealed_card_ids)
    )


def has_standard_loadout_provenance(hero: Hero) -> bool:
    """Whether public items/reveals safely identify this hero's current loadout."""
    return _definition(hero).name not in _NONSTANDARD_HEROES


def _owner_active_upgrades(hero: Hero) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(card.id)
            for card in hero.deck
            if card.tier in {CardTier.II, CardTier.III}
            and card.state not in {CardState.DECK, CardState.ITEM, CardState.RETIRED}
        )
    )


def _owner_commitments(state: GameState, hero: Hero) -> tuple[str, ...]:
    cards = []
    first = state.pending_inputs.get(hero.id)
    if first is not None:
        cards.append(str(first.id))
    second = state.pending_second_cards.get(hero.id)
    if second is not None:
        cards.append(str(second.id))
    return tuple(cards)


def build_public_card_knowledge(
    state: GameState, viewer_hero_id: str | None
) -> PublicCardKnowledge:
    """Build immutable knowledge without exposing private card identities."""
    by_hero: dict[str, HeroCardKnowledge] = {}
    revealed: dict[str, set[str]] = {}
    for record in state.public_revealed_cards:
        revealed.setdefault(record.hero_id, set()).add(record.card_id)

    for team in state.teams.values():
        for hero in team.heroes:
            hero_id = str(hero.id)
            definition = _definition(hero)
            is_owner = viewer_hero_id == hero_id
            starting_ids = _starting_ids(definition)
            revealed_ids = tuple(sorted(revealed.get(hero_id, set())))
            committed_ids = _owner_commitments(state, hero) if is_owner else None
            if is_owner and hero.id not in state.pending_upgrades:
                by_hero[hero_id] = HeroCardKnowledge(
                    starting_card_ids=starting_ids,
                    revealed_card_ids=revealed_ids,
                    committed_card_ids=committed_ids,
                    status=CardKnowledgeStatus.EXACT,
                    active_upgraded_card_ids=_owner_active_upgrades(hero),
                )
                continue

            if definition.name in _NONSTANDARD_HEROES or (
                is_owner and hero.id in state.pending_upgrades
            ):
                by_hero[hero_id] = HeroCardKnowledge(
                    starting_card_ids=starting_ids,
                    revealed_card_ids=revealed_ids,
                    committed_card_ids=committed_ids,
                    status=CardKnowledgeStatus.UNAVAILABLE,
                    active_upgraded_card_ids=None,
                )
                continue

            hypotheses = enumerate_static_loadout_hypotheses(
                definition,
                hero.level,
                public_items=hero.items,
                revealed_card_ids=revealed_ids,
            )
            status = (
                CardKnowledgeStatus.UNAVAILABLE
                if hypotheses is None
                else (
                    CardKnowledgeStatus.INFERRED if hypotheses else CardKnowledgeStatus.INCONSISTENT
                )
            )
            by_hero[hero_id] = HeroCardKnowledge(
                starting_card_ids=starting_ids,
                revealed_card_ids=revealed_ids,
                committed_card_ids=committed_ids,
                status=status,
                active_upgraded_card_ids=None,
                loadout_hypotheses=hypotheses or (),
            )

    return PublicCardKnowledge(heroes=MappingProxyType(by_hero))
