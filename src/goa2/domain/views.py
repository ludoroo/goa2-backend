"""
Player-Scoped Views - Phase 4 of Client-Readiness Roadmap.

Provides view filtering so players only see what they're allowed to see.
In GoA2, visibility is purely based on card faceup/facedown state,
not team affiliation (allies see enemies' faceup cards too).
"""

from __future__ import annotations

import time
from typing import Any

from goa2.domain.models.base import Turret
from goa2.domain.models.card import Card
from goa2.domain.models.enums import CardState, GamePhase, StatType
from goa2.domain.models.spell import SpellCard
from goa2.domain.models.unit import Hero, HeroPiece, Minion
from goa2.domain.state import GameState
from goa2.domain.time_control import public_clock_view
from goa2.domain.types import BoardEntityID, HeroID


def build_view(
    state: GameState,
    for_hero_id: HeroID | None = None,
    *,
    reveal_all: bool = False,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """
    Build a player-scoped view of the game state.

    Visibility rules:
    - Requesting hero (for_hero_id): Sees all their cards, including facedown ones
    - All other heroes: Only see faceup cards (use card.current_* pattern)
    - Discard piles: Always visible (public information)
    - Board, units, effects, life counters: Always visible (public information)

    Args:
        state: The current game state
        for_hero_id: Hero ID to scope the view to. If None, returns public view
                     (no facedown cards visible to anyone, like a spectator)
        reveal_all: Omniscient mode. If True, every hero is treated as "own" — all
                    hands and facedown cards are revealed. This MUST NEVER be set
                    from live play; it exists only for the offline replay debugger
                    (see server/routes_replays.py), which reconstructs games from
                    disk. Keyword-only and defaulting False so no client-controlled
                    value can ever reach it.

    Returns:
        Serializable dict representation of the game state with filtered cards
    """
    # Build teams view
    teams_view: dict[str, Any] = {}
    for team_color, team in state.teams.items():
        teams_view[team_color.value] = _build_team_view(state, team, for_hero_id, reveal_all)

    # Build board view (public info)
    board_view = _build_board_view(state)

    # Build effects view (public info)
    effects_view = _build_effects_view(state)

    # Build markers view (public info)
    markers_view = _build_markers_view(state)

    # Build tokens view (public info, with facedown hiding)
    tokens_view = _build_tokens_view(state, for_hero_id, reveal_all)

    # Build other public board entities view
    board_entities_view = _build_board_entities_view(state)

    # Build hero pieces view (public info)
    hero_pieces_view = _build_hero_pieces_view(state)

    # Build unresolved cards view (resolution order for frontend)
    unresolved_cards_view = _build_unresolved_cards_view(state, for_hero_id, reveal_all)

    view = {
        "phase": state.phase.value,
        "round": state.round,
        "turn": state.turn,
        "current_actor_id": state.current_actor_id,
        "unresolved_hero_ids": list(state.unresolved_hero_ids),
        "unresolved_cards": unresolved_cards_view,
        # Legacy single-lane field: populated when the game has exactly one
        # lane, null otherwise. New clients should read battle_zones.
        "active_zone_id": (
            next(iter(state.battle_zones.values())) if len(state.battle_zones) == 1 else None
        ),
        "battle_zones": dict(state.battle_zones),
        "wave_counters": dict(state.wave_counters),
        "cheats_enabled": state.cheats_enabled,
        "tie_breaker_team": state.tie_breaker_team.value,
        "teams": teams_view,
        "board": board_view,
        "effects": effects_view,
        "markers": markers_view,
        "tokens": tokens_view,
        "board_entities": board_entities_view,
        "hero_pieces": hero_pieces_view,
        # Physically placed guess cards are public table state while the effect
        # is resolving. This is the authoritative source for the guess tray:
        # it survives reconnects and exposes a card only once it is flipped.
        "card_guess": _build_card_guess_view(state),
        # Direct public hand-card reveal (Cordelia). As with card_guess, the
        # state-backed face survives reconnects and execution-context cleanup.
        "card_reveal": _build_card_reveal_view(state),
    }
    view["time_control"] = (
        state.time_control.model_dump(mode="json") if state.time_control is not None else None
    )
    view["clock"] = (
        public_clock_view(
            state.clock,
            now_ms if now_ms is not None else time.time_ns() // 1_000_000,
        )
        if state.clock is not None
        else None
    )
    return view


def _build_unresolved_cards_view(
    state: GameState, for_hero_id: HeroID | None = None, reveal_all: bool = False
) -> list[dict[str, Any]]:
    """
    Build an ordered list of unresolved cards for frontend visualization.

    Returns cards sorted by resolution priority (highest initiative first),
    with ties broken by tie_breaker_team. Only populated during RESOLUTION phase.
    """
    if state.phase != GamePhase.RESOLUTION:
        return []

    hero_ids = list(state.unresolved_hero_ids)
    if state.current_actor_id:
        hero_ids = [state.current_actor_id, *hero_ids]

    if not hero_ids:
        return []

    from goa2.engine.stats import get_computed_stat

    entries: list[dict[str, Any]] = []
    for h_id in hero_ids:
        hero = state.get_hero(h_id)
        if not hero or not hero.current_turn_card:
            continue

        card = hero.current_turn_card
        base_init = card.get_base_stat_value(StatType.INITIATIVE)
        computed_init = get_computed_stat(state, h_id, StatType.INITIATIVE, base_init)

        entries.append(
            {
                "hero_id": h_id,
                "initiative": computed_init,
                "team": hero.team,
                "card": _build_card_view(card, is_own_hero=(reveal_all or h_id == for_hero_id)),
            }
        )

    # Sort: highest initiative first, tie-breaker team favored among same initiative
    tie_breaker = state.tie_breaker_team
    entries.sort(key=lambda e: (-e["initiative"], 0 if e["team"] == tie_breaker else 1))

    # Remove internal team field before returning
    for entry in entries:
        del entry["team"]

    return entries


def _build_team_view(
    state: GameState, team, for_hero_id: HeroID | None, reveal_all: bool = False
) -> dict[str, Any]:
    """Build a view for a single team."""
    return {
        "color": team.color.value,
        "life_counters": team.life_counters,
        "heroes": [_build_hero_view(state, hero, for_hero_id, reveal_all) for hero in team.heroes],
        "minions": [_build_minion_view(minion) for minion in team.minions],
    }


def _build_hero_view(
    state: GameState, hero: Hero, for_hero_id: HeroID | None, reveal_all: bool = False
) -> dict[str, Any]:
    """
    Build a view for a single hero.

    Visibility:
    - If hero.id == for_hero_id (or reveal_all): Show all cards (hand, deck, played, current_turn)
    - Otherwise: Hand is empty, other card arrays show faceup cards, facedown cards hide sensitive fields
    - Discard pile: Always visible (public info)
    """
    is_own_hero = reveal_all or hero.id == for_hero_id

    # Whether this hero may still commit a second card (or call planning-done)
    # this turn — Emmitt's Alternative Timelines. Secret planning progress, so
    # only ever True for the genuine requesting hero, never for opponents.
    from goa2.engine.phases import planning_open_for_second_card

    can_commit_second_card = hero.id == for_hero_id and planning_open_for_second_card(
        state, hero.id
    )

    # play_card() points current_turn_card at the latest commit. While a
    # two-card hero is still in Planning, expose the first buffered commit in
    # the existing extra_turn_card slot so clients can render both cards.
    # Engine state keeps extra_turn_card reserved for the revealed second card.
    extra_turn_card = hero.extra_turn_card
    if state.phase == GamePhase.PLANNING and hero.id in state.pending_second_cards:
        extra_turn_card = state.pending_inputs.get(hero.id)

    return {
        "id": hero.id,
        "name": hero.name,
        "title": hero.title,
        "team": hero.team.value if hero.team else None,
        "level": hero.level,
        "gold": hero.gold,
        "items": hero.items,
        # Gydion's Wish victory progress is per caster and public.
        "wish_cast_count": hero.wish_cast_count,
        # Rune slots (Snorri): public to all viewers, including opponents/spectators
        "rune_slots": {str(k): v.value for k, v in hero.rune_slots.items()},
        # Hand: Own hero sees all, others see empty array
        "hand": (
            [_build_card_view(card, is_own_hero=True) for card in hero.hand] if is_own_hero else []
        ),
        # Deck: Own hero sees full deck, others see count only
        "deck": (
            [_build_card_view(card, is_own_hero=is_own_hero) for card in hero.deck]
            if is_own_hero
            else {"count": len(hero.deck)}
        ),
        # Prepared spell identities are private to their owner. Spent spells
        # are faceup public information for every viewer.
        "spellbook": (
            (
                [_build_card_view(spell, is_own_hero=True) for spell in hero.spellbook]
                if is_own_hero
                else {"count": len(hero.spellbook)}
            )
            if hero.spells
            else None
        ),
        "cast_spells": [_build_card_view(spell, is_own_hero=True) for spell in hero.cast_spells],
        # Played cards: faceup ones are public; facedown ones are hidden from
        # everyone, the owner included (see _build_card_view_outside_hand)
        "played_cards": [
            _build_card_view_outside_hand(card, is_own_hero=is_own_hero)
            for card in hero.played_cards
        ],
        # Current turn card: Own hero sees all, others see faceup only
        "current_turn_card": (
            _build_card_view(hero.current_turn_card, is_own_hero=is_own_hero)
            if hero.current_turn_card
            else None
        ),
        # Other committed card (Emmitt's Alternative Timelines): the first
        # commit during two-card Planning, then the second revealed card until
        # the retrieve choice.
        "extra_turn_card": (
            _build_card_view(extra_turn_card, is_own_hero=is_own_hero) if extra_turn_card else None
        ),
        # Two-card Planning (Emmitt's Alternative Timelines): True only for the
        # requesting hero while they may still commit a second card or call
        # planning-done. Always False for opponents and outside Planning.
        "can_commit_second_card": can_commit_second_card,
        # Discard pile: public info, except facedown cards (hidden from everyone)
        "discard_pile": [
            _build_card_view_outside_hand(card, is_own_hero=True) for card in hero.discard_pile
        ],
        # Ultimate card: Own hero sees it, others see faceup only
        "ultimate_card": (
            _build_card_view(hero.ultimate_card, is_own_hero=is_own_hero)
            if hero.ultimate_card
            else None
        ),
    }


def _build_minion_view(minion: Minion) -> dict[str, Any]:
    """Build a view for a minion (public info only)."""
    return {
        "id": minion.id,
        "type": minion.type.value,
        "team": minion.team.value if minion.team else None,
        "value": minion.value,  # 2 for MELEE/RANGED, 4 for HEAVY
        "is_heavy": minion.is_heavy,
    }


def _build_card_view_outside_hand(
    card: Card | None, is_own_hero: bool = True
) -> dict[str, Any] | None:
    """Card view for the discard pile and the resolved slots.

    A facedown card there has lost its type, color and actions per the rulebook
    — it is hidden information for EVERY viewer, its owner included (Takahide's
    Bushido puts facedown cards in those areas). Faceup cards render normally.
    """
    if card is None:
        return None
    return _build_card_view(card, is_own_hero=is_own_hero and not card.is_facedown)


def _build_card_view(card: Card | None, is_own_hero: bool = True) -> dict[str, Any] | None:
    """
    Build a view for a single card.

    Args:
        card: The card to view (may be None)
        is_own_hero: If True, show all details even if facedown.
                     If False, show all for faceup, hide sensitive fields for facedown.

    Returns:
        Dict with card details, or None if card is None
    """
    if card is None:
        return None

    if is_own_hero or not card.is_facedown:
        # Own hero or faceup card: show all details
        card_view = {
            "id": card.id,
            "name": card.name,
            "image_id": card.image_id,
            "tier": card.tier.value,
            "color": card.color.value if card.color else None,
            "primary_action": (card.primary_action.value if card.primary_action else None),
            "primary_action_value": card.primary_action_value,
            "secondary_actions": {k.value: v for k, v in card.secondary_actions.items()},
            "effect_id": card.effect_id,
            "effect_text": card.effect_text,
            "initiative": card.initiative,
            "state": card.state.value,
            "is_facedown": card.is_facedown,
            "is_ranged": card.is_ranged,
            "range_value": card.range_value,
            "radius_value": card.radius_value,
            "item": card.item.value if card.item else None,
            "is_active": card.is_active,
        }
        if isinstance(card, SpellCard):
            card_view["spell_rank"] = card.spell_rank
        return card_view
    else:
        # Other hero's facedown card: use current_* pattern and hide sensitive fields
        return {
            "tier": card.current_tier.value,
            "color": card.current_color.value if card.current_color else None,
            "primary_action": (
                card.current_primary_action.value if card.current_primary_action else None
            ),
            "primary_action_value": card.current_primary_action_value,
            "secondary_actions": {k.value: v for k, v in card.current_secondary_actions.items()},
            "effect_id": card.current_effect_id,
            "effect_text": card.current_effect_text,
            "initiative": card.current_initiative,
            "state": card.state.value,
            "is_facedown": card.is_facedown,
            # The item stat is printed on the hidden card face and can narrow
            # its identity. Keep the response shape stable but mask the value.
            "item": None,
            "is_active": card.is_active,
        }


def _build_revealed_card_view(card: Card) -> dict[str, Any]:
    """Build the complete public face of a card revealed by an effect.

    This does not mutate the card's real zone or face state. It is deliberately
    separate from normal player-scoped views: a color guess briefly makes one
    otherwise-hidden hand card public, while the rest of that hand stays
    private.
    """
    card_view = _build_card_view(card, is_own_hero=True)
    if card_view is None:  # pragma: no cover - Card is non-optional here
        raise ValueError("Cannot reveal a missing card")
    return {**card_view, "is_facedown": False}


def _build_card_guess_view(state: GameState) -> dict[str, Any] | None:
    """Public table state for a card-color guess, or None.

    Reads ``state.card_guess``, which the guess steps maintain and which
    outlives the turn's execution_context, so the final reveal is still here
    when the post-mutation view is built. The card face is resolved at view
    time rather than snapshotted, so it stays accurate whether a wrong guess
    left the card in hand or a correct one moved it to the discard pile.
    """
    guess = state.card_guess
    if not guess or not guess.get("attempts"):
        return None

    attempts: list[dict[str, Any]] = []
    for entry in guess["attempts"]:
        revealed = entry.get("correct") is not None
        card = None
        if revealed:
            victim = state.get_hero(HeroID(str(entry["victim_id"])))
            if victim is not None:
                card = next(
                    (
                        candidate
                        for candidate in [*victim.hand, *victim.discard_pile]
                        if candidate.id == entry["card_id"]
                    ),
                    None,
                )
        attempts.append(
            {
                "attempt": entry["attempt"],
                "victim_id": entry["victim_id"],
                "card": _build_revealed_card_view(card) if card else None,
                "guessed_color": entry.get("guessed_color"),
                "actual_color": entry.get("actual_color"),
                "correct": entry.get("correct"),
            }
        )

    return {"guesser_id": guess["guesser_id"], "attempts": attempts}


def _build_card_reveal_view(state: GameState) -> dict[str, Any] | None:
    """Complete public face for an intentionally revealed hand card."""
    reveal = state.card_reveal
    if not reveal:
        return None

    owner = state.get_hero(HeroID(str(reveal["owner_id"])))
    if owner is None:
        return None
    card = next(
        (
            candidate
            for candidate in [
                *owner.hand,
                *owner.discard_pile,
                *[played for played in owner.played_cards if played is not None],
            ]
            if candidate.id == reveal["card_id"]
        ),
        None,
    )
    if card is None:
        return None

    return {
        "revealer_id": reveal["revealer_id"],
        "target_unit_id": reveal["target_unit_id"],
        "owner_id": reveal["owner_id"],
        "card": _build_revealed_card_view(card),
        "tier_value": reveal["tier_value"],
        "discarded": card.state == CardState.DISCARD,
    }


def _build_board_view(state: GameState) -> dict[str, Any]:
    """Build a view of the board (public info)."""
    # Get all tiles with occupant info
    tiles_view = {}
    for hex_obj, tile in state.board.tiles.items():
        tile_id = f"{hex_obj.q}_{hex_obj.r}_{hex_obj.s}"
        tile_data = {
            "hex": {"q": hex_obj.q, "r": hex_obj.r, "s": hex_obj.s},
            "zone_id": tile.zone_id,
            "is_terrain": tile.is_terrain,
            "occupant_id": tile.occupant_id,
            "spawn_point": (
                {
                    "location": {
                        "q": tile.spawn_point.location.q,
                        "r": tile.spawn_point.location.r,
                        "s": tile.spawn_point.location.s,
                    },
                    "team": tile.spawn_point.team.value,
                    "type": tile.spawn_point.type.value,
                    "minion_type": (
                        tile.spawn_point.minion_type.value if tile.spawn_point.minion_type else None
                    ),
                }
                if tile.spawn_point
                else None
            ),
        }
        tiles_view[tile_id] = tile_data

    # Get zone info
    zones_view = {}
    for zone in state.board.zones.values():
        zones_view[zone.id] = {
            "id": zone.id,
            "neighbors": zone.neighbors,
            "spawn_points": [
                {
                    "location": {
                        "q": sp.location.q,
                        "r": sp.location.r,
                        "s": sp.location.s,
                    },
                    "team": sp.team.value,
                    "type": sp.type.value,
                    "minion_type": sp.minion_type.value if sp.minion_type else None,
                }
                for sp in zone.spawn_points
            ],
        }

    # Get entity locations
    entity_locations = {
        entity_id: {"q": h.q, "r": h.r, "s": h.s} for entity_id, h in state.entity_locations.items()
    }

    return {
        "map": state.board.map_id,
        "tiles": tiles_view,
        "zones": zones_view,
        "entity_locations": entity_locations,
    }


def _build_effects_view(state: GameState) -> list[dict[str, Any]]:
    """Build a view of active effects (public info)."""
    effects_view = []

    for effect in state.active_effects:
        origin_hex = effect.scope.origin_hex
        effect_view = {
            "id": effect.id,
            "type": effect.effect_type.value,
            "source_card_id": effect.source_card_id,
            "duration": effect.duration.value,
            "is_active": effect.is_active,
            "scope": {
                "shape": effect.scope.shape.value,
                "range": effect.scope.range,
                "origin_id": effect.scope.origin_id,
                "origin": (
                    {"q": origin_hex.q, "r": origin_hex.r, "s": origin_hex.s}
                    if origin_hex
                    else None
                ),
                "affects": effect.scope.affects.value,
            },
            "stat_type": effect.stat_type.value if effect.stat_type else None,
            "stat_value": effect.stat_value,
            # NebKher reality splits: the line of hexes where this cube
            # coordinate equals split_value, fixed at cast time.
            "split_axis": effect.split_axis,
            "split_value": effect.split_value,
            # Publicly announced card color (Imbue Doubt family).
            "named_color": effect.named_color.value if effect.named_color else None,
        }
        effects_view.append(effect_view)

    return effects_view


def _build_tokens_view(
    state: GameState, for_hero_id: HeroID | None = None, reveal_all: bool = False
) -> list[dict[str, Any]]:
    """Build a view of placed tokens with facedown identities hidden."""
    placed_tokens = sorted(
        (
            token
            for tokens in state.token_pool.values()
            for token in tokens
            if BoardEntityID(str(token.id)) in state.entity_locations
        ),
        key=lambda token: str(token.id),
    )

    tokens_view = []
    for token in placed_tokens:
        loc = state.entity_locations[BoardEntityID(str(token.id))]

        visible_type = token.token_type.value
        owner_is_viewer = for_hero_id is not None and token.owner_id == for_hero_id
        if token.is_facedown and not reveal_all and not owner_is_viewer:
            visible_type = "mine"

        is_hidden = token.is_facedown and visible_type == "mine"
        tokens_view.append(
            {
                "id": str(token.id),
                "name": "Mine" if is_hidden else token.name,
                "token_type": visible_type,
                "owner_id": str(token.owner_id) if token.owner_id else None,
                "is_facedown": token.is_facedown,
                "is_passable": token.is_passable,
                "hex": {"q": loc.q, "r": loc.r, "s": loc.s},
            }
        )
    return tokens_view


def _build_board_entities_view(state: GameState) -> list[dict[str, Any]]:
    """Build a view of non-token, non-unit board entities."""
    entities_view = []
    for entity_id, entity in state.misc_entities.items():
        if not isinstance(entity, Turret):
            continue

        loc = state.entity_locations.get(BoardEntityID(str(entity_id)))
        entities_view.append(
            {
                "id": str(entity.id),
                "name": entity.name,
                "entity_kind": entity.entity_kind,
                "owner_id": entity.owner_id,
                "is_obstacle": entity.is_obstacle,
                "hex": ({"q": loc.q, "r": loc.r, "s": loc.s} if loc else None),
            }
        )
    return entities_view


def _build_hero_pieces_view(state: GameState) -> dict[str, dict[str, Any]]:
    """Build public metadata for multi-piece hero pieces."""
    pieces_view: dict[str, dict[str, Any]] = {}
    for entity_id, entity in state.misc_entities.items():
        if not isinstance(entity, HeroPiece):
            continue
        loc = state.entity_locations.get(BoardEntityID(str(entity_id)))
        pieces_view[str(entity_id)] = {
            "owner_hero_id": entity.owner_hero_id,
            "team": entity.team.value if entity.team else None,
            "position": ({"q": loc.q, "r": loc.r, "s": loc.s} if loc else None),
        }
    return pieces_view


def _build_markers_view(state: GameState) -> dict[str, Any]:
    """Build a view of placed markers (public info)."""
    markers_view = {}

    for marker_type, marker in state.markers.items():
        markers_view[marker_type.value] = {
            "target_id": marker.target_id,
            "value": marker.value,
            "source_id": marker.source_id,
        }

    return markers_view
