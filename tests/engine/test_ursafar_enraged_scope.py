"""Who "enraged" belongs to (Ursafar x NebKher's Mind Grip).

Locked interpretation (2026-08-08):
- Enraged is a per-hero state owned by whoever PERFORMED the card that said
  "This round: You are enraged" — not by the card's owner. That is exactly the
  ``ActiveEffect.source_id`` / ``source_card_id`` split EffectManager already
  writes (see the field docstring in domain/models/effect.py).
- So NebKher performing an Ursafar rage card via Mind Grip becomes enraged
  himself, while the effect stays bound to the Ursafar card he performed.
  Ursafar does NOT become enraged from it.
- Ursafar's ultimate (Unbound Fury) still enrages him unconditionally, and it
  is HIS ultimate — no other hero's ultimate grants rage at level 8.
"""

from __future__ import annotations

import goa2.scripts.ursafar_effects  # noqa: F401  (registers the effects)
from goa2.data.heroes.registry import HeroRegistry
from goa2.domain.board import Board, Zone
from goa2.domain.hex import Hex
from goa2.domain.models import (
    Card,
    CardState,
    GamePhase,
    Hero,
    Team,
    TeamColor,
)
from goa2.domain.models.effect import (
    ActiveEffect,
    AffectsFilter,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.handler import process_stack, push_steps, submit_input
from goa2.engine.steps import PerformCardActionStep
from goa2.scripts.ursafar_effects import is_enraged


def _ursafar_card(card_id: str) -> Card:
    hero = HeroRegistry.get("Ursafar")
    assert hero is not None
    card = next((c for c in hero.deck if c.id == card_id), None)
    assert card is not None, f"{card_id} not in Ursafar's deck"
    return card


def _state() -> GameState:
    board = Board()
    hexes = {Hex(q=q, r=0, s=-q) for q in range(6)}
    board.zones = {"z1": Zone(id="z1", hexes=hexes, neighbors=[])}
    board.populate_tiles_from_zones()

    ursafar = Hero(id=HeroID("hero_ursafar"), name="Ursafar", team=TeamColor.RED, deck=[], level=1)
    nebkher = Hero(id=HeroID("hero_nebkher"), name="NebKher", team=TeamColor.BLUE, deck=[], level=1)
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[ursafar], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[nebkher], minions=[]),
        },
    )
    state.phase = GamePhase.RESOLUTION
    state.place_entity("hero_ursafar", Hex(q=0, r=0, s=0))
    state.place_entity("hero_nebkher", Hex(q=2, r=0, s=-2))
    state.current_actor_id = "hero_nebkher"
    return state


def _enraged_effect(source_id: str, source_card_id: str | None = None) -> ActiveEffect:
    return ActiveEffect(
        id=f"eff_test_{source_id}",
        source_id=source_id,
        source_card_id=source_card_id,
        effect_type=EffectType.ENRAGED,
        scope=EffectScope(shape=Shape.POINT, affects=AffectsFilter.SELF, origin_id=source_id),
        duration=DurationType.THIS_ROUND,
        created_at_turn=1,
        created_at_round=1,
        is_active=True,
    )


# =============================================================================
# Ownership of the enraged state
# =============================================================================


def test_hero_with_own_active_enraged_effect_is_enraged() -> None:
    state = _state()
    state.add_effect(_enraged_effect("hero_ursafar", "cold_ire"))

    assert is_enraged(state, state.get_hero(HeroID("hero_ursafar"))) is True


def test_enraged_effect_belongs_to_its_performer_not_the_card_owner() -> None:
    """Mind Grip: NebKher performed Ursafar's card, so the rage is NebKher's."""
    state = _state()
    ursafar = state.get_hero(HeroID("hero_ursafar"))
    rage_card = _ursafar_card("cold_ire")
    rage_card.state = CardState.RESOLVED
    # EffectManager marks the *card* active even though the effect is the
    # performer's — that mark must no longer enrage the card's owner.
    rage_card.is_active = True
    ursafar.played_cards = [rage_card]
    state.add_effect(_enraged_effect("hero_nebkher", rage_card.id))

    assert is_enraged(state, state.get_hero(HeroID("hero_nebkher"))) is True
    assert is_enraged(state, ursafar) is False


def test_dormant_enraged_effect_does_not_enrage_yet() -> None:
    """A card played but not yet resolved has not turned its rage on."""
    state = _state()
    dormant = _enraged_effect("hero_ursafar", "cold_ire")
    dormant.is_active = False
    state.add_effect(dormant)

    assert is_enraged(state, state.get_hero(HeroID("hero_ursafar"))) is False


# =============================================================================
# The ultimate is Ursafar's alone
# =============================================================================


def test_ursafar_ultimate_enrages_him() -> None:
    state = _state()
    ursafar = state.get_hero(HeroID("hero_ursafar"))
    registry_ursafar = HeroRegistry.get("Ursafar")
    ursafar.ultimate_card = registry_ursafar.ultimate_card
    ursafar.level = 8

    assert is_enraged(state, ursafar) is True


def test_another_heros_level_8_ultimate_does_not_enrage() -> None:
    """The level-8 leak: any hero with any ultimate used to read as enraged."""
    state = _state()
    nebkher = state.get_hero(HeroID("hero_nebkher"))
    registry_nebkher = HeroRegistry.get("NebKher")
    nebkher.ultimate_card = registry_nebkher.ultimate_card
    nebkher.level = 8

    assert is_enraged(state, nebkher) is False


# =============================================================================
# End to end through Mind Grip's PerformCardActionStep
# =============================================================================


def test_mind_grip_on_ursafar_rage_card_enrages_nebkher_only() -> None:
    state = _state()
    ursafar = state.get_hero(HeroID("hero_ursafar"))
    # Angry Roar's only unconditional step is "This round: You are enraged."
    roar = _ursafar_card("angry_roar")
    roar.state = CardState.RESOLVED
    roar.is_facedown = False
    ursafar.played_cards = [roar]
    nebkher = state.get_hero(HeroID("hero_nebkher"))
    nebkher.resolved_turn_count = 1

    state.execution_context["mg_target_hero"] = "hero_ursafar"
    push_steps(
        state,
        [
            PerformCardActionStep(
                card_owner_key="mg_target_hero",
                previous_slot=True,
                hero_id="hero_nebkher",
                skip_markers=True,
            )
        ],
    )

    menu = process_stack(state)
    assert menu.input_request is not None
    submit_input(state, {"request_id": menu.input_request.id, "selection": "SKILL"})
    assert process_stack(state).input_request is None

    rage = [e for e in state.active_effects if e.effect_type == EffectType.ENRAGED]
    assert len(rage) == 1
    assert rage[0].source_id == "hero_nebkher"
    assert rage[0].source_card_id == "angry_roar"
    assert is_enraged(state, nebkher) is True
    assert is_enraged(state, ursafar) is False
