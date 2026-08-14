"""Arien card effect tests."""

import pytest

import goa2.scripts.arien_effects  # noqa: F401 - register Arien effects
from goa2.domain.input import InputRequestType
from goa2.domain.models import (
    ActionType,
    ActiveEffect,
    AffectsFilter,
    Card,
    CardColor,
    CardState,
    CardTier,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.models.enums import DisplacementType

from ..builders import EffectScenarioBuilder, hero_card
from ..runner import run_card


def _melee_attack() -> Card:
    return Card(
        id="incoming_attack",
        name="Incoming Attack",
        tier=CardTier.UNTIERED,
        color=CardColor.GOLD,
        initiative=1,
        primary_action=ActionType.ATTACK,
        primary_action_value=3,
        secondary_actions={},
        range_value=1,
        effect_id="",
        effect_text="",
        is_facedown=False,
    )


def _option_set(run) -> set:
    """Set of selectable values from the current request."""
    assert run.latest_request is not None
    options = set()
    for option in run.latest_request.options:
        if hasattr(option, "metadata") and option.metadata and "raw" in option.metadata:
            options.add(option.metadata.get("raw"))
        elif hasattr(option, "id"):
            options.add(option.id)
        else:
            options.add(option)
    return options


def _base_state():
    """Arien plays Noble Blade against an adjacent enemy, with an ally to nudge.

    Board:
      (0,0,0)   Arien (RED, actor)
      (1,0,-1)  enemy minion (BLUE) - the attack target
      (2,-1,-1) ally minion (RED) - adjacent to the target, the nudge candidate
      (3,-1,-2) Wasp (BLUE) - Magnetic Dagger source in the dagger variant
      (2,0,-2), (1,-1,0), (3,-2,-1) empty - somewhere for the ally to be nudged
    """
    state = (
        EffectScenarioBuilder()
        .with_hexes(
            [
                (0, 0, 0),
                (1, 0, -1),
                (2, -1, -1),
                (3, -1, -2),
                (2, 0, -2),
                (1, -1, 0),
                (3, -2, -1),
            ]
        )
        .red_hero(
            "hero_arien",
            at=(0, 0, 0),
            current_card=hero_card("Arien", "noble_blade"),
        )
        .blue_minion("enemy_minion", at=(1, 0, -1))
        .red_minion("ally_minion", at=(2, -1, -1))
        .blue_hero("hero_wasp", at=(3, -1, -2))
        .with_actor("hero_arien")
        .build()
    )
    return state


def _magnetic_dagger_state():
    """Same board, with Wasp's Magnetic Dagger active.

    The dagger blocks PLACE and SWAP for Wasp's enemy units. It does NOT block
    MOVE, so Arien nudging his own ally one space remains legal.
    """
    state = _base_state()
    state.active_effects.append(
        ActiveEffect(
            id="magnetic_dagger_effect",
            source_id="hero_wasp",
            source_card_id="magnetic_dagger",
            effect_type=EffectType.PLACEMENT_PREVENTION,
            scope=EffectScope(
                shape=Shape.RADIUS,
                range=3,
                origin_id="hero_wasp",
                affects=AffectsFilter.ENEMY_UNITS,
            ),
            duration=DurationType.THIS_TURN,
            displacement_blocks=[DisplacementType.PLACE, DisplacementType.SWAP],
            created_at_turn=1,
            created_at_round=1,
            is_active=True,
            blocks_enemy_actors=True,
            blocks_friendly_actors=False,
            blocks_self=False,
        )
    )
    return state


@pytest.mark.effect_flow
@pytest.mark.parametrize("defense_id", ["expert_duelist", "master_duelist"])
def test_duelist_immunity_is_linked_to_the_defense_card(defense_id: str) -> None:
    attack = _melee_attack()
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1)])
        .blue_hero("hero_attacker", at=(0, 0, 0), current_card=attack)
        .red_hero("hero_arien", at=(1, 0, -1))
        .with_actor("hero_attacker")
        .build()
    )
    defense = hero_card("Arien", defense_id)
    defense.state = CardState.HAND
    arien = state.get_hero("hero_arien")
    assert arien is not None
    arien.hand = [defense]

    run = run_card(state, "hero_attacker")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("hero_arien")
    run.expect_input(InputRequestType.SELECT_CARD_OR_PASS).choose(defense_id)
    run.finish()

    immunity = next(
        effect
        for effect in state.active_effects
        if effect.effect_type == EffectType.ATTACK_IMMUNITY
    )
    assert immunity.source_id == "hero_arien"
    assert immunity.source_card_id == defense_id
    assert defense.is_active is True
    assert attack.is_active is False


@pytest.mark.effect_flow
def test_noble_blade_offers_adjacent_ally_as_nudge_target() -> None:
    """Control: with no protection effect, the ally is a legal nudge target."""
    state = _base_state()

    run = run_card(state, "hero_arien")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("enemy_minion")

    run.expect_input(InputRequestType.SELECT_UNIT)
    assert "ally_minion" in _option_set(run)


@pytest.mark.effect_flow
def test_noble_blade_can_nudge_ally_under_magnetic_dagger() -> None:
    """Magnetic Dagger blocks PLACE, not MOVE, so the ally nudge stays available.

    Regression: the nudge SelectStep screened candidates with a placement check,
    so the ally silently vanished from the options even though MoveUnitStep
    would have allowed the move.
    """
    state = _magnetic_dagger_state()

    run = run_card(state, "hero_arien")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("enemy_minion")

    run.expect_input(InputRequestType.SELECT_UNIT)
    assert "ally_minion" in _option_set(
        run
    ), "Ally should be nudgeable: Magnetic Dagger blocks PLACE/SWAP, not MOVE"


@pytest.mark.effect_flow
def test_violent_torrent_repeat_cannot_fall_back_to_the_first_target() -> None:
    """ "May repeat once on a different unit" — declining the second target must
    end the card, not hand the attack an unrestricted target picker that offers
    the unit Arien already hit.
    """
    from goa2.domain.models import ActionType, CardColor

    from ..builders import skill_card

    arena = [(q, r, -q - r) for q in range(-3, 4) for r in range(-3, 4) if abs(-q - r) <= 3]
    state = (
        EffectScenarioBuilder()
        .with_hexes(arena)
        .red_hero("hero_arien", at=(0, 0, 0), current_card=hero_card("Arien", "violent_torrent"))
        .blue_hero("first_target", at=(1, 0, -1))
        .blue_hero("other_enemy", at=(0, 1, -1))
        .with_actor("hero_arien")
        .build()
    )
    shield = skill_card("big_shield", color=CardColor.BLUE)
    shield.secondary_actions = {ActionType.DEFENSE: 20}
    defender = state.get_hero("first_target")
    assert defender is not None
    defender.hand = [shield]

    run = run_card(state, "hero_arien")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("first_target")
    run.expect_input(InputRequestType.SELECT_CARD_OR_PASS).choose("PASS")
    run.expect_input(InputRequestType.SELECT_OPTION).choose("YES")

    # The restricted select correctly excludes the unit already targeted.
    run.expect_input(InputRequestType.SELECT_UNIT)
    assert _option_set(run) == {"other_enemy"}

    # Declining it ends the card — no generic "Select Attack Target" fallback.
    run.skip().finish()


@pytest.mark.effect_flow
def test_ebb_and_flow_repeat_may_swap_back_with_the_same_minion() -> None:
    """A bare "may repeat once" — no "different" — leaves the same minion legal."""
    arena = [(q, r, -q - r) for q in range(-3, 4) for r in range(-3, 4) if abs(-q - r) <= 3]
    state = (
        EffectScenarioBuilder()
        .with_hexes(arena)
        .red_hero("hero_arien", at=(0, 0, 0), current_card=hero_card("Arien", "ebb_and_flow"))
        .blue_minion("adj_minion", at=(1, -1, 0))
        .with_actor("hero_arien")
        .build()
    )

    run = run_card(state, "hero_arien")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("adj_minion")
    # It was adjacent before the swap, so the repeat is offered.
    run.expect_input(InputRequestType.SELECT_OPTION).choose("YES")

    run.expect_input(InputRequestType.SELECT_UNIT)
    assert "adj_minion" in _option_set(run)  # swapping back is a legal play
