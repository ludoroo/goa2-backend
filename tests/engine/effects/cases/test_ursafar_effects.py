from __future__ import annotations

import pytest

import goa2.scripts.ursafar_effects  # noqa: F401
from goa2.domain.input import InputRequestType
from goa2.domain.models import ActionType, Card, CardColor, CardTier
from goa2.domain.models.effect import (
    AffectsFilter,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.engine.effect_manager import EffectManager
from goa2.engine.effects import CardEffectRegistry
from goa2.engine.steps import CheckUnitOnBoardStep, MultiSelectStep

from ..builders import EffectScenarioBuilder, hero_card
from ..runner import run_card


def _option_set(run) -> set:
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


def _filler_card(card_id: str = "filler", color: CardColor = CardColor.SILVER) -> Card:
    return Card(
        id=card_id,
        name=card_id.replace("_", " ").title(),
        tier=CardTier.UNTIERED,
        color=color,
        initiative=1,
        primary_action=ActionType.ATTACK,
        secondary_actions={},
        is_ranged=False,
        range_value=0,
        primary_action_value=1,
        effect_id="filler",
        effect_text="",
        is_facedown=False,
    )


def _make_ursafar_enraged(state) -> None:
    """Ursafar performed a rage card earlier this round.

    Rage is owned by the performer via the ENRAGED effect's ``source_id``, not
    inferred from an active card in a played slot — a card can be performed by
    someone else (NebKher's Mind Grip), and then the rage is theirs.
    """
    EffectManager.create_effect(
        state=state,
        source_id="hero_ursafar",
        effect_type=EffectType.ENRAGED,
        scope=EffectScope(
            shape=Shape.POINT,
            affects=AffectsFilter.SELF,
            origin_id="hero_ursafar",
        ),
        duration=DurationType.THIS_ROUND,
        is_active=True,
    )


def _add_brogan_minion_protection(state) -> None:
    brogan = state.get_hero("hero_brogan")
    assert brogan is not None
    brogan.hand = [_filler_card("silver_card")]

    EffectManager.create_effect(
        state=state,
        source_id="hero_brogan",
        effect_type=EffectType.MINION_PROTECTION,
        scope=EffectScope(
            shape=Shape.RADIUS,
            range=2,
            origin_id="hero_brogan",
            affects=AffectsFilter.FRIENDLY_UNITS,
        ),
        duration=DurationType.THIS_ROUND,
        is_active=True,
        allowed_discard_colors=[CardColor.SILVER],
    )


@pytest.mark.effect_flow
def test_prey_drive_skips_bonus_removal_when_attack_target_is_removed() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2)])
        .red_hero(
            "hero_ursafar",
            name="Ursafar",
            at=(0, 0, 0),
            current_card=hero_card("Ursafar", "prey_drive"),
        )
        .blue_minion("blue_target", at=(1, 0, -1))
        .blue_minion("blue_support", at=(2, 0, -2))
        .with_actor("hero_ursafar")
        .build()
    )
    _make_ursafar_enraged(state)

    run = run_card(state, "hero_ursafar")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    assert _option_set(run) == {"blue_target"}

    run.choose("blue_target").finish()

    assert "blue_target" not in state.entity_locations
    assert "blue_support" in state.entity_locations
    assert state.execution_context["target_not_removed"] is None
    assert "remove_target" not in state.execution_context


@pytest.mark.effect_flow
def test_prey_drive_offers_bonus_removal_when_attack_target_is_protected() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (1, 1, -2), (2, 0, -2)])
        .red_hero(
            "hero_ursafar",
            name="Ursafar",
            at=(0, 0, 0),
            current_card=hero_card("Ursafar", "prey_drive"),
        )
        .blue_hero("hero_brogan", name="Brogan", at=(1, 1, -2))
        .blue_minion("blue_target", at=(1, 0, -1))
        .blue_minion("blue_support", at=(2, 0, -2))
        .with_actor("hero_ursafar")
        .build()
    )
    _make_ursafar_enraged(state)
    _add_brogan_minion_protection(state)

    run = run_card(state, "hero_ursafar")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    assert _option_set(run) == {"blue_target"}

    run.choose("blue_target").expect_input(InputRequestType.SELECT_CARD)
    assert run.latest_request is not None
    assert run.latest_request.player_id == "hero_brogan"
    assert _option_set(run) == {"silver_card"}

    run.choose("silver_card").expect_input(InputRequestType.SELECT_UNIT)
    assert run.latest_request is not None
    assert run.latest_request.player_id == "hero_ursafar"
    assert _option_set(run) == {"blue_target", "blue_support"}

    run.skip().finish()

    assert "blue_target" in state.entity_locations
    assert state.execution_context["block_succeeded"] is False
    assert state.execution_context["target_not_removed"] is True


@pytest.mark.effect_contract
def test_feeding_frenzy_bonus_removal_uses_target_board_presence() -> None:
    state = (
        EffectScenarioBuilder()
        .line_board()
        .red_hero(
            "hero_ursafar",
            name="Ursafar",
            at=(0, 0, 0),
            current_card=hero_card("Ursafar", "feeding_frenzy"),
        )
        .with_actor("hero_ursafar")
        .build()
    )
    _make_ursafar_enraged(state)
    hero = state.get_hero("hero_ursafar")
    assert hero is not None
    card = hero.current_turn_card
    assert card is not None
    effect = CardEffectRegistry.get("feeding_frenzy")
    assert effect is not None

    steps = effect.get_steps(state, hero, card)

    assert any(
        isinstance(step, CheckUnitOnBoardStep)
        and step.unit_key == "victim_id"
        and step.output_key == "target_not_removed"
        for step in steps
    )
    multi_select = next(step for step in steps if isinstance(step, MultiSelectStep))
    assert multi_select.active_if_key == "target_not_removed"
