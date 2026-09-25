from __future__ import annotations

from goa2.domain.models import ActionType, Card, CardColor, CardTier
from goa2.domain.models.effect import DurationType, EffectScope, EffectType, Shape
from goa2.engine.effect_manager import EffectManager
from goa2.engine.filters_units import ImmunityFilter
from goa2.engine.steps.combat import AttackSequenceStep
from goa2.engine.steps.effects import CreateEffectStep
from tests.engine.effects.builders import EffectScenarioBuilder


def _attack_card(card_id: str, color: CardColor) -> Card:
    return Card(
        id=card_id,
        name=card_id.replace("_", " ").title(),
        tier=CardTier.UNTIERED if color in (CardColor.GOLD, CardColor.SILVER) else CardTier.I,
        color=color,
        initiative=5,
        primary_action=ActionType.ATTACK,
        primary_action_value=3,
        secondary_actions={},
        effect_id="",
        effect_text="",
        is_facedown=False,
    )


def _state(card: Card):
    return (
        EffectScenarioBuilder()
        .line_board()
        .red_hero("hero_attacker", at=(0, 0, 0), current_card=card)
        .blue_hero("hero_defender", at=(1, 0, -1))
        .with_actor("hero_attacker")
        .build()
    )


def _protect(
    state,
    *,
    basic_attacks_only: bool = False,
    non_basic_attacks_only: bool = False,
    except_attacker_ids: list[str] | None = None,
):
    return EffectManager.create_effect(
        state=state,
        source_id="hero_defender",
        effect_type=EffectType.ATTACK_IMMUNITY,
        scope=EffectScope(shape=Shape.GLOBAL),
        duration=DurationType.THIS_ROUND,
        except_attacker_ids=except_attacker_ids or [],
        basic_attacks_only=basic_attacks_only,
        non_basic_attacks_only=non_basic_attacks_only,
        is_active=True,
    )


def _target_is_allowed(state) -> bool:
    return ImmunityFilter().apply("hero_defender", state, state.execution_context)


def _classify_attack(state) -> None:
    state.execution_context["current_action_type"] = ActionType.ATTACK
    AttackSequenceStep(damage=3).resolve(state, state.execution_context)


def test_basic_only_immunity_blocks_basic_but_not_non_basic_attacks() -> None:
    state = _state(_attack_card("basic_attack", CardColor.GOLD))
    effect = _protect(state, basic_attacks_only=True)

    _classify_attack(state)
    assert effect.basic_attacks_only is True
    assert state.execution_context["attack_is_basic"] is True
    assert _target_is_allowed(state) is False

    state.get_hero("hero_attacker").current_turn_card = _attack_card(
        "colored_attack", CardColor.RED
    )
    _classify_attack(state)
    assert state.execution_context["attack_is_basic"] is False
    assert _target_is_allowed(state) is True


def test_basic_only_immunity_uses_the_nested_performing_card_source() -> None:
    outer_colored = _attack_card("outer_colored", CardColor.RED)
    inner_basic = _attack_card("inner_basic", CardColor.SILVER)
    state = _state(outer_colored)
    attacker = state.get_hero("hero_attacker")
    attacker.deck.append(inner_basic)
    _protect(state, basic_attacks_only=True)
    state.execution_context.update(
        {
            "performing_card_id": inner_basic.id,
            "performing_card_owner_id": attacker.id,
        }
    )

    _classify_attack(state)
    assert state.execution_context["attack_is_basic"] is True
    assert _target_is_allowed(state) is False

    outer_basic = _attack_card("outer_basic", CardColor.GOLD)
    inner_colored = _attack_card("inner_colored", CardColor.BLUE)
    attacker.current_turn_card = outer_basic
    attacker.deck.append(inner_colored)
    state.execution_context.update(
        {
            "performing_card_id": inner_colored.id,
            "performing_card_owner_id": attacker.id,
        }
    )

    _classify_attack(state)
    assert state.execution_context["attack_is_basic"] is False
    assert _target_is_allowed(state) is True


def test_default_attack_immunity_still_blocks_all_attacks_and_honors_exceptions() -> None:
    state = _state(_attack_card("colored_attack", CardColor.RED))
    effect = _protect(state, basic_attacks_only=False)

    _classify_attack(state)
    assert effect.basic_attacks_only is False
    assert _target_is_allowed(state) is False

    state.active_effects.clear()
    _protect(
        state,
        basic_attacks_only=True,
        except_attacker_ids=["hero_attacker"],
    )
    state.get_hero("hero_attacker").current_turn_card = _attack_card("basic_attack", CardColor.GOLD)
    _classify_attack(state)
    assert _target_is_allowed(state) is True


def test_attack_immunity_remains_fail_closed_when_current_actor_is_missing() -> None:
    state = _state(_attack_card("basic_attack", CardColor.GOLD))
    _protect(state)
    state.execution_context.update(
        {"current_action_type": ActionType.ATTACK, "attack_is_basic": True}
    )
    state.current_actor_id = None

    assert _target_is_allowed(state) is False


def test_create_effect_step_plumbs_basic_attacks_only_payload() -> None:
    state = _state(_attack_card("basic_attack", CardColor.GOLD))

    result = CreateEffectStep(
        effect_type=EffectType.ATTACK_IMMUNITY,
        scope=EffectScope(shape=Shape.GLOBAL),
        duration=DurationType.THIS_ROUND,
        basic_attacks_only=True,
        is_active=True,
        use_context_card=False,
    ).resolve(state, state.execution_context)

    assert result.is_finished is True
    assert state.active_effects[0].basic_attacks_only is True


def test_non_basic_only_immunity_blocks_non_basic_but_not_basic_attacks() -> None:
    state = _state(_attack_card("colored_attack", CardColor.RED))
    effect = _protect(state, non_basic_attacks_only=True)

    _classify_attack(state)
    assert effect.non_basic_attacks_only is True
    assert state.execution_context["attack_is_basic"] is False
    assert _target_is_allowed(state) is False

    state.get_hero("hero_attacker").current_turn_card = _attack_card("basic_attack", CardColor.GOLD)
    _classify_attack(state)
    assert state.execution_context["attack_is_basic"] is True
    assert _target_is_allowed(state) is True


def test_create_effect_step_plumbs_non_basic_attacks_only_payload() -> None:
    state = _state(_attack_card("colored_attack", CardColor.RED))

    CreateEffectStep(
        effect_type=EffectType.ATTACK_IMMUNITY,
        scope=EffectScope(shape=Shape.GLOBAL),
        duration=DurationType.THIS_ROUND,
        non_basic_attacks_only=True,
        is_active=True,
        use_context_card=False,
    ).resolve(state, state.execution_context)

    assert state.active_effects[0].non_basic_attacks_only is True


def test_basic_and_non_basic_immunities_stack_to_block_every_attack() -> None:
    state = _state(_attack_card("colored_attack", CardColor.RED))
    _protect(state, basic_attacks_only=True)
    _protect(state, non_basic_attacks_only=True)

    _classify_attack(state)
    assert _target_is_allowed(state) is False

    state.get_hero("hero_attacker").current_turn_card = _attack_card("basic_attack", CardColor.GOLD)
    _classify_attack(state)
    assert _target_is_allowed(state) is False
