from __future__ import annotations

import pytest
from pydantic import ValidationError

from goa2.domain.input import InputRequestType
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardState,
    CardTier,
)
from goa2.domain.models.effect import ActiveEffect, DurationType, EffectScope, EffectType, Shape
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.steps import ForceDiscardStep, SetContextFlagStep
from tests.engine.effects.builders import EffectScenarioBuilder


def _card(card_id: str, *, basic: bool) -> Card:
    return Card(
        id=card_id,
        name=card_id,
        tier=CardTier.UNTIERED if basic else CardTier.III,
        color=CardColor.GOLD if basic else CardColor.BLUE,
        initiative=5,
        primary_action=ActionType.SKILL,
        secondary_actions={},
        effect_id="",
        effect_text="",
        state=CardState.HAND,
        is_facedown=False,
    )


def _state():
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1)])
        .red_hero("hero_actor", at=(0, 0, 0))
        .blue_hero("hero_victim", at=(1, 0, -1))
        .with_actor("hero_actor")
        .build()
    )
    state.execution_context["victim"] = "hero_victim"
    return state


def test_filtered_force_discard_offers_only_requested_basic_status() -> None:
    state = _state()
    victim = state.get_hero("hero_victim")
    victim.hand = [_card("basic", basic=True), _card("non_basic", basic=False)]
    push_steps(state, [ForceDiscardStep(victim_key="victim", card_is_basic=False)])

    request = process_stack(state)

    assert request.input_request is not None
    assert request.input_request.request_type == InputRequestType.SELECT_CARD
    assert request.input_request.player_id == "hero_victim"
    assert {option.id for option in request.input_request.options} == {"non_basic"}


def test_filtered_force_discard_no_match_is_noop_and_continues() -> None:
    state = _state()
    state.get_hero("hero_victim").hand = [_card("non_basic", basic=False)]
    push_steps(
        state,
        [
            ForceDiscardStep(victim_key="victim", card_is_basic=True),
            SetContextFlagStep(key="continued", value=True),
        ],
    )

    result = process_stack(state)

    assert result.input_request is None
    assert state.execution_context["continued"] is True
    assert [card.id for card in state.get_hero("hero_victim").hand] == ["non_basic"]


def test_filtered_force_discard_allows_mrak_shield_as_replacement() -> None:
    state = _state()
    victim = state.get_hero("hero_victim")
    victim.hand = [_card("required_non_basic", basic=False)]
    shield = _card("discard_shield", basic=True)
    shield.state = CardState.RESOLVED
    victim.played_cards = [shield]
    state.active_effects.append(
        ActiveEffect(
            id="shield_effect",
            source_id=victim.id,
            source_card_id=shield.id,
            effect_type=EffectType.DISCARD_SHIELD,
            scope=EffectScope(shape=Shape.POINT, origin_id=victim.id),
            duration=DurationType.THIS_ROUND,
            created_at_turn=state.turn,
            created_at_round=state.round,
            is_active=True,
        )
    )
    push_steps(state, [ForceDiscardStep(victim_key="victim", card_is_basic=False)])

    request = process_stack(state)

    assert request.input_request is not None
    assert {option.id for option in request.input_request.options} == {
        "required_non_basic",
        "discard_shield",
    }
    state.execution_stack[-1].pending_input = {"selection": "discard_shield"}
    process_stack(state)
    assert [card.id for card in victim.hand] == ["required_non_basic"]
    assert [card.id for card in victim.discard_pile] == ["discard_shield"]


def test_filtered_force_discard_does_not_enable_shield_without_matching_hand_card() -> None:
    state = _state()
    victim = state.get_hero("hero_victim")
    victim.hand = [_card("wrong_basic", basic=True)]
    shield = _card("discard_shield", basic=True)
    shield.state = CardState.RESOLVED
    victim.played_cards = [shield]
    state.active_effects.append(
        ActiveEffect(
            id="shield_effect",
            source_id=victim.id,
            source_card_id=shield.id,
            effect_type=EffectType.DISCARD_SHIELD,
            scope=EffectScope(shape=Shape.POINT, origin_id=victim.id),
            duration=DurationType.THIS_ROUND,
            created_at_turn=state.turn,
            created_at_round=state.round,
            is_active=True,
        )
    )
    push_steps(state, [ForceDiscardStep(victim_key="victim", card_is_basic=False)])

    result = process_stack(state)

    assert result.input_request is None
    assert [card.id for card in victim.played_cards if card is not None] == ["discard_shield"]


# ---------------------------------------------------------------------------
# Victim source semantics: validation + non-empty literal override + fallback.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"victim_id": "", "victim_key": ""},
        {"victim_id": None, "victim_key": None},
        {"victim_id": ""},
        {"victim_key": ""},
    ],
)
def test_force_discard_rejects_missing_or_empty_victim_source(kwargs) -> None:
    """ForceDiscardStep must reject construction without a non-empty source."""
    with pytest.raises(ValidationError):
        ForceDiscardStep(**kwargs)


def test_force_discard_validator_runs_on_model_validate() -> None:
    """Validator also fires on deserialization, not just direct construction."""
    with pytest.raises(ValidationError):
        ForceDiscardStep.model_validate({"type": "force_discard"})


def _prompt(state):
    request = process_stack(state)
    assert request.input_request is not None
    assert request.input_request.request_type == InputRequestType.SELECT_CARD
    return request.input_request


def _setup_two_heroes_with_cards():
    state = _state()
    state.get_hero("hero_actor").hand = [_card("actor_card", basic=False)]
    state.get_hero("hero_victim").hand = [_card("victim_card", basic=False)]
    return state


def test_force_discard_non_empty_literal_wins_over_conflicting_key() -> None:
    """Non-empty ``victim_id`` overrides a conflicting context ``victim_key``."""
    state = _setup_two_heroes_with_cards()
    state.execution_context["victim"] = "hero_actor"  # would route to wrong hero

    push_steps(state, [ForceDiscardStep(victim_id="hero_victim", victim_key="victim")])
    req = _prompt(state)

    assert req.player_id == "hero_victim"
    assert {opt.id for opt in req.options} == {"victim_card"}


def test_force_discard_empty_literal_falls_back_to_key() -> None:
    """An empty ``victim_id`` is treated as absent and resolves from ``victim_key``."""
    state = _setup_two_heroes_with_cards()
    state.execution_context["victim"] = "hero_victim"

    push_steps(state, [ForceDiscardStep(victim_id="", victim_key="victim")])
    req = _prompt(state)

    assert req.player_id == "hero_victim"
    assert {opt.id for opt in req.options} == {"victim_card"}


def test_select_step_empty_literals_fall_back_to_keys() -> None:
    """SelectStep empty literals fall back to keys for prompt + card owner."""
    from goa2.domain.models.enums import TargetType
    from goa2.engine.steps import SelectStep

    state = _setup_two_heroes_with_cards()
    state.execution_context["victim"] = "hero_victim"

    push_steps(
        state,
        [
            SelectStep(
                target_type=TargetType.CARD,
                prompt="pick",
                context_hero_id="",
                override_player_id="",
                context_hero_id_key="victim",
                override_player_id_key="victim",
                is_mandatory=True,
            )
        ],
    )
    req = _prompt(state)

    assert req.player_id == "hero_victim"
    assert {opt.id for opt in req.options} == {"victim_card"}
