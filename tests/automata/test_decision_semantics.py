"""Typed, exhaustive decision-context classification contracts."""

from __future__ import annotations

import pytest

from automata.decision import (
    DecisionDescriptor,
    DecisionSemanticRole,
    classify_decision,
)
from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models import ActionType
from goa2.engine.setup import GameSetup

MAP = "src/goa2/data/maps/forgotten_island.json"


@pytest.fixture
def state():
    return GameSetup.create_game(
        MAP,
        ["Razzle", "Wasp"],
        ["Arien", "Brogan"],
        game_type="QUICK",
        seed=31,
    )


def _decision(request_type: InputRequestType, *, can_skip: bool = False) -> DecisionDescriptor:
    return DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            request_type=request_type,
            player_id="hero_razzle",
            can_skip=can_skip,
        ),
    )


EXPECTED_ROLES = {
    InputRequestType.NONE: DecisionSemanticRole.OPTION_SELECTION,
    InputRequestType.ACTION_CHOICE: DecisionSemanticRole.ACTION_CHOICE,
    InputRequestType.MOVEMENT_HEX: DecisionSemanticRole.MOVEMENT_DESTINATION,
    InputRequestType.DEFENSE_CARD: DecisionSemanticRole.DEFENSE_REACTION,
    InputRequestType.TIE_BREAKER: DecisionSemanticRole.ACTOR_CHOICE,
    InputRequestType.SELECT_ALLY: DecisionSemanticRole.UNIT_SELECTION,
    InputRequestType.FAST_TRAVEL_DESTINATION: DecisionSemanticRole.MOVEMENT_DESTINATION,
    InputRequestType.SELECT_ENEMY: DecisionSemanticRole.UNIT_SELECTION,
    InputRequestType.UPGRADE_CHOICE: DecisionSemanticRole.UPGRADE_CHOICE,
    InputRequestType.SELECT_UNIT: DecisionSemanticRole.UNIT_SELECTION,
    InputRequestType.SELECT_UNIT_OR_TOKEN: DecisionSemanticRole.UNIT_SELECTION,
    InputRequestType.SELECT_HEX: DecisionSemanticRole.SPATIAL_SELECTION,
    InputRequestType.SELECT_CARD: DecisionSemanticRole.CARD_SELECTION,
    InputRequestType.SELECT_NUMBER: DecisionSemanticRole.NUMBER_SELECTION,
    InputRequestType.CHOOSE_ACTION: DecisionSemanticRole.ACTION_CHOICE,
    InputRequestType.SELECT_CARD_OR_PASS: DecisionSemanticRole.DEFENSE_REACTION,
    InputRequestType.SELECT_OPTION: DecisionSemanticRole.OPTION_SELECTION,
    InputRequestType.CHOOSE_ACTOR: DecisionSemanticRole.ACTOR_CHOICE,
    InputRequestType.CHOOSE_RESPAWN: DecisionSemanticRole.RESPAWN_CHOICE,
    InputRequestType.CHOOSE_RESPAWN_HEX: DecisionSemanticRole.RESPAWN_DESTINATION,
    InputRequestType.UPGRADE_PHASE: DecisionSemanticRole.UPGRADE_CHOICE,
    InputRequestType.CONFIRM_PASSIVE: DecisionSemanticRole.PASSIVE_REACTION,
}


def test_every_input_request_type_has_one_explicit_semantic_role(state) -> None:
    assert set(EXPECTED_ROLES) == set(InputRequestType)

    for request_type, expected in EXPECTED_ROLES.items():
        semantics = classify_decision(state, _decision(request_type, can_skip=True))
        assert semantics.input_request_type == request_type.value
        assert semantics.can_skip is True
        assert semantics.semantic_role is expected


def test_card_root_has_fixed_planning_semantics(state) -> None:
    semantics = classify_decision(state, DecisionDescriptor("CARD"))

    assert semantics.input_request_type is None
    assert semantics.can_skip is False
    assert semantics.semantic_role is DecisionSemanticRole.PLANNING


def test_select_unit_during_attack_is_normalized_to_attack_target(state) -> None:
    state.execution_context["current_action_type"] = ActionType.ATTACK

    semantics = classify_decision(state, _decision(InputRequestType.SELECT_UNIT))

    assert semantics.semantic_role is DecisionSemanticRole.ATTACK_TARGET


def test_classifier_rejects_malformed_or_non_branchable_decision_shapes(state) -> None:
    with pytest.raises(ValueError, match="INPUT decision requires"):
        classify_decision(state, DecisionDescriptor("INPUT"))
    with pytest.raises(ValueError, match="unsupported decision kind"):
        classify_decision(state, DecisionDescriptor("OVER"))
