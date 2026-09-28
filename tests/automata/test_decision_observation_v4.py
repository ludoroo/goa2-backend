"""Decision-observation v4 schema boundary and invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automata.decision import DecisionDescriptor, DecisionSemanticRole
from automata.models.contracts import DecisionObservation, canonical_json_bytes
from automata.observation import encode_decision
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.engine.setup import GameSetup

MAP = Path("src/goa2/data/maps/forgotten_island.json")


def _encoded(*, can_skip: bool = False) -> DecisionObservation:
    state = GameSetup.create_game(
        str(MAP), ["Razzle", "Wasp"], ["Arien", "Brogan"], game_type="QUICK", seed=31
    )
    request = InputRequest(
        id="private-request-id",
        request_type=InputRequestType.SELECT_UNIT,
        player_id="private-routing-id",
        prompt="private prompt",
        options=[
            InputOption(
                id="hero_arien",
                text="private display text",
                metadata={"secret": "must-not-leak"},
            )
        ],
        can_skip=can_skip,
        context={"secret": "must-not-leak"},
    )
    decision = DecisionDescriptor("INPUT", request=request)
    legal = ["hero_arien", *(["SKIP"] if can_skip else [])]
    return encode_decision(
        state,
        decision,
        legal,
        decision_owner_hero_id="hero_razzle",
        perspective_team="RED",
    )


def test_encoder_emits_v4_typed_context_without_private_request_data() -> None:
    observation = _encoded(can_skip=True)

    assert observation.schema_version == 4
    assert observation.decision_kind == "INPUT"
    assert observation.input_request_type == InputRequestType.SELECT_UNIT.value
    assert observation.can_skip is True
    assert observation.semantic_role is DecisionSemanticRole.UNIT_SELECTION
    payload = canonical_json_bytes(observation)
    for private in (
        b"private-request",
        b"private-routing",
        b"private prompt",
        b"display text",
        b"must-not-leak",
    ):
        assert private not in payload


def test_v4_rejects_can_skip_candidate_mismatch_and_card_context() -> None:
    input_observation = _encoded(can_skip=True)
    payload = json.loads(canonical_json_bytes(input_observation))
    payload["can_skip"] = False
    with pytest.raises(ValueError, match="can_skip"):
        DecisionObservation.model_validate(payload)

    payload = json.loads(canonical_json_bytes(input_observation))
    payload.update(
        decision_kind="CARD",
        input_request_type=InputRequestType.SELECT_UNIT.value,
        semantic_role="UNIT_SELECTION",
    )
    with pytest.raises(ValueError, match=r"CARD|planning|request"):
        DecisionObservation.model_validate(payload)
