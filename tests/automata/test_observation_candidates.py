"""Behavioral contract for decision candidates in graph observations."""

from __future__ import annotations

import json
import math
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from automata.decision import DecisionDescriptor as Decision
from automata.models import DecisionObservation, canonical_json_bytes, from_canonical_json
from automata.search.ismcts.engine import legal_keys
from goa2.domain.hex import Hex
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))
TWO_LANE_MAP = str(Path("src/goa2/data/maps/across_the_river.json"))


def _state(red: list[str] | None = None, blue: list[str] | None = None, *, map_path: str = MAP):
    return GameSetup.create_game(
        map_path,
        red or ["Razzle", "Wasp"],
        blue or ["Arien", "Brogan"],
        game_type="QUICK",
        seed=31,
    )


def _encode(state, decision: Decision, keys: list[Any] | None = None) -> DecisionObservation:
    """Use only the proposed public, sanitized engine-to-observation seam."""
    return import_module("automata.observation").encode_decision(
        state,
        decision,
        legal_keys(decision) if keys is None else keys,
        decision_owner_hero_id="hero_razzle",
        perspective_team="RED",
    )


def _dump(value: Any) -> dict[str, Any]:
    result = value.model_dump(mode="json")
    assert isinstance(result, dict)
    return result


def _candidates(observation: Any) -> list[dict[str, Any]]:
    candidates = _dump(observation)["candidates"]
    assert isinstance(candidates, list)
    return candidates


def _tokens(observation: Any, kind: str | None = None) -> list[dict[str, Any]]:
    tokens = _dump(observation)["state"]["tokens"]
    assert isinstance(tokens, list)
    return tokens if kind is None else [token for token in tokens if token["kind"] == kind]


def _candidate_kind(candidate: dict[str, Any]) -> str:
    candidate_id = candidate["candidate_id"]
    assert isinstance(candidate_id, dict)
    return candidate_id["kind"]


def _request(
    request_type: InputRequestType,
    options: list[Any],
    *,
    can_skip: bool = False,
) -> InputRequest:
    return InputRequest(
        id="request-uuid-must-not-leak",
        request_type=request_type,
        player_id="hero_razzle",
        prompt="prompt must not become candidate identity",
        options=[InputOption.from_value(value) for value in options],
        can_skip=can_skip,
        context={"arbitrary_hidden_metadata": "must-not-leak"},
    )


def test_card_decision_maps_each_hand_card_and_explicit_finish_once() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_razzle"))
    assert hero is not None
    decision = Decision("CARD", hero=hero, can_finish_planning=True)

    observation = _encode(state, decision)
    candidates = _candidates(observation)
    card_tokens = {
        token["features"]["card_id"]: token["local_ref"]
        for token in _tokens(observation, "CARD")
        if token["features"]["card_id"] is not None
    }

    assert [candidate["selection"] for candidate in candidates] == legal_keys(decision)
    assert [_candidate_kind(candidate) for candidate in candidates].count("FINISH") == 1
    card_candidates = [
        candidate for candidate in candidates if _candidate_kind(candidate) == "CARD"
    ]
    assert {candidate["candidate_id"]["card_id"] for candidate in card_candidates} == {
        card.id for card in hero.hand
    }
    assert len(card_candidates) == len(hero.hand)
    assert all(
        candidate["target_ref"] == card_tokens[candidate["candidate_id"]["card_id"]]
        for candidate in card_candidates
    )


def test_forced_card_pass_with_no_legal_branch_needs_no_candidate_observation() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_razzle"))
    assert hero is not None
    hero.hand.clear()
    decision = Decision("CARD", hero=hero)

    assert legal_keys(decision) == []


CARD_REQUESTS = {
    InputRequestType.DEFENSE_CARD,
    InputRequestType.UPGRADE_CHOICE,
    InputRequestType.SELECT_CARD,
}
UNIT_REQUESTS = {
    InputRequestType.TIE_BREAKER,
    InputRequestType.SELECT_ALLY,
    InputRequestType.SELECT_ENEMY,
    InputRequestType.SELECT_UNIT,
    InputRequestType.CHOOSE_ACTOR,
}
HEX_REQUESTS = {
    InputRequestType.MOVEMENT_HEX,
    InputRequestType.FAST_TRAVEL_DESTINATION,
    InputRequestType.SELECT_HEX,
    InputRequestType.CHOOSE_RESPAWN_HEX,
}
ACTION_REQUESTS = {
    InputRequestType.ACTION_CHOICE,
    InputRequestType.CHOOSE_ACTION,
    InputRequestType.CHOOSE_RESPAWN,
}
OPTION_REQUESTS = {InputRequestType.SELECT_OPTION, InputRequestType.CONFIRM_PASSIVE}
UNSUPPORTED_REQUESTS = {InputRequestType.NONE, InputRequestType.UPGRADE_PHASE}


def test_every_public_input_request_type_has_an_explicit_candidate_policy() -> None:
    classified = (
        CARD_REQUESTS
        | UNIT_REQUESTS
        | HEX_REQUESTS
        | ACTION_REQUESTS
        | OPTION_REQUESTS
        | UNSUPPORTED_REQUESTS
        | {
            InputRequestType.SELECT_UNIT_OR_TOKEN,
            InputRequestType.SELECT_NUMBER,
            InputRequestType.SELECT_CARD_OR_PASS,
        }
    )
    assert classified == set(InputRequestType)


@pytest.mark.parametrize(
    ("request_type", "option", "expected_kind"),
    [
        *[(request_type, "razzle_card_1", "CARD") for request_type in CARD_REQUESTS],
        *[(request_type, "hero_wasp", "UNIT") for request_type in UNIT_REQUESTS],
        *[(request_type, Hex(q=0, r=0, s=0), "HEX") for request_type in HEX_REQUESTS],
        *[(request_type, "advance", "ACTION") for request_type in ACTION_REQUESTS],
        *[(request_type, "hold", "OPTION") for request_type in OPTION_REQUESTS],
        (InputRequestType.SELECT_NUMBER, 0, "NUMBER"),
    ],
)
def test_supported_input_request_types_encode_semantic_candidates(
    request_type: InputRequestType, option: Any, expected_kind: str
) -> None:
    state = _state()
    if expected_kind == "CARD":
        hero = state.get_hero(HeroID("hero_razzle"))
        assert hero is not None
        option = hero.hand[0].id
    if expected_kind == "HEX":
        option = next(iter(state.board.tiles))
    decision = Decision("INPUT", request=_request(request_type, [option]))

    observation = _encode(state, decision)
    candidate = _candidates(observation)[0]

    assert _candidate_kind(candidate) == expected_kind
    assert candidate["selection"] == (
        {"q": option.q, "r": option.r, "s": option.s} if isinstance(option, Hex) else option
    )
    if expected_kind in {"CARD", "UNIT", "HEX"}:
        assert candidate["target_ref"] in {token["local_ref"] for token in _tokens(observation)}


def test_respawn_choice_preserves_actual_action_options_and_order() -> None:
    state = _state()
    request = _request(InputRequestType.CHOOSE_RESPAWN, ["RESPAWN", "PASS"])
    decision = Decision("INPUT", request=request)

    observation = _encode(state, decision)
    candidates = _candidates(observation)

    assert legal_keys(decision) == ["RESPAWN", "PASS"]
    assert [_candidate_kind(candidate) for candidate in candidates] == ["ACTION", "ACTION"]
    assert [candidate["candidate_id"]["action_id"] for candidate in candidates] == [
        "RESPAWN",
        "PASS",
    ]
    assert [candidate["selection"] for candidate in candidates] == ["RESPAWN", "PASS"]


@pytest.mark.parametrize("request_type", [InputRequestType.NONE, InputRequestType.UPGRADE_PHASE])
def test_context_shaped_nonbranchable_requests_fail_closed(request_type: InputRequestType) -> None:
    state = _state()
    request = _request(request_type, [])
    request.context["players"] = {"hero_razzle": {"choices": ["private-upgrade"]}}

    with pytest.raises(ValueError, match=r"unsupported|non.?branch|candidate"):
        _encode(state, Decision("INPUT", request=request), ["private-upgrade"])


def test_unit_or_token_candidates_use_distinct_durable_types_and_graph_refs() -> None:
    state = _state()
    token = next(token for supply in state.token_pool.values() for token in supply)
    token_hex = next(
        hex_
        for hex_, tile in state.board.tiles.items()
        if not tile.is_terrain and tile.occupant_id is None
    )
    state.place_entity(token.id, token_hex)
    token_id = str(token.id)
    request = _request(InputRequestType.SELECT_UNIT_OR_TOKEN, ["hero_wasp", token_id])

    observation = _encode(state, Decision("INPUT", request=request))
    candidates = _candidates(observation)

    assert [_candidate_kind(candidate) for candidate in candidates] == ["UNIT", "ENTITY"]
    assert all(candidate["target_ref"] for candidate in candidates)
    assert {candidate["target_ref"] for candidate in candidates} <= {
        token["local_ref"] for token in _tokens(observation)
    }


def test_finish_skip_zero_option_action_and_confirm_are_unambiguous() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_razzle"))
    assert hero is not None
    observations = [
        _encode(state, Decision("CARD", hero=hero, can_finish_planning=True)),
        _encode(
            state,
            Decision("INPUT", request=_request(InputRequestType.SELECT_NUMBER, [0], can_skip=True)),
        ),
        _encode(state, Decision("INPUT", request=_request(InputRequestType.SELECT_OPTION, ["0"]))),
        _encode(state, Decision("INPUT", request=_request(InputRequestType.CHOOSE_ACTION, ["0"]))),
        _encode(
            state, Decision("INPUT", request=_request(InputRequestType.CONFIRM_PASSIVE, ["YES"]))
        ),
    ]
    ids = [
        candidate["candidate_id"]
        for observation in observations
        for candidate in _candidates(observation)
        if _candidate_kind(candidate) in {"FINISH", "SKIP", "NUMBER", "OPTION", "ACTION"}
    ]

    assert {candidate_id["kind"] for candidate_id in ids} >= {
        "FINISH",
        "SKIP",
        "NUMBER",
        "OPTION",
        "ACTION",
    }
    assert len({json.dumps(candidate_id, sort_keys=True) for candidate_id in ids}) == len(ids)


def test_card_or_pass_uses_card_and_skip_without_inventing_a_pass_identity() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_razzle"))
    assert hero is not None
    request = _request(InputRequestType.SELECT_CARD_OR_PASS, [hero.hand[0].id], can_skip=True)

    candidates = _candidates(_encode(state, Decision("INPUT", request=request)))

    assert [_candidate_kind(candidate) for candidate in candidates] == ["CARD", "SKIP"]
    assert [candidate["selection"] for candidate in candidates] == [hero.hand[0].id, "SKIP"]


def test_explicit_card_or_pass_option_without_a_card_token_uses_owner_fallback() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_razzle"))
    assert hero is not None
    request = _request(
        InputRequestType.SELECT_CARD_OR_PASS,
        [hero.hand[0].id, "PASS"],
    )
    request.options[1].text = "private display text"
    request.options[1].metadata = {"hidden_card_metadata": "must-not-leak"}

    observation = _encode(state, Decision("INPUT", request=request))
    candidates = _candidates(observation)

    assert [_candidate_kind(candidate) for candidate in candidates] == ["CARD", "CARD"]
    assert candidates[1]["candidate_id"]["card_id"] == "PASS"
    assert candidates[1]["selection"] == "PASS"
    assert candidates[1]["target_ref"] == "hero:hero_razzle"
    assert not any(
        token["kind"] == "CARD" and token["features"].get("card_id") == "PASS"
        for token in _tokens(observation)
    )
    assert b"private display text" not in canonical_json_bytes(observation)
    assert b"hidden_card_metadata" not in canonical_json_bytes(observation)


def test_numeric_candidates_preserve_zero_integers_and_fractions_exactly() -> None:
    state = _state()
    request = _request(InputRequestType.SELECT_NUMBER, [0, -2, 2.5])

    candidates = _candidates(_encode(state, Decision("INPUT", request=request)))

    assert [_candidate_kind(candidate) for candidate in candidates] == ["NUMBER"] * 3
    assert [candidate["candidate_id"]["value"] for candidate in candidates] == [0, -2, 2.5]
    assert [candidate["selection"] for candidate in candidates] == [0, -2, 2.5]


def test_request_runtime_and_hidden_metadata_do_not_enter_canonical_bytes() -> None:
    state = _state()
    first = _request(InputRequestType.SELECT_OPTION, ["hold"])
    second = _request(InputRequestType.SELECT_OPTION, ["hold"])
    second.id = "another-private-request-id"
    second.prompt = "another private prompt"
    second.player_id = "private-routing-id"
    second.context = {"secret": "enemy-card-id", "object": {"repr": "private"}}
    second.options[0].text = "secret display text"
    second.options[0].metadata = {"hidden": "enemy-card-id"}

    first_bytes = canonical_json_bytes(_encode(state, Decision("INPUT", request=first)))
    second_bytes = canonical_json_bytes(_encode(state, Decision("INPUT", request=second)))

    assert first_bytes == second_bytes
    for forbidden in (b"request-id", b"prompt", b"private-routing", b"enemy-card", b"repr"):
        assert forbidden not in first_bytes
        assert forbidden not in second_bytes


@pytest.mark.parametrize("unavailable_owner_id", ["hero_wasp", "hero_arien"])
def test_unavailable_private_card_legal_option_fails_closed_instead_of_leaking_identity(
    unavailable_owner_id: str,
) -> None:
    state = _state()
    unavailable_owner = state.get_hero(HeroID(unavailable_owner_id))
    assert unavailable_owner is not None
    hidden_card_id = unavailable_owner.hand[0].id
    request = _request(InputRequestType.SELECT_CARD, [hidden_card_id])

    with pytest.raises(ValueError, match=r"hidden|visible|reference|candidate"):
        _encode(state, Decision("INPUT", request=request))


@pytest.mark.parametrize(
    "input_request,keys",
    [
        (_request(InputRequestType.SELECT_OPTION, ["same", "same"]), ["same"]),
        (_request(InputRequestType.SELECT_OPTION, ["one", "two"]), ["one"]),
        (_request(InputRequestType.SELECT_UNIT, ["missing-unit"]), ["missing-unit"]),
        (
            _request(InputRequestType.SELECT_HEX, [{"q": 999, "r": 0, "s": -999}]),
            [("hex", 999, 0, -999)],
        ),
        (_request(InputRequestType.SELECT_NUMBER, [math.inf]), [math.inf]),
    ],
    ids=["duplicate", "legal-mismatch", "missing-ref", "missing-hex", "non-finite"],
)
def test_ambiguous_mismatched_or_unrepresentable_candidates_fail_closed(
    input_request: InputRequest, keys: list[Any]
) -> None:
    with pytest.raises(ValueError):
        _encode(_state(), Decision("INPUT", request=input_request), keys)


def test_candidate_permutation_changes_only_candidate_sequence() -> None:
    state = _state()
    options = ["hero_wasp", "hero_arien"]
    first_decision = Decision("INPUT", request=_request(InputRequestType.SELECT_UNIT, options))
    second_decision = Decision(
        "INPUT", request=_request(InputRequestType.SELECT_UNIT, list(reversed(options)))
    )

    first = _encode(state, first_decision)
    second = _encode(state, second_decision)
    first_candidates = _candidates(first)
    second_candidates = _candidates(second)

    assert _dump(first)["state"] == _dump(second)["state"]
    assert list(reversed(first_candidates)) == second_candidates


def test_complete_decision_observation_round_trips_canonically() -> None:
    state = _state()
    observation = _encode(
        state,
        Decision("INPUT", request=_request(InputRequestType.SELECT_UNIT, ["hero_wasp"])),
    )
    encoded = canonical_json_bytes(observation)

    restored = from_canonical_json(DecisionObservation, encoded)

    assert restored == observation
    assert canonical_json_bytes(restored) == encoded
    for candidate in observation.candidates:
        candidate_id = candidate.candidate_id
        candidate_bytes = canonical_json_bytes(candidate_id)
        assert from_canonical_json(type(candidate_id), candidate_bytes) == candidate_id


def _field_signature(observation: Any, collection_name: str) -> dict[str, frozenset[str]]:
    dumped = _dump(observation)
    records = (
        dumped["candidates"]
        if collection_name == "candidates"
        else dumped["state"][collection_name]
    )
    key = "kind" if collection_name != "candidates" else "candidate_id"
    grouped: dict[str, set[frozenset[str]]] = {}
    for record in records:
        kind = record[key] if isinstance(record[key], str) else record[key]["kind"]
        grouped.setdefault(kind, set()).add(frozenset(record.get("features", {})))
    assert all(len(field_sets) == 1 for field_sets in grouped.values())
    return {kind: next(iter(field_sets)) for kind, field_sets in grouped.items()}


def test_compositions_preserve_per_kind_token_edge_and_candidate_fields() -> None:
    fixtures = [
        (MAP, ["Razzle"], ["Arien"]),
        (MAP, ["Razzle", "Wasp"], ["Arien", "Brogan"]),
        (MAP, ["Razzle", "Wasp"], ["Arien", "Brogan", "Tali"]),
        (MAP, ["Razzle", "Wasp", "Xargatha"], ["Arien", "Brogan", "Tali"]),
        (TWO_LANE_MAP, ["Razzle", "Wasp", "Xargatha"], ["Arien", "Brogan", "Tali"]),
    ]
    observations = []
    for map_path, red, blue in fixtures:
        state = _state(red, blue, map_path=map_path)
        hero = state.get_hero(HeroID("hero_razzle"))
        assert hero is not None
        observations.append(_encode(state, Decision("CARD", hero=hero)))

    for collection_name in ("tokens", "relationships", "candidates"):
        signatures = [
            _field_signature(observation, collection_name) for observation in observations
        ]
        shared_kinds = set.intersection(*(set(signature) for signature in signatures))
        assert shared_kinds
        for kind in shared_kinds:
            assert len({signature[kind] for signature in signatures}) == 1
