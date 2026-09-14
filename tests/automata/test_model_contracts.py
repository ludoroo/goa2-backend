"""Public serialization and validation contracts for Phase 0 learned models."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import BaseModel

import automata.models.contracts as learned_contracts
from automata.models.contracts import (
    ActionCandidateID,
    CardCandidateID,
    DecisionObservation,
    EncodedCandidate,
    EntityCandidateID,
    FinishCandidateID,
    HexCandidateID,
    LearnedObservation,
    NumberCandidateID,
    ObservationToken,
    OptionCandidateID,
    PolicyValueOutput,
    PublicSnapshot,
    SearchOutcome,
    SkipCandidateID,
    UnitCandidateID,
    Viewer,
    canonical_json_bytes,
    from_canonical_json,
)
from automata.models.shared_encoder.artifacts.manifest import ModelArtifactManifest


def test_viewer_has_one_public_schema_v2_contract() -> None:
    viewer = learned_contracts.Viewer(
        schema_version=2,
        private_hero_id="hero_wasp",
        perspective_team="RED",
    )

    assert viewer.model_dump(mode="json") == {
        "schema_version": 2,
        "private_hero_id": "hero_wasp",
        "perspective_team": "RED",
    }
    assert [name for name in learned_contracts.__all__ if name.startswith("Viewer")] == ["Viewer"]
    with pytest.raises(ValueError, match=r"unsupported.*version"):
        from_canonical_json(
            learned_contracts.Viewer,
            b'{"hero_id":"hero_wasp","schema_version":1,"scope":"HERO","team":"RED"}',
        )
    with pytest.raises(ValueError, match=r"extra|hero_id|scope|team"):
        learned_contracts.Viewer.model_validate(
            {
                "schema_version": 2,
                "hero_id": "hero_wasp",
                "team": "RED",
                "scope": "HERO",
            }
        )


def _viewer() -> Viewer:
    return Viewer(schema_version=2, private_hero_id="hero_wasp", perspective_team="RED")


def _candidates():
    return (
        FinishCandidateID(schema_version=1),
        SkipCandidateID(schema_version=1),
        CardCandidateID(schema_version=1, card_id="wasp_card_1"),
        UnitCandidateID(schema_version=1, unit_id="hero_arien"),
        HexCandidateID(schema_version=1, q=1, r=-1, s=0),
        NumberCandidateID(schema_version=1, value=3),
        OptionCandidateID(schema_version=1, option_id="hold"),
        ActionCandidateID(schema_version=1, action_id="advance"),
        EntityCandidateID(schema_version=1, entity_ref="entity:token:0"),
    )


def _decision_observation() -> DecisionObservation:
    state = LearnedObservation(
        schema_version=2,
        viewer=_viewer(),
        tokens=(
            ObservationToken(
                schema_version=1,
                local_ref="unit:0",
                kind="UNIT",
                features={"unit_id": "hero_arien"},
            ),
        ),
    )
    return DecisionObservation(
        schema_version=3,
        state=state,
        decision_kind="INPUT",
        candidates=(
            EncodedCandidate(
                schema_version=1,
                candidate_id=UnitCandidateID(schema_version=1, unit_id="hero_arien"),
                selection="hero_arien",
                target_ref="unit:0",
                features={},
            ),
            EncodedCandidate(
                schema_version=1,
                candidate_id=SkipCandidateID(schema_version=1),
                selection="SKIP",
                features={},
            ),
        ),
    )


def _snapshot() -> PublicSnapshot:
    return PublicSnapshot(
        schema_version=2,
        viewer=_viewer(),
        map_id="forgotten_island",
        game_type="QUICK",
        public_state={"round": 1, "phase": "PLANNING"},
    )


@pytest.mark.parametrize(
    "value",
    [
        _viewer(),
        _snapshot(),
        *_candidates(),
        PolicyValueOutput(
            schema_version=1,
            candidate_ids=_candidates()[:2],
            policy_logits=(0.2, -0.2),
            value=0.25,
        ),
        SearchOutcome(
            schema_version=1,
            candidate_ids=_candidates()[:2],
            prior_probabilities=(0.4, 0.6),
            sample_counts=(4, 4),
            mean_values=(-0.1, 0.3),
            value_variances=(0.01, 0.02),
            improved_probabilities=(0.25, 0.75),
            selected_candidate_id=_candidates()[1],
        ),
        LearnedObservation(
            schema_version=2,
            viewer=_viewer(),
            decision_kind="SNAPSHOT",
            candidate_ids=(),
            features={},
        ),
        _decision_observation(),
    ],
)
def test_contracts_round_trip_through_public_canonical_json(value: BaseModel) -> None:
    encoded = canonical_json_bytes(value)

    assert isinstance(encoded, bytes)
    assert from_canonical_json(type(value), encoded) == value


def test_canonical_json_is_deterministic_across_mapping_insertion_order() -> None:
    forward = _snapshot().model_copy(update={"public_state": {"alpha": 1.0, "beta": 2.0}})
    reverse = _snapshot().model_copy(update={"public_state": {"beta": 2.0, "alpha": 1.0}})

    assert canonical_json_bytes(forward) == canonical_json_bytes(reverse)


def test_candidate_ids_have_stable_explicit_type_tags_and_values() -> None:
    serialized = [json.loads(canonical_json_bytes(candidate)) for candidate in _candidates()]

    assert [item["kind"] for item in serialized] == [
        "FINISH",
        "SKIP",
        "CARD",
        "UNIT",
        "HEX",
        "NUMBER",
        "OPTION",
        "ACTION",
        "ENTITY",
    ]
    assert serialized[2]["card_id"] == "wasp_card_1"
    assert {key: serialized[4][key] for key in ("q", "r", "s")} == {
        "q": 1,
        "r": -1,
        "s": 0,
    }
    assert serialized[5]["value"] == 3
    assert serialized[6]["option_id"] == "hold"
    assert serialized[7]["action_id"] == "advance"
    assert serialized[8]["entity_ref"] == "entity:token:0"
    assert all("repr" not in item and "hash" not in item for item in serialized)


@pytest.mark.parametrize(
    "model",
    [
        Viewer,
        PublicSnapshot,
        LearnedObservation,
        DecisionObservation,
    ],
)
def test_unknown_top_level_schema_versions_fail_clearly(model: type[BaseModel]) -> None:
    valid_by_model: dict[type[BaseModel], BaseModel] = {
        Viewer: _viewer(),
        PublicSnapshot: _snapshot(),
        LearnedObservation: LearnedObservation(
            schema_version=2,
            viewer=_viewer(),
            decision_kind="SNAPSHOT",
            candidate_ids=(),
            features={},
        ),
        DecisionObservation: _decision_observation(),
    }
    valid = valid_by_model[model]
    payload = json.loads(canonical_json_bytes(valid))
    payload["schema_version"] = 999

    with pytest.raises(ValueError, match=r"schema.*version|version.*schema|unsupported.*version"):
        from_canonical_json(model, json.dumps(payload).encode())


def test_contracts_are_frozen_and_reject_extra_fields() -> None:
    viewer = _viewer()
    with pytest.raises((AttributeError, TypeError, ValueError)):
        viewer.perspective_team = "BLUE"

    payload = json.loads(canonical_json_bytes(viewer))
    payload["future_field"] = True
    with pytest.raises(ValueError, match=r"extra|future_field|forbid"):
        from_canonical_json(Viewer, json.dumps(payload).encode())


def test_current_contracts_reject_removed_v1_payloads() -> None:
    current = _viewer()

    assert from_canonical_json(Viewer, canonical_json_bytes(current)) == current
    old_viewer = {"schema_version": 1, "hero_id": "hero_wasp", "team": "RED", "scope": "HERO"}
    old_snapshot = {
        "schema_version": 1,
        "viewer": old_viewer,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "public_state": {},
    }
    old_observation = {
        "schema_version": 1,
        "viewer": old_viewer,
        "decision_kind": "INPUT",
        "candidate_ids": [],
        "features": {},
    }
    old_training_decision = {
        "schema_version": 1,
        "game_id": "game-1",
        "decision_id": "decision-1",
        "world_seed": 17,
        "observation": old_observation,
        "candidate_ids": [],
        "improved_policy": [],
        "selected_candidate_id": None,
        "value_target": 1.0,
    }
    old_manifest = {
        "schema_version": 1,
        "model_digest": "sha256:model",
        "observation_schema_version": 1,
        "map_schema_version": 1,
        "runtime_compatibility_version": 1,
        "hero_adapter_versions": {},
        "supported_heroes": [],
        "supported_maps": [],
        "supported_game_types": [],
    }
    for model, payload in (
        (Viewer, old_viewer),
        (PublicSnapshot, old_snapshot),
        (LearnedObservation, old_observation),
        (DecisionObservation, old_training_decision),
        (ModelArtifactManifest, old_manifest),
    ):
        with pytest.raises(ValueError, match=r"unsupported.*version"):
            from_canonical_json(model, json.dumps(payload).encode())


def test_decision_observation_is_independently_versioned_from_graph_state() -> None:
    state = _decision_observation().state
    decision = _decision_observation()

    assert state.schema_version == 2
    assert decision.schema_version == 3
    assert from_canonical_json(DecisionObservation, canonical_json_bytes(decision)) == decision
    with pytest.raises(ValueError, match=r"unsupported.*version"):
        from_canonical_json(
            DecisionObservation,
            b'{"schema_version":1,"decision_kind":"INPUT","candidate_ids":[]}',
        )
    with pytest.raises(ValueError, match=r"unsupported.*version"):
        from_canonical_json(DecisionObservation, canonical_json_bytes(state))
    with pytest.raises(ValueError, match=r"unsupported.*version"):
        from_canonical_json(LearnedObservation, canonical_json_bytes(decision))


def test_decision_candidates_preserve_typed_identity_and_exact_engine_selection() -> None:
    decision = _decision_observation()

    assert [candidate.candidate_id.kind for candidate in decision.candidates] == ["UNIT", "SKIP"]
    assert [candidate.selection for candidate in decision.candidates] == ["hero_arien", "SKIP"]
    assert decision.candidates[0].target_ref == "unit:0"


@pytest.mark.parametrize(
    "candidates",
    [
        (
            EncodedCandidate(
                schema_version=1,
                candidate_id=UnitCandidateID(schema_version=1, unit_id="hero_arien"),
                selection="hero_arien",
                target_ref="missing",
                features={},
            ),
        ),
        (
            EncodedCandidate(
                schema_version=1,
                candidate_id=SkipCandidateID(schema_version=1),
                selection="SKIP",
                features={},
            ),
        )
        * 2,
    ],
    ids=["missing-target-ref", "duplicate-candidate"],
)
def test_decision_observation_rejects_invalid_candidate_refs_uniqueness_and_alignment(
    candidates,
) -> None:
    with pytest.raises(ValueError, match=r"reference|duplicate|align|selection|target"):
        DecisionObservation(
            schema_version=3,
            state=_decision_observation().state,
            decision_kind="INPUT",
            candidates=candidates,
        )


@pytest.mark.parametrize(
    "candidate_id,selection,target_ref",
    [
        (UnitCandidateID(schema_version=1, unit_id="hero_arien"), "not-the-unit", "unit:0"),
        (
            EntityCandidateID(schema_version=1, entity_ref="entity:other"),
            "public-engine-entity-id",
            "unit:0",
        ),
    ],
    ids=["engine-selection", "entity-target-ref"],
)
def test_encoded_candidate_rejects_identity_alignment_mismatches(
    candidate_id, selection, target_ref
) -> None:
    with pytest.raises(ValueError, match=r"align|selection|target"):
        EncodedCandidate(
            schema_version=1,
            candidate_id=candidate_id,
            selection=selection,
            target_ref=target_ref,
            features={},
        )


@pytest.mark.parametrize(
    ("factory", "meaning"),
    [
        (
            lambda: PolicyValueOutput(
                schema_version=1,
                candidate_ids=(_candidates()[0], _candidates()[0]),
                policy_logits=(0.0, 0.0),
                value=0.0,
            ),
            r"duplicate|candidate",
        ),
        (
            lambda: PolicyValueOutput(
                schema_version=1,
                candidate_ids=_candidates()[:2],
                policy_logits=(0.0,),
                value=0.0,
            ),
            r"align|length|candidate|logit",
        ),
        (
            lambda: SearchOutcome(
                schema_version=1,
                candidate_ids=_candidates()[:2],
                prior_probabilities=(0.5, 0.5),
                sample_counts=(1,),
                mean_values=(0.0, 0.0),
                value_variances=(0.0, 0.0),
                improved_probabilities=(0.5, 0.5),
                selected_candidate_id=_candidates()[0],
            ),
            r"align|length|candidate|sample",
        ),
    ],
)
def test_candidate_collections_reject_duplicates_and_misalignment(factory, meaning) -> None:
    with pytest.raises(ValueError, match=meaning):
        factory()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_numeric_values_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"finite|logit"):
        PolicyValueOutput(
            schema_version=1,
            candidate_ids=_candidates()[:1],
            policy_logits=(bad,),
            value=0.0,
        )
    with pytest.raises(ValueError, match=r"finite|number|value"):
        NumberCandidateID(schema_version=1, value=bad)
    with pytest.raises(ValueError, match=r"finite|mean|value"):
        SearchOutcome(
            schema_version=1,
            candidate_ids=_candidates()[:1],
            prior_probabilities=(1.0,),
            sample_counts=(1,),
            mean_values=(bad,),
            value_variances=(0.0,),
            improved_probabilities=(1.0,),
            selected_candidate_id=_candidates()[0],
        )


def test_public_snapshot_rejects_nested_non_finite_public_state_value() -> None:
    with pytest.raises(ValueError, match="serialized contract numbers must be finite"):
        PublicSnapshot(
            schema_version=2,
            viewer=_viewer(),
            map_id="forgotten_island",
            game_type="QUICK",
            public_state={"teams": [{"score": math.inf}]},
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PolicyValueOutput(
            schema_version=1,
            candidate_ids=_candidates()[:1],
            policy_logits=(0.0,),
            value=1.01,
        ),
        lambda: SearchOutcome(
            schema_version=1,
            candidate_ids=_candidates()[:2],
            prior_probabilities=(0.8, 0.8),
            sample_counts=(1, 1),
            mean_values=(0.0, 0.0),
            value_variances=(0.0, 0.0),
            improved_probabilities=(0.5, 0.5),
            selected_candidate_id=_candidates()[0],
        ),
    ],
)
def test_probabilities_and_value_targets_are_bounded_and_normalized(factory) -> None:
    with pytest.raises(ValueError, match=r"probab|sum|value|target|range"):
        factory()
