"""Behavioral contract for complete-game joint policy/value datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automata.models.contracts import (
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    Viewer,
    canonical_json_bytes,
)
from automata.training.dataset import (
    JointDatasetRecorder,
    JointDatasetRow,
    joint_decision_id,
    load_joint_dataset,
    write_joint_dataset,
)
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget


def _candidate(name: str) -> EncodedCandidate:
    return EncodedCandidate(
        schema_version=1,
        candidate_id=OptionCandidateID(schema_version=1, option_id=name),
        selection=name,
    )


def _observation(*, perspective: str = "RED", reverse: bool = False) -> DecisionObservation:
    candidates: tuple[EncodedCandidate, ...] = (_candidate("hold"), _candidate("advance"))
    if reverse:
        candidates = tuple(reversed(candidates))
    return DecisionObservation(
        schema_version=3,
        state=LearnedObservation(
            schema_version=2,
            viewer=Viewer(schema_version=2, perspective_team=perspective),
            tokens=(
                ObservationToken(
                    schema_version=1,
                    local_ref="global:0",
                    kind="GLOBAL",
                    features={"map_id": "forgotten_island", "game_type": "QUICK"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:red:0",
                    kind="HERO",
                    features={"name": "Wasp", "team_id": "RED"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:blue:0",
                    kind="HERO",
                    features={"name": "Arien", "team_id": "BLUE"},
                ),
            ),
        ),
        decision_kind="INPUT",
        candidates=candidates,
    )


def _identity(*, game_id: str = "game-7", world_seed: int = 7) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "world_seed": world_seed,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": "generation-1",
        "source_revision": "abc123",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "search-config-1",
        "generator_config_id": "generator-config-1",
    }


def _row(**changes: Any) -> JointDatasetRow:
    observation = changes.pop("observation", _observation())
    identity = _identity()
    identity.update(changes.pop("identity", {}))
    values: dict[str, Any] = {
        "schema_version": 1,
        **identity,
        "decision_index": 0,
        "perspective_team": "RED",
        "observation": observation,
        "policy_source": "HEURISTIC",
        "policy_target": (1.0, 0.0),
        "selected_candidate_id": observation.candidates[0].candidate_id,
        "selected_selection": observation.candidates[0].selection,
        "action_stats": None,
        "terminal_winner": "RED",
        "value_target": 1,
    }
    values.update(changes)
    values["decision_id"] = joint_decision_id(
        **{name: values[name] for name in identity},
        decision_index=values["decision_index"],
    )
    return JointDatasetRow(**values)


def _recorder(path: Path, **changes: Any) -> JointDatasetRecorder:
    values = _identity()
    values.update(changes)
    return JointDatasetRecorder(path, **values)


def _record_one(recorder: JointDatasetRecorder, *, perspective: str = "RED") -> None:
    observation = _observation(perspective=perspective)
    recorder.record_decision(
        observation=observation,
        policy_source="HEURISTIC",
        policy_target=(0.75, 0.25),
        selected_candidate_id=observation.candidates[0].candidate_id,
        selected_selection="hold",
    )


def test_row_is_immutable_canonical_and_preserves_exact_candidate_order() -> None:
    observation = _observation()
    stats = tuple(
        SearchActionTarget(
            schema_version=1,
            candidate=candidate,
            sample_count=count,
            mean_value=0,
            value_variance=0,
            selected=index == 0,
        )
        for index, (candidate, count) in enumerate(zip(observation.candidates, (3, 1), strict=True))
    )
    row = _row(observation=observation, action_stats=stats)

    restored = JointDatasetRow.model_validate_json(canonical_json_bytes(row))

    assert restored == row
    assert [candidate.selection for candidate in restored.observation.candidates] == [
        "hold",
        "advance",
    ]
    assert restored.policy_target == (1.0, 0.0)
    assert restored.action_stats is not None
    assert [stat.sample_count for stat in restored.action_stats] == [3, 1]
    with pytest.raises(ValidationError):
        row.value_target = -1


@pytest.mark.parametrize(
    "changes",
    [
        {"policy_target": (1.0,)},
        {"policy_target": (0.6, 0.3)},
        {"policy_target": (-0.1, 1.1)},
        {"selected_candidate_id": OptionCandidateID(schema_version=1, option_id="illegal")},
        {"selected_selection": "advance"},
        {"perspective_team": "BLUE"},
        {"terminal_winner": "BLUE", "value_target": 1},
        {"map_id": "wrong_map"},
        {"red_composition": ("PrivateHeroId",)},
    ],
    ids=[
        "policy-length",
        "policy-normalization",
        "negative-policy",
        "illegal-selection-id",
        "selection-identity",
        "perspective",
        "terminal-value",
        "map-metadata",
        "composition-metadata",
    ],
)
def test_row_rejects_misaligned_policy_selection_outcome_and_metadata(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _row(**changes)


def test_row_rejects_misaligned_optional_generic_action_stats() -> None:
    observation = _observation()
    stats = SearchPolicyTarget(
        schema_version=1,
        actions=(
            SearchActionTarget(
                schema_version=1,
                candidate=observation.candidates[1],
                sample_count=1,
                mean_value=0,
                value_variance=0,
                selected=True,
            ),
            SearchActionTarget(
                schema_version=1,
                candidate=observation.candidates[0],
                sample_count=1,
                mean_value=0,
                value_variance=0,
            ),
        ),
    )

    with pytest.raises(ValidationError):
        _row(action_stats=stats.actions)


def test_decision_id_is_stable_from_game_identity_and_decision_index() -> None:
    first = _row()
    same = _row(
        policy_target=(0.25, 0.75),
        selected_selection="advance",
        selected_candidate_id=_observation().candidates[1].candidate_id,
    )
    later = _row(decision_index=1)

    assert first.decision_id == same.decision_id
    assert first.decision_id != later.decision_id
    assert len(first.decision_id) == 64


def test_recorder_publishes_complete_game_atomically_with_contiguous_indexes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "joint.jsonl"
    recorder = _recorder(path)
    _record_one(recorder, perspective="RED")
    _record_one(recorder, perspective="BLUE")
    assert not path.exists()

    recorder.record_outcome(winner="RED", rounds=4, reason="game_over")

    dataset = load_joint_dataset(path)
    assert [row.decision_index for row in dataset.rows] == [0, 1]
    assert [row.value_target for row in dataset.rows] == [1, -1]
    assert dataset.game_ids == ("game-7",)
    assert tuple(dataset.rows_by_game) == ("game-7",)
    assert dataset.metadata.dataset_digest == dataset.digest
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("reason", ["max_steps", "timeout", "exception", "interruption"])
def test_recorder_discards_every_incomplete_game(tmp_path: Path, reason: str) -> None:
    path = tmp_path / f"{reason}.jsonl"
    recorder = _recorder(path)
    _record_one(recorder)

    recorder.record_outcome(winner=None, rounds=2, reason=reason)

    assert not path.exists()


def test_recorder_context_exception_and_close_discard_without_publication(tmp_path: Path) -> None:
    exception_path = tmp_path / "exception.jsonl"
    with pytest.raises(RuntimeError), _recorder(exception_path) as recorder:
        _record_one(recorder)
        raise RuntimeError("interrupted")
    assert not exception_path.exists()

    close_path = tmp_path / "close.jsonl"
    recorder = _recorder(close_path)
    _record_one(recorder)
    recorder.close()
    assert not close_path.exists()


def test_recorder_refuses_overwrite_and_equal_input_produces_equal_bytes(tmp_path: Path) -> None:
    paths = (tmp_path / "one.jsonl", tmp_path / "two.jsonl")
    for path in paths:
        recorder = _recorder(path)
        _record_one(recorder)
        recorder.record_outcome(winner="RED", rounds=1, reason="game_over")
    assert paths[0].read_bytes() == paths[1].read_bytes()

    with pytest.raises(FileExistsError):
        _recorder(paths[0])


def test_compressed_recorder_round_trip_is_deterministic_and_atomic(tmp_path: Path) -> None:
    paths = (tmp_path / "one.jsonl.zst", tmp_path / "two.jsonl.zst")
    for path in paths:
        recorder = _recorder(path)
        _record_one(recorder)
        assert not path.exists()
        recorder.record_outcome(winner="RED", rounds=1, reason="game_over")

    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert load_joint_dataset(paths[0]).rows == load_joint_dataset(paths[1]).rows
    assert not list(tmp_path.glob("*.tmp"))


def test_dataset_digest_is_semantic_across_plain_and_compressed_files(tmp_path: Path) -> None:
    plain = tmp_path / "joint.jsonl"
    compressed = tmp_path / "joint.jsonl.zst"
    row = _row()

    write_joint_dataset(plain, (row,))
    write_joint_dataset(compressed, (row,))

    plain_dataset = load_joint_dataset(plain)
    compressed_dataset = load_joint_dataset(compressed)
    assert compressed_dataset.canonical_bytes() == plain.read_bytes()
    assert compressed_dataset.digest == plain_dataset.digest
    assert compressed.read_bytes() != plain.read_bytes()


def test_loader_rejects_malformed_compressed_input(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl.zst"
    path.write_bytes(b"not a zstandard frame")

    with pytest.raises(ValueError, match="compressed joint dataset"):
        load_joint_dataset(path)


def _write_rows(path: Path, rows: list[dict[str, Any]], *, final_newline: bool = True) -> None:
    payload = b"\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        for row in rows
    )
    path.write_bytes(payload + (b"\n" if final_newline else b""))


@pytest.mark.parametrize(
    "mutation",
    [
        "malformed-interior",
        "truncated-final",
        "duplicate-id",
        "conflicting-game",
        "seed-reuse",
        "noncontiguous",
    ],
)
def test_strict_loader_rejects_corrupt_or_conflicting_datasets(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "bad.jsonl"
    row0 = _row().model_dump(mode="json")
    row1 = _row(decision_index=1).model_dump(mode="json")
    if mutation == "malformed-interior":
        path.write_bytes(
            canonical_json_bytes(_row())
            + b"\n{"
            + canonical_json_bytes(_row(decision_index=1))
            + b"\n"
        )
    elif mutation == "truncated-final":
        _write_rows(path, [row0], final_newline=False)
    elif mutation == "duplicate-id":
        _write_rows(path, [row0, row0])
    elif mutation == "conflicting-game":
        row1["terminal_winner"] = "BLUE"
        row1["value_target"] = 1
        _write_rows(path, [row0, row1])
    elif mutation == "seed-reuse":
        other = _row(identity=_identity(game_id="game-8"), decision_index=0).model_dump(mode="json")
        _write_rows(path, [row0, other])
    else:
        row2 = _row(decision_index=2).model_dump(mode="json")
        _write_rows(path, [row0, row2])

    with pytest.raises(ValueError):
        load_joint_dataset(path)


def test_loader_rejects_candidate_policy_tampering_and_exposes_canonical_digest(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.jsonl"
    recorder = _recorder(valid)
    _record_one(recorder)
    recorder.record_outcome(winner=None, rounds=5, reason="game_over")
    dataset = load_joint_dataset(valid)

    assert dataset.canonical_bytes() == valid.read_bytes()
    assert dataset.metadata.row_count == 1
    assert dataset.metadata.schema_version == 1

    bad = tmp_path / "bad.jsonl"
    payload = dataset.rows[0].model_dump(mode="json")
    payload["policy_target"] = [1.0]
    _write_rows(bad, [payload])
    with pytest.raises(ValueError):
        load_joint_dataset(bad)


def test_serialized_rows_contain_only_contract_data_not_runtime_state() -> None:
    payload = json.loads(canonical_json_bytes(_row()))
    encoded = json.dumps(payload)

    assert "execution_stack" not in encoded
    assert "request_id" not in encoded
    assert "rng_state" not in encoded
    assert "player_id" not in encoded
