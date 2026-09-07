"""End-to-end contract for deterministic joint learned-model training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from automata.models.contracts import (
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    RuntimeRequirements,
    Viewer,
    canonical_json_bytes,
)
from automata.models.shared_encoder.runtime import SharedEncoderRuntime
from automata.training.dataset import JointDatasetRow, joint_decision_id
from automata.training.trainer import JointTrainingConfig, batch_decision_weights, train_joint


def _row(game: int, decision: int = 0) -> JointDatasetRow:
    candidate_ids = (
        OptionCandidateID(schema_version=1, option_id="hold"),
        OptionCandidateID(schema_version=1, option_id="advance"),
    )
    observation = DecisionObservation(
        schema_version=3,
        state=LearnedObservation(
            schema_version=2,
            viewer=Viewer(schema_version=2, perspective_team="RED"),
            tokens=(
                ObservationToken(
                    schema_version=1,
                    local_ref="global:0",
                    kind="GLOBAL",
                    features={"map_id": "forgotten_island", "game_type": "QUICK"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="team:red",
                    kind="TEAM",
                    features={"team_id": "RED", "relation": "OWN"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="team:blue",
                    kind="TEAM",
                    features={"team_id": "BLUE", "relation": "ENEMY"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:red",
                    kind="HERO",
                    features={"name": "Wasp", "team_id": "RED", "team_ref": "team:red"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:blue",
                    kind="HERO",
                    features={"name": "Arien", "team_id": "BLUE", "team_ref": "team:blue"},
                ),
            ),
        ),
        decision_kind="INPUT",
        candidates=tuple(
            EncodedCandidate(schema_version=1, candidate_id=item, selection=item.option_id)
            for item in candidate_ids
        ),
    )
    identity: dict[str, Any] = {
        "game_id": f"game-{game}",
        "world_seed": 20_000 + game,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": "generation-1",
        "source_revision": "abc123",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "search-1",
        "generator_config_id": "generator-1",
    }
    target = (1.0, 0.0) if game % 2 else (0.0, 1.0)
    return JointDatasetRow(
        schema_version=1,
        decision_id=joint_decision_id(**identity, decision_index=decision),
        **identity,
        decision_index=decision,
        perspective_team="RED",
        observation=observation,
        policy_source="HEURISTIC",
        policy_target=target,
        selected_candidate_id=candidate_ids[target.index(1.0)],
        selected_selection=("hold", "advance")[target.index(1.0)],
        terminal_winner="RED" if game % 2 else "BLUE",
        value_target=1 if game % 2 else -1,
    )


def _dataset(path: Path) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(_row(game)) + b"\n" for game in range(1, 5)))


def _paths(root: Path) -> dict[str, Path]:
    return {
        "dataset_path": root / "joint.jsonl",
        "split_manifest_path": root / "split.json",
        "checkpoint_path": root / "checkpoint.pt",
        "run_manifest_path": root / "run.json",
        "artifact_path": root / "artifact",
    }


def _config(paths: dict[str, Path], **changes: object) -> JointTrainingConfig:
    values: dict[str, Any] = {
        **paths,
        "seed": 91,
        "epochs": 2,
        "games_per_batch": 1,
        "learning_rate": 0.01,
        "token_width": 4,
        "state_width": 6,
        "candidate_width": 4,
        "message_passing_layers": 1,
        "dataset_seed_purpose": "training",
    }
    values.update(changes)
    return JointTrainingConfig(**values)


def test_success_writes_canonical_manifest_and_immutable_inference_artifact(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _dataset(paths["dataset_path"])

    result = train_joint(_config(paths))

    manifest_bytes = paths["run_manifest_path"].read_bytes()
    manifest = json.loads(manifest_bytes)
    assert result.status == "SUCCEEDED"
    assert (
        manifest_bytes
        == json.dumps(
            manifest, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    )
    assert manifest["status"] == "SUCCEEDED"
    assert manifest["provenance"]["dataset_digest"]
    assert manifest["provenance"]["split_manifest"]["memberships"]
    assert manifest["metrics"]["validation"]["policy"]["overall"]["count"] > 0
    assert (paths["artifact_path"] / "manifest.json").is_file()
    assert paths["checkpoint_path"].is_file()
    runtime = SharedEncoderRuntime.from_artifact(
        paths["artifact_path"],
        requirements=RuntimeRequirements(
            runtime_compatibility_version=1,
            observation_schema_version=3,
            map_schema_version=1,
            heroes=frozenset({"Wasp", "Arien"}),
            map_id="forgotten_island",
            game_type="QUICK",
            hero_adapter_versions={"generic": 1, "Wasp": 1, "Arien": 1},
        ),
    )
    output = runtime.evaluate(_row(1).observation)
    assert output.candidate_ids == tuple(
        candidate.candidate_id for candidate in _row(1).observation.candidates
    )


def test_resume_is_bit_exact_and_configuration_mismatch_fails_closed(tmp_path: Path) -> None:
    full_paths = _paths(tmp_path / "full")
    resumed_paths = _paths(tmp_path / "resumed")
    full_paths["dataset_path"].parent.mkdir()
    resumed_paths["dataset_path"].parent.mkdir()
    _dataset(full_paths["dataset_path"])
    _dataset(resumed_paths["dataset_path"])

    full = train_joint(_config(full_paths))
    interrupted = train_joint(_config(resumed_paths), stop_after_steps=2)
    resumed = train_joint(_config(resumed_paths))

    assert interrupted.status == "INTERRUPTED"
    assert resumed.status == "SUCCEEDED"
    assert resumed.model_digest == full.model_digest

    mismatch_paths = _paths(tmp_path / "mismatch")
    mismatch_paths["dataset_path"].parent.mkdir()
    _dataset(mismatch_paths["dataset_path"])
    train_joint(_config(mismatch_paths), stop_after_steps=1)
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        train_joint(_config(mismatch_paths, learning_rate=0.02))
    assert not mismatch_paths["artifact_path"].exists()
    assert json.loads(mismatch_paths["run_manifest_path"].read_bytes())["status"] == "FAILED"


def test_persisted_global_decision_weights_survive_uneven_mini_batches() -> None:
    rows = (_row(1), _row(1, 1), _row(2))
    global_weights = (0.25, 0.25, 0.5)

    first = batch_decision_weights(rows[:2], rows, global_weights)
    second = batch_decision_weights(rows[2:], rows, global_weights)

    assert first == pytest.approx((0.25, 0.25))
    assert second == pytest.approx((0.5,))
    assert sum(first) == pytest.approx(sum(second))
