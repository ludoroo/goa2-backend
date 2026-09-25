"""Focused contract for pinned offline policy-artifact evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from automata.decision import DecisionSemanticRole
from automata.models.contracts import (
    ArtifactScope,
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    Viewer,
)
from automata.models.shared_encoder.artifacts import export_model_artifact
from automata.models.shared_encoder.model import JointModelConfig, JointPolicyValueModel
from automata.models.shared_encoder.schema import TensorFeatureSchema
from automata.policy_ranking import PolicyRankingSnapshot
from automata.training.dataset import JointDatasetRow, joint_decision_id, write_joint_dataset
from automata.training.evaluate_artifact import evaluate_policy_artifact, main
from automata.training.indexed_dataset import open_indexed_dataset
from automata.training.splits import JointSplitConfig, split_indexed_joint_dataset


def _row(game: int) -> JointDatasetRow:
    candidate_ids = (
        OptionCandidateID(schema_version=1, option_id="hold"),
        OptionCandidateID(schema_version=1, option_id="advance"),
    )
    observation = DecisionObservation(
        schema_version=4,
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
                    features={
                        "name": "Wasp",
                        "team_id": "RED",
                        "team_ref": "team:red",
                    },
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:blue",
                    kind="HERO",
                    features={
                        "name": "Arien",
                        "team_id": "BLUE",
                        "team_ref": "team:blue",
                    },
                ),
            ),
        ),
        decision_kind="INPUT",
        input_request_type="SELECT_OPTION",
        can_skip=False,
        semantic_role=DecisionSemanticRole.OPTION_SELECTION,
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
        "generation_id": "generation-v4",
        "source_revision": "abc123",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "search-1",
        "generator_config_id": "generator-1",
    }
    selected = game % 2
    target = (1.0, 0.0) if selected == 0 else (0.0, 1.0)
    return JointDatasetRow(
        schema_version=2,
        decision_id=joint_decision_id(**identity, decision_index=0),
        **identity,
        decision_index=0,
        perspective_team="RED",
        observation=observation,
        policy_source="HEURISTIC",
        policy_target=target,
        selected_candidate_id=candidate_ids[selected],
        selected_selection=("hold", "advance")[selected],
        terminal_winner="RED",
        value_target=1,
    )


def _evaluation_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    dataset_path = tmp_path / "dataset.jsonl"
    index_path = tmp_path / "index"
    split_path = tmp_path / "split.json"
    artifact_path = tmp_path / "artifact"
    write_joint_dataset(dataset_path, (_row(1), _row(2), _row(3), _row(4)))
    dataset = open_indexed_dataset(
        dataset_path,
        index_path,
        training_chunk_size=2,
        index_workers=1,
        show_progress=False,
    )
    split = split_indexed_joint_dataset(
        dataset,
        config=JointSplitConfig(
            seed=91,
            validation_fraction=0.5,
            seed_purpose="training",
        ),
    )
    split_path.write_bytes(split.canonical_bytes())

    schema = TensorFeatureSchema.current()
    model = JointPolicyValueModel(
        schema=schema,
        config=JointModelConfig(
            model_version=2,
            schema_digest=schema.digest,
            token_width=4,
            state_width=6,
            candidate_width=4,
            message_passing_layers=1,
        ),
    )
    manifest = export_model_artifact(
        artifact_path,
        model=model,
        schema=schema,
        scope=ArtifactScope(
            supported_heroes=("Arien", "Wasp"),
            supported_maps=("forgotten_island",),
            supported_game_types=("QUICK",),
            hero_adapter_versions={"generic": 1, "Arien": 1, "Wasp": 1},
            map_schema_version=1,
        ),
        runtime_compatibility_version=2,
    )
    return dataset_path, index_path, split_path, artifact_path, manifest.model_digest


def test_evaluate_policy_artifact_uses_exact_validation_membership_and_is_canonical(
    tmp_path: Path,
) -> None:
    dataset, index, split, artifact, digest = _evaluation_fixture(tmp_path)

    first = evaluate_policy_artifact(
        artifact_path=artifact,
        artifact_digest=digest,
        dataset_path=dataset,
        dataset_index_path=index,
        split_manifest_path=split,
        index_workers=1,
        decisions_per_chunk=2,
    )
    second = evaluate_policy_artifact(
        artifact_path=artifact,
        artifact_digest=digest,
        dataset_path=dataset,
        dataset_index_path=index,
        split_manifest_path=split,
        index_workers=1,
        decisions_per_chunk=2,
    )

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.artifact_digest == digest
    assert first.overall.count == 2
    assert first.overall.game_count == 2
    assert first.overall.multi_candidate_count == 2
    assert first.by_semantic_role[DecisionSemanticRole.OPTION_SELECTION] == first.overall
    assert str(tmp_path).encode() not in first.canonical_bytes()

    out = tmp_path / "evidence.json"
    assert (
        main(
            [
                "--artifact",
                str(artifact),
                "--artifact-digest",
                digest,
                "--dataset",
                str(dataset),
                "--dataset-index",
                str(index),
                "--split-manifest",
                str(split),
                "--index-workers",
                "1",
                "--decisions-per-chunk",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert out.read_bytes() == first.canonical_bytes()
    assert PolicyRankingSnapshot.from_canonical_bytes(out.read_bytes()) == first


def test_evaluate_policy_artifact_rejects_pin_and_legacy_or_incompatible_schema(
    tmp_path: Path,
) -> None:
    dataset, index, split, artifact, digest = _evaluation_fixture(tmp_path)

    with pytest.raises(ValueError, match="pinned digest"):
        evaluate_policy_artifact(
            artifact_path=artifact,
            artifact_digest="0" * 64,
            dataset_path=dataset,
            dataset_index_path=index,
            split_manifest_path=split,
            index_workers=1,
            decisions_per_chunk=2,
        )

    manifest_path = artifact / "manifest.json"
    original = json.loads(manifest_path.read_bytes())
    legacy = {**original, "observation_schema_version": 3}
    manifest_path.write_bytes(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="observation_schema_version"):
        evaluate_policy_artifact(
            artifact_path=artifact,
            artifact_digest=digest,
            dataset_path=dataset,
            dataset_index_path=index,
            split_manifest_path=split,
            index_workers=1,
            decisions_per_chunk=2,
        )

    incompatible = {
        **original,
        "tensor_schema_digest": "0" * 64,
    }
    manifest_path.write_bytes(
        json.dumps(incompatible, sort_keys=True, separators=(",", ":")).encode()
    )
    with pytest.raises(ValueError, match="incompatible tensor schemas"):
        evaluate_policy_artifact(
            artifact_path=artifact,
            artifact_digest=digest,
            dataset_path=dataset,
            dataset_index_path=index,
            split_manifest_path=split,
            index_workers=1,
            decisions_per_chunk=2,
        )
