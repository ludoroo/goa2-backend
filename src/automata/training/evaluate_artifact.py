"""Evaluate one pinned native-v4 artifact on an exact validation split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from automata.models.contracts import (
    CURRENT_DECISION_OBSERVATION_SCHEMA_VERSION,
    CURRENT_MAP_SCHEMA_VERSION,
    CURRENT_RUNTIME_COMPATIBILITY_VERSION,
    RuntimeRequirements,
    canonical_json_bytes,
)
from automata.models.shared_encoder.artifacts import (
    ModelArtifactManifest,
    load_model_artifact,
)
from automata.observation.hero_adapters import HeroObservationAdapterRegistry
from automata.policy_ranking import PolicyRankingSnapshot, snapshot_from_policy_metrics
from automata.training.evaluation import add_chunk_metric_inputs
from automata.training.indexed_dataset import (
    DEFAULT_TRAINING_CHUNK_SIZE,
    IndexedJointDataset,
    open_indexed_dataset,
)
from automata.training.io import atomic_write_bytes
from automata.training.metrics import JointMetricsAccumulator
from automata.training.splits import (
    JointSplitManifest,
    apply_indexed_joint_split_manifest,
)


def _validation_game_ids(manifest: JointSplitManifest) -> tuple[str, ...]:
    return tuple(item.game_id for item in manifest.memberships if item.split == "validation")


def _runtime_requirements(dataset: IndexedJointDataset) -> RuntimeRequirements:
    games = dataset.manifest.games
    if not games:
        raise ValueError("artifact evaluation requires a non-empty indexed dataset")
    heroes = frozenset(
        hero for game in games for hero in (*game.red_composition, *game.blue_composition)
    )
    adapters = HeroObservationAdapterRegistry()
    registered = adapters.registered_versions
    return RuntimeRequirements(
        runtime_compatibility_version=CURRENT_RUNTIME_COMPATIBILITY_VERSION,
        observation_schema_version=CURRENT_DECISION_OBSERVATION_SCHEMA_VERSION,
        map_schema_version=CURRENT_MAP_SCHEMA_VERSION,
        heroes=heroes,
        map_id=games[0].map_id,
        game_type=games[0].game_type,
        hero_adapter_versions={
            "generic": adapters.generic_version,
            **{hero: registered.get(hero, adapters.generic_version) for hero in heroes},
        },
    )


def _load_split(path: Path, dataset: IndexedJointDataset) -> JointSplitManifest:
    payload = path.read_bytes()
    manifest = JointSplitManifest.model_validate_json(payload)
    if payload != manifest.canonical_bytes():
        raise ValueError("split manifest is not canonical JSON")
    return apply_indexed_joint_split_manifest(dataset, manifest)


def evaluate_policy_artifact(
    *,
    artifact_path: Path,
    artifact_digest: str,
    dataset_path: Path,
    split_manifest_path: Path,
    dataset_index_path: Path | None = None,
    index_workers: int = 4,
    decisions_per_chunk: int = DEFAULT_TRAINING_CHUNK_SIZE,
) -> PolicyRankingSnapshot:
    """Return canonical gate evidence for one native-v4 artifact validation run."""

    dataset = open_indexed_dataset(
        dataset_path,
        dataset_index_path or Path(f"{dataset_path}.index"),
        training_chunk_size=decisions_per_chunk,
        index_workers=index_workers,
        show_progress=False,
    )
    split = _load_split(split_manifest_path, dataset)
    game_ids = _validation_game_ids(split)
    if not game_ids:
        raise ValueError("artifact evaluation requires non-empty validation membership")

    manifest_payload = (artifact_path / "manifest.json").read_bytes()
    manifest = ModelArtifactManifest.model_validate_json(manifest_payload)
    if manifest_payload != canonical_json_bytes(manifest):
        raise ValueError("artifact manifest is not canonical JSON")
    if manifest.model_digest != artifact_digest:
        raise ValueError("artifact model digest does not match pinned digest")
    if manifest.observation_schema_version != CURRENT_DECISION_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            "artifact evaluation requires a native current-schema artifact "
            f"(observation v{CURRENT_DECISION_OBSERVATION_SCHEMA_VERSION})"
        )
    index_identity = (
        dataset.manifest.tensor_schema_id,
        dataset.manifest.tensor_schema_version,
        dataset.manifest.tensor_schema_digest,
    )
    artifact_identity = (
        manifest.tensor_schema_id,
        manifest.tensor_schema_version,
        manifest.tensor_schema_digest,
    )
    if artifact_identity != index_identity:
        raise ValueError("artifact and indexed dataset use incompatible tensor schemas")
    loaded = load_model_artifact(
        artifact_path,
        requirements=_runtime_requirements(dataset),
    )

    supported_maps = set(loaded.manifest.supported_maps)
    supported_game_types = set(loaded.manifest.supported_game_types)
    dataset_maps = {dataset.game(game_id).map_id for game_id in game_ids}
    dataset_game_types = {dataset.game(game_id).game_type for game_id in game_ids}
    if not dataset_maps <= supported_maps or not dataset_game_types <= supported_game_types:
        raise ValueError("artifact scope does not cover the validation membership")

    accumulator = JointMetricsAccumulator()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    loaded.model.eval()
    with torch.inference_mode():
        for game_id in game_ids:
            for chunk in dataset.iter_game_training_chunks(game_id):
                add_chunk_metric_inputs(accumulator, loaded.model(chunk.batch), chunk)
    policy_metrics = accumulator.compute()["policy"]
    return snapshot_from_policy_metrics(
        policy_metrics,
        artifact_digest=artifact_digest,
        dataset_digest=dataset.digest,
        split_digest=split.digest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--index-workers", type=int, default=4)
    parser.add_argument(
        "--decisions-per-chunk",
        type=int,
        default=DEFAULT_TRAINING_CHUNK_SIZE,
        help="must match the training index chunk size when reusing --dataset-index",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot = evaluate_policy_artifact(
        artifact_path=args.artifact,
        artifact_digest=args.artifact_digest,
        dataset_path=args.dataset,
        dataset_index_path=args.dataset_index,
        split_manifest_path=args.split_manifest,
        index_workers=args.index_workers,
        decisions_per_chunk=args.decisions_per_chunk,
    )
    payload = snapshot.canonical_bytes()
    if args.out is None:
        sys.stdout.buffer.write(payload + b"\n")
    else:
        atomic_write_bytes(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_policy_artifact", "main"]
