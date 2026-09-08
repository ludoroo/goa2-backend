"""Crash-safe coordinator from persistent self-play to an immutable candidate.

The coordinator deliberately owns no process pool.  Generation, training, and
offline validation are explicit callbacks so callers can choose process
boundaries while this module owns identities, reconciliation, and publication.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from automata.models.shared_encoder.artifacts import ModelArtifactManifest
from automata.training.io import atomic_write_bytes as _atomic_write
from automata.training.io import canonical_json_bytes as _canonical
from automata.training.io import content_digest as _digest
from automata.training.io import file_digest as _file_digest

from .dataset import JointDataset, JointDatasetRow, load_joint_dataset, write_joint_dataset
from .generation import CheckpointRow, WorkerSpec
from .registry import CandidateManifest, CandidateMetadata, ChampionRegistry
from .replay_buffer import ReplayBuffer
from .splits import JointSplitManifest, apply_joint_split_manifest
from .trainer import JointTrainingConfig, JointTrainingResult

StageName = Literal[
    "SELF_PLAY", "RECONCILE", "REPLAY_SAMPLE", "TRAIN", "OFFLINE_VALIDATE", "CANDIDATE"
]
_STAGES: tuple[StageName, ...] = (
    "SELF_PLAY",
    "RECONCILE",
    "REPLAY_SAMPLE",
    "TRAIN",
    "OFFLINE_VALIDATE",
    "CANDIDATE",
)


class PipelineError(RuntimeError):
    """The run cannot safely proceed from its durable state."""


class StalePipelineProductError(PipelineError):
    """A completed product no longer matches its recorded identity."""


class PipelineInterruptedError(PipelineError):
    """A resumable process-heavy callback stopped before completion."""


@dataclass(frozen=True, slots=True)
class WorkerOutput:
    """Published files belonging to one ``SelfPlayWorker`` invocation."""

    worker_id: int
    checkpoint_path: Path
    fragment_paths: tuple[Path, ...]


class GenerateWorker(Protocol):
    def __call__(self, spec: WorkerSpec) -> WorkerOutput: ...


class TrainCandidate(Protocol):
    def __call__(self, config: JointTrainingConfig) -> JointTrainingResult: ...


class OfflineValidate(Protocol):
    def __call__(
        self, artifact_path: Path, training_manifest_path: Path
    ) -> Mapping[str, JsonValue]: ...


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    """Explicit process-heavy seams and an optional crash-test boundary."""

    generate_worker: GenerateWorker
    offline_validate: OfflineValidate
    train_candidate: TrainCandidate
    after_product: Callable[[StageName], None] = field(default=lambda _stage: None)


@dataclass(frozen=True, slots=True)
class ResumableGenerationPipelineConfig:
    root: Path
    workers: tuple[WorkerSpec, ...]
    parent_artifact: ModelArtifactManifest
    parent_candidate_digest: str
    candidate_generation: int
    replay_game_count: int
    replay_seed: int
    replay_snapshot_digest: str
    training: JointTrainingConfig
    training_identity: Mapping[str, JsonValue]
    source_revision: str
    source_tree_digest: str

    def __post_init__(self) -> None:
        if not self.workers:
            raise ValueError("pipeline requires at least one self-play worker")
        if len({worker.worker_id for worker in self.workers}) != len(self.workers):
            raise ValueError("pipeline worker IDs must be unique")
        generator_ids = {worker.config.generator_config_id for worker in self.workers}
        if len(generator_ids) != 1:
            raise ValueError("pipeline workers must share one generation configuration")
        generation = self.workers[0].config
        if generation.parent_model_digest != self.parent_artifact.model_digest:
            raise ValueError("worker parent model does not match parent artifact")
        if generation.source_revision != self.source_revision:
            raise ValueError("pipeline source revision does not match worker generation")
        if self.candidate_generation != generation.parent_generation + 1:
            raise ValueError("candidate generation must immediately follow its parent")
        for label, value in (
            ("parent_candidate_digest", self.parent_candidate_digest),
            ("replay_snapshot_digest", self.replay_snapshot_digest),
            ("source_tree_digest", self.source_tree_digest),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if self.replay_game_count <= 0 or self.replay_seed < 0:
            raise ValueError("replay game count must be positive and seed non-negative")
        if not self.source_revision:
            raise ValueError("source_revision must be non-empty")
        # Copy and JSON-validate caller-owned mutable metadata.
        copied = json.loads(_canonical(dict(self.training_identity)))
        object.__setattr__(self, "training_identity", copied)

    @property
    def journal_path(self) -> Path:
        return self.root / "stage-journal.json"

    @property
    def receipts_path(self) -> Path:
        return self.root / "self-play-outputs.json"

    @property
    def reconciled_path(self) -> Path:
        return self.root / "generation.jsonl.zst"

    @property
    def sample_manifest_path(self) -> Path:
        return self.root / "replay-sample.json"

    @property
    def validation_path(self) -> Path:
        return self.root / "offline-validation.json"


class StageRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["RUNNING", "COMPLETED", "FAILED"]
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error: str | None = None


class StageJournal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    pipeline_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stages: dict[StageName, StageRecord] = Field(default_factory=dict)


def _training_identity(config: JointTrainingConfig) -> dict[str, JsonValue]:
    paths = {
        "dataset_path",
        "split_manifest_path",
        "checkpoint_path",
        "run_manifest_path",
        "artifact_path",
    }
    values = {key: value for key, value in asdict(config).items() if key not in paths}
    return json.loads(_canonical(values))


class ResumableGenerationPipeline:
    """Run or resume the exact content-addressed generation pipeline."""

    def __init__(
        self,
        config: ResumableGenerationPipelineConfig,
        *,
        replay_buffer: ReplayBuffer,
        registry: ChampionRegistry,
        dependencies: PipelineDependencies,
    ) -> None:
        self.config = config
        self.replay_buffer = replay_buffer
        self.registry = registry
        self.dependencies = dependencies
        if dict(config.training_identity) != _training_identity(config.training):
            raise ValueError("declared training identity does not match training configuration")
        if config.training.dataset_path != config.root / "training-sample.jsonl.zst":
            raise ValueError(
                "training dataset path must be <pipeline root>/training-sample.jsonl.zst"
            )
        self.pipeline_digest = self._pipeline_digest()
        self._journal = self._load_journal()

    def run(self) -> CandidateManifest:
        self.config.root.mkdir(parents=True, exist_ok=True)
        self._stage("SELF_PLAY", self._self_play_input(), self._generate, self._generation_digest)
        self._stage(
            "RECONCILE",
            self._stage_output("SELF_PLAY"),
            self._reconcile,
            self._reconciled_digest,
        )
        self._stage(
            "REPLAY_SAMPLE",
            _digest(
                {
                    "generation": self._stage_output("RECONCILE"),
                    "snapshot": self.config.replay_snapshot_digest,
                    "count": self.config.replay_game_count,
                    "seed": self.config.replay_seed,
                }
            ),
            self._sample,
            self._sample_digest,
        )
        self._stage(
            "TRAIN",
            _digest(
                {
                    "sample": self._stage_output("REPLAY_SAMPLE"),
                    "config": dict(self.config.training_identity),
                }
            ),
            self._train,
            self._trained_digest,
        )
        self._stage(
            "OFFLINE_VALIDATE",
            self._stage_output("TRAIN"),
            self._validate,
            self._validation_digest,
        )
        self._stage(
            "CANDIDATE",
            _digest(
                {
                    "artifact": self._stage_output("TRAIN"),
                    "validation": self._stage_output("OFFLINE_VALIDATE"),
                    "metadata": self._metadata().model_dump(mode="json"),
                }
            ),
            self._publish,
            self._candidate_digest,
        )
        return self.registry.candidate(self._stage_output("CANDIDATE"))

    def _pipeline_digest(self) -> str:
        generation = self.config.workers[0].config
        return _digest(
            {
                "worker_specs": [
                    {
                        "worker_id": worker.worker_id,
                        "worker_config_id": worker.worker_config_id,
                        "games": [game.game_id(worker.config) for game in worker.games],
                    }
                    for worker in sorted(self.config.workers, key=lambda item: item.worker_id)
                ],
                "generator_config_id": generation.generator_config_id,
                "parent_candidate_digest": self.config.parent_candidate_digest,
                "parent_artifact_digest": self.config.parent_artifact.model_digest,
                "candidate_generation": self.config.candidate_generation,
                "replay_game_count": self.config.replay_game_count,
                "replay_seed": self.config.replay_seed,
                "replay_snapshot_digest": self.config.replay_snapshot_digest,
                "training": dict(self.config.training_identity),
                "source_revision": self.config.source_revision,
                "source_tree_digest": self.config.source_tree_digest,
            }
        )

    def _load_journal(self) -> StageJournal:
        path = self.config.journal_path
        if not path.exists():
            return StageJournal(pipeline_digest=self.pipeline_digest)
        payload = path.read_bytes()
        try:
            journal = StageJournal.model_validate_json(payload)
        except ValueError as exc:
            raise PipelineError("stage journal is invalid") from exc
        if payload != _canonical(journal):
            raise PipelineError("stage journal is not canonical JSON")
        if journal.pipeline_digest != self.pipeline_digest:
            raise StalePipelineProductError("pipeline inputs changed for an existing journal")
        return journal

    def _write_record(self, stage: StageName, record: StageRecord) -> None:
        stages = {**self._journal.stages, stage: record}
        self._journal = self._journal.model_copy(update={"stages": stages})
        _atomic_write(self.config.journal_path, _canonical(self._journal))

    def _stage(
        self,
        name: StageName,
        input_digest: str,
        execute: Callable[[], None],
        validate: Callable[[], str],
    ) -> None:
        record = self._journal.stages.get(name)
        if record is not None and record.input_digest != input_digest:
            raise StalePipelineProductError(f"{name} inputs changed")
        if record is not None and record.status == "COMPLETED":
            try:
                actual = validate()
            except Exception as exc:
                raise StalePipelineProductError(f"{name} product is invalid") from exc
            if actual != record.output_digest:
                raise StalePipelineProductError(f"{name} product identity changed")
            return

        # A callback may have published its product immediately before a crash.
        if record is not None:
            try:
                recovered = validate()
            except Exception as exc:
                if name == "TRAIN" and self.config.training.artifact_path.exists():
                    raise StalePipelineProductError("TRAIN product is invalid") from exc
            else:
                self._write_record(
                    name,
                    StageRecord(
                        status="COMPLETED", input_digest=input_digest, output_digest=recovered
                    ),
                )
                return

        self._write_record(name, StageRecord(status="RUNNING", input_digest=input_digest))
        try:
            execute()
            output_digest = validate()
            self.dependencies.after_product(name)
        except Exception as exc:
            self._write_record(
                name,
                StageRecord(
                    status="FAILED",
                    input_digest=input_digest,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise
        self._write_record(
            name,
            StageRecord(status="COMPLETED", input_digest=input_digest, output_digest=output_digest),
        )

    def _stage_output(self, stage: StageName) -> str:
        record = self._journal.stages.get(stage)
        if record is None or record.status != "COMPLETED" or record.output_digest is None:
            raise PipelineError(f"stage {stage} has no completed output")
        return record.output_digest

    def _self_play_input(self) -> str:
        return _digest([worker.worker_config_id for worker in self.config.workers])

    def _generate(self) -> None:
        outputs = [self.dependencies.generate_worker(spec) for spec in self.config.workers]
        by_id = {output.worker_id: output for output in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {
            item.worker_id for item in self.config.workers
        }:
            raise ValueError("self-play callback returned the wrong worker outputs")
        payload = {
            "schema_version": 1,
            "workers": [
                {
                    "worker_id": output.worker_id,
                    "checkpoint_path": str(output.checkpoint_path),
                    "fragment_paths": [str(path) for path in output.fragment_paths],
                }
                for output in sorted(outputs, key=lambda item: item.worker_id)
            ],
        }
        _atomic_write(self.config.receipts_path, _canonical(payload))

    def _outputs(self) -> tuple[WorkerOutput, ...]:
        payload = json.loads(self.config.receipts_path.read_bytes())
        if _canonical(payload) != self.config.receipts_path.read_bytes():
            raise ValueError("self-play output receipt is not canonical")
        if payload.get("schema_version") != 1 or not isinstance(payload.get("workers"), list):
            raise ValueError("self-play output receipt is invalid")
        return tuple(
            WorkerOutput(
                worker_id=item["worker_id"],
                checkpoint_path=Path(item["checkpoint_path"]),
                fragment_paths=tuple(Path(path) for path in item["fragment_paths"]),
            )
            for item in payload["workers"]
        )

    def _validated_generation(self) -> tuple[tuple[JointDatasetRow, ...], list[dict[str, Any]]]:
        specs = {spec.worker_id: spec for spec in self.config.workers}
        outputs = self._outputs()
        if {output.worker_id for output in outputs} != set(specs):
            raise ValueError("self-play receipt worker coverage mismatch")
        rows: list[JointDatasetRow] = []
        identities: list[dict[str, Any]] = []
        for output in outputs:
            spec = specs[output.worker_id]
            checkpoints: dict[str, CheckpointRow] = {}
            raw_checkpoint = output.checkpoint_path.read_bytes()
            if raw_checkpoint and not raw_checkpoint.endswith(b"\n"):
                raise ValueError("self-play checkpoint has a partial final row")
            for raw in raw_checkpoint.splitlines():
                checkpoint = CheckpointRow.model_validate_json(raw)
                if raw != _canonical(checkpoint):
                    raise ValueError("self-play checkpoint row is not canonical")
                if checkpoint.worker_config_id != spec.worker_config_id:
                    raise ValueError("self-play checkpoint worker identity mismatch")
                if checkpoint.game_id in checkpoints:
                    raise ValueError("self-play checkpoint contains duplicate games")
                checkpoints[checkpoint.game_id] = checkpoint
            expected = {game.game_id(spec.config) for game in spec.games}
            if set(checkpoints) != expected:
                raise ValueError("self-play checkpoint game coverage mismatch")
            fragments = {path.name: path for path in output.fragment_paths}
            if len(fragments) != len(output.fragment_paths):
                raise ValueError("self-play receipt contains duplicate fragments")
            seen: set[str] = set()
            for path in output.fragment_paths:
                dataset = load_joint_dataset(path)
                if len(dataset.game_ids) != 1:
                    raise ValueError("self-play fragment must contain one complete game")
                game_id = dataset.game_ids[0]
                game = next(
                    (item for item in spec.games if item.game_id(spec.config) == game_id), None
                )
                receipt = checkpoints.get(game_id)
                if game is None or receipt is None or game_id in seen:
                    raise ValueError("self-play fragment has no unique checkpoint")
                if receipt.fragment_digest != _file_digest(path) or receipt.row_count != len(
                    dataset.rows
                ):
                    raise ValueError("self-play fragment and checkpoint disagree")
                if any(
                    row.world_seed != game.world_seed
                    or row.map_id != game.map_id
                    or row.game_type != game.game_type
                    or row.red_composition != game.red_composition
                    or row.blue_composition != game.blue_composition
                    or row.generation_id != spec.config.generation_id
                    or row.source_revision != spec.config.source_revision
                    or row.dirty_tree_hash != spec.config.dirty_tree_hash
                    or row.source_model_digest != spec.config.parent_model_digest
                    or row.search_config_id != spec.config.search_config_id
                    or row.generator_config_id != spec.config.generator_config_id
                    or row.observation.schema_version != spec.config.observation_schema_version
                    for row in dataset.rows
                ):
                    raise ValueError(
                        "self-play fragment identity does not match worker specification"
                    )
                seen.add(game_id)
                rows.extend(dataset.rows)
                identities.append(
                    {
                        "worker_id": output.worker_id,
                        "game_id": game_id,
                        "fragment_digest": receipt.fragment_digest,
                        "row_count": receipt.row_count,
                    }
                )
            if seen != expected:
                raise ValueError("self-play fragment game coverage mismatch")
        rows.sort(key=lambda row: (row.world_seed, row.game_id, row.decision_index))
        return tuple(rows), sorted(
            identities, key=lambda item: (item["worker_id"], item["game_id"])
        )

    def _generation_digest(self) -> str:
        _rows, identities = self._validated_generation()
        return _digest(identities)

    def _reconcile(self) -> None:
        rows, _identities = self._validated_generation()
        decisions: dict[str, JointDatasetRow] = {}
        indexes: dict[str, list[int]] = {}
        for row in rows:
            previous = decisions.get(row.decision_id)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting self-play decision {row.decision_id}")
            decisions[row.decision_id] = row
            indexes.setdefault(row.game_id, []).append(row.decision_index)
        if any(sorted(values) != list(range(len(values))) for values in indexes.values()):
            raise ValueError("self-play decision indexes are not contiguous")
        ordered = sorted(
            decisions.values(), key=lambda row: (row.world_seed, row.game_id, row.decision_index)
        )
        write_joint_dataset(self.config.reconciled_path, ordered)

    def _reconciled_digest(self) -> str:
        return load_joint_dataset(self.config.reconciled_path).digest

    def _ensure_generation_in_replay(self, dataset: JointDataset) -> None:
        present = set(dataset.game_ids) & set(self.replay_buffer.game_ids)
        if present == set(dataset.game_ids):
            return
        if present:
            raise ValueError("replay buffer contains only part of this generation")
        self.replay_buffer.add_generation(dataset, champion_parent=self.config.parent_artifact)

    def _sample(self) -> None:
        generation = load_joint_dataset(self.config.reconciled_path)
        self._ensure_generation_in_replay(generation)
        sample = self.replay_buffer.sample(
            game_count=self.config.replay_game_count, seed=self.config.replay_seed
        )
        write_joint_dataset(self.config.training.dataset_path, sample.rows)
        dataset = load_joint_dataset(self.config.training.dataset_path)
        manifest = {
            "schema_version": 1,
            "dataset_digest": dataset.digest,
            "game_ids": list(sample.game_ids),
            "decision_weights": list(sample.decision_weights),
            "replay_snapshot_digest": self.config.replay_snapshot_digest,
        }
        _atomic_write(self.config.sample_manifest_path, _canonical(manifest))

    def _sample_digest(self) -> str:
        dataset = load_joint_dataset(self.config.training.dataset_path)
        payload = json.loads(self.config.sample_manifest_path.read_bytes())
        if _canonical(payload) != self.config.sample_manifest_path.read_bytes():
            raise ValueError("replay sample manifest is not canonical")
        if (
            payload.get("dataset_digest") != dataset.digest
            or payload.get("replay_snapshot_digest") != self.config.replay_snapshot_digest
        ):
            raise ValueError("replay sample manifest disagrees with its dataset")
        return _digest(payload)

    def _train(self) -> None:
        training = replace(
            self.config.training, decision_weights_path=self.config.sample_manifest_path
        )
        result = self.dependencies.train_candidate(training)
        if result.status == "INTERRUPTED":
            raise PipelineInterruptedError("joint training was interrupted")
        if result.model_digest is None:
            raise PipelineError("successful joint training omitted its model digest")

    def _training_manifest(self) -> dict[str, Any]:
        payload = json.loads(self.config.training.run_manifest_path.read_bytes())
        if _canonical(payload) != self.config.training.run_manifest_path.read_bytes():
            raise ValueError("training manifest is not canonical")
        if payload.get("status") != "SUCCEEDED" or not isinstance(payload.get("model_digest"), str):
            raise ValueError("training manifest is not successful")
        return payload

    def _artifact_manifest(self) -> ModelArtifactManifest:
        payload = (self.config.training.artifact_path / "manifest.json").read_bytes()
        manifest = ModelArtifactManifest.model_validate_json(payload)
        if payload != _canonical(manifest):
            raise ValueError("artifact manifest is not canonical")
        return manifest

    def _split_manifest(self) -> JointSplitManifest:
        path = self.config.training.split_manifest_path
        payload = path.read_bytes()
        manifest = JointSplitManifest.model_validate_json(payload)
        if payload != manifest.canonical_bytes():
            raise ValueError("training split manifest is not canonical")
        dataset = load_joint_dataset(self.config.training.dataset_path)
        splits = apply_joint_split_manifest(dataset, manifest)
        if not splits.rows("train") or not splits.rows("validation"):
            raise ValueError("training split requires non-empty train and validation evidence")
        return manifest

    def _artifact_provenance(self, artifact: ModelArtifactManifest) -> dict[str, Any]:
        path = self.config.training.artifact_path / "provenance.json"
        payload = path.read_bytes()
        provenance = json.loads(payload)
        if not isinstance(provenance, dict) or payload != _canonical(provenance):
            raise ValueError("artifact provenance is not canonical")
        record = artifact.files.get("provenance.json")
        if (
            record is None
            or "provenance.json" not in artifact.non_executable_files
            or record.length != len(payload)
            or record.sha256 != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError("artifact manifest does not bind its provenance")
        return provenance

    def _training_product_identity(self) -> dict[str, JsonValue]:
        dataset = load_joint_dataset(self.config.training.dataset_path)
        sample_payload = json.loads(self.config.sample_manifest_path.read_bytes())
        if self._sample_digest() != self._stage_output("REPLAY_SAMPLE"):
            raise ValueError("training sample identity changed")
        split = self._split_manifest()
        run = self._training_manifest()
        artifact = self._artifact_manifest()
        artifact_provenance = self._artifact_provenance(artifact)
        expected: dict[str, JsonValue] = {
            "dataset_digest": dataset.digest,
            "split_digest": split.digest,
            "split_manifest": split.model_dump(mode="json"),
            "config": dict(self.config.training_identity),
        }
        run_provenance = run.get("provenance")
        if not isinstance(run_provenance, dict) or any(
            run_provenance.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("training manifest provenance disagrees with pipeline inputs")
        if any(artifact_provenance.get(key) != value for key, value in expected.items()):
            raise ValueError("artifact provenance disagrees with pipeline inputs")
        if (
            run["model_digest"] != artifact.model_digest
            or artifact_provenance.get("metrics") != run.get("metrics")
            or artifact_provenance.get("step") != run.get("step")
        ):
            raise ValueError("training and artifact manifests disagree")
        if sample_payload.get("dataset_digest") != dataset.digest:
            raise ValueError("training dataset disagrees with replay sample")
        return {
            "model_digest": artifact.model_digest,
            "dataset_digest": dataset.digest,
            "replay_sample_digest": _file_digest(self.config.sample_manifest_path),
            "train_input_digest": self._journal.stages["TRAIN"].input_digest,
            "split_manifest_digest": split.digest,
            "training_config_digest": _digest(dict(self.config.training_identity)),
        }

    def _trained_digest(self) -> str:
        return _digest(self._training_product_identity())

    def _validate(self) -> None:
        identity = self._training_product_identity()
        metrics = dict(
            self.dependencies.offline_validate(
                self.config.training.artifact_path, self.config.training.run_manifest_path
            )
        )
        if not metrics:
            raise ValueError("offline validation must emit metrics")
        _atomic_write(
            self.config.validation_path,
            _canonical({"schema_version": 1, "training_product": identity, "metrics": metrics}),
        )

    def _validation(self) -> dict[str, JsonValue]:
        payload = json.loads(self.config.validation_path.read_bytes())
        if _canonical(payload) != self.config.validation_path.read_bytes():
            raise ValueError("offline validation result is not canonical")
        if payload.get("training_product") != self._training_product_identity() or not payload.get(
            "metrics"
        ):
            raise ValueError("offline validation result disagrees with trained artifact")
        return dict(payload["metrics"])

    def _validation_digest(self) -> str:
        self._validation()
        return _file_digest(self.config.validation_path)

    def _metadata(self) -> CandidateMetadata:
        generation = self.config.workers[0].config
        identity = self._training_product_identity()
        return CandidateMetadata(
            generation=self.config.candidate_generation,
            parent_champion_digest=self.config.parent_candidate_digest,
            training_data_digests=(
                str(identity["dataset_digest"]),
                str(identity["split_manifest_digest"]),
            ),
            source_revision=self.config.source_revision,
            source_tree_digest=self.config.source_tree_digest,
            search_config=dict(generation.search_config),
            training_config={
                **dict(self.config.training_identity),
                "replay_sample_digest": identity["replay_sample_digest"],
                "train_input_digest": identity["train_input_digest"],
                "split_manifest_digest": identity["split_manifest_digest"],
            },
            offline_metrics=self._validation(),
            arena_results={},
        )

    def _expected_candidate(self) -> CandidateManifest:
        artifact = self._artifact_manifest()
        metadata = self._metadata()
        values: dict[str, Any] = {
            "artifact_digest": artifact.model_digest,
            "generation": metadata.generation,
            "parent_champion_digest": metadata.parent_champion_digest,
            "observation_schema_version": artifact.observation_schema_version,
            "hero_adapter_versions": artifact.hero_adapter_versions,
            "map_schema_version": artifact.map_schema_version,
            "training_data_digests": tuple(metadata.training_data_digests),
            "source_revision": metadata.source_revision,
            "source_tree_digest": metadata.source_tree_digest,
            "search_config": metadata.search_config,
            "training_config": metadata.training_config,
            "supported_heroes": tuple(artifact.supported_heroes),
            "supported_maps": tuple(artifact.supported_maps),
            "supported_game_types": tuple(artifact.supported_game_types),
            "offline_metrics": metadata.offline_metrics,
            "arena_results": metadata.arena_results,
            "runtime_format": metadata.runtime_format,
            "runtime_compatibility_version": artifact.runtime_compatibility_version,
        }
        provisional = CandidateManifest.model_construct(digest="0" * 64, **values)
        digest = hashlib.sha256(
            _canonical(provisional.model_dump(mode="json", exclude={"digest"}))
        ).hexdigest()
        return CandidateManifest(digest=digest, **values)

    def _publish(self) -> None:
        expected = self._expected_candidate()
        try:
            existing = self.registry.candidate(expected.digest)
        except (FileNotFoundError, OSError, ValueError):
            published = self.registry.register_candidate(
                self.config.training.artifact_path, metadata=self._metadata()
            )
            if published != expected:
                raise ValueError("candidate publisher returned an unexpected manifest") from None
        else:
            if existing != expected:
                raise ValueError("existing candidate manifest has stale content")

    def _candidate_digest(self) -> str:
        expected = self._expected_candidate()
        if self.registry.candidate(expected.digest) != expected:
            raise ValueError("registered candidate manifest disagrees with pipeline inputs")
        return expected.digest


def run_generation_pipeline(
    config: ResumableGenerationPipelineConfig,
    *,
    replay_buffer: ReplayBuffer,
    registry: ChampionRegistry,
    dependencies: PipelineDependencies,
) -> CandidateManifest:
    """Run/resume generation through candidate publication for policy iteration.

    This is the public composition seam: callers provide concrete generation,
    training, validation, replay, and registry dependencies.  The policy-iteration
    command owns construction of those dependencies and subsequent arena/promotion.
    """

    return ResumableGenerationPipeline(
        config,
        replay_buffer=replay_buffer,
        registry=registry,
        dependencies=dependencies,
    ).run()


__all__ = [
    "PipelineDependencies",
    "PipelineError",
    "PipelineInterruptedError",
    "ResumableGenerationPipeline",
    "ResumableGenerationPipelineConfig",
    "StageJournal",
    "StageName",
    "StalePipelineProductError",
    "WorkerOutput",
    "run_generation_pipeline",
]
