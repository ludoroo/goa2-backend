"""Deterministic, resumable CPU trainer for strict joint policy/value data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from tqdm import tqdm

from automata.models.contracts import ArtifactScope
from automata.models.shared_encoder.artifacts import export_model_artifact
from automata.models.shared_encoder.model import JointModelConfig, JointPolicyValueModel
from automata.models.shared_encoder.schema import TensorFeatureSchema
from automata.training.dataset import JointDatasetRow
from automata.training.indexed_dataset import (
    IndexedJointDataset,
    IndexedTrainingChunk,
    open_indexed_dataset,
)
from automata.training.io import TQDM_BAR_FORMAT
from automata.training.losses import (
    JointLossConfig,
    clip_gradients,
    joint_policy_value_loss,
)
from automata.training.metrics import (
    JointMetricsAccumulator,
    PolicyMetricInput,
    ValueMetricInput,
)
from automata.training.splits import (
    JointSplitConfig,
    JointSplitManifest,
    SplitName,
    apply_indexed_joint_split_manifest,
    split_indexed_joint_dataset,
)


@dataclass(frozen=True, slots=True)
class JointTrainingConfig:
    """All paths and numerical choices needed to identify one training run."""

    dataset_path: Path
    split_manifest_path: Path
    checkpoint_path: Path
    run_manifest_path: Path
    artifact_path: Path
    seed: int
    epochs: int = 10
    games_per_batch: int = 8
    learning_rate: float = 1e-3
    max_gradient_norm: float = 1.0
    token_width: int = 32
    state_width: int = 64
    candidate_width: int = 32
    message_passing_layers: int = 2
    dropout: float = 0.0
    validation_fraction: float = 0.2
    dataset_seed_purpose: str = "bootstrap"
    holdout_map_ids: tuple[str, ...] = ()
    holdout_compositions: tuple[tuple[str, ...], ...] = ()
    holdout_game_modes: tuple[str, ...] = ()
    value_weight: float = 1.0
    entropy_weight: float = 0.0
    l2_weight: float = 0.0
    decision_weights_path: Path | None = None
    dataset_index_path: Path | None = None
    decisions_per_chunk: int = 32
    index_workers: int = 4

    def validate(self) -> None:
        if (
            self.epochs <= 0
            or self.games_per_batch <= 0
            or self.decisions_per_chunk <= 0
            or self.index_workers <= 0
        ):
            raise ValueError(
                "epochs, games_per_batch, decisions_per_chunk, and index_workers must be positive"
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.dataset_seed_purpose:
            raise ValueError("dataset_seed_purpose must be non-empty")
        for name in ("learning_rate", "max_gradient_norm", "value_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("entropy_weight", "l2_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.validation_fraction) or not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if any(mode not in {"QUICK", "LONG"} for mode in self.holdout_game_modes):
            raise ValueError("holdout_game_modes must contain only QUICK or LONG")


@dataclass(frozen=True, slots=True)
class JointTrainingResult:
    status: Literal["INTERRUPTED", "SUCCEEDED"]
    step: int
    model_digest: str | None = None


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _config_identity(config: JointTrainingConfig) -> dict[str, Any]:
    excluded = {
        "dataset_path",
        "split_manifest_path",
        "checkpoint_path",
        "run_manifest_path",
        "artifact_path",
        "decision_weights_path",
        "dataset_index_path",
        "index_workers",
    }
    return {key: value for key, value in asdict(config).items() if key not in excluded}


def _identity_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_or_create_splits(
    config: JointTrainingConfig, dataset: IndexedJointDataset
) -> JointSplitManifest:
    split_config = JointSplitConfig(
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        seed_purpose=config.dataset_seed_purpose,
        holdout_map_ids=config.holdout_map_ids,
        holdout_compositions=config.holdout_compositions,
        holdout_game_modes=config.holdout_game_modes,
    )
    path = config.split_manifest_path
    if path.exists():
        payload = path.read_bytes()
        manifest = JointSplitManifest.model_validate_json(payload)
        if payload != manifest.canonical_bytes():
            raise ValueError("split manifest is not canonical JSON")
        if manifest.config != split_config:
            raise ValueError("split manifest configuration mismatch")
        return apply_indexed_joint_split_manifest(dataset, manifest)
    manifest = split_indexed_joint_dataset(dataset, config=split_config)
    _atomic_bytes(path, manifest.canonical_bytes())
    return manifest


def _split_game_ids(manifest: JointSplitManifest, split: SplitName) -> tuple[str, ...]:
    return tuple(item.game_id for item in manifest.memberships if item.split == split)


def _batches(game_ids: Sequence[str], *, seed: int, epoch: int, size: int) -> list[tuple[str, ...]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    order = torch.randperm(len(game_ids), generator=generator).tolist()
    game_list = list(game_ids)
    shuffled = [game_list[index] for index in order]
    return [tuple(shuffled[index : index + size]) for index in range(0, len(shuffled), size)]


def batch_decision_weights(
    rows: Sequence[JointDatasetRow],
    all_rows: Sequence[JointDatasetRow],
    decision_weights: Sequence[float],
) -> tuple[float, ...]:
    """Select persisted global row weights without mini-batch renormalization."""
    if len(all_rows) != len(decision_weights):
        raise ValueError("decision weights must align with the complete dataset")
    by_decision = dict(zip((row.decision_id for row in all_rows), decision_weights, strict=True))
    if len(by_decision) != len(all_rows):
        raise ValueError("dataset decision IDs must be unique")
    try:
        return tuple(float(by_decision[row.decision_id]) for row in rows)
    except KeyError as exc:
        raise ValueError("batch row is absent from persisted decision weights") from exc


@dataclass(frozen=True, slots=True)
class _DecisionWeightStore:
    dataset: IndexedJointDataset
    persisted: tuple[float, ...] | None
    digest: str

    def for_chunk(self, game_id: str, offset: int, count: int) -> tuple[float, ...]:
        game = self.dataset.game(game_id)
        if offset < 0 or count <= 0 or offset + count > game.row_count:
            raise ValueError("decision-weight chunk falls outside its indexed game")
        if self.persisted is None:
            weight = 1.0 / self.dataset.manifest.game_count / game.row_count
            return (weight,) * count
        start = game.global_row_start + offset
        return self.persisted[start : start + count]


def _global_decision_weights(
    config: JointTrainingConfig, dataset: IndexedJointDataset
) -> _DecisionWeightStore:
    if config.decision_weights_path is None:
        return _DecisionWeightStore(dataset, None, "derived-equal-game-weights")
    payload = config.decision_weights_path.read_bytes()
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("decision weights manifest is invalid JSON") from exc
    if payload != _canonical(manifest) or manifest.get("dataset_digest") != dataset.digest:
        raise ValueError("decision weights manifest disagrees with the training dataset")
    raw = manifest.get("decision_weights")
    if not isinstance(raw, list) or len(raw) != dataset.manifest.row_count:
        raise ValueError("decision weights must align with the complete dataset")
    weights: tuple[float, ...] = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("decision weights must be finite and non-negative")
    game_totals = [
        sum(weights[game.global_row_start : game.global_row_start + game.row_count])
        for game in dataset.manifest.games
    ]
    if (
        not game_totals
        or game_totals[0] <= 0
        or not all(
            math.isclose(total, game_totals[0], rel_tol=1e-9, abs_tol=1e-9) for total in game_totals
        )
    ):
        raise ValueError("decision weights must give every source game equal influence")
    return _DecisionWeightStore(dataset, weights, hashlib.sha256(payload).hexdigest())


def _checkpoint_payload(
    *,
    identity: Mapping[str, Any],
    model: JointPolicyValueModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    batch_index: int,
    step: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "identity": dict(identity),
        "identity_digest": _identity_digest(identity),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "batch_index": batch_index,
        "step": step,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    _atomic_bytes(path, stream.getvalue())


def _restore_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    model: JointPolicyValueModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint is malformed or unsafe") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("checkpoint is malformed or unsupported")
    if payload.get("identity_digest") != _identity_digest(identity) or payload.get(
        "identity"
    ) != dict(identity):
        raise ValueError("checkpoint identity mismatch")
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng_state"])
        random.setstate(payload["python_rng_state"])
        epoch = int(payload["epoch"])
        batch_index = int(payload["batch_index"])
        step = int(payload["step"])
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("checkpoint state is incompatible") from exc
    return epoch, batch_index, step


def _add_metric_inputs(
    accumulator: JointMetricsAccumulator,
    output: Any,
    chunk: IndexedTrainingChunk,
) -> None:
    policies: list[PolicyMetricInput] = []
    values: list[ValueMetricInput] = []
    for index, metadata in enumerate(chunk.metric_metadata):
        candidate_count = metadata.candidate_count
        policies.append(
            PolicyMetricInput(
                game_id=chunk.game_ids[index],
                candidate_family=metadata.candidate_family,
                target_probabilities=metadata.target_probabilities,
                predicted_logits=tuple(
                    float(item) for item in output.policy_logits[index, :candidate_count]
                ),
                prior_probabilities=metadata.prior_probabilities,
                q_variances=metadata.q_variances,
                hero=metadata.hero,
                map_id=metadata.map_id,
                composition=metadata.composition,
                round_bucket=metadata.round_bucket,
            )
        )
        values.append(
            ValueMetricInput(
                game_id=chunk.game_ids[index],
                target_value=int(chunk.value_targets[index]),
                predicted_value=float(output.value[index]),
                candidate_family=metadata.candidate_family,
                hero=metadata.hero,
                map_id=metadata.map_id,
                composition=metadata.composition,
                round_bucket=metadata.round_bucket,
            )
        )
    accumulator.update(policies, values)


def _evaluate_split(
    model: JointPolicyValueModel,
    dataset: IndexedJointDataset,
    game_ids: Sequence[str],
    *,
    split: SplitName,
    show_progress: bool,
) -> dict[str, Any]:
    accumulator = JointMetricsAccumulator()
    model.eval()
    games = tqdm(
        game_ids,
        desc=f"Metrics {split.replace('_', ' ')}",
        unit="game",
        bar_format=TQDM_BAR_FORMAT,
        mininterval=2.0,
        disable=not show_progress,
    )
    with torch.no_grad():
        for game_id in games:
            for chunk in dataset.iter_game_training_chunks(game_id):
                _add_metric_inputs(accumulator, model(chunk.batch), chunk)
    return accumulator.compute()


def _source_identities(dataset: IndexedJointDataset) -> list[dict[str, Any]]:
    values = {
        (
            game.generation_id,
            game.source_revision,
            game.dirty_tree_hash,
            game.source_model_digest,
            game.search_config_id,
            game.generator_config_id,
        )
        for game in dataset.manifest.games
    }
    names = (
        "generation_id",
        "source_revision",
        "dirty_tree_hash",
        "source_model_digest",
        "search_config_id",
        "generator_config_id",
    )
    return [dict(zip(names, value, strict=True)) for value in sorted(values, key=repr)]


def _scope(dataset: IndexedJointDataset) -> ArtifactScope:
    heroes = tuple(
        sorted(
            {
                hero
                for game in dataset.manifest.games
                for hero in (*game.red_composition, *game.blue_composition)
            }
        )
    )
    return ArtifactScope(
        supported_heroes=heroes,
        supported_maps=tuple(sorted({game.map_id for game in dataset.manifest.games})),
        supported_game_types=tuple(sorted({game.game_type for game in dataset.manifest.games})),
        hero_adapter_versions={"generic": 1, **{hero: 1 for hero in heroes}},
        map_schema_version=1,
    )


def train_joint(
    config: JointTrainingConfig,
    *,
    stop_after_steps: int | None = None,
    show_progress: bool = True,
) -> JointTrainingResult:
    """Train or resume one exact run, exporting inference state only after success."""
    base_manifest: dict[str, Any] = {"schema_version": 1, "status": "RUNNING"}
    try:
        config.validate()
        if stop_after_steps is not None and stop_after_steps <= 0:
            raise ValueError("stop_after_steps must be positive")
        if config.artifact_path.exists():
            raise FileExistsError(f"artifact destination already exists: {config.artifact_path}")
        index_path = config.dataset_index_path or Path(f"{config.dataset_path}.index")
        dataset = open_indexed_dataset(
            config.dataset_path,
            index_path,
            show_progress=show_progress,
            training_chunk_size=config.decisions_per_chunk,
            index_workers=config.index_workers,
        )
        decision_weights = _global_decision_weights(config, dataset)
        split_manifest = _load_or_create_splits(config, dataset)
        train_game_ids = _split_game_ids(split_manifest, "train")
        validation_game_ids = _split_game_ids(split_manifest, "validation")
        if not train_game_ids or not validation_game_ids:
            raise ValueError("joint training requires non-empty train and validation splits")
        schema = TensorFeatureSchema.current()
        architecture = JointModelConfig(
            model_version=1,
            schema_digest=schema.digest,
            token_width=config.token_width,
            state_width=config.state_width,
            candidate_width=config.candidate_width,
            message_passing_layers=config.message_passing_layers,
            dropout=config.dropout,
        )
        source_identities = _source_identities(dataset)
        identity = {
            "dataset_digest": dataset.digest,
            "split_digest": split_manifest.digest,
            "split_manifest": split_manifest.model_dump(mode="json"),
            "schema_digest": schema.digest,
            "config": _config_identity(config),
            "source_identities": source_identities,
            "decision_weights_digest": decision_weights.digest,
        }
        provenance = {
            **identity,
            "dataset_row_count": dataset.manifest.row_count,
            "dataset_game_count": dataset.manifest.game_count,
        }
        base_manifest.update({"config": _config_identity(config), "provenance": provenance})
        _atomic_bytes(config.run_manifest_path, _canonical(base_manifest))

        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        model = JointPolicyValueModel(schema=schema, config=architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        epoch = batch_index = step = 0
        if config.checkpoint_path.exists():
            epoch, batch_index, step = _restore_checkpoint(
                config.checkpoint_path, identity=identity, model=model, optimizer=optimizer
            )

        loss_config = JointLossConfig(
            value_weight=config.value_weight,
            entropy_weight=config.entropy_weight,
            l2_weight=config.l2_weight,
        )
        chunk_loss_config = JointLossConfig(
            value_weight=config.value_weight,
            entropy_weight=config.entropy_weight,
            l2_weight=0.0,
        )
        invocation_steps = 0
        batch_count = math.ceil(len(train_game_ids) / config.games_per_batch)
        with tqdm(
            total=config.epochs * batch_count,
            initial=step,
            desc="Training",
            unit="batch",
            bar_format=TQDM_BAR_FORMAT,
            mininterval=2.0,
            disable=not show_progress,
        ) as progress:
            while epoch < config.epochs:
                epoch_batches = _batches(
                    train_game_ids,
                    seed=config.seed,
                    epoch=epoch,
                    size=config.games_per_batch,
                )
                while batch_index < len(epoch_batches):
                    game_batch = epoch_batches[batch_index]
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = policy_loss = value_loss = 0.0
                    for game_id in game_batch:
                        game = dataset.game(game_id)
                        for chunk in dataset.iter_game_training_chunks(game_id):
                            game_offset = chunk.global_row_offsets[0] - game.global_row_start
                            chunk_weights = decision_weights.for_chunk(
                                game_id, game_offset, chunk.row_count
                            )
                            if not any(chunk_weights):
                                continue
                            output = model(chunk.batch)
                            loss = joint_policy_value_loss(
                                output,
                                legal_mask=chunk.batch.candidates.mask,
                                policy_targets=chunk.policy_targets,
                                value_targets=chunk.value_targets,
                                game_ids=chunk.game_ids,
                                decision_weights=chunk_weights,
                                config=chunk_loss_config,
                            )
                            loss.total.backward()
                            total_loss += float(loss.total.detach())
                            policy_loss += float(loss.policy.detach())
                            value_loss += float(loss.value.detach())
                    if loss_config.l2_weight:
                        l2 = sum(
                            (parameter.square().sum() for parameter in model.parameters()),
                            start=torch.zeros(()),
                        )
                        regularization = loss_config.l2_weight * l2
                        regularization.backward()
                        total_loss += float(regularization.detach())
                    clip_gradients(model.parameters(), config.max_gradient_norm)
                    optimizer.step()
                    batch_index += 1
                    step += 1
                    invocation_steps += 1
                    next_epoch, next_batch = epoch, batch_index
                    if next_batch == len(epoch_batches):
                        next_epoch, next_batch = epoch + 1, 0
                    _save_checkpoint(
                        config.checkpoint_path,
                        _checkpoint_payload(
                            identity=identity,
                            model=model,
                            optimizer=optimizer,
                            epoch=next_epoch,
                            batch_index=next_batch,
                            step=step,
                        ),
                    )
                    progress.set_postfix(
                        epoch=f"{epoch + 1}/{config.epochs}",
                        loss=total_loss,
                        policy=policy_loss,
                        value=value_loss,
                    )
                    progress.update()
                    if stop_after_steps is not None and invocation_steps >= stop_after_steps:
                        interrupted = {**base_manifest, "status": "INTERRUPTED", "step": step}
                        _atomic_bytes(config.run_manifest_path, _canonical(interrupted))
                        return JointTrainingResult(status="INTERRUPTED", step=step)
                epoch += 1
                batch_index = 0

        metric_splits: tuple[SplitName, ...] = (
            "train",
            "validation",
            "map_holdout",
            "composition_holdout",
            "game_mode_holdout",
        )
        metrics = {
            name: _evaluate_split(
                model,
                dataset,
                _split_game_ids(split_manifest, name),
                split=name,
                show_progress=show_progress,
            )
            for name in metric_splits
        }
        final_provenance = {**provenance, "metrics": metrics, "step": step}
        artifact_manifest = export_model_artifact(
            config.artifact_path,
            model=model,
            schema=schema,
            scope=_scope(dataset),
            runtime_compatibility_version=1,
            provenance=final_provenance,
        )
        succeeded = {
            **base_manifest,
            "status": "SUCCEEDED",
            "step": step,
            "model_digest": artifact_manifest.model_digest,
            "metrics": metrics,
        }
        _atomic_bytes(config.run_manifest_path, _canonical(succeeded))
        return JointTrainingResult("SUCCEEDED", step, artifact_manifest.model_digest)
    except BaseException as exc:
        failed = {**base_manifest, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        _atomic_bytes(config.run_manifest_path, _canonical(failed))
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--games-per-batch", type=int, default=8)
    parser.add_argument("--decisions-per-chunk", type=int, default=32)
    parser.add_argument("--dataset-index", type=Path)
    parser.add_argument("--index-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--l2-weight", type=float, default=0.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--holdout-game-mode", action="append")
    parser.add_argument("--dataset-seed-purpose", default="bootstrap")
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false", help="disable progress output"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_joint(
        JointTrainingConfig(
            dataset_path=args.dataset,
            split_manifest_path=args.split_manifest,
            checkpoint_path=args.checkpoint,
            run_manifest_path=args.run_manifest,
            artifact_path=args.artifact,
            seed=args.seed,
            epochs=args.epochs,
            games_per_batch=args.games_per_batch,
            decisions_per_chunk=args.decisions_per_chunk,
            dataset_index_path=args.dataset_index,
            index_workers=args.index_workers,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            entropy_weight=args.entropy_weight,
            l2_weight=args.l2_weight,
            value_weight=args.value_weight,
            validation_fraction=args.validation_fraction,
            holdout_game_modes=tuple(args.holdout_game_mode or ()),
            dataset_seed_purpose=args.dataset_seed_purpose,
        ),
        show_progress=args.progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JointTrainingConfig",
    "JointTrainingResult",
    "batch_decision_weights",
    "main",
    "train_joint",
]
