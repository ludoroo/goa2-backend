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
from automata.models.shared_encoder.batching import collate_decisions
from automata.models.shared_encoder.model import JointModelConfig, JointPolicyValueModel
from automata.models.shared_encoder.schema import TensorFeatureSchema
from automata.training.dataset import JointDataset, JointDatasetRow, load_joint_dataset
from automata.training.losses import (
    JointLossConfig,
    clip_gradients,
    equal_game_weights,
    joint_policy_value_loss,
)
from automata.training.metrics import PolicyMetricInput, ValueMetricInput, joint_metrics
from automata.training.splits import (
    JointSplitConfig,
    JointSplitManifest,
    JointSplits,
    SplitName,
    apply_joint_split_manifest,
    split_joint_dataset,
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

    def validate(self) -> None:
        if self.epochs <= 0 or self.games_per_batch <= 0:
            raise ValueError("epochs and games_per_batch must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.dataset_seed_purpose:
            raise ValueError("dataset_seed_purpose must be non-empty")
        for name in ("learning_rate", "max_gradient_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


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
    }
    return {key: value for key, value in asdict(config).items() if key not in excluded}


def _identity_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_or_create_splits(config: JointTrainingConfig, dataset: JointDataset) -> JointSplits:
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
        return apply_joint_split_manifest(dataset, manifest)
    splits = split_joint_dataset(dataset, config=split_config)
    _atomic_bytes(path, splits.manifest.canonical_bytes())
    return splits


def _batches(game_ids: Sequence[str], *, seed: int, epoch: int, size: int) -> list[tuple[str, ...]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    order = torch.randperm(len(game_ids), generator=generator).tolist()
    game_list = list(game_ids)
    shuffled = [game_list[index] for index in order]
    return [tuple(shuffled[index : index + size]) for index in range(0, len(shuffled), size)]


def _rows_for_games(
    rows_by_game: Mapping[str, tuple[JointDatasetRow, ...]], ids: Sequence[str]
) -> tuple[JointDatasetRow, ...]:
    return tuple(row for game_id in ids for row in rows_by_game[game_id])


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


def _global_decision_weights(
    config: JointTrainingConfig, dataset: JointDataset
) -> tuple[tuple[float, ...], str]:
    if config.decision_weights_path is None:
        derived_weights = equal_game_weights(
            [row.game_id for row in dataset.rows], like=torch.empty(len(dataset.rows))
        ).tolist()
        return tuple(float(value) for value in derived_weights), "derived-equal-game-weights"
    payload = config.decision_weights_path.read_bytes()
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("decision weights manifest is invalid JSON") from exc
    if payload != _canonical(manifest) or manifest.get("dataset_digest") != dataset.digest:
        raise ValueError("decision weights manifest disagrees with the training dataset")
    raw = manifest.get("decision_weights")
    if not isinstance(raw, list) or len(raw) != len(dataset.rows):
        raise ValueError("decision weights must align with the complete dataset")
    weights: tuple[float, ...] = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("decision weights must be finite and non-negative")
    game_totals: dict[str, float] = {}
    for row, weight in zip(dataset.rows, weights, strict=True):
        game_totals[row.game_id] = game_totals.get(row.game_id, 0.0) + weight
    if not game_totals or not all(
        math.isclose(total, next(iter(game_totals.values())), rel_tol=1e-9, abs_tol=1e-9)
        for total in game_totals.values()
    ):
        raise ValueError("decision weights must give every source game equal influence")
    return weights, hashlib.sha256(payload).hexdigest()


def _targets(rows: Sequence[JointDatasetRow], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    policy = torch.zeros((len(rows), width), dtype=torch.float32)
    for index, row in enumerate(rows):
        policy[index, : len(row.policy_target)] = torch.tensor(row.policy_target)
    value = torch.tensor([row.value_target for row in rows], dtype=torch.float32)
    return policy, value


def _source_identities(rows: Sequence[JointDatasetRow]) -> list[dict[str, Any]]:
    values = {
        (
            row.generation_id,
            row.source_revision,
            row.dirty_tree_hash,
            row.source_model_digest,
            row.search_config_id,
            row.generator_config_id,
        )
        for row in rows
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


def _metric_inputs(
    model: JointPolicyValueModel,
    schema: TensorFeatureSchema,
    rows: Sequence[JointDatasetRow],
) -> dict[str, Any]:
    if not rows:
        return joint_metrics((), ())
    model.eval()
    batch = collate_decisions([row.observation for row in rows], schema=schema, training=True)
    with torch.no_grad():
        output = model(batch)
    policies: list[PolicyMetricInput] = []
    values: list[ValueMetricInput] = []
    for index, row in enumerate(rows):
        candidate_count = len(row.policy_target)
        global_features = next(
            token.features for token in row.observation.state.tokens if token.kind == "GLOBAL"
        )
        perspective_heroes = [
            token
            for token in row.observation.state.tokens
            if token.kind == "HERO" and token.features.get("team_id") == row.perspective_team
        ]
        hero_token = next(
            (
                token
                for token in perspective_heroes
                if token.features.get("is_decision_owner") or token.features.get("is_current_actor")
            ),
            perspective_heroes[0] if perspective_heroes else None,
        )
        hero = str(hero_token.features["name"]) if hero_token is not None else None
        candidate_kinds = sorted(
            {candidate.candidate_id.kind for candidate in row.observation.candidates}
        )
        family = "+".join(candidate_kinds)
        priors: tuple[float, ...] | None = None
        variances: tuple[float, ...] | None = None
        if row.action_stats is not None:
            counts = tuple(float(item.sample_count) for item in row.action_stats)
            total = sum(counts)
            priors = tuple(value / total for value in counts) if total else None
            variances = tuple(float(item.value_variance) for item in row.action_stats)
        composition = f"{'/'.join(row.red_composition)} vs {'/'.join(row.blue_composition)}"
        round_bucket = str(global_features["round"]) if "round" in global_features else None
        policies.append(
            PolicyMetricInput(
                game_id=row.game_id,
                candidate_family=family,
                target_probabilities=row.policy_target,
                predicted_logits=tuple(
                    float(item) for item in output.policy_logits[index, :candidate_count]
                ),
                prior_probabilities=priors,
                q_variances=variances,
                hero=hero,
                map_id=row.map_id,
                composition=composition,
                round_bucket=round_bucket,
            )
        )
        values.append(
            ValueMetricInput(
                game_id=row.game_id,
                target_value=row.value_target,
                predicted_value=float(output.value[index]),
                candidate_family=family,
                hero=hero,
                map_id=row.map_id,
                composition=composition,
                round_bucket=round_bucket,
            )
        )
    return joint_metrics(policies, values)


def _scope(rows: Sequence[JointDatasetRow]) -> ArtifactScope:
    heroes = tuple(
        sorted({hero for row in rows for hero in (*row.red_composition, *row.blue_composition)})
    )
    return ArtifactScope(
        supported_heroes=heroes,
        supported_maps=tuple(sorted({row.map_id for row in rows})),
        supported_game_types=tuple(sorted({row.game_type for row in rows})),
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
        dataset = load_joint_dataset(config.dataset_path)
        decision_weights, decision_weights_digest = _global_decision_weights(config, dataset)
        splits = _load_or_create_splits(config, dataset)
        train_game_ids = splits.game_ids("train")
        if not train_game_ids or not splits.rows("validation"):
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
        source_identities = _source_identities(dataset.rows)
        identity = {
            "dataset_digest": dataset.digest,
            "split_digest": splits.digest,
            "split_manifest": splits.manifest.model_dump(mode="json"),
            "schema_digest": schema.digest,
            "config": _config_identity(config),
            "source_identities": source_identities,
            "decision_weights_digest": decision_weights_digest,
        }
        provenance = {
            **identity,
            "dataset_row_count": len(dataset.rows),
            "dataset_game_count": len(dataset.game_ids),
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
        invocation_steps = 0
        batch_count = math.ceil(len(train_game_ids) / config.games_per_batch)
        with tqdm(
            total=config.epochs * batch_count,
            initial=step,
            desc="Training",
            unit="batch",
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
                    rows = _rows_for_games(dataset.rows_by_game, epoch_batches[batch_index])
                    batch = collate_decisions(
                        [row.observation for row in rows], schema=schema, training=True
                    )
                    policy_targets, value_targets = _targets(rows, batch.candidates.mask.shape[1])
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch)
                    loss = joint_policy_value_loss(
                        output,
                        legal_mask=batch.candidates.mask,
                        policy_targets=policy_targets,
                        value_targets=value_targets,
                        game_ids=[row.game_id for row in rows],
                        decision_weights=batch_decision_weights(
                            rows, dataset.rows, decision_weights
                        ),
                        parameters=model.parameters(),
                        config=loss_config,
                    )
                    loss.total.backward()
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
                        loss=float(loss.total.detach()),
                        policy=float(loss.policy.detach()),
                        value=float(loss.value.detach()),
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
        metrics = {name: _metric_inputs(model, schema, splits.rows(name)) for name in metric_splits}
        final_provenance = {**provenance, "metrics": metrics, "step": step}
        artifact_manifest = export_model_artifact(
            config.artifact_path,
            model=model,
            schema=schema,
            scope=_scope(dataset.rows),
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
            learning_rate=args.learning_rate,
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
