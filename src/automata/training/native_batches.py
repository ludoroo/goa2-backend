"""Typed, per-head collation for native Gen1 training records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor

from automata.decision import DecisionSemanticRole
from automata.models.contracts import canonical_json_bytes
from automata.models.shared_encoder.batching import (
    DecisionBatch,
    StableValueBatch,
    collate_decisions,
    collate_stable_values,
)
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.training.metrics import PolicyMetricInput, ValueMetricInput
from automata.training.native_dataset import (
    NATIVE_RECORD_ADAPTER,
    PolicyDatasetRecord,
    ValueDatasetRecord,
)

BoundaryKind = Literal["ACTOR_READY", "PLANNING_READY"]


@dataclass(frozen=True, slots=True)
class NativePolicyMetricMetadata:
    """Policy-only attributes retained for the existing metric primitives."""

    game_id: str
    candidate_count: int
    candidate_family: str
    target_probabilities: tuple[float, ...]
    prior_probabilities: tuple[float, ...] | None
    visit_counts: tuple[int, ...]
    q_variances: tuple[float, ...]
    hero: str
    map_id: str
    composition: str
    round_bucket: str
    input_request_type: str | None
    semantic_role: DecisionSemanticRole
    can_skip: bool

    def to_metric_input(self, predicted_logits: Tensor | Sequence[float]) -> PolicyMetricInput:
        """Build one typed metric row, ignoring only model padding past legal candidates."""
        if isinstance(predicted_logits, Tensor):
            if predicted_logits.ndim != 1:
                raise ValueError("predicted policy logits must be a one-dimensional row")
            logits = tuple(float(value) for value in predicted_logits.detach().cpu().tolist())
        else:
            logits = tuple(float(value) for value in predicted_logits)
        if len(logits) < self.candidate_count:
            raise ValueError("predicted policy logits do not cover every legal candidate")
        return PolicyMetricInput(
            game_id=self.game_id,
            candidate_family=self.candidate_family,
            target_probabilities=self.target_probabilities,
            predicted_logits=logits[: self.candidate_count],
            prior_probabilities=self.prior_probabilities,
            q_variances=self.q_variances,
            hero=self.hero,
            map_id=self.map_id,
            composition=self.composition,
            round_bucket=self.round_bucket,
            input_request_type=self.input_request_type,
            semantic_role=self.semantic_role,
            can_skip=self.can_skip,
        )


@dataclass(frozen=True, slots=True)
class NativeValueMetricMetadata:
    """Candidate-free value attributes retained for the existing value metrics."""

    game_id: str
    target_value: int
    hero: str
    map_id: str
    composition: str
    round_bucket: str
    boundary_kind: BoundaryKind

    def to_metric_input(self, predicted_value: Tensor | float) -> ValueMetricInput:
        """Build one typed candidate-free metric row."""
        if isinstance(predicted_value, Tensor):
            if predicted_value.numel() != 1:
                raise ValueError("predicted value must contain exactly one scalar")
            prediction = float(predicted_value.detach().cpu().item())
        else:
            prediction = float(predicted_value)
        return ValueMetricInput(
            game_id=self.game_id,
            target_value=self.target_value,
            predicted_value=prediction,
            hero=self.hero,
            map_id=self.map_id,
            composition=self.composition,
            round_bucket=self.round_bucket,
        )


@dataclass(frozen=True, slots=True)
class NativePolicyTrainingBatch:
    """One homogeneous native policy batch and its explicit supervision."""

    batch: DecisionBatch
    policy_targets: Tensor
    row_weights: Tensor
    row_mask: Tensor
    sample_ids: tuple[str, ...]
    sample_indexes: tuple[int, ...]
    game_ids: tuple[str, ...]
    metric_metadata: tuple[NativePolicyMetricMetadata, ...]


@dataclass(frozen=True, slots=True)
class NativeValueTrainingBatch:
    """One homogeneous candidate-free native value batch and its supervision."""

    batch: StableValueBatch
    value_targets: Tensor
    row_weights: Tensor
    row_mask: Tensor
    sample_ids: tuple[str, ...]
    sample_indexes: tuple[int, ...]
    game_ids: tuple[str, ...]
    metric_metadata: tuple[NativeValueMetricMetadata, ...]


def _revalidate_policy_records(
    records: Sequence[PolicyDatasetRecord],
) -> tuple[PolicyDatasetRecord, ...]:
    if not records:
        raise ValueError("native policy batch cannot be empty")
    validated: list[PolicyDatasetRecord] = []
    for record in records:
        if not isinstance(record, (PolicyDatasetRecord, ValueDatasetRecord)):
            raise TypeError("native policy batch items must be native dataset records")
        restored = NATIVE_RECORD_ADAPTER.validate_json(canonical_json_bytes(record), strict=True)
        if not isinstance(restored, PolicyDatasetRecord):
            raise ValueError("native policy batch can contain only POLICY records")
        validated.append(restored)
    _require_unique_sample_ids(validated)
    return tuple(validated)


def _revalidate_value_records(
    records: Sequence[ValueDatasetRecord],
) -> tuple[ValueDatasetRecord, ...]:
    if not records:
        raise ValueError("native value batch cannot be empty")
    validated: list[ValueDatasetRecord] = []
    for record in records:
        if not isinstance(record, (PolicyDatasetRecord, ValueDatasetRecord)):
            raise TypeError("native value batch items must be native dataset records")
        restored = NATIVE_RECORD_ADAPTER.validate_json(canonical_json_bytes(record), strict=True)
        if not isinstance(restored, ValueDatasetRecord):
            raise ValueError("native value batch can contain only VALUE records")
        validated.append(restored)
    _require_unique_sample_ids(validated)
    return tuple(validated)


def _require_unique_sample_ids(
    records: Sequence[PolicyDatasetRecord | ValueDatasetRecord],
) -> None:
    sample_ids = tuple(record.sample_id for record in records)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("native batch sample IDs must be unique")


def _row_weights(
    records: Sequence[PolicyDatasetRecord | ValueDatasetRecord],
    counts: Mapping[str, int],
    *,
    head: str,
) -> Tensor:
    if not isinstance(counts, Mapping):
        raise TypeError(f"game {head} counts must be a mapping")
    for game_id, count in counts.items():
        if not isinstance(game_id, str) or not game_id:
            raise TypeError(f"game {head} count mapping keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"game {head} counts must be strict positive integers")

    observed = Counter(record.game.game_id for record in records)
    for game_id, observed_count in observed.items():
        if game_id not in counts:
            raise ValueError(f"game {head} count mapping is missing a batch game")
        if counts[game_id] < observed_count:
            raise ValueError(
                f"full-game {head} count cannot be smaller than observed batch records"
            )
    return torch.tensor(
        [1.0 / counts[record.game.game_id] for record in records],
        dtype=torch.float32,
    )


def _global_features(record: PolicyDatasetRecord | ValueDatasetRecord) -> dict[str, object]:
    global_tokens = tuple(
        token for token in record.observation.state.tokens if token.kind == "GLOBAL"
    )
    # Native-record revalidation guarantees this, but keeping the extraction
    # total makes this module robust if that invariant is ever weakened.
    if len(global_tokens) != 1:
        raise ValueError("native metric metadata requires exactly one GLOBAL token")
    return cast(dict[str, object], global_tokens[0].features)


def _composition(record: PolicyDatasetRecord | ValueDatasetRecord) -> str:
    return (
        f"{'/'.join(record.game.red_composition)} vs " f"{'/'.join(record.game.blue_composition)}"
    )


def _policy_metadata(record: PolicyDatasetRecord) -> NativePolicyMetricMetadata:
    actions = record.target.actions
    targets = tuple(cast(float, action.improved_probability) for action in actions)
    raw_priors = tuple(action.prior_probability for action in actions)
    priors = None if raw_priors[0] is None else tuple(cast(float, prior) for prior in raw_priors)
    viewer_tokens = tuple(
        token
        for token in record.observation.state.tokens
        if token.kind == "HERO" and token.features.get("relation") == "SELF"
    )
    if len(viewer_tokens) != 1:
        raise ValueError("native policy metric metadata requires exactly one viewer hero")
    global_features = _global_features(record)
    return NativePolicyMetricMetadata(
        game_id=record.game.game_id,
        candidate_count=len(actions),
        candidate_family="+".join(
            sorted({candidate.candidate_id.kind for candidate in record.observation.candidates})
        ),
        target_probabilities=targets,
        prior_probabilities=priors,
        visit_counts=tuple(action.sample_count for action in actions),
        q_variances=tuple(float(action.value_variance) for action in actions),
        hero=str(viewer_tokens[0].features["name"]),
        map_id=record.game.map_id,
        composition=_composition(record),
        round_bucket=str(global_features["round"]),
        input_request_type=record.observation.input_request_type,
        semantic_role=record.observation.semantic_role,
        can_skip=record.observation.can_skip,
    )


def _value_metadata(record: ValueDatasetRecord) -> NativeValueMetricMetadata:
    viewer = next(
        (
            token
            for token in record.observation.state.tokens
            if token.kind == "HERO" and token.local_ref == record.boundary.viewer_ref
        ),
        None,
    )
    if viewer is None:
        raise ValueError("native value metric metadata requires the boundary viewer hero")
    return NativeValueMetricMetadata(
        game_id=record.game.game_id,
        target_value=record.value_target,
        hero=str(viewer.features["name"]),
        map_id=record.game.map_id,
        composition=_composition(record),
        round_bucket=str(record.boundary.round),
        boundary_kind=record.boundary.kind,
    )


def collate_native_policy_records(
    records: Sequence[PolicyDatasetRecord],
    *,
    game_policy_counts: Mapping[str, int],
    schema: TensorFeatureSchema | None = None,
) -> NativePolicyTrainingBatch:
    """Revalidate and collate actual native POLICY rows without a value stand-in."""
    rows = _revalidate_policy_records(records)
    if schema is None:
        schema = TensorFeatureSchema.current()
    elif not isinstance(schema, TensorFeatureSchema):
        raise TypeError("native policy schema must be a TensorFeatureSchema")

    weights = _row_weights(rows, game_policy_counts, head="policy")
    batch = collate_decisions(
        [record.observation for record in rows],
        schema=schema,
        training=True,
    )
    targets = torch.zeros(batch.candidates.mask.shape, dtype=torch.float32)
    for row_index, record in enumerate(rows):
        probabilities = tuple(
            cast(float, action.improved_probability) for action in record.target.actions
        )
        targets[row_index, : len(probabilities)] = torch.tensor(probabilities, dtype=torch.float32)
    return NativePolicyTrainingBatch(
        batch=batch,
        policy_targets=targets,
        row_weights=weights,
        row_mask=torch.ones(len(rows), dtype=torch.bool),
        sample_ids=tuple(record.sample_id for record in rows),
        sample_indexes=tuple(record.sample_index for record in rows),
        game_ids=tuple(record.game.game_id for record in rows),
        metric_metadata=tuple(_policy_metadata(record) for record in rows),
    )


def collate_native_value_records(
    records: Sequence[ValueDatasetRecord],
    *,
    game_value_counts: Mapping[str, int],
    schema: StableValueTensorSchema | None = None,
) -> NativeValueTrainingBatch:
    """Revalidate and collate actual native VALUE rows without policy candidates."""
    rows = _revalidate_value_records(records)
    if schema is None:
        schema = StableValueTensorSchema.current()
    elif not isinstance(schema, StableValueTensorSchema):
        raise TypeError("native value schema must be a StableValueTensorSchema")

    weights = _row_weights(rows, game_value_counts, head="value")
    batch = collate_stable_values(
        [record.observation for record in rows],
        schema=schema,
    )
    return NativeValueTrainingBatch(
        batch=batch,
        value_targets=torch.tensor([record.value_target for record in rows], dtype=torch.float32),
        row_weights=weights,
        row_mask=torch.ones(len(rows), dtype=torch.bool),
        sample_ids=tuple(record.sample_id for record in rows),
        sample_indexes=tuple(record.sample_index for record in rows),
        game_ids=tuple(record.game.game_id for record in rows),
        metric_metadata=tuple(_value_metadata(record) for record in rows),
    )


__all__ = [
    "NativePolicyMetricMetadata",
    "NativePolicyTrainingBatch",
    "NativeValueMetricMetadata",
    "NativeValueTrainingBatch",
    "collate_native_policy_records",
    "collate_native_value_records",
]
