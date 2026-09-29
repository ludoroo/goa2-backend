"""Explicit PyTorch boundary for ragged learned-model decision batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor

from ..contracts import CandidateID, DecisionObservation, StableValueObservation, Viewer
from .schema import (
    RecordFeatureSchema,
    StableValueTensorSchema,
    TensorFeatureSchema,
    VectorizedDecision,
    VectorizedStableValue,
    expanded_numeric_width,
)


@dataclass(frozen=True)
class FeatureTable:
    """A padded table for one record kind."""

    numeric: Tensor
    numeric_valid: Tensor
    categorical: Tensor
    references: Tensor
    reference_valid: Tensor
    reference_kind_indices: Tensor
    mask: Tensor


@dataclass(frozen=True)
class RelationshipTable(FeatureTable):
    """A feature table whose endpoints use kind-local token indexes."""

    source_indices: Tensor
    target_indices: Tensor
    source_kind_indices: Tensor
    target_kind_indices: Tensor


@dataclass(frozen=True)
class CandidateTable(FeatureTable):
    """Padded legal candidates in engine order."""

    kind_indices: Tensor
    target_indices: Tensor
    target_kind_indices: Tensor
    target_valid: Tensor


TokenEmbeddings: TypeAlias = dict[str, Tensor]


@dataclass(frozen=True)
class GraphBatch:
    """Padded graph tensors shared by policy and stable-value batches."""

    tokens: dict[str, FeatureTable]
    relationships: dict[str, RelationshipTable]
    token_kinds: tuple[str, ...]


@dataclass(frozen=True)
class DecisionBatch:
    """CPU tensors plus the typed identities needed to decode model output."""

    tokens: dict[str, FeatureTable]
    relationships: dict[str, RelationshipTable]
    decision_context: FeatureTable | None
    candidates: CandidateTable
    candidate_ids: tuple[tuple[CandidateID, ...], ...]
    token_kinds: tuple[str, ...]
    candidate_kinds: tuple[str, ...]

    @property
    def graph(self) -> GraphBatch:
        """Return a nonserialized view without changing legacy batch fields."""
        return GraphBatch(self.tokens, self.relationships, self.token_kinds)

    def gather_candidate_targets(self, embeddings: TokenEmbeddings) -> tuple[Tensor, Tensor]:
        """Gather graph-bound candidate targets without reading padded rows."""
        if not embeddings:
            raise ValueError("token embeddings cannot be empty")
        first = next(iter(embeddings.values()))
        if first.ndim != 3 or first.shape[0] != self.candidates.mask.shape[0]:
            raise ValueError("token embeddings must have shape [batch, rows, width]")
        width = first.shape[-1]
        output = first.new_zeros((*self.candidates.mask.shape, width))
        valid = torch.zeros_like(self.candidates.mask)

        for kind_index, kind in enumerate(self.token_kinds):
            source = embeddings.get(kind)
            if source is None:
                raise ValueError(f"missing token embeddings for {kind!r}")
            if source.shape[:2] != self.tokens[kind].mask.shape or source.shape[-1] != width:
                raise ValueError(f"token embeddings for {kind!r} have incompatible shape")
            kind_valid = (
                self.candidates.target_valid
                & self.candidates.mask
                & (self.candidates.target_kind_indices == kind_index)
            )
            gathered, gathered_valid = safe_gather(
                source,
                self.candidates.target_indices,
                kind_valid,
                source_mask=self.tokens[kind].mask,
            )
            output = torch.where(gathered_valid.unsqueeze(-1), gathered, output)
            valid |= gathered_valid
        return output, valid


@dataclass(frozen=True)
class StableValueBatch:
    """Candidate-free stable graphs plus value and viewer context."""

    graph: GraphBatch
    value_context: FeatureTable
    viewers: tuple[Viewer, ...]


# Short name retained for model-facing annotations.
Batch = DecisionBatch


def _expanded_mask(mask: Tensor, values: Tensor) -> Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a bool tensor")
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return mask


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Mean over valid entries; an empty reduction is defined as zero."""
    expanded = _expanded_mask(mask, values)
    clean = torch.where(expanded, values, torch.zeros((), dtype=values.dtype, device=values.device))
    count = expanded.sum(dim=dim)
    total = clean.sum(dim=dim)
    return torch.where(count > 0, total / count.clamp_min(1), torch.zeros_like(total))


def masked_max(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Maximum over valid entries; an empty reduction is defined as zero."""
    expanded = _expanded_mask(mask, values)
    if values.shape[dim] == 0:
        return values.sum(dim=dim)
    clean = torch.where(
        expanded,
        values,
        torch.full((), -torch.inf, dtype=values.dtype, device=values.device),
    )
    maximum = clean.amax(dim=dim)
    any_valid = expanded.any(dim=dim)
    return torch.where(any_valid, maximum, torch.zeros_like(maximum))


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Stable softmax over legal entries, with exactly-zero padded output."""
    if logits.shape != mask.shape:
        raise ValueError("logits and mask must have identical shapes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a bool tensor")
    if not mask.any(dim=dim).all():
        raise ValueError("masked softmax requires at least one legal candidate per row")
    finite_logits = torch.nan_to_num(logits)
    masked = torch.where(mask, finite_logits, torch.full_like(finite_logits, -torch.inf))
    probabilities = torch.softmax(masked, dim=dim)
    return torch.where(mask, probabilities, torch.zeros_like(probabilities))


def safe_gather(
    source: Tensor,
    indices: Tensor,
    valid: Tensor,
    *,
    source_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Gather ``[batch, row, width]`` tensors while making bad refs inert."""
    if source.ndim != 3 or indices.ndim != 2 or indices.shape != valid.shape:
        raise ValueError("safe_gather expects source [B,N,D] and aligned indexes [B,M]")
    if indices.dtype != torch.int64 or valid.dtype != torch.bool:
        raise TypeError("indices must be int64 and valid must be bool")
    if source.shape[0] != indices.shape[0]:
        raise ValueError("safe_gather batch dimensions must match")

    rows = source.shape[1]
    effective = valid & (indices >= 0) & (indices < rows)
    if source_mask is not None:
        if source_mask.shape != source.shape[:2] or source_mask.dtype != torch.bool:
            raise ValueError("source_mask must be bool with shape [B,N]")
        if rows:
            sanitized_for_mask = torch.where(effective, indices, torch.zeros_like(indices))
            referenced_rows = source_mask.gather(1, sanitized_for_mask.clamp(0, rows - 1))
            effective &= referenced_rows
    if rows == 0:
        return source.new_zeros((*indices.shape, source.shape[-1])), effective

    # Replace invalid sentinels before clamping so -1 can never address the last row.
    sanitized = torch.where(effective, indices, torch.zeros_like(indices)).clamp(0, rows - 1)
    gathered = source.gather(1, sanitized.unsqueeze(-1).expand(-1, -1, source.shape[-1]))
    gathered = torch.where(effective.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    return gathered, effective


def _empty_table(
    batch_size: int,
    rows: int,
    numeric_width: int,
    categorical_width: int,
    reference_width: int,
) -> FeatureTable:
    shape = (batch_size, rows)
    return FeatureTable(
        numeric=torch.zeros((*shape, numeric_width), dtype=torch.float32),
        numeric_valid=torch.zeros((*shape, numeric_width), dtype=torch.bool),
        categorical=torch.zeros((*shape, categorical_width), dtype=torch.int64),
        references=torch.full((*shape, reference_width), -1, dtype=torch.int64),
        reference_valid=torch.zeros((*shape, reference_width), dtype=torch.bool),
        reference_kind_indices=torch.full((*shape, reference_width), -1, dtype=torch.int64),
        mask=torch.zeros(shape, dtype=torch.bool),
    )


def _empty_feature_table(batch_size: int, rows: int, schema: RecordFeatureSchema) -> FeatureTable:
    return _empty_table(
        batch_size,
        rows,
        expanded_numeric_width(schema),
        len(schema.categorical),
        len(schema.references),
    )


def _collate_graphs(
    vectorized: Sequence[VectorizedDecision | VectorizedStableValue],
    *,
    token_schemas: tuple[RecordFeatureSchema, ...],
    relationship_schemas: tuple[RecordFeatureSchema, ...],
) -> tuple[GraphBatch, list[dict[str, tuple[str, int]]]]:
    batch_size = len(vectorized)
    token_kinds = tuple(item.kind for item in token_schemas)
    token_kind_index = {kind: index for index, kind in enumerate(token_kinds)}
    token_tables: dict[str, FeatureTable] = {}
    local_maps: list[dict[str, tuple[str, int]]] = []
    for graph in vectorized:
        counts: dict[str, int] = {}
        local: dict[str, tuple[str, int]] = {}
        for token in graph.tokens:
            offset = counts.get(token.kind, 0)
            counts[token.kind] = offset + 1
            local[token.local_ref] = (token.kind, offset)
        local_maps.append(local)

    for record_schema in token_schemas:
        token_records_by_batch = [
            [record for record in graph.tokens if record.kind == record_schema.kind]
            for graph in vectorized
        ]
        rows = max(map(len, token_records_by_batch))
        table = _empty_feature_table(batch_size, rows, record_schema)
        for batch_index, token_records in enumerate(token_records_by_batch):
            for row, token_record in enumerate(token_records):
                table.mask[batch_index, row] = True
                table.numeric[batch_index, row] = torch.tensor(
                    token_record.numeric, dtype=torch.float32
                )
                table.numeric_valid[batch_index, row] = torch.tensor(
                    token_record.numeric_valid, dtype=torch.bool
                )
                table.categorical[batch_index, row] = torch.tensor(
                    token_record.categorical, dtype=torch.int64
                )
                for column, (reference, is_valid) in enumerate(
                    zip(token_record.references, token_record.reference_valid, strict=True)
                ):
                    if is_valid:
                        target = vectorized[batch_index].tokens[reference]
                        target_kind, target_offset = local_maps[batch_index][target.local_ref]
                        table.references[batch_index, row, column] = target_offset
                        table.reference_kind_indices[batch_index, row, column] = token_kind_index[
                            target_kind
                        ]
                        table.reference_valid[batch_index, row, column] = True
        token_tables[record_schema.kind] = table

    relationship_tables: dict[str, RelationshipTable] = {}
    for record_schema in relationship_schemas:
        relationship_records_by_batch = [
            [record for record in graph.relationships if record.kind == record_schema.kind]
            for graph in vectorized
        ]
        rows = max(map(len, relationship_records_by_batch))
        base = _empty_feature_table(batch_size, rows, record_schema)
        shape = (batch_size, rows)
        table = RelationshipTable(
            **base.__dict__,
            source_indices=torch.full(shape, -1, dtype=torch.int64),
            target_indices=torch.full(shape, -1, dtype=torch.int64),
            source_kind_indices=torch.full(shape, -1, dtype=torch.int64),
            target_kind_indices=torch.full(shape, -1, dtype=torch.int64),
        )
        for batch_index, relationship_records in enumerate(relationship_records_by_batch):
            for row, relationship_record in enumerate(relationship_records):
                source_kind, source_offset = local_maps[batch_index][relationship_record.source_ref]
                target_kind, target_offset = local_maps[batch_index][relationship_record.target_ref]
                table.mask[batch_index, row] = True
                table.numeric[batch_index, row] = torch.tensor(
                    relationship_record.numeric, dtype=torch.float32
                )
                table.numeric_valid[batch_index, row] = torch.tensor(
                    relationship_record.numeric_valid, dtype=torch.bool
                )
                table.categorical[batch_index, row] = torch.tensor(
                    relationship_record.categorical, dtype=torch.int64
                )
                table.source_indices[batch_index, row] = source_offset
                table.target_indices[batch_index, row] = target_offset
                table.source_kind_indices[batch_index, row] = token_kind_index[source_kind]
                table.target_kind_indices[batch_index, row] = token_kind_index[target_kind]
        relationship_tables[record_schema.kind] = table

    return GraphBatch(token_tables, relationship_tables, token_kinds), local_maps


def collate_decisions(
    decisions: Sequence[DecisionObservation | VectorizedDecision],
    *,
    schema: TensorFeatureSchema,
    training: bool = False,
) -> DecisionBatch:
    """Collate variable decision graphs using an explicitly pinned schema."""
    if not decisions:
        raise ValueError("decision batch cannot be empty")
    vectorized = [
        schema.vectorize(item, training=training) if isinstance(item, DecisionObservation) else item
        for item in decisions
    ]
    if any(not item.candidates for item in vectorized):
        raise ValueError("each decision must contain at least one candidate")
    if any(len(item.candidates) != len(item.candidate_ids) for item in vectorized):
        raise ValueError("candidate records and IDs must be aligned")

    batch_size = len(vectorized)
    decision_context: FeatureTable | None = None
    if schema.decision_context is not None:
        decision_context = _empty_feature_table(batch_size, 1, schema.decision_context)
        for batch_index, decision in enumerate(vectorized):
            context = decision.decision_context
            if context is None:
                raise ValueError("tensor schema requires one decision-context row")
            decision_context.mask[batch_index, 0] = True
            decision_context.numeric[batch_index, 0] = torch.tensor(
                context.numeric, dtype=torch.float32
            )
            decision_context.numeric_valid[batch_index, 0] = torch.tensor(
                context.numeric_valid, dtype=torch.bool
            )
            decision_context.categorical[batch_index, 0] = torch.tensor(
                context.categorical, dtype=torch.int64
            )
    graph, local_maps = _collate_graphs(
        vectorized,
        token_schemas=schema.tokens,
        relationship_schemas=schema.relationships,
    )
    token_kinds = graph.token_kinds
    token_kind_index = {kind: index for index, kind in enumerate(token_kinds)}

    candidate_kinds = tuple(item.kind for item in schema.candidates)
    candidate_kind_index = {kind: index for index, kind in enumerate(candidate_kinds)}
    max_candidates = max(len(item.candidates) for item in vectorized)
    base = _empty_table(
        batch_size,
        max_candidates,
        max(expanded_numeric_width(item) for item in schema.candidates),
        max(len(item.categorical) for item in schema.candidates),
        max(len(item.references) for item in schema.candidates),
    )
    shape = (batch_size, max_candidates)
    candidates = CandidateTable(
        **base.__dict__,
        kind_indices=torch.full(shape, -1, dtype=torch.int64),
        target_indices=torch.full(shape, -1, dtype=torch.int64),
        target_kind_indices=torch.full(shape, -1, dtype=torch.int64),
        target_valid=torch.zeros(shape, dtype=torch.bool),
    )
    for batch_index, decision in enumerate(vectorized):
        for row, candidate in enumerate(decision.candidates):
            candidates.mask[batch_index, row] = True
            candidates.kind_indices[batch_index, row] = candidate_kind_index[candidate.kind]
            candidates.numeric[batch_index, row, : len(candidate.numeric)] = torch.tensor(
                candidate.numeric, dtype=torch.float32
            )
            candidates.numeric_valid[batch_index, row, : len(candidate.numeric_valid)] = (
                torch.tensor(candidate.numeric_valid, dtype=torch.bool)
            )
            candidates.categorical[batch_index, row, : len(candidate.categorical)] = torch.tensor(
                candidate.categorical, dtype=torch.int64
            )
            for column, (reference, is_valid) in enumerate(
                zip(candidate.references, candidate.reference_valid, strict=True)
            ):
                if is_valid:
                    target = decision.tokens[reference]
                    target_kind, target_offset = local_maps[batch_index][target.local_ref]
                    candidates.references[batch_index, row, column] = target_offset
                    candidates.reference_valid[batch_index, row, column] = True
                    candidates.reference_kind_indices[batch_index, row, column] = token_kind_index[
                        target_kind
                    ]
                    if column == 0:
                        candidates.target_indices[batch_index, row] = target_offset
                        candidates.target_kind_indices[batch_index, row] = token_kind_index[
                            target_kind
                        ]
                        candidates.target_valid[batch_index, row] = True

    return DecisionBatch(
        tokens=graph.tokens,
        relationships=graph.relationships,
        decision_context=decision_context,
        candidates=candidates,
        candidate_ids=tuple(item.candidate_ids for item in vectorized),
        token_kinds=token_kinds,
        candidate_kinds=candidate_kinds,
    )


def _validate_values(
    record: object,
    record_schema: RecordFeatureSchema,
    *,
    label: str,
    references: bool,
) -> None:
    # model_copy/model_construct can bypass Pydantic field validation; cached
    # or caller-supplied vectors must still satisfy the actual tensor layout.
    numeric = getattr(record, "numeric", None)
    numeric_valid = getattr(record, "numeric_valid", None)
    categorical = getattr(record, "categorical", None)
    if not isinstance(numeric, tuple) or len(numeric) != expanded_numeric_width(record_schema):
        raise ValueError(f"{label} numeric width does not match stable-value schema")
    if not isinstance(numeric_valid, tuple) or len(numeric_valid) != len(numeric):
        raise ValueError(f"{label} numeric-valid width does not match stable-value schema")
    limit = torch.finfo(torch.float32).max
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not -limit <= value <= limit
        for value in numeric
    ):
        raise ValueError(f"{label} numeric values must be finite and representable as float32")
    if any(not isinstance(value, bool) for value in numeric_valid):
        raise ValueError(f"{label} numeric validity values must be boolean")
    if not isinstance(categorical, tuple) or len(categorical) != len(record_schema.categorical):
        raise ValueError(f"{label} categorical width does not match stable-value schema")
    for value, feature in zip(categorical, record_schema.categorical, strict=True):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} categorical values must be integer indexes")
        if value < 0 or value >= len(feature.vocabulary):
            raise ValueError(f"{label} categorical index is outside its vocabulary")
    if not references:
        return
    record_references = getattr(record, "references", None)
    reference_valid = getattr(record, "reference_valid", None)
    if not isinstance(record_references, tuple) or len(record_references) != len(
        record_schema.references
    ):
        raise ValueError(f"{label} reference width does not match stable-value schema")
    if not isinstance(reference_valid, tuple) or len(reference_valid) != len(record_references):
        raise ValueError(f"{label} reference-valid width does not match stable-value schema")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in record_references):
        raise ValueError(f"{label} references must be integer indexes")
    if any(not isinstance(value, bool) for value in reference_valid):
        raise ValueError(f"{label} reference validity values must be boolean")
    for reference_feature, valid in zip(record_schema.references, reference_valid, strict=True):
        if reference_feature.required and not valid:
            raise ValueError(f"required reference {reference_feature.source!r} is missing")


def _validate_stable_value(
    value: VectorizedStableValue,
    schema: StableValueTensorSchema,
) -> None:
    if value.tensor_schema_id != schema.schema_id:
        raise ValueError("prevectorized stable value schema ID does not match")
    if value.tensor_schema_version != schema.schema_version:
        raise ValueError("prevectorized stable value schema version does not match")
    if value.tensor_schema_digest != schema.digest:
        raise ValueError("prevectorized stable value schema digest does not match")
    if not isinstance(value.viewer, Viewer):
        raise ValueError("prevectorized stable value has malformed viewer context")
    if (
        value.viewer.schema_version != 2
        or not value.viewer.private_hero_id
        or value.viewer.perspective_team not in {"RED", "BLUE"}
    ):
        raise ValueError("prevectorized stable value requires viewer entitlement and perspective")

    _validate_values(
        value.value_context,
        schema.value_context,
        label="stable value context",
        references=False,
    )
    boundary_column = next(
        (
            index
            for index, feature in enumerate(schema.value_context.categorical)
            if feature.source == "boundary_kind"
        ),
        None,
    )
    if boundary_column is None:
        raise ValueError("stable-value schema has no boundary-kind context")
    boundary_feature = schema.value_context.categorical[boundary_column]
    boundary_index = value.value_context.categorical[boundary_column]
    boundary_kind = boundary_feature.vocabulary[boundary_index]
    if boundary_kind not in {"ACTOR_READY", "PLANNING_READY"}:
        raise ValueError("stable value context has unknown boundary kind")

    token_by_kind = {item.kind: item for item in schema.tokens}
    indexes_by_ref: dict[str, int] = {}
    for index, token in enumerate(value.tokens):
        record_schema = token_by_kind.get(token.kind)
        if record_schema is None:
            raise ValueError(f"unknown token kind: {token.kind!r}")
        if not token.local_ref or token.local_ref in indexes_by_ref:
            raise ValueError("stable value token local refs must be non-empty and unique")
        indexes_by_ref[token.local_ref] = index
        _validate_values(token, record_schema, label=f"token {token.kind!r}", references=True)
    for token in value.tokens:
        for reference, valid in zip(token.references, token.reference_valid, strict=True):
            if valid and not 0 <= reference < len(value.tokens):
                raise ValueError("stable value token reference is outside the graph")
            if not valid and reference != -1:
                raise ValueError("invalid stable value token reference must use -1")

    relationship_by_kind = {item.kind: item for item in schema.relationships}
    for relationship in value.relationships:
        record_schema = relationship_by_kind.get(relationship.kind)
        if record_schema is None:
            raise ValueError(f"unknown relationship kind: {relationship.kind!r}")
        _validate_values(
            relationship,
            record_schema,
            label=f"relationship {relationship.kind!r}",
            references=False,
        )
        if (
            isinstance(relationship.source_index, bool)
            or not isinstance(relationship.source_index, int)
            or isinstance(relationship.target_index, bool)
            or not isinstance(relationship.target_index, int)
        ):
            raise ValueError("stable value relationship indexes must be integers")
        if (
            relationship.source_ref not in indexes_by_ref
            or relationship.target_ref not in indexes_by_ref
        ):
            raise ValueError("stable value relationship ref does not identify a token")
        if relationship.source_index != indexes_by_ref[relationship.source_ref]:
            raise ValueError("stable value relationship source index/ref mismatch")
        if relationship.target_index != indexes_by_ref[relationship.target_ref]:
            raise ValueError("stable value relationship target index/ref mismatch")


def collate_stable_values(
    observations: Sequence[StableValueObservation | VectorizedStableValue],
    *,
    schema: StableValueTensorSchema,
) -> StableValueBatch:
    """Collate candidate-free stable observations under a pinned value schema."""
    if not isinstance(schema, StableValueTensorSchema):
        raise TypeError(
            "schema must be a StableValueTensorSchema, not a decision TensorFeatureSchema"
        )
    if not observations:
        raise ValueError("stable value batch cannot be empty")

    vectorized: list[VectorizedStableValue] = []
    for observation in observations:
        if isinstance(observation, StableValueObservation):
            vectorized.append(schema.vectorize(observation))
        elif isinstance(observation, VectorizedStableValue):
            vectorized.append(observation)
        else:
            raise TypeError(
                "stable value batch items must be StableValueObservation or "
                "VectorizedStableValue"
            )
    for item in vectorized:
        _validate_stable_value(item, schema)
    graph, _ = _collate_graphs(
        vectorized,
        token_schemas=schema.tokens,
        relationship_schemas=schema.relationships,
    )
    value_context = _empty_feature_table(len(vectorized), 1, schema.value_context)
    for batch_index, value in enumerate(vectorized):
        context = value.value_context
        value_context.mask[batch_index, 0] = True
        value_context.numeric[batch_index, 0] = torch.tensor(context.numeric, dtype=torch.float32)
        value_context.numeric_valid[batch_index, 0] = torch.tensor(
            context.numeric_valid, dtype=torch.bool
        )
        value_context.categorical[batch_index, 0] = torch.tensor(
            context.categorical, dtype=torch.int64
        )
    return StableValueBatch(
        graph=graph,
        value_context=value_context,
        viewers=tuple(item.viewer for item in vectorized),
    )


__all__ = [
    "Batch",
    "CandidateTable",
    "DecisionBatch",
    "FeatureTable",
    "GraphBatch",
    "RelationshipTable",
    "StableValueBatch",
    "collate_decisions",
    "collate_stable_values",
    "masked_max",
    "masked_mean",
    "masked_softmax",
    "safe_gather",
]
