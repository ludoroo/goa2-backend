"""Explicit PyTorch boundary for ragged learned-model decision batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor

from ..contracts import CandidateID, DecisionObservation
from .schema import RecordFeatureSchema, TensorFeatureSchema, VectorizedDecision


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
class DecisionBatch:
    """CPU tensors plus the typed identities needed to decode model output."""

    tokens: dict[str, FeatureTable]
    relationships: dict[str, RelationshipTable]
    candidates: CandidateTable
    candidate_ids: tuple[tuple[CandidateID, ...], ...]
    token_kinds: tuple[str, ...]
    candidate_kinds: tuple[str, ...]

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
        len(schema.numeric),
        len(schema.categorical),
        len(schema.references),
    )


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
    token_kinds = tuple(item.kind for item in schema.tokens)
    token_kind_index = {kind: index for index, kind in enumerate(token_kinds)}
    token_tables: dict[str, FeatureTable] = {}
    local_maps: list[dict[str, tuple[str, int]]] = []
    for decision in vectorized:
        counts: dict[str, int] = {}
        local: dict[str, tuple[str, int]] = {}
        for token in decision.tokens:
            offset = counts.get(token.kind, 0)
            counts[token.kind] = offset + 1
            local[token.local_ref] = (token.kind, offset)
        local_maps.append(local)

    for record_schema in schema.tokens:
        token_records_by_batch = [
            [record for record in decision.tokens if record.kind == record_schema.kind]
            for decision in vectorized
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
    for record_schema in schema.relationships:
        relationship_records_by_batch = [
            [record for record in decision.relationships if record.kind == record_schema.kind]
            for decision in vectorized
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

    candidate_kinds = tuple(item.kind for item in schema.candidates)
    candidate_kind_index = {kind: index for index, kind in enumerate(candidate_kinds)}
    max_candidates = max(len(item.candidates) for item in vectorized)
    base = _empty_table(
        batch_size,
        max_candidates,
        max(len(item.numeric) for item in schema.candidates),
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
        tokens=token_tables,
        relationships=relationship_tables,
        candidates=candidates,
        candidate_ids=tuple(item.candidate_ids for item in vectorized),
        token_kinds=token_kinds,
        candidate_kinds=candidate_kinds,
    )


__all__ = [
    "Batch",
    "CandidateTable",
    "DecisionBatch",
    "FeatureTable",
    "RelationshipTable",
    "collate_decisions",
    "masked_max",
    "masked_mean",
    "masked_softmax",
    "safe_gather",
]
