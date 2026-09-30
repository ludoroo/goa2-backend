"""Gen1 shared graph encoder with isolated policy and stable-value forwards.

This module intentionally leaves the released joint model untouched.  Policy
and stable-boundary value inference share graph processing, while their context
encoders and output heads remain separate semantic parameter groups.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .batching import (
    DecisionBatch,
    FeatureTable,
    GraphBatch,
    StableValueBatch,
    masked_mean,
)
from .model import _MessageLayer, _RecordEncoder
from .schema import (
    STABLE_VALUE_TENSOR_SCHEMA_ID,
    STABLE_VALUE_TENSOR_SCHEMA_VERSION,
    TENSOR_SCHEMA_ID,
    TENSOR_SCHEMA_VERSION,
    StableValueTensorSchema,
    StableValueTensorSchemaID,
    StableValueTensorSchemaVersion,
    TensorFeatureSchema,
    TensorSchemaID,
    TensorSchemaVersion,
)

GEN1_ARCHITECTURE_ID: Literal["goa2-gen1-policy-stable-value-v1"] = (
    "goa2-gen1-policy-stable-value-v1"
)


@dataclass(frozen=True, slots=True)
class Gen1ModelConfig:
    """Versioned architecture parameters pinned to both tensor schemas."""

    decision_schema_digest: str
    stable_value_schema_digest: str
    token_width: int
    state_width: int
    candidate_width: int
    message_passing_layers: int
    dropout: float = 0.0
    architecture_id: Literal["goa2-gen1-policy-stable-value-v1"] = GEN1_ARCHITECTURE_ID
    architecture_version: Literal[1] = 1
    decision_schema_id: TensorSchemaID = TENSOR_SCHEMA_ID
    decision_schema_version: TensorSchemaVersion = TENSOR_SCHEMA_VERSION
    stable_value_schema_id: StableValueTensorSchemaID = STABLE_VALUE_TENSOR_SCHEMA_ID
    stable_value_schema_version: StableValueTensorSchemaVersion = STABLE_VALUE_TENSOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        identity = (
            self.architecture_id,
            self.architecture_version,
            self.decision_schema_id,
            self.decision_schema_version,
            self.stable_value_schema_id,
            self.stable_value_schema_version,
        )
        expected = (
            GEN1_ARCHITECTURE_ID,
            1,
            TENSOR_SCHEMA_ID,
            TENSOR_SCHEMA_VERSION,
            STABLE_VALUE_TENSOR_SCHEMA_ID,
            STABLE_VALUE_TENSOR_SCHEMA_VERSION,
        )
        versions = (
            self.architecture_version,
            self.decision_schema_version,
            self.stable_value_schema_version,
        )
        if identity != expected or any(type(version) is not int for version in versions):
            raise ValueError("unsupported gen1 architecture/tensor schema identity")
        for name in ("decision_schema_digest", "stable_value_schema_digest"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "token_width",
            "state_width",
            "candidate_width",
            "message_passing_layers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in the range [0, 1)")


@dataclass(frozen=True)
class PolicyHeadOutput:
    """Policy-only model output aligned with a decision batch's candidates."""

    policy_logits: Tensor


@dataclass(frozen=True)
class StableValueHeadOutput:
    """Candidate-free stable-boundary value output."""

    value: Tensor


class Gen1PolicyValueModel(nn.Module):
    """Compact graph model with native, isolated policy and stable-value paths."""

    def __init__(
        self,
        *,
        decision_schema: TensorFeatureSchema,
        stable_value_schema: StableValueTensorSchema,
        config: Gen1ModelConfig,
    ) -> None:
        super().__init__()
        decision_identity = (
            config.decision_schema_id,
            config.decision_schema_version,
            config.decision_schema_digest,
        )
        if decision_identity != (
            decision_schema.schema_id,
            decision_schema.schema_version,
            decision_schema.digest,
        ):
            raise ValueError("model configuration is incompatible with decision schema")
        stable_value_identity = (
            config.stable_value_schema_id,
            config.stable_value_schema_version,
            config.stable_value_schema_digest,
        )
        if stable_value_identity != (
            stable_value_schema.schema_id,
            stable_value_schema.schema_version,
            stable_value_schema.digest,
        ):
            raise ValueError("model configuration is incompatible with stable-value schema")
        if (
            decision_schema.tokens != stable_value_schema.tokens
            or decision_schema.relationships != stable_value_schema.relationships
        ):
            raise ValueError("decision and stable-value schemas must declare the same graph")
        if decision_schema.decision_context is None:
            raise ValueError("gen1 model requires a decision-context feature schema")

        self.config = config
        self.token_kinds = tuple(item.kind for item in decision_schema.tokens)
        self.relationship_kinds = tuple(item.kind for item in decision_schema.relationships)
        self.candidate_kinds = tuple(item.kind for item in decision_schema.candidates)

        self.token_encoders = nn.ModuleDict(
            {
                item.kind: _RecordEncoder(item, config.token_width, config.dropout)
                for item in decision_schema.tokens
            }
        )
        self.edge_encoders = nn.ModuleDict(
            {
                item.kind: _RecordEncoder(item, config.token_width, config.dropout)
                for item in decision_schema.relationships
            }
        )
        self.message_layers = nn.ModuleList(
            _MessageLayer(
                self.token_kinds,
                self.relationship_kinds,
                config.token_width,
                config.dropout,
            )
            for _ in range(config.message_passing_layers)
        )
        self.state_encoder = nn.Sequential(
            nn.Linear((len(self.token_kinds) + 1) * config.token_width, config.state_width),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_width, config.state_width),
            nn.ReLU(),
        )

        self.decision_encoder = _RecordEncoder(
            decision_schema.decision_context,
            config.token_width,
            config.dropout,
        )
        self.candidate_encoders = nn.ModuleDict(
            {
                item.kind: _RecordEncoder(item, config.candidate_width, config.dropout)
                for item in decision_schema.candidates
            }
        )
        policy_input_width = config.state_width + config.candidate_width + config.token_width + 1
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_width, config.candidate_width),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.candidate_width, 1),
        )

        self.value_context_encoder = _RecordEncoder(
            stable_value_schema.value_context,
            config.token_width,
            config.dropout,
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.state_width, config.state_width),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_width, 1),
            nn.Tanh(),
        )

    def forward_policy(self, batch: DecisionBatch) -> PolicyHeadOutput:
        """Score legal candidates without evaluating the stable-value head."""
        if batch.candidate_kinds != self.candidate_kinds:
            raise ValueError("batch candidate kinds are incompatible with model schema")
        if batch.decision_context is None:
            raise ValueError("gen1 policy forward requires one decision-context row")
        embeddings = self._encode_graph(batch.graph)
        context = self._encode_single_context(
            self.decision_encoder,
            batch.decision_context,
            batch_size=batch.candidates.mask.shape[0],
            label="decision context",
        )
        state = self._encode_state(embeddings, batch.graph, context)
        candidates = self._encode_candidates(batch)
        targets, target_valid = batch.gather_candidate_targets(embeddings)
        policy_state = state.unsqueeze(1).expand(-1, batch.candidates.mask.shape[1], -1)
        logits = self.policy_head(
            torch.cat(
                (
                    candidates,
                    targets,
                    target_valid.unsqueeze(-1).to(state.dtype),
                    policy_state,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        logits = torch.where(batch.candidates.mask, logits, torch.zeros_like(logits))
        return PolicyHeadOutput(policy_logits=logits)

    def forward_stable_value(self, batch: StableValueBatch) -> StableValueHeadOutput:
        """Evaluate stable-boundary state value without touching policy components."""
        embeddings = self._encode_graph(batch.graph)
        batch_size = batch.graph.tokens[self.token_kinds[0]].mask.shape[0]
        if len(batch.viewers) != batch_size:
            raise ValueError("stable-value viewers must align with graph batch items")
        context = self._encode_single_context(
            self.value_context_encoder,
            batch.value_context,
            batch_size=batch_size,
            label="stable-value context",
        )
        state = self._encode_state(embeddings, batch.graph, context)
        return StableValueHeadOutput(value=self.value_head(state).squeeze(-1))

    def _encode_graph(self, graph: GraphBatch) -> dict[str, Tensor]:
        if graph.token_kinds != self.token_kinds:
            raise ValueError("batch token kinds are incompatible with model schema")
        if set(graph.tokens) != set(self.token_kinds):
            raise ValueError("batch token tables are incompatible with model schema")
        if set(graph.relationships) != set(self.relationship_kinds):
            raise ValueError("batch relationship tables are incompatible with model schema")
        embeddings = {
            kind: self.token_encoders[kind](graph.tokens[kind]) for kind in self.token_kinds
        }
        edge_embeddings = {
            kind: self.edge_encoders[kind](graph.relationships[kind])
            for kind in self.relationship_kinds
        }
        for layer in self.message_layers:
            embeddings = layer(
                embeddings,
                graph.tokens,
                graph.relationships,
                edge_embeddings,
                self.token_kinds,
            )
        return embeddings

    def _encode_state(
        self,
        embeddings: dict[str, Tensor],
        graph: GraphBatch,
        context: Tensor,
    ) -> Tensor:
        pooled_parts = [
            masked_mean(embeddings[kind], graph.tokens[kind].mask, dim=1)
            for kind in self.token_kinds
        ]
        pooled_parts.append(context)
        return self.state_encoder(torch.cat(pooled_parts, dim=-1))

    @staticmethod
    def _encode_single_context(
        encoder: _RecordEncoder,
        table: FeatureTable,
        *,
        batch_size: int,
        label: str,
    ) -> Tensor:
        if table.mask.shape != (batch_size, 1) or not bool(table.mask.all()):
            raise ValueError(f"{label} must contain exactly one valid row per batch item")
        return encoder(table).squeeze(1)

    def _encode_candidates(self, batch: DecisionBatch) -> Tensor:
        shape = (*batch.candidates.mask.shape, self.config.candidate_width)
        output = batch.candidates.numeric.new_zeros(shape)
        for index, kind in enumerate(self.candidate_kinds):
            kind_mask = batch.candidates.mask & (batch.candidates.kind_indices == index)
            table = FeatureTable(
                numeric=batch.candidates.numeric,
                numeric_valid=batch.candidates.numeric_valid,
                categorical=batch.candidates.categorical,
                references=batch.candidates.references,
                reference_valid=batch.candidates.reference_valid,
                reference_kind_indices=batch.candidates.reference_kind_indices,
                mask=kind_mask,
            )
            encoded = self.candidate_encoders[kind](table)
            output = torch.where(kind_mask.unsqueeze(-1), encoded, output)
        return output

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Return the exhaustive, disjoint semantic parameter partition."""
        groups = {
            "shared": tuple(
                self._parameters_of(
                    (
                        self.token_encoders,
                        self.edge_encoders,
                        self.message_layers,
                        self.state_encoder,
                    )
                )
            ),
            "policy": tuple(
                self._parameters_of(
                    (self.decision_encoder, self.candidate_encoders, self.policy_head)
                )
            ),
            "value": tuple(self._parameters_of((self.value_context_encoder, self.value_head))),
        }
        grouped_parameter_ids = [
            id(parameter) for parameters in groups.values() for parameter in parameters
        ]
        model_parameter_ids = {id(parameter) for parameter in self.parameters()}
        if (
            len(grouped_parameter_ids) != len(set(grouped_parameter_ids))
            or set(grouped_parameter_ids) != model_parameter_ids
        ):
            raise RuntimeError("parameter groups must be an exhaustive and disjoint partition")
        return groups

    @staticmethod
    def _parameters_of(modules: Iterable[nn.Module]) -> Iterable[nn.Parameter]:
        for module in modules:
            yield from module.parameters()


__all__ = [
    "GEN1_ARCHITECTURE_ID",
    "Gen1ModelConfig",
    "Gen1PolicyValueModel",
    "PolicyHeadOutput",
    "StableValueHeadOutput",
]
