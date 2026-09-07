"""Compact, explicit PyTorch joint policy/value model.

This module is intentionally outside :mod:`automata.models`' facade so importing
the observation contracts does not make PyTorch a server dependency.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .batching import DecisionBatch, FeatureTable, RelationshipTable, masked_mean, safe_gather
from .schema import RecordFeatureSchema, TensorFeatureSchema


@dataclass(frozen=True, slots=True)
class JointModelConfig:
    """Versioned architecture parameters pinned to one tensor schema."""

    model_version: int
    schema_digest: str
    token_width: int
    state_width: int
    candidate_width: int
    message_passing_layers: int
    dropout: float = 0.0
    schema_id: str = "goa2-tensor-features-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.model_version != 1:
            raise ValueError("unsupported model version")
        if self.schema_version != 1 or not self.schema_id:
            raise ValueError("invalid tensor schema identity")
        if len(self.schema_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.schema_digest
        ):
            raise ValueError("schema digest must be a lowercase SHA-256 digest")
        for name in ("token_width", "state_width", "candidate_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.message_passing_layers <= 0:
            raise ValueError("message_passing_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")


@dataclass(frozen=True, slots=True)
class JointModelOutput:
    policy_logits: Tensor
    value: Tensor


class _RecordEncoder(nn.Module):
    """Encode one schema kind without ever embedding reference or public IDs."""

    def __init__(self, schema: RecordFeatureSchema, width: int, dropout: float) -> None:
        super().__init__()
        self.numeric_width = len(schema.numeric)
        self.categorical_width = len(schema.categorical)
        embedding_widths = [min(8, max(2, width // 4)) for _ in schema.categorical]
        self.embeddings = nn.ModuleList(
            nn.Embedding(len(feature.vocabulary), embedding_widths[index], padding_idx=0)
            for index, feature in enumerate(schema.categorical)
        )
        input_width = self.numeric_width * 2 + sum(embedding_widths)
        # A leading constant gives featureless record kinds a learned kind representation.
        self.network = nn.Sequential(
            nn.Linear(input_width + 1, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, table: FeatureTable) -> Tensor:
        mask = table.mask.unsqueeze(-1)
        numeric = torch.where(
            mask,
            table.numeric[..., : self.numeric_width],
            torch.zeros((), dtype=table.numeric.dtype, device=table.numeric.device),
        )
        numeric_valid = torch.where(
            mask,
            table.numeric_valid[..., : self.numeric_width],
            torch.zeros((), dtype=torch.bool, device=table.numeric_valid.device),
        ).to(numeric.dtype)
        pieces = [torch.ones((*table.mask.shape, 1), dtype=numeric.dtype, device=numeric.device)]
        pieces.extend((numeric, numeric_valid))
        for column, embedding in enumerate(self.embeddings):
            values = torch.where(
                table.mask,
                table.categorical[..., column],
                torch.zeros((), dtype=torch.int64, device=table.categorical.device),
            )
            pieces.append(embedding(values))
        encoded = self.network(torch.cat(pieces, dim=-1))
        return torch.where(mask, encoded, torch.zeros_like(encoded))


class _MessageLayer(nn.Module):
    def __init__(
        self,
        token_kinds: tuple[str, ...],
        relationship_kinds: tuple[str, ...],
        width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.messages = nn.ModuleDict(
            {
                kind: nn.Sequential(
                    nn.Linear(width * 3, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, width),
                )
                for kind in relationship_kinds
            }
        )
        self.updates = nn.ModuleDict(
            {
                kind: nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Dropout(dropout))
                for kind in token_kinds
            }
        )

    def forward(
        self,
        embeddings: dict[str, Tensor],
        token_tables: dict[str, FeatureTable],
        relationships: dict[str, RelationshipTable],
        edge_embeddings: dict[str, Tensor],
        token_kinds: tuple[str, ...],
    ) -> dict[str, Tensor]:
        totals = {kind: torch.zeros_like(value) for kind, value in embeddings.items()}
        counts = {
            kind: value.new_zeros((*value.shape[:2], 1)) for kind, value in embeddings.items()
        }
        for relationship_kind, table in relationships.items():
            edge_values = edge_embeddings[relationship_kind]
            for source_kind_index, source_kind in enumerate(token_kinds):
                source = embeddings[source_kind]
                for target_kind_index, target_kind in enumerate(token_kinds):
                    valid = (
                        table.mask
                        & (table.source_kind_indices == source_kind_index)
                        & (table.target_kind_indices == target_kind_index)
                    )
                    if not valid.any():
                        continue
                    source_values, source_valid = safe_gather(
                        source,
                        table.source_indices,
                        valid,
                        source_mask=token_tables[source_kind].mask,
                    )
                    target_values, endpoint_valid = safe_gather(
                        embeddings[target_kind],
                        table.target_indices,
                        source_valid,
                        source_mask=token_tables[target_kind].mask,
                    )
                    message = self.messages[relationship_kind](
                        torch.cat((source_values, target_values, edge_values), dim=-1)
                    )
                    message = torch.where(
                        endpoint_valid.unsqueeze(-1), message, torch.zeros_like(message)
                    )
                    self._scatter_add(
                        totals[target_kind],
                        counts[target_kind],
                        message,
                        table.target_indices,
                        endpoint_valid,
                    )
        updated: dict[str, Tensor] = {}
        for kind, values in embeddings.items():
            aggregate = totals[kind] / counts[kind].clamp_min(1)
            residual = values + self.updates[kind](torch.cat((values, aggregate), dim=-1))
            updated[kind] = torch.where(
                token_tables[kind].mask.unsqueeze(-1), residual, torch.zeros_like(residual)
            )
        return updated

    @staticmethod
    def _scatter_add(
        output: Tensor, counts: Tensor, values: Tensor, indices: Tensor, valid: Tensor
    ) -> None:
        rows = output.shape[1]
        if rows == 0:
            return
        batch_offsets = torch.arange(output.shape[0], device=output.device).unsqueeze(1) * rows
        safe_indices = torch.where(valid, indices, torch.zeros_like(indices)) + batch_offsets
        flat_indices = safe_indices[valid]
        output.view(-1, output.shape[-1]).index_add_(0, flat_indices, values[valid])
        counts.view(-1, 1).index_add_(
            0,
            flat_indices,
            torch.ones((flat_indices.numel(), 1), dtype=counts.dtype, device=counts.device),
        )


class JointPolicyValueModel(nn.Module):
    """Small relation-aware model with one shared state and two output heads."""

    def __init__(self, *, schema: TensorFeatureSchema, config: JointModelConfig) -> None:
        super().__init__()
        if (
            config.schema_id != schema.schema_id
            or config.schema_version != schema.schema_version
            or config.schema_digest != schema.digest
        ):
            raise ValueError("model configuration is incompatible with tensor schema")
        self.config = config
        self.token_kinds = tuple(item.kind for item in schema.tokens)
        self.relationship_kinds = tuple(item.kind for item in schema.relationships)
        self.candidate_kinds = tuple(item.kind for item in schema.candidates)
        self.token_encoders = nn.ModuleDict(
            {
                item.kind: _RecordEncoder(item, config.token_width, config.dropout)
                for item in schema.tokens
            }
        )
        self.edge_encoders = nn.ModuleDict(
            {
                item.kind: _RecordEncoder(item, config.token_width, config.dropout)
                for item in schema.relationships
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
            nn.Linear(len(self.token_kinds) * config.token_width, config.state_width),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_width, config.state_width),
            nn.ReLU(),
        )
        self.candidate_encoders = nn.ModuleDict(
            {
                item.kind: _RecordEncoder(item, config.candidate_width, config.dropout)
                for item in schema.candidates
            }
        )
        policy_input = config.state_width + config.candidate_width + config.token_width + 1
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input, config.candidate_width),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.candidate_width, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.state_width, config.state_width),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_width, 1),
            nn.Tanh(),
        )

    def forward(self, batch: DecisionBatch) -> JointModelOutput:
        if batch.token_kinds != self.token_kinds or batch.candidate_kinds != self.candidate_kinds:
            raise ValueError("batch kinds are incompatible with model schema")
        embeddings = {
            kind: self.token_encoders[kind](batch.tokens[kind]) for kind in self.token_kinds
        }
        edge_embeddings = {
            kind: self.edge_encoders[kind](batch.relationships[kind])
            for kind in self.relationship_kinds
        }
        for layer in self.message_layers:
            embeddings = layer(
                embeddings,
                batch.tokens,
                batch.relationships,
                edge_embeddings,
                self.token_kinds,
            )
        pooled = torch.cat(
            [
                masked_mean(embeddings[kind], batch.tokens[kind].mask, dim=1)
                for kind in self.token_kinds
            ],
            dim=-1,
        )
        state = self.state_encoder(pooled)
        candidates = self._encode_candidates(batch)
        targets, target_valid = batch.gather_candidate_targets(embeddings)
        policy_state = state.unsqueeze(1).expand(-1, batch.candidates.mask.shape[1], -1)
        logits = self.policy_head(
            torch.cat(
                (candidates, targets, target_valid.unsqueeze(-1).to(state.dtype), policy_state),
                dim=-1,
            )
        ).squeeze(-1)
        logits = torch.where(batch.candidates.mask, logits, torch.zeros_like(logits))
        return JointModelOutput(policy_logits=logits, value=self.value_head(state).squeeze(-1))

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
        """Return an exhaustive, disjoint semantic partition of model parameters."""

        shared_modules: tuple[nn.Module, ...] = (
            self.token_encoders,
            self.edge_encoders,
            self.message_layers,
            self.state_encoder,
        )
        groups = {
            "shared": tuple(self._parameters_of(shared_modules)),
            "policy": tuple(self._parameters_of((self.candidate_encoders, self.policy_head))),
            "value": tuple(self.value_head.parameters()),
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


__all__ = ["JointModelConfig", "JointModelOutput", "JointPolicyValueModel"]
