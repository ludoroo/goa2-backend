"""Shared tensor-chunk evaluation helpers for offline training tools."""

from __future__ import annotations

from typing import Any

from automata.training.indexed_dataset import IndexedTrainingChunk
from automata.training.metrics import (
    JointMetricsAccumulator,
    PolicyMetricInput,
    ValueMetricInput,
)


def add_chunk_metric_inputs(
    accumulator: JointMetricsAccumulator,
    output: Any,
    chunk: IndexedTrainingChunk,
) -> None:
    """Add one validated model/chunk result to the bounded metric accumulator."""

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
                input_request_type=metadata.input_request_type,
                semantic_role=metadata.semantic_role,
                can_skip=metadata.can_skip,
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


__all__ = ["add_chunk_metric_inputs"]
