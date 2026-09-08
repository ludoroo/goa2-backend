"""Joint policy/value objectives for the bounded policy/value model.

The value head exposes a score in ``[-1, 1]``, not a logit.  This module maps
that score to a win probability with ``p = (value + 1) / 2`` and computes
binary cross-entropy after clamping ``p`` away from zero and one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from automata.models.shared_encoder.model import JointModelOutput


@dataclass(frozen=True, slots=True)
class JointLossConfig:
    """Coefficients and numerical bounds for the joint objective."""

    value_weight: float = 1.0
    entropy_weight: float = 0.0
    l2_weight: float = 0.0
    value_probability_epsilon: float = 1e-7

    def validate(self) -> None:
        for name in ("value_weight", "entropy_weight", "l2_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.value_probability_epsilon) or not (
            0.0 < self.value_probability_epsilon < 0.5
        ):
            raise ValueError("value_probability_epsilon must be finite and between zero and 0.5")


@dataclass(frozen=True, slots=True)
class JointLossResult:
    """Differentiable scalar components of the joint objective."""

    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    l2: Tensor


def equal_game_weights(game_ids: Sequence[str], *, like: Tensor) -> Tensor:
    """Return row weights such that every represented game sums to equal weight."""
    if not game_ids:
        raise ValueError("game_ids must not be empty")
    counts: dict[str, int] = {}
    for game_id in game_ids:
        if not game_id:
            raise ValueError("game_ids must contain non-empty strings")
        counts[game_id] = counts.get(game_id, 0) + 1
    game_count = len(counts)
    return like.new_tensor([1.0 / (game_count * counts[item]) for item in game_ids])


def joint_policy_value_loss(
    output: JointModelOutput,
    *,
    legal_mask: Tensor,
    policy_targets: Tensor,
    value_targets: Tensor,
    game_ids: Sequence[str],
    decision_weights: Sequence[float] | None = None,
    parameters: Iterable[nn.Parameter] = (),
    config: JointLossConfig | None = None,
) -> JointLossResult:
    """Compute the weighted policy CE and bounded-probability value BCE."""
    config = config or JointLossConfig()
    config.validate()
    logits = output.policy_logits
    _validate_training_tensors(
        logits, output.value, legal_mask, policy_targets, value_targets, len(game_ids)
    )
    weights = (
        equal_game_weights(game_ids, like=logits)
        if decision_weights is None
        else logits.new_tensor(decision_weights)
    )
    if weights.shape != (len(game_ids),) or not torch.isfinite(weights).all().item():
        raise ValueError("decision_weights must be finite and align with the batch")
    if (weights < 0).any().item() or weights.sum().item() <= 0:
        raise ValueError("decision_weights must be non-negative with positive total weight")
    masked_logits = logits.masked_fill(~legal_mask, -torch.inf)
    log_probabilities = torch.log_softmax(masked_logits, dim=-1)
    policy_per_row = -torch.where(
        legal_mask, policy_targets * log_probabilities, torch.zeros_like(policy_targets)
    ).sum(dim=-1)
    policy = (weights * policy_per_row).sum()

    probabilities = ((output.value + 1.0) / 2.0).clamp(
        config.value_probability_epsilon, 1.0 - config.value_probability_epsilon
    )
    binary_targets = (value_targets + 1.0) / 2.0
    value_per_row = -(
        binary_targets * probabilities.log() + (1.0 - binary_targets) * (1.0 - probabilities).log()
    )
    value = (weights * value_per_row).sum()

    policy_probabilities = log_probabilities.exp()
    finite_log_probabilities = torch.where(
        legal_mask, log_probabilities, torch.zeros_like(log_probabilities)
    )
    entropy_per_row = -(policy_probabilities * finite_log_probabilities).sum(dim=-1)
    entropy = (weights * entropy_per_row).sum()

    parameter_tuple = tuple(parameters)
    if any(not torch.isfinite(parameter).all().item() for parameter in parameter_tuple):
        raise ValueError("regularized parameters must contain only finite values")
    l2 = sum(
        (parameter.square().sum() for parameter in parameter_tuple), start=logits.new_zeros(())
    )
    total = (
        policy
        + config.value_weight * value
        - config.entropy_weight * entropy
        + config.l2_weight * l2
    )
    if not torch.isfinite(total).item():
        raise ValueError("joint loss must be finite")
    return JointLossResult(total, policy, value, entropy, l2)


def _validate_training_tensors(
    logits: Tensor,
    values: Tensor,
    legal_mask: Tensor,
    policy_targets: Tensor,
    value_targets: Tensor,
    game_id_count: int,
) -> None:
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("policy_logits must be a rank-two floating tensor")
    if legal_mask.dtype != torch.bool or legal_mask.shape != logits.shape:
        raise ValueError("legal_mask must be boolean and match policy_logits")
    if policy_targets.shape != logits.shape or not policy_targets.is_floating_point():
        raise ValueError("policy_targets must be floating and match policy_logits")
    batch_size = logits.shape[0]
    if values.shape != (batch_size,) or not values.is_floating_point():
        raise ValueError("value output must be a rank-one floating tensor aligned to the batch")
    if value_targets.shape != (batch_size,) or not value_targets.is_floating_point():
        raise ValueError("value_targets must be a rank-one floating tensor aligned to the batch")
    if game_id_count != batch_size:
        raise ValueError("game_ids must align with the batch")
    tensors = (legal_mask, policy_targets, values, value_targets)
    if any(tensor.device != logits.device for tensor in tensors):
        raise ValueError("all training tensors must use the policy_logits device")
    if policy_targets.dtype != logits.dtype or values.dtype != logits.dtype:
        raise ValueError("model outputs and policy_targets must use one floating dtype")
    if value_targets.dtype != logits.dtype:
        raise ValueError("value_targets must use the model output dtype")
    if not torch.isfinite(logits).all().item():
        raise ValueError("policy_logits must contain only finite values")
    if not torch.isfinite(policy_targets).all().item():
        raise ValueError("policy_targets must contain only finite values")
    if (policy_targets < 0.0).any().item():
        raise ValueError("policy_targets must be non-negative")
    if not legal_mask.any(dim=-1).all().item():
        raise ValueError("each decision must have at least one legal candidate")
    if (policy_targets.masked_select(~legal_mask) != 0.0).any().item():
        raise ValueError("policy_targets for masked candidates must be zero")
    target_sums = policy_targets.sum(dim=-1)
    if not torch.allclose(target_sums, torch.ones_like(target_sums), rtol=1e-6, atol=1e-7):
        raise ValueError("policy_targets for each decision must sum to one")
    if not torch.isfinite(values).all().item() or ((values < -1.0) | (values > 1.0)).any().item():
        raise ValueError("value output must be finite and in the range [-1, 1]")
    if not torch.isfinite(value_targets).all().item():
        raise ValueError("value_targets must contain only finite values")
    if ((value_targets < -1.0) | (value_targets > 1.0)).any().item():
        raise ValueError("value_targets must be in the range [-1, 1]")


def clip_gradients(parameters: Iterable[nn.Parameter], max_norm: float = 1.0) -> Tensor:
    """Clip a parameter iterable to a finite global L2 gradient norm."""
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be finite and positive")
    return nn.utils.clip_grad_norm_(tuple(parameters), max_norm, error_if_nonfinite=True)


__all__ = [
    "JointLossConfig",
    "JointLossResult",
    "clip_gradients",
    "equal_game_weights",
    "joint_policy_value_loss",
]
