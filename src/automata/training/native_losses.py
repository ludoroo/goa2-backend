"""Independent policy and stable-value losses for native training rows.

Callers provide full-game row weights and a head-specific normalizer.  This
keeps each head additive when a logical batch is split into arbitrary chunks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from automata.models.shared_encoder.gen1_model import PolicyHeadOutput, StableValueHeadOutput


@dataclass(frozen=True, slots=True)
class NativePolicyLossResult:
    """Differentiable scalar components of the native policy objective."""

    total: Tensor
    cross_entropy: Tensor
    entropy: Tensor


@dataclass(frozen=True, slots=True)
class NativeValueLossResult:
    """Differentiable scalar components of the native stable-value objective."""

    total: Tensor
    bce: Tensor


def native_policy_loss(
    output: PolicyHeadOutput,
    *,
    legal_mask: Tensor,
    policy_targets: Tensor,
    row_weights: Tensor,
    head_normalizer: float = 1.0,
    row_mask: Tensor | None = None,
    entropy_weight: float = 0.0,
) -> NativePolicyLossResult:
    """Compute additive weighted policy CE and entropy over active rows."""
    _validate_positive_finite("head_normalizer", head_normalizer)
    _validate_nonnegative_finite("entropy_weight", entropy_weight)

    logits = output.policy_logits
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("policy_logits must be a rank-two floating tensor")
    if legal_mask.dtype != torch.bool or legal_mask.shape != logits.shape:
        raise ValueError("legal_mask must be boolean and match policy_logits")
    if policy_targets.shape != logits.shape or not policy_targets.is_floating_point():
        raise ValueError("policy_targets must be floating and match policy_logits")

    batch_size = logits.shape[0]
    _validate_row_weights(row_weights, batch_size=batch_size, like=logits)
    active_mask = _validate_row_mask(row_mask, batch_size=batch_size, like=logits)
    _validate_devices(
        like=logits,
        tensors=(legal_mask, policy_targets, row_weights, active_mask),
        label="policy tensors and masks",
    )
    if policy_targets.dtype != logits.dtype or row_weights.dtype != logits.dtype:
        raise ValueError(
            "policy_logits, policy_targets, and row_weights must use one floating dtype"
        )

    active_logits = logits[active_mask]
    if active_logits.shape[0] == 0:
        zero = active_logits.sum()
        return NativePolicyLossResult(total=zero, cross_entropy=zero, entropy=zero)

    active_legal = legal_mask[active_mask]
    active_targets = policy_targets[active_mask]
    active_weights = row_weights[active_mask]
    if not torch.isfinite(active_logits.masked_select(active_legal)).all().item():
        raise ValueError("policy_logits for legal candidates must contain only finite values")
    if not torch.isfinite(active_targets).all().item():
        raise ValueError("policy_targets for active rows must contain only finite values")
    if (active_targets < 0.0).any().item():
        raise ValueError("policy_targets for active rows must be non-negative")
    if not active_legal.any(dim=-1).all().item():
        raise ValueError("each active policy row must have at least one legal candidate")
    if (active_targets.masked_select(~active_legal) != 0.0).any().item():
        raise ValueError("policy_targets for illegal candidates must be zero")
    target_sums = active_targets.sum(dim=-1)
    if not torch.allclose(target_sums, torch.ones_like(target_sums), rtol=1e-6, atol=1e-7):
        raise ValueError("policy_targets for each active row must sum to one")

    masked_logits = active_logits.masked_fill(~active_legal, -torch.inf)
    log_probabilities = torch.log_softmax(masked_logits, dim=-1)
    finite_log_probabilities = log_probabilities.masked_fill(~active_legal, 0.0)
    cross_entropy_per_row = -(active_targets * finite_log_probabilities).sum(dim=-1)
    probabilities = log_probabilities.exp()
    entropy_per_row = -(probabilities * finite_log_probabilities).sum(dim=-1)

    cross_entropy = (active_weights * cross_entropy_per_row).sum() / head_normalizer
    entropy = (active_weights * entropy_per_row).sum() / head_normalizer
    total = cross_entropy - entropy_weight * entropy
    if not all(torch.isfinite(component).item() for component in (cross_entropy, entropy, total)):
        raise ValueError("native policy loss must be finite")
    return NativePolicyLossResult(
        total=total,
        cross_entropy=cross_entropy,
        entropy=entropy,
    )


def native_stable_value_loss(
    output: StableValueHeadOutput,
    *,
    value_targets: Tensor,
    row_weights: Tensor,
    head_normalizer: float = 1.0,
    row_mask: Tensor | None = None,
    probability_epsilon: float = 1e-7,
) -> NativeValueLossResult:
    """Compute additive BCE for candidate-free bounded stable-value rows."""
    _validate_positive_finite("head_normalizer", head_normalizer)
    if not _is_finite_real(probability_epsilon) or not 0.0 < probability_epsilon < 0.5:
        raise ValueError("probability_epsilon must be finite and between zero and 0.5")

    values = output.value
    if values.ndim != 1 or not values.is_floating_point():
        raise ValueError("value output must be a rank-one floating tensor")
    batch_size = values.shape[0]
    if value_targets.shape != (batch_size,) or not value_targets.is_floating_point():
        raise ValueError("value_targets must be a rank-one floating tensor aligned to the batch")
    _validate_row_weights(row_weights, batch_size=batch_size, like=values)
    active_mask = _validate_row_mask(row_mask, batch_size=batch_size, like=values)
    _validate_devices(
        like=values,
        tensors=(value_targets, row_weights, active_mask),
        label="stable-value tensors and masks",
    )
    if value_targets.dtype != values.dtype or row_weights.dtype != values.dtype:
        raise ValueError("value output, value_targets, and row_weights must use one floating dtype")

    active_values = values[active_mask]
    if active_values.shape[0] == 0:
        zero = active_values.sum()
        return NativeValueLossResult(total=zero, bce=zero)

    active_targets = value_targets[active_mask]
    active_weights = row_weights[active_mask]
    if (
        not torch.isfinite(active_values).all().item()
        or ((active_values < -1.0) | (active_values > 1.0)).any().item()
    ):
        raise ValueError("value output for active rows must be finite and in the range [-1, 1]")
    if not torch.isfinite(active_targets).all().item():
        raise ValueError("value_targets for active rows must contain only finite values")
    if ((active_targets < -1.0) | (active_targets > 1.0)).any().item():
        raise ValueError("value_targets for active rows must be in the range [-1, 1]")

    probabilities = ((active_values + 1.0) / 2.0).clamp(
        probability_epsilon, 1.0 - probability_epsilon
    )
    binary_targets = (active_targets + 1.0) / 2.0
    bce_per_row = -(
        binary_targets * probabilities.log() + (1.0 - binary_targets) * (1.0 - probabilities).log()
    )
    bce = (active_weights * bce_per_row).sum() / head_normalizer
    if not torch.isfinite(bce).item():
        raise ValueError("native stable-value loss must be finite")
    return NativeValueLossResult(total=bce, bce=bce)


def _validate_row_weights(row_weights: Tensor, *, batch_size: int, like: Tensor) -> None:
    if row_weights.shape != (batch_size,) or not row_weights.is_floating_point():
        raise ValueError("row_weights must be a rank-one floating tensor aligned to the batch")
    if row_weights.device != like.device:
        raise ValueError("row_weights must use the model output device")
    if not torch.isfinite(row_weights).all().item() or (row_weights < 0.0).any().item():
        raise ValueError("row_weights must contain only finite non-negative values")


def _validate_row_mask(
    row_mask: Tensor | None,
    *,
    batch_size: int,
    like: Tensor,
) -> Tensor:
    if row_mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device=like.device)
    if row_mask.dtype != torch.bool or row_mask.shape != (batch_size,):
        raise ValueError("row_mask must be boolean and align with the batch")
    if row_mask.device != like.device:
        raise ValueError("row_mask must use the model output device")
    return row_mask


def _validate_devices(*, like: Tensor, tensors: tuple[Tensor, ...], label: str) -> None:
    if any(tensor.device != like.device for tensor in tensors):
        raise ValueError(f"{label} must use the model output device")


def _is_finite_real(value: float) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def _validate_positive_finite(name: str, value: float) -> None:
    if not _is_finite_real(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_nonnegative_finite(name: str, value: float) -> None:
    if not _is_finite_real(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "NativePolicyLossResult",
    "NativeValueLossResult",
    "native_policy_loss",
    "native_stable_value_loss",
]
