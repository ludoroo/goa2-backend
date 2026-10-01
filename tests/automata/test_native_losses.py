"""Behavioral contract for native, independently normalized head losses."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from automata.models.shared_encoder.gen1_model import PolicyHeadOutput, StableValueHeadOutput
from automata.training.native_losses import (
    NativePolicyLossResult,
    NativeValueLossResult,
    native_policy_loss,
    native_stable_value_loss,
)


def _policy_loss(
    logits: Tensor,
    *,
    targets: Tensor | None = None,
    legal_mask: Tensor | None = None,
    row_weights: Tensor | None = None,
    row_mask: Tensor | None = None,
    head_normalizer: float = 1.0,
    entropy_weight: float = 0.0,
) -> NativePolicyLossResult:
    rows = logits.shape[0] if logits.ndim > 0 else 1
    candidates = logits.shape[1] if logits.ndim == 2 else 1
    return native_policy_loss(
        PolicyHeadOutput(policy_logits=logits),
        legal_mask=(
            torch.ones((rows, candidates), dtype=torch.bool, device=logits.device)
            if legal_mask is None
            else legal_mask
        ),
        policy_targets=(
            torch.full(
                (rows, candidates),
                1.0 / candidates,
                dtype=logits.dtype,
                device=logits.device,
            )
            if targets is None
            else targets
        ),
        row_weights=(
            torch.ones(rows, dtype=logits.dtype, device=logits.device)
            if row_weights is None
            else row_weights
        ),
        row_mask=row_mask,
        head_normalizer=head_normalizer,
        entropy_weight=entropy_weight,
    )


def _value_loss(
    values: Tensor,
    *,
    targets: Tensor | None = None,
    row_weights: Tensor | None = None,
    row_mask: Tensor | None = None,
    head_normalizer: float = 1.0,
    probability_epsilon: float = 1e-7,
) -> NativeValueLossResult:
    return native_stable_value_loss(
        StableValueHeadOutput(value=values),
        value_targets=torch.zeros_like(values) if targets is None else targets,
        row_weights=torch.ones_like(values) if row_weights is None else row_weights,
        row_mask=row_mask,
        head_normalizer=head_normalizer,
        probability_epsilon=probability_epsilon,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_policy_ce_entropy_and_gradients_are_separate_and_exact(dtype: torch.dtype) -> None:
    logits = torch.tensor(
        [[0.0, math.log(3.0), float("nan")], [0.0, 0.0, -100.0]],
        dtype=dtype,
        requires_grad=True,
    )
    result = _policy_loss(
        logits,
        legal_mask=torch.tensor([[True, True, False], [True, True, False]]),
        targets=torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=dtype),
        row_weights=torch.tensor([0.5, 1.5], dtype=dtype),
        head_normalizer=2.0,
        entropy_weight=0.25,
    )

    first_entropy = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    expected_ce = (0.5 * math.log(4.0) + 1.5 * math.log(2.0)) / 2.0
    expected_entropy = (0.5 * first_entropy + 1.5 * math.log(2.0)) / 2.0
    assert result.cross_entropy.item() == pytest.approx(expected_ce)
    assert result.entropy.item() == pytest.approx(expected_entropy)
    assert result.total.item() == pytest.approx(expected_ce - 0.25 * expected_entropy)

    result.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 2].tolist() == [0.0, 0.0]


def test_stable_value_uses_bounded_probability_bce_and_has_finite_gradients() -> None:
    values = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float64, requires_grad=True)
    result = _value_loss(
        values,
        targets=torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64),
        row_weights=torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64),
        head_normalizer=0.5,
    )

    third_bce = -0.5 * (math.log(0.25) + math.log(0.75))
    expected = (0.5 * math.log(2.0) + 0.25 * math.log(4.0) + 0.25 * third_bce) / 0.5
    assert result.bce.item() == pytest.approx(expected)
    assert result.total.item() == pytest.approx(expected)

    result.total.backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert values.grad.tolist() == pytest.approx([-1.0, 1.0, -1.0 / 3.0])


def test_value_probability_clamp_is_finite_at_bounded_extremes() -> None:
    epsilon = 1e-4
    result = _value_loss(
        torch.tensor([1.0, -1.0], dtype=torch.float64),
        targets=torch.tensor([-1.0, 1.0], dtype=torch.float64),
        row_weights=torch.tensor([0.5, 0.5], dtype=torch.float64),
        probability_epsilon=epsilon,
    )

    assert torch.isfinite(result.bce)
    assert result.bce.item() == pytest.approx(-math.log(epsilon))


def test_row_masks_ignore_poisoned_inactive_rows_without_nan_leakage() -> None:
    policy_logits = torch.tensor(
        [[0.0, 0.0], [float("nan"), float("nan")]],
        dtype=torch.float64,
        requires_grad=True,
    )
    policy = _policy_loss(
        policy_logits,
        legal_mask=torch.tensor([[True, True], [False, False]]),
        targets=torch.tensor([[1.0, 0.0], [float("nan"), float("nan")]], dtype=torch.float64),
        row_weights=torch.tensor([1.0, 9.0], dtype=torch.float64),
        row_mask=torch.tensor([True, False]),
    )
    values = torch.tensor([0.0, float("nan")], dtype=torch.float64, requires_grad=True)
    value = _value_loss(
        values,
        targets=torch.tensor([1.0, float("nan")], dtype=torch.float64),
        row_weights=torch.tensor([1.0, 9.0], dtype=torch.float64),
        row_mask=torch.tensor([True, False]),
    )

    assert policy.cross_entropy.item() == pytest.approx(math.log(2.0))
    assert value.bce.item() == pytest.approx(math.log(2.0))
    (policy.total + value.total).backward()
    assert policy_logits.grad is not None
    assert policy_logits.grad[1].tolist() == [0.0, 0.0]
    assert values.grad is not None
    assert values.grad.tolist() == pytest.approx([-1.0, 0.0])


def test_policy_and_value_heads_use_independent_row_masks() -> None:
    policy = _policy_loss(
        torch.tensor([[0.0, 0.0], [float("nan"), float("nan")]], dtype=torch.float64),
        legal_mask=torch.tensor([[True, True], [False, False]]),
        targets=torch.tensor([[1.0, 0.0], [float("nan"), float("nan")]], dtype=torch.float64),
        row_weights=torch.tensor([1.0, 1.0], dtype=torch.float64),
        row_mask=torch.tensor([True, False]),
    )
    value = _value_loss(
        torch.tensor([float("nan"), 0.0], dtype=torch.float64),
        targets=torch.tensor([float("nan"), -1.0], dtype=torch.float64),
        row_weights=torch.tensor([1.0, 1.0], dtype=torch.float64),
        row_mask=torch.tensor([False, True]),
    )

    assert policy.cross_entropy.item() == pytest.approx(math.log(2.0))
    assert value.bce.item() == pytest.approx(math.log(2.0))
    assert set(policy.__dataclass_fields__) == {"total", "cross_entropy", "entropy"}
    assert set(value.__dataclass_fields__) == {"total", "bce"}


def test_empty_heads_return_finite_differentiable_zero_without_reading_poison() -> None:
    logits = torch.full((2, 3), float("nan"), dtype=torch.float64, requires_grad=True)
    policy = _policy_loss(
        logits,
        legal_mask=torch.zeros((2, 3), dtype=torch.bool),
        targets=torch.full((2, 3), float("nan"), dtype=torch.float64),
        row_weights=torch.tensor([1.0, 1.0], dtype=torch.float64),
        row_mask=torch.tensor([False, False]),
        entropy_weight=0.5,
    )
    values = torch.full((2,), float("nan"), dtype=torch.float64, requires_grad=True)
    value = _value_loss(
        values,
        targets=torch.full((2,), float("nan"), dtype=torch.float64),
        row_weights=torch.tensor([1.0, 1.0], dtype=torch.float64),
        row_mask=torch.tensor([False, False]),
    )

    for component in (policy.total, policy.cross_entropy, policy.entropy, value.total, value.bce):
        assert component.item() == 0.0
        assert component.requires_grad
    (policy.total + value.total).backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))
    assert values.grad is not None
    assert torch.equal(values.grad, torch.zeros_like(values.grad))


def test_full_game_weights_make_policy_loss_additive_across_arbitrary_chunks() -> None:
    logits = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.5, -0.5], [-0.5, 0.5], [0.0, 0.0], [2.0, 0.0]],
        dtype=torch.float64,
    )
    targets = torch.tensor([[1.0, 0.0]] * 6, dtype=torch.float64)
    # Four rows from one game and two from another: each game has unit head mass.
    weights = torch.tensor([0.25] * 4 + [0.5] * 2, dtype=torch.float64)
    whole = _policy_loss(
        logits,
        targets=targets,
        row_weights=weights,
        head_normalizer=2.0,
        entropy_weight=0.1,
    )
    chunks = [
        _policy_loss(
            logits[indices],
            targets=targets[indices],
            row_weights=weights[indices],
            head_normalizer=2.0,
            entropy_weight=0.1,
        )
        for indices in (slice(0, 1), slice(1, 5), slice(5, 6))
    ]

    assert whole.cross_entropy == pytest.approx(sum(item.cross_entropy for item in chunks))
    assert whole.entropy == pytest.approx(sum(item.entropy for item in chunks))
    assert whole.total == pytest.approx(sum(item.total for item in chunks))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_game_weights_make_value_loss_additive_across_arbitrary_chunks(
    dtype: torch.dtype,
) -> None:
    values = torch.tensor([-0.5, 0.0, 0.5, 0.25, -0.25], dtype=dtype)
    targets = torch.tensor([-1.0, 0.0, 1.0, 1.0, -1.0], dtype=dtype)
    # Three rows from one game and two from another: each game has unit head mass.
    weights = torch.tensor([1.0 / 3.0] * 3 + [0.5] * 2, dtype=dtype)
    whole = _value_loss(values, targets=targets, row_weights=weights, head_normalizer=2.0)
    chunks = [
        _value_loss(
            values[indices],
            targets=targets[indices],
            row_weights=weights[indices],
            head_normalizer=2.0,
        )
        for indices in (slice(0, 2), slice(2, 4), slice(4, 5))
    ]

    assert whole.bce == pytest.approx(sum(item.bce for item in chunks))
    assert whole.total == pytest.approx(sum(item.total for item in chunks))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("logit-rank", "policy_logits"),
        ("legal-shape", "legal_mask"),
        ("legal-dtype", "legal_mask"),
        ("target-shape", "policy_targets"),
        ("target-dtype", "floating dtype"),
        ("weight-shape", "row_weights"),
        ("weight-dtype", "floating dtype"),
        ("row-mask-shape", "row_mask"),
        ("row-mask-dtype", "row_mask"),
        ("nan-logit", "policy_logits"),
        ("no-legal", "legal candidate"),
        ("negative-target", "non-negative"),
        ("illegal-target", "illegal candidates"),
        ("target-sum", "sum to one"),
        ("nan-weight", "row_weights"),
        ("negative-weight", "row_weights"),
        ("zero-normalizer", "head_normalizer"),
        ("nan-normalizer", "head_normalizer"),
        ("negative-entropy", "entropy_weight"),
        ("infinite-entropy", "entropy_weight"),
    ],
)
def test_policy_validation(change: str, message: str) -> None:
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    legal_mask = torch.tensor([[True, True]])
    targets = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    weights = torch.tensor([1.0], dtype=torch.float64)
    row_mask = torch.tensor([True])
    normalizer = 1.0
    entropy_weight = 0.0
    if change == "logit-rank":
        logits = logits.squeeze(0)
    elif change == "legal-shape":
        legal_mask = torch.tensor([[True]])
    elif change == "legal-dtype":
        legal_mask = legal_mask.to(torch.float64)
    elif change == "target-shape":
        targets = torch.tensor([[1.0]], dtype=torch.float64)
    elif change == "target-dtype":
        targets = targets.to(torch.float32)
    elif change == "weight-shape":
        weights = weights.unsqueeze(1)
    elif change == "weight-dtype":
        weights = weights.to(torch.float32)
    elif change == "row-mask-shape":
        row_mask = row_mask.unsqueeze(1)
    elif change == "row-mask-dtype":
        row_mask = row_mask.to(torch.int64)
    elif change == "nan-logit":
        logits[0, 0] = torch.nan
    elif change == "no-legal":
        legal_mask[:] = False
    elif change == "negative-target":
        targets = torch.tensor([[1.1, -0.1]], dtype=torch.float64)
    elif change == "illegal-target":
        legal_mask[0, 1] = False
        targets = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    elif change == "target-sum":
        targets = torch.tensor([[0.4, 0.4]], dtype=torch.float64)
    elif change == "nan-weight":
        weights[0] = torch.nan
    elif change == "negative-weight":
        weights[0] = -1.0
    elif change == "zero-normalizer":
        normalizer = 0.0
    elif change == "nan-normalizer":
        normalizer = math.nan
    elif change == "negative-entropy":
        entropy_weight = -0.1
    else:
        entropy_weight = math.inf

    with pytest.raises(ValueError, match=message):
        _policy_loss(
            logits,
            legal_mask=legal_mask,
            targets=targets,
            row_weights=weights,
            row_mask=row_mask,
            head_normalizer=normalizer,
            entropy_weight=entropy_weight,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("value-rank", "value output"),
        ("target-shape", "value_targets"),
        ("target-dtype", "floating dtype"),
        ("weight-shape", "row_weights"),
        ("weight-dtype", "floating dtype"),
        ("row-mask-shape", "row_mask"),
        ("row-mask-dtype", "row_mask"),
        ("nan-value", "value output"),
        ("large-value", "value output"),
        ("nan-target", "value_targets"),
        ("large-target", "value_targets"),
        ("nan-weight", "row_weights"),
        ("negative-weight", "row_weights"),
        ("zero-normalizer", "head_normalizer"),
        ("infinite-normalizer", "head_normalizer"),
        ("zero-epsilon", "probability_epsilon"),
        ("large-epsilon", "probability_epsilon"),
        ("nan-epsilon", "probability_epsilon"),
    ],
)
def test_value_validation(change: str, message: str) -> None:
    values = torch.tensor([0.0], dtype=torch.float64)
    targets = torch.tensor([1.0], dtype=torch.float64)
    weights = torch.tensor([1.0], dtype=torch.float64)
    row_mask = torch.tensor([True])
    normalizer = 1.0
    epsilon = 1e-7
    if change == "value-rank":
        values = values.unsqueeze(1)
    elif change == "target-shape":
        targets = targets.unsqueeze(1)
    elif change == "target-dtype":
        targets = targets.to(torch.float32)
    elif change == "weight-shape":
        weights = weights.unsqueeze(1)
    elif change == "weight-dtype":
        weights = weights.to(torch.float32)
    elif change == "row-mask-shape":
        row_mask = row_mask.unsqueeze(1)
    elif change == "row-mask-dtype":
        row_mask = row_mask.to(torch.int64)
    elif change == "nan-value":
        values[0] = torch.nan
    elif change == "large-value":
        values[0] = 1.01
    elif change == "nan-target":
        targets[0] = torch.nan
    elif change == "large-target":
        targets[0] = -1.01
    elif change == "nan-weight":
        weights[0] = torch.nan
    elif change == "negative-weight":
        weights[0] = -1.0
    elif change == "zero-normalizer":
        normalizer = 0.0
    elif change == "infinite-normalizer":
        normalizer = math.inf
    elif change == "zero-epsilon":
        epsilon = 0.0
    elif change == "large-epsilon":
        epsilon = 0.5
    else:
        epsilon = math.nan

    with pytest.raises(ValueError, match=message):
        _value_loss(
            values,
            targets=targets,
            row_weights=weights,
            row_mask=row_mask,
            head_normalizer=normalizer,
            probability_epsilon=epsilon,
        )


def test_masks_and_value_inputs_must_share_the_output_device() -> None:
    logits = torch.tensor([[0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="device"):
        _policy_loss(logits, legal_mask=torch.ones((1, 1), dtype=torch.bool, device="meta"))

    values = torch.tensor([0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="device"):
        _value_loss(values, targets=torch.ones(1, dtype=torch.float64, device="meta"))
