"""Behavioral contract for joint policy/value training objectives."""

from __future__ import annotations

import math

import pytest
import torch

from automata.models.shared_encoder.model import JointModelOutput
from automata.training.losses import (
    JointLossConfig,
    clip_gradients,
    equal_game_weights,
    joint_policy_value_loss,
)


def test_masked_losses_are_exact_and_each_game_has_equal_total_weight() -> None:
    logits = torch.tensor(
        [[0.0, math.log(3.0), 100.0], [0.0, 0.0, -100.0], [math.log(3.0), 0.0, 50.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    values = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float64, requires_grad=True)
    output = JointModelOutput(policy_logits=logits, value=values)
    legal_mask = torch.tensor([[True, True, False], [True, True, False], [True, True, False]])
    policy_targets = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    value_targets = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)

    result = joint_policy_value_loss(
        output,
        legal_mask=legal_mask,
        policy_targets=policy_targets,
        value_targets=value_targets,
        game_ids=("long", "long", "short"),
    )

    expected_policy = 0.25 * math.log(4.0) + 0.25 * math.log(2.0) + 0.5 * math.log(4 / 3)
    third_value_loss = -0.5 * (math.log(0.25) + math.log(0.75))
    expected_value = 0.25 * math.log(2.0) + 0.25 * math.log(4.0) + 0.5 * third_value_loss
    assert equal_game_weights(("long", "long", "short"), like=logits).tolist() == [
        0.25,
        0.25,
        0.5,
    ]
    assert result.policy.item() == pytest.approx(expected_policy)
    assert result.value.item() == pytest.approx(expected_value)
    assert result.total.item() == pytest.approx(expected_policy + expected_value)


def test_entropy_and_l2_are_declared_components_of_the_total() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
    output = JointModelOutput(
        policy_logits=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        value=torch.tensor([0.0], dtype=torch.float64),
    )

    result = joint_policy_value_loss(
        output,
        legal_mask=torch.tensor([[True, True]]),
        policy_targets=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        value_targets=torch.tensor([0.0], dtype=torch.float64),
        game_ids=("game",),
        parameters=(parameter,),
        config=JointLossConfig(value_weight=2.0, entropy_weight=0.5, l2_weight=0.1),
    )

    assert result.policy.item() == pytest.approx(math.log(2.0))
    assert result.value.item() == pytest.approx(math.log(2.0))
    assert result.entropy.item() == pytest.approx(math.log(2.0))
    assert result.l2.item() == pytest.approx(25.0)
    assert result.total.item() == pytest.approx(2.5 + 2.5 * math.log(2.0))


def test_gradients_are_exact_for_policy_and_bounded_value_probability() -> None:
    logits = torch.tensor([[0.0, 0.0, 99.0]], dtype=torch.float64, requires_grad=True)
    value = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

    result = joint_policy_value_loss(
        JointModelOutput(policy_logits=logits, value=value),
        legal_mask=torch.tensor([[True, True, False]]),
        policy_targets=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        value_targets=torch.tensor([1.0], dtype=torch.float64),
        game_ids=("game",),
    )
    result.total.backward()

    assert logits.grad is not None
    assert logits.grad[0].tolist() == pytest.approx([-0.5, 0.5, 0.0])
    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([-1.0])


def test_value_bce_clamps_bounded_probabilities_at_declared_epsilon() -> None:
    epsilon = 1e-4
    result = joint_policy_value_loss(
        JointModelOutput(
            policy_logits=torch.tensor([[0.0], [0.0]], dtype=torch.float64),
            value=torch.tensor([1.0, -1.0], dtype=torch.float64),
        ),
        legal_mask=torch.tensor([[True], [True]]),
        policy_targets=torch.tensor([[1.0], [1.0]], dtype=torch.float64),
        value_targets=torch.tensor([-1.0, 1.0], dtype=torch.float64),
        game_ids=("game", "game"),
        config=JointLossConfig(value_probability_epsilon=epsilon),
    )

    assert torch.isfinite(result.value)
    assert result.value.item() == pytest.approx(-math.log(epsilon))


def test_gradient_clipping_uses_a_finite_global_l2_norm() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])

    original_norm = clip_gradients((parameter,), max_norm=2.0)

    assert original_norm.item() == pytest.approx(5.0)
    assert parameter.grad.norm().item() == pytest.approx(2.0)
    parameter.grad[0] = torch.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_gradients((parameter,))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("nan-logit", "policy_logits"),
        ("masked-target", "masked candidates"),
        ("target-sum", "sum to one"),
        ("no-legal-candidate", "legal candidate"),
        ("bad-value", "range"),
        ("nan-value-target", "value_targets"),
    ],
)
def test_loss_rejects_non_finite_or_invalid_training_tensors(change: str, message: str) -> None:
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    legal_mask = torch.tensor([[True, True]])
    policy_targets = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    value = torch.tensor([0.0], dtype=torch.float64)
    value_targets = torch.tensor([1.0], dtype=torch.float64)
    if change == "nan-logit":
        logits[0, 0] = torch.nan
    elif change == "masked-target":
        legal_mask[0, 1] = False
        policy_targets = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    elif change == "target-sum":
        policy_targets = torch.tensor([[0.4, 0.4]], dtype=torch.float64)
    elif change == "no-legal-candidate":
        legal_mask[:] = False
    elif change == "bad-value":
        value[0] = 1.01
    else:
        value_targets[0] = torch.nan

    with pytest.raises(ValueError, match=message):
        joint_policy_value_loss(
            JointModelOutput(policy_logits=logits, value=value),
            legal_mask=legal_mask,
            policy_targets=policy_targets,
            value_targets=value_targets,
            game_ids=("game",),
        )


@pytest.mark.parametrize(
    "config",
    [
        JointLossConfig(value_weight=-1.0),
        JointLossConfig(entropy_weight=math.inf),
        JointLossConfig(l2_weight=-0.1),
        JointLossConfig(value_probability_epsilon=0.5),
    ],
)
def test_invalid_loss_configuration_is_rejected(config: JointLossConfig) -> None:
    with pytest.raises(ValueError):
        joint_policy_value_loss(
            JointModelOutput(policy_logits=torch.tensor([[0.0]]), value=torch.tensor([0.0])),
            legal_mask=torch.tensor([[True]]),
            policy_targets=torch.tensor([[1.0]]),
            value_targets=torch.tensor([1.0]),
            game_ids=("game",),
            config=config,
        )
