"""Behavioral tests for joint policy/value metric primitives."""

from __future__ import annotations

import json
import math

import pytest

from automata.decision import DecisionSemanticRole
from automata.training.metrics import (
    JointMetricsAccumulator,
    PolicyMetricInput,
    ValueMetricInput,
    joint_metrics,
    policy_metrics,
    value_metrics,
)
from goa2.domain.input import InputRequestType


def _assert_metrics_equal(actual: object, expected: object) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_metrics_equal(actual[key], value)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def test_policy_metrics_are_game_equal_and_grouped_by_candidate_family() -> None:
    examples = [
        PolicyMetricInput(
            game_id="long",
            candidate_family="CARD",
            target_probabilities=(1.0, 0.0),
            predicted_logits=(math.log(0.8), math.log(0.2)),
        ),
        PolicyMetricInput(
            game_id="long",
            candidate_family="CARD",
            target_probabilities=(1.0, 0.0),
            predicted_logits=(math.log(0.8), math.log(0.2)),
        ),
        PolicyMetricInput(
            game_id="short",
            candidate_family="HEX",
            target_probabilities=(1.0, 0.0),
            predicted_logits=(math.log(0.4), math.log(0.6)),
        ),
    ]

    result = policy_metrics(examples, top_k=2)

    assert result["overall"] == pytest.approx(
        {
            "count": 3,
            "game_count": 2,
            "multi_candidate_count": 3,
            "pairwise_count": 3,
            "pairwise_game_count": 2,
            "cross_entropy": (-math.log(0.8) - math.log(0.4)) / 2,
            "top1_accuracy": 0.5,
            "top2_recall": 1.0,
            "pairwise_accuracy": 0.5,
            "entropy": (
                -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
                - (0.4 * math.log(0.4) + 0.6 * math.log(0.6))
            )
            / 2,
            "search_overturn_rate": None,
            "q_variance": None,
        }
    )
    assert list(result["by_candidate_family"]) == ["CARD", "HEX"]
    assert result["by_candidate_family"]["CARD"]["top1_accuracy"] == 1.0
    assert result["by_candidate_family"]["HEX"]["top1_accuracy"] == 0.0
    json.dumps(result, allow_nan=False, sort_keys=True)


def test_joint_metrics_have_explicit_json_safe_empty_behavior() -> None:
    result = joint_metrics([], [], top_k=1)

    assert result["policy"]["overall"] == {
        "count": 0,
        "game_count": 0,
        "multi_candidate_count": 0,
        "pairwise_count": 0,
        "pairwise_game_count": 0,
        "cross_entropy": None,
        "top1_accuracy": None,
        "top1_recall": None,
        "pairwise_accuracy": None,
        "entropy": None,
        "search_overturn_rate": None,
        "q_variance": None,
    }
    assert result["value"]["overall"] == {
        "count": 0,
        "game_count": 0,
        "log_loss": None,
        "brier_score": None,
        "expected_calibration_error": None,
        "accuracy": None,
        "saturation_rate": None,
    }
    for head in ("policy", "value"):
        for dimension in (
            "by_candidate_family",
            "by_hero",
            "by_map",
            "by_composition",
            "by_round",
        ):
            assert result[head][dimension] == {}
    assert all(
        bucket["count"] == 0
        for dimension in (
            "by_input_request_type",
            "by_semantic_role",
            "by_can_skip",
        )
        for bucket in result["policy"][dimension].values()
    )
    assert json.loads(json.dumps(result, allow_nan=False, sort_keys=True)) == result


def test_policy_metrics_emit_complete_typed_decision_buckets_deterministically() -> None:
    examples = [
        PolicyMetricInput(
            game_id="planning",
            candidate_family="CARD",
            target_probabilities=(1.0, 0.0),
            predicted_logits=(1.0, 0.0),
            input_request_type=None,
            semantic_role=DecisionSemanticRole.PLANNING,
            can_skip=False,
        ),
        PolicyMetricInput(
            game_id="reaction",
            candidate_family="CARD+SKIP",
            target_probabilities=(0.0, 1.0),
            predicted_logits=(0.0, 1.0),
            input_request_type=InputRequestType.SELECT_CARD_OR_PASS.value,
            semantic_role=DecisionSemanticRole.DEFENSE_REACTION,
            can_skip=True,
        ),
    ]

    result = policy_metrics(examples)

    assert tuple(result["by_input_request_type"]) == (
        "null",
        *sorted(item.value for item in InputRequestType),
    )
    assert tuple(result["by_semantic_role"]) == tuple(
        sorted(item.value for item in DecisionSemanticRole)
    )
    assert tuple(result["by_can_skip"]) == ("false", "true")
    assert result["by_input_request_type"]["null"]["count"] == 1
    assert result["by_semantic_role"]["PLANNING"]["count"] == 1
    assert result["by_can_skip"]["true"]["count"] == 1
    empty = result["by_semantic_role"][DecisionSemanticRole.ATTACK_TARGET.value]
    assert empty["count"] == 0
    assert empty["cross_entropy"] is None
    assert empty["top1_accuracy"] is None
    assert empty["pairwise_accuracy"] is None
    assert json.dumps(result, allow_nan=False, separators=(",", ":")) == json.dumps(
        policy_metrics(reversed(examples)), allow_nan=False, separators=(",", ":")
    )


def test_pairwise_metrics_exclude_singleton_and_all_tied_targets() -> None:
    examples = [
        PolicyMetricInput(
            game_id="singleton",
            candidate_family="CARD",
            target_probabilities=(1.0,),
            predicted_logits=(0.0,),
        ),
        PolicyMetricInput(
            game_id="tied",
            candidate_family="OPTION",
            target_probabilities=(0.5, 0.5),
            predicted_logits=(1.0, -1.0),
        ),
        PolicyMetricInput(
            game_id="ranked",
            candidate_family="OPTION",
            target_probabilities=(0.75, 0.25),
            predicted_logits=(1.0, 0.0),
        ),
    ]

    result = policy_metrics(examples)

    assert result["overall"]["count"] == 3
    assert result["overall"]["multi_candidate_count"] == 2
    assert result["overall"]["pairwise_count"] == 1
    assert result["overall"]["pairwise_game_count"] == 1
    assert result["overall"]["pairwise_accuracy"] == 1.0
    assert result["by_candidate_family"]["CARD"]["pairwise_accuracy"] is None
    assert result["by_candidate_family"]["OPTION"]["pairwise_count"] == 1


def test_policy_metric_context_rejects_partial_or_incoherent_metadata() -> None:
    common = {
        "game_id": "g1",
        "candidate_family": "OPTION",
        "target_probabilities": (1.0, 0.0),
        "predicted_logits": (1.0, 0.0),
    }

    with pytest.raises(ValueError, match="supplied together"):
        PolicyMetricInput(**common, input_request_type="SELECT_OPTION")
    with pytest.raises(ValueError, match="null request and no skip"):
        PolicyMetricInput(
            **common,
            input_request_type="SELECT_OPTION",
            semantic_role=DecisionSemanticRole.PLANNING,
            can_skip=False,
        )
    with pytest.raises(ValueError, match="requires an input request"):
        PolicyMetricInput(
            **common,
            semantic_role=DecisionSemanticRole.OPTION_SELECTION,
            can_skip=False,
        )


def test_policy_metrics_include_optional_search_diagnostics_and_supplied_buckets() -> None:
    example = PolicyMetricInput(
        game_id="g1",
        candidate_family="UNIT",
        target_probabilities=(0.1, 0.9),
        predicted_logits=(0.0, 1.0),
        prior_probabilities=(0.8, 0.2),
        q_variances=(0.2, 0.6),
        hero="Arien",
        map_id="island",
        composition="Arien+Wasp|Bain+Misa",
        round_bucket="4-6",
    )

    result = policy_metrics([example])

    assert result["overall"]["search_overturn_rate"] == 1.0
    assert result["overall"]["q_variance"] == pytest.approx(0.4)
    assert list(result["by_hero"]) == ["Arien"]
    assert list(result["by_map"]) == ["island"]
    assert list(result["by_composition"]) == ["Arien+Wasp|Bain+Misa"]
    assert list(result["by_round"]) == ["4-6"]


def test_joint_metrics_accumulator_matches_batch_metrics_incrementally() -> None:
    policies = [
        PolicyMetricInput(
            game_id="long",
            candidate_family="CARD",
            target_probabilities=(0.8, 0.2),
            predicted_logits=(1.0, -0.5),
            prior_probabilities=(0.1, 0.9),
            q_variances=(0.2, 0.6),
            hero="Arien",
            map_id="island",
            composition="Arien+Wasp|Bain+Misa",
            round_bucket="1-3",
            input_request_type=None,
            semantic_role=DecisionSemanticRole.PLANNING,
            can_skip=False,
        ),
        PolicyMetricInput(
            game_id="short",
            candidate_family="HEX",
            target_probabilities=(0.0, 0.5, 0.5),
            predicted_logits=(-1.0, 0.5, 0.5),
            hero="Wasp",
            map_id="island",
            composition="Arien+Wasp|Bain+Misa",
            round_bucket="4-6",
            input_request_type=InputRequestType.SELECT_HEX.value,
            semantic_role=DecisionSemanticRole.SPATIAL_SELECTION,
            can_skip=False,
        ),
        PolicyMetricInput(
            game_id="long",
            candidate_family="CARD",
            target_probabilities=(0.3, 0.7),
            predicted_logits=(0.4, 0.6),
            prior_probabilities=(0.3, 0.7),
            hero="Arien",
            map_id="cove",
            input_request_type=InputRequestType.SELECT_CARD.value,
            semantic_role=DecisionSemanticRole.CARD_SELECTION,
            can_skip=False,
        ),
    ]
    values = [
        ValueMetricInput(
            game_id="short",
            target_value=-1,
            predicted_value=0.8,
            candidate_family="HEX",
            hero="Wasp",
            map_id="island",
            composition="Arien+Wasp|Bain+Misa",
            round_bucket="4-6",
        ),
        ValueMetricInput(
            game_id="long",
            target_value=1,
            predicted_value=0.8,
            candidate_family="CARD",
            hero="Arien",
            map_id="island",
            composition="Arien+Wasp|Bain+Misa",
            round_bucket="1-3",
        ),
        ValueMetricInput(
            game_id="long",
            target_value=0,
            predicted_value=0.0,
            candidate_family="CARD",
            hero="Arien",
            map_id="cove",
        ),
    ]
    accumulator = JointMetricsAccumulator(top_k=2, ece_bins=5, saturation_threshold=0.75)

    # Deliberately add heads and games in different orders; no complete collection is supplied.
    accumulator.add_policy(policies[0])
    accumulator.add_value(values[0])
    accumulator.add_policy(policies[1])
    accumulator.add_value(values[1])
    accumulator.add_value(values[2])
    accumulator.add_policy(policies[2])

    expected = joint_metrics(
        policies,
        values,
        top_k=2,
        ece_bins=5,
        saturation_threshold=0.75,
    )
    _assert_metrics_equal(accumulator.compute(), expected)
    _assert_metrics_equal(accumulator.compute()["policy"], policy_metrics(policies, top_k=2))
    assert accumulator.compute()["policy"]["by_semantic_role"]["PLANNING"]["count"] == 1
    assert accumulator.compute()["policy"]["by_input_request_type"]["SELECT_HEX"]["count"] == 1
    _assert_metrics_equal(
        accumulator.compute()["value"],
        value_metrics(values, ece_bins=5, saturation_threshold=0.75),
    )


def test_joint_metrics_accumulator_matches_empty_batch_then_accepts_examples() -> None:
    accumulator = JointMetricsAccumulator(top_k=1)

    assert accumulator.compute() == joint_metrics([], [], top_k=1)

    policy = PolicyMetricInput(
        game_id="g1",
        candidate_family="UNIT",
        target_probabilities=(1.0, 0.0),
        predicted_logits=(2.0, 0.0),
    )
    value = ValueMetricInput(game_id="g1", target_value=1, predicted_value=0.5)
    accumulator.add_policy(policy)
    accumulator.add_value(value)

    _assert_metrics_equal(accumulator.compute(), joint_metrics([policy], [value], top_k=1))


def test_value_metrics_are_game_equal_calibrated_and_stratified() -> None:
    examples = [
        ValueMetricInput(
            game_id="long",
            target_value=1,
            predicted_value=0.8,
            candidate_family="CARD",
            hero="Wasp",
            map_id="island",
            composition="Wasp|Arien",
            round_bucket="1-3",
        ),
        ValueMetricInput(
            game_id="long",
            target_value=1,
            predicted_value=0.8,
            candidate_family="CARD",
            hero="Wasp",
            map_id="island",
            composition="Wasp|Arien",
            round_bucket="1-3",
        ),
        ValueMetricInput(
            game_id="short",
            target_value=-1,
            predicted_value=0.8,
            candidate_family="HEX",
        ),
    ]

    result = value_metrics(examples, ece_bins=5, saturation_threshold=0.75)

    assert result["overall"] == pytest.approx(
        {
            "count": 3,
            "game_count": 2,
            "log_loss": (-math.log(0.9) - math.log(0.1)) / 2,
            "brier_score": (0.01 + 0.81) / 2,
            "expected_calibration_error": 0.4,
            "accuracy": 0.5,
            "saturation_rate": 1.0,
        }
    )
    assert list(result["by_candidate_family"]) == ["CARD", "HEX"]
    assert result["by_candidate_family"]["CARD"]["accuracy"] == 1.0
    assert list(result["by_hero"]) == ["Wasp"]
    assert list(result["by_map"]) == ["island"]
    assert list(result["by_composition"]) == ["Wasp|Arien"]
    assert list(result["by_round"]) == ["1-3"]
    json.dumps(result, allow_nan=False, sort_keys=True)
