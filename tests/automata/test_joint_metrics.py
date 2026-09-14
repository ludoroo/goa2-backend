"""Behavioral tests for joint policy/value metric primitives."""

from __future__ import annotations

import json
import math

import pytest

from automata.training.metrics import (
    PolicyMetricInput,
    ValueMetricInput,
    joint_metrics,
    policy_metrics,
    value_metrics,
)


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
    assert json.loads(json.dumps(result, allow_nan=False, sort_keys=True)) == result


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
