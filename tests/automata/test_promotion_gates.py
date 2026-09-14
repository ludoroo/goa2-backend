"""Behavioral tests for the neutral promotion-gate evaluator."""

from dataclasses import replace
from typing import Any

import pytest

from automata.evaluation.promotion_gates import (
    ArtifactLoad,
    LatencySLO,
    PromotionGateConfig,
    PromotionMetrics,
    evaluate_promotion_gates,
)


def test_clean_evidence_passes_every_declared_gate() -> None:
    config = PromotionGateConfig(
        max_timeout_or_max_step_rate=0.01,
        latency_slos=(LatencySLO(tier="standard", percentile=95.0, max_ms=100.0),),
        practical_margin=0.02,
        required_strata=("map:a",),
        max_stratum_regression=0.01,
        required_artifact_loads=2,
        expected_artifact_digest="a" * 64,
    )
    metrics = PromotionMetrics(
        total_games=100,
        engine_errors=0,
        illegal_choices=0,
        agent_errors=0,
        timeout_terminations=0,
        max_step_terminations=1,
        candidate_score=0.54,
        champion_score=0.50,
        decision_latencies_ms={"standard": (10.0, 20.0, 30.0)},
        stratum_score_margins={"map:a": -0.005},
        artifact_loads=(
            ArtifactLoad(digest="a" * 64, error=None),
            ArtifactLoad(digest="a" * 64, error=None),
        ),
    )

    verdict = evaluate_promotion_gates(metrics, config)

    assert verdict.promoted is True
    assert verdict.failures == ()
    assert verdict.checks
    assert all(check.passed for check in verdict.checks)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_timeout_or_max_step_rate": -0.01},
        {"max_timeout_or_max_step_rate": 1.01},
        {"latency_slos": ()},
        {"latency_slos": (LatencySLO(tier="", percentile=95.0, max_ms=100.0),)},
        {"latency_slos": (LatencySLO(tier="fast", percentile=0.0, max_ms=100.0),)},
        {"latency_slos": (LatencySLO(tier="fast", percentile=101.0, max_ms=100.0),)},
        {"latency_slos": (LatencySLO(tier="fast", percentile=95.0, max_ms=0.0),)},
        {"practical_margin": -0.01},
        {"required_strata": ()},
        {"max_stratum_regression": -0.01},
        {"required_artifact_loads": 1},
    ],
)
def test_config_rejects_missing_or_invalid_policy(changes: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "max_timeout_or_max_step_rate": 0.01,
        "latency_slos": (LatencySLO(tier="standard", percentile=95.0, max_ms=100.0),),
        "practical_margin": 0.02,
        "required_strata": ("map:a",),
        "max_stratum_regression": 0.01,
        "required_artifact_loads": 2,
        "expected_artifact_digest": "a" * 64,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        PromotionGateConfig(**values)


def test_all_mandatory_failures_are_retained_without_short_circuiting() -> None:
    config = PromotionGateConfig(
        max_timeout_or_max_step_rate=0.02,
        latency_slos=(LatencySLO(tier="fast", percentile=95.0, max_ms=50.0),),
        practical_margin=0.03,
        required_strata=("map:a", "composition:b"),
        max_stratum_regression=0.01,
        required_artifact_loads=3,
        expected_artifact_digest="a" * 64,
    )
    metrics = PromotionMetrics(
        total_games=100,
        engine_errors=1,
        illegal_choices=2,
        agent_errors=3,
        timeout_terminations=1,
        max_step_terminations=2,
        candidate_score=0.51,
        champion_score=0.50,
        decision_latencies_ms={},
        stratum_score_margins={"map:a": -0.02},
        artifact_loads=(
            ArtifactLoad(digest="a" * 64, error=None),
            ArtifactLoad(digest=None, error="load failed"),
            ArtifactLoad(digest="b" * 64, error=None),
        ),
    )

    verdict = evaluate_promotion_gates(metrics, config)

    assert verdict.promoted is False
    assert {failure.gate for failure in verdict.failures} == {
        "engine_errors",
        "illegal_choices",
        "agent_errors",
        "timeout_or_max_step_rate",
        "latency:fast:p95",
        "practical_margin",
        "required_stratum:map:a",
        "required_stratum:composition:b",
        "deterministic_artifact_load",
    }
    artifact_failure = next(
        failure for failure in verdict.failures if failure.gate == "deterministic_artifact_load"
    )
    assert (None, "load failed") in artifact_failure.actual


def test_latency_uses_nearest_rank_percentiles_for_each_declared_tier() -> None:
    config = PromotionGateConfig(
        max_timeout_or_max_step_rate=0.0,
        latency_slos=(
            LatencySLO(tier="fast", percentile=95.0, max_ms=19.0),
            LatencySLO(tier="strong", percentile=100.0, max_ms=99.0),
        ),
        practical_margin=0.0,
        required_strata=("all",),
        max_stratum_regression=0.0,
        required_artifact_loads=2,
        expected_artifact_digest="a" * 64,
    )
    metrics = PromotionMetrics(
        total_games=1,
        engine_errors=0,
        illegal_choices=0,
        agent_errors=0,
        timeout_terminations=0,
        max_step_terminations=0,
        candidate_score=0.5,
        champion_score=0.5,
        decision_latencies_ms={
            "fast": (*tuple(float(value) for value in range(1, 20)), 100.0),
            "strong": (20.0, 100.0),
        },
        stratum_score_margins={"all": 0.0},
        artifact_loads=(
            ArtifactLoad(digest="a" * 64, error=None),
            ArtifactLoad(digest="a" * 64, error=None),
        ),
    )

    verdict = evaluate_promotion_gates(metrics, config)

    latency_checks = {
        check.gate: check for check in verdict.checks if check.gate.startswith("latency")
    }
    assert latency_checks["latency:fast:p95"].actual == 19.0
    assert latency_checks["latency:fast:p95"].passed is True
    assert latency_checks["latency:strong:p100"].actual == 100.0
    assert latency_checks["latency:strong:p100"].passed is False


def test_metrics_reject_impossible_counts_before_evaluation() -> None:
    metrics = PromotionMetrics(
        total_games=1,
        engine_errors=0,
        illegal_choices=0,
        agent_errors=0,
        timeout_terminations=0,
        max_step_terminations=0,
        candidate_score=0.5,
        champion_score=0.5,
        decision_latencies_ms={"fast": (1.0,)},
        stratum_score_margins={"all": 0.0},
        artifact_loads=(ArtifactLoad(digest="a" * 64, error=None),),
    )

    with pytest.raises(ValueError):
        replace(metrics, total_games=0)
    with pytest.raises(ValueError):
        replace(metrics, illegal_choices=-1)
    with pytest.raises(ValueError):
        replace(metrics, timeout_terminations=2)


def test_repeatable_load_of_the_wrong_artifact_fails_identity_gate() -> None:
    config = PromotionGateConfig(
        max_timeout_or_max_step_rate=0.0,
        latency_slos=(LatencySLO(tier="fast", percentile=100.0, max_ms=10.0),),
        practical_margin=0.0,
        required_strata=("all",),
        max_stratum_regression=0.0,
        required_artifact_loads=2,
        expected_artifact_digest="a" * 64,
    )
    metrics = PromotionMetrics(
        total_games=1,
        engine_errors=0,
        illegal_choices=0,
        agent_errors=0,
        timeout_terminations=0,
        max_step_terminations=0,
        candidate_score=0.5,
        champion_score=0.5,
        decision_latencies_ms={"fast": (1.0,)},
        stratum_score_margins={"all": 0.0},
        artifact_loads=(
            ArtifactLoad(digest="b" * 64, error=None),
            ArtifactLoad(digest="b" * 64, error=None),
        ),
    )

    verdict = evaluate_promotion_gates(metrics, config)

    assert "deterministic_artifact_load" in {failure.gate for failure in verdict.failures}
