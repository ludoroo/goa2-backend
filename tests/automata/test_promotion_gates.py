"""Behavioral tests for the neutral promotion-gate evaluator."""

from dataclasses import replace
from typing import Any

import pytest

from automata.decision import DecisionSemanticRole
from automata.evaluation.promotion_gates import (
    ArtifactLoad,
    LatencySLO,
    PolicyRankingGateConfig,
    PromotionGateConfig,
    PromotionMetrics,
    evaluate_policy_ranking_gates,
    evaluate_promotion_gates,
)
from automata.policy_ranking import (
    PolicyRankingBucket,
    PolicyRankingSnapshot,
    snapshot_from_policy_metrics,
)
from automata.training.metrics import PolicyMetricInput, policy_metrics

_DATASET_DIGEST = "d" * 64
_SPLIT_DIGEST = "e" * 64


def _ranking_bucket(
    count: int = 10,
    accuracy: float | None = 0.7,
    *,
    multi_candidate_count: int | None = None,
    game_count: int = 2,
) -> PolicyRankingBucket:
    return PolicyRankingBucket(
        count=count,
        multi_candidate_count=(count if multi_candidate_count is None else multi_candidate_count),
        game_count=game_count,
        pairwise_accuracy=accuracy,
    )


def _ranking_snapshot(
    *,
    artifact: str,
    overall: PolicyRankingBucket | None = None,
    roles: dict[DecisionSemanticRole, PolicyRankingBucket] | None = None,
    dataset_digest: str = _DATASET_DIGEST,
    split_digest: str = _SPLIT_DIGEST,
) -> PolicyRankingSnapshot:
    return PolicyRankingSnapshot(
        artifact_digest=artifact,
        dataset_digest=dataset_digest,
        split_digest=split_digest,
        overall=overall or _ranking_bucket(),
        by_semantic_role=roles or {DecisionSemanticRole.PLANNING: _ranking_bucket()},
    )


def test_offline_policy_ranking_gate_requires_overall_and_every_declared_role() -> None:
    config = PolicyRankingGateConfig(
        required_semantic_roles=(
            DecisionSemanticRole.PLANNING,
            DecisionSemanticRole.DEFENSE_REACTION,
        ),
        min_samples=2,
        min_games=1,
    )
    parent = _ranking_snapshot(
        artifact="a" * 64,
        overall=_ranking_bucket(count=100, accuracy=0.70, game_count=8),
        roles={
            DecisionSemanticRole.PLANNING: _ranking_bucket(count=40, accuracy=0.65, game_count=8),
            DecisionSemanticRole.DEFENSE_REACTION: _ranking_bucket(
                count=10, accuracy=0.60, game_count=4
            ),
        },
    )
    passing_candidate = _ranking_snapshot(
        artifact="b" * 64,
        overall=_ranking_bucket(count=100, accuracy=0.71, game_count=8),
        roles={
            DecisionSemanticRole.PLANNING: _ranking_bucket(count=40, accuracy=0.66, game_count=8),
            DecisionSemanticRole.DEFENSE_REACTION: _ranking_bucket(
                count=10, accuracy=0.61, game_count=4
            ),
        },
    )
    missing_role_candidate = _ranking_snapshot(
        artifact="b" * 64,
        overall=passing_candidate.overall,
        roles={
            DecisionSemanticRole.PLANNING: passing_candidate.by_semantic_role[
                DecisionSemanticRole.PLANNING
            ]
        },
    )

    passing = evaluate_policy_ranking_gates(passing_candidate, parent, config)
    verdict = evaluate_policy_ranking_gates(missing_role_candidate, parent, config)

    assert passing.promoted is True
    assert verdict.promoted is False
    assert tuple(check.gate for check in verdict.checks) == (
        "offline_ranking:dataset_identity",
        "offline_ranking:split_identity",
        "offline_ranking:overall",
        "offline_ranking:DEFENSE_REACTION",
        "offline_ranking:PLANNING",
    )
    assert {failure.gate for failure in verdict.failures} == {"offline_ranking:DEFENSE_REACTION"}


def test_offline_policy_ranking_gate_boundary_and_evidence_failures() -> None:
    config = PolicyRankingGateConfig(
        required_semantic_roles=(DecisionSemanticRole.PLANNING,),
        min_samples=4,
        min_games=2,
        max_overall_regression=0.01,
        max_role_regression=0.02,
    )
    parent = _ranking_snapshot(
        artifact="a" * 64,
        overall=_ranking_bucket(accuracy=0.70),
        roles={DecisionSemanticRole.PLANNING: _ranking_bucket(count=4, accuracy=0.70)},
    )
    boundary = _ranking_snapshot(
        artifact="b" * 64,
        overall=_ranking_bucket(accuracy=0.69),
        roles={DecisionSemanticRole.PLANNING: _ranking_bucket(count=4, accuracy=0.68)},
    )
    unequal = _ranking_snapshot(
        artifact="b" * 64,
        overall=_ranking_bucket(
            count=10,
            accuracy=0.69,
            multi_candidate_count=11,
        ),
        roles={
            DecisionSemanticRole.PLANNING: _ranking_bucket(
                count=4,
                accuracy=0.68,
                game_count=3,
            )
        },
    )

    assert evaluate_policy_ranking_gates(boundary, parent, config).promoted is True
    verdict = evaluate_policy_ranking_gates(unequal, parent, config)

    assert verdict.promoted is False
    assert {failure.gate for failure in verdict.failures} == {
        "offline_ranking:overall",
        "offline_ranking:PLANNING",
    }
    overall = next(check for check in verdict.checks if check.gate.endswith("overall"))
    assert overall.actual["candidate_pairwise_accuracy"] == 0.69
    assert overall.actual["parent_pairwise_accuracy"] == 0.70


def test_offline_policy_ranking_gate_rejects_dataset_and_split_misalignment() -> None:
    config = PolicyRankingGateConfig(
        required_semantic_roles=(DecisionSemanticRole.PLANNING,),
        min_samples=2,
        min_games=1,
    )
    parent = _ranking_snapshot(artifact="a" * 64)
    candidate = _ranking_snapshot(
        artifact="b" * 64,
        dataset_digest="c" * 64,
        split_digest="f" * 64,
    )

    verdict = evaluate_policy_ranking_gates(candidate, parent, config)

    assert verdict.promoted is False
    assert {failure.gate for failure in verdict.failures} == {
        "offline_ranking:dataset_identity",
        "offline_ranking:split_identity",
    }


def test_snapshot_from_policy_metrics_is_complete_and_canonical() -> None:
    metrics = policy_metrics(
        [
            PolicyMetricInput(
                game_id="game-1",
                candidate_family="CARD",
                target_probabilities=(0.8, 0.2),
                predicted_logits=(1.0, 0.0),
                input_request_type=None,
                semantic_role=DecisionSemanticRole.PLANNING,
                can_skip=False,
            )
        ]
    )

    snapshot = snapshot_from_policy_metrics(
        metrics,
        artifact_digest="a" * 64,
        dataset_digest=_DATASET_DIGEST,
        split_digest=_SPLIT_DIGEST,
    )

    assert snapshot.overall == _ranking_bucket(count=1, accuracy=1.0, game_count=1)
    assert snapshot.by_semantic_role[DecisionSemanticRole.PLANNING] == snapshot.overall
    assert snapshot.by_semantic_role[DecisionSemanticRole.ATTACK_TARGET] == _ranking_bucket(
        count=0,
        accuracy=None,
        multi_candidate_count=0,
        game_count=0,
    )
    payload = snapshot.canonical_bytes()
    assert PolicyRankingSnapshot.from_canonical_bytes(payload) == snapshot
    assert b'"artifact_digest":"' + b"a" * 64 in payload
    with pytest.raises(ValueError, match="canonical JSON"):
        PolicyRankingSnapshot.from_canonical_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="invalid"):
        PolicyRankingSnapshot.from_canonical_bytes(
            payload.replace(b'"schema_version":1', b'"schema_version":true')
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
