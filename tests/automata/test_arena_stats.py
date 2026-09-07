from __future__ import annotations

import pytest

from automata.evaluation.arena_stats import (
    SequentialBoundary,
    SequentialDecision,
    SequentialPlan,
    evaluate_sequential,
    pair_seed_scores,
    paired_confidence_interval,
)
from automata.evaluation.protocol import EvaluationGameResult


def _observation(seed: int, a_side: str, winner_side: str | None) -> EvaluationGameResult:
    return EvaluationGameResult(
        case_id=f"{seed}-{a_side}",
        world_seed=seed,
        a_side=a_side,
        winner_side=winner_side,
        rounds=1,
        steps=1,
        reason="game_over",
    )


def test_pair_seed_scores_counts_draws_as_half_points_before_averaging() -> None:
    pairs = pair_seed_scores(
        [
            _observation(7, "RED", "RED"),
            _observation(7, "BLUE", None),
            _observation(3, "RED", "BLUE"),
            _observation(3, "BLUE", "BLUE"),
        ]
    )

    assert [(pair.world_seed, pair.score) for pair in pairs] == [(3, 0.5), (7, 0.75)]


def test_pairing_is_invariant_to_observation_order() -> None:
    rows = [
        _observation(7, "RED", "RED"),
        _observation(3, "BLUE", "BLUE"),
        _observation(7, "BLUE", None),
        _observation(3, "RED", "BLUE"),
    ]

    assert pair_seed_scores(rows) == pair_seed_scores(reversed(rows))


def test_pairing_rejects_a_seed_with_a_missing_side() -> None:
    with pytest.raises(ValueError, match=r"seed 7.*missing.*BLUE"):
        pair_seed_scores([_observation(7, "RED", "RED")])


def test_pairing_rejects_conflicting_rows_for_the_same_seed_and_side() -> None:
    with pytest.raises(ValueError, match=r"seed 7.*RED"):
        pair_seed_scores(
            [
                _observation(7, "RED", "RED"),
                _observation(7, "RED", "BLUE"),
                _observation(7, "BLUE", "BLUE"),
            ]
        )


@pytest.mark.parametrize("bad_side", ["red", "GREEN"])
def test_pairing_rejects_unknown_candidate_side(bad_side: str) -> None:
    with pytest.raises(ValueError, match="a_side"):
        pair_seed_scores([_observation(7, bad_side, "RED")])


def test_pairing_rejects_unknown_winner_side() -> None:
    with pytest.raises(ValueError, match="winner_side"):
        pair_seed_scores([_observation(7, "RED", "GREEN"), _observation(7, "BLUE", "BLUE")])


def test_paired_confidence_interval_matches_hoeffding_reference_calculation() -> None:
    rows: list[EvaluationGameResult] = []
    for seed in range(100):
        rows.extend(
            [
                _observation(seed, "RED", "RED"),
                _observation(seed, "BLUE", None),
            ]
        )

    interval = paired_confidence_interval(rows, confidence_level=0.95)

    # Every pair is (1 + .5) / 2 = .75. For bounded independent pair scores,
    # Hoeffding's two-sided 95% radius is sqrt(log(40) / 200).
    assert interval.mean == pytest.approx(0.75)
    assert interval.lower == pytest.approx(0.6141898484)
    assert interval.upper == pytest.approx(0.8858101516)
    assert interval.pair_count == 100
    assert interval.confidence_level == pytest.approx(0.95)


def test_paired_confidence_interval_rejects_empty_or_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        paired_confidence_interval([])
    for bad_confidence in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="confidence_level"):
            paired_confidence_interval([], confidence_level=bad_confidence)


def _uniform_pairs(
    count: int, candidate_score: float, *, start_seed: int = 0
) -> list[EvaluationGameResult]:
    rows: list[EvaluationGameResult] = []
    for seed in range(start_seed, start_seed + count):
        red_winner: str | None = "RED" if candidate_score == 1.0 else "BLUE"
        blue_winner: str | None = "BLUE" if candidate_score == 1.0 else "RED"
        if candidate_score == 0.5:
            red_winner = blue_winner = None
        rows.extend(
            [
                _observation(seed, "RED", red_winner),
                _observation(seed, "BLUE", blue_winner),
            ]
        )
    return rows


def test_sequential_test_stops_only_at_a_predeclared_boundary() -> None:
    plan = SequentialPlan(boundaries=(SequentialBoundary(pair_count=10, alpha=0.05),))

    before = evaluate_sequential(_uniform_pairs(9, 1.0), plan)
    at_boundary = evaluate_sequential(_uniform_pairs(10, 1.0), plan)

    assert before.decision is SequentialDecision.CONTINUE
    assert before.evaluated_pair_count == 0
    assert at_boundary.decision is SequentialDecision.PROMOTE
    assert at_boundary.evaluated_pair_count == 10


def test_sequential_test_has_symmetric_promote_reject_and_inconclusive_decisions() -> None:
    plan = SequentialPlan(boundaries=(SequentialBoundary(pair_count=10, alpha=0.05),))

    promote = evaluate_sequential(_uniform_pairs(10, 1.0), plan)
    reject = evaluate_sequential(_uniform_pairs(10, 0.0), plan)
    inconclusive = evaluate_sequential(_uniform_pairs(10, 0.5), plan)

    assert promote.decision is SequentialDecision.PROMOTE
    assert reject.decision is SequentialDecision.REJECT
    assert inconclusive.decision is SequentialDecision.INCONCLUSIVE


def test_sequential_test_uses_each_predeclared_alpha_spend() -> None:
    plan = SequentialPlan(boundaries=(SequentialBoundary(pair_count=100, alpha=0.01),))

    result = evaluate_sequential(_uniform_pairs(100, 1.0), plan)

    assert result.interval is not None
    assert result.interval.confidence_level == pytest.approx(0.99)
    assert result.interval.lower == pytest.approx(0.8372376369)
    assert plan.familywise_alpha == pytest.approx(0.01)


def test_sequential_practical_margin_uses_candidate_advantage_scale() -> None:
    plan = SequentialPlan(
        boundaries=(SequentialBoundary(pair_count=100, alpha=0.01),),
        practical_margin=0.10,
    )

    # A +0.10 candidate-vs-champion advantage is a 0.55 candidate score.
    assert plan.target_score == pytest.approx(0.55)


def test_sequential_test_is_invariant_to_observation_order() -> None:
    rows = _uniform_pairs(10, 0.5) + _uniform_pairs(10, 1.0, start_seed=10)
    plan = SequentialPlan(
        boundaries=(
            SequentialBoundary(pair_count=10, alpha=0.4),
            SequentialBoundary(pair_count=20, alpha=0.4),
        )
    )

    forward = evaluate_sequential(rows, plan)
    reverse = evaluate_sequential(reversed(rows), plan)
    assert forward == reverse
    assert forward.decision is SequentialDecision.PROMOTE
    assert forward.evaluated_pair_count == 20


@pytest.mark.parametrize("pair_count,alpha", [(0, 0.05), (10, 0.0), (10, 1.0)])
def test_sequential_boundary_rejects_invalid_values(pair_count: int, alpha: float) -> None:
    with pytest.raises(ValueError):
        SequentialBoundary(pair_count=pair_count, alpha=alpha)


def test_sequential_plan_requires_increasing_nonempty_boundaries() -> None:
    with pytest.raises(ValueError):
        SequentialPlan(boundaries=())
    with pytest.raises(ValueError):
        SequentialPlan(
            boundaries=(
                SequentialBoundary(pair_count=10, alpha=0.05),
                SequentialBoundary(pair_count=10, alpha=0.05),
            )
        )
    with pytest.raises(ValueError, match="sum"):
        SequentialPlan(
            boundaries=(
                SequentialBoundary(pair_count=10, alpha=0.5),
                SequentialBoundary(pair_count=20, alpha=0.5),
            )
        )


def test_sequential_test_rejects_samples_beyond_the_predeclared_maximum() -> None:
    plan = SequentialPlan(boundaries=(SequentialBoundary(pair_count=10, alpha=0.05),))

    with pytest.raises(ValueError, match="predeclared maximum"):
        evaluate_sequential(_uniform_pairs(11, 1.0), plan)
