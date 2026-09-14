"""Paired statistics for candidate-versus-champion arena games."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from .protocol import EvaluationGameResult


@dataclass(frozen=True)
class PairedSeedScore:
    """Candidate scores from both side assignments for one world seed."""

    world_seed: int
    red_score: float
    blue_score: float

    @property
    def score(self) -> float:
        return (self.red_score + self.blue_score) / 2.0


@dataclass(frozen=True)
class ConfidenceInterval:
    """Distribution-free interval over independent, bounded pair scores."""

    mean: float
    lower: float
    upper: float
    confidence_level: float
    pair_count: int


@dataclass(frozen=True)
class SequentialBoundary:
    """One predeclared look and its portion of the total type-I error budget."""

    pair_count: int
    alpha: float

    def __post_init__(self) -> None:
        if self.pair_count <= 0:
            raise ValueError("boundary pair_count must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("boundary alpha must be strictly between 0 and 1")


@dataclass(frozen=True)
class SequentialPlan:
    """Stopping schedule with margin on candidate-minus-champion effect scale."""

    boundaries: tuple[SequentialBoundary, ...]
    practical_margin: float = 0.0

    def __post_init__(self) -> None:
        if not self.boundaries:
            raise ValueError("at least one stopping boundary is required")
        counts = tuple(boundary.pair_count for boundary in self.boundaries)
        if any(later <= earlier for earlier, later in pairwise(counts)):
            raise ValueError("boundary pair counts must be strictly increasing")
        if not 0.0 <= self.practical_margin < 1.0:
            raise ValueError("practical_margin must be in [0, 1) on the effect scale")
        if self.familywise_alpha >= 1.0:
            raise ValueError("the sum of boundary alpha spends must be less than 1")

    @property
    def familywise_alpha(self) -> float:
        """Bonferroni upper bound on type-I error across every declared look."""
        return math.fsum(boundary.alpha for boundary in self.boundaries)

    @property
    def target_score(self) -> float:
        """Convert candidate-minus-champion advantage to candidate score share."""
        return 0.5 + self.practical_margin / 2.0


class SequentialDecision(StrEnum):
    """Current outcome of a predeclared sequential test."""

    CONTINUE = "CONTINUE"
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class SequentialResult:
    """Decision at the earliest eligible predeclared stopping boundary."""

    decision: SequentialDecision
    observed_pair_count: int
    evaluated_pair_count: int
    interval: ConfidenceInterval | None
    boundary: SequentialBoundary | None


def _candidate_score(observation: EvaluationGameResult) -> float:
    if observation.winner_side is None:
        return 0.5
    return 1.0 if observation.winner_side == observation.a_side else 0.0


def pair_seed_scores(observations: Iterable[EvaluationGameResult]) -> tuple[PairedSeedScore, ...]:
    """Pair A-on-RED and A-on-BLUE games by world seed.

    Every seed must have exactly one observation for each side. Invalid sides,
    winners, duplicate side assignments, and incomplete pairs are rejected
    rather than silently biasing the sampling unit.
    """
    grouped: dict[int, dict[str, EvaluationGameResult]] = {}
    for observation in observations:
        if observation.a_side not in ("RED", "BLUE"):
            raise ValueError(f"invalid a_side {observation.a_side!r}")
        if observation.winner_side not in (None, "RED", "BLUE"):
            raise ValueError(f"invalid winner_side {observation.winner_side!r}")
        by_side = grouped.setdefault(observation.world_seed, {})
        if observation.a_side in by_side:
            raise ValueError(
                f"conflicting observations for seed {observation.world_seed} "
                f"and side {observation.a_side}"
            )
        by_side[observation.a_side] = observation

    for seed, by_side in grouped.items():
        missing = {"RED", "BLUE"} - by_side.keys()
        if missing:
            raise ValueError(f"seed {seed} is missing side(s): {', '.join(sorted(missing))}")

    return tuple(
        PairedSeedScore(
            world_seed=seed,
            red_score=_candidate_score(by_side["RED"]),
            blue_score=_candidate_score(by_side["BLUE"]),
        )
        for seed, by_side in sorted(grouped.items())
    )


def _confidence_interval_from_pairs(
    pairs: tuple[PairedSeedScore, ...], confidence_level: float
) -> ConfidenceInterval:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if not pairs:
        raise ValueError("at least one complete seed pair is required")

    pair_count = len(pairs)
    mean = math.fsum(pair.score for pair in pairs) / pair_count
    alpha = 1.0 - confidence_level
    radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * pair_count))
    return ConfidenceInterval(
        mean=mean,
        lower=max(0.0, mean - radius),
        upper=min(1.0, mean + radius),
        confidence_level=confidence_level,
        pair_count=pair_count,
    )


def paired_confidence_interval(
    observations: Iterable[EvaluationGameResult], *, confidence_level: float = 0.95
) -> ConfidenceInterval:
    """Calculate a two-sided Hoeffding interval using seed pairs as samples."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    return _confidence_interval_from_pairs(pair_seed_scores(observations), confidence_level)


def evaluate_sequential(
    observations: Iterable[EvaluationGameResult], plan: SequentialPlan
) -> SequentialResult:
    """Evaluate paired outcomes only at the plan's alpha-spending looks.

    A promotion or rejection is returned at the first boundary whose interval
    lies wholly above or below the required score. Reaching the final boundary
    without crossing either side is inconclusive. Before then, the test
    continues. Pair sorting by world seed makes replay independent of row order.
    """
    pairs = pair_seed_scores(observations)
    observed_count = len(pairs)
    maximum = plan.boundaries[-1].pair_count
    if observed_count > maximum:
        raise ValueError(f"observed {observed_count} pairs exceeds predeclared maximum {maximum}")

    last_interval: ConfidenceInterval | None = None
    last_boundary: SequentialBoundary | None = None
    for boundary in plan.boundaries:
        if observed_count < boundary.pair_count:
            break
        look_pairs = pairs[: boundary.pair_count]
        interval = _confidence_interval_from_pairs(look_pairs, 1.0 - boundary.alpha)
        last_interval = interval
        last_boundary = boundary
        if interval.lower > plan.target_score:
            decision = SequentialDecision.PROMOTE
        elif interval.upper < plan.target_score:
            decision = SequentialDecision.REJECT
        else:
            continue
        return SequentialResult(
            decision=decision,
            observed_pair_count=observed_count,
            evaluated_pair_count=boundary.pair_count,
            interval=interval,
            boundary=boundary,
        )

    final_reached = observed_count == maximum
    return SequentialResult(
        decision=(
            SequentialDecision.INCONCLUSIVE if final_reached else SequentialDecision.CONTINUE
        ),
        observed_pair_count=observed_count,
        evaluated_pair_count=last_boundary.pair_count if last_boundary else 0,
        interval=last_interval,
        boundary=last_boundary,
    )
