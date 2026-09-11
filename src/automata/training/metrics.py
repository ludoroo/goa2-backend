"""Deterministic, dependency-free diagnostics for joint policy/value models."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class PolicyMetricInput:
    """One decision's aligned target and model prediction."""

    game_id: str
    candidate_family: str
    target_probabilities: tuple[float, ...]
    predicted_logits: tuple[float, ...]
    prior_probabilities: tuple[float, ...] | None = None
    q_variances: tuple[float, ...] | None = None
    hero: str | None = None
    map_id: str | None = None
    composition: str | None = None
    round_bucket: str | None = None

    def __post_init__(self) -> None:
        size = len(self.target_probabilities)
        if not self.game_id or not self.candidate_family or size == 0:
            raise ValueError("policy metric inputs require a game, family, and candidates")
        if len(self.predicted_logits) != size:
            raise ValueError("policy targets and logits must have aligned lengths")
        _probability_distribution(self.target_probabilities, "policy target")
        _finite(self.predicted_logits, "policy logits")
        if self.prior_probabilities is not None:
            if len(self.prior_probabilities) != size:
                raise ValueError("policy priors must align with candidates")
            _probability_distribution(self.prior_probabilities, "policy priors")
        if self.q_variances is not None:
            if len(self.q_variances) != size:
                raise ValueError("Q variances must align with candidates")
            _finite(self.q_variances, "Q variances")
            if any(value < 0.0 for value in self.q_variances):
                raise ValueError("Q variances must be non-negative")


@dataclass(frozen=True, slots=True)
class ValueMetricInput:
    """One terminal value target and its perspective-correct prediction."""

    game_id: str
    target_value: int
    predicted_value: float
    candidate_family: str | None = None
    hero: str | None = None
    map_id: str | None = None
    composition: str | None = None
    round_bucket: str | None = None

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("value metric inputs require a game")
        if self.target_value not in {-1, 0, 1}:
            raise ValueError("value target must be -1, 0, or 1")
        if not math.isfinite(self.predicted_value) or not -1.0 <= self.predicted_value <= 1.0:
            raise ValueError("predicted value must be finite and in [-1, 1]")


@dataclass(frozen=True, slots=True)
class _PolicyDecision:
    cross_entropy: float
    top1_accuracy: float
    topk_recall: float
    pairwise_accuracy: float
    entropy: float
    search_overturn: float | None
    q_variance: float | None


def policy_metrics(examples: Iterable[PolicyMetricInput], *, top_k: int = 3) -> dict[str, Any]:
    """Return game-equal policy diagnostics overall and for supplied buckets."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows = tuple(examples)
    measured = {id(row): _measure_policy(row, top_k) for row in rows}
    result: dict[str, Any] = {
        "overall": _aggregate_policy(rows, measured, top_k),
        "by_candidate_family": _policy_buckets(
            rows, measured, top_k, lambda row: row.candidate_family
        ),
    }
    for output_name, attribute in (
        ("by_hero", "hero"),
        ("by_map", "map_id"),
        ("by_composition", "composition"),
        ("by_round", "round_bucket"),
    ):
        result[output_name] = _policy_buckets(
            rows,
            measured,
            top_k,
            cast(Callable[[PolicyMetricInput], str | None], attrgetter(attribute)),
        )
    return result


def value_metrics(
    examples: Iterable[ValueMetricInput],
    *,
    ece_bins: int = 10,
    saturation_threshold: float = 0.95,
) -> dict[str, Any]:
    """Return game-equal value quality, calibration, and saturation diagnostics."""
    if ece_bins <= 0:
        raise ValueError("ece_bins must be positive")
    if not math.isfinite(saturation_threshold) or not 0.0 <= saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must be in [0, 1]")
    rows = tuple(examples)
    result: dict[str, Any] = {
        "overall": _aggregate_value(rows, ece_bins, saturation_threshold),
    }
    dimensions = (
        ("by_candidate_family", "candidate_family"),
        ("by_hero", "hero"),
        ("by_map", "map_id"),
        ("by_composition", "composition"),
        ("by_round", "round_bucket"),
    )
    for output_name, attribute in dimensions:
        getter = cast(Callable[[ValueMetricInput], str | None], attrgetter(attribute))
        grouped: dict[str, list[ValueMetricInput]] = defaultdict(list)
        for row in rows:
            bucket = getter(row)
            if bucket is not None:
                grouped[bucket].append(row)
        result[output_name] = {
            bucket: _aggregate_value(grouped[bucket], ece_bins, saturation_threshold)
            for bucket in sorted(grouped)
        }
    return result


def joint_metrics(
    policy_examples: Iterable[PolicyMetricInput],
    value_examples: Iterable[ValueMetricInput],
    *,
    top_k: int = 3,
    ece_bins: int = 10,
    saturation_threshold: float = 0.95,
) -> dict[str, Any]:
    """Compute both heads' diagnostics without coupling them to a trainer."""
    return {
        "policy": policy_metrics(policy_examples, top_k=top_k),
        "value": value_metrics(
            value_examples,
            ece_bins=ece_bins,
            saturation_threshold=saturation_threshold,
        ),
    }


def _aggregate_value(
    rows: Sequence[ValueMetricInput], ece_bins: int, saturation_threshold: float
) -> dict[str, Any]:
    weights = _game_equal_weights(rows)
    if not rows:
        return {
            "count": 0,
            "game_count": 0,
            "log_loss": None,
            "brier_score": None,
            "expected_calibration_error": None,
            "accuracy": None,
            "saturation_rate": None,
        }

    probabilities = tuple((row.predicted_value + 1.0) / 2.0 for row in rows)
    targets = tuple((row.target_value + 1.0) / 2.0 for row in rows)
    clipped = tuple(min(max(value, 1e-15), 1.0 - 1e-15) for value in probabilities)
    log_loss = sum(
        weight * -(target * math.log(prediction) + (1.0 - target) * math.log1p(-prediction))
        for target, prediction, weight in zip(targets, clipped, weights, strict=True)
    )
    brier = sum(
        weight * (prediction - target) ** 2
        for prediction, target, weight in zip(probabilities, targets, weights, strict=True)
    )
    accuracy = sum(
        weight * float(_sign(row.predicted_value) == row.target_value)
        for row, weight in zip(rows, weights, strict=True)
    )
    saturation = sum(
        weight * float(abs(row.predicted_value) >= saturation_threshold)
        for row, weight in zip(rows, weights, strict=True)
    )

    calibration = 0.0
    for bin_index in range(ece_bins):
        members = [
            index
            for index, probability in enumerate(probabilities)
            if min(int(probability * ece_bins), ece_bins - 1) == bin_index
        ]
        bin_weight = sum(weights[index] for index in members)
        if bin_weight:
            predicted = sum(weights[index] * probabilities[index] for index in members) / bin_weight
            observed = sum(weights[index] * targets[index] for index in members) / bin_weight
            calibration += bin_weight * abs(predicted - observed)

    return {
        "count": len(rows),
        "game_count": len({row.game_id for row in rows}),
        "log_loss": log_loss,
        "brier_score": brier,
        "expected_calibration_error": calibration,
        "accuracy": accuracy,
        "saturation_rate": saturation,
    }


def _measure_policy(row: PolicyMetricInput, top_k: int) -> _PolicyDecision:
    probabilities = _softmax(row.predicted_logits)
    cross_entropy = -sum(
        target * math.log(max(prediction, 1e-300))
        for target, prediction in zip(row.target_probabilities, probabilities, strict=True)
    )
    target_best = _max_indices(row.target_probabilities)
    predicted_best = _max_indices(probabilities)
    top1 = len(target_best & predicted_best) / len(predicted_best)
    ranked = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    topk = len(target_best & set(ranked[:top_k])) / len(target_best)

    comparisons: list[float] = []
    for left in range(len(probabilities)):
        for right in range(left + 1, len(probabilities)):
            target_delta = row.target_probabilities[left] - row.target_probabilities[right]
            if target_delta == 0.0:
                continue
            prediction_delta = probabilities[left] - probabilities[right]
            comparisons.append(
                1.0
                if target_delta * prediction_delta > 0.0
                else 0.5 if prediction_delta == 0.0 else 0.0
            )
    pairwise = sum(comparisons) / len(comparisons) if comparisons else 0.5
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    overturn = None
    if row.prior_probabilities is not None:
        overturn = float(_max_indices(row.prior_probabilities).isdisjoint(target_best))
    q_variance = (
        sum(row.q_variances) / len(row.q_variances) if row.q_variances is not None else None
    )
    return _PolicyDecision(cross_entropy, top1, topk, pairwise, entropy, overturn, q_variance)


def _aggregate_policy(
    rows: Sequence[PolicyMetricInput],
    measured: dict[int, _PolicyDecision],
    top_k: int,
) -> dict[str, Any]:
    def average(field: str) -> float | None:
        available_rows = [row for row in rows if getattr(measured[id(row)], field) is not None]
        if not available_rows:
            return None
        available_weights = _game_equal_weights(available_rows)
        return sum(
            float(getattr(measured[id(row)], field)) * weight
            for row, weight in zip(available_rows, available_weights, strict=True)
        )

    return {
        "count": len(rows),
        "game_count": len({row.game_id for row in rows}),
        "cross_entropy": average("cross_entropy"),
        "top1_accuracy": average("top1_accuracy"),
        f"top{top_k}_recall": average("topk_recall"),
        "pairwise_accuracy": average("pairwise_accuracy"),
        "entropy": average("entropy"),
        "search_overturn_rate": average("search_overturn"),
        "q_variance": average("q_variance"),
    }


def _policy_buckets(
    rows: Sequence[PolicyMetricInput],
    measured: dict[int, _PolicyDecision],
    top_k: int,
    key: Callable[[PolicyMetricInput], str | None],
) -> dict[str, Any]:
    grouped: dict[str, list[PolicyMetricInput]] = defaultdict(list)
    for row in rows:
        bucket = key(row)
        if bucket is not None:
            grouped[bucket].append(row)
    return {
        bucket: _aggregate_policy(grouped[bucket], measured, top_k) for bucket in sorted(grouped)
    }


def _game_equal_weights(
    rows: Sequence[PolicyMetricInput | ValueMetricInput],
) -> tuple[float, ...]:
    if not rows:
        return ()
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.game_id] += 1
    game_count = len(counts)
    return tuple(1.0 / game_count / counts[row.game_id] for row in rows)


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    maximum = max(logits)
    exponentials = tuple(math.exp(value - maximum) for value in logits)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _max_indices(values: Sequence[float]) -> set[int]:
    maximum = max(values)
    return {index for index, value in enumerate(values) if value == maximum}


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _finite(values: Sequence[float], name: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite values")


def _probability_distribution(values: Sequence[float], name: str) -> None:
    _finite(values, name)
    if any(value < 0.0 for value in values) or not math.isclose(
        sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(f"{name} must be a probability distribution")


@dataclass(slots=True)
class _RunningMean:
    total: float = 0.0
    count: int = 0

    def add(self, value: float | None) -> None:
        if value is not None:
            self.total += value
            self.count += 1

    def compute(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass(slots=True)
class _PolicyGameMetrics:
    cross_entropy: _RunningMean = field(default_factory=_RunningMean)
    top1_accuracy: _RunningMean = field(default_factory=_RunningMean)
    topk_recall: _RunningMean = field(default_factory=_RunningMean)
    pairwise_accuracy: _RunningMean = field(default_factory=_RunningMean)
    entropy: _RunningMean = field(default_factory=_RunningMean)
    search_overturn: _RunningMean = field(default_factory=_RunningMean)
    q_variance: _RunningMean = field(default_factory=_RunningMean)

    def add(self, decision: _PolicyDecision) -> None:
        self.cross_entropy.add(decision.cross_entropy)
        self.top1_accuracy.add(decision.top1_accuracy)
        self.topk_recall.add(decision.topk_recall)
        self.pairwise_accuracy.add(decision.pairwise_accuracy)
        self.entropy.add(decision.entropy)
        self.search_overturn.add(decision.search_overturn)
        self.q_variance.add(decision.q_variance)


@dataclass(slots=True)
class _PolicyAggregate:
    count: int = 0
    games: dict[str, _PolicyGameMetrics] = field(default_factory=dict)

    def add(self, game_id: str, decision: _PolicyDecision) -> None:
        self.count += 1
        game = self.games.get(game_id)
        if game is None:
            game = self.games[game_id] = _PolicyGameMetrics()
        game.add(decision)

    def compute(self, top_k: int) -> dict[str, Any]:
        def game_equal_average(name: str) -> float | None:
            means = [
                value
                for game in self.games.values()
                if (value := getattr(game, name).compute()) is not None
            ]
            return sum(means) / len(means) if means else None

        return {
            "count": self.count,
            "game_count": len(self.games),
            "cross_entropy": game_equal_average("cross_entropy"),
            "top1_accuracy": game_equal_average("top1_accuracy"),
            f"top{top_k}_recall": game_equal_average("topk_recall"),
            "pairwise_accuracy": game_equal_average("pairwise_accuracy"),
            "entropy": game_equal_average("entropy"),
            "search_overturn_rate": game_equal_average("search_overturn"),
            "q_variance": game_equal_average("q_variance"),
        }


@dataclass(slots=True)
class _ValueGameMetrics:
    count: int
    log_loss: float
    brier_score: float
    accuracy: float
    saturation_rate: float
    calibration_differences: list[float]

    @classmethod
    def empty(cls, ece_bins: int) -> _ValueGameMetrics:
        return cls(0, 0.0, 0.0, 0.0, 0.0, [0.0] * ece_bins)


@dataclass(slots=True)
class _ValueAggregate:
    ece_bins: int
    saturation_threshold: float
    count: int = 0
    games: dict[str, _ValueGameMetrics] = field(default_factory=dict)

    def add(self, row: ValueMetricInput) -> None:
        probability = (row.predicted_value + 1.0) / 2.0
        target = (row.target_value + 1.0) / 2.0
        clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
        game = self.games.get(row.game_id)
        if game is None:
            game = self.games[row.game_id] = _ValueGameMetrics.empty(self.ece_bins)

        self.count += 1
        game.count += 1
        game.log_loss += -(target * math.log(clipped) + (1.0 - target) * math.log1p(-clipped))
        game.brier_score += (probability - target) ** 2
        game.accuracy += float(_sign(row.predicted_value) == row.target_value)
        game.saturation_rate += float(abs(row.predicted_value) >= self.saturation_threshold)
        bin_index = min(int(probability * self.ece_bins), self.ece_bins - 1)
        game.calibration_differences[bin_index] += probability - target

    def compute(self) -> dict[str, Any]:
        if not self.games:
            return {
                "count": 0,
                "game_count": 0,
                "log_loss": None,
                "brier_score": None,
                "expected_calibration_error": None,
                "accuracy": None,
                "saturation_rate": None,
            }

        game_count = len(self.games)

        def game_equal_average(name: str) -> float:
            return (
                sum(getattr(game, name) / game.count for game in self.games.values()) / game_count
            )

        calibration = sum(
            abs(
                sum(
                    game.calibration_differences[bin_index] / game.count
                    for game in self.games.values()
                )
                / game_count
            )
            for bin_index in range(self.ece_bins)
        )
        return {
            "count": self.count,
            "game_count": game_count,
            "log_loss": game_equal_average("log_loss"),
            "brier_score": game_equal_average("brier_score"),
            "expected_calibration_error": calibration,
            "accuracy": game_equal_average("accuracy"),
            "saturation_rate": game_equal_average("saturation_rate"),
        }


_METRIC_DIMENSIONS = (
    ("by_candidate_family", "candidate_family"),
    ("by_hero", "hero"),
    ("by_map", "map_id"),
    ("by_composition", "composition"),
    ("by_round", "round_bucket"),
)


class _PolicyMetricsAccumulator:
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k
        self.overall = _PolicyAggregate()
        self.buckets: dict[str, dict[str, _PolicyAggregate]] = {
            output_name: {} for output_name, _ in _METRIC_DIMENSIONS
        }

    def add(self, row: PolicyMetricInput) -> None:
        decision = _measure_policy(row, self.top_k)
        self.overall.add(row.game_id, decision)
        for output_name, attribute in _METRIC_DIMENSIONS:
            bucket = getattr(row, attribute)
            if bucket is not None:
                groups = self.buckets[output_name]
                aggregate = groups.get(bucket)
                if aggregate is None:
                    aggregate = groups[bucket] = _PolicyAggregate()
                aggregate.add(row.game_id, decision)

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {"overall": self.overall.compute(self.top_k)}
        for output_name, _ in _METRIC_DIMENSIONS:
            groups = self.buckets[output_name]
            result[output_name] = {
                bucket: groups[bucket].compute(self.top_k) for bucket in sorted(groups)
            }
        return result


class _ValueMetricsAccumulator:
    def __init__(self, ece_bins: int, saturation_threshold: float) -> None:
        self.ece_bins = ece_bins
        self.saturation_threshold = saturation_threshold
        self.overall = self._new_aggregate()
        self.buckets: dict[str, dict[str, _ValueAggregate]] = {
            output_name: {} for output_name, _ in _METRIC_DIMENSIONS
        }

    def _new_aggregate(self) -> _ValueAggregate:
        return _ValueAggregate(self.ece_bins, self.saturation_threshold)

    def add(self, row: ValueMetricInput) -> None:
        self.overall.add(row)
        for output_name, attribute in _METRIC_DIMENSIONS:
            bucket = getattr(row, attribute)
            if bucket is not None:
                groups = self.buckets[output_name]
                aggregate = groups.get(bucket)
                if aggregate is None:
                    aggregate = groups[bucket] = self._new_aggregate()
                aggregate.add(row)

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {"overall": self.overall.compute()}
        for output_name, _ in _METRIC_DIMENSIONS:
            groups = self.buckets[output_name]
            result[output_name] = {bucket: groups[bucket].compute() for bucket in sorted(groups)}
        return result


class JointMetricsAccumulator:
    """Incrementally compute exact game-equal joint metrics without retaining examples.

    The accumulator keeps only per-game summaries for the overall result and populated
    buckets. Candidate arrays, logits, and individual examples are discarded after each add.
    """

    def __init__(
        self,
        *,
        top_k: int = 3,
        ece_bins: int = 10,
        saturation_threshold: float = 0.95,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if ece_bins <= 0:
            raise ValueError("ece_bins must be positive")
        if not math.isfinite(saturation_threshold) or not 0.0 <= saturation_threshold <= 1.0:
            raise ValueError("saturation_threshold must be in [0, 1]")
        self._policy = _PolicyMetricsAccumulator(top_k)
        self._value = _ValueMetricsAccumulator(ece_bins, saturation_threshold)

    def add_policy(self, example: PolicyMetricInput) -> None:
        """Add one policy decision."""
        self._policy.add(example)

    def add_value(self, example: ValueMetricInput) -> None:
        """Add one value target and prediction."""
        self._value.add(example)

    def update(
        self,
        policy_examples: Iterable[PolicyMetricInput] = (),
        value_examples: Iterable[ValueMetricInput] = (),
    ) -> None:
        """Add zero or more policy and value examples."""
        for policy_example in policy_examples:
            self.add_policy(policy_example)
        for value_example in value_examples:
            self.add_value(value_example)

    def compute(self) -> dict[str, Any]:
        """Return a JSON-safe metrics snapshot; more examples may be added afterward."""
        return {"policy": self._policy.compute(), "value": self._value.compute()}


__all__ = [
    "JointMetricsAccumulator",
    "PolicyMetricInput",
    "ValueMetricInput",
    "joint_metrics",
    "policy_metrics",
    "value_metrics",
]
