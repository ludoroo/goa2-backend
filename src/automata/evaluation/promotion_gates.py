"""Neutral, deterministic evaluation of predeclared promotion gates.

The evaluator consumes measurements; it does not run games, load models, or
choose policy.  Every gate is represented in the returned evidence so callers
can explain a rejection without relying on the first failure encountered.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class LatencySLO:
    """One deployment tier's nearest-rank percentile latency budget."""

    tier: str
    percentile: float
    max_ms: float


@dataclass(frozen=True, slots=True)
class ArtifactLoad:
    """Outcome of one independent load of the candidate artifact."""

    digest: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class PromotionGateConfig:
    """Policy choices; practical margin is candidate-minus-champion advantage."""

    max_timeout_or_max_step_rate: float
    latency_slos: tuple[LatencySLO, ...]
    practical_margin: float
    required_strata: tuple[str, ...]
    max_stratum_regression: float
    required_artifact_loads: int
    expected_artifact_digest: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_timeout_or_max_step_rate) or not (
            0.0 <= self.max_timeout_or_max_step_rate <= 1.0
        ):
            raise ValueError("max timeout/max-step rate must be finite and between zero and one")
        if not self.latency_slos:
            raise ValueError("at least one latency SLO must be declared")
        tiers = [slo.tier for slo in self.latency_slos]
        if len(tiers) != len(set(tiers)):
            raise ValueError("latency SLO tiers must be unique")
        for slo in self.latency_slos:
            if not slo.tier:
                raise ValueError("latency SLO tier must be non-empty")
            if not math.isfinite(slo.percentile) or not 0.0 < slo.percentile <= 100.0:
                raise ValueError("latency percentile must be finite and in (0, 100]")
            if not math.isfinite(slo.max_ms) or slo.max_ms <= 0.0:
                raise ValueError("latency maximum must be finite and positive")
        if not math.isfinite(self.practical_margin) or not 0.0 <= self.practical_margin < 1.0:
            raise ValueError("practical margin must be finite and in [0, 1) on the effect scale")
        if not self.required_strata or any(not stratum for stratum in self.required_strata):
            raise ValueError("required strata must be non-empty")
        if len(self.required_strata) != len(set(self.required_strata)):
            raise ValueError("required strata must be unique")
        if not math.isfinite(self.max_stratum_regression) or self.max_stratum_regression < 0.0:
            raise ValueError("maximum stratum regression must be finite and non-negative")
        if self.required_artifact_loads < 2:
            raise ValueError("at least two artifact loads are required to establish determinism")
        if re.fullmatch(r"[0-9a-f]{64}", self.expected_artifact_digest) is None:
            raise ValueError("expected artifact digest must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class PromotionMetrics:
    """Measured candidate evidence supplied to the neutral evaluator."""

    total_games: int
    engine_errors: int
    illegal_choices: int
    agent_errors: int
    timeout_terminations: int
    max_step_terminations: int
    candidate_score: float
    champion_score: float
    decision_latencies_ms: Mapping[str, tuple[float, ...]]
    stratum_score_margins: Mapping[str, float]
    artifact_loads: tuple[ArtifactLoad, ...]

    def __post_init__(self) -> None:
        if self.total_games <= 0:
            raise ValueError("total games must be positive")
        counts = (
            self.engine_errors,
            self.illegal_choices,
            self.agent_errors,
            self.timeout_terminations,
            self.max_step_terminations,
        )
        if any(count < 0 for count in counts):
            raise ValueError("failure counts must be non-negative")
        if self.timeout_terminations + self.max_step_terminations > self.total_games:
            raise ValueError("timeout and max-step terminations cannot exceed total games")
        if not all(math.isfinite(score) for score in (self.candidate_score, self.champion_score)):
            raise ValueError("candidate and champion scores must be finite")
        if any(
            not math.isfinite(latency) or latency < 0.0
            for values in self.decision_latencies_ms.values()
            for latency in values
        ):
            raise ValueError("decision latencies must be finite and non-negative")
        if any(not math.isfinite(margin) for margin in self.stratum_score_margins.values()):
            raise ValueError("stratum score margins must be finite")


@dataclass(frozen=True, slots=True)
class GateCheck:
    """Evidence and outcome for one mandatory gate."""

    gate: str
    passed: bool
    actual: Any
    limit: Any
    message: str


@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    """Complete, non-short-circuiting result of gate evaluation."""

    promoted: bool
    checks: tuple[GateCheck, ...]

    @property
    def failures(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def _check(gate: str, passed: bool, actual: Any, limit: Any, message: str) -> GateCheck:
    return GateCheck(gate=gate, passed=passed, actual=actual, limit=limit, message=message)


def _nearest_rank(values: tuple[float, ...], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(percentile / 100.0 * len(ordered))
    return ordered[max(rank - 1, 0)]


def evaluate_promotion_gates(
    metrics: PromotionMetrics, config: PromotionGateConfig
) -> PromotionVerdict:
    """Evaluate every mandatory gate and retain every failure as evidence."""

    checks: list[GateCheck] = []
    for gate, count in (
        ("engine_errors", metrics.engine_errors),
        ("illegal_choices", metrics.illegal_choices),
        ("agent_errors", metrics.agent_errors),
    ):
        checks.append(_check(gate, count == 0, count, 0, f"{gate} must be zero"))

    failed_games = metrics.timeout_terminations + metrics.max_step_terminations
    rate = failed_games / metrics.total_games
    checks.append(
        _check(
            "timeout_or_max_step_rate",
            rate <= config.max_timeout_or_max_step_rate,
            rate,
            config.max_timeout_or_max_step_rate,
            "combined timeout and max-step rate must meet the declared SLO",
        )
    )

    for slo in config.latency_slos:
        actual = _nearest_rank(
            tuple(metrics.decision_latencies_ms.get(slo.tier, ())), slo.percentile
        )
        checks.append(
            _check(
                f"latency:{slo.tier}:p{slo.percentile:g}",
                actual is not None and actual <= slo.max_ms,
                actual,
                slo.max_ms,
                f"{slo.tier} p{slo.percentile:g} latency must meet its declared SLO",
            )
        )

    margin = metrics.candidate_score - metrics.champion_score
    checks.append(
        _check(
            "practical_margin",
            margin >= config.practical_margin,
            margin,
            config.practical_margin,
            "candidate score advantage must meet the declared practical margin",
        )
    )

    for stratum in config.required_strata:
        actual = metrics.stratum_score_margins.get(stratum)
        checks.append(
            _check(
                f"required_stratum:{stratum}",
                actual is not None and actual >= -config.max_stratum_regression,
                actual,
                -config.max_stratum_regression,
                f"required stratum {stratum!r} must be present and within regression tolerance",
            )
        )

    loads = metrics.artifact_loads
    successful_digests = tuple(load.digest for load in loads if load.error is None)
    load_passed = (
        len(loads) >= config.required_artifact_loads
        and all(load.error is None and load.digest for load in loads)
        and len(set(successful_digests)) == 1
        and successful_digests[0] == config.expected_artifact_digest
    )
    checks.append(
        _check(
            "deterministic_artifact_load",
            load_passed,
            tuple((load.digest, load.error) for load in loads),
            MappingProxyType(
                {
                    "required_loads": config.required_artifact_loads,
                    "distinct_digests": 1,
                    "expected_digest": config.expected_artifact_digest,
                }
            ),
            "every required artifact load must succeed with the same digest",
        )
    )

    frozen_checks = tuple(checks)
    return PromotionVerdict(
        promoted=all(check.passed for check in frozen_checks), checks=frozen_checks
    )
