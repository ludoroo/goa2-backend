"""Equal-budget, paired evaluation of shallow-search boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tqdm import tqdm


class SearchBoundary(StrEnum):
    """The four predeclared search horizons compared by evaluation."""

    POLICY_ONLY = "policy_only"
    IMMEDIATE_CONSEQUENCE = "immediate_consequence"
    ONE_OPPOSING_RESPONSE = "one_opposing_response"
    TWO_RESPONSE_CYCLES = "two_response_cycles"


class ReliabilityOutcome(StrEnum):
    """Mutually exclusive completion outcome reported by the runner."""

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    MAX_STEPS = "max_steps"
    ILLEGAL_ACTION = "illegal_action"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SearchBoundaryConfig:
    boundary: SearchBoundary
    wall_clock_budget_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, SearchBoundary):
            raise ValueError("boundary must be a SearchBoundary")
        if not math.isfinite(self.wall_clock_budget_seconds) or self.wall_clock_budget_seconds <= 0:
            raise ValueError("wall-clock budget must be finite and positive")


@dataclass(frozen=True, slots=True)
class BoundaryCase:
    case_id: str
    boundary: SearchBoundary
    world_seed: int
    candidate_side: str
    wall_clock_budget_seconds: float


@dataclass(frozen=True, slots=True)
class BoundaryMeasurement:
    """Strength, reliability, and latency evidence returned by an injected runner."""

    candidate_score: float
    reliability: ReliabilityOutcome
    decision_latencies_ms: tuple[float, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.reliability, ReliabilityOutcome):
            raise ValueError("reliability must be a ReliabilityOutcome")
        if not math.isfinite(self.candidate_score) or not 0.0 <= self.candidate_score <= 1.0:
            raise ValueError("candidate score must be finite and between zero and one")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed seconds must be finite and non-negative")
        if any(
            not math.isfinite(latency) or latency < 0.0 for latency in self.decision_latencies_ms
        ):
            raise ValueError("decision latencies must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BoundaryObservation:
    case_id: str
    boundary: SearchBoundary
    world_seed: int
    candidate_side: str
    candidate_score: float
    reliability: ReliabilityOutcome
    decision_latencies_ms: tuple[float, ...]
    elapsed_seconds: float

    @classmethod
    def from_case(cls, case: BoundaryCase, measurement: BoundaryMeasurement) -> BoundaryObservation:
        return cls(
            case_id=case.case_id,
            boundary=case.boundary,
            world_seed=case.world_seed,
            candidate_side=case.candidate_side,
            candidate_score=measurement.candidate_score,
            reliability=measurement.reliability,
            decision_latencies_ms=measurement.decision_latencies_ms,
            elapsed_seconds=measurement.elapsed_seconds,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "boundary": self.boundary.value,
            "candidate_score": self.candidate_score,
            "candidate_side": self.candidate_side,
            "case_id": self.case_id,
            "decision_latencies_ms": list(self.decision_latencies_ms),
            "elapsed_seconds": self.elapsed_seconds,
            "reliability": self.reliability.value,
            "world_seed": self.world_seed,
        }

    @classmethod
    def from_mapping(cls, value: object) -> BoundaryObservation:
        if not isinstance(value, dict):
            raise ValueError("boundary observation must be an object")
        required = {
            "boundary",
            "candidate_score",
            "candidate_side",
            "case_id",
            "decision_latencies_ms",
            "elapsed_seconds",
            "reliability",
            "world_seed",
        }
        if set(value) != required:
            raise ValueError("boundary observation has an invalid schema")
        raw_latencies = value["decision_latencies_ms"]
        if not isinstance(raw_latencies, list):
            raise ValueError("decision latencies must be a list")
        measurement = BoundaryMeasurement(
            candidate_score=float(value["candidate_score"]),
            reliability=ReliabilityOutcome(str(value["reliability"])),
            decision_latencies_ms=tuple(float(item) for item in raw_latencies),
            elapsed_seconds=float(value["elapsed_seconds"]),
        )
        return cls(
            case_id=str(value["case_id"]),
            boundary=SearchBoundary(str(value["boundary"])),
            world_seed=int(value["world_seed"]),
            candidate_side=str(value["candidate_side"]),
            candidate_score=measurement.candidate_score,
            reliability=measurement.reliability,
            decision_latencies_ms=measurement.decision_latencies_ms,
            elapsed_seconds=measurement.elapsed_seconds,
        )


@dataclass(frozen=True, slots=True)
class BoundaryEvidence:
    boundary: SearchBoundary
    case_count: int
    pair_count: int
    mean_paired_score: float | None
    reliability_rate: float
    decision_latencies_ms: tuple[float, ...]
    elapsed_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BoundaryBenchmarkResult:
    benchmark_digest: str
    expected_case_count: int
    observations: tuple[BoundaryObservation, ...]
    evidence: Mapping[SearchBoundary, BoundaryEvidence]

    @property
    def complete(self) -> bool:
        return len(self.observations) == self.expected_case_count


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """Predeclared confidence and practical-separation requirements."""

    confidence_level: float
    practical_margin: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must be finite and strictly between zero and one")
        if not math.isfinite(self.practical_margin) or self.practical_margin < 0.0:
            raise ValueError("practical margin must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BoundaryGateConfig:
    """Reliability and nearest-rank latency policy for boundary eligibility."""

    max_unreliable_rate: float
    latency_percentile: float
    max_latency_ms: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_unreliable_rate) or not (
            0.0 <= self.max_unreliable_rate <= 1.0
        ):
            raise ValueError("maximum unreliable rate must be finite and between zero and one")
        if not math.isfinite(self.latency_percentile) or not (
            0.0 < self.latency_percentile <= 100.0
        ):
            raise ValueError("latency percentile must be finite and in (0, 100]")
        if not math.isfinite(self.max_latency_ms) or self.max_latency_ms <= 0.0:
            raise ValueError("maximum latency must be finite and positive")


@dataclass(frozen=True, slots=True)
class RankedBoundary:
    rank: int
    boundary: SearchBoundary
    mean_paired_score: float
    confidence_lower: float
    confidence_upper: float
    reliability_rate: float
    latency_ms: float | None
    eligible: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundaryRanking:
    entries: tuple[RankedBoundary, ...]
    selected_boundary: SearchBoundary | None


@dataclass(frozen=True, slots=True)
class BoundaryBenchmarkConfig:
    """Complete experiment identity and equal-budget case declaration."""

    experiment_id: str
    world_seeds: tuple[int, ...]
    boundaries: tuple[SearchBoundaryConfig, ...]

    def __post_init__(self) -> None:
        if not self.experiment_id or self.experiment_id != self.experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty normalized string")
        if not self.world_seeds:
            raise ValueError("world_seeds must be non-empty")
        if any(type(seed) is not int or seed < 0 for seed in self.world_seeds):
            raise ValueError("world seeds must be non-negative integers")
        if len(set(self.world_seeds)) != len(self.world_seeds):
            raise ValueError("world seeds must be unique")
        declared = tuple(config.boundary for config in self.boundaries)
        if len(set(declared)) != len(declared) or set(declared) != set(SearchBoundary):
            raise ValueError("every search boundary must be configured exactly once")
        budgets = {config.wall_clock_budget_seconds for config in self.boundaries}
        if len(budgets) != 1:
            raise ValueError("all search boundaries must have an equal wall-clock budget")

    def _identity(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "world_seeds": sorted(self.world_seeds),
            "boundaries": [
                {
                    "boundary": config.boundary.value,
                    "wall_clock_budget_seconds": config.wall_clock_budget_seconds,
                }
                for config in sorted(self.boundaries, key=lambda item: item.boundary.value)
            ],
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_bytes(self._identity())).hexdigest()

    def cases(self) -> tuple[BoundaryCase, ...]:
        configs = {config.boundary: config for config in self.boundaries}
        cases: list[BoundaryCase] = []
        for seed in sorted(self.world_seeds):
            for side in ("RED", "BLUE"):
                for boundary in SearchBoundary:
                    config = configs[boundary]
                    identity = {
                        "benchmark_digest": self.digest,
                        "boundary": boundary.value,
                        "world_seed": seed,
                        "candidate_side": side,
                    }
                    case_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:24]
                    cases.append(
                        BoundaryCase(
                            case_id=case_id,
                            boundary=boundary,
                            world_seed=seed,
                            candidate_side=side,
                            wall_clock_budget_seconds=config.wall_clock_budget_seconds,
                        )
                    )
        return tuple(cases)


def _evidence(
    observations: tuple[BoundaryObservation, ...],
) -> dict[SearchBoundary, BoundaryEvidence]:
    result: dict[SearchBoundary, BoundaryEvidence] = {}
    for boundary in SearchBoundary:
        rows = tuple(row for row in observations if row.boundary is boundary)
        by_seed: dict[int, dict[str, float]] = {}
        for row in rows:
            by_seed.setdefault(row.world_seed, {})[row.candidate_side] = row.candidate_score
        # Unreliable-case scores stay in paired strength; reliability is gated separately.
        pairs = tuple(
            (scores["RED"] + scores["BLUE"]) / 2.0
            for _seed, scores in sorted(by_seed.items())
            if set(scores) == {"RED", "BLUE"}
        )
        result[boundary] = BoundaryEvidence(
            boundary=boundary,
            case_count=len(rows),
            pair_count=len(pairs),
            mean_paired_score=math.fsum(pairs) / len(pairs) if pairs else None,
            reliability_rate=(
                sum(row.reliability is ReliabilityOutcome.COMPLETED for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            decision_latencies_ms=tuple(
                latency for row in rows for latency in row.decision_latencies_ms
            ),
            elapsed_seconds=tuple(row.elapsed_seconds for row in rows),
        )
    return result


def _result(
    config: BoundaryBenchmarkConfig, observations: tuple[BoundaryObservation, ...]
) -> BoundaryBenchmarkResult:
    return BoundaryBenchmarkResult(
        benchmark_digest=config.digest,
        expected_case_count=len(config.cases()),
        observations=observations,
        evidence=_evidence(observations),
    )


def _payload(
    config: BoundaryBenchmarkConfig, observations: tuple[BoundaryObservation, ...]
) -> dict[str, object]:
    return {
        "benchmark_digest": config.digest,
        "observations": [observation.to_mapping() for observation in observations],
        "schema_version": 1,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _write_checkpoint(
    config: BoundaryBenchmarkConfig,
    path: Path,
    observations: tuple[BoundaryObservation, ...],
) -> None:
    _atomic_write(path, _canonical_bytes(_payload(config, observations)) + b"\n")


def load_boundary_benchmark(
    config: BoundaryBenchmarkConfig, checkpoint_path: Path
) -> BoundaryBenchmarkResult:
    """Load and validate a canonical partial or complete benchmark artifact."""

    path = Path(checkpoint_path)
    if not path.exists():
        return _result(config, ())
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("boundary checkpoint is not valid JSON") from exc
    if raw != _canonical_bytes(payload) + b"\n":
        raise ValueError("boundary checkpoint is not canonical JSON")
    if not isinstance(payload, dict) or set(payload) != {
        "benchmark_digest",
        "observations",
        "schema_version",
    }:
        raise ValueError("boundary checkpoint has an invalid schema")
    if payload["schema_version"] != 1 or payload["benchmark_digest"] != config.digest:
        raise ValueError("boundary checkpoint identity does not match the benchmark")
    raw_observations = payload["observations"]
    if not isinstance(raw_observations, list):
        raise ValueError("boundary checkpoint observations must be a list")
    observations = tuple(BoundaryObservation.from_mapping(item) for item in raw_observations)
    expected = config.cases()
    if len(observations) > len(expected):
        raise ValueError("boundary checkpoint contains too many observations")
    for observation, case in zip(observations, expected, strict=False):
        if (
            observation.case_id,
            observation.boundary,
            observation.world_seed,
            observation.candidate_side,
        ) != (case.case_id, case.boundary, case.world_seed, case.candidate_side):
            raise ValueError("boundary checkpoint does not follow the deterministic case schedule")
    return _result(config, observations)


def run_boundary_benchmark(
    config: BoundaryBenchmarkConfig,
    *,
    checkpoint_path: Path,
    run_case: Callable[[BoundaryCase], BoundaryMeasurement],
    show_progress: bool = False,
) -> BoundaryBenchmarkResult:
    """Run missing cases in order, atomically checkpointing each measurement."""

    loaded = load_boundary_benchmark(config, checkpoint_path)
    observations = list(loaded.observations)
    all_cases = config.cases()
    missing_cases = all_cases[len(observations) :]
    cases = (
        tqdm(
            missing_cases,
            desc="Search boundaries",
            initial=len(observations),
            total=len(all_cases),
            unit="case",
        )
        if show_progress
        else missing_cases
    )
    for case in cases:
        measurement = run_case(case)
        if not isinstance(measurement, BoundaryMeasurement):
            raise ValueError("run_case must return BoundaryMeasurement")
        observations.append(BoundaryObservation.from_case(case, measurement))
        _write_checkpoint(config, Path(checkpoint_path), tuple(observations))
    return _result(config, tuple(observations))


def _nearest_rank(values: tuple[float, ...], percentile: float) -> float | None:
    if not values:
        return None
    rank = math.ceil(percentile / 100.0 * len(values))
    return sorted(values)[rank - 1]


def rank_boundaries(
    result: BoundaryBenchmarkResult,
    *,
    confidence: ConfidenceConfig | None,
    gates: BoundaryGateConfig | None,
) -> BoundaryRanking:
    """Rank measured boundaries, selecting one only with credible separation.

    Ranking is forbidden for partial evidence or implicit policy. A boundary
    must meet both declared operational gates to be selectable. Selection then
    requires its Hoeffding lower bound to exceed every other eligible boundary's
    upper bound by the practical margin; inconclusive evidence selects nothing.
    """

    if not result.complete:
        raise ValueError("boundary evidence must be complete before ranking")
    if confidence is None:
        raise ValueError("confidence config is required before ranking")
    if gates is None:
        raise ValueError("gate config is required before ranking")

    provisional: list[RankedBoundary] = []
    for boundary in SearchBoundary:
        evidence = result.evidence[boundary]
        mean = evidence.mean_paired_score
        if mean is None or evidence.pair_count <= 0:
            raise ValueError("complete boundary evidence must contain paired scores")
        alpha = 1.0 - confidence.confidence_level
        radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * evidence.pair_count))
        latency = _nearest_rank(evidence.decision_latencies_ms, gates.latency_percentile)
        failures: list[str] = []
        if 1.0 - evidence.reliability_rate > gates.max_unreliable_rate:
            failures.append("reliability")
        if latency is None or latency > gates.max_latency_ms:
            failures.append("latency")
        provisional.append(
            RankedBoundary(
                rank=0,
                boundary=boundary,
                mean_paired_score=mean,
                confidence_lower=max(0.0, mean - radius),
                confidence_upper=min(1.0, mean + radius),
                reliability_rate=evidence.reliability_rate,
                latency_ms=latency,
                eligible=not failures,
                gate_failures=tuple(failures),
            )
        )

    ordered = sorted(
        provisional,
        key=lambda entry: (-entry.mean_paired_score, list(SearchBoundary).index(entry.boundary)),
    )
    entries = tuple(
        RankedBoundary(
            rank=index,
            boundary=entry.boundary,
            mean_paired_score=entry.mean_paired_score,
            confidence_lower=entry.confidence_lower,
            confidence_upper=entry.confidence_upper,
            reliability_rate=entry.reliability_rate,
            latency_ms=entry.latency_ms,
            eligible=entry.eligible,
            gate_failures=entry.gate_failures,
        )
        for index, entry in enumerate(ordered, start=1)
    )
    eligible = tuple(entry for entry in entries if entry.eligible)
    selected: SearchBoundary | None = None
    if eligible:
        strongest = eligible[0]
        competitors = eligible[1:]
        if all(
            strongest.confidence_lower > competitor.confidence_upper + confidence.practical_margin
            for competitor in competitors
        ):
            selected = strongest.boundary
    return BoundaryRanking(entries=entries, selected_boundary=selected)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


__all__ = [
    "BoundaryBenchmarkConfig",
    "BoundaryBenchmarkResult",
    "BoundaryCase",
    "BoundaryEvidence",
    "BoundaryGateConfig",
    "BoundaryMeasurement",
    "BoundaryObservation",
    "BoundaryRanking",
    "ConfidenceConfig",
    "RankedBoundary",
    "ReliabilityOutcome",
    "SearchBoundary",
    "SearchBoundaryConfig",
    "load_boundary_benchmark",
    "rank_boundaries",
    "run_boundary_benchmark",
]
