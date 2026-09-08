"""Behavioral tests for equal-budget search-boundary benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from automata.evaluation.search_boundaries import (
    BoundaryBenchmarkConfig,
    BoundaryCase,
    BoundaryGateConfig,
    BoundaryMeasurement,
    ConfidenceConfig,
    ReliabilityOutcome,
    SearchBoundary,
    SearchBoundaryConfig,
    load_boundary_benchmark,
    rank_boundaries,
    run_boundary_benchmark,
)


def _config(**changes: object) -> BoundaryBenchmarkConfig:
    values: dict[str, Any] = {
        "experiment_id": "artifact-a-vs-baseline",
        "world_seeds": (12, 10),
        "boundaries": tuple(
            SearchBoundaryConfig(boundary=boundary, wall_clock_budget_seconds=0.25)
            for boundary in SearchBoundary
        ),
    }
    values.update(changes)
    return BoundaryBenchmarkConfig(**values)


def _canonical_checkpoint(payload: object) -> bytes:
    return (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )


def _valid_checkpoint(tmp_path: Path) -> tuple[BoundaryBenchmarkConfig, Path, dict[str, Any]]:
    config = _config(world_seeds=(10,))
    checkpoint = tmp_path / "boundaries.json"
    run_boundary_benchmark(config, checkpoint_path=checkpoint, run_case=_FakeRunner())
    return config, checkpoint, json.loads(checkpoint.read_bytes())


def test_schedule_covers_every_boundary_with_paired_deterministic_cases() -> None:
    config = _config()

    cases = config.cases()

    assert tuple(SearchBoundary) == (
        SearchBoundary.POLICY_ONLY,
        SearchBoundary.IMMEDIATE_CONSEQUENCE,
        SearchBoundary.ONE_OPPOSING_RESPONSE,
        SearchBoundary.TWO_RESPONSE_CYCLES,
    )
    assert len(cases) == 16
    assert [(case.world_seed, case.candidate_side) for case in cases[:8]] == [
        (10, "RED"),
        (10, "RED"),
        (10, "RED"),
        (10, "RED"),
        (10, "BLUE"),
        (10, "BLUE"),
        (10, "BLUE"),
        (10, "BLUE"),
    ]
    assert {case.boundary for case in cases} == set(SearchBoundary)
    assert {case.wall_clock_budget_seconds for case in cases} == {0.25}
    assert len({case.case_id for case in cases}) == len(cases)
    assert cases == _config(world_seeds=(10, 12)).cases()


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, case: BoundaryCase) -> BoundaryMeasurement:
        self.calls.append(case.case_id)
        score = 1.0 if case.candidate_side == "RED" else 0.5
        return BoundaryMeasurement(
            candidate_score=score,
            reliability=ReliabilityOutcome.COMPLETED,
            decision_latencies_ms=(10.0, 20.0 + list(SearchBoundary).index(case.boundary)),
            elapsed_seconds=0.2,
        )


def test_runner_collects_strength_reliability_and_latency_in_canonical_result(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "boundaries.json"

    result = run_boundary_benchmark(_config(), checkpoint_path=checkpoint, run_case=_FakeRunner())

    assert result.complete is True
    assert len(result.observations) == 16
    assert result.evidence[SearchBoundary.POLICY_ONLY].mean_paired_score == 0.75
    assert result.evidence[SearchBoundary.POLICY_ONLY].reliability_rate == 1.0
    assert result.evidence[SearchBoundary.POLICY_ONLY].decision_latencies_ms == (
        10.0,
        20.0,
        10.0,
        20.0,
        10.0,
        20.0,
        10.0,
        20.0,
    )
    raw = checkpoint.read_bytes()
    assert raw.endswith(b"\n")
    assert (
        json.dumps(
            json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
        == raw
    )


def test_config_rejects_unequal_wall_clock_budgets() -> None:
    unequal = tuple(
        SearchBoundaryConfig(
            boundary=boundary,
            wall_clock_budget_seconds=(
                0.5 if boundary is SearchBoundary.TWO_RESPONSE_CYCLES else 0.25
            ),
        )
        for boundary in SearchBoundary
    )

    with pytest.raises(ValueError, match="equal wall-clock"):
        _config(boundaries=unequal)


def test_interrupted_run_resumes_only_missing_cases(tmp_path: Path) -> None:
    checkpoint = tmp_path / "boundaries.json"
    first = _FakeRunner()

    def interrupt(case: BoundaryCase) -> BoundaryMeasurement:
        if len(first.calls) == 5:
            raise RuntimeError("stop")
        return first(case)

    with pytest.raises(RuntimeError, match="stop"):
        run_boundary_benchmark(_config(), checkpoint_path=checkpoint, run_case=interrupt)
    assert len(load_boundary_benchmark(_config(), checkpoint).observations) == 5

    resumed = _FakeRunner()
    result = run_boundary_benchmark(_config(), checkpoint_path=checkpoint, run_case=resumed)

    assert result.complete is True
    assert len(resumed.calls) == 11
    assert set(first.calls).isdisjoint(resumed.calls)


def test_load_rejects_noncanonical_bytes(tmp_path: Path) -> None:
    config, checkpoint, payload = _valid_checkpoint(tmp_path)
    noncanonical = json.dumps(payload, indent=2).encode("utf-8")
    checkpoint.write_bytes(noncanonical)

    with pytest.raises(ValueError, match="not canonical JSON"):
        load_boundary_benchmark(config, checkpoint)

    assert checkpoint.read_bytes() == noncanonical


def test_load_rejects_wrong_config_and_digest_without_altering_checkpoint(
    tmp_path: Path,
) -> None:
    config, checkpoint, payload = _valid_checkpoint(tmp_path)
    valid_bytes = checkpoint.read_bytes()
    wrong_config = _config(experiment_id="different-experiment", world_seeds=(10,))

    with pytest.raises(ValueError, match="identity does not match"):
        load_boundary_benchmark(wrong_config, checkpoint)

    assert checkpoint.read_bytes() == valid_bytes
    assert load_boundary_benchmark(config, checkpoint).complete is True

    payload["benchmark_digest"] = "0" * 64
    wrong_digest = _canonical_checkpoint(payload)
    checkpoint.write_bytes(wrong_digest)
    with pytest.raises(ValueError, match="identity does not match"):
        load_boundary_benchmark(config, checkpoint)
    assert checkpoint.read_bytes() == wrong_digest


def test_load_rejects_too_many_observations(tmp_path: Path) -> None:
    config, checkpoint, payload = _valid_checkpoint(tmp_path)
    payload["observations"].append(payload["observations"][0])
    too_many = _canonical_checkpoint(payload)
    checkpoint.write_bytes(too_many)

    with pytest.raises(ValueError, match="too many observations"):
        load_boundary_benchmark(config, checkpoint)

    assert checkpoint.read_bytes() == too_many


def test_load_rejects_observations_outside_the_case_schedule(tmp_path: Path) -> None:
    config, checkpoint, payload = _valid_checkpoint(tmp_path)
    payload["observations"][0], payload["observations"][1] = (
        payload["observations"][1],
        payload["observations"][0],
    )
    out_of_schedule = _canonical_checkpoint(payload)
    checkpoint.write_bytes(out_of_schedule)

    with pytest.raises(ValueError, match="deterministic case schedule"):
        load_boundary_benchmark(config, checkpoint)

    assert checkpoint.read_bytes() == out_of_schedule


def test_ranking_requires_complete_evidence_and_declared_confidence_and_gates(
    tmp_path: Path,
) -> None:
    config = _config(world_seeds=tuple(range(8)))
    checkpoint = tmp_path / "boundaries.json"
    confidence = ConfidenceConfig(confidence_level=0.5, practical_margin=0.05)
    gates = BoundaryGateConfig(
        max_unreliable_rate=0.0,
        latency_percentile=95.0,
        max_latency_ms=50.0,
    )

    partial = load_boundary_benchmark(config, checkpoint)
    with pytest.raises(ValueError, match="complete"):
        rank_boundaries(partial, confidence=confidence, gates=gates)

    scores = {
        SearchBoundary.POLICY_ONLY: 0.1,
        SearchBoundary.IMMEDIATE_CONSEQUENCE: 0.2,
        SearchBoundary.ONE_OPPOSING_RESPONSE: 0.9,
        SearchBoundary.TWO_RESPONSE_CYCLES: 1.0,
    }

    def measured(case: BoundaryCase) -> BoundaryMeasurement:
        return BoundaryMeasurement(
            candidate_score=scores[case.boundary],
            reliability=(
                ReliabilityOutcome.TIMEOUT
                if case.boundary is SearchBoundary.TWO_RESPONSE_CYCLES
                else ReliabilityOutcome.COMPLETED
            ),
            decision_latencies_ms=(
                75.0 if case.boundary is SearchBoundary.IMMEDIATE_CONSEQUENCE else 25.0,
            ),
            elapsed_seconds=0.2,
        )

    complete = run_boundary_benchmark(config, checkpoint_path=checkpoint, run_case=measured)
    with pytest.raises(ValueError, match="confidence"):
        rank_boundaries(complete, confidence=None, gates=gates)
    with pytest.raises(ValueError, match="gate"):
        rank_boundaries(complete, confidence=confidence, gates=None)

    ranking = rank_boundaries(complete, confidence=confidence, gates=gates)

    assert [entry.boundary for entry in ranking.entries] == [
        SearchBoundary.TWO_RESPONSE_CYCLES,
        SearchBoundary.ONE_OPPOSING_RESPONSE,
        SearchBoundary.IMMEDIATE_CONSEQUENCE,
        SearchBoundary.POLICY_ONLY,
    ]
    assert ranking.entries[0].eligible is False
    assert ranking.entries[0].gate_failures == ("reliability",)
    immediate = next(
        entry for entry in ranking.entries if entry.boundary is SearchBoundary.IMMEDIATE_CONSEQUENCE
    )
    assert immediate.gate_failures == ("latency",)
    assert ranking.selected_boundary is SearchBoundary.ONE_OPPOSING_RESPONSE

    inconclusive = rank_boundaries(
        complete,
        confidence=ConfidenceConfig(confidence_level=0.95, practical_margin=0.05),
        gates=gates,
    )
    assert inconclusive.selected_boundary is None
