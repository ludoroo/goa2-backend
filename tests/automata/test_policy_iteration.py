"""Behavioral contract for end-to-end resumable policy iteration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from automata.evaluation.arena import ArenaResult, ArtifactIdentity
from automata.evaluation.promotion_gates import (
    ArtifactLoad,
    LatencySLO,
    PromotionGateConfig,
    PromotionMetrics,
    PromotionVerdict,
    evaluate_promotion_gates,
)
from automata.training.policy_iteration import (
    PolicyIterationStatus,
    run_policy_iteration,
)
from automata.training.registry import RegistryError

PARENT = "a" * 64
PARENT_ARTIFACT = "b" * 64
CANDIDATE = "c" * 64
CANDIDATE_ARTIFACT = "d" * 64


def _candidate() -> Any:
    return SimpleNamespace(
        digest=CANDIDATE,
        artifact_digest=CANDIDATE_ARTIFACT,
        generation=5,
        parent_champion_digest=PARENT,
        training_data_digests=("1" * 64,),
        source_revision="revision-7",
        source_tree_digest="e" * 64,
        search_config={"simulations": 64},
        training_config={"epochs": 3},
        offline_metrics={"loss": 0.2},
    )


class _Registry:
    def __init__(self) -> None:
        pointer = SimpleNamespace(
            digest="0" * 64,
            manifest_digest=PARENT,
            artifact_digest=PARENT_ARTIFACT,
            generation=4,
        )
        self.current: SimpleNamespace | None = SimpleNamespace(
            pointer=pointer, manifest=SimpleNamespace(digest=PARENT)
        )
        self.candidates: dict[str, Any] = {}
        self.promoted: list[str] = []
        self.rejected: list[tuple[str, dict[str, Any], Any]] = []

    def load_champion(self) -> Any:
        assert self.current is not None
        return self.current

    def champion_pointer(self) -> Any:
        return None if self.current is None else self.current.pointer

    def candidate(self, digest: str) -> Any:
        try:
            return self.candidates[digest]
        except KeyError as exc:
            raise RegistryError("missing candidate") from exc

    def promote(self, candidate_digest: str) -> Any:
        self.promoted.append(candidate_digest)
        candidate = self.candidates[candidate_digest]
        pointer = SimpleNamespace(
            digest="f" * 64,
            manifest_digest=candidate_digest,
            artifact_digest=candidate.artifact_digest,
            generation=candidate.generation,
        )
        self.current = SimpleNamespace(pointer=pointer, manifest=candidate)
        return pointer

    def finalize_candidate(self, candidate_digest: str, *, arena_results: dict[str, Any]) -> Any:
        candidate = self.candidates[candidate_digest]
        finalized = SimpleNamespace(**vars(candidate), arena_results=arena_results)
        finalized.digest = "8" * 64
        self.candidates[finalized.digest] = finalized
        return finalized

    def reject(self, candidate_digest: str, *, evidence: dict[str, Any]) -> Any:
        record = SimpleNamespace(digest="9" * 64, evidence=evidence)
        self.rejected.append((candidate_digest, evidence, record))
        return record

    def rejection(self, candidate_digest: str) -> Any:
        if not self.rejected:
            raise RegistryError("missing rejection")
        assert self.rejected[-1][0] == candidate_digest
        return self.rejected[-1][2]


class _Pipeline:
    pipeline_digest = "2" * 64

    def __init__(self, registry: _Registry) -> None:
        self.registry = registry
        self.calls = 0

    def run(self) -> Any:
        self.calls += 1
        candidate = _candidate()
        self.registry.candidates[candidate.digest] = candidate
        return candidate


def _gates(artifact_digest: str = CANDIDATE_ARTIFACT) -> PromotionGateConfig:
    return PromotionGateConfig(
        max_timeout_or_max_step_rate=0.0,
        latency_slos=(LatencySLO("standard", 100.0, 10.0),),
        practical_margin=0.0,
        required_strata=("all",),
        max_stratum_regression=0.0,
        required_artifact_loads=2,
        expected_artifact_digest=artifact_digest,
    )


def _metrics(*, agent_errors: int = 0) -> PromotionMetrics:
    return PromotionMetrics(
        total_games=2,
        engine_errors=0,
        illegal_choices=0,
        agent_errors=agent_errors,
        timeout_terminations=0,
        max_step_terminations=0,
        candidate_score=1.0,
        champion_score=0.0,
        decision_latencies_ms={"standard": (1.0,)},
        stratum_score_margins={"all": 1.0},
        artifact_loads=(ArtifactLoad(CANDIDATE_ARTIFACT, None),) * 2,
    )


def _arena_result(
    *, metrics: PromotionMetrics | None = None, verdict: PromotionVerdict | None = None
) -> ArenaResult:
    metrics = metrics or _metrics()
    verdict = verdict or evaluate_promotion_gates(metrics, _gates())
    return ArenaResult(
        candidate=ArtifactIdentity(CANDIDATE, CANDIDATE_ARTIFACT),
        champion=ArtifactIdentity(PARENT, PARENT_ARTIFACT),
        stages=(),
        promotion_gate_config=_gates(),
        promotion_metrics=metrics,
        promotion_gates=verdict,
        promoted=verdict.promoted,
    )


def _run(
    tmp_path: Path,
    registry: _Registry,
    pipeline: Any,
    arena: Any,
    *,
    stage_hook: Any = None,
) -> Any:
    return run_policy_iteration(
        work_dir=tmp_path / "iteration",
        registry=registry,
        generation_pipeline=pipeline,
        arena_runner=arena,
        promotion_gate_config=_gates(),
        stage_hook=stage_hook,
    )


def test_pipeline_candidate_is_evaluated_against_its_exact_parent_and_promoted(
    tmp_path: Path,
) -> None:
    registry = _Registry()
    pipeline = _Pipeline(registry)
    arena_calls: list[tuple[Any, Any, Path]] = []

    def arena(candidate: Any, champion: Any, stage_dir: Path) -> ArenaResult:
        arena_calls.append((candidate, champion, stage_dir))
        return _arena_result()

    result = _run(tmp_path, registry, pipeline, arena)

    assert result.status is PolicyIterationStatus.PROMOTED
    assert pipeline.calls == 1
    assert arena_calls[0][0].digest == CANDIDATE
    assert arena_calls[0][1].pointer.manifest_digest == PARENT
    assert result.manifest.generation == 5
    assert result.manifest.parent_champion_digest == PARENT
    assert result.manifest.pipeline_digest == pipeline.pipeline_digest
    assert result.manifest.promotion_gate_config_digest
    assert result.manifest.promotion_gate_config["expected_artifact_digest"] == CANDIDATE_ARTIFACT
    assert result.candidate.arena_results["arena_result"]["promotion_metrics"]["total_games"] == 2
    assert registry.promoted == [result.candidate.digest]


def test_any_failed_gate_rejects_with_complete_arena_evidence(tmp_path: Path) -> None:
    registry = _Registry()
    metrics = _metrics(agent_errors=1)
    verdict = evaluate_promotion_gates(metrics, _gates())

    result = _run(
        tmp_path,
        registry,
        _Pipeline(registry),
        lambda candidate, champion, stage_dir: _arena_result(metrics=metrics, verdict=verdict),
    )

    assert result.status is PolicyIterationStatus.REJECTED
    assert registry.promoted == []
    _, evidence, _ = registry.rejected[0]
    assert evidence["arena_result"]["promotion_metrics"]["agent_errors"] == 1
    assert any(not check["passed"] for check in evidence["promotion_verdict"]["checks"])


@pytest.mark.parametrize("crash_stage", ["generation", "training"])
def test_generation_pipeline_can_resume_internal_generation_and_training_crashes(
    tmp_path: Path, crash_stage: str
) -> None:
    registry = _Registry()

    class ResumablePipeline(_Pipeline):
        def __init__(self, registry: _Registry) -> None:
            super().__init__(registry)
            self.completed: set[str] = set()
            self.attempts = {"generation": 0, "training": 0}
            self.crashed = False

        def run(self) -> Any:
            for stage in ("generation", "training"):
                if stage in self.completed:
                    continue
                self.attempts[stage] += 1
                self.completed.add(stage)
                if stage == crash_stage and not self.crashed:
                    self.crashed = True
                    raise RuntimeError(f"crash after {stage}")
            return super().run()

    pipeline = ResumablePipeline(registry)

    def arena(candidate: Any, champion: Any, stage_dir: Path) -> ArenaResult:
        return _arena_result()

    with pytest.raises(RuntimeError, match=f"crash after {crash_stage}"):
        _run(tmp_path, registry, pipeline, arena)

    assert _run(tmp_path, registry, pipeline, arena).status is PolicyIterationStatus.PROMOTED
    assert pipeline.attempts == {"generation": 1, "training": 1}


def test_arena_journal_is_canonical_and_resume_does_not_rerun_arena(tmp_path: Path) -> None:
    registry = _Registry()
    pipeline = _Pipeline(registry)
    arena_calls = 0

    def arena(candidate: Any, champion: Any, stage_dir: Path) -> ArenaResult:
        nonlocal arena_calls
        arena_calls += 1
        return _arena_result()

    def crash(stage: str) -> None:
        if stage == "arena":
            raise RuntimeError("crash after arena")

    with pytest.raises(RuntimeError, match="crash after arena"):
        _run(tmp_path, registry, pipeline, arena, stage_hook=crash)
    assert _run(tmp_path, registry, pipeline, arena).status is PolicyIterationStatus.PROMOTED
    assert pipeline.calls == 1
    assert arena_calls == 1

    for name in ("generation-manifest.json", "stage-journal.json"):
        raw = (tmp_path / "iteration" / name).read_bytes()
        assert (
            raw
            == json.dumps(
                json.loads(raw),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )


def test_resume_recovers_pointer_published_before_journal_update(tmp_path: Path) -> None:
    class CrashAfterPointerRegistry(_Registry):
        crashed = False

        def promote(self, candidate_digest: str) -> Any:
            pointer = super().promote(candidate_digest)
            if not self.crashed:
                self.crashed = True
                raise RuntimeError("crash after pointer")
            return pointer

    registry = CrashAfterPointerRegistry()
    pipeline = _Pipeline(registry)
    arena_calls = 0

    def arena(candidate: Any, champion: Any, stage_dir: Path) -> ArenaResult:
        nonlocal arena_calls
        arena_calls += 1
        return _arena_result()

    with pytest.raises(RuntimeError, match="crash after pointer"):
        _run(tmp_path, registry, pipeline, arena)

    assert _run(tmp_path, registry, pipeline, arena).status is PolicyIterationStatus.PROMOTED
    assert registry.promoted == ["8" * 64]
    assert pipeline.calls == 1
    assert arena_calls == 1


def test_wrong_generation_or_parent_fails_before_arena_or_pointer(tmp_path: Path) -> None:
    registry = _Registry()

    class WrongPipeline(_Pipeline):
        def run(self) -> Any:
            candidate = _candidate()
            candidate.generation = 6
            self.registry.candidates[candidate.digest] = candidate
            return candidate

    arena_calls = 0

    def arena(candidate: Any, champion: Any, stage_dir: Path) -> ArenaResult:
        nonlocal arena_calls
        arena_calls += 1
        return _arena_result()

    with pytest.raises(ValueError, match=r"parent|generation"):
        _run(tmp_path, registry, WrongPipeline(registry), arena)
    assert arena_calls == 0
    assert registry.promoted == []


def test_uninitialized_registry_bootstraps_generation_zero_then_runs_generation_one(
    tmp_path: Path,
) -> None:
    registry = _Registry()
    registry.current = None
    genesis = _candidate()
    genesis.digest = PARENT
    genesis.artifact_digest = PARENT_ARTIFACT
    genesis.generation = 0
    genesis.parent_champion_digest = None
    registry.candidates[PARENT] = genesis
    candidate = _candidate()
    candidate.generation = 1

    class FirstPipeline(_Pipeline):
        def run(self) -> Any:
            self.calls += 1
            self.registry.candidates[candidate.digest] = candidate
            return candidate

    initialized = 0

    def initialize(target: Any) -> None:
        nonlocal initialized
        initialized += 1
        pointer = SimpleNamespace(
            digest="0" * 64,
            manifest_digest=PARENT,
            artifact_digest=PARENT_ARTIFACT,
            generation=0,
            previous_manifest_digest=None,
        )
        target.current = SimpleNamespace(pointer=pointer, manifest=genesis)

    pipeline = FirstPipeline(registry)
    result = run_policy_iteration(
        work_dir=tmp_path / "iteration",
        registry=registry,
        generation_pipeline=pipeline,
        arena_runner=lambda candidate, champion, stage_dir: _arena_result(),
        promotion_gate_config=_gates(),
        initialize_champion=initialize,
    )
    resumed = run_policy_iteration(
        work_dir=tmp_path / "iteration",
        registry=registry,
        generation_pipeline=pipeline,
        arena_runner=lambda candidate, champion, stage_dir: _arena_result(),
        promotion_gate_config=_gates(),
        initialize_champion=initialize,
    )

    assert result.manifest.generation == 1
    assert resumed.manifest == result.manifest
    assert result.manifest.parent_champion_digest == PARENT
    assert initialized == 1


def test_inconsistent_arena_verdict_fails_closed(tmp_path: Path) -> None:
    registry = _Registry()
    metrics = _metrics(agent_errors=1)
    forged = _arena_result(metrics=metrics, verdict=evaluate_promotion_gates(_metrics(), _gates()))

    with pytest.raises(ValueError, match="verdict"):
        _run(tmp_path, registry, _Pipeline(registry), lambda *args: forged)

    assert registry.promoted == []
