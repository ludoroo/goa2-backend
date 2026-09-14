"""Crash-safe composition of candidate generation, arena, and promotion."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from automata.evaluation.arena import ArenaResult, ArtifactIdentity
from automata.evaluation.promotion_gates import (
    ArtifactLoad,
    GateCheck,
    LatencySLO,
    PromotionGateConfig,
    PromotionMetrics,
    PromotionVerdict,
    evaluate_promotion_gates,
)
from automata.training.io import atomic_write_bytes as _atomic_write
from automata.training.io import canonical_json_bytes as _canonical
from automata.training.io import content_digest as _digest
from automata.training.registry import (
    CandidateManifest,
    ChampionPointer,
    LoadedChampion,
    RegistryError,
    RejectionRecord,
)


def _json_value(value: Any) -> JsonValue:
    return json.loads(_canonical(value))


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GenerationManifest(_FrozenModel):
    """Canonical handoff from the generation pipeline to evaluation."""

    schema_version: Literal[1] = 1
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    pipeline_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(ge=1)
    parent_champion_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_data_digests: tuple[str, ...]
    source_revision: str
    source_tree_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_config: dict[str, JsonValue]
    training_config: dict[str, JsonValue]
    offline_metrics: dict[str, JsonValue]
    promotion_gate_config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    promotion_gate_config: dict[str, JsonValue]

    @model_validator(mode="after")
    def _validate_digest(self) -> GenerationManifest:
        if self.digest != _digest(self.model_dump(mode="json", exclude={"digest"})):
            raise ValueError("generation manifest digest mismatch")
        return self


class PolicyIterationStatus(StrEnum):
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class PolicyIterationResult:
    status: PolicyIterationStatus
    manifest: GenerationManifest
    candidate: CandidateManifest
    pointer: ChampionPointer | None = None
    rejection: RejectionRecord | None = None


class CandidateGenerationPipeline(Protocol):
    """The existing resumable generation-to-candidate pipeline boundary."""

    pipeline_digest: str

    def run(self) -> CandidateManifest: ...


class ArenaRunner(Protocol):
    def __call__(
        self,
        candidate: CandidateManifest,
        champion: LoadedChampion,
        stage_dir: Path,
    ) -> ArenaResult: ...


class PolicyRegistry(Protocol):
    """ChampionRegistry surface consumed by this coordinator."""

    def load_champion(self) -> Any: ...

    def champion_pointer(self) -> Any: ...

    def candidate(self, digest: str) -> Any: ...

    def promote(self, candidate_digest: str) -> Any: ...

    def finalize_candidate(
        self, candidate_digest: str, *, arena_results: dict[str, JsonValue]
    ) -> Any: ...

    def reject(self, candidate_digest: str, *, evidence: dict[str, JsonValue]) -> Any: ...

    def rejection(self, candidate_digest: str) -> Any: ...


StageHook = Callable[[str], None]
InitializeChampion = Callable[[PolicyRegistry], None]


def _generation_manifest(
    pipeline_digest: str,
    candidate: CandidateManifest,
    parent_artifact_digest: str,
    promotion_gate_config: PromotionGateConfig,
) -> GenerationManifest:
    if promotion_gate_config.expected_artifact_digest != candidate.artifact_digest:
        raise ValueError("promotion gate config must identify the generated candidate artifact")
    gate_config = _object(_json_value(promotion_gate_config), "promotion gate config")
    values: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_digest": pipeline_digest,
        "candidate_digest": candidate.digest,
        "artifact_digest": candidate.artifact_digest,
        "generation": candidate.generation,
        "parent_champion_digest": candidate.parent_champion_digest,
        "parent_artifact_digest": parent_artifact_digest,
        "training_data_digests": candidate.training_data_digests,
        "source_revision": candidate.source_revision,
        "source_tree_digest": candidate.source_tree_digest,
        "search_config": candidate.search_config,
        "training_config": candidate.training_config,
        "offline_metrics": candidate.offline_metrics,
        "promotion_gate_config_digest": _digest(gate_config),
        "promotion_gate_config": gate_config,
    }
    return GenerationManifest(digest=_digest(values), **values)


def _read_canonical(path: Path) -> Any:
    payload = path.read_bytes()
    value = json.loads(payload)
    if payload != _canonical(value):
        raise ValueError(f"{path.name} is not canonical JSON")
    return value


def _load_or_run_generation(
    path: Path,
    pipeline: CandidateGenerationPipeline,
    registry: PolicyRegistry,
    parent: ChampionPointer,
    promotion_gate_config: PromotionGateConfig,
) -> tuple[GenerationManifest, CandidateManifest]:
    if path.exists():
        manifest = GenerationManifest.model_validate(_read_canonical(path))
        if manifest.pipeline_digest != pipeline.pipeline_digest:
            raise ValueError("generation manifest belongs to a different pipeline")
        candidate = registry.candidate(manifest.candidate_digest)
        expected = _generation_manifest(
            pipeline.pipeline_digest,
            candidate,
            manifest.parent_artifact_digest,
            promotion_gate_config,
        )
        if expected != manifest:
            raise ValueError("generation manifest and registered candidate disagree")
        return manifest, candidate

    candidate = pipeline.run()
    if (
        candidate.parent_champion_digest != parent.manifest_digest
        or candidate.generation != parent.generation + 1
    ):
        raise ValueError("candidate parent or generation does not match the exact champion")
    manifest = _generation_manifest(
        pipeline.pipeline_digest, candidate, parent.artifact_digest, promotion_gate_config
    )
    _atomic_write(path, _canonical(manifest))
    return manifest, candidate


class _StageJournal:
    def __init__(self, path: Path, manifest_digest: str) -> None:
        self.path = path
        self.manifest_digest = manifest_digest
        self.stages: dict[str, dict[str, JsonValue]] = {}
        if not path.exists():
            return
        raw = _read_canonical(path)
        if not isinstance(raw, dict):
            raise ValueError("stage journal must be a JSON object")
        body = {key: value for key, value in raw.items() if key != "digest"}
        if raw.get("digest") != _digest(body):
            raise ValueError("stage journal digest mismatch")
        if raw.get("schema_version") != 1 or raw.get("manifest_digest") != manifest_digest:
            raise ValueError("stage journal does not match generation manifest")
        stages = raw.get("stages")
        if not isinstance(stages, dict) or any(
            not isinstance(name, str) or not isinstance(payload, dict)
            for name, payload in stages.items()
        ):
            raise ValueError("stage journal stages must be an object of objects")
        if set(stages) - {"arena", "promoted", "rejected"}:
            raise ValueError("stage journal contains an unknown stage")
        terminals = {"promoted", "rejected"} & stages.keys()
        if len(terminals) > 1 or (terminals and "arena" not in stages):
            raise ValueError("stage journal has an invalid terminal transition")
        self.stages = stages

    def record(self, stage: str, payload: Mapping[str, JsonValue]) -> None:
        value = dict(payload)
        if existing := self.stages.get(stage):
            if existing != value:
                raise ValueError(f"completed stage {stage!r} has conflicting evidence")
            return
        self.stages[stage] = value
        body = {
            "schema_version": 1,
            "manifest_digest": self.manifest_digest,
            "stages": self.stages,
        }
        _atomic_write(self.path, _canonical({**body, "digest": _digest(body)}))


def _object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"journal {label} must be an object")
    return value


def _verdict_from(value: JsonValue) -> PromotionVerdict:
    verdict = _object(value, "verdict")
    promoted = verdict["promoted"]
    raw_checks = verdict["checks"]
    if not isinstance(promoted, bool) or not isinstance(raw_checks, list):
        raise ValueError("journal verdict has invalid types")
    checks: list[GateCheck] = []
    for raw_check in raw_checks:
        check = _object(raw_check, "gate check")
        if not isinstance(check["gate"], str) or not isinstance(check["passed"], bool):
            raise ValueError("journal gate check has invalid types")
        if not isinstance(check["message"], str):
            raise ValueError("journal gate message must be a string")
        checks.append(
            GateCheck(
                gate=check["gate"],
                passed=check["passed"],
                actual=check["actual"],
                limit=check["limit"],
                message=check["message"],
            )
        )
    return PromotionVerdict(promoted=promoted, checks=tuple(checks))


def _gate_config_from(value: JsonValue) -> PromotionGateConfig:
    config = _object(value, "promotion gate config")
    raw_slos = config.get("latency_slos")
    if not isinstance(raw_slos, list):
        raise ValueError("journal promotion gate latency SLOs must be a list")
    values: dict[str, Any] = dict(config)
    values["latency_slos"] = tuple(
        LatencySLO(**cast(Any, _object(item, "latency SLO"))) for item in raw_slos
    )
    values["required_strata"] = tuple(cast(list[str], config["required_strata"]))
    return PromotionGateConfig(**cast(Any, values))


def _metrics_from(value: JsonValue) -> PromotionMetrics:
    metrics = _object(value, "promotion metrics")
    raw_loads = metrics.get("artifact_loads")
    if not isinstance(raw_loads, list):
        raise ValueError("journal artifact loads must be a list")
    values: dict[str, Any] = dict(metrics)
    values["decision_latencies_ms"] = {
        key: tuple(cast(list[float], samples))
        for key, samples in _object(metrics["decision_latencies_ms"], "latencies").items()
    }
    values["artifact_loads"] = tuple(
        ArtifactLoad(**cast(Any, _object(item, "artifact load"))) for item in raw_loads
    )
    return PromotionMetrics(**cast(Any, values))


def _validate_arena_result(
    arena: ArenaResult,
    *,
    candidate: CandidateManifest,
    parent_manifest_digest: str,
    parent_artifact_digest: str,
    promotion_gate_config: PromotionGateConfig,
) -> tuple[PromotionVerdict | None, bool]:
    expected_candidate = ArtifactIdentity(candidate.digest, candidate.artifact_digest)
    expected_champion = ArtifactIdentity(parent_manifest_digest, parent_artifact_digest)
    if arena.candidate != expected_candidate or arena.champion != expected_champion:
        raise ValueError("arena contestants do not match candidate and exact champion")
    if arena.promotion_gate_config != promotion_gate_config:
        raise ValueError("arena promotion gate config disagrees with declared config")
    if arena.promotion_gates is None or arena.promotion_metrics is None:
        if (
            arena.promotion_gates is not None
            or arena.promotion_metrics is not None
            or arena.promoted
        ):
            raise ValueError("arena early verdict is inconsistent")
        return None, False
    expected = evaluate_promotion_gates(arena.promotion_metrics, promotion_gate_config)
    if _canonical(arena.promotion_gates) != _canonical(expected) or (
        arena.promoted and not expected.promoted
    ):
        raise ValueError("arena promotion verdict is inconsistent with gate metrics")
    return expected, arena.promoted


def _arena_authority_from(payload: Mapping[str, JsonValue]) -> ArenaResult:
    result = _object(payload["arena_result"], "arena result")
    candidate = _object(result["candidate"], "candidate identity")
    champion = _object(result["champion"], "champion identity")
    raw_metrics = result.get("promotion_metrics")
    raw_verdict = result.get("promotion_gates")
    promoted = result.get("promoted")
    if not isinstance(promoted, bool):
        raise ValueError("journal arena promoted flag must be boolean")
    return ArenaResult(
        candidate=ArtifactIdentity(**cast(Any, candidate)),
        champion=ArtifactIdentity(**cast(Any, champion)),
        stages=(),
        promotion_gate_config=_gate_config_from(result["promotion_gate_config"]),
        promotion_metrics=None if raw_metrics is None else _metrics_from(raw_metrics),
        promotion_gates=None if raw_verdict is None else _verdict_from(raw_verdict),
        promoted=promoted,
    )


def run_policy_iteration(
    *,
    work_dir: str | Path,
    registry: PolicyRegistry,
    generation_pipeline: CandidateGenerationPipeline,
    arena_runner: ArenaRunner,
    promotion_gate_config: PromotionGateConfig,
    initialize_champion: InitializeChampion | None = None,
    stage_hook: StageHook | None = None,
) -> PolicyIterationResult:
    """Run or resume one exact generation and promote only when every gate passes."""

    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    if registry.champion_pointer() is None:
        if initialize_champion is None:
            raise RegistryError("champion is not initialized and no initializer was provided")
        initialize_champion(registry)
        initialized = registry.champion_pointer()
        if (
            initialized is None
            or initialized.generation != 0
            or initialized.previous_manifest_digest is not None
        ):
            raise ValueError("champion initializer must promote one generation 0 genesis candidate")
    current_champion = registry.load_champion()
    manifest, candidate = _load_or_run_generation(
        root / "generation-manifest.json",
        generation_pipeline,
        registry,
        current_champion.pointer,
        promotion_gate_config,
    )
    journal = _StageJournal(root / "stage-journal.json", manifest.digest)

    current = registry.champion_pointer()
    if current is None:
        raise ValueError("current champion disappeared during policy iteration")
    if current.manifest_digest not in {manifest.parent_champion_digest, candidate.digest}:
        published = registry.candidate(current.manifest_digest)
        if (
            published.artifact_digest != candidate.artifact_digest
            or published.generation != candidate.generation
            or published.parent_champion_digest != candidate.parent_champion_digest
            or not published.arena_results
        ):
            raise ValueError("current champion is neither the generation parent nor its candidate")
    if current.manifest_digest == manifest.parent_champion_digest and (
        current.generation + 1 != manifest.generation
        or current.artifact_digest != manifest.parent_artifact_digest
    ):
        raise ValueError("current champion does not exactly match the generation parent")

    arena_payload = journal.stages.get("arena")
    if arena_payload is None:
        if current.manifest_digest != manifest.parent_champion_digest:
            raise ValueError("candidate pointer was published before arena evidence")
        arena = arena_runner(candidate, current_champion, root / "arena")
        _validate_arena_result(
            arena,
            candidate=candidate,
            parent_manifest_digest=manifest.parent_champion_digest,
            parent_artifact_digest=manifest.parent_artifact_digest,
            promotion_gate_config=promotion_gate_config,
        )
        arena_payload = _object(
            _json_value({"arena_result": arena}),
            "arena stage",
        )
        journal.record("arena", arena_payload)
        if stage_hook is not None:
            stage_hook("arena")
    arena = _arena_authority_from(arena_payload)
    verdict, arena_promoted = _validate_arena_result(
        arena,
        candidate=candidate,
        parent_manifest_digest=manifest.parent_champion_digest,
        parent_artifact_digest=manifest.parent_artifact_digest,
        promotion_gate_config=promotion_gate_config,
    )
    evidence = dict(arena_payload)
    evidence["promotion_gate_config_digest"] = manifest.promotion_gate_config_digest
    evidence["promotion_verdict"] = _json_value(verdict)

    if verdict is not None and arena_promoted:
        candidate = registry.finalize_candidate(candidate.digest, arena_results=evidence)
        current = registry.champion_pointer()
        completed = journal.stages.get("promoted")
        if completed is not None:
            if current is None or current.manifest_digest != candidate.digest:
                raise ValueError("promotion journal and champion pointer disagree")
            pointer = current
        elif current is not None and current.manifest_digest == candidate.digest:
            pointer = current
            journal.record("promoted", {"pointer_digest": pointer.digest})
        else:
            pointer = registry.promote(candidate.digest)
            journal.record("promoted", {"pointer_digest": pointer.digest})
        return PolicyIterationResult(
            PolicyIterationStatus.PROMOTED, manifest, candidate, pointer=pointer
        )

    completed = journal.stages.get("rejected")
    if completed is None:
        try:
            rejection = registry.rejection(candidate.digest)
        except RegistryError:
            rejection = registry.reject(candidate.digest, evidence=evidence)
        else:
            if rejection.evidence != evidence:
                raise ValueError("stored rejection evidence disagrees with arena evidence")
        journal.record("rejected", {"rejection_digest": rejection.digest})
    else:
        rejection = registry.rejection(candidate.digest)
    return PolicyIterationResult(
        PolicyIterationStatus.REJECTED, manifest, candidate, rejection=rejection
    )


__all__ = [
    "ArenaRunner",
    "CandidateGenerationPipeline",
    "GenerationManifest",
    "PolicyIterationResult",
    "PolicyIterationStatus",
    "run_policy_iteration",
]
