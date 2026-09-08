"""Resumable candidate-versus-champion arena orchestration.

The arena keeps game execution in :mod:`protocol`, statistics in
:mod:`arena_stats`, and policy gates in :mod:`promotion_gates`.  This module
only adapts those neutral boundaries into the smoke, screen, and promotion
workflow.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .arena_stats import (
    PairedSeedScore,
    SequentialDecision,
    SequentialPlan,
    SequentialResult,
    evaluate_sequential,
    pair_seed_scores,
)
from .promotion_gates import (
    ArtifactLoad,
    PromotionGateConfig,
    PromotionMetrics,
    PromotionVerdict,
    evaluate_promotion_gates,
)
from .protocol import (
    AgentSpec,
    EvaluationGameResult,
    EvaluationProtocol,
    GameCase,
    load_observations,
    run_protocol,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ArenaStage(StrEnum):
    """The three ordered arena stages."""

    SMOKE = "SMOKE"
    SCREEN = "SCREEN"
    PROMOTION = "PROMOTION"


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Immutable manifest and executable-content identity used by the arena."""

    manifest_digest: str
    artifact_digest: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.manifest_digest) is None:
            raise ValueError("manifest_digest must be a lowercase SHA-256 digest")
        if _SHA256.fullmatch(self.artifact_digest) is None:
            raise ValueError("artifact_digest must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ArenaStageConfig:
    """One explicit protocol/checkpoint/statistical policy stage."""

    stage: ArenaStage
    protocol: EvaluationProtocol
    checkpoint_path: Path
    sequential_plan: SequentialPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        plan = self.sequential_plan
        if plan is not None and plan.boundaries[-1].pair_count > len(self.protocol.world_seeds):
            raise ValueError("sequential plan exceeds the stage's predeclared seed pairs")


@dataclass(frozen=True, slots=True)
class ArenaOperationalEvidence:
    """Neutral non-strength measurements collected around arena decisions."""

    engine_errors: int
    illegal_choices: int
    agent_errors: int
    decision_latencies_ms: Mapping[str, tuple[float, ...]]
    stratum_score_margins: Mapping[str, float]
    artifact_loads: tuple[ArtifactLoad, ...]

    def __post_init__(self) -> None:
        if min(self.engine_errors, self.illegal_choices, self.agent_errors) < 0:
            raise ValueError("operational error counts must be non-negative")
        object.__setattr__(
            self,
            "decision_latencies_ms",
            MappingProxyType(dict(self.decision_latencies_ms)),
        )
        object.__setattr__(
            self,
            "stratum_score_margins",
            MappingProxyType(dict(self.stratum_score_margins)),
        )


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    """Complete predeclared arena policy and the two immutable contestants."""

    candidate: ArtifactIdentity
    champion: ArtifactIdentity
    candidate_agent: AgentSpec
    champion_agent: AgentSpec
    smoke: ArenaStageConfig
    screen: ArenaStageConfig
    promotion: ArenaStageConfig
    promotion_gates: PromotionGateConfig
    candidate_artifact_digest_param: str | None = None
    champion_artifact_digest_param: str | None = None

    def __post_init__(self) -> None:
        configured = (self.smoke.stage, self.screen.stage, self.promotion.stage)
        expected = (ArenaStage.SMOKE, ArenaStage.SCREEN, ArenaStage.PROMOTION)
        if configured != expected:
            raise ValueError("smoke, screen, and promotion stage configs must be explicit")
        if self.promotion.sequential_plan is None:
            raise ValueError("promotion requires a predeclared sequential plan")
        if self.promotion.sequential_plan.practical_margin != self.promotion_gates.practical_margin:
            raise ValueError(
                "promotion sequential plan and promotion gates must use the same "
                "effect-scale practical margin"
            )
        if self.promotion_gates.expected_artifact_digest != self.candidate.artifact_digest:
            raise ValueError("promotion gate artifact digest must identify the candidate")

        protocols = (self.smoke.protocol, self.screen.protocol, self.promotion.protocol)
        expected_matchup = {
            "agent_a": self.candidate_agent,
            "agent_b": self.champion_agent,
            "red_heroes": protocols[0].red_heroes,
            "blue_heroes": protocols[0].blue_heroes,
            "map_path": protocols[0].map_path,
            "game_type": protocols[0].game_type,
        }
        for protocol in protocols:
            for field_name, expected_value in expected_matchup.items():
                if getattr(protocol, field_name) != expected_value:
                    raise ValueError(f"arena protocols must share matchup field {field_name!r}")

        _validate_agent_artifact_binding(
            "candidate",
            self.candidate_agent,
            self.candidate,
            self.candidate_artifact_digest_param,
        )
        _validate_agent_artifact_binding(
            "champion",
            self.champion_agent,
            self.champion,
            self.champion_artifact_digest_param,
        )


def _validate_agent_artifact_binding(
    role: str,
    agent: AgentSpec,
    artifact: ArtifactIdentity,
    digest_param: str | None,
) -> None:
    identity_params = agent.identity()["params"]
    digest_params = {
        key
        for key in identity_params
        if key == "model_digest"
        or key == "artifact_digest"
        or key.endswith("_model_digest")
        or key.endswith("_artifact_digest")
    }
    if digest_params and digest_param is None:
        raise ValueError(
            f"{role} agent carries a model/artifact digest; declare its artifact digest param"
        )
    if digest_param is None:
        return
    if digest_param not in digest_params:
        raise ValueError(f"{role} artifact digest param {digest_param!r} is not in agent identity")
    if identity_params[digest_param] != artifact.artifact_digest:
        raise ValueError(f"{role} agent artifact digest does not match ArtifactIdentity")


@dataclass(frozen=True, slots=True)
class ArenaStageResult:
    """Protocol observations and paired statistical result for one stage."""

    stage: ArenaStage
    protocol_identity: str
    observations: tuple[EvaluationGameResult, ...]
    pairs: tuple[PairedSeedScore, ...]
    sequential: SequentialResult | None


@dataclass(frozen=True, slots=True)
class ArenaResult:
    """Complete arena outcome with deterministic audit evidence."""

    candidate: ArtifactIdentity
    champion: ArtifactIdentity
    stages: tuple[ArenaStageResult, ...]
    promotion_gate_config: PromotionGateConfig
    promotion_metrics: PromotionMetrics | None
    promotion_gates: PromotionVerdict | None
    promoted: bool

    def stage(self, stage: ArenaStage) -> ArenaStageResult:
        for result in self.stages:
            if result.stage is stage:
                return result
        raise KeyError(f"stage {stage.value} was not run")

    def canonical_evidence(self) -> str:
        """Return path-independent canonical JSON suitable for durable evidence."""

        return json.dumps(
            _json_value(self),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _protocol_prefix(protocol: EvaluationProtocol, pair_count: int) -> EvaluationProtocol:
    seeds = tuple(sorted(protocol.world_seeds)[:pair_count])
    return EvaluationProtocol(
        agent_a=protocol.agent_a,
        agent_b=protocol.agent_b,
        red_heroes=protocol.red_heroes,
        blue_heroes=protocol.blue_heroes,
        world_seeds=seeds,
        map_path=protocol.map_path,
        game_type=protocol.game_type,
        max_steps=protocol.max_steps,
        source_revision=protocol.source_revision,
        dirty_tree_hash=protocol.dirty_tree_hash,
        case_timeout_seconds=protocol.case_timeout_seconds,
    )


def _valid_cached_observations(config: ArenaStageConfig) -> tuple[EvaluationGameResult, ...]:
    """Adapt checkpoint rows only when they exactly match a scheduled case."""

    expected = {case.case_id: case for case in config.protocol.cases()}
    valid: list[EvaluationGameResult] = []
    seen: set[str] = set()
    for observation in load_observations(config.checkpoint_path):
        case = expected.get(observation.case_id)
        if case is None:
            continue  # protocol owns stale-identity compaction
        if observation.case_id in seen:
            raise ValueError(f"duplicate protocol observation {observation.case_id!r}")
        if observation.world_seed != case.world_seed or observation.a_side != case.a_side:
            raise ValueError("cached protocol observation does not match its scheduled case")
        seen.add(observation.case_id)
        valid.append(observation)
    return tuple(valid)


def _completed_prefix_count(config: ArenaStageConfig) -> int:
    observations = _valid_cached_observations(config)
    sides_by_seed: dict[int, set[str]] = {}
    for observation in observations:
        sides_by_seed.setdefault(observation.world_seed, set()).add(observation.a_side)

    # The invariant is simply: zero or more complete pairs, optionally followed
    # by one partial pair. No observations may occur after a gap or partial pair.
    seeds = sorted(config.protocol.world_seeds)
    observed_indexes = [index for index, seed in enumerate(seeds) if seed in sides_by_seed]
    if not observed_indexes:
        return 0
    last_observed = observed_indexes[-1]
    for index in range(last_observed):
        if sides_by_seed.get(seeds[index]) != {"RED", "BLUE"}:
            raise ValueError("cached protocol observations are not a seed-prefix")
    return last_observed + int(sides_by_seed[seeds[last_observed]] == {"RED", "BLUE"})


def _run_stage(
    config: ArenaStageConfig,
    run_case: Callable[[GameCase], EvaluationGameResult],
) -> ArenaStageResult:
    plan = config.sequential_plan
    if plan is None:
        complete_observations = tuple(
            run_protocol(
                config.protocol,
                checkpoint_path=config.checkpoint_path,
                run_case=run_case,
            )
        )
        return ArenaStageResult(
            stage=config.stage,
            protocol_identity=config.protocol.identity_digest(),
            observations=complete_observations,
            pairs=pair_seed_scores(complete_observations),
            sequential=None,
        )

    completed = _completed_prefix_count(config)
    result: SequentialResult | None = None
    observations: tuple[EvaluationGameResult, ...] = ()
    for boundary in plan.boundaries:
        if boundary.pair_count < completed:
            continue
        prefix = _protocol_prefix(config.protocol, boundary.pair_count)
        observations = tuple(
            run_protocol(prefix, checkpoint_path=config.checkpoint_path, run_case=run_case)
        )
        result = evaluate_sequential(observations, plan)
        if result.decision is not SequentialDecision.CONTINUE:
            break
    if result is None:
        observations = _valid_cached_observations(config)
        result = evaluate_sequential(observations, plan)
    return ArenaStageResult(
        stage=config.stage,
        protocol_identity=config.protocol.identity_digest(),
        observations=observations,
        pairs=pair_seed_scores(observations),
        sequential=result,
    )


def _gate_metrics(
    stages: tuple[ArenaStageResult, ...], evidence: ArenaOperationalEvidence
) -> PromotionMetrics:
    observations = tuple(obs for stage in stages for obs in stage.observations)
    promotion = next(stage for stage in stages if stage.stage is ArenaStage.PROMOTION)
    candidate_score = sum(pair.score for pair in promotion.pairs) / len(promotion.pairs)
    return PromotionMetrics(
        total_games=len(observations),
        engine_errors=evidence.engine_errors,
        illegal_choices=evidence.illegal_choices,
        agent_errors=evidence.agent_errors,
        timeout_terminations=sum(obs.reason == "wall_clock_timeout" for obs in observations),
        max_step_terminations=sum(obs.reason == "max_steps" for obs in observations),
        candidate_score=candidate_score,
        champion_score=1.0 - candidate_score,
        decision_latencies_ms=MappingProxyType(dict(evidence.decision_latencies_ms)),
        stratum_score_margins=MappingProxyType(dict(evidence.stratum_score_margins)),
        artifact_loads=evidence.artifact_loads,
    )


def run_arena(
    config: ArenaConfig,
    *,
    run_cases: Mapping[ArenaStage, Callable[[GameCase], EvaluationGameResult]],
    operational_evidence: ArenaOperationalEvidence,
) -> ArenaResult:
    """Run the configured arena, resuming each stage at complete seed pairs."""

    stage_results: list[ArenaStageResult] = []
    for stage_config in (config.smoke, config.screen, config.promotion):
        try:
            runner = run_cases[stage_config.stage]
        except KeyError as exc:
            raise ValueError(f"missing runner for stage {stage_config.stage.value}") from exc
        result = _run_stage(stage_config, runner)
        stage_results.append(result)
        sequential = result.sequential
        if sequential is not None and sequential.decision is SequentialDecision.REJECT:
            return ArenaResult(
                candidate=config.candidate,
                champion=config.champion,
                stages=tuple(stage_results),
                promotion_gate_config=config.promotion_gates,
                promotion_metrics=None,
                promotion_gates=None,
                promoted=False,
            )

    stages = tuple(stage_results)
    promotion_result = stages[-1].sequential
    assert promotion_result is not None
    metrics = _gate_metrics(stages, operational_evidence)
    verdict = evaluate_promotion_gates(metrics, config.promotion_gates)
    return ArenaResult(
        candidate=config.candidate,
        champion=config.champion,
        stages=stages,
        promotion_gate_config=config.promotion_gates,
        promotion_metrics=metrics,
        promotion_gates=verdict,
        promoted=(promotion_result.decision is SequentialDecision.PROMOTE and verdict.promoted),
    )


__all__ = [
    "ArenaConfig",
    "ArenaOperationalEvidence",
    "ArenaResult",
    "ArenaStage",
    "ArenaStageConfig",
    "ArenaStageResult",
    "ArtifactIdentity",
    "run_arena",
]
