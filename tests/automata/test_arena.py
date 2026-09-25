from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from automata.evaluation.arena import (
    ArenaConfig,
    ArenaOperationalEvidence,
    ArenaStage,
    ArenaStageConfig,
    ArtifactIdentity,
    run_arena,
)
from automata.evaluation.arena_stats import SequentialBoundary, SequentialPlan
from automata.evaluation.promotion_gates import (
    ArtifactLoad,
    LatencySLO,
    PromotionGateConfig,
)
from automata.evaluation.protocol import (
    AgentSpec,
    EvaluationGameResult,
    EvaluationProtocol,
    GameCase,
)


def _protocol(seeds: tuple[int, ...], **overrides: Any) -> EvaluationProtocol:
    values: dict[str, Any] = {
        "agent_a": AgentSpec("candidate", "fake"),
        "agent_b": AgentSpec("champion", "fake"),
        "red_heroes": ("Wasp",),
        "blue_heroes": ("Arien",),
        "world_seeds": seeds,
        "map_path": "map.json",
        "game_type": "QUICK",
        "max_steps": 100,
        "source_revision": "revision",
        "dirty_tree_hash": "clean",
    }
    values.update(overrides)
    return EvaluationProtocol(**values)


def _stage(
    tmp_path: Path,
    stage: ArenaStage,
    seeds: tuple[int, ...],
    *,
    looks: tuple[int, ...] = (),
) -> ArenaStageConfig:
    plan = None
    if looks:
        plan = SequentialPlan(
            boundaries=tuple(SequentialBoundary(pair_count=count, alpha=0.4) for count in looks)
        )
    return ArenaStageConfig(
        stage=stage,
        protocol=_protocol(seeds),
        checkpoint_path=tmp_path / f"{stage.value.lower()}.jsonl",
        sequential_plan=plan,
    )


def _config(
    tmp_path: Path,
    *,
    candidate_agent: AgentSpec | None = None,
    champion_agent: AgentSpec | None = None,
    smoke_protocol: EvaluationProtocol | None = None,
    screen_protocol: EvaluationProtocol | None = None,
    promotion_protocol: EvaluationProtocol | None = None,
    candidate_artifact_digest_param: str | None = None,
    champion_artifact_digest_param: str | None = None,
) -> ArenaConfig:
    digest = "a" * 64
    candidate_agent = candidate_agent or AgentSpec("candidate", "fake")
    champion_agent = champion_agent or AgentSpec("champion", "fake")
    agent_overrides = {"agent_a": candidate_agent, "agent_b": champion_agent}
    return ArenaConfig(
        candidate=ArtifactIdentity(manifest_digest="c" * 64, artifact_digest=digest),
        champion=ArtifactIdentity(manifest_digest="d" * 64, artifact_digest="b" * 64),
        candidate_agent=candidate_agent,
        champion_agent=champion_agent,
        smoke=(
            ArenaStageConfig(ArenaStage.SMOKE, smoke_protocol, tmp_path / "smoke.jsonl")
            if smoke_protocol is not None
            else ArenaStageConfig(
                ArenaStage.SMOKE,
                _protocol((1, 2), **agent_overrides),
                tmp_path / "smoke.jsonl",
            )
        ),
        screen=(
            ArenaStageConfig(
                ArenaStage.SCREEN,
                screen_protocol,
                tmp_path / "screen.jsonl",
                SequentialPlan((SequentialBoundary(pair_count=10, alpha=0.4),)),
            )
            if screen_protocol is not None
            else ArenaStageConfig(
                ArenaStage.SCREEN,
                _protocol(tuple(range(10)), **agent_overrides),
                tmp_path / "screen.jsonl",
                SequentialPlan((SequentialBoundary(pair_count=10, alpha=0.4),)),
            )
        ),
        promotion=(
            ArenaStageConfig(
                ArenaStage.PROMOTION,
                promotion_protocol,
                tmp_path / "promotion.jsonl",
                SequentialPlan(
                    (
                        SequentialBoundary(pair_count=10, alpha=0.4),
                        SequentialBoundary(pair_count=20, alpha=0.4),
                    )
                ),
            )
            if promotion_protocol is not None
            else ArenaStageConfig(
                ArenaStage.PROMOTION,
                _protocol(tuple(range(20)), **agent_overrides),
                tmp_path / "promotion.jsonl",
                SequentialPlan(
                    (
                        SequentialBoundary(pair_count=10, alpha=0.4),
                        SequentialBoundary(pair_count=20, alpha=0.4),
                    )
                ),
            )
        ),
        promotion_gates=PromotionGateConfig(
            max_timeout_or_max_step_rate=0.0,
            latency_slos=(LatencySLO(tier="standard", percentile=100.0, max_ms=10.0),),
            practical_margin=0.0,
            required_strata=("all",),
            max_stratum_regression=0.0,
            required_artifact_loads=2,
            expected_artifact_digest=digest,
        ),
        candidate_artifact_digest_param=candidate_artifact_digest_param,
        champion_artifact_digest_param=champion_artifact_digest_param,
    )


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("agent_a", AgentSpec("other-candidate", "fake")),
        ("agent_b", AgentSpec("other-champion", "fake")),
        ("red_heroes", ("Arien",)),
        ("blue_heroes", ("Wasp",)),
        ("map_path", "other-map.json"),
        ("game_type", "STANDARD"),
    ],
)
def test_arena_rejects_cross_stage_matchup_mismatch(
    tmp_path: Path, field: str, different: object
) -> None:
    protocol = _protocol(tuple(range(10)), **{field: different})

    with pytest.raises(ValueError, match=field):
        _config(tmp_path, screen_protocol=protocol)


def test_arena_rejects_mismatched_sequential_and_gate_practical_margins(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    assert config.promotion.sequential_plan is not None
    promotion = replace(
        config.promotion,
        sequential_plan=replace(config.promotion.sequential_plan, practical_margin=0.10),
    )

    with pytest.raises(ValueError, match="practical margin"):
        replace(config, promotion=promotion)


@pytest.mark.parametrize("role", ["candidate", "champion"])
def test_arena_binds_declared_agent_model_digest_to_artifact(tmp_path: Path, role: str) -> None:
    kwargs: dict[str, Any]
    if role == "candidate":
        kwargs = {
            "candidate_agent": AgentSpec("candidate", "fake", {"value_model_digest": "f" * 64}),
            "candidate_artifact_digest_param": "value_model_digest",
        }
    else:
        kwargs = {
            "champion_agent": AgentSpec("champion", "fake", {"policy_model_digest": "f" * 64}),
            "champion_artifact_digest_param": "policy_model_digest",
        }

    with pytest.raises(ValueError, match=rf"{role}.*digest"):
        _config(tmp_path, **kwargs)


def test_agent_model_digest_requires_an_explicit_artifact_binding(tmp_path: Path) -> None:
    candidate = AgentSpec("candidate", "fake", {"value_model_digest": "a" * 64})

    with pytest.raises(ValueError, match=r"candidate.*declare.*digest param"):
        _config(tmp_path, candidate_agent=candidate)


def test_matching_explicit_agent_artifact_bindings_are_accepted(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        candidate_agent=AgentSpec("candidate", "fake", {"value_model_digest": "a" * 64}),
        champion_agent=AgentSpec("champion", "fake", {"policy_model_digest": "b" * 64}),
        candidate_artifact_digest_param="value_model_digest",
        champion_artifact_digest_param="policy_model_digest",
    )

    assert config.smoke.protocol.agent_a == config.candidate_agent
    assert config.smoke.protocol.agent_b == config.champion_agent


class FakeRunner:
    def __init__(self, *, draw_seeds: frozenset[int] = frozenset()) -> None:
        self.calls: list[tuple[int, str]] = []
        self.draw_seeds = draw_seeds

    def __call__(self, case: GameCase) -> EvaluationGameResult:
        self.calls.append((case.world_seed, case.a_side))
        return EvaluationGameResult(
            case_id=case.case_id,
            world_seed=case.world_seed,
            a_side=case.a_side,
            winner_side=None if case.world_seed in self.draw_seeds else case.a_side,
            rounds=2,
            steps=20,
            reason="game_over",
        )


def _evidence() -> ArenaOperationalEvidence:
    return ArenaOperationalEvidence(
        engine_errors=0,
        illegal_choices=0,
        agent_errors=0,
        decision_latencies_ms={"standard": (2.0, 1.0)},
        stratum_score_margins={"all": 0.0},
        artifact_loads=(
            ArtifactLoad(digest="a" * 64, error=None),
            ArtifactLoad(digest="a" * 64, error=None),
        ),
    )


def test_operational_evidence_freezes_caller_owned_mappings(tmp_path: Path) -> None:
    latencies: dict[str, tuple[float, ...]] = {"standard": (2.0, 1.0)}
    margins: dict[str, float] = {"all": 0.0}
    evidence = ArenaOperationalEvidence(
        engine_errors=0,
        illegal_choices=0,
        agent_errors=0,
        decision_latencies_ms=latencies,
        stratum_score_margins=margins,
        artifact_loads=(
            ArtifactLoad(digest="a" * 64, error=None),
            ArtifactLoad(digest="a" * 64, error=None),
        ),
    )

    latencies["standard"] = (100.0,)
    margins["all"] = -1.0

    assert evidence.decision_latencies_ms == {"standard": (2.0, 1.0)}
    assert evidence.stratum_score_margins == {"all": 0.0}
    with pytest.raises(TypeError):
        cast(dict[str, tuple[float, ...]], evidence.decision_latencies_ms)["standard"] = (100.0,)
    with pytest.raises(TypeError):
        cast(dict[str, float], evidence.stratum_score_margins)["all"] = -1.0

    result = run_arena(
        _config(tmp_path),
        run_cases={stage: FakeRunner() for stage in ArenaStage},
        operational_evidence=evidence,
    )
    canonical = result.canonical_evidence()
    latencies["new"] = (1000.0,)
    margins["new"] = -10.0

    assert result.promoted is True
    assert result.canonical_evidence() == canonical


def test_arena_runs_explicit_side_swapped_stages_and_scores_draws_as_half(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    runners = {
        ArenaStage.SMOKE: FakeRunner(draw_seeds=frozenset({1})),
        ArenaStage.SCREEN: FakeRunner(),
        ArenaStage.PROMOTION: FakeRunner(),
    }

    result = run_arena(config, run_cases=runners, operational_evidence=_evidence())

    smoke = result.stage(ArenaStage.SMOKE)
    assert [(pair.world_seed, pair.score) for pair in smoke.pairs] == [(1, 0.5), (2, 1.0)]
    assert runners[ArenaStage.SMOKE].calls == [(1, "RED"), (1, "BLUE"), (2, "RED"), (2, "BLUE")]
    assert result.promoted is True
    assert [stage.stage for stage in result.stages] == [
        ArenaStage.SMOKE,
        ArenaStage.SCREEN,
        ArenaStage.PROMOTION,
    ]


def test_promotion_stops_at_first_predeclared_complete_pair_boundary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    promotion_runner = FakeRunner()

    result = run_arena(
        config,
        run_cases={
            ArenaStage.SMOKE: FakeRunner(),
            ArenaStage.SCREEN: FakeRunner(),
            ArenaStage.PROMOTION: promotion_runner,
        },
        operational_evidence=_evidence(),
    )

    sequential = result.stage(ArenaStage.PROMOTION).sequential
    assert sequential is not None
    assert sequential.evaluated_pair_count == 10
    assert len(promotion_runner.calls) == 20
    assert all(
        {side for seed, side in promotion_runner.calls if seed == world_seed} == {"RED", "BLUE"}
        for world_seed in range(10)
    )


def test_resume_completes_a_partial_pair_before_any_sequential_decision(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stage = config.promotion
    first_case = next(iter(stage.protocol.cases()))
    first = FakeRunner()(first_case)
    stage.checkpoint_path.write_text(first.to_json() + "\n", encoding="utf-8")
    runner = FakeRunner()

    result = run_arena(
        config,
        run_cases={
            ArenaStage.SMOKE: FakeRunner(),
            ArenaStage.SCREEN: FakeRunner(),
            ArenaStage.PROMOTION: runner,
        },
        operational_evidence=_evidence(),
    )

    assert (first_case.world_seed, first_case.a_side) not in runner.calls
    assert (first_case.world_seed, "BLUE") in runner.calls
    sequential = result.stage(ArenaStage.PROMOTION).sequential
    assert sequential is not None
    assert sequential.observed_pair_count == 10


def test_canonical_evidence_is_order_independent_and_carries_artifact_identities(
    tmp_path: Path,
) -> None:
    ordered = _config(tmp_path / "ordered")
    ordered_result = run_arena(
        ordered,
        run_cases={stage: FakeRunner() for stage in ArenaStage},
        operational_evidence=_evidence(),
    )

    reordered = _config(
        tmp_path / "reordered",
        smoke_protocol=_protocol((2, 1)),
        screen_protocol=_protocol(tuple(reversed(range(10)))),
        promotion_protocol=_protocol(tuple(reversed(range(20)))),
    )
    completed_seeds = {ArenaStage.SMOKE: 2, ArenaStage.SCREEN: 10, ArenaStage.PROMOTION: 10}
    for stage in (reordered.smoke, reordered.screen, reordered.promotion):
        observations = [
            FakeRunner()(case)
            for case in stage.protocol.cases()
            if case.world_seed in sorted(stage.protocol.world_seeds)[: completed_seeds[stage.stage]]
        ]
        stage.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        stage.checkpoint_path.write_text(
            "".join(observation.to_json() + "\n" for observation in reversed(observations)),
            encoding="utf-8",
        )
    reordered_result = run_arena(
        reordered,
        run_cases={stage: FakeRunner() for stage in ArenaStage},
        operational_evidence=_evidence(),
    )

    encoded = ordered_result.canonical_evidence()
    payload = json.loads(encoded)
    assert encoded == reordered_result.canonical_evidence()
    assert encoded == json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert payload["candidate"]["artifact_digest"] == "a" * 64
    assert payload["champion"]["manifest_digest"] == "d" * 64
    assert payload["promotion_gates"]["promoted"] is True


def test_neutral_operational_failures_are_all_retained_and_block_promotion(
    tmp_path: Path,
) -> None:
    evidence = ArenaOperationalEvidence(
        engine_errors=1,
        illegal_choices=1,
        agent_errors=1,
        decision_latencies_ms={"standard": (11.0,)},
        stratum_score_margins={"all": -0.1},
        artifact_loads=(
            ArtifactLoad(digest="a" * 64, error=None),
            ArtifactLoad(digest=None, error="reload failed"),
        ),
    )

    result = run_arena(
        _config(tmp_path),
        run_cases={stage: FakeRunner() for stage in ArenaStage},
        operational_evidence=evidence,
    )

    assert result.promoted is False
    assert result.promotion_gates is not None
    assert {failure.gate for failure in result.promotion_gates.failures} == {
        "engine_errors",
        "illegal_choices",
        "agent_errors",
        "latency:standard:p100",
        "required_stratum:all",
        "deterministic_artifact_load",
    }


def test_screen_rejection_does_not_run_the_promotion_stage(tmp_path: Path) -> None:
    class LosingRunner(FakeRunner):
        def __call__(self, case: GameCase) -> EvaluationGameResult:
            observation = super().__call__(case)
            return EvaluationGameResult(
                case_id=observation.case_id,
                world_seed=observation.world_seed,
                a_side=observation.a_side,
                winner_side="BLUE" if case.a_side == "RED" else "RED",
                rounds=observation.rounds,
                steps=observation.steps,
                reason=observation.reason,
            )

    promotion_runner = FakeRunner()
    result = run_arena(
        _config(tmp_path),
        run_cases={
            ArenaStage.SMOKE: FakeRunner(),
            ArenaStage.SCREEN: LosingRunner(),
            ArenaStage.PROMOTION: promotion_runner,
        },
        operational_evidence=_evidence(),
    )

    assert result.promoted is False
    assert result.promotion_gates is None
    assert promotion_runner.calls == []
