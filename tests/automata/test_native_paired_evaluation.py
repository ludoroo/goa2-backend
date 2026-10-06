"""Behavior tests for the concrete native paired gameplay runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import automata.training.native_paired as runner
from automata.harness.game_runner import RunResult
from automata.models.contracts import ArtifactError
from automata.models.shared_encoder.gen1_model import GEN1_ARCHITECTURE_ID
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.search.continuation import PriorSamplingContinuationPolicy
from automata.search.contracts import (
    ComponentInferenceError,
    CutoffUnit,
    LeafMode,
    PolicyScores,
    ScoreSemantics,
)
from automata.search.heuristic import HeuristicLeafEvaluator
from automata.search.ismcts.strategy import ISMCTSStrategy
from automata.search.learned import LearnedStableValueEvaluator
from automata.search.root import RootTarget
from automata.training.native_paired_contracts import (
    NativeEvaluatedArtifactIdentity,
    NativeEvaluationFixture,
    NativePairedEvaluationAuthorities,
    NativePairedEvaluationConfig,
    create_native_paired_evaluation_manifest,
    load_native_paired_evaluation_result,
)
from automata.training.native_run_contracts import NativeRunSearchConfig
from automata.training.native_splits import NativeSeedRange, NativeSplitConfig
from goa2.data.heroes import HeroRegistry
from goa2.domain.models import TeamColor

_DIGEST = "a" * 64
_COMPARISONS = (
    "CANDIDATE_VS_HEURISTIC_FULL_SEARCH",
    "CANDIDATE_VALUE_VS_HEURISTIC_VALUE_FIXED_CANDIDATE_POLICY",
    "CANDIDATE_VS_HEURISTIC_POLICY_ONLY",
)


def _search() -> NativeRunSearchConfig:
    return NativeRunSearchConfig(
        iterations=2,
        decision_timeout_seconds=None,
        max_advance_steps=100,
        uct_c=1.4,
        cutoff_limit=2,
        cutoff_unit=CutoffUnit.ROUNDS,
        max_advance_transitions=100,
        max_forced_decisions=50,
        leaf_mode=LeafMode.STABLE_TRANSITION,
        widening_c=2.0,
        widening_alpha=0.5,
        root_widening_c=None,
        root_widening_alpha=None,
        adaptive_hex_root_schedule_version=None,
        request_schedule_version=None,
        seed=0,
        use_prior=True,
        puct_c=1.0,
        root_puct_c=None,
    )


def _manifest(tmp_path: Path):
    config = NativePairedEvaluationConfig(
        evaluation_id="native-paired-runner-test",
        source_revision="test-revision",
        dirty_tree_hash="fixture",
        candidate_model_digest=_DIGEST,
        parent_model_digest=None,
        split_config=NativeSplitConfig(
            namespace="native-paired-runner-tests",
            salt="fixed",
            validation_fraction=0.2,
            seed_ranges=(NativeSeedRange(purpose="evaluation", start=500, stop=501),),
        ),
        comparisons=_COMPARISONS,
        fixtures=(
            NativeEvaluationFixture(
                fixture_id="quick-1v1",
                map_id="forgotten_island",
                game_type="QUICK",
                red_composition=("Wasp",),
                blue_composition=("Arien",),
                world_seeds=(500,),
            ),
        ),
        search=_search(),
        random_stream_namespace="native-paired-runner-streams",
        max_steps=30,
        max_rounds=3,
    )
    return create_native_paired_evaluation_manifest(
        config,
        NativePairedEvaluationAuthorities(
            output_root=(tmp_path / "evaluation").absolute(),
            candidate_artifact_path=(tmp_path / "candidate").absolute(),
            parent_artifact_path=None,
        ),
    )


def _parent_manifest(tmp_path: Path):
    base = _manifest(tmp_path)
    config = base.config.model_copy(
        update={
            "parent_model_digest": "c" * 64,
            "comparisons": (
                *base.config.comparisons,
                "CANDIDATE_VS_GEN1_PARENT_FULL_SEARCH",
            ),
        }
    )
    return create_native_paired_evaluation_manifest(
        config,
        NativePairedEvaluationAuthorities(
            output_root=base.authorities.output_root,
            candidate_artifact_path=base.authorities.candidate_artifact_path,
            parent_artifact_path=(tmp_path / "parent").absolute(),
        ),
    )


def _identity(manifest) -> NativeEvaluatedArtifactIdentity:
    return NativeEvaluatedArtifactIdentity(
        role="CANDIDATE",
        model_digest=manifest.config.candidate_model_digest,
        manifest_sha256="b" * 64,
        model_id=GEN1_ARCHITECTURE_ID,
        runtime_compatibility_version=1,
        decision_tensor_schema_digest=TensorFeatureSchema.current().digest,
        stable_value_tensor_schema_digest=StableValueTensorSchema.current().digest,
        value_semantics="stable-boundary-outcome-v1",
        current_scope_digest=manifest.current_scope.digest,
        device="cpu",
        floating_dtype="float32",
    )


def _isolate_gameplay(monkeypatch: pytest.MonkeyPatch, manifest) -> None:
    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda _: runner._LoadedArtifacts(None, None, (_identity(manifest),)),
    )
    monkeypatch.setattr(
        runner,
        "_build_runtimes",
        lambda *_: {manifest.config.fixtures[0].fixture_id: object()},
    )
    monkeypatch.setattr(
        runner,
        "_build_case_agents",
        lambda *_: ({"hero_wasp": Mock(), "hero_arien": Mock()}, {}),
    )


def _outcome(
    *,
    winner: str | None = "RED",
    reason: str = "game_over",
    rounds: int = 1,
    steps: int = 3,
) -> RunResult:
    return RunResult(
        winner=winner if reason == "game_over" else None,
        winner_side=winner if reason == "game_over" else None,
        rounds=rounds,
        turns=2,
        steps=steps,
        reason=reason,
    )


def test_native_paired_runner_exposes_the_public_execution_seam() -> None:
    assert callable(runner.derive_native_evaluation_seed)
    assert callable(runner.run_native_paired_evaluation)


def test_seed_streams_are_side_common_role_independent_and_domain_separated(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    red_legs = manifest.planned_cases[:2]

    red_search = runner.derive_native_evaluation_seed(
        manifest,
        comparison=red_legs[0].comparison,
        fixture_id=red_legs[0].fixture_id,
        world_seed=red_legs[0].world_seed,
        stream="RED_SEARCH",
    )
    same_red_search = runner.derive_native_evaluation_seed(
        manifest,
        comparison=red_legs[1].comparison,
        fixture_id=red_legs[1].fixture_id,
        world_seed=red_legs[1].world_seed,
        stream="RED_SEARCH",
    )
    streams = {
        runner.derive_native_evaluation_seed(
            manifest,
            comparison=red_legs[0].comparison,
            fixture_id=red_legs[0].fixture_id,
            world_seed=red_legs[0].world_seed,
            stream=stream,
        )
        for stream in (
            "RED_SEARCH",
            "RED_ENVIRONMENT",
            "BLUE_SEARCH",
            "BLUE_ENVIRONMENT",
        )
    }

    assert red_search == same_red_search
    assert len(streams) == 4
    assert red_search == runner.derive_native_evaluation_seed(
        manifest,
        comparison=red_legs[0].comparison,
        fixture_id=red_legs[0].fixture_id,
        world_seed=red_legs[0].world_seed,
        stream="RED_SEARCH",
    )


class _FakeRuntime:
    def evaluate_policy(self, observation):  # pragma: no cover - wiring only
        raise AssertionError("not evaluated by this wiring test")

    def evaluate_stable_value(self, observation):  # pragma: no cover - wiring only
        raise AssertionError("not evaluated by this wiring test")


def test_component_recipes_include_real_search_fixed_policy_value_and_zero_search(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    runtime = _FakeRuntime()
    runtimes = runner._FixtureRuntimes(runtime, None)

    full_candidate = runner._build_arm_agent(
        manifest,
        manifest.planned_cases[0],
        runtimes,
        side="RED",
        candidate_arm=True,
    )
    full_heuristic = runner._build_arm_agent(
        manifest,
        manifest.planned_cases[0],
        runtimes,
        side="BLUE",
        candidate_arm=False,
    )
    fixed_learned = runner._build_arm_agent(
        manifest,
        manifest.planned_cases[2],
        runtimes,
        side="RED",
        candidate_arm=True,
    )
    fixed_heuristic = runner._build_arm_agent(
        manifest,
        manifest.planned_cases[2],
        runtimes,
        side="BLUE",
        candidate_arm=False,
    )
    policy_candidate = runner._build_arm_agent(
        manifest,
        manifest.planned_cases[4],
        runtimes,
        side="RED",
        candidate_arm=True,
    )
    policy_heuristic = runner._build_arm_agent(
        manifest,
        manifest.planned_cases[4],
        runtimes,
        side="BLUE",
        candidate_arm=False,
    )

    assert isinstance(full_candidate.strategy, ISMCTSStrategy)
    assert isinstance(full_heuristic.strategy, ISMCTSStrategy)
    assert isinstance(fixed_learned.strategy, ISMCTSStrategy)
    assert isinstance(fixed_heuristic.strategy, ISMCTSStrategy)
    assert fixed_learned.prior.runtime is runtime
    assert fixed_heuristic.prior.runtime is runtime
    assert isinstance(fixed_learned.continuation, PriorSamplingContinuationPolicy)
    assert isinstance(fixed_heuristic.continuation, PriorSamplingContinuationPolicy)
    assert isinstance(fixed_learned.leaf, LearnedStableValueEvaluator)
    assert isinstance(fixed_heuristic.leaf, HeuristicLeafEvaluator)
    assert isinstance(policy_candidate.strategy, runner._PriorOnlyStrategy)
    assert isinstance(policy_heuristic.strategy, runner._PriorOnlyStrategy)
    assert policy_candidate.leaf is policy_heuristic.leaf is None
    second_candidate = runner._build_arm_agent(
        manifest,
        manifest.planned_cases[0],
        runtimes,
        side="RED",
        candidate_arm=True,
    )
    assert second_candidate.agent is not full_candidate.agent
    assert second_candidate.environment is not full_candidate.environment
    assert (
        second_candidate.environment._rng.getstate() == full_candidate.environment._rng.getstate()
    )


def test_optional_parent_full_search_uses_the_exact_parent_runtime(tmp_path: Path) -> None:
    manifest = _parent_manifest(tmp_path)
    candidate_runtime = _FakeRuntime()
    parent_runtime = _FakeRuntime()
    parent_case = manifest.planned_cases[-2]

    parent = runner._build_arm_agent(
        manifest,
        parent_case,
        runner._FixtureRuntimes(candidate_runtime, parent_runtime),
        side="BLUE",
        candidate_arm=False,
    )

    assert isinstance(parent.strategy, ISMCTSStrategy)
    assert parent.prior.runtime is parent_runtime
    assert isinstance(parent.continuation, PriorSamplingContinuationPolicy)
    assert isinstance(parent.leaf, LearnedStableValueEvaluator)
    assert parent.leaf.runtime is parent_runtime


def test_policy_only_strategy_calls_public_scores_and_stable_first_argmax_without_search() -> None:
    hero = HeroRegistry.get("Wasp")
    assert hero is not None
    hero.team = TeamColor.RED
    state = Mock()
    state.get_hero.return_value = hero
    policy = Mock()
    policy.score.return_value = PolicyScores(("first", "second"), (7.0, 7.0), ScoreSemantics.LOGITS)
    strategy = runner._PriorOnlyStrategy(policy)

    result = strategy.select(
        state,
        TeamColor.RED,
        RootTarget.card(hero_id=str(hero.id), owned_hero_ids=frozenset({str(hero.id)})),
        ("first", "second"),
    )

    assert result.selected_candidate == "first"
    assert result.search_result is None
    policy.score.assert_called_once()


def test_runner_keeps_board_rosters_fixed_continues_after_censor_and_scores_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    _isolate_gameplay(monkeypatch, manifest)
    calls: list[tuple[tuple[str, ...], tuple[str, ...], int]] = []
    outcomes = iter(
        (
            _outcome(winner="RED"),
            _outcome(winner="BLUE"),
            _outcome(winner="BLUE"),
            _outcome(winner="RED"),
            _outcome(winner=None, reason="max_steps", steps=30),
            _outcome(winner="BLUE"),
        )
    )

    def fake_run(red, blue, agents, **kwargs):
        del agents
        calls.append((tuple(red), tuple(blue), kwargs["seed"]))
        return next(outcomes)

    monkeypatch.setattr(runner, "run_game", fake_run)

    result = runner.run_native_paired_evaluation(manifest)

    assert result.status == "SUCCEEDED"
    assert len(calls) == len(manifest.planned_cases) == 6
    assert calls == [(("Wasp",), ("Arien",), 500)] * 6
    assert result.attempted_case_ids == tuple(case.case_id for case in manifest.planned_cases)
    assert result.aggregates[0].descriptive_mean_completed_pair_score == 1.0
    assert result.aggregates[2].censored_case_count == 1
    assert result.aggregates[2].excluded_completed_singletons == 1
    assert not any(manifest.authorities.output_root.glob("**/*replay*"))


def test_round_limit_censors_preserve_truthful_harness_counter_and_full_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    _isolate_gameplay(monkeypatch, manifest)
    monkeypatch.setattr(
        runner,
        "run_game",
        lambda *args, **kwargs: _outcome(
            winner=None,
            reason="max_rounds",
            rounds=manifest.config.max_rounds + 1,
        ),
    )

    result = runner.run_native_paired_evaluation(manifest)

    assert result.status == "SUCCEEDED"
    assert result.attempted_case_ids == tuple(case.case_id for case in manifest.planned_cases)
    assert all(item.status == "CENSORED" for item in result.observations)
    assert all(item.rounds == manifest.config.max_rounds + 1 for item in result.observations)
    assert result.completed_pairs == ()
    assert all(
        aggregate.descriptive_mean_completed_pair_score is None for aggregate in result.aggregates
    )


def test_inference_failure_publishes_partial_failed_evidence_and_reraises_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    _isolate_gameplay(monkeypatch, manifest)
    failure = ComponentInferenceError("model failed")
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            return _outcome(winner="RED")
        raise failure

    monkeypatch.setattr(runner, "run_game", fake_run)

    with pytest.raises(ComponentInferenceError) as raised:
        runner.run_native_paired_evaluation(manifest)

    result = load_native_paired_evaluation_result(manifest.authorities.output_root / "result.json")
    assert raised.value is failure
    assert calls == 2
    assert result.status == "FAILED"
    assert result.failure is not None
    assert result.failure.category == "INFERENCE_FAILURE"
    assert result.failure.case_id == manifest.planned_cases[1].case_id
    assert result.attempted_case_ids == tuple(case.case_id for case in manifest.planned_cases[:2])
    assert len(result.observations) == 1
    assert not result.completed_pairs and not result.aggregates
    assert not (manifest.authorities.output_root / "complete.json").exists()


def test_boundary_space_error_persists_failed_status_and_reraises_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    _isolate_gameplay(monkeypatch, manifest)
    failure = RuntimeError("x" * 999 + " tail")
    monkeypatch.setattr(runner, "run_game", Mock(side_effect=failure))

    with pytest.raises(RuntimeError) as raised:
        runner.run_native_paired_evaluation(manifest)

    failed = load_native_paired_evaluation_result(manifest.authorities.output_root / "result.json")
    assert raised.value is failure
    assert failed.status == "FAILED"
    assert failed.failure is not None
    assert failed.failure.message == "x" * 999


@pytest.mark.parametrize(
    ("outcome", "category"),
    [
        (_outcome(winner=None), "OUTCOME_NORMALIZATION_FAILURE"),
        (RuntimeError("engine exploded"), "GAMEPLAY_FAILURE"),
    ],
)
def test_malformed_terminal_and_engine_errors_are_never_draws_or_inference_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: RunResult | BaseException,
    category: str,
) -> None:
    manifest = _manifest(tmp_path)
    _isolate_gameplay(monkeypatch, manifest)

    def fake_run(*args, **kwargs):
        del args, kwargs
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(runner, "run_game", fake_run)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run_native_paired_evaluation(manifest)

    failed = load_native_paired_evaluation_result(manifest.authorities.output_root / "result.json")
    assert failed.failure is not None
    assert failed.failure.category == category
    assert failed.observations == ()


def test_existing_output_root_rejects_before_artifact_access(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.authorities.output_root.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_native_paired_evaluation(manifest)


def test_artifact_failure_is_preflighted_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    manifest.authorities.candidate_artifact_path.mkdir()
    expected = ArtifactError("bad artifact")
    monkeypatch.setattr(
        runner,
        "load_current_gen1_parent_artifact",
        Mock(side_effect=expected),
    )

    with pytest.raises(ArtifactError) as raised:
        runner.run_native_paired_evaluation(manifest)

    assert raised.value is expected
    assert not manifest.authorities.output_root.exists()


def test_runtime_constructor_failure_happens_before_output_root_is_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda _: runner._LoadedArtifacts(None, None, (_identity(manifest),)),
    )
    expected = RuntimeError("runtime construction failed")
    monkeypatch.setattr(runner, "_build_runtimes", Mock(side_effect=expected))

    with pytest.raises(RuntimeError) as raised:
        runner.run_native_paired_evaluation(manifest)

    assert raised.value is expected
    assert not manifest.authorities.output_root.exists()


def test_marker_publication_failure_rewrites_owned_result_to_truthful_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    _isolate_gameplay(monkeypatch, manifest)
    monkeypatch.setattr(runner, "run_game", lambda *args, **kwargs: _outcome(winner="RED"))
    original_publish = runner._publish_new
    marker_error = OSError("marker fsync failed")

    def fail_marker(path: Path, payload: bytes):
        if path.name == "complete.json":
            raise marker_error
        return original_publish(path, payload)

    monkeypatch.setattr(runner, "_publish_new", fail_marker)

    with pytest.raises(OSError) as raised:
        runner.run_native_paired_evaluation(manifest)

    failed = load_native_paired_evaluation_result(manifest.authorities.output_root / "result.json")
    assert raised.value is marker_error
    assert failed.status == "FAILED"
    assert failed.failure is not None
    assert failed.failure.category == "PUBLICATION_FAILURE"
    assert failed.failure.case_id is None
    assert len(failed.observations) == len(manifest.planned_cases)
    assert not (manifest.authorities.output_root / "complete.json").exists()


def test_result_publication_race_does_not_delete_or_replace_competing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    _isolate_gameplay(monkeypatch, manifest)
    monkeypatch.setattr(runner, "run_game", lambda *args, **kwargs: _outcome(winner="RED"))
    original_publish = runner._publish_new
    competing = b"competitor-owned\n"

    def race_result(path: Path, payload: bytes):
        if path.name == "result.json":
            path.write_bytes(competing)
            raise FileExistsError("publication race")
        return original_publish(path, payload)

    monkeypatch.setattr(runner, "_publish_new", race_result)

    with pytest.raises(FileExistsError, match="publication race") as raised:
        runner.run_native_paired_evaluation(manifest)

    assert (manifest.authorities.output_root / "result.json").read_bytes() == competing
    assert any("FAILED status" in note for note in (raised.value.__notes__ or ()))
    assert not (manifest.authorities.output_root / "complete.json").exists()


def test_seed_recipe_has_no_case_or_contestant_identity_material(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.planned_cases[0]
    derived = runner.derive_native_evaluation_seed(
        manifest,
        comparison=case.comparison,
        fixture_id=case.fixture_id,
        world_seed=case.world_seed,
        stream="RED_SEARCH",
    )
    expected_payload = json.dumps(
        (
            "native-gen1-paired-evaluation-stream-v1",
            manifest.config.random_stream_namespace,
            case.comparison,
            case.fixture_id,
            str(case.world_seed),
            "RED",
            "SEARCH",
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert derived == int.from_bytes(hashlib.sha256(expected_payload).digest(), "big")
