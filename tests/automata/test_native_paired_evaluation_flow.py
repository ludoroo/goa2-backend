"""Real native train/export → paired gameplay handoff, not a strength experiment.

The reused raw-stack terminal fixture preserves real setup, input ownership,
search, inference, and terminal normalization. It replaces a remaining card body
only to bound offline wiring tests; it is not a character-effect test.
"""

import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from test_native_run_flow import _config as _run_config
from test_native_trainer_generator_flow import terminal_game as terminal_game

from automata.models.shared_encoder.artifacts.gen1_io import export_gen1_model_artifact
from automata.models.shared_encoder.gen1_model import Gen1ModelConfig, Gen1PolicyValueModel
from automata.models.shared_encoder.gen1_runtime import Gen1SharedEncoderRuntime
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import ComponentInferenceError, LeafMode
from automata.training.native_gen1 import current_gen1_artifact_scope
from automata.training.native_paired import (
    derive_native_evaluation_seed,
    run_native_paired_evaluation,
)
from automata.training.native_paired_contracts import (
    NativeEvaluationFixture,
    NativePairedEvaluationAuthorities,
    NativePairedEvaluationConfig,
    create_native_paired_evaluation_manifest,
    load_completed_native_paired_evaluation,
    load_native_paired_evaluation_result,
)
from automata.training.native_run import run_native_one
from automata.training.native_run_contracts import (
    NativeRunAuthorities,
    NativeRunSearchConfig,
    create_native_run_manifest,
)
from automata.training.native_splits import NativeSeedRange, NativeSplitConfig
from goa2.domain.models import GamePhase, TargetType
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.setup import GameSetup
from goa2.engine.steps import EndPhaseCleanupStep, SelectStep

_COMPARISONS = (
    "CANDIDATE_VS_HEURISTIC_FULL_SEARCH",
    "CANDIDATE_VALUE_VS_HEURISTIC_VALUE_FIXED_CANDIDATE_POLICY",
    "CANDIDATE_VS_HEURISTIC_POLICY_ONLY",
)


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _split() -> NativeSplitConfig:
    old = _run_config().split_config
    return old.model_copy(
        update={
            "seed_ranges": (
                *old.seed_ranges,
                NativeSeedRange(purpose="evaluation", start=2000, stop=2010),
            )
        }
    )


def _evaluation_config(model_digest: str, *, max_steps: int = 20):
    return NativePairedEvaluationConfig(
        schema_version=1,
        evaluation_id="native-paired-flow",
        source_revision="native-paired-flow-test",
        dirty_tree_hash="fixture",
        candidate_model_digest=model_digest,
        parent_model_digest=None,
        split_config=_split(),
        comparisons=_COMPARISONS,
        fixtures=(
            NativeEvaluationFixture(
                fixture_id="wasp-arien",
                map_id="forgotten_island",
                game_type="QUICK",
                red_composition=("Wasp",),
                blue_composition=("Arien",),
                world_seeds=(2000,),
            ),
        ),
        search=NativeRunSearchConfig(
            **asdict(SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TRANSITION))
        ),
        random_stream_namespace="native-paired-flow",
        max_steps=max_steps,
        max_rounds=4,
    )


def _fresh_artifact(root: Path) -> tuple[Path, str]:
    # Match a real generation/export process before capturing complete scope.
    register_all_effects()
    decision = TensorFeatureSchema.current()
    value = StableValueTensorSchema.current()
    config = Gen1ModelConfig(
        decision_schema_digest=decision.digest,
        stable_value_schema_digest=value.digest,
        token_width=8,
        state_width=12,
        candidate_width=8,
        message_passing_layers=1,
        dropout=0.0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(19)
        model = Gen1PolicyValueModel(
            decision_schema=decision, stable_value_schema=value, config=config
        )
    artifact = export_gen1_model_artifact(
        root,
        model=model,
        decision_schema=decision,
        stable_value_schema=value,
        scope=current_gen1_artifact_scope(),
        provenance={"purpose": "bounded-pytest-fixture"},
    )
    return root, artifact.model_digest


def test_seed_namespace_and_fixture_boundaries_cannot_alias(tmp_path: Path) -> None:
    base = _evaluation_config("a" * 64)
    comparison = _COMPARISONS[0]
    manifests = []
    for index, (namespace, fixture_id) in enumerate(
        ((f"n\0{comparison}\0f", "g"), ("n", f"f\0{comparison}\0g"))
    ):
        config = base.model_copy(
            update={
                "random_stream_namespace": namespace,
                "fixtures": (base.fixtures[0].model_copy(update={"fixture_id": fixture_id}),),
            }
        )
        manifests.append(
            create_native_paired_evaluation_manifest(
                config,
                NativePairedEvaluationAuthorities(
                    output_root=tmp_path / f"evaluation-{index}",
                    candidate_artifact_path=tmp_path / "candidate",
                    parent_artifact_path=None,
                ),
            )
        )
    assert manifests[0].config.digest != manifests[1].config.digest
    seeds = tuple(
        derive_native_evaluation_seed(
            manifest,
            comparison=comparison,
            fixture_id=manifest.config.fixtures[0].fixture_id,
            world_seed=2000,
            stream="RED_SEARCH",
        )
        for manifest in manifests
    )
    assert seeds[0] != seeds[1]


def test_actual_trained_artifact_to_all_three_paired_gameplay_comparisons(
    tmp_path: Path, terminal_game: None
) -> None:
    base = _run_config()
    split = _split()
    training_config = base.model_copy(
        update={
            "split_config": split,
            "replay_config": base.replay_config.model_copy(update={"split_config": split}),
            "games": base.games[:2],
            "optimizer_steps": 1,
            "sampling_seeds": (11,),
        }
    )
    training_root = tmp_path / "training"
    trained = run_native_one(
        create_native_run_manifest(
            training_config,
            NativeRunAuthorities(output_root=training_root, parent_artifact_path=None),
        )
    )
    assert trained.products is not None
    training_before = _hash_tree(training_root)
    root = tmp_path / "evaluation"
    manifest = create_native_paired_evaluation_manifest(
        _evaluation_config(trained.products.artifact_model_digest),
        NativePairedEvaluationAuthorities(
            output_root=root,
            candidate_artifact_path=training_root / "artifact",
            parent_artifact_path=None,
        ),
    )
    rng_before = torch.random.get_rng_state().clone()
    result = run_native_paired_evaluation(manifest)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert result.status == "SUCCEEDED"
    assert result.evaluation_scope == "DECLARED_EVALUATION_SEEDS"
    assert result.artifact_training_exposure == "UNKNOWN"
    assert result.strength_claim == "DESCRIPTIVE_COMPLETED_PAIRS_ONLY"
    assert load_completed_native_paired_evaluation(root) == (manifest, result)
    assert result.planned_case_count == 6
    assert result.attempted_case_ids == tuple(case.case_id for case in manifest.planned_cases)
    assert len(result.observations) == 6
    assert len(result.completed_pairs) == 3
    assert len(result.artifact_identities) == 1
    assert result.artifact_identities[0].model_digest == trained.products.artifact_model_digest
    for planned, observed in zip(manifest.planned_cases, result.observations, strict=True):
        assert observed.case_id == planned.case_id
        assert observed.pair_id == planned.pair_id
        assert observed.candidate_side == planned.candidate_side
        assert observed.world_seed == 2000
        assert observed.status == "COMPLETED"
        assert observed.reason == "game_over"
        # The bounded fixture always declares hero_wasp, fixed on RED, winner.
        assert observed.raw_winner == "hero_wasp"
        assert observed.winner_side == "RED"
    for pair in result.completed_pairs:
        assert pair.candidate_red_won is True
        assert pair.candidate_blue_won is False
        assert pair.candidate_pair_score == 0.5
    for aggregate in result.aggregates:
        assert aggregate.planned_pair_count == 1
        assert aggregate.completed_pair_count == 1
        assert aggregate.censored_case_count == 0
        assert aggregate.excluded_completed_singletons == 0
        assert aggregate.candidate_game_wins_in_completed_pairs == 1
        assert aggregate.comparator_game_wins_in_completed_pairs == 1
        assert aggregate.descriptive_mean_completed_pair_score == 0.5
    assert _hash_tree(training_root) == training_before
    assert not tuple(root.rglob("*.jsonl.zst"))
    assert not any((root / name).exists() for name in ("replay", "receipts", "index", "artifact"))


def test_actual_censored_pairs_finish_declared_schedule_without_scores_or_replacements(
    tmp_path: Path, terminal_game: None
) -> None:
    artifact_path, digest = _fresh_artifact(tmp_path / "candidate")
    artifact_before = _hash_tree(artifact_path)
    root = tmp_path / "censored"
    manifest = create_native_paired_evaluation_manifest(
        _evaluation_config(digest, max_steps=1),
        NativePairedEvaluationAuthorities(
            output_root=root, candidate_artifact_path=artifact_path, parent_artifact_path=None
        ),
    )
    result = run_native_paired_evaluation(manifest)
    assert result.status == "SUCCEEDED"  # the protocol finished, not six scored games
    assert result.attempted_case_ids == tuple(case.case_id for case in manifest.planned_cases)
    assert len(result.observations) == 6
    assert result.completed_pairs == ()
    for observed in result.observations:
        assert observed.status == "CENSORED"
        assert observed.reason == "max_steps"
        assert observed.winner_side is None
        assert observed.raw_winner is None
    for aggregate in result.aggregates:
        assert aggregate.attempted_case_count == 2
        assert aggregate.completed_case_count == 0
        assert aggregate.censored_case_count == 2
        assert aggregate.completed_pair_count == 0
        assert aggregate.censored_pair_count == 1
        assert aggregate.descriptive_mean_completed_pair_score is None
        assert aggregate.candidate_game_wins_in_completed_pairs == 0
        assert aggregate.comparator_game_wins_in_completed_pairs == 0
    assert load_completed_native_paired_evaluation(root) == (manifest, result)
    assert _hash_tree(artifact_path) == artifact_before


def test_actual_round_limit_censor_preserves_harness_round_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Offline handoff fixture: a real choice and real end-round cleanup reach
    # round 2. The harness's max_rounds=1 guard then returns a normal censor.
    original_setup = GameSetup.create_game

    def end_of_round_setup(*args, **kwargs):
        state = original_setup(*args, **kwargs)
        state.phase = GamePhase.RESOLUTION
        state.pending_inputs.clear()
        state.execution_stack.clear()
        state.current_actor_id = HeroID("hero_wasp")
        state.resolution_owner_id = HeroID("hero_wasp")
        push_steps(
            state,
            [
                SelectStep(
                    target_type=TargetType.NUMBER,
                    number_options=[1, 2],
                    prompt="Complete the round",
                    override_player_id="hero_wasp",
                ),
                EndPhaseCleanupStep(),
            ],
        )
        return state

    monkeypatch.setattr(GameSetup, "create_game", staticmethod(end_of_round_setup))
    artifact_path, digest = _fresh_artifact(tmp_path / "candidate")
    root = tmp_path / "round-limit"
    manifest = create_native_paired_evaluation_manifest(
        _evaluation_config(digest).model_copy(update={"max_rounds": 1}),
        NativePairedEvaluationAuthorities(
            output_root=root, candidate_artifact_path=artifact_path, parent_artifact_path=None
        ),
    )
    result = run_native_paired_evaluation(manifest)
    assert result.status == "SUCCEEDED"
    assert result.attempted_case_ids == tuple(case.case_id for case in manifest.planned_cases)
    assert len(result.observations) == 6
    assert result.completed_pairs == ()
    for observed in result.observations:
        assert observed.status == "CENSORED"
        assert observed.reason == "max_rounds"
        assert observed.rounds == 2
        assert observed.steps > 0
        assert observed.winner_side is None and observed.raw_winner is None
    assert all(item.descriptive_mean_completed_pair_score is None for item in result.aggregates)
    assert load_completed_native_paired_evaluation(root) == (manifest, result)


def test_actual_policy_inference_error_is_failed_not_censored_or_heuristic_fallback(
    tmp_path: Path, terminal_game: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path, digest = _fresh_artifact(tmp_path / "candidate")
    artifact_before = _hash_tree(artifact_path)
    root = tmp_path / "inference-failure"
    manifest = create_native_paired_evaluation_manifest(
        _evaluation_config(digest),
        NativePairedEvaluationAuthorities(
            output_root=root, candidate_artifact_path=artifact_path, parent_artifact_path=None
        ),
    )
    original = ComponentInferenceError("injected real-decision inference failure")

    def fail_policy(self, observation):
        raise original

    monkeypatch.setattr(Gen1SharedEncoderRuntime, "evaluate_policy", fail_policy)
    with pytest.raises(ComponentInferenceError) as caught:
        run_native_paired_evaluation(manifest)
    assert caught.value is original
    failed = load_native_paired_evaluation_result(root / "result.json")
    assert failed.status == "FAILED"
    assert failed.failure is not None
    assert failed.failure.category == "INFERENCE_FAILURE"
    assert failed.failure.case_id == manifest.planned_cases[0].case_id
    assert failed.attempted_case_ids == (manifest.planned_cases[0].case_id,)
    assert failed.observations == ()
    assert failed.completed_pairs == () and failed.aggregates == ()
    assert not (root / "complete.json").exists()
    assert _hash_tree(artifact_path) == artifact_before
