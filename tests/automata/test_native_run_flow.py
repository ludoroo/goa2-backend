"""Actual-play one-run handoff, not a learning or playing-strength experiment.

Reuse the existing raw-stack terminal fixture: it preserves real setup, search,
inputs, actor finalization, boundaries, and terminal normalization. Its replaced
card body bounds offline wiring tests; this does not test a character effect.
"""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from test_native_trainer_generator_flow import (
    _trainer_config,
)
from test_native_trainer_generator_flow import (
    terminal_game as terminal_game,
)

from automata.models.shared_encoder.gen1_runtime import Gen1SharedEncoderRuntime
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode
from automata.training import native_run
from automata.training.native_dataset import iter_native_game_records
from automata.training.native_gen1 import (
    current_gen1_artifact_scope,
    current_gen1_runtime_requirements,
    load_current_gen1_parent_artifact,
)
from automata.training.native_generation import NativeGenerationGame
from automata.training.native_replay import (
    NativeReplayConfig,
    load_native_replay_catalog,
    sample_native_replay,
)
from automata.training.native_run import run_native_one
from automata.training.native_run_contracts import (
    NativeRunAuthorities,
    NativeRunConfig,
    NativeRunSearchConfig,
    create_native_run_manifest,
    load_completed_native_run,
    load_native_run_result,
)
from automata.training.native_splits import (
    NativeSeedRange,
    NativeSplitConfig,
    create_native_split_ledger,
    extend_native_split_ledger,
)
from automata.training.native_trainer import NativeTrainerInitialization
from goa2.domain.models import TargetType
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.setup import GameSetup
from goa2.engine.steps import SelectStep, TriggerGameOverStep


def _config(*, parent_digest: str | None = None, max_steps: int = 20) -> NativeRunConfig:
    split = NativeSplitConfig(
        namespace="native-one-run-flow",
        salt="fixed",
        validation_fraction=0.2,
        seed_ranges=(
            NativeSeedRange(purpose="training", start=100, stop=200),
            NativeSeedRange(purpose="validation", start=1000, stop=1010),
        ),
    )
    ledger = extend_native_split_ledger(create_native_split_ledger(split), range(100, 200))
    seeds = [entry.world_seed for entry in ledger.assignments if entry.split == "train"]
    planned = ((seeds[0], "training"), (1000, "validation"), (seeds[1], "training"))
    games = tuple(
        NativeGenerationGame(
            world_seed=seed,
            seed_purpose=purpose,
            map_id="forgotten_island",
            game_type="QUICK",
            red_composition=("Wasp",),
            blue_composition=("Arien",),
        )
        for seed, purpose in planned
    )
    search = SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TRANSITION)
    return NativeRunConfig(
        run_id="warm-run" if parent_digest else "bootstrap-run",
        generation_id="warm-generation" if parent_digest else "bootstrap-generation",
        source_revision="native-run-flow-test",
        dirty_tree_hash="fixture",
        teacher_kind="GEN1_PARENT" if parent_digest else "HEURISTIC_BOOTSTRAP",
        source_model_digest=parent_digest,
        search=NativeRunSearchConfig(**asdict(search)),
        split_config=split,
        random_stream_namespace="native-run-flow",
        visit_temperature=1.0,
        max_steps=max_steps,
        max_rounds=None,
        games=games,
        replay_config=NativeReplayConfig(split_config=split, max_games=8),
        trainer_config=_trainer_config(),
        initialization=NativeTrainerInitialization(
            mode="GEN1_PARENT" if parent_digest else "FRESH_BOOTSTRAP",
            parent_model_digest=parent_digest,
        ),
        optimizer_steps=1 if parent_digest else 2,
        games_per_step=1,
        sampling_seeds=(31,) if parent_digest else (11, 12),
        index_chunk_size=1,
    )


def test_bounded_actual_run_publishes_split_isolated_metrics_and_parent_loadable_artifact(
    tmp_path: Path, terminal_game: None
) -> None:
    root = tmp_path / "bootstrap"
    config = _config()
    manifest = create_native_run_manifest(
        config, NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )
    rng_before = torch.random.get_rng_state().clone()
    result = run_native_one(manifest)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert result.status == "SUCCEEDED"
    assert result.products is not None
    products = result.products
    loaded_manifest, loaded_result = load_completed_native_run(root)
    assert loaded_manifest == manifest and loaded_result == result
    assert (root / "complete.json").is_file()
    assert products.validation_scope == "CURRENT_RUN_UPDATES"
    assert products.parent_training_exposure == "NONE_FRESH_INITIALIZATION"

    planned_ids = tuple(game.expected_game_id for game in manifest.planned_games)
    validation_ids = tuple(
        game.expected_game_id for game in manifest.planned_games if game.split == "validation"
    )
    train_ids = set(planned_ids) - set(validation_ids)
    assert len(validation_ids) == 1 and len(train_ids) == 2
    assert products.generated_game_ids == planned_ids
    catalog = load_native_replay_catalog(root / "replay" / "catalog.json")
    assert catalog.digest == products.catalog_digest
    assert catalog.split_ledger == manifest.planned_split_ledger
    assert {game.game.game_id for game in catalog.games} == train_ids
    assert len(catalog.generations) == 1
    assert len(products.training_steps) == config.optimizer_steps
    for index, step in enumerate(products.training_steps):
        assert step.sampling_seed == config.sampling_seeds[index]
        assert step.result.optimizer_step == index + 1
        assert len(step.selected_game_ids) == config.games_per_step
        assert (
            step.selected_game_ids
            == sample_native_replay(
                catalog, game_count=config.games_per_step, sampling_seed=step.sampling_seed
            ).game_ids
        )
        assert set(step.selected_game_ids) <= train_ids
        assert not set(step.selected_game_ids) & set(validation_ids)
        assert step.result.provenance.selected_game_ids == step.selected_game_ids
        assert step.result.provenance.replay_catalog_digest == catalog.digest

    initial = products.initial_validation
    final = products.final_validation
    for metrics in (initial, final):
        assert metrics.validation_scope == "CURRENT_RUN_UPDATES"
        assert metrics.game_ids == validation_ids
        assert metrics.validation_ledger_digest == products.validation_ledger_digest
        assert metrics.dataset_digests == (products.dataset_digest,)
        assert metrics.source_digests == (products.source_receipt_digest,)
        assert metrics.completion_receipt_digests == (products.completion_receipt_digest,)
        assert metrics.game_count == 1
        assert metrics.policy_contributing_game_count == 1
        assert metrics.value_contributing_game_count == 1
        assert metrics.policy_cross_entropy is not None
        assert metrics.policy_entropy is not None
        assert metrics.value_bce is not None
    assert initial.policy_row_count == final.policy_row_count
    assert initial.value_row_count == final.value_row_count

    parent = load_current_gen1_parent_artifact(
        root / "artifact", expected_model_digest=products.artifact_model_digest
    )
    assert parent.manifest.supported_heroes == current_gen1_artifact_scope().supported_heroes
    assert "Brogan" in parent.manifest.supported_heroes
    parent_provenance = json.loads((root / "artifact" / "provenance.json").read_text())
    assert len(parent_provenance["successful_steps"]) == config.optimizer_steps
    runtime = Gen1SharedEncoderRuntime.from_artifact(
        root / "artifact",
        requirements=current_gen1_runtime_requirements(
            heroes=("Wasp", "Arien"), map_id="forgotten_island", game_type="QUICK"
        ),
    )
    rows = list(iter_native_game_records(root / manifest.planned_games[0].source_logical_name))
    policy_row = next(row for row in rows if row.sample_kind == "POLICY")
    value_row = next(row for row in rows if row.sample_kind == "VALUE")
    assert runtime.evaluate_policy(policy_row.observation).candidate_ids
    assert -1.0 <= runtime.evaluate_stable_value(value_row.observation).value <= 1.0

    # A second one-run invocation is an explicit test fixture, not automatic iteration.
    # The inherited model's past validation exposure is UNKNOWN, not magically held out.
    warm_root = tmp_path / "warm"
    warm_config = _config(parent_digest=products.artifact_model_digest)
    warm_manifest = create_native_run_manifest(
        warm_config,
        NativeRunAuthorities(output_root=warm_root, parent_artifact_path=root / "artifact"),
    )
    warm = run_native_one(warm_manifest)
    assert warm.status == "SUCCEEDED" and warm.products is not None
    assert warm.products.parent_training_exposure == "UNKNOWN"
    assert warm.products.validation_scope == "CURRENT_RUN_UPDATES"
    assert len(warm.products.training_steps) == 1
    child = load_current_gen1_parent_artifact(
        warm_root / "artifact", expected_model_digest=warm.products.artifact_model_digest
    )
    assert child.manifest.model_digest == warm.products.artifact_model_digest
    child_provenance = json.loads((warm_root / "artifact" / "provenance.json").read_text())
    assert child_provenance["initialization"]["parent_model_digest"] == (
        products.artifact_model_digest
    )
    assert child_provenance["optimizer"]["step_count"] == 1


def test_censored_game_fails_run_without_replacement_and_preserves_earlier_game(
    tmp_path: Path, terminal_game: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_setup = GameSetup.create_game
    setups = []

    def first_game_finishes_earlier(*args, **kwargs):
        state = original_setup(*args, **kwargs)
        setups.append(state)
        if len(setups) == 1:
            state.execution_stack.clear()
            push_steps(
                state,
                [
                    SelectStep(
                        target_type=TargetType.NUMBER,
                        number_options=[1, 2],
                        prompt="One-input terminal fixture",
                        override_player_id="hero_wasp",
                    ),
                    TriggerGameOverStep(individual_winner_id=HeroID("hero_wasp"), condition="TEST"),
                ],
            )
        return state

    monkeypatch.setattr(GameSetup, "create_game", staticmethod(first_game_finishes_earlier))
    root = tmp_path / "censored"
    manifest = create_native_run_manifest(
        _config(max_steps=2), NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )
    with pytest.raises((RuntimeError, ValueError)):
        run_native_one(manifest)
    failed = load_native_run_result(root / "result.json")
    assert failed.status == "FAILED" and failed.products is None
    assert failed.failure is not None
    assert len(setups) == 2  # The third planned game is never attempted or substituted.
    first, second, third = manifest.planned_games
    assert failed.progress.completed_game_ids == (first.expected_game_id,)
    assert failed.progress.attempted_game_ids == (first.expected_game_id, second.expected_game_id)
    assert failed.progress.uncertified_game_ids == (second.expected_game_id,)
    assert failed.progress.optimizer_steps_completed == 0
    assert (root / first.source_logical_name).is_file()
    assert (root / first.completion_relative_path).is_file()
    assert not (root / second.source_logical_name).exists()
    assert not (root / third.source_logical_name).exists()
    assert not (root / "complete.json").exists()
    assert not (root / "artifact").exists()
    assert not (root / "receipts" / "completion-set.json").exists()
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        load_completed_native_run(root)


def test_marker_failure_preserves_competing_file_after_publication_race(
    tmp_path: Path, terminal_game: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "marker-race"
    base = _config()
    config = base.model_copy(
        update={"games": base.games[:2], "optimizer_steps": 1, "sampling_seeds": (11,)}
    )
    manifest = create_native_run_manifest(
        config, NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )
    marker = root / "complete.json"
    competing_bytes = b"a competing writer's marker must survive"
    original_link = native_run.os.link
    original_fsync = native_run.fsync_directory

    def replace_just_linked_marker(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if Path(destination) == marker:
            marker.unlink()
            marker.write_bytes(competing_bytes)

    def fail_marker_fsync(directory):
        if Path(directory) == root and marker.exists():
            raise OSError("simulated completion-marker fsync failure")
        original_fsync(directory)

    monkeypatch.setattr(native_run.os, "link", replace_just_linked_marker)
    monkeypatch.setattr(native_run, "fsync_directory", fail_marker_fsync)
    with pytest.raises(OSError, match="simulated completion-marker fsync failure"):
        run_native_one(manifest)
    assert marker.read_bytes() == competing_bytes
    assert load_native_run_result(root / "result.json").status == "FAILED"
    with pytest.raises((RuntimeError, ValueError)):
        load_completed_native_run(root)
