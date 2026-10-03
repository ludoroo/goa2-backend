"""Bounded actual-play → replay → optimizer → artifact → learned-play handoff.

This tests offline library wiring, not a card effect or playing strength. The
raw-stack terminal fixture mirrors test_native_replay_flow: real game setup,
input/search, finalization, boundaries, and terminal normalization still execute.
Only the remaining card body is replaced to keep the fixture deterministically
short. No training/generation experiment or production artifact is created.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from automata.models.shared_encoder.gen1_runtime import Gen1SharedEncoderRuntime
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode
from automata.training.native_dataset import iter_native_game_records
from automata.training.native_gen1 import (
    current_gen1_artifact_scope,
    current_gen1_runtime_requirements,
)
from automata.training.native_generation import (
    NativeGenerationConfig,
    NativeGenerationGame,
    NativeGenerationOutput,
    generate_native_game,
)
from automata.training.native_indexed_dataset import (
    build_native_indexed_dataset,
    create_native_source_receipt_from_completions,
)
from automata.training.native_receipts import create_native_dataset_completion_receipt
from automata.training.native_replay import (
    NativeReplayConfig,
    sample_native_replay,
    update_native_replay_catalog,
)
from automata.training.native_splits import (
    NativeSeedRange,
    NativeSplitConfig,
    create_native_split_ledger,
    extend_native_split_ledger,
)
from automata.training.native_trainer import (
    NativeReplayDatasetBinding,
    NativeTrainerConfig,
    NativeTrainerInitialization,
    bind_native_replay_sample,
    create_native_trainer,
)
from goa2.domain.models import CardState, GamePhase, TargetType
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.setup import GameSetup
from goa2.engine.steps import (
    FinalizeHeroTurnStep,
    ResolveCardStep,
    SelectStep,
    StepResult,
    TriggerGameOverStep,
)


@pytest.fixture
def terminal_game(monkeypatch: pytest.MonkeyPatch) -> None:
    original_setup = GameSetup.create_game

    def create_game(*args, **kwargs):
        state = original_setup(*args, **kwargs)
        state.phase = GamePhase.RESOLUTION
        state.pending_inputs.clear()
        state.execution_stack.clear()
        state.current_actor_id = HeroID("hero_wasp")
        state.resolution_owner_id = HeroID("hero_wasp")
        state.unresolved_hero_ids = [HeroID("hero_arien")]
        for hero_id in ("hero_wasp", "hero_arien"):
            hero = state.get_hero(HeroID(hero_id))
            assert hero is not None
            card = hero.hand.pop()
            card.state = CardState.UNRESOLVED
            card.is_facedown = False
            hero.current_turn_card = card
        push_steps(
            state,
            [
                SelectStep(
                    target_type=TargetType.NUMBER,
                    number_options=[1, 2],
                    prompt="Played root choice",
                    override_player_id="hero_wasp",
                ),
                FinalizeHeroTurnStep(hero_id="hero_wasp"),
            ],
        )
        return state

    def resolve(step, state, context):
        return StepResult(
            is_finished=True,
            new_steps=[
                SelectStep(
                    target_type=TargetType.NUMBER,
                    number_options=[1, 2],
                    prompt="Played next-actor choice",
                    override_player_id=str(step.hero_id),
                ),
                TriggerGameOverStep(individual_winner_id=HeroID("hero_wasp"), condition="TEST"),
            ],
        )

    monkeypatch.setattr(GameSetup, "create_game", staticmethod(create_game))
    monkeypatch.setattr(ResolveCardStep, "resolve", resolve)


def _trainer_config() -> NativeTrainerConfig:
    return NativeTrainerConfig(
        seed=7,
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        policy_weight=1.0,
        value_weight=1.0,
        entropy_weight=0.01,
        l2_weight=0.0001,
        max_gradient_norm=1.0,
        token_width=8,
        state_width=12,
        candidate_width=8,
        message_passing_layers=1,
        dropout=0.0,
    )


def _index_completion(root: Path, output: NativeGenerationOutput):
    completion = create_native_dataset_completion_receipt(
        output.source_root, (output.completion_receipt_path,)
    )
    inventory = create_native_source_receipt_from_completions(output.source_root, completion)
    inventory_path = root / "inventory.json"
    completion_path = root / "completion-set.json"
    inventory_path.write_bytes(inventory.canonical_bytes())
    completion_path.write_bytes(completion.canonical_bytes())
    cache = root / "index"
    dataset = build_native_indexed_dataset(output.source_root, inventory_path, cache, chunk_size=1)
    binding = NativeReplayDatasetBinding(
        dataset_digest=dataset.digest,
        source_digest=dataset.source_digest,
        completion_receipt_digest=completion.digest,
        source_root=output.source_root,
        source_receipt_path=inventory_path,
        completion_receipt_path=completion_path,
        index_cache_dir=cache,
        chunk_size=1,
    )
    return dataset, completion, binding


def test_actual_native_generation_to_trainer_export_and_learned_generation(
    tmp_path: Path, terminal_game: None
) -> None:
    split = NativeSplitConfig(
        namespace="native-trainer-generator-flow",
        salt="fixed",
        validation_fraction=0.2,
        seed_ranges=(
            NativeSeedRange(purpose="training", start=100, stop=200),
            NativeSeedRange(purpose="validation", start=1000, stop=1010),
        ),
    )
    ledger = extend_native_split_ledger(create_native_split_ledger(split), range(100, 200))
    train_seed = next(item.world_seed for item in ledger.assignments if item.split == "train")
    config = NativeGenerationConfig(
        generation_id="bootstrap",
        source_revision="native-flow-test",
        dirty_tree_hash="fixture",
        teacher_kind="HEURISTIC_BOOTSTRAP",
        source_model_digest=None,
        search_config=SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TRANSITION),
        split_config=split,
        random_stream_namespace="native-flow",
        visit_temperature=1.0,
        max_steps=20,
        max_rounds=None,
    )
    replay_config = NativeReplayConfig(split_config=split)
    catalog_path = tmp_path / "catalog.json"
    bindings = []
    training_output = None
    for generation, seed, purpose in (
        ("bootstrap", train_seed, "training"),
        ("held-out", 1000, "validation"),
    ):
        folder = tmp_path / generation
        source = folder / "source"
        source.mkdir(parents=True)
        output = NativeGenerationOutput(
            source_root=source,
            logical_name="game.jsonl.zst",
            completion_receipt_path=folder / "completed.json",
        )
        result = generate_native_game(
            NativeGenerationGame(
                world_seed=seed,
                seed_purpose=purpose,
                map_id="forgotten_island",
                game_type="QUICK",
                red_composition=("Wasp",),
                blue_composition=("Arien",),
            ),
            replace(config, generation_id=generation),
            output,
        )
        assert result.completed and result.outcome.winner_side == "RED"
        assert result.completion_receipt is not None
        assert result.completion_receipt.policy_row_count > 0
        assert result.completion_receipt.value_row_count > 0
        dataset, completion, binding = _index_completion(folder, output)
        catalog = update_native_replay_catalog(
            catalog_path,
            config=replay_config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=None,
        )
        if purpose == "training":
            bindings.append(binding)
            training_output = output

    sample = sample_native_replay(catalog, game_count=1, sampling_seed=13)
    assert all(item.game.world_seed == train_seed for item in sample.games)
    assert catalog.split_ledger.split_for_seed(1000) == "validation"
    assert catalog.compatibility is not None
    incompatible = catalog.model_copy(
        update={
            "compatibility": catalog.compatibility.model_copy(
                update={"decision_tensor_schema_digest": "f" * 64}
            )
        }
    )
    with pytest.raises(ValueError):
        bind_native_replay_sample(incompatible, sample, bindings=tuple(bindings))
    bound = bind_native_replay_sample(catalog, sample, bindings=tuple(bindings))
    trainer = create_native_trainer(
        _trainer_config(), initialization=NativeTrainerInitialization(mode="FRESH_BOOTSTRAP")
    )
    before = {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }
    step = trainer.train_logical_batch(bound)
    assert step.optimizer_step == 1
    assert step.provenance.selected_game_ids == sample.game_ids
    assert step.provenance.policy_contributing_game_count == 1
    assert step.provenance.value_contributing_game_count == 1
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in trainer.model.named_parameters()
    )
    artifact_path = tmp_path / "child-artifact"
    artifact = trainer.export_artifact(
        artifact_path, step=step, source_revision="native-flow-test", dirty_tree_hash="fixture"
    )
    scope = current_gen1_artifact_scope()
    assert artifact.supported_heroes == scope.supported_heroes
    assert artifact.supported_maps == scope.supported_maps
    assert artifact.supported_game_types == scope.supported_game_types
    assert "Brogan" in artifact.supported_heroes
    runtime = Gen1SharedEncoderRuntime.from_artifact(
        artifact_path,
        requirements=current_gen1_runtime_requirements(
            heroes=("Wasp", "Arien"), map_id="forgotten_island", game_type="QUICK"
        ),
    )
    assert training_output is not None
    rows = list(
        iter_native_game_records(training_output.source_root / training_output.logical_name)
    )
    policy_row = next(row for row in rows if row.sample_kind == "POLICY")
    value_row = next(row for row in rows if row.sample_kind == "VALUE")
    assert runtime.evaluate_policy(policy_row.observation).candidate_ids
    assert -1.0 <= runtime.evaluate_stable_value(value_row.observation).value <= 1.0
    rng_before_parent_load = torch.random.get_rng_state().clone()
    warm = create_native_trainer(
        _trainer_config(),
        initialization=NativeTrainerInitialization(
            mode="GEN1_PARENT", parent_model_digest=artifact.model_digest
        ),
        parent_artifact_path=artifact_path,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before_parent_load)
    assert warm.optimizer_steps == 0 and not warm.optimizer.state
    for name, parameter in warm.model.named_parameters():
        assert torch.equal(parameter, trainer.model.state_dict()[name])

    learned_folder = tmp_path / "learned"
    learned_source = learned_folder / "source"
    learned_source.mkdir(parents=True)
    learned_output = NativeGenerationOutput(
        source_root=learned_source,
        logical_name="game.jsonl.zst",
        completion_receipt_path=learned_folder / "completed.json",
    )
    game = NativeGenerationGame(
        world_seed=train_seed,
        seed_purpose="training",
        map_id="forgotten_island",
        game_type="QUICK",
        red_composition=("Wasp",),
        blue_composition=("Arien",),
    )
    learned_config = replace(
        config,
        generation_id="learned",
        teacher_kind="GEN1_PARENT",
        source_model_digest=artifact.model_digest,
    )
    learned = generate_native_game(
        game, learned_config, learned_output, parent_artifact_path=artifact_path
    )
    assert learned.completed
    assert learned.game.source_model_digest == artifact.model_digest
    dataset, completion, _ = _index_completion(learned_folder, learned_output)
    appended = update_native_replay_catalog(
        catalog_path,
        config=replay_config,
        dataset=dataset,
        completion_receipt=completion,
        parent_artifact=artifact,
    )
    assert len(appended.games) == 2
    assert appended.split_ledger == catalog.split_ledger

    censored_output = NativeGenerationOutput(
        source_root=learned_source,
        logical_name="censored.jsonl.zst",
        completion_receipt_path=learned_folder / "censored-completed.json",
    )
    censored = generate_native_game(
        game,
        replace(learned_config, generation_id="censored", max_steps=2),
        censored_output,
        parent_artifact_path=artifact_path,
    )
    assert not censored.completed and censored.completion_receipt is None
    assert not (learned_source / censored_output.logical_name).exists()
    assert not censored_output.completion_receipt_path.exists()
