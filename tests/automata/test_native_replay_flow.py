"""Controlled played-game completion feeds native replay without split leakage.

The raw-stack fixture isolates the offline handoff, not a card effect: only the
next card's body is replaced with a terminal sequence. Actual input decisions,
search, finalization, stable-boundary observation, and outcome handling still run.
"""

from pathlib import Path

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.decision import DecisionDescriptor
from automata.harness.game_runner import DEFAULT_MAP, continue_game
from automata.observation import encode_decision
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from automata.search.ismcts.strategy import ISMCTSStrategy
from automata.training.native_dataset import NativeGameIdentity, native_game_id
from automata.training.native_indexed_dataset import (
    build_native_indexed_dataset,
    create_native_source_receipt_from_completions,
)
from automata.training.native_receipts import (
    NativeCompletionTarget,
    create_native_dataset_completion_receipt,
)
from automata.training.native_recorder import NativeDatasetRecorder
from automata.training.native_replay import (
    NativeReplayConfig,
    load_native_replay_catalog,
    sample_native_replay,
    update_native_replay_catalog,
)
from automata.training.native_splits import (
    NativeSeedRange,
    NativeSplitConfig,
    create_native_split_ledger,
    extend_native_split_ledger,
)
from automata.training.search_targets import search_policy_target_from_result
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


def _game(seed: int):
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=seed)
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


def _identity(seed: int, generation: str) -> NativeGameIdentity:
    fields = dict(
        world_seed=seed,
        map_id="forgotten_island",
        game_type="QUICK",
        red_composition=("Wasp",),
        blue_composition=("Arien",),
        generation_id=generation,
        source_revision="native-replay-flow-test",
        dirty_tree_hash="test-fixture",
        source_model_digest=None,
        search_config_id="stable-transition-fixture",
        generator_config_id="controlled-recorder-fixture",
    )
    return NativeGameIdentity(game_id=native_game_id(**fields), **fields)


class _PlayedPolicySink:
    strategy_id = "controlled-replay-test"

    def __init__(self, delegate, recorder):
        self.delegate = delegate
        self.recorder = recorder

    def select(self, state, team, root, legal):
        result = self.delegate.select(state, team, root, legal)
        owner = state.get_hero(HeroID(root.decision_owner_hero_id))
        descriptor = DecisionDescriptor(
            root.kind,
            hero=owner if root.kind == "CARD" else None,
            request=root.request,
            can_finish_planning=root.kind == "CARD" and None in legal,
        )
        observation = encode_decision(
            state,
            descriptor,
            legal,
            decision_owner_hero_id=root.decision_owner_hero_id,
            perspective_team=team.value,
        )
        self.recorder.record_policy(
            observation=observation,
            target=search_policy_target_from_result(result, observation.candidates),
            perspective_team=team.value,
        )
        return result


def _play(root: Path, *, seed: int, generation: str, max_steps: int = 20):
    destination = root / generation / "game.jsonl.zst"
    receipt_path = root / generation / "completion.json"
    config = SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TRANSITION, seed=19)
    policy = HeuristicAgent(19)
    with NativeDatasetRecorder(
        destination,
        game=_identity(seed, generation),
        completion_target=NativeCompletionTarget(source_root=root, receipt_path=receipt_path),
    ) as recorder:
        strategy = ISMCTSStrategy(
            environment_policy=policy,
            config=config,
            prior=HeuristicPrior(policy),
            leaf_evaluator=HeuristicLeafEvaluator(),
        )
        agent = ISMCTSAgent(config, strategy=_PlayedPolicySink(strategy, recorder))
        outcome = continue_game(
            _game(seed),
            {"hero_wasp": agent, "hero_arien": agent},
            max_steps=max_steps,
            boundary_observer=recorder,
        )
    return outcome, recorder, receipt_path


@pytest.fixture
def terminal_card(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(ResolveCardStep, "resolve", resolve)


def test_controlled_live_completion_to_persistent_replay_keeps_seed_splits(
    tmp_path: Path, terminal_card: None
) -> None:
    split_config = NativeSplitConfig(
        namespace="native-replay-flow-test",
        salt="fixed",
        validation_fraction=0.2,
        seed_ranges=(
            NativeSeedRange(purpose="training", start=100, stop=200),
            NativeSeedRange(purpose="validation", start=1000, stop=1010),
            NativeSeedRange(purpose="arena", start=2000, stop=2010),
        ),
    )
    probe = extend_native_split_ledger(create_native_split_ledger(split_config), range(100, 200))
    train_seed = next(seed for seed in range(100, 200) if probe.split_for_seed(seed) == "train")
    config = NativeReplayConfig(split_config=split_config, max_games=1)
    root = tmp_path / "data"
    root.mkdir()
    catalog_path = tmp_path / "catalog.json"
    train_game_ids = []
    for generation, seed in (
        ("train-one", train_seed),
        ("validation", 1000),
        ("train-two", train_seed),
    ):
        outcome, recorder, receipt_path = _play(root, seed=seed, generation=generation)
        assert outcome.reason == "game_over" and outcome.winner_side == "RED"
        assert recorder.completion_receipt is not None
        completion = create_native_dataset_completion_receipt(root, (receipt_path,))
        inventory = create_native_source_receipt_from_completions(root, completion)
        inventory_path = tmp_path / f"{generation}-inventory.json"
        inventory_path.write_bytes(inventory.canonical_bytes())
        dataset = build_native_indexed_dataset(
            root, inventory_path, tmp_path / f"{generation}-index", chunk_size=1
        )
        for game_id in dataset.game_ids:
            assert dataset.game(game_id).policy_row_count > 0
            assert dataset.game(game_id).value_row_count > 0
        catalog = update_native_replay_catalog(
            catalog_path,
            config=config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=None,
        )
        assert (
            load_native_replay_catalog(catalog_path).canonical_bytes() == catalog.canonical_bytes()
        )
        assert catalog.split_ledger.split_for_seed(train_seed) == "train"
        if seed == 1000:
            assert catalog.split_ledger.split_for_seed(seed) == "validation"
            assert all(game.game.world_seed != seed for game in catalog.games)
        else:
            train_game_ids.extend(dataset.game_ids)

    assert len(catalog.generations) == 3
    assert len(catalog.split_ledger.assignments) == 2
    assert len(catalog.games) == 1
    assert catalog.games[0].game.game_id == train_game_ids[-1]
    sample = sample_native_replay(catalog, game_count=1, sampling_seed=7)
    assert sample == sample_native_replay(catalog, game_count=1, sampling_seed=7)
    assert sample.game_ids == (train_game_ids[-1],)
    assert sample.policy_contributing_game_count == 1
    assert sample.value_contributing_game_count == 1
    assert any(train_game_ids[0] in generation.all_game_ids for generation in catalog.generations)


def test_censored_live_game_cannot_issue_completion_provenance(
    tmp_path: Path, terminal_card: None
) -> None:
    outcome, recorder, receipt_path = _play(tmp_path, seed=101, generation="censored", max_steps=2)
    assert outcome.reason != "game_over"
    assert recorder.completion_receipt is None
    assert not receipt_path.exists()
    assert not (receipt_path.parent / "game.jsonl.zst").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        create_native_dataset_completion_receipt(tmp_path, (receipt_path,))
