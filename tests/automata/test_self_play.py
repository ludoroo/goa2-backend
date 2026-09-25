from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.decision import DecisionSemanticRole
from automata.harness.game_runner import DEFAULT_MAP, RunResult
from automata.search import REQUEST_AWARE_SCHEDULE_V1_ID
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode
from automata.search.ismcts import (
    SearchProgressionDiagnostics,
    SearchProgressionError,
    SearchResult,
)
from automata.search.ismcts.strategy import ISMCTSStrategy, StrategyResult
from automata.search.node import Node
from automata.training.dataset import load_joint_dataset
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def _module() -> Any:
    return importlib.import_module("automata.training.generation")


def _config(
    *,
    visit_temperature: float = 0.0,
    visit_temperature_schedule: str = "constant",
    decision_timeout_seconds: float | None = None,
    random_stream_namespace: str | None = None,
) -> Any:
    module = _module()
    namespace = (
        {}
        if random_stream_namespace is None
        else {"random_stream_namespace": random_stream_namespace}
    )
    return module.GenerationConfig(
        generation_id="generation-7",
        parent_model_digest="a" * 64,
        parent_generation=6,
        observation_schema_version=4,
        source_revision="revision",
        dirty_tree_hash="clean",
        search_config={"iterations": 4},
        source_config={"map_path": DEFAULT_MAP, "recipe": "visits-v1"},
        max_steps=20,
        timeout_seconds=5.0,
        visit_temperature=visit_temperature,
        visit_temperature_schedule=visit_temperature_schedule,
        decision_timeout_seconds=decision_timeout_seconds,
        **namespace,
    )


def _games(*seeds: int) -> tuple[Any, ...]:
    module = _module()
    scope = PHASE0_EXPERIMENT
    return tuple(
        module.GameSpec(
            world_seed=seed,
            map_id=scope.map_id,
            map_path=DEFAULT_MAP,
            game_type=scope.game_type,
            red_composition=scope.red_heroes,
            blue_composition=scope.blue_heroes,
        )
        for seed in seeds
    )


class _ImprovedStrategy:
    strategy_id = "test-improved"

    def select(self, state: GameState, team: TeamColor, target: Any, legal: Any) -> Any:
        legal = tuple(legal)
        del state, team, target
        selected = Node(visits=4, total_value=1.0, total_squared_value=0.25)
        return StrategyResult(legal, 0, SearchResult(Node(children={legal[0]: selected}), legal[0]))


class _CapturingRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record_decision(self, **record: Any) -> None:
        self.records.append(record)


class _NoVisitStrategy:
    strategy_id = "test-no-visits"

    def select(self, state: GameState, team: TeamColor, target: Any, legal: Any) -> Any:
        legal = tuple(legal)
        del state, team, target
        return StrategyResult(legal, 0, SearchResult(Node(), legal[0]))


class _VisitProbeStrategy:
    strategy_id = "test-visit-probe"

    def select(self, state: GameState, team: TeamColor, target: Any, legal: Any) -> Any:
        legal = tuple(legal)
        del state, team, target
        return StrategyResult(
            legal,
            0,
            SearchResult(Node(children={legal[1]: Node(visits=1)}), legal[0]),
        )


class _SampledNonBestStrategy:
    strategy_id = "test-sampled-non-best"

    def select(self, state: GameState, team: TeamColor, target: Any, legal: Any) -> Any:
        legal = tuple(legal)
        del state, team, target
        root = Node(
            children={
                legal[0]: Node(visits=3, total_value=2.0, total_squared_value=2.0),
                legal[1]: Node(visits=1, total_value=0.25, total_squared_value=0.0625),
            }
        )
        return StrategyResult(legal, 1, SearchResult(root, legal[0]))


def _short_game(
    red: list[str], blue: list[str], agents: dict[str, Any], **kwargs: Any
) -> RunResult:
    state = GameSetup.create_game(
        kwargs["map_path"], red, blue, game_type=kwargs["game_type"], seed=kwargs["seed"]
    )
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    agents[hero.id].choose_planning(state, hero)
    result = RunResult("hero_wasp", 1, 1, 1, "game_over", winner_side="RED")
    kwargs["decision_observer"].record_outcome(winner_side="RED", rounds=1, reason="game_over")
    return result


def test_singleton_real_search_without_visits_records_aligned_forced_target() -> None:
    module = _module()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=3)
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    hero.hand = hero.hand[:1]
    expected_card_id = hero.hand[0].id

    recorder = _CapturingRecorder()
    delegate = ISMCTSStrategy(
        environment_policy=HeuristicAgent(3),
        config=SearchConfig(iterations=1, seed=3),
    )
    strategy = module._RecordingStrategy(delegate, recorder)
    agent = ISMCTSAgent(SearchConfig(iterations=1, seed=3), strategy=strategy)

    decision = agent.choose_planning(state, hero)

    assert decision.card is not None
    assert decision.card.id == expected_card_id
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record["policy_target"] == (1.0,)
    assert record["selected_candidate_id"] == record["observation"].candidates[0].candidate_id
    assert record["selected_selection"] == expected_card_id
    assert len(record["action_stats"]) == 1
    action = record["action_stats"][0]
    assert action.candidate == record["observation"].candidates[0]
    assert action.candidate.selection == expected_card_id
    assert action.sample_count == 0
    assert action.mean_value == 0.0
    assert action.value_variance == 0.0
    assert action.improved_probability == 1.0
    assert action.selected is True


def test_non_singleton_search_without_visits_still_fails_closed() -> None:
    module = _module()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=3)
    request = InputRequest(
        id="no-visit-request",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        prompt="Choose",
        options=[InputOption.from_value("FIRST"), InputOption.from_value("SECOND")],
    )
    recorder = _CapturingRecorder()
    strategy = module._RecordingStrategy(_NoVisitStrategy(), recorder)
    agent = ISMCTSAgent(SearchConfig(iterations=1), strategy=strategy)

    with pytest.raises(ValueError, match="search statistics contain no root visits"):
        agent.choose_input(
            state,
            request,
            owned_hero_ids=frozenset({"hero_wasp"}),
            decision_owner_hero_id="hero_wasp",
        )

    assert recorder.records == []


def test_automatic_input_without_pending_stack_records_exact_surfaced_request() -> None:
    module = _module()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=3)
    request = InputRequest(
        id="automatic-request",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        prompt="Choose the automatic action",
        options=[InputOption.from_value("ATTACK"), InputOption.from_value("MOVE")],
    )
    assert state.input_stack == []

    recorder = _CapturingRecorder()
    strategy = module._RecordingStrategy(_ImprovedStrategy(), recorder)
    agent = ISMCTSAgent(SearchConfig(iterations=1), strategy=strategy)

    selection = agent.choose_input(
        state,
        request,
        owned_hero_ids=frozenset({"hero_wasp"}),
        decision_owner_hero_id="hero_wasp",
    )

    assert selection == "ATTACK"
    assert len(recorder.records) == 1
    observation = recorder.records[0]["observation"]
    assert observation.decision_kind == "INPUT"
    assert tuple(candidate.selection for candidate in observation.candidates) == (
        "ATTACK",
        "MOVE",
    )


def test_recording_marks_sampled_non_best_action_without_changing_visit_target() -> None:
    module = _module()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=3)
    request = InputRequest(
        id="sampled-request",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        prompt="Choose",
        options=[InputOption.from_value("BEST"), InputOption.from_value("SAMPLED")],
    )
    recorder = _CapturingRecorder()
    strategy = module._RecordingStrategy(_SampledNonBestStrategy(), recorder)
    agent = ISMCTSAgent(SearchConfig(iterations=1), strategy=strategy)

    selection = agent.choose_input(
        state,
        request,
        owned_hero_ids=frozenset({"hero_wasp"}),
        decision_owner_hero_id="hero_wasp",
    )

    assert selection == "SAMPLED"
    record = recorder.records[0]
    assert record["policy_target"] == (0.75, 0.25)
    assert record["selected_selection"] == "SAMPLED"
    assert [action.sample_count for action in record["action_stats"]] == [3, 1]
    assert [action.selected for action in record["action_stats"]] == [False, True]


def test_recording_rejects_pending_request_that_differs_from_surfaced_request() -> None:
    module = _module()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=3)
    request = InputRequest(
        id="automatic-request",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        prompt="Surfaced prompt",
        options=[InputOption.from_value("ATTACK"), InputOption.from_value("MOVE")],
    )
    state.input_stack.append(request.model_copy(update={"prompt": "Different prompt"}))
    strategy = module._RecordingStrategy(_ImprovedStrategy(), _CapturingRecorder())
    agent = ISMCTSAgent(SearchConfig(iterations=1), strategy=strategy)

    with pytest.raises(ValueError, match="search root request does not match"):
        agent.choose_input(
            state,
            request,
            owned_hero_ids=frozenset({"hero_wasp"}),
            decision_owner_hero_id="hero_wasp",
        )


def test_worker_loads_once_builds_fresh_agents_and_records_improved_roots(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config()
    spec = module.WorkerSpec(worker_id=0, config=config, games=_games(20_000, 20_001))
    loads: list[object] = []
    strategies: list[object] = []

    def load_runtime(_config: Any) -> Any:
        loads.append(object())
        return module.LoadedChampionRuntime(
            runtime=object(),
            model_digest="a" * 64,
            generation=6,
            observation_schema_version=4,
        )

    def strategy_factory(_runtime: object, _game: Any, _side: str, _seed: int) -> Any:
        strategy = _ImprovedStrategy()
        strategies.append(strategy)
        return strategy

    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        runtime_loader=load_runtime,
        strategy_factory=strategy_factory,
        game_runner=_short_game,
    )

    assert worker.run() == 2
    assert len(loads) == 1
    assert len(strategies) == 4
    datasets = [
        load_joint_dataset(path) for path in sorted((tmp_path / "fragments").glob("*.jsonl.zst"))
    ]
    assert [dataset.rows[0].world_seed for dataset in datasets] == [20_000, 20_001]
    assert all(dataset.rows[0].policy_source == "ISMCTS_VISITS" for dataset in datasets)
    assert all(dataset.rows[0].action_stats is not None for dataset in datasets)
    assert all(dataset.rows[0].source_model_digest == "a" * 64 for dataset in datasets)


def test_partial_visit_budget_discards_the_entire_provisional_game(tmp_path: Path) -> None:
    module = _module()
    spec = module.WorkerSpec(worker_id=0, config=_config(), games=_games(20_007))

    class PartiallyCompletedStrategy(_ImprovedStrategy):
        def __init__(self) -> None:
            self.calls = 0

        def select(self, *args: Any) -> Any:
            result = super().select(*args)
            self.calls += 1
            # The first decision completes four visits; the second recovers
            # only four of an eight-visit root budget, as serving may do.
            return replace(
                result,
                search_result=replace(
                    result.search_result,
                    effective_iterations=4 if self.calls == 1 else 8,
                ),
            )

    def two_decisions(red: list[str], blue: list[str], agents: dict[str, Any], **kwargs: Any):
        state = GameSetup.create_game(
            kwargs["map_path"], red, blue, game_type=kwargs["game_type"], seed=kwargs["seed"]
        )
        hero = state.get_hero(HeroID("hero_wasp"))
        assert hero is not None
        agents[hero.id].choose_planning(state, hero)
        agents[hero.id].choose_planning(state, hero)
        kwargs["decision_observer"].record_outcome(winner_side="RED", rounds=1, reason="game_over")
        return RunResult("RED", 1, 2, 2, "game_over", winner_side="RED")

    checkpoint = tmp_path / "checkpoint.jsonl"
    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=checkpoint,
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        strategy_factory=lambda *_args: PartiallyCompletedStrategy(),
        game_runner=two_decisions,
    )

    with pytest.raises(ValueError, match="effective visit budget"):
        worker.run()

    assert not checkpoint.exists()
    assert not worker.fragment_path(spec.games[0]).exists()
    assert not list((tmp_path / "fragments").iterdir())


def test_worker_applies_configured_visit_sampling_only_to_self_play(tmp_path: Path) -> None:
    module = _module()
    spec = module.WorkerSpec(
        worker_id=0,
        config=_config(visit_temperature=1.0),
        games=_games(20_007),
    )

    def assert_sampled(
        red: list[str], blue: list[str], agents: dict[str, Any], **kwargs: Any
    ) -> RunResult:
        state = GameSetup.create_game(
            kwargs["map_path"],
            red,
            blue,
            game_type=kwargs["game_type"],
            seed=kwargs["seed"],
        )
        hero = state.get_hero(HeroID("hero_wasp"))
        assert hero is not None
        expected = hero.hand[1].id
        decision = agents[hero.id].choose_planning(state, hero)
        assert decision.card is not None
        assert decision.card.id == expected
        kwargs["decision_observer"].record_outcome(winner_side="RED", rounds=1, reason="game_over")
        return RunResult("RED", 1, 1, 1, "game_over", winner_side="RED")

    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        strategy_factory=lambda *_args: _VisitProbeStrategy(),
        game_runner=assert_sampled,
    )

    assert worker.run() == 1


def test_checkpoint_records_visit_temperature_schedule_through_generator_identity(
    tmp_path: Path,
) -> None:
    module = _module()
    constant = _config(visit_temperature=0.5)
    decay = _config(
        visit_temperature=0.5,
        visit_temperature_schedule="round-decay-v1",
    )
    game = _games(20_007)[0]
    spec = module.WorkerSpec(worker_id=0, config=decay, games=(game,))
    checkpoint = tmp_path / "checkpoint.jsonl"
    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=checkpoint,
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        strategy_factory=lambda *_args: _ImprovedStrategy(),
        game_runner=_short_game,
    )

    assert worker.run() == 1
    row = module.CheckpointRow.model_validate_json(checkpoint.read_bytes())
    assert row.generator_config_id == decay.generator_config_id
    assert row.generator_config_id != constant.generator_config_id
    assert row.game_id == game.game_id(decay)
    assert row.game_id != game.game_id(constant)


def test_resume_skips_complete_games_without_loading_or_duplicates(tmp_path: Path) -> None:
    module = _module()
    spec = module.WorkerSpec(worker_id=0, config=_config(), games=_games(20_002))
    loads = 0

    def load_runtime(_config: Any) -> Any:
        nonlocal loads
        loads += 1
        return module.LoadedChampionRuntime(object(), "a" * 64, 6, 4)

    kwargs = dict(
        output_dir=tmp_path / "fragments",
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        runtime_loader=load_runtime,
        strategy_factory=lambda *_args: _ImprovedStrategy(),
        game_runner=_short_game,
    )
    assert module.SelfPlayWorker(spec, **kwargs).run() == 1
    checkpoint = (tmp_path / "checkpoint.jsonl").read_bytes()
    assert module.SelfPlayWorker(spec, **kwargs).run() == 0
    assert loads == 1
    assert (tmp_path / "checkpoint.jsonl").read_bytes() == checkpoint


def test_outcome_disagreement_discards_fragment_and_resume_replays_game(tmp_path: Path) -> None:
    module = _module()
    spec = module.WorkerSpec(worker_id=0, config=_config(), games=_games(20_002))
    checkpoint = tmp_path / "checkpoint.jsonl"
    fragments = tmp_path / "fragments"
    common = dict(
        output_dir=fragments,
        checkpoint_path=checkpoint,
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        strategy_factory=lambda *_args: _ImprovedStrategy(),
    )

    def contradictory(*args: Any, **kwargs: Any) -> RunResult:
        _short_game(*args, **kwargs)  # Observer published RED.
        return RunResult("BLUE", 1, 1, 1, "game_over", winner_side="BLUE")

    with pytest.raises(ValueError, match="winner side disagree"):
        module.SelfPlayWorker(spec, game_runner=contradictory, **common).run()

    assert list(fragments.iterdir()) == []
    assert not checkpoint.exists()
    assert module.SelfPlayWorker(spec, game_runner=_short_game, **common).run() == 1
    receipt = module.CheckpointRow.model_validate_json(checkpoint.read_bytes())
    assert receipt.winner == "RED"


def test_resume_rejects_checkpoint_winner_that_disagrees_with_fragment(tmp_path: Path) -> None:
    from automata.models.contracts import canonical_json_bytes

    module = _module()
    spec = module.WorkerSpec(worker_id=0, config=_config(), games=_games(20_002))
    checkpoint = tmp_path / "checkpoint.jsonl"
    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=checkpoint,
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        strategy_factory=lambda *_args: _ImprovedStrategy(),
        game_runner=_short_game,
    )
    assert worker.run() == 1
    receipt = module.CheckpointRow.model_validate_json(checkpoint.read_bytes())
    contradictory = receipt.model_copy(update={"winner": "BLUE"})
    checkpoint.write_bytes(canonical_json_bytes(contradictory) + b"\n")

    with pytest.raises(ValueError, match="checkpoint and fragment disagree"):
        worker.run()


def test_outcome_contract_changes_generation_and_game_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _config()
    game = _games(20_002)[0]
    before_id = before.generator_config_id
    before_game_id = game.game_id(before)
    monkeypatch.setattr(_module(), "OUTCOME_CONTRACT", "different-outcome-contract")
    after = _config()

    assert before_id != after.generator_config_id
    assert before_game_id != game.game_id(after)


def test_four_worker_assignment_is_deterministic_disjoint_and_training_only() -> None:
    module = _module()
    specs = module.build_worker_specs(_config(), _games(20_003, 20_000, 20_002, 20_001))

    assert len(specs) == 4
    assert [[game.world_seed for game in spec.games] for spec in specs] == [
        [20_000],
        [20_001],
        [20_002],
        [20_003],
    ]
    with pytest.raises(ValueError, match="training"):
        module.build_worker_specs(_config(), _games(10_000))


def test_worker_fails_closed_on_runtime_or_checkpoint_identity_mismatch(tmp_path: Path) -> None:
    module = _module()
    spec = module.WorkerSpec(worker_id=0, config=_config(), games=_games(20_004))
    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "b" * 64, 6, 4),
        strategy_factory=lambda *_args: _ImprovedStrategy(),
        game_runner=_short_game,
    )
    with pytest.raises(ValueError, match=r"runtime.*digest"):
        worker.run()

    (tmp_path / "checkpoint.jsonl").write_text('{"schema_version":1}\n')
    with pytest.raises(ValueError, match="checkpoint"):
        module.SelfPlayWorker(
            spec,
            **{
                "output_dir": tmp_path / "fragments",
                "checkpoint_path": tmp_path / "checkpoint.jsonl",
                "runtime_loader": lambda _config: module.LoadedChampionRuntime(
                    object(), "a" * 64, 6, 3
                ),
                "strategy_factory": lambda *_args: _ImprovedStrategy(),
                "game_runner": _short_game,
            },
        ).run()


def test_interruption_after_fragment_publication_resumes_without_duplicate(
    tmp_path: Path,
) -> None:
    module = _module()
    spec = module.WorkerSpec(worker_id=2, config=_config(), games=_games(20_005))
    events: list[Any] = []
    runs = 0

    def interrupted(*args: Any, **kwargs: Any) -> RunResult:
        nonlocal runs
        runs += 1
        _short_game(*args, **kwargs)
        raise RuntimeError("worker stopped")

    common = {
        "output_dir": tmp_path / "fragments",
        "checkpoint_path": tmp_path / "checkpoint.jsonl",
        "runtime_loader": lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        "strategy_factory": lambda *_args: _ImprovedStrategy(),
        "telemetry": events.append,
    }
    with pytest.raises(RuntimeError, match="worker stopped"):
        module.SelfPlayWorker(spec, game_runner=interrupted, **common).run()
    assert events[-1].event == "error"

    assert module.SelfPlayWorker(spec, game_runner=_short_game, **common).run() == 0
    assert runs == 1
    assert len((tmp_path / "checkpoint.jsonl").read_text().splitlines()) == 1


def test_nonterminal_game_emits_timeout_telemetry_without_fragment_or_checkpoint(
    tmp_path: Path,
) -> None:
    module = _module()
    spec = module.WorkerSpec(worker_id=0, config=_config(), games=_games(20_006))
    events: list[Any] = []

    def capped(_red: Any, _blue: Any, _agents: Any, **kwargs: Any) -> RunResult:
        kwargs["decision_observer"].record_outcome(winner_side=None, rounds=1, reason="max_steps")
        return RunResult(None, 1, 1, 20, "max_steps", winner_side=None)

    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        strategy_factory=lambda *_args: _ImprovedStrategy(),
        game_runner=capped,
        telemetry=events.append,
    )
    assert worker.run() == 0
    assert events[-1].event == "timeout"
    assert events[-1].reason == "max_steps"
    assert not list((tmp_path / "fragments").glob("*.jsonl*"))
    assert not (tmp_path / "checkpoint.jsonl").exists()


def test_round_decay_visit_temperature_schedule_has_exact_round_boundaries() -> None:
    module = _module()
    provider = module.visit_temperature_provider("round-decay-v1", 0.5)

    assert [
        provider(SimpleNamespace(round=round_number)) for round_number in (1, 4, 5, 8, 9, 20)
    ] == [
        0.5,
        0.5,
        0.25,
        0.25,
        0.0,
        0.0,
    ]


def test_visit_temperature_schedule_is_validated_and_changes_content_identity() -> None:
    constant = _config(visit_temperature=0.5)
    decay = _config(visit_temperature=0.5, visit_temperature_schedule="round-decay-v1")
    game = _games(20_000)[0]

    assert constant.visit_temperature_schedule == "constant"
    assert constant.generator_config_id != decay.generator_config_id
    assert game.game_id(constant) != game.game_id(decay)
    with pytest.raises(ValueError, match="visit_temperature_schedule"):
        _config(visit_temperature_schedule="unknown")


def test_decision_timeout_is_disabled_by_default_and_part_of_generator_identity() -> None:
    default = _config()
    enabled = _config(decision_timeout_seconds=1.25)

    assert default.decision_timeout_seconds is None
    assert enabled.decision_timeout_seconds == 1.25
    assert default.generator_config_id != enabled.generator_config_id
    with pytest.raises(ValueError, match="decision_timeout_seconds"):
        _config(decision_timeout_seconds=0.0)


def test_namespaced_agent_seed_is_comparable_across_generation_knobs() -> None:
    module = _module()
    namespace = "fair-temperature-arms-v1"
    control = _config(random_stream_namespace=namespace, visit_temperature=0.0)
    treatment = replace(
        _config(random_stream_namespace=namespace, visit_temperature=1.5),
        search_config={"iterations": 8, "adaptive_schedule": "widening-v2"},
    )

    assert control.generator_config_id != treatment.generator_config_id
    assert module.agent_seed(control, 20_000, "RED") == module.agent_seed(treatment, 20_000, "RED")
    assert module.agent_seed(control, 20_000, "BLUE") == module.agent_seed(
        treatment, 20_000, "BLUE"
    )


def test_namespaced_agent_seed_separates_namespaces_and_content_identity() -> None:
    module = _module()
    first = _config(random_stream_namespace="experiment-arm-set-a")
    second = _config(random_stream_namespace="experiment-arm-set-b")
    game = _games(20_000)[0]

    assert module.agent_seed(first, game.world_seed, "RED") != module.agent_seed(
        second, game.world_seed, "RED"
    )
    assert first.generator_config_id != second.generator_config_id
    assert game.game_id(first) != game.game_id(second)


def test_default_agent_seed_and_identity_bind_the_current_schema_generation() -> None:
    module = _module()
    config = _config()

    # DEFAULT_MAP is checkout-relative in production configuration. Test the
    # identity contract, not a digest tied to one developer's absolute path.
    identical = _config()
    next_schema = replace(config, observation_schema_version=config.observation_schema_version + 1)
    assert config.generator_config_id == identical.generator_config_id
    assert config.generator_config_id != next_schema.generator_config_id
    seeds = {side: module.agent_seed(config, 20_000, side) for side in ("RED", "BLUE")}
    assert seeds["RED"] != seeds["BLUE"]
    for side, seed in seeds.items():
        assert seed == module.agent_seed(identical, 20_000, side)
        assert seed != module.agent_seed(next_schema, 20_000, side)
        assert seed != module.agent_seed(config, 20_001, side)


def test_random_stream_namespace_is_nonempty_and_bounded() -> None:
    for invalid in ("", "   ", "x" * 129):
        with pytest.raises(ValueError, match="random_stream_namespace"):
            _config(random_stream_namespace=invalid)


def test_namespaced_agent_seed_is_stable_across_worker_sharding() -> None:
    module = _module()
    base = _config(random_stream_namespace="fair-sharding-v1")
    two_config = replace(base, source_config={**base.source_config, "worker_count": 2})
    four_config = replace(base, source_config={**base.source_config, "worker_count": 4})
    games = _games(20_000, 20_001, 20_002, 20_003)
    two_workers = module.build_worker_specs(two_config, games, worker_count=2)
    four_workers = module.build_worker_specs(four_config, games, worker_count=4)

    def seeds_by_world(specs: tuple[Any, ...]) -> dict[tuple[int, str], int]:
        return {
            (game.world_seed, side): module.agent_seed(spec.config, game.world_seed, side)
            for spec in specs
            for game in spec.games
            for side in ("RED", "BLUE")
        }

    assert two_config.generator_config_id != four_config.generator_config_id
    assert seeds_by_world(two_workers) == seeds_by_world(four_workers)


def test_nested_decision_deadline_restores_elapsed_game_deadline() -> None:
    if not hasattr(signal, "setitimer"):
        pytest.skip("POSIX interval timers are unavailable")
    module = _module()

    with module._game_timeout(0.5):
        outer_before = signal.getitimer(signal.ITIMER_REAL)[0]
        with module._decision_timeout(0.2):
            inner = signal.getitimer(signal.ITIMER_REAL)[0]
            assert 0 < inner <= 0.2
            time.sleep(0.04)
        outer_after = signal.getitimer(signal.ITIMER_REAL)[0]

        assert outer_after < outer_before - 0.02
        assert outer_after > 0.25


def test_nested_decision_deadline_preserves_whichever_timeout_expires_first() -> None:
    if not hasattr(signal, "setitimer"):
        pytest.skip("POSIX interval timers are unavailable")
    module = _module()

    with (
        module._game_timeout(0.5),
        pytest.raises(module.SourceDecisionTimeout),
        module._decision_timeout(0.02),
    ):
        time.sleep(0.1)

    with (
        pytest.raises(module.SourceGameTimeout),
        module._game_timeout(0.02),
        module._decision_timeout(0.5),
    ):
        time.sleep(0.1)


def test_decision_plan_holder_does_not_leak_into_a_later_preplan_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=3)
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    hero.hand = hero.hand[:1]
    legal = (hero.hand[0].id,)
    target = module.RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id}))
    delegate = ISMCTSStrategy(
        environment_policy=HeuristicAgent(3),
        config=SearchConfig(iterations=1, seed=3, request_schedule_version=1),
    )
    events: list[tuple[str, Any]] = []
    timeout_calls = 0

    @contextmanager
    def timeout_on_second_call(_seconds: float | None) -> Any:
        nonlocal timeout_calls
        timeout_calls += 1
        if timeout_calls == 2:
            raise module.SourceDecisionTimeout
        yield

    def capture(*args: Any) -> None:
        events.append((args[0], args[-2]))

    monkeypatch.setattr(module, "_decision_timeout", timeout_on_second_call)
    strategy = module._RecordingStrategy(
        delegate,
        _CapturingRecorder(),
        decision_timeout_seconds=1.0,
        decision_events=capture,
    )

    strategy.select(state, TeamColor.RED, target, legal)
    with pytest.raises(module.SourceDecisionTimeout):
        strategy.select(state, TeamColor.RED, target, legal)

    completed_plan = next(plan for event, plan in events if event == "decision_completed")
    timeout_plan = next(plan for event, plan in events if event == "decision_timeout")
    assert completed_plan is not None
    assert completed_plan.schedule_id == REQUEST_AWARE_SCHEDULE_V1_ID
    assert timeout_plan is None


def test_decision_timeout_discards_game_and_continues_with_actionable_telemetry(
    tmp_path: Path,
) -> None:
    if not hasattr(signal, "setitimer"):
        pytest.skip("POSIX interval timers are unavailable")
    module = _module()
    spec = module.WorkerSpec(
        worker_id=0,
        config=_config(decision_timeout_seconds=1.0),
        games=_games(20_000, 20_001),
    )
    events: list[Any] = []

    class SlowSecondStrategy:
        strategy_id = "test-slow-after-real-plan"

        def __init__(self, seed: int) -> None:
            self.calls = 0
            self.delegate = ISMCTSStrategy(
                environment_policy=HeuristicAgent(seed),
                config=SearchConfig(iterations=1, seed=seed, request_schedule_version=1),
            )

        def select(self, state: GameState, team: TeamColor, target: Any, legal: Any) -> Any:
            self.calls += 1
            result = self.delegate.select(state, team, target, legal)
            if self.calls == 2:
                time.sleep(1.5)
            return result

    def strategy_factory(_runtime: object, game: Any, _side: str, seed: int) -> Any:
        return SlowSecondStrategy(seed) if game.world_seed == 20_000 else _ImprovedStrategy()

    def run_with_partial_row(
        red: list[str], blue: list[str], agents: dict[str, Any], **kwargs: Any
    ) -> RunResult:
        state = GameSetup.create_game(
            kwargs["map_path"],
            red,
            blue,
            game_type=kwargs["game_type"],
            seed=kwargs["seed"],
        )
        wasp = state.get_hero(HeroID("hero_wasp"))
        assert wasp is not None
        agents[wasp.id].choose_planning(state, wasp)
        if kwargs["seed"] == 20_000:
            xargatha = state.get_hero(HeroID("hero_xargatha"))
            assert xargatha is not None
            agents[xargatha.id].choose_planning(state, xargatha)
        kwargs["decision_observer"].record_outcome(winner_side="RED", rounds=1, reason="game_over")
        return RunResult("RED", 1, 1, 1, "game_over", winner_side="RED")

    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        strategy_factory=strategy_factory,
        game_runner=run_with_partial_row,
        telemetry=events.append,
    )

    assert worker.run() == 1
    assert [path.name[:20] for path in (tmp_path / "fragments").glob("*.jsonl.zst")] == [
        f"{20_001:020d}"
    ]
    assert len((tmp_path / "checkpoint.jsonl").read_text().splitlines()) == 1

    timed_out = next(event for event in events if event.event == "decision_timeout")
    assert timed_out.world_seed == 20_000
    assert timed_out.game_id == _games(20_000)[0].game_id(spec.config)
    assert timed_out.decision_index == 1
    assert timed_out.round == 1
    assert timed_out.phase == "PLANNING"
    assert timed_out.perspective_team == "RED"
    assert timed_out.decision_owner_hero_id == "hero_xargatha"
    assert timed_out.root_kind == "CARD"
    assert timed_out.request_type is None
    assert timed_out.request_id is None
    assert timed_out.schedule_id == REQUEST_AWARE_SCHEDULE_V1_ID
    assert timed_out.effective_leaf_mode == "BOUNDED_CONTINUATION"
    assert timed_out.semantic_role == "PLANNING"
    assert timed_out.requested_iterations == 1
    assert timed_out.effective_iterations == 1
    assert timed_out.root_coverage_target is None
    assert timed_out.legal_count > 0
    assert timed_out.legal_family == "CARD"
    assert timed_out.decision_elapsed_seconds >= 0.01
    assert timed_out.completed_visits is None
    assert timed_out.visited_legal_count is None
    assert timed_out.legal_coverage is None

    successful = [event for event in events if event.event == "decision_completed"]
    assert [(event.world_seed, event.decision_index) for event in successful] == [
        (20_000, 0),
        (20_001, 0),
    ]
    assert [event.completed_visits for event in successful] == [1, 4]
    assert all(event.visited_legal_count == 1 for event in successful)
    assert all(event.legal_coverage == pytest.approx(1 / event.legal_count) for event in successful)


def test_search_progression_failure_discards_game_and_continues_with_diagnostics(
    tmp_path: Path,
) -> None:
    module = _module()
    first_game, second_game = _games(20_000, 20_001)
    spec = module.WorkerSpec(
        worker_id=0,
        config=_config(),
        games=(first_game, second_game),
    )
    events: list[Any] = []
    diagnostics = SearchProgressionDiagnostics(
        phase="RESOLUTION",
        round=4,
        actor="hero_arien",
        pending_request="request-17",
        stack_depth=3,
        top_step="SELECT_TARGET",
        transition_counts=(
            ("advance_calls", 2),
            ("advance_transitions", 257),
            ("current_advance_transitions", 256),
            ("session_advances", 129),
            ("environment_planning", 4),
            ("environment_inputs", 124),
            ("forced_decisions", 1),
        ),
    )

    class FailingPrior:
        def score(self, context: Any, state: GameState, legal: Any) -> Any:
            raise SearchProgressionError("advance transition limit exceeded (255)", diagnostics)

    class FailSecondDecisionStrategy(_ImprovedStrategy):
        def __init__(self, seed: int) -> None:
            self.calls = 0
            self.failing = ISMCTSStrategy(
                environment_policy=HeuristicAgent(seed),
                config=SearchConfig(iterations=1, seed=seed, request_schedule_version=1),
                prior=FailingPrior(),
            )

        def select(self, state: GameState, team: TeamColor, target: Any, legal: Any) -> Any:
            self.calls += 1
            if self.calls == 2:
                return self.failing.select(state, team, target, legal)
            return super().select(state, team, target, legal)

    def strategy_factory(_runtime: object, game: Any, _side: str, seed: int) -> Any:
        if game.world_seed == first_game.world_seed:
            return FailSecondDecisionStrategy(seed)
        return _ImprovedStrategy()

    def run_with_partial_row(
        red: list[str], blue: list[str], agents: dict[str, Any], **kwargs: Any
    ) -> RunResult:
        state = GameSetup.create_game(
            kwargs["map_path"],
            red,
            blue,
            game_type=kwargs["game_type"],
            seed=kwargs["seed"],
        )
        wasp = state.get_hero(HeroID("hero_wasp"))
        assert wasp is not None
        agents[wasp.id].choose_planning(state, wasp)
        if kwargs["seed"] == first_game.world_seed:
            xargatha = state.get_hero(HeroID("hero_xargatha"))
            assert xargatha is not None
            agents[xargatha.id].choose_planning(state, xargatha)
        kwargs["decision_observer"].record_outcome(winner_side="RED", rounds=1, reason="game_over")
        return RunResult("RED", 1, 1, 1, "game_over", winner_side="RED")

    fragments = tmp_path / "fragments"
    checkpoint = tmp_path / "checkpoint.jsonl"
    worker = module.SelfPlayWorker(
        spec,
        output_dir=fragments,
        checkpoint_path=checkpoint,
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        strategy_factory=strategy_factory,
        game_runner=run_with_partial_row,
        telemetry=events.append,
    )

    assert worker.run() == 1
    assert [path.name[:20] for path in fragments.glob("*.jsonl.zst")] == [f"{20_001:020d}"]
    checkpoint_rows = [
        module.CheckpointRow.model_validate_json(line)
        for line in checkpoint.read_bytes().splitlines()
    ]
    assert [row.game_id for row in checkpoint_rows] == [second_game.game_id(spec.config)]

    failed = next(event for event in events if event.event == "decision_failed")
    assert failed.reason == "search_progression"
    assert failed.error_type == "SearchProgressionError"
    assert failed.world_seed == first_game.world_seed
    assert failed.game_id == first_game.game_id(spec.config)
    assert failed.decision_index == 1
    assert failed.round == 1
    assert failed.phase == "PLANNING"
    assert failed.side == "RED"
    assert failed.perspective_team == "RED"
    assert failed.decision_owner_hero_id == "hero_xargatha"
    assert failed.root_kind == "CARD"
    assert failed.request_type is None
    assert failed.request_id is None
    assert failed.schedule_id == REQUEST_AWARE_SCHEDULE_V1_ID
    assert failed.effective_leaf_mode == "BOUNDED_CONTINUATION"
    assert failed.semantic_role == "PLANNING"
    assert failed.requested_iterations == 1
    assert failed.effective_iterations == 1
    assert failed.root_coverage_target is None
    assert failed.legal_count > 0
    assert failed.legal_family == "CARD"
    assert failed.decision_elapsed_seconds >= 0.0
    assert failed.completed_visits is None
    assert failed.visited_legal_count is None
    assert failed.legal_coverage is None
    assert failed.progression_reason == "advance transition limit exceeded (255)"
    assert failed.progression_phase == diagnostics.phase
    assert failed.progression_round == diagnostics.round
    assert failed.progression_actor == diagnostics.actor
    assert failed.progression_pending_request == diagnostics.pending_request
    assert failed.progression_stack_depth == diagnostics.stack_depth
    assert failed.progression_top_step == diagnostics.top_step
    assert failed.progression_transition_counts == dict(diagnostics.transition_counts)

    receipt = next(event for event in events if event.event == "timeout")
    assert receipt.world_seed == first_game.world_seed
    assert receipt.reason == "search_progression"
    assert receipt.error_type == "SearchProgressionError"
    assert [event.world_seed for event in events if event.event == "game_complete"] == [
        second_game.world_seed
    ]


def test_input_decision_telemetry_is_bounded_and_progress_keeps_round_and_steps(
    tmp_path: Path,
) -> None:
    module = _module()
    spec = module.WorkerSpec(
        worker_id=0,
        config=_config(decision_timeout_seconds=1.0),
        games=_games(20_002),
    )
    events: list[Any] = []

    def input_game(_red: Any, _blue: Any, agents: dict[str, Any], **kwargs: Any) -> RunResult:
        state = GameSetup.create_game(
            kwargs["map_path"],
            ["Wasp", "Xargatha"],
            ["Arien", "Brogan"],
            game_type=kwargs["game_type"],
            seed=kwargs["seed"],
        )
        state.round = 3
        request = InputRequest(
            id="bounded-request-id",
            request_type=InputRequestType.SELECT_OPTION,
            player_id="hero_wasp",
            prompt="private prompt must not enter telemetry",
            options=[InputOption.from_value("FIRST"), InputOption.from_value("SECOND")],
        )
        agents["hero_wasp"].choose_input(
            state,
            request,
            owned_hero_ids=frozenset({"hero_wasp", "hero_xargatha"}),
            decision_owner_hero_id="hero_wasp",
        )
        kwargs["progress_callback"](3, 9)
        kwargs["decision_observer"].record_outcome(winner_side="RED", rounds=3, reason="game_over")
        return RunResult("RED", 3, 2, 9, "game_over", winner_side="RED")

    class ScheduledStrategy(_ImprovedStrategy):
        def select(self, state: GameState, team: TeamColor, target: Any, legal: Any) -> Any:
            result = super().select(state, team, target, legal)
            assert result.search_result is not None
            result.search_result.schedule_id = REQUEST_AWARE_SCHEDULE_V1_ID
            result.search_result.effective_leaf_mode = LeafMode.IMMEDIATE_ACTION
            result.search_result.request_type = InputRequestType.SELECT_OPTION.value
            result.search_result.semantic_role = DecisionSemanticRole.OPTION_SELECTION
            result.search_result.requested_iterations = 1
            result.search_result.effective_iterations = 2
            result.search_result.root_coverage_target = 2
            result.search_result.root = Node(
                visits=2,
                total_value=0.5,
                total_squared_value=0.125,
                children={
                    key: Node(visits=1, total_value=0.25, total_squared_value=0.0625)
                    for key in legal
                },
            )
            return result

    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 4),
        strategy_factory=lambda *_args: ScheduledStrategy(),
        game_runner=input_game,
        telemetry=events.append,
    )

    assert worker.run() == 1
    started, completed = [event for event in events if event.event.startswith("decision_")]
    assert started.event == "decision_started"
    assert completed.event == "decision_completed"
    assert completed.decision_index == started.decision_index == 0
    assert completed.request_type == "SELECT_OPTION"
    assert completed.request_id == "bounded-request-id"
    assert completed.schedule_id == REQUEST_AWARE_SCHEDULE_V1_ID
    assert completed.effective_leaf_mode == "IMMEDIATE_ACTION"
    assert completed.semantic_role == "OPTION_SELECTION"
    assert completed.requested_iterations == 1
    assert completed.effective_iterations == 2
    assert completed.root_coverage_target == 2
    assert completed.legal_family == "SELECT_OPTION"
    assert completed.legal_count == 2
    assert (
        completed.model_dump(exclude_none=True)
        .keys()
        .isdisjoint({"prompt", "options", "legal_candidates", "selected_candidate"})
    )
    progress = next(event for event in events if event.event == "progress")
    assert progress.round == 3
    assert progress.steps == 9


def test_self_play_command_import_is_torch_free() -> None:
    script = """
import builtins
import sys
real_import = builtins.__import__
def reject(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch crossed the worker runtime boundary')
    return real_import(name, *args, **kwargs)
builtins.__import__ = reject
import automata.scripts.run_self_play
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _self_play_command_args(tmp_path: Path, worker_id: int, *extra: str) -> list[str]:
    return [
        "--artifact",
        str(tmp_path / "artifact"),
        "--parent-model-digest",
        "a" * 64,
        "--parent-generation",
        "6",
        "--observation-schema-version",
        "4",
        "--generation-id",
        "generation-7",
        "--seed-start",
        "20000",
        "--seed-end",
        "20008",
        "--worker-id",
        str(worker_id),
        "--output-dir",
        str(tmp_path / "output"),
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--search-config",
        '{"iterations":4}',
        "--max-steps",
        "20",
        "--timeout-seconds",
        "5",
        "--source-revision",
        "revision",
        "--dirty-tree-hash",
        "clean",
        "--no-progress",
        *extra,
    ]


def _capture_self_play_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> list[Any]:
    command = importlib.import_module("automata.scripts.run_self_play")
    captured: list[Any] = []

    class CapturingWorker:
        def __init__(self, spec: Any, **_kwargs: Any) -> None:
            captured.append(spec)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(command, "SelfPlayWorker", CapturingWorker)
    for worker_id in range(4):
        assert (
            command.main(
                _self_play_command_args(tmp_path, worker_id, *extra),
                runtime_loader=lambda _config: None,
                strategy_factory=lambda *_args: _ImprovedStrategy(),
            )
            == 0
        )
    return captured


def _variant_tuple(game: Any) -> tuple[Any, ...]:
    return game.game_type, game.red_composition, game.blue_composition


def test_self_play_cli_records_native_parent_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "observation_schema_version": 4,
                "runtime_compatibility_version": 2,
                "tensor_schema_digest": "b" * 64,
            }
        )
    )
    specs = _capture_self_play_specs(tmp_path, monkeypatch)
    identity = specs[0].config.source_config["parent_runtime_identity"]
    assert identity["observation_schema_version"] == 4
    assert identity["runtime_compatibility_version"] == 2
    assert identity["tensor_schema_digest"] == "b" * 64


def test_self_play_cli_rejects_retired_observation_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        _capture_self_play_specs(
            tmp_path, monkeypatch, "--parent-observation-bridge", "decision-v4-to-v3-v1"
        )
    assert raised.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_self_play_cli_visit_temperature_schedule_defaults_to_constant_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = _capture_self_play_specs(tmp_path, monkeypatch, "--visit-temperature", "0.5")
    explicit = _capture_self_play_specs(
        tmp_path,
        monkeypatch,
        "--visit-temperature",
        "0.5",
        "--visit-temperature-schedule",
        "constant",
    )
    decay = _capture_self_play_specs(
        tmp_path,
        monkeypatch,
        "--visit-temperature",
        "0.5",
        "--visit-temperature-schedule",
        "round-decay-v1",
    )

    assert [spec.worker_config_id for spec in default] == [
        spec.worker_config_id for spec in explicit
    ]
    assert default[0].config.source_config["visit_temperature_schedule"] == "constant"
    assert decay[0].config.source_config["visit_temperature_schedule"] == "round-decay-v1"
    assert default[0].config.generator_config_id != decay[0].config.generator_config_id
    assert default[0].games[0].game_id(default[0].config) != decay[0].games[0].game_id(
        decay[0].config
    )

    command = importlib.import_module("automata.scripts.run_self_play")
    with pytest.raises(SystemExit):
        command.main(
            _self_play_command_args(
                tmp_path,
                0,
                "--visit-temperature-schedule",
                "unknown",
            ),
            runtime_loader=lambda _config: None,
            strategy_factory=lambda *_args: _ImprovedStrategy(),
        )


def test_self_play_cli_decision_timeout_is_optional_positive_and_changes_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = _capture_self_play_specs(tmp_path, monkeypatch)
    enabled = _capture_self_play_specs(tmp_path, monkeypatch, "--decision-timeout-seconds", "1.25")

    assert default[0].config.decision_timeout_seconds is None
    assert enabled[0].config.decision_timeout_seconds == 1.25
    assert default[0].config.generator_config_id != enabled[0].config.generator_config_id

    command = importlib.import_module("automata.scripts.run_self_play")
    with pytest.raises(SystemExit):
        command.main(
            _self_play_command_args(tmp_path, 0, "--decision-timeout-seconds", "0"),
            runtime_loader=lambda _config: None,
            strategy_factory=lambda *_args: _ImprovedStrategy(),
        )


def test_self_play_cli_records_random_stream_namespace_in_provenance_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = _capture_self_play_specs(tmp_path, monkeypatch)
    namespaced = _capture_self_play_specs(
        tmp_path,
        monkeypatch,
        "--random-stream-namespace",
        "fair-temperature-arms-v1",
    )

    assert default[0].config.random_stream_namespace is None
    assert "random_stream_namespace" not in default[0].config.source_config
    assert namespaced[0].config.random_stream_namespace == "fair-temperature-arms-v1"
    assert (
        namespaced[0].config.source_config["random_stream_namespace"] == "fair-temperature-arms-v1"
    )
    assert default[0].config.source_config_id != namespaced[0].config.source_config_id
    assert default[0].config.generator_config_id != namespaced[0].config.generator_config_id
    assert default[0].games[0].game_id(default[0].config) != namespaced[0].games[0].game_id(
        namespaced[0].config
    )

    command = importlib.import_module("automata.scripts.run_self_play")
    for invalid in ("", "   ", "x" * 129):
        with pytest.raises(SystemExit):
            command.main(
                _self_play_command_args(tmp_path, 0, "--random-stream-namespace", invalid),
                runtime_loader=lambda _config: None,
                strategy_factory=lambda *_args: _ImprovedStrategy(),
            )


def test_self_play_balanced_schedule_has_exact_owned_seed_order_and_survives_sharding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _capture_self_play_specs(tmp_path, monkeypatch, "--variant-schedule", "balanced")
    games = sorted(
        (game for spec in specs for game in spec.games), key=lambda game: game.world_seed
    )

    expected_cycle = [
        ("QUICK", ("Wasp", "Xargatha"), ("Arien", "Brogan")),
        ("QUICK", ("Arien", "Brogan"), ("Wasp", "Xargatha")),
        ("LONG", ("Wasp", "Xargatha"), ("Arien", "Brogan")),
        ("LONG", ("Arien", "Brogan"), ("Wasp", "Xargatha")),
    ]
    assert [_variant_tuple(game) for game in games] == expected_cycle * 2
    assert [[game.world_seed for game in spec.games] for spec in specs] == [
        [20_000, 20_004],
        [20_001, 20_005],
        [20_002, 20_006],
        [20_003, 20_007],
    ]
    assert [_variant_tuple(spec.games[0]) for spec in specs] == expected_cycle
    assert [_variant_tuple(spec.games[1]) for spec in specs] == expected_cycle


def test_self_play_balanced_schedule_is_stable_for_resume_and_records_resolved_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _capture_self_play_specs(tmp_path, monkeypatch, "--variant-schedule", "balanced")
    second = _capture_self_play_specs(tmp_path, monkeypatch, "--variant-schedule", "balanced")

    assert [spec.worker_config_id for spec in first] == [spec.worker_config_id for spec in second]
    source_config = dict(first[0].config.source_config)
    assert source_config["variant_schedule_name"] == "balanced"
    assert source_config["variant_schedule_seed_start"] == 20_000
    assert source_config["variant_schedule"] == [
        {
            "game_type": "QUICK",
            "red_composition": ["Wasp", "Xargatha"],
            "blue_composition": ["Arien", "Brogan"],
        },
        {
            "game_type": "QUICK",
            "red_composition": ["Arien", "Brogan"],
            "blue_composition": ["Wasp", "Xargatha"],
        },
        {
            "game_type": "LONG",
            "red_composition": ["Wasp", "Xargatha"],
            "blue_composition": ["Arien", "Brogan"],
        },
        {
            "game_type": "LONG",
            "red_composition": ["Arien", "Brogan"],
            "blue_composition": ["Wasp", "Xargatha"],
        },
    ]


def test_self_play_variant_schedule_default_is_fixed_and_changes_generation_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = _capture_self_play_specs(tmp_path, monkeypatch)
    explicit_fixed = _capture_self_play_specs(tmp_path, monkeypatch, "--variant-schedule", "fixed")
    balanced = _capture_self_play_specs(tmp_path, monkeypatch, "--variant-schedule", "balanced")

    expected = (
        PHASE0_EXPERIMENT.game_type,
        PHASE0_EXPERIMENT.red_heroes,
        PHASE0_EXPERIMENT.blue_heroes,
    )
    assert all(_variant_tuple(game) == expected for spec in default for game in spec.games)
    assert [spec.worker_config_id for spec in default] == [
        spec.worker_config_id for spec in explicit_fixed
    ]
    assert default[0].config.source_config["variant_schedule_name"] == "fixed"
    assert default[0].config.source_config["variant_schedule"] == [
        {
            "game_type": "QUICK",
            "red_composition": ["Wasp", "Xargatha"],
            "blue_composition": ["Arien", "Brogan"],
        }
    ]
    assert default[0].config.generator_config_id != balanced[0].config.generator_config_id


def test_self_play_pins_current_scope_and_requires_every_scheduled_mode_and_hero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automata.models.contracts import ArtifactError
    from automata.models.shared_encoder.runtime import SharedEncoderRuntime

    command = importlib.import_module("automata.scripts.run_self_play")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_compatibility_version": 2,
                "observation_schema_version": 4,
                # These claims must never become the loader's requirements.
                "map_schema_version": 999,
                "hero_adapter_versions": {"generic": 999},
                "model_digest": "a" * 64,
            }
        )
    )
    supported_game_types = frozenset({"QUICK", "LONG"})

    def capturing_load(_cls: Any, _path: Path, *, requirements: Any) -> Any:
        captured.append(requirements)
        return SimpleNamespace(
            schema=SimpleNamespace(observation_schema_version=4),
            supported_game_types=supported_game_types,
            supported_heroes=frozenset({"Wasp", "Xargatha", "Arien", "Brogan"}),
        )

    captured: list[Any] = []
    monkeypatch.setattr(SharedEncoderRuntime, "from_artifact", classmethod(capturing_load))
    variants = command._resolved_variant_schedule("balanced")

    loaded = command._load_learned_model_runtime(artifact, _config(), variants)

    assert loaded.model_digest == "a" * 64
    assert captured[0].game_type == "QUICK"
    assert captured[0].heroes == frozenset({"Wasp", "Xargatha", "Arien", "Brogan"})
    assert captured[0].runtime_compatibility_version == 2
    assert captured[0].observation_schema_version == 4
    assert captured[0].map_schema_version == 1
    assert captured[0].hero_adapter_versions == {
        "generic": 1,
        "Wasp": 1,
        "Xargatha": 1,
        "Arien": 1,
        "Brogan": 1,
    }

    supported_game_types = frozenset({"QUICK"})
    with pytest.raises(ArtifactError, match=r"scheduled game type.*LONG"):
        command._load_learned_model_runtime(artifact, _config(), variants)


@pytest.mark.parametrize(
    "manifest_change",
    [
        {"observation_schema_version": 3},
        {"runtime_compatibility_version": 1},
    ],
)
def test_self_play_rejects_obsolete_formats_before_runtime_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_change: dict[str, int],
) -> None:
    from automata.models.shared_encoder.runtime import SharedEncoderRuntime

    command = importlib.import_module("automata.scripts.run_self_play")
    artifact = tmp_path / "legacy-artifact"
    artifact.mkdir()
    manifest = {
        "runtime_compatibility_version": 2,
        "observation_schema_version": 4,
        "map_schema_version": 1,
        "hero_adapter_versions": {"generic": 1},
    }
    manifest.update(manifest_change)
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    called = False

    def unexpected_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("runtime load must not be reached")

    monkeypatch.setattr(SharedEncoderRuntime, "from_artifact", unexpected_load)

    with pytest.raises(ValueError, match="current-format"):
        command._load_learned_model_runtime(artifact, _config())
    assert not called


def test_self_play_rejects_non_v4_external_schema_before_reading_artifact(
    tmp_path: Path,
) -> None:
    command = importlib.import_module("automata.scripts.run_self_play")

    with pytest.raises(ValueError, match=r"schema 4; got 3"):
        command._load_learned_model_runtime(
            tmp_path / "does-not-exist",
            replace(_config(), observation_schema_version=3),
        )


def _replace_self_play_search_config(args: list[str], value: str) -> list[str]:
    replaced = list(args)
    replaced[replaced.index("--search-config") + 1] = value
    return replaced


def test_self_play_builtin_lh_preset_constructs_resolved_seeded_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automata.search.contracts import CutoffUnit, LeafMode
    from automata.search.heuristic import HeuristicLeafEvaluator
    from automata.search.learned import LearnedSearchPolicy

    command = importlib.import_module("automata.scripts.run_self_play")
    captured: list[tuple[Any, dict[str, Any]]] = []

    class CapturingWorker:
        def __init__(self, spec: Any, **kwargs: Any) -> None:
            captured.append((spec, kwargs))

        def run(self) -> int:
            return 0

    monkeypatch.setattr(command, "SelfPlayWorker", CapturingWorker)
    args = _replace_self_play_search_config(
        _self_play_command_args(
            tmp_path,
            0,
            "--strategy-preset",
            "learned-policy-heuristic-value",
        ),
        json.dumps(
            {
                "iterations": 8,
                "cutoff_limit": 3,
                "cutoff_unit": "DECISIONS",
                "leaf_mode": "IMMEDIATE_ACTION",
                "root_puct_c": 1.75,
                "root_widening_c": 0.75,
                "root_widening_alpha": 0.25,
                "adaptive_hex_root_schedule_version": 1,
                "request_schedule_version": 1,
                "max_advance_transitions": 2048,
                "max_forced_decisions": 512,
            }
        ),
    )

    assert command.main(args, runtime_loader=lambda _config: None) == 0

    spec, worker_kwargs = captured[0]
    runtime = object()
    strategy = worker_kwargs["strategy_factory"](runtime, spec.games[0], "RED", 9876)
    assert isinstance(strategy, ISMCTSStrategy)
    assert strategy._config == SearchConfig(
        iterations=8,
        seed=9876,
        cutoff_limit=3,
        cutoff_unit=CutoffUnit.DECISIONS,
        leaf_mode=LeafMode.IMMEDIATE_ACTION,
        use_prior=True,
        root_puct_c=1.75,
        root_widening_c=0.75,
        root_widening_alpha=0.25,
        adaptive_hex_root_schedule_version=1,
        request_schedule_version=1,
        max_advance_transitions=2048,
        max_forced_decisions=512,
    )
    assert isinstance(strategy._environment_policy, HeuristicAgent)
    assert isinstance(strategy._prior, LearnedSearchPolicy)
    assert strategy._prior.runtime is runtime
    assert isinstance(strategy._leaf_evaluator, HeuristicLeafEvaluator)
    assert spec.config.search_config["seed"] == "agent_seed"
    assert spec.config.source_config["strategy_preset"] == {
        "name": "learned-policy-heuristic-value",
        "matrix_cell": "L/H",
        "continuation_policy": "learned-argmax-v1",
        "value_recipe": HeuristicLeafEvaluator.recipe_id,
        "search_config": dict(spec.config.search_config),
    }


@pytest.mark.parametrize(
    ("search_config", "message"),
    [
        ({"iterations": 8, "unknown": True}, "unknown"),
        ({"iterations": "8"}, "iterations"),
        ({"iterations": 8, "leaf_mode": "ROLLOUT"}, "leaf_mode"),
        ({"iterations": 8, "cutoff_unit": "TURNS"}, "cutoff_unit"),
        ({"iterations": 8, "seed": 42}, "seed"),
        ({"iterations": 8, "root_puct_c": "1.5"}, "root_puct_c"),
    ],
)
def test_self_play_builtin_lh_preset_rejects_invalid_search_config(
    tmp_path: Path,
    search_config: dict[str, Any],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = importlib.import_module("automata.scripts.run_self_play")
    args = _replace_self_play_search_config(
        _self_play_command_args(
            tmp_path,
            0,
            "--strategy-preset",
            "learned-policy-heuristic-value",
        ),
        json.dumps(search_config),
    )

    with pytest.raises(SystemExit):
        command.main(args, runtime_loader=lambda _config: None)

    assert message in capsys.readouterr().err


def test_self_play_cli_strategy_selection_is_mutually_exclusive_and_still_required(
    tmp_path: Path,
) -> None:
    command = importlib.import_module("automata.scripts.run_self_play")
    with pytest.raises(SystemExit):
        command.main(
            _self_play_command_args(
                tmp_path,
                0,
                "--strategy-preset",
                "learned-policy-heuristic-value",
                "--strategy-factory",
                "package.module:factory",
            ),
            runtime_loader=lambda _config: None,
        )
    with pytest.raises(SystemExit):
        command.main(_self_play_command_args(tmp_path, 0), runtime_loader=lambda _config: None)


def test_self_play_builtin_lh_resolved_config_and_preset_change_generator_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _capture_self_play_specs(
        tmp_path,
        monkeypatch,
        "--strategy-preset",
        "learned-policy-heuristic-value",
    )
    second_args = _replace_self_play_search_config(
        _self_play_command_args(
            tmp_path,
            0,
            "--strategy-preset",
            "learned-policy-heuristic-value",
        ),
        '{"iterations":5}',
    )
    command = importlib.import_module("automata.scripts.run_self_play")
    captured: list[Any] = []

    class CapturingWorker:
        def __init__(self, spec: Any, **_kwargs: Any) -> None:
            captured.append(spec)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(command, "SelfPlayWorker", CapturingWorker)
    assert (
        command.main(
            second_args,
            runtime_loader=lambda _config: None,
            strategy_factory=lambda *_args: _ImprovedStrategy(),
        )
        == 0
    )

    assert first[0].config.generator_config_id != captured[0].config.generator_config_id
    assert first[0].config.source_config_id != captured[0].config.source_config_id
    assert first[0].games[0].game_id(first[0].config) != captured[0].games[0].game_id(
        captured[0].config
    )
