from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from automata.harness.game_runner import DEFAULT_MAP, RunResult
from automata.search.ismcts import SearchResult
from automata.search.ismcts.strategy import StrategyResult
from automata.search.node import Node
from automata.training.dataset import load_joint_dataset
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def _module() -> Any:
    return importlib.import_module("automata.training.generation")


def _config() -> Any:
    module = _module()
    return module.GenerationConfig(
        generation_id="generation-7",
        parent_model_digest="a" * 64,
        parent_generation=6,
        observation_schema_version=3,
        source_revision="revision",
        dirty_tree_hash="clean",
        search_config={"iterations": 4},
        source_config={"map_path": DEFAULT_MAP, "recipe": "visits-v1"},
        max_steps=20,
        timeout_seconds=5.0,
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


def _short_game(
    red: list[str], blue: list[str], agents: dict[str, Any], **kwargs: Any
) -> RunResult:
    state = GameSetup.create_game(
        kwargs["map_path"], red, blue, game_type=kwargs["game_type"], seed=kwargs["seed"]
    )
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    agents[hero.id].choose_planning(state, hero)
    result = RunResult("RED", 1, 1, 1, "game_over")
    kwargs["decision_observer"].record_outcome(winner="RED", rounds=1, reason="game_over")
    return result


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
            observation_schema_version=3,
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


def test_resume_skips_complete_games_without_loading_or_duplicates(tmp_path: Path) -> None:
    module = _module()
    spec = module.WorkerSpec(worker_id=0, config=_config(), games=_games(20_002))
    loads = 0

    def load_runtime(_config: Any) -> Any:
        nonlocal loads
        loads += 1
        return module.LoadedChampionRuntime(object(), "a" * 64, 6, 3)

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
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "b" * 64, 6, 3),
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
        "runtime_loader": lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 3),
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
        kwargs["decision_observer"].record_outcome(winner=None, rounds=1, reason="max_steps")
        return RunResult(None, 1, 1, 20, "max_steps")

    worker = module.SelfPlayWorker(
        spec,
        output_dir=tmp_path / "fragments",
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        runtime_loader=lambda _config: module.LoadedChampionRuntime(object(), "a" * 64, 6, 3),
        strategy_factory=lambda *_args: _ImprovedStrategy(),
        game_runner=capped,
        telemetry=events.append,
    )
    assert worker.run() == 0
    assert events[-1].event == "timeout"
    assert events[-1].reason == "max_steps"
    assert not list((tmp_path / "fragments").glob("*.jsonl*"))
    assert not (tmp_path / "checkpoint.jsonl").exists()


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
