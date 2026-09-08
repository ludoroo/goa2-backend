from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from automata.agents.contracts import PlanningDecision
from automata.harness.game_runner import DEFAULT_MAP, RunResult
from automata.runtime.driver import BotDecision, DecisionKind
from automata.runtime.effects import register_all_effects
from automata.training.dataset import (
    JointDatasetRecorder,
    load_joint_dataset,
    write_joint_dataset,
)
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def _module() -> Any:
    return importlib.import_module("automata.scripts.generate_joint_bootstrap")


def _args(out: Path, checkpoint: Path, start: int = 10_000, end: int = 10_002) -> list[str]:
    return [
        "--out",
        str(out),
        "--checkpoint",
        str(checkpoint),
        "--seed-start",
        str(start),
        "--seed-end",
        str(end),
        "--target-source",
        "heuristic",
        "--target-recipe",
        "one-hot-exact-choice",
        "--max-steps",
        "123",
        "--timeout-seconds",
        "30",
    ]


class _RecorderSpy:
    calls: ClassVar[list[_RecorderSpy]] = []

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        self.path, self.kwargs, self.closed = Path(path), kwargs, False
        self.calls.append(self)

    def close(self) -> None:
        self.closed = True


def test_generator_uses_fresh_deterministic_agents_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _RecorderSpy.calls = []
    monkeypatch.setattr(module, "JointDatasetRecorder", _RecorderSpy)
    monkeypatch.setattr(module, "source_identity", lambda **_kwargs: ("rev", "dirty"))
    monkeypatch.setattr(module, "_publish_output", lambda *_a, **_kw: None)
    runs: list[dict[str, Any]] = []

    def fake_run(red: Any, blue: Any, agents: Any, **kwargs: Any) -> RunResult:
        runs.append({"red": list(red), "blue": list(blue), "agents": dict(agents), **kwargs})
        return RunResult("RED", 1, 2, 3, "game_over")

    monkeypatch.setattr(module, "run_game", fake_run)
    out, checkpoint = tmp_path / "joint.jsonl", tmp_path / "checkpoint.jsonl"
    args = _args(out, checkpoint)

    assert module.main(args) == 0
    assert [captured["seed"] for captured in runs] == [10_000, 10_001]
    for captured in runs:
        red_agent = captured["agents"]["hero_wasp"]
        blue_agent = captured["agents"]["hero_arien"]
        assert captured["agents"]["hero_xargatha"] is red_agent
        assert captured["agents"]["hero_brogan"] is blue_agent
        assert red_agent is not blue_agent
        assert captured["decision_observer"].recorder in _RecorderSpy.calls
        assert captured["max_steps"] == 123
    assert len({id(captured["agents"]["hero_wasp"]) for captured in runs}) == 2
    assert all(call.path.name.endswith(".jsonl.zst") for call in _RecorderSpy.calls)

    assert module.main(args) == 0
    assert len(runs) == 2
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert all(row["completed"] and row["reason"] == "game_over" for row in rows)


def test_rejects_seed_range_outside_bootstrap_before_touching_files(tmp_path: Path) -> None:
    out, checkpoint = tmp_path / "joint.jsonl", tmp_path / "checkpoint.jsonl"
    with pytest.raises(SystemExit):
        _module().main(_args(out, checkpoint, 9_999, 10_001))
    assert not out.exists() and not checkpoint.exists()


def test_checkpoint_tolerates_only_a_truncated_final_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _RecorderSpy.calls = []
    monkeypatch.setattr(module, "JointDatasetRecorder", _RecorderSpy)
    monkeypatch.setattr(module, "source_identity", lambda **_kwargs: ("rev", "dirty"))
    monkeypatch.setattr(module, "_publish_output", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        module, "run_game", lambda *_a, **_kw: RunResult("RED", 1, 2, 3, "game_over")
    )
    out, checkpoint = tmp_path / "joint.jsonl", tmp_path / "checkpoint.jsonl"
    checkpoint.write_bytes(b'{"broken":')
    assert module.main(_args(out, checkpoint, 10_000, 10_001)) == 0
    assert len(checkpoint.read_text().splitlines()) == 1

    checkpoint.write_bytes(b'{"broken":true}\n{"truncated":')
    with pytest.raises(ValueError, match="checkpoint row 1"):
        module.main(_args(out, checkpoint, 10_000, 10_001))


def test_resume_reconciles_empty_aggregate_from_an_old_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "source_identity", lambda **_kwargs: ("rev", "dirty"))
    runs: list[int] = []

    def incomplete_run(*_args: Any, **kwargs: Any) -> RunResult:
        runs.append(kwargs["seed"])
        return RunResult(None, 1, 2, 123, "max_steps")

    monkeypatch.setattr(module, "run_game", incomplete_run)
    out, checkpoint = tmp_path / "joint.jsonl.zst", tmp_path / "checkpoint.jsonl"
    write_joint_dataset(out, ())
    module._append_checkpoint(
        checkpoint,
        module.CheckpointRow(
            config_id="old-config",
            game_id="old-game",
            world_seed=10_000,
            completed=True,
            reason="game_over",
            winner="RED",
            rounds=1,
            turns=2,
            steps=3,
        ),
    )
    stale_fragment = module._fragment_path(module._fragment_dir(out), "old-game")
    stale_fragment.parent.mkdir()
    stale_fragment.write_bytes(b"stale")

    assert module.main(_args(out, checkpoint, 10_000, 10_001)) == 0

    assert runs == [10_000]
    assert not out.exists()
    assert not stale_fragment.exists()


def test_incomplete_games_do_not_publish_an_empty_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "source_identity", lambda **_kwargs: ("rev", "dirty"))
    monkeypatch.setattr(
        module,
        "run_game",
        lambda *_args, **_kwargs: RunResult(None, 1, 2, 123, "max_steps"),
    )
    out, checkpoint = tmp_path / "joint.jsonl.zst", tmp_path / "checkpoint.jsonl"

    assert module.main(_args(out, checkpoint, 10_000, 10_001)) == 0

    assert not out.exists()


def test_real_heuristic_decision_records_exact_v3_one_hot_row(tmp_path: Path) -> None:
    module = _module()
    register_all_effects()
    scope = PHASE0_EXPERIMENT
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(scope.red_heroes),
        list(scope.blue_heroes),
        game_type=scope.game_type,
        seed=10_000,
    )
    from automata.runtime.driver import inspect_next_decision

    decision = inspect_next_decision(state, module.build_agents(10_000))
    assert decision is not None
    path = tmp_path / "one-game.jsonl"
    recorder = JointDatasetRecorder(
        path,
        game_id="game",
        world_seed=10_000,
        map_id=scope.map_id,
        game_type=scope.game_type,
        red_composition=scope.red_heroes,
        blue_composition=scope.blue_heroes,
        generation_id="generation",
        source_revision="rev",
        dirty_tree_hash="clean",
        source_model_digest=None,
        search_config_id="none",
        generator_config_id="config",
    )
    observer = module.HeuristicJointObserver(recorder)
    observer.record_decision(state, decision)
    observer.record_outcome(winner="RED", rounds=1, reason="game_over")

    [row] = load_joint_dataset(path).rows
    assert row.observation.schema_version == 3
    selections = [candidate.selection for candidate in row.observation.candidates]
    assert row.selected_selection in selections
    assert row.policy_target == tuple(
        1.0 if candidate.candidate_id == row.selected_candidate_id else 0.0
        for candidate in row.observation.candidates
    )


def test_observer_preserves_finish_and_skip_selections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    register_all_effects()
    scope = PHASE0_EXPERIMENT
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(scope.red_heroes),
        list(scope.blue_heroes),
        game_type=scope.game_type,
        seed=10_000,
    )
    path = tmp_path / "sentinels.jsonl"
    recorder = JointDatasetRecorder(
        path,
        game_id="sentinels",
        world_seed=10_000,
        map_id=scope.map_id,
        game_type=scope.game_type,
        red_composition=scope.red_heroes,
        blue_composition=scope.blue_heroes,
        generation_id="generation",
        source_revision="rev",
        dirty_tree_hash="clean",
        source_model_digest=None,
        search_config_id="none",
        generator_config_id="config",
    )
    observer = module.HeuristicJointObserver(recorder)
    monkeypatch.setattr(module, "planning_open_for_second_card", lambda *_a: True)
    observer.record_decision(
        state,
        BotDecision(
            kind=DecisionKind.PLANNING,
            hero_id=HeroID("hero_wasp"),
            planning=PlanningDecision.finish(),
        ),
    )
    request = InputRequest(
        id="skip",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value("hold")],
        can_skip=True,
    )
    observer.record_decision(
        state,
        BotDecision(
            kind=DecisionKind.INPUT,
            hero_id=HeroID("hero_wasp"),
            request=request,
            selection="SKIP",
        ),
    )
    observer.record_outcome(winner="RED", rounds=1, reason="game_over")

    rows = load_joint_dataset(path).rows
    assert [row.selected_selection for row in rows] == [None, "SKIP"]


def test_observer_preserves_numeric_engine_selection(tmp_path: Path) -> None:
    module = _module()
    register_all_effects()
    scope = PHASE0_EXPERIMENT
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(scope.red_heroes),
        list(scope.blue_heroes),
        game_type=scope.game_type,
        seed=10_000,
    )
    path = tmp_path / "numeric.jsonl"
    recorder = JointDatasetRecorder(
        path,
        game_id="numeric",
        world_seed=10_000,
        map_id=scope.map_id,
        game_type=scope.game_type,
        red_composition=scope.red_heroes,
        blue_composition=scope.blue_heroes,
        generation_id="generation",
        source_revision="rev",
        dirty_tree_hash="clean",
        source_model_digest=None,
        search_config_id="none",
        generator_config_id="config",
    )
    observer = module.HeuristicJointObserver(recorder)
    request = InputRequest(
        id="number-like-option",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value(2), InputOption.from_value(3)],
    )
    observer.record_decision(
        state,
        BotDecision(
            kind=DecisionKind.INPUT,
            hero_id=HeroID("hero_wasp"),
            request=request,
            selection=2,
        ),
    )
    observer.record_outcome(winner="RED", rounds=1, reason="game_over")

    [row] = load_joint_dataset(path).rows
    assert row.selected_selection == 2
    assert [candidate.selection for candidate in row.observation.candidates] == [2, 3]
