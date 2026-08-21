"""RED behavioral tests for the resumable expert-policy generator CLI."""

from __future__ import annotations

import importlib
import json
import signal
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest

from automata.runtime.harness import RunResult

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]
HERO_IDS = {"hero_wasp", "hero_xargatha", "hero_arien", "hero_brogan"}


def _module() -> Any:
    try:
        return importlib.import_module("automata.scripts.generate_policy_data")
    except ModuleNotFoundError:
        pytest.fail("automata.scripts.generate_policy_data is not implemented")


def _args(
    out: Path,
    checkpoint: Path,
    *,
    iterations: int = 17,
    timeout: str = "30",
) -> list[str]:
    return [
        "--out",
        str(out),
        "--checkpoint",
        str(checkpoint),
        "--seed-start",
        "41",
        "--seed-end",
        "43",
        "--expert-iterations",
        str(iterations),
        "--expert-cutoff-rounds",
        "4",
        "--source-max-steps",
        "900",
        "--source-timeout-seconds",
        timeout,
        "--game-type",
        "QUICK",
    ]


class _RecorderSpy:
    calls: ClassVar[list[_RecorderSpy]] = []

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        self.path = Path(path)
        self.kwargs = kwargs
        self.calls.append(self)

    def emit(self, marker: str = "decision") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"game_id": self.kwargs["game_id"], "marker": marker}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


class _AgentSpy:
    calls: ClassVar[list[_AgentSpy]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls.append(self)


def _install_spies(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    _RecorderSpy.calls = []
    _AgentSpy.calls = []
    monkeypatch.setattr(module, "PolicyDatasetRecorder", _RecorderSpy, raising=True)
    monkeypatch.setattr(module, "ISMCTSAgent", _AgentSpy, raising=True)
    monkeypatch.setattr(module, "source_identity", lambda: ("revision-a", "dirty-a"))


def _result(reason: str = "game_over", winner: str | None = "RED") -> RunResult:
    return RunResult(winner=winner, rounds=2, turns=8, steps=40, reason=reason)


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_half_open_range_builds_fresh_experts_and_wires_one_recorder_per_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    runs: list[dict[str, Any]] = []

    def run(red: Any, blue: Any, agents: Any, **kwargs: Any) -> RunResult:
        runs.append({"red": list(red), "blue": list(blue), "agents": dict(agents), **kwargs})
        return _result()

    monkeypatch.setattr(module, "run_game", run, raising=True)
    out, checkpoint = tmp_path / "policy.jsonl", tmp_path / "games.jsonl"

    assert module.main(_args(out, checkpoint)) == 0
    assert [run["seed"] for run in runs] == [41, 42]
    assert len(_RecorderSpy.calls) == 2

    seen_agents: set[int] = set()
    for captured, recorder in zip(runs, _RecorderSpy.calls, strict=True):
        assert captured["red"] == RED and captured["blue"] == BLUE
        assert captured["game_type"] == "QUICK" and captured["max_steps"] == 900
        assert captured["recorder"] is recorder
        agents = captured["agents"]
        assert set(agents) == HERO_IDS
        red_agent, blue_agent = agents["hero_wasp"], agents["hero_arien"]
        assert agents["hero_xargatha"] is red_agent
        assert agents["hero_brogan"] is blue_agent
        assert red_agent is not blue_agent
        assert id(red_agent) not in seen_agents and id(blue_agent) not in seen_agents
        seen_agents.update((id(red_agent), id(blue_agent)))
        assert red_agent.kwargs["root_observer"] is recorder
        assert blue_agent.kwargs["root_observer"] is recorder
        for side, agent in (("RED", red_agent), ("BLUE", blue_agent)):
            config = agent.kwargs.get("config", agent.args[0] if agent.args else None)
            assert config.iterations == 17 and config.cutoff_rounds == 4
            assert config.seed == module.agent_seed(captured["seed"], side)
            assert not ({"default_policy", "value_fn", "prior"} & agent.kwargs.keys())
        assert red_agent.kwargs["config"].seed != blue_agent.kwargs["config"].seed


def test_source_and_full_config_define_deterministic_resume_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    out, checkpoint = tmp_path / "policy.jsonl", tmp_path / "games.jsonl"
    calls: list[int] = []

    def run(*_: Any, **kwargs: Any) -> RunResult:
        calls.append(kwargs["seed"])
        return _result()

    monkeypatch.setattr(module, "run_game", run)
    args = _args(out, checkpoint)
    assert module.main(args) == 0
    first_recorders = list(_RecorderSpy.calls)
    first_checkpoints = _rows(checkpoint)
    assert calls == [41, 42]
    assert module.main(args) == 0
    assert calls == [41, 42]

    assert len({row["config_id"] for row in first_checkpoints}) == 1
    assert [row["game_id"] for row in first_checkpoints] == [
        recorder.kwargs["game_id"] for recorder in first_recorders
    ]
    for recorder in first_recorders:
        identity = recorder.kwargs
        assert identity["source_revision"] == "revision-a"
        assert identity["dirty_tree_hash"] == "dirty-a"
        assert identity["expert_identity"]
        assert identity["expert_config"]["iterations"] == 17
        assert identity["expert_config"]["cutoff_rounds"] == 4
        assert identity["expert_config"]["use_prior"] is True

    monkeypatch.setattr(module, "source_identity", lambda: ("revision-b", "dirty-b"))
    assert module.main(args) == 0
    assert calls == [41, 42, 41, 42]
    changed = _RecorderSpy.calls[-2:]
    assert {r.kwargs["game_id"] for r in first_recorders}.isdisjoint(
        r.kwargs["game_id"] for r in changed
    )
    assert {r.kwargs["expert_identity"] for r in first_recorders}.isdisjoint(
        r.kwargs["expert_identity"] for r in changed
    )


def test_only_game_over_is_completed_and_failures_are_durable_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    out, checkpoint = tmp_path / "policy.jsonl", tmp_path / "games.jsonl"
    attempts: list[int] = []
    outcomes = {41: _result(), 42: _result("max_steps", None)}
    fsyncs: list[int] = []
    monkeypatch.setattr(module.os, "fsync", fsyncs.append)

    def run(*_: Any, **kwargs: Any) -> RunResult:
        attempts.append(kwargs["seed"])
        return outcomes[kwargs["seed"]]

    monkeypatch.setattr(module, "run_game", run)
    args = _args(out, checkpoint)
    assert module.main(args) == 0
    assert attempts == [41, 42]
    rows = _rows(checkpoint)
    assert len(rows) == 2 and len(fsyncs) == 2
    assert rows[0]["completed"] is True and rows[0]["reason"] == "game_over"
    assert rows[1]["completed"] is False and rows[1]["reason"] == "max_steps"

    outcomes[42] = _result()
    assert module.main(args) == 0
    assert attempts == [41, 42, 42]
    assert _rows(checkpoint)[-1]["completed"] is True


def test_exception_is_checkpointed_as_incomplete_then_propagates_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    out, checkpoint = tmp_path / "policy.jsonl", tmp_path / "games.jsonl"
    calls = 0

    def fail(*_: Any, **__: Any) -> RunResult:
        nonlocal calls
        calls += 1
        raise RuntimeError("source exploded")

    monkeypatch.setattr(module, "run_game", fail)
    with pytest.raises(RuntimeError, match="source exploded"):
        module.main(_args(out, checkpoint))
    [row] = _rows(checkpoint)
    assert row["completed"] is False and row["reason"] == "exception"

    monkeypatch.setattr(module, "run_game", lambda *_a, **_kw: _result())
    assert module.main(_args(out, checkpoint)) == 0
    assert calls == 1


def test_startup_removes_orphan_rows_without_losing_checkpointed_other_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    out, checkpoint = tmp_path / "policy.jsonl", tmp_path / "games.jsonl"

    def complete(*_: Any, **kwargs: Any) -> RunResult:
        kwargs["recorder"].emit(f"seed-{kwargs['seed']}")
        return _result()

    monkeypatch.setattr(module, "run_game", complete)
    assert module.main(_args(out, checkpoint)) == 0
    initial = _rows(checkpoint)
    completed_game_ids = {row["game_id"] for row in initial}
    unrelated_id = "completed-by-another-config"
    with checkpoint.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "config_id": "other-config",
                    "game_id": unrelated_id,
                    "world_seed": 999,
                    "completed": True,
                    "reason": "game_over",
                }
            )
            + "\n"
        )
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"game_id": unrelated_id, "marker": "keep"}) + "\n")
        handle.write(json.dumps({"game_id": "orphan-after-crash", "marker": "drop"}) + "\n")

    assert module.main(_args(out, checkpoint)) == 0
    output_rows = _rows(out)
    assert {row["game_id"] for row in output_rows} == completed_game_ids | {unrelated_id}
    assert len(output_rows) == 3


def test_expert_or_source_config_change_is_disjoint_and_reruns_seed_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    out, checkpoint = tmp_path / "policy.jsonl", tmp_path / "games.jsonl"
    calls: list[int] = []
    monkeypatch.setattr(
        module,
        "run_game",
        lambda *_a, **kw: (calls.append(kw["seed"]) or _result()),
    )

    variants = [
        _args(out, checkpoint, iterations=17),
        _args(out, checkpoint, iterations=18),
        _args(out, checkpoint, iterations=17, timeout="31"),
    ]
    source_steps = _args(out, checkpoint, iterations=17)
    source_steps[source_steps.index("--source-max-steps") + 1] = "901"
    variants.append(source_steps)
    game_type = _args(out, checkpoint, iterations=17)
    game_type[game_type.index("--game-type") + 1] = "STANDARD"
    variants.append(game_type)

    prior_ids: set[str] = set()
    for args in variants:
        assert module.main(args) == 0
        current = _rows(checkpoint)[-2:]
        current_ids = {row["game_id"] for row in current}
        assert prior_ids.isdisjoint(current_ids)
        prior_ids.update(current_ids)

    assert calls == [41, 42] * len(variants)
    assert len({row["config_id"] for row in _rows(checkpoint)}) == len(variants)


def test_parser_requires_explicit_paths_range_and_positive_finite_limits() -> None:
    parser = _module()._parser()
    required = [
        "--out",
        "policy.jsonl",
        "--checkpoint",
        "games.jsonl",
        "--seed-start",
        "1",
        "--seed-end",
        "2",
        "--game-type",
        "QUICK",
    ]
    parsed = parser.parse_args(required)
    assert parsed.seed_start == 1 and parsed.seed_end == 2
    assert parsed.out == "policy.jsonl" and parsed.checkpoint == "games.jsonl"
    assert parsed.expert_iterations == 16
    assert parsed.expert_cutoff_rounds == 2
    assert parsed.source_timeout_seconds == pytest.approx(1800.0)
    for flag in ("--expert-iterations", "--expert-cutoff-rounds", "--source-max-steps"):
        with pytest.raises(SystemExit):
            parser.parse_args([*required, flag, "0"])
    for value in ("0", "-1", "nan", "inf", "-inf"):
        with pytest.raises(SystemExit):
            parser.parse_args([*required, "--source-timeout-seconds", value])
    with pytest.raises(SystemExit):
        parser.parse_args(required[:4])


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX timers")
def test_timeout_restores_signal_state_is_diagnostic_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    out, checkpoint = tmp_path / "policy.jsonl", tmp_path / "games.jsonl"
    calls: list[int] = []

    def slow(*_: Any, **kwargs: Any) -> RunResult:
        calls.append(kwargs["seed"])
        if kwargs["seed"] == 41:
            time.sleep(0.1)
        return _result()

    monkeypatch.setattr(module, "run_game", slow)
    before_handler = signal.getsignal(signal.SIGALRM)
    before_timer = signal.getitimer(signal.ITIMER_REAL)
    args = _args(out, checkpoint, timeout="0.02")
    assert module.main(args) == 0
    assert signal.getsignal(signal.SIGALRM) == before_handler
    assert signal.getitimer(signal.ITIMER_REAL) == before_timer
    assert calls == [41, 42]
    timeout_row = next(row for row in _rows(checkpoint) if row["world_seed"] == 41)
    assert timeout_row["completed"] is False
    assert timeout_row["reason"] == "wall_clock_timeout"

    monkeypatch.setattr(
        module,
        "run_game",
        lambda *_a, **kw: (calls.append(kw["seed"]) or _result()),
    )
    assert module.main(args) == 0
    assert calls == [41, 42, 41]
