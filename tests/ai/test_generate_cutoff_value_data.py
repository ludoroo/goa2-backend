"""RED behavioral tests for the resumable cutoff-label generator CLI."""

from __future__ import annotations

import importlib
import json
import signal
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]


def _module() -> Any:
    try:
        return importlib.import_module("automata.scripts.generate_cutoff_value_data")
    except ModuleNotFoundError:
        pytest.fail("generate_cutoff_value_data CLI is not implemented")


def _args(out: Path, checkpoint: Path | None = None, *, iterations: int = 7) -> list[str]:
    args = [
        "--out",
        str(out),
        "--seed-start",
        "11",
        "--seed-end",
        "13",
        "--iterations",
        str(iterations),
        "--cutoff-rounds",
        "3",
        "--sample-every",
        "5",
        "--max-samples-per-side",
        "6",
        "--source-max-steps",
        "700",
        "--continuation-max-steps",
        "800",
        "--continuation-max-rounds",
        "9",
    ]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    return args


class _ObserverSpy:
    calls: ClassVar[list[tuple[tuple[Any, ...], dict[str, Any]]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class _AgentSpy:
    calls: ClassVar[list[tuple[tuple[Any, ...], dict[str, Any], object]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs, self))


def _install_spies(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    _ObserverSpy.calls = []
    _AgentSpy.calls = []
    monkeypatch.setattr(module, "TerminalLabelObserver", _ObserverSpy)
    monkeypatch.setattr(module, "ISMCTSAgent", _AgentSpy)
    monkeypatch.setattr(module, "source_identity", lambda: ("rev", "dirty"))


def test_builds_fresh_seeded_ismcts_sides_and_wires_cutoff_observers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    runs: list[dict[str, Any]] = []

    def capture_run(red_heroes: Any, blue_heroes: Any, agents: Any, **kwargs: Any) -> None:
        runs.append(
            {
                "red_heroes": red_heroes,
                "blue_heroes": blue_heroes,
                "agents": agents,
                **kwargs,
            }
        )

    monkeypatch.setattr(module, "run_game", capture_run)
    out = tmp_path / "labels.jsonl"

    assert module.main(_args(out)) == 0

    assert {run["seed"] for run in runs} == {11, 12}
    all_agents: set[object] = set()
    for run in runs:
        seed = run["seed"]
        agents = run["agents"]
        red_agent = agents["hero_wasp"]
        blue_agent = agents["hero_arien"]
        red_call = next(call for call in _AgentSpy.calls if call[2] is red_agent)
        blue_call = next(call for call in _AgentSpy.calls if call[2] is blue_agent)
        assert agents["hero_xargatha"] is red_agent
        assert agents["hero_brogan"] is blue_agent
        assert red_agent is not blue_agent
        assert red_agent not in all_agents and blue_agent not in all_agents
        all_agents.update((red_agent, blue_agent))
        assert run["max_steps"] == 700
        configs = [red_call[1]["config"], blue_call[1]["config"]]
        assert all(c.iterations == 7 and c.cutoff_rounds == 3 for c in configs)
        assert configs[0].seed == module.agent_seed(seed, "RED")
        assert configs[1].seed == module.agent_seed(seed, "BLUE")
        assert configs[0].seed != configs[1].seed
        assert red_call[1]["cutoff_observer"] is not None
        assert blue_call[1]["cutoff_observer"] is not None
        assert red_call[1]["cutoff_observer"] is not blue_call[1]["cutoff_observer"]

    for args, kwargs in _ObserverSpy.calls:
        assert Path(args[0] if args else kwargs["path"]) == out
        assert kwargs["sample_every"] == 5 and kwargs["max_samples"] == 6
        assert kwargs["continuation_max_steps"] == 800
        assert kwargs["continuation_max_rounds"] == 9
        assert kwargs["red_heroes"] == RED and kwargs["blue_heroes"] == BLUE
        assert kwargs["source_revision"] == "rev" and kwargs["dirty_tree_hash"] == "dirty"
    labels = [call[1]["agent_label"].upper() for call in _ObserverSpy.calls[:2]]
    assert any("RED" in label for label in labels) and any("BLUE" in label for label in labels)


def test_checkpoint_resume_preserves_output_and_is_scoped_to_full_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    out = tmp_path / "labels.jsonl"
    out.write_text('{"partial":true}\n')
    runs: list[int] = []
    fsyncs: list[int] = []
    monkeypatch.setattr(module.os, "fsync", fsyncs.append)
    monkeypatch.setattr(module, "run_game", lambda *a, **kw: runs.append(kw["seed"]))

    assert module.main(_args(out)) == 0
    checkpoint = Path(str(out) + ".games.jsonl")
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert runs == [11, 12] and [row["world_seed"] for row in rows] == [11, 12]
    assert len(fsyncs) == 2
    first_ids = [call[1]["source_game_id"] for call in _ObserverSpy.calls]
    assert out.read_text() == '{"partial":true}\n'

    assert module.main(_args(out)) == 0
    assert runs == [11, 12]
    assert len(checkpoint.read_text().splitlines()) == 2

    assert module.main(_args(out, iterations=8)) == 0
    changed_ids = [call[1]["source_game_id"] for call in _ObserverSpy.calls[-4:]]
    assert runs == [11, 12, 11, 12]
    assert set(first_ids).isdisjoint(changed_ids)
    assert len(checkpoint.read_text().splitlines()) == 4


def test_failed_source_game_is_not_checkpointed_and_can_resume_partial_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    out = tmp_path / "labels.jsonl"
    checkpoint = tmp_path / "custom-checkpoint.jsonl"
    calls = 0

    def interrupted(*_: Any, **__: Any) -> None:
        nonlocal calls
        calls += 1
        out.write_text('{"sample_id":"already-durable"}\n')
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "run_game", interrupted)
    args = _args(out, checkpoint)
    index = args.index("--iterations")
    del args[index : index + 2]
    with pytest.raises(RuntimeError, match="interrupted"):
        module.main(args)
    assert _AgentSpy.calls[0][1]["config"].iterations == 4
    assert not checkpoint.exists() or checkpoint.read_text() == ""

    monkeypatch.setattr(module, "run_game", lambda *_a, **_kw: None)
    assert module.main(args) == 0
    assert calls == 1
    assert out.read_text() == '{"sample_id":"already-durable"}\n'


def test_feature_schema_is_validated_wired_and_part_of_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    runs: list[int] = []
    monkeypatch.setattr(module, "run_game", lambda *a, **kw: runs.append(kw["seed"]))
    out = tmp_path / "labels.jsonl"

    assert module.main([*_args(out), "--feature-schema", "rich-v1"]) == 0
    rich_ids = {call[1]["source_game_id"] for call in _ObserverSpy.calls}
    assert {call[1]["feature_schema"] for call in _ObserverSpy.calls} == {"rich-v1"}
    assert module.main(_args(out)) == 0
    base_ids = {call[1]["source_game_id"] for call in _ObserverSpy.calls[4:]}
    assert runs == [11, 12, 11, 12]
    assert rich_ids.isdisjoint(base_ids)
    with pytest.raises(SystemExit):
        module.main([*_args(out), "--feature-schema", "unknown-v1"])


def test_source_timeout_cli_is_a_positive_float_defaulting_to_1800() -> None:
    parser = _module()._parser()
    required = ["--out", "labels.jsonl", "--seed-start", "1", "--seed-end", "2"]

    assert parser.parse_args(required).source_timeout_seconds == 1800.0
    assert (
        parser.parse_args([*required, "--source-timeout-seconds", "0.02"]).source_timeout_seconds
        == 0.02
    )
    for value in ("0", "-0.1"):
        with pytest.raises(SystemExit):
            parser.parse_args([*required, "--source-timeout-seconds", value])


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX timers")
def test_source_timeout_checkpoints_and_continues_and_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _install_spies(monkeypatch, module)
    out = tmp_path / "labels.jsonl"
    out.write_text('{"partial_terminal_label":true}\n')
    checkpoint = tmp_path / "games.jsonl"
    calls: list[int] = []

    def run(*_: Any, **kwargs: Any) -> None:
        calls.append(kwargs["seed"])
        if kwargs["seed"] == 11:
            time.sleep(0.1)

    monkeypatch.setattr(module, "run_game", run)
    args = [*_args(out, checkpoint), "--source-timeout-seconds", "0.02"]
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    assert module.main(args) == 0

    assert calls == [11, 12]
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == previous_timer
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    timeout = rows[0]
    assert timeout["world_seed"] == 11
    assert isinstance(timeout["source_config_id"], str)
    assert timeout["reason"] == "wall_clock_timeout"
    assert timeout["winner"] is None
    assert out.read_text() == '{"partial_terminal_label":true}\n'

    calls.clear()
    assert module.main(args) == 0
    assert calls == []

    changed = [*_args(out, checkpoint), "--source-timeout-seconds", "0.03"]
    monkeypatch.setattr(module, "run_game", lambda *a, **kw: calls.append(kw["seed"]))
    assert module.main(changed) == 0
    assert calls == [11, 12]
