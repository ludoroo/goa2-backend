"""RED behavioral tests for deterministic terminal labels at search cutoffs."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.evaluation.features import FEATURE_NAMES, RICH_FEATURE_NAMES
from automata.runtime.harness import DEFAULT_MAP, RunResult
from goa2.domain.models import TeamColor
from goa2.engine.setup import GameSetup


def _observer_type() -> type[Any]:
    try:
        module = importlib.import_module("automata.evaluation.terminal_labels")
    except ModuleNotFoundError:
        pytest.fail("TerminalLabelObserver is not implemented")
    return module.TerminalLabelObserver


def _state(seed: int = 7):
    return GameSetup.create_game(
        map_path=DEFAULT_MAP,
        red_heroes=["Wasp"],
        blue_heroes=["Arien"],
        game_type="QUICK",
        seed=seed,
    )


def _kwargs(path: Path, continuation: Any, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "path": path,
        "source_game_id": "source-17",
        "world_seed": 17,
        "agent_label": "ismcts-a",
        "red_heroes": ["Wasp"],
        "blue_heroes": ["Arien"],
        "source_revision": "rev-abc",
        "dirty_tree_hash": "clean",
        "sample_every": 2,
        "max_samples": 3,
        "continuation_max_steps": 80,
        "continuation_max_rounds": 4,
        "continuation_fn": continuation,
    }
    values.update(overrides)
    return values


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_samples_fixed_stride_clones_state_and_writes_value_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, dict[str, HeuristicAgent], dict[str, Any]]] = []

    def continue_fake(state: Any, agents: Any, **caps: Any) -> RunResult:
        calls.append((state, agents, caps))
        state.teams[TeamColor.RED].life_counters = 0
        winner = "BLUE" if len(calls) == 1 else "RED"
        return RunResult(winner, state.round + 1, 2, 9, "game_over")

    source = _state()
    before = source.model_dump(mode="json")
    out = tmp_path / "labels.jsonl"
    observer_type = _observer_type()
    fsync_calls: list[int] = []
    module = importlib.import_module("automata.evaluation.terminal_labels")
    monkeypatch.setattr(module.os, "fsync", fsync_calls.append)
    observer = observer_type(**_kwargs(out, continue_fake))

    for _ in range(4):
        observer(source, TeamColor.RED, 0.25)

    assert len(calls) == 2  # zero-based ordinals 0 and 2
    assert all(call[0] is not source for call in calls)
    assert source.model_dump(mode="json") == before
    assert calls[0][2] == {"max_steps": 80, "max_rounds": 4}
    assert set(calls[0][1]) == {"hero_wasp", "hero_arien"}
    assert all(isinstance(agent, HeuristicAgent) for agent in calls[0][1].values())

    rows = _rows(out)
    assert len(rows) == 2
    assert [item["winner"] for item in rows] == ["BLUE", "RED"]
    assert len(fsync_calls) == 2
    row = rows[0]
    assert row["game_id"] == "source-17"
    assert row["world_seed"] == 17
    assert row["team"] == "RED"
    assert len(row["features"]) == 6
    assert row["feature_names"] == list(FEATURE_NAMES)
    assert row["winner"] == "BLUE"
    assert row["red_heroes"] == ["Wasp"]
    assert row["blue_heroes"] == ["Arien"]
    assert row["source_revision"] == "rev-abc"
    assert row["dirty_tree_hash"] == "clean"
    assert row["agent_label"] == "ismcts-a"
    assert row["cutoff_ordinal"] == 0
    assert row["cutoff_round"] == source.round
    assert row["active_value"] == 0.25
    assert row["continuation"] == {
        "max_steps": 80,
        "max_rounds": 4,
        "rounds": source.round + 1,
        "steps": 9,
        "reason": "game_over",
    }
    assert isinstance(row["sample_id"], str) and len(row["sample_id"]) == 64


def test_output_is_byte_deterministic_and_does_not_consume_search_rng(tmp_path: Path) -> None:
    def run(path: Path, burn: int) -> tuple[bytes, object]:
        seen: list[int] = []

        def continuation(_: Any, agents: Any, **__: Any) -> RunResult:
            seen.extend(agent._rng.getrandbits(64) for agent in agents.values())
            return RunResult("RED", 3, 4, 12, "game_over")

        active = HeuristicAgent(seed=991)
        for _ in range(burn):
            active._rng.random()
        rng_before = active._rng.getstate()
        observer = _observer_type()(**_kwargs(path, continuation, sample_every=1))
        observer(_state(), TeamColor.BLUE, -0.5)
        assert active._rng.getstate() == rng_before
        return path.read_bytes(), seen

    left_bytes, left_seeds = run(tmp_path / "left.jsonl", 0)
    right_bytes, right_seeds = run(tmp_path / "right.jsonl", 200)
    assert left_bytes == right_bytes
    assert left_seeds == right_seeds


def test_resume_skips_existing_ids_and_config_changes_identity(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    calls = 0

    def continuation(*_: Any, **__: Any) -> RunResult:
        nonlocal calls
        calls += 1
        return RunResult("RED", 2, 1, 3, "game_over")

    first = _observer_type()(**_kwargs(out, continuation, sample_every=1))
    first(_state(), TeamColor.RED, 0.0)
    original = out.read_bytes()

    resumed = _observer_type()(**_kwargs(out, continuation, sample_every=1))
    resumed(_state(), TeamColor.RED, 0.0)
    assert out.read_bytes() == original
    assert calls == 1

    changed = _observer_type()(
        **_kwargs(out, continuation, sample_every=1, continuation_max_steps=81)
    )
    changed(_state(), TeamColor.RED, 0.0)
    rows = _rows(out)
    assert len(rows) == 2
    assert rows[0]["sample_id"] != rows[1]["sample_id"]


def test_cap_and_incomplete_continuations_are_counted_not_written(tmp_path: Path) -> None:
    outcomes = iter(
        [
            RunResult(None, 5, 2, 80, "max_steps"),
            RunResult("BLUE", 6, 2, 10, "game_over"),
            RunResult("RED", 7, 2, 11, "game_over"),
        ]
    )

    def continuation(*_: Any, **__: Any) -> RunResult:
        return next(outcomes)

    out = tmp_path / "labels.jsonl"
    observer = _observer_type()(**_kwargs(out, continuation, sample_every=1, max_samples=2))
    for _ in range(4):
        observer(_state(), TeamColor.RED, 0.0)

    assert [row["winner"] for row in _rows(out)] == ["BLUE"]
    assert observer.stats.recorded_samples == 1
    assert observer.stats.skipped_samples == 1
    assert observer.stats.sampled == 2


def test_explicit_rich_schema_controls_rows_and_sample_identity(tmp_path: Path) -> None:
    def continuation(*_: Any, **__: Any) -> RunResult:
        return RunResult("RED", 2, 1, 3, "game_over")

    state = _state()
    base_path, rich_path = tmp_path / "base.jsonl", tmp_path / "rich.jsonl"
    _observer_type()(**_kwargs(base_path, continuation, sample_every=1))(state, TeamColor.RED, 0)
    _observer_type()(**_kwargs(rich_path, continuation, sample_every=1, feature_schema="rich-v1"))(
        state, TeamColor.RED, 0
    )

    base, rich = _rows(base_path)[0], _rows(rich_path)[0]
    assert base["feature_names"] == list(FEATURE_NAMES)
    assert rich["feature_schema"] == "rich-v1"
    assert rich["feature_names"] == list(RICH_FEATURE_NAMES)
    assert len(rich["features"]) == len(RICH_FEATURE_NAMES)
    assert (rich["config_id"], rich["sample_id"]) != (base["config_id"], base["sample_id"])
