"""Replay parity and rewind semantics for override records."""

import json

import pytest

from goa2.engine.overrides import apply_override_decision
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup
from goa2.server.replay import (
    ReplayCursor,
    ReplayRecorder,
    effective_decisions,
    effective_indices,
    load_replay,
    rebuild_session_for_rewind,
    replay_game,
)

MAP_PATH = "src/goa2/data/maps/forgotten_island.json"


def _fresh_session(seed=42) -> GameSession:
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"], False, "QUICK", seed=seed)
    return GameSession(state)


def _write_replay(tmp_path, decisions, seed=42) -> str:
    rec = ReplayRecorder("g1", replay_dir=str(tmp_path))
    rec.record_setup(
        map_name="forgotten_island",
        red_heroes=["Arien"],
        blue_heroes=["Wasp"],
        game_type="QUICK",
        cheats=False,
        seed=seed,
    )
    for d in decisions:
        rec._append(d)
    return str(rec.path)


# ---- effective_indices / effective_decisions ----


def test_effective_indices_no_rewind():
    ds = [{"type": "pass"}, {"type": "pass"}]
    assert effective_indices(ds) == [0, 1]


def test_effective_indices_simple_rewind():
    ds = [
        {"type": "pass", "hero": "a"},  # 0
        {"type": "pass", "hero": "b"},  # 1
        {"type": "ov_rewind", "to": 1},  # 2  -> keep only decision 0
        {"type": "pass", "hero": "c"},  # 3
    ]
    assert effective_indices(ds) == [0, 3]
    assert [d["hero"] for d in effective_decisions(ds)] == ["a", "c"]


def test_effective_indices_nested_rewind():
    ds = [
        {"type": "pass", "hero": "a"},  # 0
        {"type": "ov_rewind", "to": 0},  # 1  -> back to start
        {"type": "pass", "hero": "b"},  # 2
        {"type": "ov_rewind", "to": 3},  # 3  -> effective prefix of 0..2 = [b]
        {"type": "pass", "hero": "c"},  # 4
    ]
    assert [d["hero"] for d in effective_decisions(ds)] == ["b", "c"]


# ---- replay parity (the load-bearing test) ----


def test_override_patch_replay_parity(tmp_path):
    live = _fresh_session()
    apply_override_decision(live, "set_gold", {"hero_id": "hero_arien", "value": 9})
    record = {
        "type": "ov_patch",
        "r": live.state.round,
        "t": live.state.turn,
        "hero": "hero_arien",
        "op": "set_gold",
        "args": {"hero_id": "hero_arien", "value": 9},
        "voters": ["hero_arien", "hero_wasp"],
    }
    path = _write_replay(tmp_path, [record])
    replayed = replay_game(path)
    assert replayed.state.model_dump(
        mode="json", exclude={"clock", "time_control"}
    ) == live.state.model_dump(mode="json", exclude={"clock", "time_control"})


def test_voters_field_ignored_by_reconstruction(tmp_path):
    record = {
        "type": "ov_patch",
        "r": 1,
        "t": 1,
        "hero": "hero_arien",
        "op": "set_gold",
        "args": {"hero_id": "hero_arien", "value": 5},
        "voters": [],
    }
    path = _write_replay(tmp_path, [record])
    replayed = replay_game(path)
    assert replayed.state.get_hero("hero_arien").gold == 5


def test_cheat_gold_legacy_branch_still_loads(tmp_path):
    record = {"type": "cheat_gold", "r": 1, "t": 1, "hero": "hero_arien", "amount": 3}
    path = _write_replay(tmp_path, [record])
    start_gold = _fresh_session().state.get_hero("hero_arien").gold
    replayed = replay_game(path)
    assert replayed.state.get_hero("hero_arien").gold == start_gold + 3


# ---- rewind determinism ----


def test_rewind_record_rebuilds_from_seed(tmp_path):
    ds = [
        {
            "type": "ov_patch",
            "r": 1,
            "t": 1,
            "hero": "hero_arien",
            "op": "set_gold",
            "args": {"hero_id": "hero_arien", "value": 9},
        },
        {"type": "ov_rewind", "r": 1, "t": 1, "hero": "hero_arien", "to": 0, "voters": []},
        {
            "type": "ov_patch",
            "r": 1,
            "t": 1,
            "hero": "hero_arien",
            "op": "set_gold",
            "args": {"hero_id": "hero_arien", "value": 4},
        },
    ]
    path = _write_replay(tmp_path, ds)
    replayed = replay_game(path)
    # The gold=9 segment is dead; only gold=4 applies.
    assert replayed.state.get_hero("hero_arien").gold == 4


def test_cursor_seek_through_rewind_and_back(tmp_path):
    ds = [
        {
            "type": "ov_patch",
            "r": 1,
            "t": 1,
            "hero": "hero_arien",
            "op": "set_gold",
            "args": {"hero_id": "hero_arien", "value": 9},
        },
        {"type": "ov_rewind", "r": 1, "t": 1, "hero": "hero_arien", "to": 0},
        {
            "type": "ov_patch",
            "r": 1,
            "t": 1,
            "hero": "hero_arien",
            "op": "set_gold",
            "args": {"hero_id": "hero_arien", "value": 4},
        },
    ]
    path = _write_replay(tmp_path, ds)
    setup, decisions = load_replay(path)
    cursor = ReplayCursor(setup, decisions)
    s1 = cursor.seek(1)
    assert s1.state.get_hero("hero_arien").gold == 9
    s2 = cursor.seek(2)  # crosses the rewind record
    assert s2.state.get_hero("hero_arien").gold != 9
    s3 = cursor.seek(3)
    assert s3.state.get_hero("hero_arien").gold == 4
    # Backward seek still rebuilds correctly
    s1b = cursor.seek(1)
    assert s1b.state.get_hero("hero_arien").gold == 9


def test_rebuild_session_for_rewind(tmp_path):
    ds = [
        {
            "type": "ov_patch",
            "r": 1,
            "t": 1,
            "hero": "hero_arien",
            "op": "set_gold",
            "args": {"hero_id": "hero_arien", "value": 9},
        },
    ]
    path = _write_replay(tmp_path, ds)
    session = rebuild_session_for_rewind(path, 0)
    assert (
        session.state.get_hero("hero_arien").gold
        == _fresh_session().state.get_hero("hero_arien").gold
    )
    with pytest.raises(ValueError):
        rebuild_session_for_rewind(path, 5)


def test_record_override_appends_with_ts(tmp_path):
    rec = ReplayRecorder("g2", replay_dir=str(tmp_path))
    rec.record_setup(
        map_name="forgotten_island",
        red_heroes=["Arien"],
        blue_heroes=["Wasp"],
        game_type="QUICK",
        cheats=False,
        seed=1,
    )
    rec.record_override(
        {
            "type": "ov_unstick",
            "r": 2,
            "t": 1,
            "hero": "hero_arien",
            "op": "abort_action",
            "args": {},
            "voters": ["hero_arien"],
        }
    )
    with open(rec.path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert lines[-1]["type"] == "ov_unstick"
    assert "ts" in lines[-1]


def test_record_override_rejects_non_override_types(tmp_path):
    rec = ReplayRecorder("g3", replay_dir=str(tmp_path))
    with pytest.raises(ValueError):
        rec.record_override({"type": "pass", "hero": "hero_arien"})
