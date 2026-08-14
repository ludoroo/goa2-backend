"""Tests for the minimal game replay log (record + reconstruct)."""

from __future__ import annotations

import asyncio
import json

import pytest

from goa2.domain.input import InputResponse
from goa2.domain.types import HeroID
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.setup import GameSetup
from goa2.server.registry import ManagedGame
from goa2.server.replay import (
    ReplayRecorder,
    _resolve_map_path,
    cleanup_old_replays,
    replay_game,
)
from goa2.server.ws import _handle_commit_card, _handle_finish_planning, _handle_pass_turn

MAP = "forgotten_island"
RED = ["Arien"]
BLUE = ["Wasp"]


def _strip_volatile(obj):
    """Drop non-deterministic instance identifiers (step_id = id(object()))."""
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v)
            for k, v in obj.items()
            if k not in {"step_id", "pending_request_id"}
        }
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _hero_ids(state) -> list[str]:
    return [h.id for team in state.teams.values() for h in team.heroes]


def _first_card_id(state, hero_id: str) -> str:
    hero = state.get_hero(HeroID(hero_id))
    return hero.hand[0].id


def _live_game(seed: int = 42) -> GameSession:
    state = GameSetup.create_game(_resolve_map_path(MAP), RED, BLUE, False, "QUICK", seed=seed)
    return GameSession(state)


def _record_setup(rec: ReplayRecorder, seed: int = 42) -> None:
    rec.record_setup(
        map_name=MAP,
        red_heroes=RED,
        blue_heroes=BLUE,
        game_type="QUICK",
        cheats=False,
        seed=seed,
    )


def test_record_writes_setup_header_first(tmp_path):
    rec = ReplayRecorder("g1", str(tmp_path))
    _record_setup(rec)
    lines = (tmp_path / "g1.jsonl").read_text().splitlines()
    header = json.loads(lines[0])
    assert header["type"] == "setup"
    assert header["map"] == MAP
    assert header["red"] == RED
    assert header["blue"] == BLUE
    assert header["seed"] == 42
    assert header["v"] == 1


def test_record_decisions_carry_wall_clock_timestamps(tmp_path):
    import time as _time

    rec = ReplayRecorder("g1", str(tmp_path))
    _record_setup(rec)
    before = _time.time()
    rec.record_commit("hero_arien", "arien_basic_1", 1, 1)
    rec.record_input("hero_arien", "minion_1", 1, 2)
    rec.record_rollback("hero_arien", 1, 2)
    rec.record_cheat_gold("hero_arien", 5, 1, 1)
    rec.record_pass("hero_arien", 1, 2)
    rec.record_uncommit("hero_arien", 1, 1)
    rec.record_timer_timeout(action="commit", hero_id="hero_arien", round_num=1, turn=1)
    after = _time.time()

    records = [json.loads(line) for line in (tmp_path / "g1.jsonl").read_text().splitlines()]
    assert records[0]["type"] == "setup"
    assert isinstance(records[0]["ts"], float)
    for record in records[1:]:
        assert isinstance(record["ts"], float)
        assert before <= record["ts"] <= after
    # Decision timestamps are non-decreasing (appended in write order).
    assert [r["ts"] for r in records] == sorted(r["ts"] for r in records)


def test_record_and_replay_roundtrip(tmp_path):
    seed = 42
    live = _live_game(seed)

    rec = ReplayRecorder("g1", str(tmp_path))
    _record_setup(rec, seed)

    for hero_id in _hero_ids(live.state):
        card_id = _first_card_id(live.state, hero_id)
        rec.record_commit(hero_id, card_id, live.state.round, live.state.turn)
        card = next(c for c in live.state.get_hero(HeroID(hero_id)).hand if c.id == card_id)
        live.commit_card(HeroID(hero_id), card)

    replayed = replay_game(str(tmp_path / "g1.jsonl"))

    assert _strip_volatile(replayed.state.model_dump(mode="json")) == _strip_volatile(
        live.state.model_dump(mode="json")
    )


def test_rejected_websocket_actions_do_not_leave_phantom_replay_decisions(tmp_path):
    live = _live_game()
    recorder = ReplayRecorder("g1", str(tmp_path))
    _record_setup(recorder)
    game = ManagedGame(
        game_id="g1",
        session=live,
        player_tokens={},
        spectator_token="spectator",
        hero_to_token={},
        replay_recorder=recorder,
    )
    hero_id = "hero_arien"
    first_card = _first_card_id(live.state, hero_id)

    with pytest.raises(ValueError):
        asyncio.run(_handle_finish_planning(game, hero_id))
    asyncio.run(_handle_commit_card(game, hero_id, {"card_id": first_card}))
    second_card = _first_card_id(live.state, hero_id)
    with pytest.raises(ValueError):
        asyncio.run(_handle_commit_card(game, hero_id, {"card_id": second_card}))
    with pytest.raises(ValueError):
        asyncio.run(_handle_pass_turn(game, hero_id))

    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [record["type"] for record in records] == ["setup", "commit"]
    replay_game(str(recorder.path))


def test_replay_stops_at_decision_index(tmp_path):
    """until_decision=1 applies only the first decision (one commit)."""
    live = _live_game()
    rec = ReplayRecorder("g1", str(tmp_path))
    _record_setup(rec)

    hero_ids = _hero_ids(live.state)
    for hero_id in hero_ids:
        card_id = _first_card_id(live.state, hero_id)
        rec.record_commit(hero_id, card_id, live.state.round, live.state.turn)
        card = next(c for c in live.state.get_hero(HeroID(hero_id)).hand if c.id == card_id)
        live.commit_card(HeroID(hero_id), card)

    partial = replay_game(str(tmp_path / "g1.jsonl"), until_decision=1)
    # Only the first hero has committed -> exactly one pending input recorded.
    assert len(partial.state.pending_inputs) == 1


def test_cleanup_old_replays_removes_only_stale(tmp_path):
    import os
    import time

    old = tmp_path / "old.jsonl"
    fresh = tmp_path / "fresh.jsonl"
    old.write_text('{"type":"setup"}\n')
    fresh.write_text('{"type":"setup"}\n')

    # Age the old file 31 days.
    stale = time.time() - 31 * 86400
    os.utime(old, (stale, stale))

    removed = cleanup_old_replays(str(tmp_path), ttl_days=30)
    assert removed == 1
    assert not old.exists()
    assert fresh.exists()


def test_replay_unknown_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay_game(str(tmp_path / "missing.jsonl"))


def _build_log(tmp_path, seed: int = 42) -> str:
    """Drive a short game while recording, returning the replay file path."""
    live = _live_game(seed)
    rec = ReplayRecorder("cli", str(tmp_path))
    _record_setup(rec, seed)
    for hero_id in _hero_ids(live.state):
        card_id = _first_card_id(live.state, hero_id)
        rec.record_commit(hero_id, card_id, live.state.round, live.state.turn)
        card = next(c for c in live.state.get_hero(HeroID(hero_id)).hand if c.id == card_id)
        live.commit_card(HeroID(hero_id), card)
    return str(tmp_path / "cli.jsonl")


def test_cli_replay_prints_summary(tmp_path, capsys):
    from goa2.scripts.replay import main

    path = _build_log(tmp_path)
    rc = main([path])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Replay:" in out
    assert "Stopped at:" in out


def test_cli_decision_stop(tmp_path, capsys):
    from goa2.scripts.replay import main

    path = _build_log(tmp_path)
    assert main([path, "--decision", "1"]) == 0


def test_cli_missing_file_returns_error(tmp_path, capsys):
    from goa2.scripts.replay import main

    rc = main([str(tmp_path / "nope.jsonl")])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_end_to_end_record_via_server_then_replay(tmp_path, monkeypatch):
    """Play a game through the real REST server, then replay its recorded log."""
    import os

    from fastapi.testclient import TestClient

    from goa2.server.app import create_app

    monkeypatch.setenv("GOA2_SAVE_DIR", str(tmp_path / "saves"))
    replay_dir = tmp_path / "replays"
    monkeypatch.setenv("GOA2_REPLAY_DIR", str(replay_dir))

    with TestClient(create_app()) as client:
        resp = client.post(
            "/games",
            json={"map_name": MAP, "red_heroes": RED, "blue_heroes": BLUE, "game_type": "QUICK"},
        )
        assert resp.status_code == 201
        data = resp.json()
        game_id = data["game_id"]
        tokens = {pt["hero_id"]: pt["token"] for pt in data["player_tokens"]}

        registry = client.app.state.registry
        live_state = registry.get(game_id).session.state

        for hero_id, token in tokens.items():
            view = client.get(
                f"/games/{game_id}", headers={"Authorization": f"Bearer {token}"}
            ).json()
            hand = next(
                h["hand"]
                for team in view["view"]["teams"].values()
                for h in team["heroes"]
                if h["id"] == hero_id
            )
            client.post(
                f"/games/{game_id}/cards",
                json={"card_id": hand[0]["id"]},
                headers={"Authorization": f"Bearer {token}"},
            )

        replay_file = replay_dir / f"{game_id}.jsonl"
        assert replay_file.is_file()

        replayed = replay_game(str(replay_file))
        assert _strip_volatile(replayed.state.model_dump(mode="json")) == _strip_volatile(
            live_state.model_dump(mode="json")
        )

    assert os.path.exists(replay_dir)


def _pick(req: dict):
    """First legal selection for an input request (test auto-player)."""
    if req.get("valid_options"):
        return req["valid_options"][0]
    if req.get("valid_hexes"):
        return req["valid_hexes"][0]
    if req.get("options"):
        opt = req["options"][0]
        return opt["id"] if isinstance(opt, dict) else opt
    return "SKIP"


def test_replay_roundtrip_with_cheat_gold(tmp_path):
    """A recorded gold cheat reconstructs the same gold (was previously lost)."""
    seed = 11
    state = GameSetup.create_game(_resolve_map_path(MAP), RED, BLUE, True, "QUICK", seed=seed)
    live = GameSession(state)

    rec = ReplayRecorder("ch", str(tmp_path))
    rec.record_setup(
        map_name=MAP, red_heroes=RED, blue_heroes=BLUE, game_type="QUICK", cheats=True, seed=seed
    )

    hero_id = _hero_ids(live.state)[0]
    # Apply a cheat the way the server does, and record it.
    live.state.get_hero(HeroID(hero_id)).gold += 7
    rec.record_cheat_gold(hero_id, 7, live.state.round, live.state.turn)
    # Plus a normal commit so the log isn't cheat-only.
    card = live.state.get_hero(HeroID(hero_id)).hand[0]
    rec.record_commit(hero_id, card.id, live.state.round, live.state.turn)
    live.commit_card(HeroID(hero_id), card)

    replayed = replay_game(str(tmp_path / "ch.jsonl"))
    assert replayed.state.get_hero(HeroID(hero_id)).gold == 7
    assert _strip_volatile(replayed.state.model_dump(mode="json")) == _strip_volatile(
        live.state.model_dump(mode="json")
    )


def test_replay_roundtrip_with_rollback(tmp_path):
    """A game where the actor rolls back mid-turn reconstructs faithfully."""
    seed = 7
    live = _live_game(seed)

    rec = ReplayRecorder("rb", str(tmp_path))
    _record_setup(rec, seed)

    # Commit every hero (enters resolution); keep the last result to drive from.
    result = None
    for hero_id in _hero_ids(live.state):
        card = live.state.get_hero(HeroID(hero_id)).hand[0]
        rec.record_commit(hero_id, card.id, live.state.round, live.state.turn)
        result = live.commit_card(HeroID(hero_id), card)

    did_rollback = False
    for _ in range(300):
        if live.state.phase.value == "GAME_OVER":
            break
        if result.result_type == SessionResultType.INPUT_NEEDED and result.input_request:
            req = result.input_request.to_dict()
            sel = _pick(req)
            r, t = live.state.round, live.state.turn
            rec.record_input("", sel, r, t)
            assert result.input_request is not None
            result = live.advance(InputResponse(request_id=result.input_request.id, selection=sel))

            # The first rollback-able input: undo and re-answer it.
            if req.get("can_rollback") and not did_rollback:
                rr, rt = live.state.round, live.state.turn
                rec.record_rollback("", rr, rt)
                result = live.rollback()
                did_rollback = True
        elif live.state.phase.value == "PLANNING":
            # Resolution (including our rollback) finished; stop at this boundary.
            break
        else:
            result = live.advance()

    assert did_rollback, "expected a rollback-able resolution input in this game"

    replayed = replay_game(str(tmp_path / "rb.jsonl"))
    assert _strip_volatile(replayed.state.model_dump(mode="json")) == _strip_volatile(
        live.state.model_dump(mode="json")
    )


def test_record_and_replay_roundtrip_with_uncommit(tmp_path):
    """commit -> uncommit -> commit(other) reconstructs to the live state."""
    seed = 42
    live = _live_game(seed)
    rec = ReplayRecorder("g1", str(tmp_path))
    _record_setup(rec, seed)

    first_hero = _hero_ids(live.state)[0]
    hero = live.state.get_hero(HeroID(first_hero))
    card0, card1 = hero.hand[0], hero.hand[1]

    rec.record_commit(first_hero, card0.id, live.state.round, live.state.turn)
    live.commit_card(HeroID(first_hero), card0)
    rec.record_uncommit(first_hero, live.state.round, live.state.turn)
    live.uncommit_card(HeroID(first_hero))
    rec.record_commit(first_hero, card1.id, live.state.round, live.state.turn)
    live.commit_card(HeroID(first_hero), card1)

    replayed = replay_game(str(tmp_path / "g1.jsonl"))

    assert replayed.state.pending_inputs[first_hero].id == card1.id
    replayed_hero = replayed.state.get_hero(HeroID(first_hero))
    assert any(c.id == card0.id for c in replayed_hero.hand)
    assert _strip_volatile(replayed.state.model_dump(mode="json")) == _strip_volatile(
        live.state.model_dump(mode="json")
    )
