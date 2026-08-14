"""Tests for the read-only replay-debugger API (omniscient view)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from goa2.domain.types import HeroID
from goa2.domain.views import build_view
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup
from goa2.server import routes_replays
from goa2.server.app import create_app
from goa2.server.replay import ReplayRecorder, _resolve_map_path

MAP = "forgotten_island"
RED = ["Arien"]
BLUE = ["Wasp"]
GAME_ID = "replaytest1"


def _hero_ids(state) -> list[str]:
    return [h.id for team in state.teams.values() for h in team.heroes]


def _record_two_commit_game(seed: int = 42) -> GameSession:
    """Record a short deterministic game (one commit per hero) into GOA2_REPLAY_DIR.

    Returns the live GameSession it was recorded from, so tests can compare the
    reconstructed state against ground truth.
    """
    state = GameSetup.create_game(_resolve_map_path(MAP), RED, BLUE, False, "QUICK", seed=seed)
    live = GameSession(state)

    rec = ReplayRecorder(GAME_ID)
    rec.record_setup(
        map_name=MAP, red_heroes=RED, blue_heroes=BLUE, game_type="QUICK", cheats=False, seed=seed
    )
    for hero_id in _hero_ids(live.state):
        card = live.state.get_hero(HeroID(hero_id)).hand[0]
        rec.record_commit(hero_id, card.id, live.state.round, live.state.turn)
        live.commit_card(HeroID(hero_id), card)
    return live


def _record_partial_game(game_id: str, seed: int = 42):
    """Record a game that is still being played: first hero committed, second not.

    Returns (live_session, finish) where calling ``finish()`` commits the second
    hero and appends that decision to the same on-disk log — the shape of a
    replay being viewed while its game is still running.
    """
    state = GameSetup.create_game(_resolve_map_path(MAP), RED, BLUE, False, "QUICK", seed=seed)
    live = GameSession(state)
    rec = ReplayRecorder(game_id)
    rec.record_setup(
        map_name=MAP, red_heroes=RED, blue_heroes=BLUE, game_type="QUICK", cheats=False, seed=seed
    )
    first, second = _hero_ids(live.state)

    def commit(hero_id: str) -> None:
        card = live.state.get_hero(HeroID(hero_id)).hand[0]
        rec.record_commit(hero_id, card.id, live.state.round, live.state.turn)
        live.commit_card(HeroID(hero_id), card)

    commit(first)
    return live, lambda: commit(second)


@pytest.fixture
def client_with_replay():
    """A TestClient with the replay API enabled and a recorded game on disk."""
    routes_replays._CACHE.clear()
    prev = os.environ.get("GOA2_REPLAY_API")
    os.environ["GOA2_REPLAY_API"] = "1"
    try:
        _record_two_commit_game()
        with TestClient(create_app()) as client:
            yield client
    finally:
        routes_replays._CACHE.clear()
        if prev is None:
            os.environ.pop("GOA2_REPLAY_API", None)
        else:
            os.environ["GOA2_REPLAY_API"] = prev


# --- listing & metadata ---------------------------------------------------


def test_list_replays_returns_recorded_game(client_with_replay):
    rows = client_with_replay.get("/replays").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["game_id"] == GAME_ID
    assert row["map"] == MAP
    assert row["red"] == RED
    assert row["blue"] == BLUE
    assert row["num_decisions"] == 2


def test_meta_returns_full_decision_list(client_with_replay):
    meta = client_with_replay.get(f"/replays/{GAME_ID}").json()
    assert meta["setup"]["map"] == MAP
    decisions = meta["decisions"]
    assert len(decisions) == 2
    assert [d["index"] for d in decisions] == [0, 1]
    assert all(d["type"] == "commit" for d in decisions)


# --- stepping -------------------------------------------------------------


def test_state_at_each_decision_index(client_with_replay):
    for n in (0, 1, 2):
        body = client_with_replay.get(f"/replays/{GAME_ID}/state?decision={n}").json()
        assert body["position"]["decision_index"] == n
        assert body["position"]["total_decisions"] == 2
        assert body["view"]["teams"]  # a real view came back


def test_state_clamps_out_of_range(client_with_replay):
    body = client_with_replay.get(f"/replays/{GAME_ID}/state?decision=999").json()
    assert body["position"]["decision_index"] == 2  # clamped to total


def test_round_jump_past_end_lands_at_total(client_with_replay):
    body = client_with_replay.get(f"/replays/{GAME_ID}/state?round=99").json()
    assert body["position"]["decision_index"] == 2


# --- omniscient view (the security-sensitive part) ------------------------


def test_reveal_all_shows_every_hand(client_with_replay):
    body = client_with_replay.get(f"/replays/{GAME_ID}/state?decision=0").json()
    heroes = [h for team in body["view"]["teams"].values() for h in team["heroes"]]
    assert len(heroes) == 2
    # Omniscient: every hero's hand is populated, not just one player's.
    assert all(len(h["hand"]) > 0 for h in heroes)


def test_cache_reloads_when_the_log_grows(client_with_replay):
    """A replay of a *live* game keeps growing on disk; the cursor must follow it.

    Caching by game_id alone froze `total_decisions` at whatever the file held
    on first view, so the transport counter capped below the real end of the
    game until the server restarted.
    """
    # A second game recorded mid-play: only the first hero has committed so far.
    live, pending = _record_partial_game("growing")
    before = client_with_replay.get("/replays/growing/state?decision=999").json()
    assert before["position"]["total_decisions"] == 1

    # The game continues: the remaining decision is appended to the same log.
    pending()

    after = client_with_replay.get("/replays/growing/state?decision=999").json()
    assert after["position"]["total_decisions"] == 2
    assert after["position"]["decision_index"] == 2
    assert _strip_volatile(after["view"]) == _strip_volatile(
        build_view(live.state, reveal_all=True)
    )


def test_growing_log_reuses_the_cursor_instead_of_rebuilding(client_with_replay):
    """Appends must not throw away applied work — the log is append-only.

    Rebuilding from the seed on every append costs a full re-simulation; reusing
    the cursor costs a re-parse. Cursor identity is the observable proxy for
    "the session was kept".
    """
    _live, pending = _record_partial_game("growing2")
    client_with_replay.get("/replays/growing2/state?decision=1")
    before = routes_replays._CACHE["growing2"][1]
    assert before.cursor == 1

    pending()
    client_with_replay.get("/replays/growing2/state?decision=2")
    after = routes_replays._CACHE["growing2"][1]

    assert after is before, "cursor was rebuilt instead of extended"
    assert after.total == 2
    assert after.cursor == 2


def test_divergent_log_still_rebuilds(client_with_replay):
    """If the prefix no longer matches, reuse would be wrong — rebuild instead."""
    _record_partial_game("diverging")
    client_with_replay.get("/replays/diverging/state?decision=1")
    before = routes_replays._CACHE["diverging"][1]

    # Rewrite the log with a different first decision (not a real append).
    path = Path(os.environ["GOA2_REPLAY_DIR"]) / "diverging.jsonl"
    lines = path.read_text().splitlines()
    header, first = lines[0], json.loads(lines[1])
    first["card"] = "some_other_card"
    path.write_text(header + "\n" + json.dumps(first) + "\n")

    client_with_replay.get("/replays/diverging/state?decision=0")
    assert routes_replays._CACHE["diverging"][1] is not before


def test_concurrent_seeks_on_one_game_stay_consistent(client_with_replay):
    """Two threads sharing a cursor must not interleave into a corrupted session."""
    expected = {
        n: _strip_volatile(client_with_replay.get(f"/replays/{GAME_ID}/state?decision={n}").json())
        for n in (0, 1, 2)
    }
    errors: list[str] = []

    def hammer() -> None:
        for _ in range(10):
            for n in (0, 2, 1):
                body = client_with_replay.get(f"/replays/{GAME_ID}/state?decision={n}").json()
                if _strip_volatile(body) != expected[n]:
                    errors.append(f"decision {n} returned a mismatched view")

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]


def test_cache_path_matches_cold_rebuild(client_with_replay):
    # Forward through the cache: 0 -> 1 -> 2.
    client_with_replay.get(f"/replays/{GAME_ID}/state?decision=1")
    warm = client_with_replay.get(f"/replays/{GAME_ID}/state?decision=2").json()
    # Cold rebuild at the same index.
    routes_replays._CACHE.clear()
    cold = client_with_replay.get(f"/replays/{GAME_ID}/state?decision=2").json()
    assert warm["position"] == cold["position"]
    assert _strip_volatile(warm["view"]) == _strip_volatile(cold["view"])


# --- errors & gating ------------------------------------------------------


def test_unknown_replay_404(client_with_replay):
    assert client_with_replay.get("/replays/nope/state?decision=0").status_code == 404


def test_path_traversal_rejected(client_with_replay):
    assert client_with_replay.get("/replays/..%2F..%2Fsecret/state").status_code == 404


def test_api_disabled_by_default():
    """Without GOA2_REPLAY_API the router is not mounted — endpoints 404."""
    prev = os.environ.get("GOA2_REPLAY_API")
    os.environ.pop("GOA2_REPLAY_API", None)
    try:
        with TestClient(create_app()) as client:
            assert client.get("/replays").status_code == 404
    finally:
        if prev is not None:
            os.environ["GOA2_REPLAY_API"] = prev


# --- build_view security lock (pure, no server) ---------------------------


def test_build_view_hides_other_hands_by_default():
    """The live view path must keep hiding non-owner hands; reveal_all is opt-in."""
    state = GameSetup.create_game(_resolve_map_path(MAP), RED, BLUE, False, "QUICK", seed=7)
    hero_ids = _hero_ids(state)
    me, other = HeroID(hero_ids[0]), hero_ids[1]

    scoped = build_view(state, for_hero_id=me)

    def hand_of(view, hid):
        for team in view["teams"].values():
            for h in team["heroes"]:
                if h["id"] == hid:
                    return h["hand"]
        raise AssertionError(hid)

    assert len(hand_of(scoped, str(me))) > 0  # I see my own hand
    assert hand_of(scoped, other) == []  # opponent's hand hidden

    # Omniscient reveals everyone.
    omni = build_view(state, reveal_all=True)
    assert len(hand_of(omni, str(me))) > 0
    assert len(hand_of(omni, other)) > 0


def _strip_volatile(obj):
    """Drop non-deterministic instance identifiers (step_id = id(object()))."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k != "step_id"}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj
