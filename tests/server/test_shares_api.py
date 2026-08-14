"""Tests for baked, shareable replay artifacts (public read + admin mint/revoke)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from goa2.domain.types import HeroID
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup
from goa2.server import routes_replays, shares
from goa2.server.app import create_app
from goa2.server.replay import ReplayRecorder, _resolve_map_path, cleanup_old_replays

MAP = "forgotten_island"
RED = ["Arien"]
BLUE = ["Wasp"]
FINISHED = "sharefin1"
UNFINISHED = "shareopen1"


def _hero_ids(state) -> list[str]:
    return [h.id for team in state.teams.values() for h in team.heroes]


def _record_game(game_id: str, *, finish: bool, seed: int = 42) -> None:
    """Record a short game; optionally end it with a recorded override.

    Playing a real game to its natural end takes thousands of decisions. An
    ``ov_patch`` dropping a team to zero life counters is a genuine, replayable
    decision that produces a real GAME_OVER state, so the finished fixture
    exercises the same reconstruction path a naturally-won game would.
    """
    state = GameSetup.create_game(_resolve_map_path(MAP), RED, BLUE, False, "QUICK", seed=seed)
    live = GameSession(state)
    rec = ReplayRecorder(game_id)
    rec.record_setup(
        map_name=MAP, red_heroes=RED, blue_heroes=BLUE, game_type="QUICK", cheats=False, seed=seed
    )
    for hero_id in _hero_ids(live.state):
        card = live.state.get_hero(HeroID(hero_id)).hand[0]
        rec.record_commit(hero_id, card.id, live.state.round, live.state.turn)
        live.commit_card(HeroID(hero_id), card)

    if finish:
        rec.record_override(
            {
                "type": "ov_patch",
                "r": live.state.round,
                "t": live.state.turn,
                "hero": _hero_ids(live.state)[0],
                "op": "set_life_counters",
                "args": {"team": "BLUE", "value": 0},
                "voters": _hero_ids(live.state),
            }
        )


@pytest.fixture
def client():
    """TestClient with the admin API enabled and both fixture games recorded."""
    routes_replays._CACHE.clear()
    prev = os.environ.get("GOA2_REPLAY_API")
    os.environ["GOA2_REPLAY_API"] = "1"
    try:
        _record_game(FINISHED, finish=True)
        _record_game(UNFINISHED, finish=False)
        with TestClient(create_app()) as c:
            yield c
    finally:
        routes_replays._CACHE.clear()
        if prev is None:
            os.environ.pop("GOA2_REPLAY_API", None)
        else:
            os.environ["GOA2_REPLAY_API"] = prev


def _mint(client) -> str:
    res = client.post(f"/replays/{FINISHED}/share")
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["url"] == f"/shared/{body['token']}"
    return body["token"]


def _strip_volatile(obj):
    """Drop non-deterministic instance identifiers (step_id = id(object()))."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k != "step_id"}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


# --- minting --------------------------------------------------------------


def test_mint_requires_a_finished_game(client):
    res = client.post(f"/replays/{UNFINISHED}/share")
    assert res.status_code == 409
    assert "finished" in res.json()["detail"]


def test_rejected_mint_leaves_no_artifact(client):
    client.post(f"/replays/{UNFINISHED}/share")
    assert shares.list_shares() == []


def test_mint_unknown_game_404(client):
    assert client.post("/replays/nope/share").status_code == 404


@pytest.mark.parametrize(
    "lines, expected_status",
    [
        # No setup header at all, and a corrupt line: rejected before any bake.
        ([{"type": "pass", "r": 1, "t": 1, "hero": "hero_arien"}], 422),
        ("{not json", 422),
        # Malformed decision records: these reach the engine and must still come
        # back as 422, not a 500. A missing key raises KeyError rather than the
        # ValueError engine drift produces, which once escaped as a 500.
        ([{"type": "commit", "r": 1, "t": 1, "card": "x"}], 422),
        ([{"type": "teleport", "r": 1, "t": 1, "hero": "hero_arien"}], 422),
    ],
)
def test_broken_replays_are_rejected_not_500(client, lines, expected_status):
    path = Path(os.environ["GOA2_REPLAY_DIR"]) / "broken.jsonl"
    header = (Path(os.environ["GOA2_REPLAY_DIR"]) / f"{FINISHED}.jsonl").read_text().splitlines()[0]
    if isinstance(lines, str):
        path.write_text(header + "\n" + lines + "\n")
    else:
        body = "\n".join(json.dumps(d) for d in lines)
        # First case deliberately omits the header to exercise that path.
        prefix = "" if lines[0].get("type") == "pass" else header + "\n"
        path.write_text(prefix + body + "\n")

    res = client.post("/replays/broken/share")
    assert res.status_code == expected_status, res.text
    assert shares.list_shares() == []  # nothing published, no staging left behind


def test_minting_twice_returns_the_same_share(client):
    """A finished game's artifact can never change, so re-sharing must not re-bake."""
    first = _mint(client)
    second = _mint(client)
    assert first == second
    assert len(shares.list_shares()) == 1


# --- listing (drives the replay-list UI) ----------------------------------


def test_list_shares_reports_what_the_ui_needs(client):
    token = _mint(client)
    rows = client.get("/shares").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["token"] == token
    assert row["game_id"] == FINISHED
    assert row["total_decisions"] == 3
    assert row["size_bytes"] > 0  # recorded at bake time, not stat'd per request
    assert row["created_at"] is not None


def test_list_shares_empty_before_minting(client):
    assert client.get("/shares").json() == []


def test_listing_requires_admin(client):
    prev = os.environ.pop("GOA2_REPLAY_API", None)
    try:
        with TestClient(create_app()) as anon:
            assert anon.get("/shares").status_code == 404
    finally:
        if prev is not None:
            os.environ["GOA2_REPLAY_API"] = prev


# --- the load-bearing property -------------------------------------------


def test_every_baked_position_matches_dynamic_reconstruction(client):
    """A baked position must equal what /replays/{id}/state returns at that index."""
    token = _mint(client)
    total = client.get(f"/shared/{token}").json()
    assert len(total["decisions"]) == 3

    for n in range(4):
        baked = client.get(f"/shared/{token}/state?decision={n}").json()
        live = client.get(f"/replays/{FINISHED}/state?decision={n}").json()
        assert _strip_volatile(baked) == _strip_volatile(live), f"mismatch at decision {n}"


def test_shared_meta_matches_replay_meta(client):
    token = _mint(client)
    shared = client.get(f"/shared/{token}").json()
    admin = client.get(f"/replays/{FINISHED}").json()
    assert shared["decisions"] == admin["decisions"]
    assert shared["setup"] == admin["setup"]


# --- public serving -------------------------------------------------------


def test_state_is_served_gzipped(client):
    token = _mint(client)
    # No automatic decompression, so the stored encoding is observable.
    res = client.get(f"/shared/{token}/state?decision=0", headers={"Accept-Encoding": "identity"})
    assert res.status_code == 200
    assert res.headers["content-encoding"] == "gzip"


def test_state_clamps_out_of_range(client):
    token = _mint(client)
    body = client.get(f"/shared/{token}/state?decision=999").json()
    assert body["position"]["decision_index"] == 3
    assert body["position"]["total_decisions"] == 3


def test_state_defaults_to_end_of_game(client):
    token = _mint(client)
    body = client.get(f"/shared/{token}/state").json()
    assert body["position"]["decision_index"] == 3
    assert body["winner"] is not None


def test_round_jump_works_without_the_engine(client):
    token = _mint(client)
    baked = client.get(f"/shared/{token}/state?round=1").json()
    live = client.get(f"/replays/{FINISHED}/state?round=1").json()
    assert baked["position"] == live["position"]


# --- revocation & errors --------------------------------------------------


def test_revoke_makes_the_link_dead(client):
    token = _mint(client)
    assert client.get(f"/shared/{token}").status_code == 200
    assert client.delete(f"/shares/{token}").status_code == 204
    assert client.get(f"/shared/{token}").status_code == 404
    assert client.get(f"/shared/{token}/state?decision=0").status_code == 404


def test_revoke_unknown_token_404(client):
    assert client.delete("/shares/doesnotexist").status_code == 404


def test_unknown_token_404(client):
    assert client.get("/shared/doesnotexist").status_code == 404


@pytest.mark.parametrize("bad", ["..%2F..%2Fetc", "with/slash", "with.dot"])
def test_malformed_tokens_are_rejected(client, bad):
    assert client.get(f"/shared/{bad}").status_code == 404
    assert client.get(f"/shared/{bad}/state?decision=0").status_code == 404


# --- gating ---------------------------------------------------------------


def test_shares_readable_without_admin_but_not_mintable(client):
    """The share router is public; minting and revoking are not."""
    token = _mint(client)
    prev = os.environ.get("GOA2_REPLAY_API")
    os.environ.pop("GOA2_REPLAY_API", None)
    try:
        with TestClient(create_app()) as anon:
            assert anon.get(f"/shared/{token}").status_code == 200
            assert anon.get(f"/shared/{token}/state?decision=0").status_code == 200
            # Admin routers are not mounted at all without a token or the dev flag.
            assert anon.post(f"/replays/{FINISHED}/share").status_code == 404
            assert anon.delete(f"/shares/{token}").status_code == 404
    finally:
        if prev is not None:
            os.environ["GOA2_REPLAY_API"] = prev


# --- retention ------------------------------------------------------------


def test_shared_replay_is_pinned_against_ttl(client):
    _mint(client)
    # TTL of 0 days: everything unpinned is stale.
    removed = cleanup_old_replays(ttl_days=0)
    assert removed == 1  # only the unshared game went
    assert shares.shared_game_ids() == {FINISHED}


def test_revoked_share_releases_the_pin(client):
    token = _mint(client)
    client.delete(f"/shares/{token}")
    removed = cleanup_old_replays(ttl_days=0)
    assert removed == 2
