"""REST endpoints: op schema catalogue + player-scoped history."""

import os

import pytest
from fastapi.testclient import TestClient

from goa2.engine.overrides import OVERRIDE_OPS
from goa2.server.app import create_app


@pytest.fixture
def client(tmp_path):
    os.environ["GOA2_SAVE_DIR"] = str(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        yield c
    os.environ.pop("GOA2_SAVE_DIR", None)


def test_schema_lists_every_registered_op(client):
    resp = client.get("/overrides/schema")
    assert resp.status_code == 200
    ops = {o["name"]: o for o in resp.json()["ops"]}
    # Schema completeness: every registered op appears with a valid arg schema.
    assert set(ops) == set(OVERRIDE_OPS)
    for op in ops.values():
        assert op["family"] in ("patch", "unstick")
        assert op["label"] and op["description"]
        assert isinstance(op["args_schema"], dict)
        assert op["args_schema"].get("type") == "object"


def test_schema_is_static_and_unauthenticated(client):
    # Game-independent; clients fetch once and cache.
    assert client.get("/overrides/schema").json() == client.get("/overrides/schema").json()


# ---------------------------------------------------------------------------
# Player-scoped decision history (Task 9)
# ---------------------------------------------------------------------------


@pytest.fixture
def game_data(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
        },
    )
    return resp.json()


def _token_for(game_data, hero_id):
    for pt in game_data["player_tokens"]:
        if pt["hero_id"] == hero_id:
            return pt["token"]
    raise ValueError(hero_id)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _commit_first_card(client, game_data, hero_id):
    token = _token_for(game_data, hero_id)
    gid = game_data["game_id"]
    view = client.get(f"/games/{gid}", headers=_auth(token)).json()["view"]
    card_id = None
    for team in view["teams"].values():
        for h in team["heroes"]:
            if h["id"] == hero_id:
                card_id = h["hand"][0]["id"]
    assert card_id is not None
    resp = client.post(f"/games/{gid}/cards", json={"card_id": card_id}, headers=_auth(token))
    assert resp.status_code == 200
    return card_id


def test_history_requires_auth(client, game_data):
    gid = game_data["game_id"]
    assert client.get(f"/games/{gid}/overrides/history").status_code == 401


def test_history_masks_opponent_facedown_commit(client, game_data):
    gid = game_data["game_id"]
    arien_card = _commit_first_card(client, game_data, "hero_arien")

    # Wasp must NOT learn the identity of Arien's facedown commit.
    wasp = client.get(
        f"/games/{gid}/overrides/history",
        headers=_auth(_token_for(game_data, "hero_wasp")),
    ).json()
    commit_rows = [d for d in wasp["decisions"] if d["type"] == "commit"]
    assert commit_rows, "commit decision missing from history"
    assert arien_card not in commit_rows[0]["label"]
    assert "a card" in commit_rows[0]["label"]

    # Arien sees their own card (by name, not the anonymous form).
    arien = client.get(
        f"/games/{gid}/overrides/history",
        headers=_auth(_token_for(game_data, "hero_arien")),
    ).json()
    own_rows = [d for d in arien["decisions"] if d["type"] == "commit"]
    assert own_rows[0]["label"] != commit_rows[0]["label"]


def test_history_spectator_gets_fully_masked_form(client, game_data):
    gid = game_data["game_id"]
    arien_card = _commit_first_card(client, game_data, "hero_arien")
    spec = client.get(
        f"/games/{gid}/overrides/history",
        headers=_auth(game_data["spectator_token"]),
    ).json()
    for row in spec["decisions"]:
        assert arien_card not in row["label"]


def test_history_marks_superseded_segments(client, game_data):
    """Records behind a rewind carry superseded=True."""
    gid = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")
    # Fabricate the scenario at the replay-log level: one decision + a rewind.
    from goa2.server.replay import create_replay_recorder

    rec = create_replay_recorder(gid)
    rec.record_pass("hero_arien", 1, 1)
    rec.record_override(
        {"type": "ov_rewind", "r": 1, "t": 1, "hero": "hero_arien", "to": 0, "voters": []}
    )
    hist = client.get(f"/games/{gid}/overrides/history", headers=_auth(token)).json()
    pass_row = next(d for d in hist["decisions"] if d["type"] == "pass")
    rewind_row = next(d for d in hist["decisions"] if d["type"] == "ov_rewind")
    assert pass_row["superseded"] is True
    assert rewind_row["superseded"] is False
    assert "rew" in rewind_row["label"].lower()
    assert hist["total"] == len(hist["decisions"])
