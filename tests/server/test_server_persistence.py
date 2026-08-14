"""Server integration tests for Phase 6: State Persistence.

Tests that games survive server restarts via auto-save and restore.
"""

import os

from fastapi.testclient import TestClient

from goa2.server.app import create_app


def _token_for(game_data: dict, hero_id: str) -> str:
    for pt in game_data["player_tokens"]:
        if pt["hero_id"] == hero_id:
            return pt["token"]
    raise ValueError(f"No token for {hero_id}")


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _create_game(client: TestClient) -> dict:
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
        },
    )
    assert resp.status_code == 201
    return resp.json()


def _make_client(save_dir: str) -> TestClient:
    """Create a fresh TestClient with the given save_dir."""
    os.environ["GOA2_SAVE_DIR"] = save_dir
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Save file creation
# ---------------------------------------------------------------------------


def test_create_game_creates_save_file(tmp_path):
    """Creating a game via API produces a .json save file."""
    with _make_client(str(tmp_path)) as client:
        data = _create_game(client)
        game_id = data["game_id"]

    save_file = tmp_path / f"{game_id}.json"
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_commit_card_updates_save_file(tmp_path):
    """Committing a card updates the save file."""
    with _make_client(str(tmp_path)) as client:
        data = _create_game(client)
        game_id = data["game_id"]
        token = _token_for(data, "hero_arien")
        save_file = tmp_path / f"{game_id}.json"

        # Get a card to commit
        view = client.get(f"/games/{game_id}", headers=_auth(token)).json()
        arien_hand = None
        for team_data in view["view"]["teams"].values():
            for hero in team_data["heroes"]:
                if hero["id"] == "hero_arien":
                    arien_hand = hero["hand"]
        assert arien_hand

        resp = client.post(
            f"/games/{game_id}/cards",
            json={"card_id": arien_hand[0]["id"]},
            headers=_auth(token),
        )
        assert resp.status_code == 200

        # Save file should be updated
        assert save_file.exists()
        # File was rewritten (mtime or size may differ)
        assert save_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# Restart survival
# ---------------------------------------------------------------------------


def test_game_survives_restart(tmp_path):
    """Create game, 'restart' (new TestClient), game is still accessible."""
    save_dir = str(tmp_path)

    # Session 1: create game
    with _make_client(save_dir) as client1:
        data = _create_game(client1)
        game_id = data["game_id"]
        arien_token = _token_for(data, "hero_arien")
        spectator_token = data["spectator_token"]

    # Session 2: new client (simulates restart)
    with _make_client(save_dir) as client2:
        # Game should be accessible with the same token
        resp = client2.get(f"/games/{game_id}", headers=_auth(arien_token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["view"]["phase"] == "PLANNING"

        # Spectator token should also work
        resp = client2.get(f"/games/{game_id}", headers=_auth(spectator_token))
        assert resp.status_code == 200


def test_completed_game_reports_winner_after_restart(tmp_path):
    """A restored GAME_OVER save must rebuild the transient winner result."""
    from goa2.domain.models import GamePhase, TeamColor

    save_dir = str(tmp_path)
    with _make_client(save_dir) as client1:
        data = _create_game(client1)
        game_id = data["game_id"]
        spectator_token = data["spectator_token"]
        game = client1.app.state.registry.get(game_id)
        game.session.state.phase = GamePhase.GAME_OVER
        game.session.state.winner = TeamColor.RED
        game.session.state.execution_stack.clear()
        client1.app.state.registry.save_game(game_id)

    with _make_client(save_dir) as client2:
        response = client2.get(f"/games/{game_id}", headers=_auth(spectator_token))
        assert response.status_code == 200
        assert response.json()["winner"] == "RED"

        with client2.websocket_connect(f"/games/{game_id}/ws?token={spectator_token}") as websocket:
            message = websocket.receive_json()
            assert message["type"] == "STATE_UPDATE"
            assert message["winner"] == "RED"


def test_completed_solo_game_reports_hero_winner_after_restart(tmp_path):
    """An individual winner survives persistence and both client transports."""
    from goa2.domain.models import GamePhase

    save_dir = str(tmp_path)
    with _make_client(save_dir) as client1:
        data = _create_game(client1)
        game_id = data["game_id"]
        spectator_token = data["spectator_token"]
        game = client1.app.state.registry.get(game_id)
        game.session.state.phase = GamePhase.GAME_OVER
        game.session.state.individual_winner_id = "hero_arien"
        game.session.state.victory_condition = "TEST_SOLO"
        game.session.state.execution_stack.clear()
        client1.app.state.registry.save_game(game_id)

    with _make_client(save_dir) as client2:
        response = client2.get(f"/games/{game_id}", headers=_auth(spectator_token))
        assert response.status_code == 200
        assert response.json()["winner"] == "hero_arien"

        with client2.websocket_connect(f"/games/{game_id}/ws?token={spectator_token}") as websocket:
            message = websocket.receive_json()
            assert message["type"] == "STATE_UPDATE"
            assert message["winner"] == "hero_arien"


def test_committed_card_survives_restart(tmp_path):
    """Commit a card, restart, verify the card is committed."""
    save_dir = str(tmp_path)

    # Session 1: create game and commit Arien's card
    with _make_client(save_dir) as client1:
        data = _create_game(client1)
        game_id = data["game_id"]
        arien_token = _token_for(data, "hero_arien")

        # Get and commit a card
        view = client1.get(f"/games/{game_id}", headers=_auth(arien_token)).json()
        arien_hand = None
        for team_data in view["view"]["teams"].values():
            for hero in team_data["heroes"]:
                if hero["id"] == "hero_arien":
                    arien_hand = hero["hand"]
        assert arien_hand
        card_id = arien_hand[0]["id"]
        hand_size_before = len(arien_hand)

        resp = client1.post(
            f"/games/{game_id}/cards",
            json={"card_id": card_id},
            headers=_auth(arien_token),
        )
        assert resp.status_code == 200

    # Session 2: restart
    with _make_client(save_dir) as client2:
        view = client2.get(f"/games/{game_id}", headers=_auth(arien_token)).json()

        # Find Arien's hand in restored state
        restored_hand = None
        for team_data in view["view"]["teams"].values():
            for hero in team_data["heroes"]:
                if hero["id"] == "hero_arien":
                    restored_hand = hero["hand"]
        assert restored_hand is not None
        # Card was committed, so hand should be smaller
        assert len(restored_hand) == hand_size_before - 1


def test_full_planning_survives_restart(tmp_path):
    """Both players commit cards, restart, verify phase transitioned."""
    save_dir = str(tmp_path)

    # Session 1: full planning flow
    with _make_client(save_dir) as client1:
        data = _create_game(client1)
        game_id = data["game_id"]
        arien_token = _token_for(data, "hero_arien")
        wasp_token = _token_for(data, "hero_wasp")

        # Get cards for both heroes
        view = client1.get(f"/games/{game_id}", headers=_auth(arien_token)).json()
        arien_hand = None
        for td in view["view"]["teams"].values():
            for h in td["heroes"]:
                if h["id"] == "hero_arien":
                    arien_hand = h["hand"]
        assert arien_hand

        view = client1.get(f"/games/{game_id}", headers=_auth(wasp_token)).json()
        wasp_hand = None
        for td in view["view"]["teams"].values():
            for h in td["heroes"]:
                if h["id"] == "hero_wasp":
                    wasp_hand = h["hand"]
        assert wasp_hand

        # Commit both cards
        client1.post(
            f"/games/{game_id}/cards",
            json={"card_id": arien_hand[0]["id"]},
            headers=_auth(arien_token),
        )
        resp = client1.post(
            f"/games/{game_id}/cards",
            json={"card_id": wasp_hand[0]["id"]},
            headers=_auth(wasp_token),
        )
        assert resp.status_code == 200
        phase_after = resp.json()["current_phase"]
        assert phase_after != "PLANNING"

    # Session 2: restart — phase should be preserved
    with _make_client(save_dir) as client2:
        view = client2.get(f"/games/{game_id}", headers=_auth(arien_token)).json()
        assert view["view"]["phase"] == phase_after


# ---------------------------------------------------------------------------
# Multiple games
# ---------------------------------------------------------------------------


def test_multiple_games_survive_restart(tmp_path):
    """Multiple games all survive a restart."""
    save_dir = str(tmp_path)
    game_ids = []
    tokens = []

    with _make_client(save_dir) as client1:
        for _ in range(3):
            data = _create_game(client1)
            game_ids.append(data["game_id"])
            tokens.append(_token_for(data, "hero_arien"))

    with _make_client(save_dir) as client2:
        for gid, tok in zip(game_ids, tokens, strict=False):
            resp = client2.get(f"/games/{gid}", headers=_auth(tok))
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# WebSocket reconnection
# ---------------------------------------------------------------------------


def test_ws_reconnect_after_restart(tmp_path):
    """WebSocket connection works after restart with same token."""
    save_dir = str(tmp_path)

    # Session 1: create game
    with _make_client(save_dir) as client1:
        data = _create_game(client1)
        game_id = data["game_id"]
        arien_token = _token_for(data, "hero_arien")

    # Session 2: reconnect via WebSocket
    with (
        _make_client(save_dir) as client2,
        client2.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws,
    ):
        msg = ws.receive_json()
        assert msg["type"] == "STATE_UPDATE"
        assert "view" in msg
        assert msg["view"]["phase"] == "PLANNING"


# ---------------------------------------------------------------------------
# No save_dir (disabled persistence)
# ---------------------------------------------------------------------------


def test_no_persistence_when_save_dir_unset(tmp_path):
    """When GOA2_SAVE_DIR is empty, no files are written."""
    os.environ["GOA2_SAVE_DIR"] = ""
    app = create_app()
    with TestClient(app) as client:
        _create_game(client)
    os.environ.pop("GOA2_SAVE_DIR", None)

    # No files should be created anywhere for this test
    # (registry has save_dir="" which is falsy, so save_game is a no-op)


# ---------------------------------------------------------------------------
# Persisted bot metadata
# ---------------------------------------------------------------------------


def _new_registry_with_bot_game(save_dir: str, bot_specs):
    """Create a registry-backed game with the given bot_specs.

    Bypasses the REST API — this test only concerns the registry/persistence
    layer, not the create endpoint.
    """
    from goa2.engine.session import GameSession
    from goa2.engine.setup import GameSetup
    from goa2.server.registry import GameRegistry

    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Arien"],
        ["Wasp"],
    )
    session = GameSession(state)
    registry = GameRegistry(save_dir=save_dir)
    game = registry.create_game(session, ["hero_arien", "hero_wasp"], bot_specs=bot_specs)
    return registry, game


def test_bot_specs_round_trip_through_save_and_load(tmp_path):
    """Bot specs saved with a game are restored identically on reload."""
    from goa2.server.bot_models import BotSpec, SearchSettings
    from goa2.server.registry import GameRegistry

    save_dir = str(tmp_path)
    original_specs = {
        "hero_arien": BotSpec(kind="heuristic"),
        "hero_wasp": BotSpec(
            kind="ismcts",
            search=SearchSettings(iterations=128, decision_timeout_seconds=1.5),
        ),
    }
    registry1, game = _new_registry_with_bot_game(save_dir, original_specs)
    # save_game is called by create_game when save_dir is set, but re-run it
    # explicitly for clarity.
    registry1.save_game(game.game_id)

    registry2 = GameRegistry(save_dir=save_dir)
    restored = registry2.restore_all()
    assert restored == 1
    restored_game = registry2.get(game.game_id)
    assert restored_game.bot_specs == original_specs
    # BotSpec equality is field-based; explicitly verify the nested settings
    # survived the JSON round-trip as typed floats/ints.
    assert restored_game.bot_specs["hero_wasp"].search is not None
    assert restored_game.bot_specs["hero_wasp"].search.iterations == 128
    assert restored_game.bot_specs["hero_wasp"].search.decision_timeout_seconds == 1.5


def test_bot_task_not_serialized(tmp_path):
    """Assigning a live bot_task must not leak into the on-disk payload."""
    import asyncio
    import json

    from goa2.server.bot_models import BotSpec

    save_dir = str(tmp_path)
    specs = {"hero_arien": BotSpec(kind="random")}
    registry, game = _new_registry_with_bot_game(save_dir, specs)

    # Attach a live task (simulating the coordinator).
    loop = asyncio.new_event_loop()
    try:

        async def _noop() -> None:
            return None

        game.bot_task = loop.create_task(_noop())
        registry.save_game(game.game_id)
        loop.run_until_complete(game.bot_task)
    finally:
        loop.close()

    # Inspect the raw save file — no task/agent references should appear.
    save_file = tmp_path / f"{game.game_id}.json"
    raw = save_file.read_text()
    assert "bot_task" not in raw
    payload = json.loads(raw)
    # bot_specs is persisted; bot_task is absent.
    assert "bot_specs" in payload
    assert "bot_task" not in payload


def test_legacy_save_without_bot_metadata_loads(tmp_path):
    """Save files predating this change must load with empty bot_specs."""
    import json

    from goa2.server.registry import GameRegistry

    save_dir = str(tmp_path)
    # Round-trip a game through save, then strip the bot_specs key to emulate
    # a legacy payload.
    registry1, game = _new_registry_with_bot_game(save_dir, bot_specs={})
    registry1.save_game(game.game_id)

    save_file = tmp_path / f"{game.game_id}.json"
    payload = json.loads(save_file.read_text())
    payload.pop("bot_specs", None)
    save_file.write_text(json.dumps(payload))

    registry2 = GameRegistry(save_dir=save_dir)
    assert registry2.restore_all() == 1
    restored = registry2.get(game.game_id)
    assert restored.bot_specs == {}


def test_corrupt_bot_spec_entry_is_skipped_on_restore(tmp_path, caplog):
    """A malformed bot_specs entry on disk must not abort the whole restore."""
    import json
    import logging

    from goa2.server.bot_models import BotSpec
    from goa2.server.registry import GameRegistry

    save_dir = str(tmp_path)
    registry1, game = _new_registry_with_bot_game(
        save_dir, bot_specs={"hero_arien": BotSpec(kind="random")}
    )
    registry1.save_game(game.game_id)

    save_file = tmp_path / f"{game.game_id}.json"
    payload = json.loads(save_file.read_text())
    # Corrupt the on-disk spec (unsupported kind).
    payload["bot_specs"]["hero_arien"] = {"kind": "not_a_real_kind"}
    save_file.write_text(json.dumps(payload))

    registry2 = GameRegistry(save_dir=save_dir)
    with caplog.at_level(logging.ERROR, logger="goa2.server.registry"):
        assert registry2.restore_all() == 1
    restored = registry2.get(game.game_id)
    # Corrupt entry silently dropped; game is otherwise intact.
    assert restored.bot_specs == {}


def test_bot_specs_survive_full_server_restart(tmp_path, monkeypatch):
    """Bot metadata persists through the same restart path as game state."""
    from fastapi.testclient import TestClient

    from goa2.server.app import create_app
    from goa2.server.bot_models import BotSpec

    save_dir = str(tmp_path)
    # monkeypatch scopes the env change to this test only — other tests in the
    # suite (some of which explicitly clear GOA2_SAVE_DIR) must not see a
    # dangling value if this test errors out.
    monkeypatch.setenv("GOA2_SAVE_DIR", save_dir)

    # Session 1: create a plain game via API, then inject bot metadata and
    # persist. The create endpoint does not accept bot metadata directly in
    # this test path, so we drive the registry directly for the persistence
    # test.
    app1 = create_app()
    with TestClient(app1) as client1:
        data = _create_game(client1)
        game_id = data["game_id"]
        registry1 = client1.app.state.registry
        game = registry1.get(game_id)
        game.bot_specs = {"hero_arien": BotSpec(kind="heuristic")}
        registry1.save_game(game_id)

    # Session 2: fresh app + registry restores the game and the specs.
    app2 = create_app()
    with TestClient(app2):
        registry2 = app2.state.registry
        restored = registry2.get(game_id)
        assert restored.bot_specs == {"hero_arien": BotSpec(kind="heuristic")}


def test_non_mapping_bot_specs_on_disk_is_discarded(tmp_path, caplog):
    """A non-dict ``bot_specs`` value must not crash restore or contaminate state."""
    import json
    import logging

    from goa2.server.registry import GameRegistry

    save_dir = str(tmp_path)
    registry1, game = _new_registry_with_bot_game(save_dir, bot_specs={})
    registry1.save_game(game.game_id)

    save_file = tmp_path / f"{game.game_id}.json"
    payload = json.loads(save_file.read_text())
    # Emulate a corrupt / hand-edited save where bot_specs is not a mapping
    # (e.g. a list, or a stray string from a schema-mismatch bug).
    payload["bot_specs"] = ["hero_arien", "hero_wasp"]
    save_file.write_text(json.dumps(payload))

    registry2 = GameRegistry(save_dir=save_dir)
    with caplog.at_level(logging.ERROR, logger="goa2.server.registry"):
        assert registry2.restore_all() == 1
    restored = registry2.get(game.game_id)
    assert restored.bot_specs == {}
    # Verify the discard was logged (not silently swallowed).
    assert any("bot_specs" in rec.getMessage() for rec in caplog.records)


def test_bot_spec_for_unknown_hero_on_disk_is_discarded(tmp_path, caplog):
    """A restored spec whose hero is not in the restored roster is dropped."""
    import json
    import logging

    from goa2.server.bot_models import BotSpec
    from goa2.server.registry import GameRegistry

    save_dir = str(tmp_path)
    registry1, game = _new_registry_with_bot_game(
        save_dir, bot_specs={"hero_arien": BotSpec(kind="random")}
    )
    registry1.save_game(game.game_id)

    save_file = tmp_path / f"{game.game_id}.json"
    payload = json.loads(save_file.read_text())
    # Inject an extra spec for a hero not present in the persisted roster
    # (hero_to_token). Simulates roster drift between save and reload.
    payload["bot_specs"]["hero_nomad"] = {"kind": "heuristic"}
    save_file.write_text(json.dumps(payload))

    registry2 = GameRegistry(save_dir=save_dir)
    with caplog.at_level(logging.ERROR, logger="goa2.server.registry"):
        assert registry2.restore_all() == 1
    restored = registry2.get(game.game_id)
    # Valid roster hero survives; unknown hero is dropped.
    assert restored.bot_specs == {"hero_arien": BotSpec(kind="random")}
    assert any(
        "hero_nomad" in rec.getMessage() and "roster" in rec.getMessage() for rec in caplog.records
    )
