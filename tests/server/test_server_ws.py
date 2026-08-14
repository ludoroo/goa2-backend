"""WebSocket integration tests."""

import os

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from goa2.server.app import create_app


@pytest.fixture
def client(tmp_path):
    os.environ["GOA2_SAVE_DIR"] = str(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        yield c
    os.environ.pop("GOA2_SAVE_DIR", None)


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


def _token_for(game_data: dict, hero_id: str) -> str:
    for pt in game_data["player_tokens"]:
        if pt["hero_id"] == hero_id:
            return pt["token"]
    raise ValueError(f"No token for {hero_id}")


# ---- Connection tests ----


def test_ws_connect_and_receive_initial_state(client, game_data):
    token = _token_for(game_data, "hero_arien")
    game_id = game_data["game_id"]
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "STATE_UPDATE"
        assert msg["view"]["phase"] == "PLANNING"


def test_ws_connect_spectator(client, game_data):
    game_id = game_data["game_id"]
    token = game_data["spectator_token"]
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "STATE_UPDATE"


def test_ws_initial_state_only_includes_input_for_responder(client, game_data):
    from goa2.domain.input import InputRequestType, create_input_request
    from goa2.engine.session import SessionResult, SessionResultType

    game_id = game_data["game_id"]
    game = client.app.state.registry.get(game_id)
    game.last_result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        current_phase=game.session.current_phase,
        input_request=create_input_request(
            InputRequestType.SELECT_CARD_OR_PASS,
            player_id="hero_wasp",
            prompt="Choose a defense",
            options=[{"id": "magnetic_dagger", "text": "Magnetic Dagger"}],
        ),
    )

    wasp_token = _token_for(game_data, "hero_wasp")
    arien_token = _token_for(game_data, "hero_arien")
    spectator_token = game_data["spectator_token"]

    with client.websocket_connect(f"/games/{game_id}/ws?token={wasp_token}") as ws:
        msg = ws.receive_json()
        assert msg["input_request"]["player_id"] == "hero_wasp"
        assert msg["input_request"]["options"][0]["id"] == "magnetic_dagger"

    for token in (arien_token, spectator_token):
        with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
            assert "input_request" not in ws.receive_json()


def test_ws_invalid_token(client, game_data):
    game_id = game_data["game_id"]
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(f"/games/{game_id}/ws?token=badtoken") as ws,
    ):
        ws.receive_json()


def test_ws_wrong_game(client, game_data):
    token = _token_for(game_data, "hero_arien")
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(f"/games/wrong_game/ws?token={token}") as ws,
    ):
        ws.receive_json()


def test_new_connection_supersedes_old_connection(client, game_data):
    """A reconnect owns the token and continues receiving broadcasts."""
    token = _token_for(game_data, "hero_arien")
    game_id = game_data["game_id"]

    view = client.get(
        f"/games/{game_id}",
        headers={"Authorization": f"Bearer {token}"},
    ).json()
    card_id = next(
        hero["hand"][0]["id"]
        for team in view["view"]["teams"].values()
        for hero in team["heroes"]
        if hero["id"] == "hero_arien"
    )

    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as old_ws:
        old_ws.receive_json()

        with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as new_ws:
            new_ws.receive_json()

            with pytest.raises(WebSocketDisconnect) as exc_info:
                old_ws.receive_json()
            assert exc_info.value.code == 4002

            # The old handler's disconnect cleanup must not unregister the new
            # socket. A mutation therefore yields both its direct result and
            # the authoritative state broadcast on the replacement socket.
            new_ws.send_json({"type": "COMMIT_CARD", "card_id": card_id})
            assert new_ws.receive_json()["type"] == "ACTION_RESULT"
            assert new_ws.receive_json()["type"] == "STATE_UPDATE"


def test_shared_spectator_token_supports_multiple_live_connections(client, game_data):
    game_id = game_data["game_id"]
    player_token = _token_for(game_data, "hero_arien")
    spectator_token = game_data["spectator_token"]

    with (
        client.websocket_connect(f"/games/{game_id}/ws?token={player_token}") as player_ws,
        client.websocket_connect(f"/games/{game_id}/ws?token={spectator_token}") as spectator_one,
        client.websocket_connect(f"/games/{game_id}/ws?token={spectator_token}") as spectator_two,
    ):
        player_view = player_ws.receive_json()
        spectator_one.receive_json()
        spectator_two.receive_json()
        arien = next(
            hero
            for team in player_view["view"]["teams"].values()
            for hero in team["heroes"]
            if hero["id"] == "hero_arien"
        )

        player_ws.send_json({"type": "COMMIT_CARD", "card_id": arien["hand"][0]["id"]})
        assert player_ws.receive_json()["type"] == "ACTION_RESULT"
        assert player_ws.receive_json()["type"] == "STATE_UPDATE"
        assert spectator_one.receive_json()["type"] == "STATE_UPDATE"
        assert spectator_two.receive_json()["type"] == "STATE_UPDATE"


# ---- GET_VIEW ----


def test_ws_get_view(client, game_data):
    token = _token_for(game_data, "hero_arien")
    game_id = game_data["game_id"]
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()  # initial state
        ws.send_json({"type": "GET_VIEW"})
        msg = ws.receive_json()
        assert msg["type"] == "STATE_UPDATE"


# ---- Ephemeral table pings ----


def test_ws_ping_broadcasts_authenticated_identity_to_players_and_spectators(client, game_data):
    game_id = game_data["game_id"]
    arien_token = _token_for(game_data, "hero_arien")
    wasp_token = _token_for(game_data, "hero_wasp")
    spectator_token = game_data["spectator_token"]

    with (
        client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws_a,
        client.websocket_connect(f"/games/{game_id}/ws?token={wasp_token}") as ws_w,
        client.websocket_connect(f"/games/{game_id}/ws?token={spectator_token}") as ws_s,
    ):
        initial = ws_a.receive_json()
        ws_w.receive_json()
        ws_s.receive_json()
        ping_hex = next(iter(initial["view"]["board"]["tiles"].values()))["hex"]

        # Clients cannot spoof who pinged; identity comes from the token.
        ws_a.send_json(
            {
                "type": "PING",
                "hero_id": "hero_wasp",
                "target": {"kind": "HEX", "hex": ping_hex},
            }
        )

        messages = [ws_a.receive_json(), ws_w.receive_json(), ws_s.receive_json()]
        assert all(message["type"] == "PING" for message in messages)
        assert all(message["hero_id"] == "hero_arien" for message in messages)
        assert all(message["target"] == {"kind": "HEX", "hex": ping_hex} for message in messages)
        assert len({message["ping_id"] for message in messages}) == 1


def test_ws_ping_card_uses_public_table_location_without_card_identity(client, game_data):
    game_id = game_data["game_id"]
    arien_token = _token_for(game_data, "hero_arien")

    with client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws:
        initial = ws.receive_json()
        arien = next(
            hero
            for team in initial["view"]["teams"].values()
            for hero in team["heroes"]
            if hero["id"] == "hero_arien"
        )
        ws.send_json({"type": "COMMIT_CARD", "card_id": arien["hand"][0]["id"]})
        assert ws.receive_json()["type"] == "ACTION_RESULT"
        assert ws.receive_json()["type"] == "STATE_UPDATE"

        ws.send_json(
            {
                "type": "PING",
                "target": {
                    "kind": "CARD",
                    "hero_id": "hero_arien",
                    "zone": "CURRENT",
                    # Even if a hostile client supplies it, the server strips it.
                    "card_id": arien["hand"][0]["id"],
                },
            }
        )
        message = ws.receive_json()

        assert message["type"] == "PING"
        assert message["target"] == {
            "kind": "CARD",
            "hero_id": "hero_arien",
            "zone": "CURRENT",
        }


@pytest.mark.parametrize(
    "target, detail",
    [
        ({"kind": "HEX", "hex": {"q": 999, "r": -999, "s": 0}}, "not on the board"),
        (
            {"kind": "CARD", "hero_id": "hero_arien", "zone": "DISCARD", "index": 0},
            "not currently on the table",
        ),
    ],
)
def test_ws_ping_rejects_targets_that_are_not_on_the_table(client, game_data, target, detail):
    token = _token_for(game_data, "hero_arien")
    game_id = game_data["game_id"]
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()
        ws.send_json({"type": "PING", "target": target})
        message = ws.receive_json()
        assert message["type"] == "ERROR"
        assert detail in message["detail"]


# ---- Invalid JSON ----


def test_ws_invalid_json(client, game_data):
    token = _token_for(game_data, "hero_arien")
    game_id = game_data["game_id"]
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()  # initial state
        ws.send_text("not json")
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"
        assert "Invalid JSON" in msg["detail"]


# ---- Unknown message type ----


def test_ws_unknown_type(client, game_data):
    token = _token_for(game_data, "hero_arien")
    game_id = game_data["game_id"]
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()  # initial state
        ws.send_json({"type": "FOOBAR"})
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"
        assert "Unknown" in msg["detail"]


# ---- Spectator restrictions ----


def test_ws_spectator_cannot_commit_card(client, game_data):
    game_id = game_data["game_id"]
    token = game_data["spectator_token"]
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()
        ws.send_json({"type": "COMMIT_CARD", "card_id": "x"})
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"
        assert "Spectator" in msg["detail"]


def test_ws_spectator_can_get_view(client, game_data):
    game_id = game_data["game_id"]
    token = game_data["spectator_token"]
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()
        ws.send_json({"type": "GET_VIEW"})
        msg = ws.receive_json()
        assert msg["type"] == "STATE_UPDATE"


# ---- COMMIT_CARD via WS ----


def test_ws_commit_card(client, game_data):
    game_id = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")

    # Get a card ID from REST
    view = client.get(f"/games/{game_id}", headers={"Authorization": f"Bearer {token}"}).json()
    arien_hand = None
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_arien":
                arien_hand = hero["hand"]
    assert arien_hand

    card_id = arien_hand[0]["id"]

    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()  # initial state
        ws.send_json({"type": "COMMIT_CARD", "card_id": card_id})
        msg = ws.receive_json()
        assert msg["type"] == "ACTION_RESULT"
        assert msg["result_type"] in (
            "ACTION_COMPLETE",
            "PHASE_CHANGED",
            "INPUT_NEEDED",
        )


# ---- Full WS flow ----


def test_ws_full_planning_flow(client, game_data):
    """Both players commit cards via WS."""
    game_id = game_data["game_id"]
    arien_token = _token_for(game_data, "hero_arien")
    wasp_token = _token_for(game_data, "hero_wasp")

    # Get card IDs via REST
    arien_view = client.get(
        f"/games/{game_id}", headers={"Authorization": f"Bearer {arien_token}"}
    ).json()
    wasp_view = client.get(
        f"/games/{game_id}", headers={"Authorization": f"Bearer {wasp_token}"}
    ).json()

    arien_card = wasp_card = None
    for team_data in arien_view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_arien":
                arien_card = hero["hand"][0]["id"]
    for team_data in wasp_view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_wasp":
                wasp_card = hero["hand"][0]["id"]

    assert arien_card and wasp_card

    # Arien commits
    with client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws_a:
        ws_a.receive_json()  # initial
        ws_a.send_json({"type": "COMMIT_CARD", "card_id": arien_card})
        msg = ws_a.receive_json()
        assert msg["type"] == "ACTION_RESULT"
        assert msg["result_type"] == "ACTION_COMPLETE"

    # Wasp commits -> phase transition
    with client.websocket_connect(f"/games/{game_id}/ws?token={wasp_token}") as ws_w:
        ws_w.receive_json()  # initial
        ws_w.send_json({"type": "COMMIT_CARD", "card_id": wasp_card})
        msg = ws_w.receive_json()
        assert msg["type"] == "ACTION_RESULT"
        assert msg["current_phase"] != "PLANNING"


# ---- Cheats WebSocket tests ----


def test_ws_cheats_gold_success(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")

    with client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws:
        ws.receive_json()  # initial state

        # Give gold to Arien
        ws.send_json({"type": "CHEATS_GOLD", "hero_id": "hero_arien", "amount": 5})
        msg = ws.receive_json()
        assert msg["type"] == "ACTION_RESULT"
        assert msg["result_type"] == "ACTION_COMPLETE"
        assert msg["events"]
        assert msg["events"][0]["event_type"] == "GOLD_GAINED"
        assert msg["events"][0]["metadata"]["amount"] == 5
        assert msg["events"][0]["metadata"]["reason"] == "cheat"


def test_ws_cheats_gold_disabled(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": False,
        },
    )
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")

    with client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws:
        ws.receive_json()  # initial state

        # Try to give gold when cheats are disabled
        ws.send_json({"type": "CHEATS_GOLD", "hero_id": "hero_arien", "amount": 5})
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"
        assert "Cheats are not enabled" in msg["detail"]


def test_ws_cheats_gold_invalid_hero(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")

    with client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws:
        ws.receive_json()  # initial state

        # Try to give gold to non-existent hero
        ws.send_json({"type": "CHEATS_GOLD", "hero_id": "hero_does_not_exist", "amount": 5})
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"
        assert "not found" in msg["detail"]


def test_ws_cheats_gold_negative_amount(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")

    with client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws:
        ws.receive_json()  # initial state

        # Try to give negative gold
        ws.send_json({"type": "CHEATS_GOLD", "hero_id": "hero_arien", "amount": -5})
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"
        assert "Amount must be a positive integer" in msg["detail"]


def test_ws_cheats_gold_broadcasts_state(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")
    wasp_token = _token_for(data, "hero_wasp")

    # Connect both players
    with (
        client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws_a,
        client.websocket_connect(f"/games/{game_id}/ws?token={wasp_token}") as ws_w,
    ):
        initial_a = ws_a.receive_json()  # initial Arien
        ws_w.receive_json()  # initial Wasp
        # Initial connect state carries no events (nothing to animate yet).
        assert "events" not in initial_a

        # Arien gives gold
        ws_a.send_json({"type": "CHEATS_GOLD", "hero_id": "hero_arien", "amount": 5})

        # Arien gets ACTION_RESULT
        msg_a1 = ws_a.receive_json()
        assert msg_a1["type"] == "ACTION_RESULT"
        assert msg_a1["events"][0]["event_type"] == "GOLD_GAINED"

        # Both get STATE_UPDATE broadcast
        msg_a2 = ws_a.receive_json()
        assert msg_a2["type"] == "STATE_UPDATE"

        msg_w = ws_w.receive_json()
        assert msg_w["type"] == "STATE_UPDATE"

        # The broadcast carries the action's events so the non-acting player
        # (Wasp) can animate too, not just the actor.
        assert msg_w["events"][0]["event_type"] == "GOLD_GAINED"
        assert msg_a2["events"][0]["event_type"] == "GOLD_GAINED"

        # Verify gold was updated in both views
        arien_gold = None
        for team_data in msg_a2["view"]["teams"].values():
            for hero in team_data["heroes"]:
                if hero["id"] == "hero_arien":
                    arien_gold = hero["gold"]
        assert arien_gold == 5

        wasp_view_arien_gold = None
        for team_data in msg_w["view"]["teams"].values():
            for hero in team_data["heroes"]:
                if hero["id"] == "hero_arien":
                    wasp_view_arien_gold = hero["gold"]
        assert wasp_view_arien_gold == 5


def test_ws_broadcast_names_the_awaited_player_without_leaking_their_options(client, game_data):
    import asyncio

    from goa2.domain.input import InputRequestType, create_input_request
    from goa2.engine.session import SessionResult, SessionResultType
    from goa2.server.ws import broadcast

    class RecordingSocket:
        def __init__(self):
            self.message = None

        async def send_json(self, data):
            self.message = data

    game = client.app.state.registry.get(game_data["game_id"])
    game.last_result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        current_phase=game.session.state.phase,
        input_request=create_input_request(
            InputRequestType.SELECT_CARD_OR_PASS,
            player_id="hero_wasp",
            prompt="Choose a defense",
            options=[{"id": "magnetic_dagger", "text": "Magnetic Dagger"}],
        ),
    )

    sockets = {
        _token_for(game_data, "hero_arien"): RecordingSocket(),
        _token_for(game_data, "hero_wasp"): RecordingSocket(),
    }
    spectator_socket = RecordingSocket()
    game.ws_connections = sockets
    game.spectator_ws_connections = {id(spectator_socket): spectator_socket}

    asyncio.run(broadcast(game, client.app.state.registry))

    attacker_msg = sockets[_token_for(game_data, "hero_arien")].message
    defender_msg = sockets[_token_for(game_data, "hero_wasp")].message

    assert attacker_msg["awaiting_input"] == ["hero_wasp"]
    assert "input_request" not in attacker_msg
    assert spectator_socket.message["awaiting_input"] == ["hero_wasp"]
    assert defender_msg["awaiting_input"] == ["hero_wasp"]
    assert defender_msg["input_request"]["options"][0]["id"] == "magnetic_dagger"


def test_ws_broadcast_projects_hidden_mine_event_per_connection(client, game_data):
    import asyncio

    from goa2.domain.events import GameEvent, GameEventType
    from goa2.domain.models import TokenType
    from goa2.server.ws import broadcast

    class RecordingSocket:
        def __init__(self):
            self.message = None

        async def send_json(self, data):
            self.message = data

    game = client.app.state.registry.get(game_data["game_id"])
    state = game.session.state
    mine = state.token_pool[TokenType.MINE_BLAST][0]
    mine.owner_id = "hero_arien"
    destination = next(hex_ for hex_, tile in state.board.tiles.items() if not tile.is_occupied)
    state.place_entity(mine.id, destination)
    event = GameEvent(
        event_type=GameEventType.TOKEN_PLACED,
        actor_id="hero_arien",
        target_id=mine.id,
        metadata={"token_type": TokenType.MINE_BLAST.value},
    ).model_dump()

    sockets = {
        _token_for(game_data, "hero_arien"): RecordingSocket(),
        _token_for(game_data, "hero_wasp"): RecordingSocket(),
    }
    spectator_socket = RecordingSocket()
    game.ws_connections = sockets
    game.spectator_ws_connections = {id(spectator_socket): spectator_socket}

    asyncio.run(broadcast(game, client.app.state.registry, events=[event]))

    arien_event = sockets[_token_for(game_data, "hero_arien")].message["events"][0]
    wasp_event = sockets[_token_for(game_data, "hero_wasp")].message["events"][0]
    spectator_event = spectator_socket.message["events"][0]
    assert arien_event["metadata"]["token_type"] == "mine_blast"
    assert wasp_event["metadata"]["token_type"] == "mine"
    assert spectator_event["metadata"]["token_type"] == "mine"


def test_ws_broadcast_captures_one_immutable_state_snapshot(client, game_data):
    import asyncio

    from goa2.server.ws import broadcast

    class BlockingSocket:
        def __init__(self, entered, release):
            self.entered = entered
            self.release = release
            self.message = None

        async def send_json(self, data):
            self.message = data
            self.entered.set()
            await self.release.wait()

    class RecordingSocket:
        def __init__(self):
            self.message = None

        async def send_json(self, data):
            self.message = data

    game = client.app.state.registry.get(game_data["game_id"])
    state = game.session.state
    original_turn = state.turn

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        first = BlockingSocket(entered, release)
        second = RecordingSocket()
        game.ws_connections = {
            _token_for(game_data, "hero_arien"): first,
            _token_for(game_data, "hero_wasp"): second,
        }

        task = asyncio.create_task(broadcast(game, client.app.state.registry))
        await entered.wait()
        # Simulate another request mutating live state while network I/O is
        # stalled. Every recipient must still receive the pre-I/O snapshot.
        state.turn = original_turn + 1
        release.set()
        await task
        return first.message, second.message

    first_message, second_message = asyncio.run(scenario())

    assert first_message["view"]["turn"] == original_turn
    assert second_message["view"]["turn"] == original_turn


def test_ws_broadcasts_are_serialized_per_game(client, game_data):
    import asyncio

    from goa2.domain.events import GameEvent, GameEventType
    from goa2.server.ws import broadcast

    class FirstSendBlocksSocket:
        def __init__(self, entered, release):
            self.entered = entered
            self.release = release
            self.messages = []

        async def send_json(self, data):
            self.messages.append(data)
            if len(self.messages) == 1:
                self.entered.set()
                await self.release.wait()

    game = client.app.state.registry.get(game_data["game_id"])

    def event(sequence):
        return GameEvent(
            event_type=GameEventType.TURN_ENDED,
            actor_id="hero_arien",
            metadata={"sequence": sequence},
        ).model_dump()

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        second_invoked = asyncio.Event()
        socket = FirstSendBlocksSocket(entered, release)
        game.ws_connections = {_token_for(game_data, "hero_arien"): socket}

        first = asyncio.create_task(
            broadcast(game, client.app.state.registry, events=[event("first")])
        )
        await entered.wait()

        async def send_second():
            second_invoked.set()
            await broadcast(game, client.app.state.registry, events=[event("second")])

        second = asyncio.create_task(send_second())
        await second_invoked.wait()
        await asyncio.sleep(0)
        messages_while_first_is_blocked = len(socket.messages)

        release.set()
        await asyncio.gather(first, second)
        return messages_while_first_is_blocked, socket.messages

    blocked_count, messages = asyncio.run(scenario())

    assert blocked_count == 1
    assert [message["events"][0]["metadata"]["sequence"] for message in messages] == [
        "first",
        "second",
    ]


# ---- Game over tests ----


def test_ws_state_update_includes_winner_on_game_over(client):
    """Verify STATE_UPDATE includes winner field when game ends."""
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
        },
    )
    data = resp.json()
    game_id = data["game_id"]
    _token_for(data, "hero_arien")
    _token_for(data, "hero_wasp")
    spectator_token = data["spectator_token"]

    # Connect spectator (non-acting player)
    with client.websocket_connect(f"/games/{game_id}/ws?token={spectator_token}") as ws_s:
        ws_s.receive_json()  # initial state

        # Access the game and set up game over state
        app = client.app
        registry = app.state.registry
        game = registry.get(game_id)

        # Simulate game over by setting last_result.winner and state phase
        from goa2.domain.models import GamePhase
        from goa2.engine.session import SessionResult, SessionResultType

        game.session.state.phase = GamePhase.GAME_OVER
        game.last_result = SessionResult(
            result_type=SessionResultType.GAME_OVER,
            current_phase=GamePhase.GAME_OVER,
            winner="RED",
            events=[],
        )

        # Manually trigger broadcast by sending a GET_VIEW request
        ws_s.send_json({"type": "GET_VIEW"})
        msg = ws_s.receive_json()

        # Verify STATE_UPDATE includes winner
        assert msg["type"] == "STATE_UPDATE"
        assert msg.get("winner") == "RED"
        assert msg["view"]["phase"] == "GAME_OVER"


# ---- ROLLBACK ----


def test_ws_rollback_spectator_blocked(client, game_data):
    """Spectators cannot send ROLLBACK messages."""
    game_id = game_data["game_id"]
    token = game_data["spectator_token"]
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()  # initial state
        ws.send_json({"type": "ROLLBACK"})
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"


def test_ws_rollback_no_active_actor(client, game_data):
    """ROLLBACK fails when no one is acting (PLANNING phase)."""
    game_id = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")
    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()  # initial state
        ws.send_json({"type": "ROLLBACK"})
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"


# ---- FINISH_PLANNING (Emmitt's Alternative Timelines) ----


def test_ws_finish_planning(client):
    """A two-card-capable hero can close planning with one card via WS."""
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Emmitt"],
            "blue_heroes": ["Wasp"],
        },
    )
    data = resp.json()
    game_id = data["game_id"]
    client.app.state.registry.get(game_id).session.state.get_hero("hero_emmitt").level = 8
    em_token = _token_for(data, "hero_emmitt")
    wa_token = _token_for(data, "hero_wasp")

    # Wasp commits via REST
    wa_view = client.get(f"/games/{game_id}", headers={"Authorization": f"Bearer {wa_token}"})
    wasp_card = next(
        h
        for t in wa_view.json()["view"]["teams"].values()
        for h in t["heroes"]
        if h["id"] == "hero_wasp"
    )["hand"][0]["id"]
    client.post(
        f"/games/{game_id}/cards",
        json={"card_id": wasp_card},
        headers={"Authorization": f"Bearer {wa_token}"},
    )

    with client.websocket_connect(f"/games/{game_id}/ws?token={em_token}") as ws:
        ws.receive_json()  # initial state
        ws.send_json({"type": "COMMIT_CARD", "card_id": "reverse_time"})
        msg = ws.receive_json()
        assert msg["type"] == "ACTION_RESULT"
        assert msg["current_phase"] == "PLANNING"  # still open for Emmitt
        ws.receive_json()  # broadcast STATE_UPDATE from the commit

        ws.send_json({"type": "FINISH_PLANNING"})
        msg = ws.receive_json()
        assert msg["type"] == "ACTION_RESULT"
        assert msg["current_phase"] == "RESOLUTION"


# ---- UNCOMMIT_CARD via WS ----


def test_ws_uncommit_card(client, game_data):
    game_id = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")

    view = client.get(f"/games/{game_id}", headers={"Authorization": f"Bearer {token}"}).json()
    hand = None
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_arien":
                hand = hero["hand"]
    assert hand
    card_id = hand[0]["id"]
    hand_size = len(hand)

    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()  # initial state
        ws.send_json({"type": "COMMIT_CARD", "card_id": card_id})
        assert ws.receive_json()["type"] == "ACTION_RESULT"
        assert ws.receive_json()["type"] == "STATE_UPDATE"  # broadcast (incl. sender)

        ws.send_json({"type": "UNCOMMIT_CARD"})
        msg = ws.receive_json()
        assert msg["type"] == "ACTION_RESULT"
        assert msg["current_phase"] == "PLANNING"

        # The broadcast after the uncommit shows the card back in hand.
        state = ws.receive_json()
        assert state["type"] == "STATE_UPDATE"
        for team_data in state["view"]["teams"].values():
            for hero in team_data["heroes"]:
                if hero["id"] == "hero_arien":
                    assert hero["current_turn_card"] is None
                    assert len(hero["hand"]) == hand_size


def test_ws_uncommit_nothing_committed_is_error(client, game_data):
    game_id = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")

    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()  # initial state
        ws.send_json({"type": "UNCOMMIT_CARD"})
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"
        assert "no committed card" in msg["detail"]


def test_ws_spectator_cannot_uncommit(client, game_data):
    game_id = game_data["game_id"]
    token = game_data["spectator_token"]

    with client.websocket_connect(f"/games/{game_id}/ws?token={token}") as ws:
        ws.receive_json()
        ws.send_json({"type": "UNCOMMIT_CARD"})
        msg = ws.receive_json()
        assert msg["type"] == "ERROR"
