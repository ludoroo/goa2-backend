from __future__ import annotations

import json
import os
import random

import pytest
from fastapi.testclient import TestClient

from goa2.domain.input import SKIP, InputOption, InputRequest, InputRequestType
from goa2.domain.models import GamePhase
from goa2.domain.time_control import ClockKind, ClockStatus, TimeControlConfig
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.session import GameSession, SessionResult, SessionResultType
from goa2.engine.setup import GameSetup
from goa2.engine.steps.cards import ResolveUpgradesStep
from goa2.server.app import create_app
from goa2.server.map_paths import resolve_map_path
from goa2.server.registry import GameRegistry, ManagedGame
from goa2.server.replay import ReplayRecorder, replay_game
from goa2.server.time_control import (
    _apply_input_timeout,
    _timeout_selection,
    apply_due_timeouts,
    client_decision_timed_out,
    mark_human_action,
    prepare_timed_mutation,
    reconcile_game_clock,
    set_player_ready,
    stop_clock_for_accepted_decision,
)
from goa2.server.visibility import events_for_viewer


def _config(**overrides: int) -> TimeControlConfig:
    values = {
        "planning_allowance_seconds": 10,
        "resolution_allowance_seconds": 20,
        "response_grant_seconds": 15,
        "initial_time_bank_seconds": 30,
        "time_bank_increment_seconds": 5,
        "max_time_bank_seconds": 60,
        "upgrade_allowance_seconds": 10,
    }
    values.update(overrides)
    return TimeControlConfig(**values)


def _game(config: TimeControlConfig | None = None) -> ManagedGame:
    state = GameSetup.create_game(
        resolve_map_path("forgotten_island"),
        ["Arien"],
        ["Wasp"],
        time_control=config,
        seed=123,
    )
    session = GameSession(state)
    hero_ids = [str(hero.id) for team in state.teams.values() for hero in team.heroes]
    return GameRegistry().create_game(session, hero_ids, game_id="timed-test")


def _game_2v2(config: TimeControlConfig) -> ManagedGame:
    state = GameSetup.create_game(
        resolve_map_path("forgotten_island"),
        ["Arien", "Min"],
        ["Wasp", "Brogan"],
        time_control=config,
        seed=321,
    )
    session = GameSession(state)
    hero_ids = [str(hero.id) for team in state.teams.values() for hero in team.heroes]
    return GameRegistry().create_game(session, hero_ids, game_id="timed-team-test")


def _emmitt_game(config: TimeControlConfig) -> ManagedGame:
    state = GameSetup.create_game(
        resolve_map_path("forgotten_island"),
        ["Emmitt"],
        ["Wasp"],
        time_control=config,
        seed=999,
    )
    session = GameSession(state)
    hero_ids = [str(hero.id) for team in state.teams.values() for hero in team.heroes]
    return GameRegistry().create_game(session, hero_ids, game_id="timed-emmitt-test")


def _start(game: ManagedGame, at_ms: int = 0) -> None:
    set_player_ready(game, "hero_arien", True, at_ms)
    set_player_ready(game, "hero_wasp", True, at_ms)


@pytest.fixture
def client(tmp_path):
    os.environ["GOA2_SAVE_DIR"] = str(tmp_path)
    with TestClient(create_app()) as test_client:
        yield test_client
    os.environ.pop("GOA2_SAVE_DIR", None)


def _api_config() -> dict[str, int]:
    return _config().model_dump(mode="json")


def _api_token(game_data: dict, hero_id: str) -> str:
    return next(
        entry["token"] for entry in game_data["player_tokens"] if entry["hero_id"] == hero_id
    )


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_rest_timed_game_ready_check_gates_gameplay(client: TestClient) -> None:
    created = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "time_control": _api_config(),
        },
    )
    assert created.status_code == 201
    game = created.json()
    arien_token = _api_token(game, "hero_arien")
    wasp_token = _api_token(game, "hero_wasp")

    initial = client.get(f"/games/{game['game_id']}", headers=_auth(arien_token)).json()["view"]
    assert initial["clock"]["status"] == "WAITING_FOR_PLAYERS"
    arien = next(
        hero
        for team in initial["teams"].values()
        for hero in team["heroes"]
        if hero["id"] == "hero_arien"
    )
    blocked = client.post(
        f"/games/{game['game_id']}/cards",
        json={"card_id": arien["hand"][0]["id"]},
        headers=_auth(arien_token),
    )
    assert blocked.status_code == 400
    assert "every player" in blocked.json()["detail"]

    first = client.post(
        f"/games/{game['game_id']}/ready",
        json={"ready": True},
        headers=_auth(arien_token),
    )
    assert first.json()["view"]["clock"]["status"] == "WAITING_FOR_PLAYERS"
    second = client.post(
        f"/games/{game['game_id']}/ready",
        json={"ready": True},
        headers=_auth(wasp_token),
    )
    assert second.json()["view"]["clock"]["status"] == "RUNNING"
    assert set(second.json()["view"]["clock"]["active"]["hero_ids"]) == {
        "hero_arien",
        "hero_wasp",
    }


def test_draft_time_control_can_be_configured_and_explicitly_disabled(
    client: TestClient,
) -> None:
    created = client.post(
        "/drafts",
        json={"host_name": "Host", "time_control": _api_config()},
    )
    assert created.status_code == 201
    data = created.json()
    headers = _auth(data["player_token"])
    view = client.get(f"/drafts/{data['draft_id']}", headers=headers).json()
    assert view["draft"]["time_control"]["response_grant_seconds"] == 15

    disabled = client.patch(
        f"/drafts/{data['draft_id']}/settings",
        json={"time_control": None},
        headers=headers,
    )
    assert disabled.status_code == 200
    assert disabled.json()["draft"]["time_control"] is None


def test_websocket_ready_updates_are_broadcast_and_final_ready_starts_clock(
    client: TestClient,
) -> None:
    created = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "time_control": _api_config(),
        },
    ).json()
    game_id = created["game_id"]
    arien_token = _api_token(created, "hero_arien")
    wasp_token = _api_token(created, "hero_wasp")

    with (
        client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as arien_ws,
        client.websocket_connect(f"/games/{game_id}/ws?token={wasp_token}") as wasp_ws,
    ):
        assert arien_ws.receive_json()["view"]["clock"]["status"] == "WAITING_FOR_PLAYERS"
        assert wasp_ws.receive_json()["view"]["clock"]["status"] == "WAITING_FOR_PLAYERS"

        arien_ws.send_json({"type": "SET_READY", "ready": True})
        assert arien_ws.receive_json()["type"] == "READY_UPDATED"
        assert arien_ws.receive_json()["view"]["clock"]["ready_hero_ids"] == ["hero_arien"]
        assert wasp_ws.receive_json()["view"]["clock"]["ready_hero_ids"] == ["hero_arien"]

        wasp_ws.send_json({"type": "SET_READY", "ready": True})
        assert wasp_ws.receive_json()["type"] == "READY_UPDATED"
        arien_started = arien_ws.receive_json()
        wasp_started = wasp_ws.receive_json()
        assert arien_started["view"]["clock"]["status"] == "RUNNING"
        assert wasp_started["view"]["clock"]["status"] == "RUNNING"


def test_rest_ready_updates_are_broadcast_to_connected_clients(client: TestClient) -> None:
    created = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "time_control": _api_config(),
        },
    ).json()
    game_id = created["game_id"]
    arien_token = _api_token(created, "hero_arien")
    wasp_token = _api_token(created, "hero_wasp")

    with client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws:
        assert ws.receive_json()["view"]["clock"]["ready_hero_ids"] == []
        response = client.post(
            f"/games/{game_id}/ready",
            json={"ready": True},
            headers=_auth(wasp_token),
        )
        assert response.status_code == 200
        update = ws.receive_json()
        assert update["type"] == "STATE_UPDATE"
        assert update["view"]["clock"]["ready_hero_ids"] == ["hero_wasp"]


def test_rejected_rest_action_restarts_paused_planning_clocks(client: TestClient) -> None:
    created = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "time_control": _api_config(),
        },
    ).json()
    game_id = created["game_id"]
    arien_token = _api_token(created, "hero_arien")
    wasp_token = _api_token(created, "hero_wasp")
    client.post(f"/games/{game_id}/ready", json={"ready": True}, headers=_auth(arien_token))
    client.post(f"/games/{game_id}/ready", json={"ready": True}, headers=_auth(wasp_token))

    # Passing with cards in hand is rejected after the route has paused every
    # Planning clock to exclude backend processing time.
    rejected = client.post(f"/games/{game_id}/pass", headers=_auth(arien_token))
    assert rejected.status_code == 400
    assert "cannot pass while holding" in rejected.json()["detail"]

    game = client.app.state.registry.get(game_id)
    clock = game.session.state.clock
    assert clock is not None
    assert set(clock.active_hero_ids) == {"hero_arien", "hero_wasp"}
    assert game.timer_task is not None and not game.timer_task.done()


def test_rejected_websocket_action_restarts_paused_planning_clocks(
    client: TestClient,
) -> None:
    created = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "time_control": _api_config(),
        },
    ).json()
    game_id = created["game_id"]
    arien_token = _api_token(created, "hero_arien")
    wasp_token = _api_token(created, "hero_wasp")
    client.post(f"/games/{game_id}/ready", json={"ready": True}, headers=_auth(arien_token))
    client.post(f"/games/{game_id}/ready", json={"ready": True}, headers=_auth(wasp_token))

    with client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws:
        ws.receive_json()
        ws.send_json({"type": "PASS_TURN"})
        error = ws.receive_json()
        assert error["type"] == "ERROR"
        assert "cannot pass while holding" in error["detail"]

        game = client.app.state.registry.get(game_id)
        clock = game.session.state.clock
        assert clock is not None
        assert set(clock.active_hero_ids) == {"hero_arien", "hero_wasp"}
        assert game.timer_task is not None and not game.timer_task.done()


def test_rest_deadline_race_broadcasts_automatic_timeout(client: TestClient) -> None:
    created = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "time_control": _api_config(),
        },
    ).json()
    game_id = created["game_id"]
    arien_token = _api_token(created, "hero_arien")
    wasp_token = _api_token(created, "hero_wasp")
    client.post(f"/games/{game_id}/ready", json={"ready": True}, headers=_auth(arien_token))
    client.post(f"/games/{game_id}/ready", json={"ready": True}, headers=_auth(wasp_token))

    game = client.app.state.registry.get(game_id)
    clock = game.session.state.clock
    arien = game.session.state.get_hero(HeroID("hero_arien"))
    assert clock is not None and arien is not None
    clock.players["hero_arien"].planning_allowance_ms = 0
    clock.players["hero_arien"].time_bank_ms = 0

    with client.websocket_connect(f"/games/{game_id}/ws?token={wasp_token}") as ws:
        ws.receive_json()
        late = client.post(
            f"/games/{game_id}/cards",
            json={"card_id": arien.hand[0].id},
            headers=_auth(arien_token),
        )
        assert late.status_code == 400
        assert late.json()["detail"] == "Decision already timed out"

        update = ws.receive_json()
        assert update["type"] == "STATE_UPDATE"
        assert any(event["event_type"] == "TIMER_EXPIRED" for event in update["events"])
        assert update["view"]["clock"]["players"]["hero_arien"]["planning_locked_by_timeout"]


def test_websocket_ready_requires_a_boolean(client: TestClient) -> None:
    created = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "time_control": _api_config(),
        },
    ).json()
    game_id = created["game_id"]
    arien_token = _api_token(created, "hero_arien")

    with client.websocket_connect(f"/games/{game_id}/ws?token={arien_token}") as ws:
        ws.receive_json()
        ws.send_json({"type": "SET_READY", "ready": "false"})
        error = ws.receive_json()
        assert error == {"type": "ERROR", "detail": "ready must be a boolean"}


def test_timed_match_blocks_play_until_all_players_are_ready() -> None:
    game = _game(_config())
    clock = game.session.state.clock
    assert clock is not None
    assert clock.status == ClockStatus.WAITING_FOR_PLAYERS

    with pytest.raises(ValueError, match="every player"):
        prepare_timed_mutation(game, 0)

    assert not set_player_ready(game, "hero_arien", True, 0)
    assert set_player_ready(game, "hero_wasp", True, 0)
    assert clock.status == ClockStatus.RUNNING
    assert set(clock.active_hero_ids) == {"hero_arien", "hero_wasp"}


def test_fully_automatic_turn_limit_suspends_and_all_ready_resumes() -> None:
    game = _game(_config(automatic_turn_limit=2))
    _start(game)
    state = game.session.state
    clock = state.clock
    assert clock is not None

    # The first completed turn had no accepted human decision. It increments
    # the counter but the next Planning phase still starts normally.
    state.turn = 2
    reconcile_game_clock(game, 1_000)
    assert clock.status == ClockStatus.RUNNING
    assert clock.consecutive_automatic_turns == 1
    assert (clock.turn_round, clock.turn_number) == (1, 2)

    # The second fully automatic turn reaches the limit. The engine has entered
    # Turn 3, but its fresh pools/clocks wait for a new ready check.
    state.turn = 3
    reconcile_game_clock(game, 2_000)
    assert clock.status == ClockStatus.SUSPENDED_FOR_INACTIVITY
    assert clock.consecutive_automatic_turns == 2
    assert (clock.turn_round, clock.turn_number) == (1, 2)
    assert clock.ready_hero_ids == []
    assert clock.active_kind is None
    with pytest.raises(ValueError, match="suspended for inactivity"):
        prepare_timed_mutation(game, 2_000)

    assert not set_player_ready(game, "hero_arien", True, 3_000)
    assert set_player_ready(game, "hero_wasp", True, 3_000)
    assert clock.status == ClockStatus.RUNNING
    assert clock.consecutive_automatic_turns == 0
    assert (clock.turn_round, clock.turn_number) == (1, 3)
    assert set(clock.active_hero_ids) == {"hero_arien", "hero_wasp"}
    assert clock.players["hero_arien"].planning_allowance_ms == 10_000


def test_human_action_resets_automatic_turn_counter() -> None:
    game = _game(_config(automatic_turn_limit=1))
    _start(game)
    state = game.session.state
    clock = state.clock
    assert clock is not None

    mark_human_action(game)
    state.turn = 2
    reconcile_game_clock(game, 1_000)

    assert clock.status == ClockStatus.RUNNING
    assert clock.consecutive_automatic_turns == 0
    assert not clock.human_action_seen_this_turn


def test_zero_automatic_turn_limit_disables_inactivity_suspension() -> None:
    game = _game(_config(automatic_turn_limit=0))
    _start(game)
    state = game.session.state
    clock = state.clock
    assert clock is not None

    for turn in range(2, 6):
        state.turn = turn
        reconcile_game_clock(game, turn * 1_000)

    assert clock.status == ClockStatus.RUNNING
    assert clock.consecutive_automatic_turns == 4
    assert (clock.turn_round, clock.turn_number) == (1, 5)


def test_resolution_timeout_fallback_priority() -> None:
    hold = InputRequest(
        request_type=InputRequestType.CHOOSE_ACTION,
        player_id="hero_arien",
        can_skip=True,
        options=[InputOption(id="MOVE", text="Move"), InputOption(id="HOLD", text="Hold")],
    )
    assert _timeout_selection(hold, random.Random(1)) == "HOLD"

    skippable = InputRequest(
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_arien",
        can_skip=True,
        options=[InputOption(id="A", text="A")],
    )
    assert _timeout_selection(skippable, random.Random(1)) == SKIP

    passable = InputRequest(
        request_type=InputRequestType.SELECT_CARD_OR_PASS,
        player_id="hero_arien",
        options=[InputOption(id="card-a", text="Card"), InputOption(id="PASS", text="Pass")],
    )
    assert _timeout_selection(passable, random.Random(1)) == "PASS"

    mandatory = InputRequest(
        request_type=InputRequestType.SELECT_NUMBER,
        player_id="hero_arien",
        options=[InputOption(id="1", text="One"), InputOption(id="2", text="Two")],
    )
    assert _timeout_selection(mandatory, random.Random(0)) in {1, 2}


def test_planning_timeout_commits_one_random_card_and_locks_takeback() -> None:
    game = _game(
        _config(
            planning_allowance_seconds=1,
            initial_time_bank_seconds=0,
            max_time_bank_seconds=0,
        )
    )
    _start(game)
    clock = game.session.state.clock
    assert clock is not None
    # Leave the opponent enough time so this assertion isolates one timeout.
    clock.players["hero_wasp"].planning_allowance_ms = 5_000

    events = apply_due_timeouts(game, 1_000, rng=random.Random(7))

    committed = game.session.state.pending_inputs[HeroID("hero_arien")]
    assert committed is not None
    assert clock.players["hero_arien"].planning_locked_by_timeout
    assert "hero_arien" not in clock.active_hero_ids
    assert [event.event_type.value for event in events] == ["TIMER_EXPIRED"]
    assert events[0].metadata["automatic_action"] == "commit"
    assert events[0].metadata["card_id"] == committed.id
    assert client_decision_timed_out(events, hero_id="hero_arien")

    raw = [event.model_dump() for event in events]
    owner_event = events_for_viewer(raw, game.session.state, "hero_arien")[0]
    opponent_event = events_for_viewer(raw, game.session.state, "hero_wasp")[0]
    assert owner_event["metadata"]["card_id"] == committed.id
    assert opponent_event["metadata"]["card_id"] is None

    assert game.replay_recorder is not None
    replay_record = json.loads(game.replay_recorder.path.read_text().splitlines()[-1])
    assert replay_record["type"] == "timer_timeout"
    assert replay_record["action"] == "commit"
    assert replay_record["r"] == 1
    assert replay_record["t"] == 1
    assert replay_record["hero"] == "hero_arien"
    assert replay_record["card"] == committed.id
    assert isinstance(replay_record["ts"], float)


def test_emmitt_timeout_keeps_an_existing_first_commit_and_finishes_planning() -> None:
    game = _emmitt_game(
        _config(
            planning_allowance_seconds=1,
            initial_time_bank_seconds=0,
            max_time_bank_seconds=0,
        )
    )
    emmitt = game.session.state.get_hero(HeroID("hero_emmitt"))
    assert emmitt is not None
    emmitt.level = 8
    clock = game.session.state.clock
    assert clock is not None
    for hero_id in list(clock.players):
        set_player_ready(game, hero_id, True, 0)

    first = emmitt.hand[0]
    game.last_result = game.session.commit_card(emmitt.id, first)
    reconcile_game_clock(game, 0)
    clock.players["hero_wasp"].planning_allowance_ms = 5_000

    apply_due_timeouts(game, 1_000, rng=random.Random(4))

    assert game.session.state.pending_inputs[emmitt.id] is first
    assert emmitt.id not in game.session.state.pending_second_cards
    assert emmitt.id in game.session.state.planning_done
    assert clock.players["hero_emmitt"].planning_locked_by_timeout


def test_out_of_turn_and_team_requests_get_response_time_once() -> None:
    game = _game(_config(response_grant_seconds=15))
    _start(game)
    state = game.session.state
    clock = state.clock
    assert clock is not None
    state.phase = GamePhase.RESOLUTION
    state.resolution_owner_id = HeroID("hero_arien")

    request = InputRequest(
        id="defense-1",
        request_type=InputRequestType.SELECT_CARD_OR_PASS,
        player_id="hero_wasp",
    )
    game.last_result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=request,
        current_phase=GamePhase.RESOLUTION,
    )
    reconcile_game_clock(game, 0)
    assert clock.active_kind == ClockKind.RESPONSE
    assert clock.active_hero_ids == ["hero_wasp"]
    assert clock.players["hero_wasp"].response_time_ms == 15_000

    reconcile_game_clock(game, 0)
    assert clock.players["hero_wasp"].response_time_ms == 15_000

    team_request = request.model_copy(update={"id": "team-1", "player_id": "team:BLUE"})
    game.last_result = game.last_result.model_copy(update={"input_request": team_request})
    reconcile_game_clock(game, 0)
    assert clock.active_kind == ClockKind.RESPONSE
    assert clock.active_hero_ids == ["hero_wasp"]
    assert clock.players["hero_wasp"].response_time_ms == 30_000


def test_first_primary_resolution_actor_receives_initiative_bonus_once() -> None:
    game = _game(_config(initiative_bonus_seconds=15))
    _start(game)
    state = game.session.state
    clock = state.clock
    assert clock is not None
    state.phase = GamePhase.RESOLUTION
    state.resolution_owner_id = HeroID("hero_arien")

    primary = InputRequest(
        id="primary-1",
        request_type=InputRequestType.CHOOSE_ACTION,
        player_id="hero_arien",
        options=[InputOption(id="HOLD", text="Hold")],
    )
    game.last_result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=primary,
        current_phase=GamePhase.RESOLUTION,
    )
    reconcile_game_clock(game, 0)

    arien_clock = clock.players["hero_arien"]
    wasp_clock = clock.players["hero_wasp"]
    assert clock.initiative_bonus_hero_id == "hero_arien"
    assert arien_clock.initiative_bonus_ms == 15_000
    assert wasp_clock.initiative_bonus_ms == 0

    # Reconciliation is idempotent and cannot grant the same shared-turn
    # bonus again.
    reconcile_game_clock(game, 0)
    assert arien_clock.initiative_bonus_ms == 15_000

    # A later response by the first actor does not spend the bonus.
    state.resolution_owner_id = HeroID("hero_wasp")
    response = primary.model_copy(update={"id": "response-1", "player_id": "hero_arien"})
    game.last_result = game.last_result.model_copy(update={"input_request": response})
    reconcile_game_clock(game, 0)
    assert clock.active_kind == ClockKind.RESPONSE
    reconcile_game_clock(game, 5_000)
    assert arien_clock.initiative_bonus_ms == 15_000

    # The next primary actor cannot claim the first-actor bonus in the same
    # shared turn.
    state.resolution_owner_id = HeroID("hero_wasp")
    next_primary = response.model_copy(update={"id": "primary-2", "player_id": "hero_wasp"})
    game.last_result = game.last_result.model_copy(update={"input_request": next_primary})
    reconcile_game_clock(game, 5_000)
    assert clock.active_kind == ClockKind.RESOLUTION
    assert wasp_clock.initiative_bonus_ms == 0

    # Moving to a fresh shared turn resets the old grant and makes the new
    # first primary actor eligible again.
    state.turn = 2
    reconcile_game_clock(game, 5_000)
    assert clock.initiative_bonus_hero_id == "hero_wasp"
    assert arien_clock.initiative_bonus_ms == 0
    assert wasp_clock.initiative_bonus_ms == 15_000


def test_turn_boundary_prompt_without_owner_cannot_claim_initiative_bonus() -> None:
    """Expiring-effect finishing steps run with no resolution owner.

    They are still RESOLUTION-phase, hero-scoped prompts, so they charge the
    Resolution allowance — but they belong to nobody's turn and must not
    consume the shared turn's one-shot bonus.
    """
    game = _game(_config(initiative_bonus_seconds=15))
    _start(game)
    state = game.session.state
    clock = state.clock
    assert clock is not None
    state.phase = GamePhase.RESOLUTION
    # FinalizeHeroTurnStep cleared the owner before the turn-boundary steps.
    state.resolution_owner_id = None

    finishing = InputRequest(
        id="delayed-jump-1",
        request_type=InputRequestType.SELECT_HEX,
        player_id="hero_arien",
        options=[InputOption(id="hex", text="Hex")],
    )
    game.last_result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=finishing,
        current_phase=GamePhase.RESOLUTION,
    )
    reconcile_game_clock(game, 0)

    assert clock.active_kind == ClockKind.RESOLUTION
    assert clock.initiative_bonus_hero_id is None
    assert clock.players["hero_arien"].initiative_bonus_ms == 0

    # The bonus survives for the next shared turn's real first actor.
    state.turn = 2
    state.resolution_owner_id = HeroID("hero_wasp")
    primary = finishing.model_copy(update={"id": "primary-1", "player_id": "hero_wasp"})
    game.last_result = game.last_result.model_copy(update={"input_request": primary})
    reconcile_game_clock(game, 0)
    assert clock.initiative_bonus_hero_id == "hero_wasp"
    assert clock.players["hero_wasp"].initiative_bonus_ms == 15_000


def test_team_timeout_waits_until_every_eligible_person_exhausts(monkeypatch) -> None:
    game = _game_2v2(
        _config(
            response_grant_seconds=0,
            initial_time_bank_seconds=0,
            max_time_bank_seconds=0,
        )
    )
    clock = game.session.state.clock
    assert clock is not None
    for hero_id in clock.players:
        set_player_ready(game, hero_id, True, 0)
    state = game.session.state
    state.phase = GamePhase.RESOLUTION
    state.resolution_owner_id = HeroID("hero_arien")
    request = InputRequest(
        id="team-choice",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="team:BLUE",
        options=[InputOption(id="PASS", text="Pass")],
    )
    game.last_result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=request,
        current_phase=GamePhase.RESOLUTION,
    )
    reconcile_game_clock(game, 0)
    clock.players["hero_wasp"].resolution_allowance_ms = 1_000
    clock.players["hero_brogan"].resolution_allowance_ms = 3_000
    assert set(clock.active_decision_hero_ids) == {"hero_wasp", "hero_brogan"}

    submitted = []

    def fake_advance(response):
        submitted.append(response.selection)
        return SessionResult(
            result_type=SessionResultType.ACTION_COMPLETE,
            current_phase=GamePhase.RESOLUTION,
        )

    monkeypatch.setattr(game.session, "advance", fake_advance)

    assert apply_due_timeouts(game, 1_000, rng=random.Random(1)) == []
    assert submitted == []
    assert clock.active_hero_ids == ["hero_brogan"]

    events = apply_due_timeouts(game, 3_000, rng=random.Random(1))
    assert submitted == ["PASS"]
    timeout = next(event for event in events if event.event_type.value == "TIMER_EXPIRED")
    assert timeout.metadata["team"] == "team:BLUE"
    assert set(timeout.metadata["eligible_hero_ids"]) == {"hero_wasp", "hero_brogan"}
    assert game.replay_recorder is not None
    replay_record = json.loads(game.replay_recorder.path.read_text().splitlines()[-1])
    assert replay_record["team"] == "team:BLUE"
    assert set(replay_record["eligible_heroes"]) == {"hero_wasp", "hero_brogan"}


def test_authoritative_server_processing_time_is_not_charged_to_players() -> None:
    game = _game(_config(initial_time_bank_seconds=0, max_time_bank_seconds=0))
    _start(game)
    clock = game.session.state.clock
    assert clock is not None

    prepare_timed_mutation(game, 1_000)
    assert clock.players["hero_arien"].planning_allowance_ms == 9_000
    assert clock.players["hero_wasp"].planning_allowance_ms == 9_000
    stop_clock_for_accepted_decision(
        game,
        hero_id="hero_arien",
        completes_planning=True,
    )
    arien = game.session.state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    game.last_result = game.session.commit_card(arien.id, arien.hand[0])

    # Simulate four seconds of backend work while the game lock is held.
    reconcile_game_clock(game, 5_000)
    assert clock.players["hero_arien"].planning_allowance_ms == 9_000
    assert clock.players["hero_wasp"].planning_allowance_ms == 9_000


def test_action_choice_timeout_selects_hold() -> None:
    game = _game(_config())
    _start(game)
    state = game.session.state

    arien = state.get_hero(HeroID("hero_arien"))
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert arien is not None and wasp is not None
    arien_card = max(arien.hand, key=lambda card: card.initiative)
    wasp_card = min(wasp.hand, key=lambda card: card.initiative)
    result = game.session.commit_card(arien.id, arien_card)
    result = game.session.commit_card(wasp.id, wasp_card)
    game.last_result = result
    request = result.input_request
    assert request is not None
    assert request.request_type == InputRequestType.CHOOSE_ACTION

    reconcile_game_clock(game, 0)
    clock = state.clock
    assert clock is not None and clock.active_hero_ids
    active = clock.active_hero_ids[0]
    clock.players[active].resolution_allowance_ms = 0
    clock.players[active].time_bank_ms = 0

    events = apply_due_timeouts(game, 0, rng=random.Random(2))

    timeout = next(event for event in events if event.event_type.value == "TIMER_EXPIRED")
    assert timeout.metadata["selection"] == "HOLD"
    assert game.replay_recorder is not None
    replay_records = [
        json.loads(line) for line in game.replay_recorder.path.read_text().splitlines()
    ]
    assert any(
        record["type"] == "timer_timeout"
        and record["action"] == "input"
        and record["sel"] == "HOLD"
        for record in replay_records
    )
    assert game.last_result is not None
    assert game.last_result.input_request is None or game.last_result.input_request.id != request.id


def test_resolution_timeout_freezes_rollback_before_automatic_input(monkeypatch) -> None:
    game = _game(_config())
    request = InputRequest(
        id="mandatory-choice",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_arien",
        options=[InputOption(id="A", text="A")],
    )
    observed: dict[str, object] = {}

    def fake_advance(response):
        observed["selection"] = response.selection
        observed["rollback_frozen"] = game.session.state.execution_context.get("rollback_frozen")
        observed["snapshot"] = game.session._rollback_snapshot
        return SessionResult(
            result_type=SessionResultType.ACTION_COMPLETE,
            current_phase=GamePhase.RESOLUTION,
        )

    game.session._rollback_snapshot = {"some": "state"}
    game.session._rollback_actor_id = "hero_arien"
    monkeypatch.setattr(game.session, "advance", fake_advance)

    _apply_input_timeout(
        game,
        request,
        ClockKind.RESOLUTION,
        "hero_arien",
        random.Random(1),
    )

    assert observed == {
        "selection": "A",
        "rollback_frozen": True,
        "snapshot": None,
    }


def test_resolution_timeout_replay_matches_live_control_flow(tmp_path) -> None:
    config = _config(
        resolution_allowance_seconds=0,
        response_grant_seconds=0,
        initial_time_bank_seconds=0,
        time_bank_increment_seconds=0,
        max_time_bank_seconds=0,
    )
    game = _game(config)
    game.replay_recorder = ReplayRecorder("timed-replay", str(tmp_path))
    game.replay_recorder.record_setup(
        map_name="forgotten_island",
        red_heroes=["Arien"],
        blue_heroes=["Wasp"],
        game_type="LONG",
        cheats=False,
        seed=123,
        time_control=config,
    )
    _start(game)

    state = game.session.state
    arien = state.get_hero(HeroID("hero_arien"))
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert arien is not None and wasp is not None
    arien_card = max(arien.hand, key=lambda card: card.initiative)
    wasp_card = min(wasp.hand, key=lambda card: card.initiative)

    game.replay_recorder.record_commit(arien.id, arien_card.id, state.round, state.turn)
    game.session.commit_card(arien.id, arien_card)
    game.replay_recorder.record_commit(wasp.id, wasp_card.id, state.round, state.turn)
    game.last_result = game.session.commit_card(wasp.id, wasp_card)
    reconcile_game_clock(game, 0)

    apply_due_timeouts(game, 0, rng=random.Random(2))
    live_request = game.last_result.input_request if game.last_result else None
    assert live_request is not None

    replayed = replay_game(str(game.replay_recorder.path))
    replay_result = replayed.advance()
    replay_request = replay_result.input_request
    assert replay_request is not None
    assert replay_request.player_id == live_request.player_id
    assert replay_request.request_type == live_request.request_type
    assert replayed.state.current_actor_id == state.current_actor_id
    assert replayed.state.resolution_owner_id == state.resolution_owner_id
    assert replayed.state.time_control == config
    assert replayed.state.clock is None


def test_timeout_chain_yields_when_resolution_target_changes(monkeypatch) -> None:
    game = _game(
        _config(
            resolution_allowance_seconds=0,
            response_grant_seconds=0,
            initial_time_bank_seconds=0,
            max_time_bank_seconds=0,
        )
    )
    _start(game)
    state = game.session.state
    clock = state.clock
    assert clock is not None
    state.phase = GamePhase.RESOLUTION
    state.resolution_owner_id = HeroID("hero_arien")
    first = InputRequest(
        id="first-zero-choice",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_arien",
        options=[InputOption(id="A", text="A")],
    )
    second = InputRequest(
        id="second-zero-choice",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption(id="B", text="B")],
    )
    game.last_result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=first,
        current_phase=GamePhase.RESOLUTION,
    )
    reconcile_game_clock(game, 0)
    submitted: list[str] = []

    def fake_advance(response):
        submitted.append(response.selection)
        return SessionResult(
            result_type=SessionResultType.INPUT_NEEDED,
            input_request=second,
            current_phase=GamePhase.RESOLUTION,
        )

    monkeypatch.setattr(game.session, "advance", fake_advance)

    events = apply_due_timeouts(game, 0, rng=random.Random(1))

    assert submitted == ["A"]
    assert len([event for event in events if event.event_type.value == "TIMER_EXPIRED"]) == 1
    assert clock.active_request_id == second.id
    assert clock.active_hero_ids == ["hero_wasp"]
    assert clock.players["hero_wasp"].response_time_ms == 0


def test_rules_rollback_never_refunds_clock_time() -> None:
    game = _game(_config(initial_time_bank_seconds=0, max_time_bank_seconds=0))
    _start(game)
    state = game.session.state
    arien = state.get_hero(HeroID("hero_arien"))
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert arien is not None and wasp is not None
    game.session.commit_card(arien.id, max(arien.hand, key=lambda card: card.initiative))
    game.last_result = game.session.commit_card(
        wasp.id,
        min(wasp.hand, key=lambda card: card.initiative),
    )
    assert game.last_result.input_request is not None
    reconcile_game_clock(game, 0)
    prepare_timed_mutation(game, 1_000)
    clock = game.session.state.clock
    assert clock is not None
    before = clock.model_dump(mode="json")

    game.session.rollback()

    assert game.session.state.clock is clock
    assert clock.model_dump(mode="json") == before


def test_level_up_timeout_completes_every_pending_upgrade_for_player() -> None:
    game = _game(
        _config(upgrade_allowance_seconds=0, initial_time_bank_seconds=0, max_time_bank_seconds=0)
    )
    _start(game)
    state = game.session.state
    state.phase = GamePhase.LEVEL_UP
    state.pending_upgrades[HeroID("hero_arien")] = 2
    push_steps(state, [ResolveUpgradesStep()])
    game.last_result = game.session.advance()
    reconcile_game_clock(game, 0)

    events = apply_due_timeouts(game, 0, rng=random.Random(3))

    assert HeroID("hero_arien") not in state.pending_upgrades
    timeouts = [event for event in events if event.event_type.value == "TIMER_EXPIRED"]
    assert len(timeouts) == 2
    assert all(event.metadata["clock_kind"] == "LEVEL_UP" for event in timeouts)
