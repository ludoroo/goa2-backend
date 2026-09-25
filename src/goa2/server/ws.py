"""WebSocket handler for real-time game events."""

from __future__ import annotations

import asyncio
import json
import re
import time
from concurrent.futures.process import BrokenProcessPool
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from goa2.domain.events import GameEvent, GameEventType
from goa2.domain.hex import Hex
from goa2.domain.input import InputResponse
from goa2.domain.models import GamePhase
from goa2.domain.time_control import ClockStatus
from goa2.domain.types import HeroID
from goa2.domain.views import _build_board_view, build_view
from goa2.engine.overrides import OverrideRejectedError, apply_override_decision
from goa2.engine.session import SessionResult
from goa2.server import overrides as ov
from goa2.server.errors import (
    CardNotInHandError,
    GameNotFoundError,
    InvalidPhaseError,
    NotYourTurnError,
    validate_input_turn,
    validate_simultaneous_input_scope,
)
from goa2.server.registry import GameRegistry, ManagedGame
from goa2.server.replay import (
    load_replay,
    rebuild_session_for_rewind,
)
from goa2.server.result_logging import log_session_result
from goa2.server.time_control import (
    client_decision_timed_out,
    finalize_timed_mutation,
    mark_human_action,
    now_ms,
    pause_game_for_consensus,
    prepare_timed_mutation,
    reconcile_game_clock,
    set_player_ready,
    stop_clock_for_accepted_decision,
)
from goa2.server.visibility import (
    awaiting_input_hero_ids,
    events_for_viewer,
    input_request_for_viewer,
)
from goa2.server.workers import run_heavy

router = APIRouter()

PING_MIN_INTERVAL_SECONDS = 0.45
POINTER_MIN_INTERVAL_SECONDS = 0.06
POINTER_ZONES = frozenset(
    {
        "COMMIT",
        "PLAYED",
        "ULTIMATE",
        "DECK",
        "DISCARD",
        "SPELLBOOK",
        "STATUS",
        "WISH",
        "HAND",
        "SHEET",
    }
)
PING_CARD_ZONES = frozenset({"CURRENT", "EXTRA", "PLAYED", "DISCARD", "ULTIMATE", "CAST"})
# The table stays quiet while a turn or an upgrade is being resolved.
PING_BLOCKED_PHASES = frozenset({GamePhase.RESOLUTION, GamePhase.LEVEL_UP})

MUTATION_MESSAGE_TYPES = frozenset(
    {
        "SUBMIT_INPUT",
        "COMMIT_CARD",
        "UNCOMMIT_CARD",
        "PASS_TURN",
        "FINISH_PLANNING",
        "ROLLBACK",
        "CHEATS_GOLD",
        "SET_READY",
        "STARTING_POSITION",
    }
)
CLIENT_ACTION_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,64}\Z")

CapturedMessage = tuple[str | None, WebSocket, dict[str, Any]]
CapturedBroadcast = list[CapturedMessage]


def _normalize_ping_target(game: ManagedGame, data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an ephemeral table-ping target.

    Card pings deliberately use a public table location rather than a card ID.
    A facedown committed card can therefore be pointed at without revealing its
    private identity to opponents or spectators.
    """
    target = data.get("target")
    if not isinstance(target, dict):
        raise ValueError("PING target must be an object")

    kind = target.get("kind")
    state = game.session.state
    if kind == "HEX":
        raw_hex = target.get("hex")
        if not isinstance(raw_hex, dict):
            raise ValueError("PING hex target must include hex coordinates")
        q = raw_hex.get("q")
        r = raw_hex.get("r")
        s = raw_hex.get("s")
        if type(q) is not int or type(r) is not int or type(s) is not int:
            raise ValueError("PING hex coordinates must be integers")
        ping_hex = Hex(q=q, r=r, s=s)
        if ping_hex not in state.board.tiles:
            raise ValueError("PING hex is not on the board")
        return {
            "kind": "HEX",
            "hex": {"q": ping_hex.q, "r": ping_hex.r, "s": ping_hex.s},
        }

    if kind != "CARD":
        raise ValueError("PING target kind must be HEX or CARD")

    hero_id = target.get("hero_id")
    zone = target.get("zone")
    if not isinstance(hero_id, str) or not hero_id or len(hero_id) > 100:
        raise ValueError("PING card target must include a valid hero_id")
    if zone not in PING_CARD_ZONES:
        raise ValueError("PING card target has an invalid zone")

    hero = state.get_hero(HeroID(hero_id))
    if hero is None:
        raise ValueError("PING card target hero was not found")

    normalized: dict[str, Any] = {
        "kind": "CARD",
        "hero_id": str(hero.id),
        "zone": zone,
    }
    if zone == "CURRENT":
        exists = hero.current_turn_card is not None
    elif zone == "EXTRA":
        extra_card = hero.extra_turn_card
        if state.phase == GamePhase.PLANNING and hero.id in state.pending_second_cards:
            extra_card = state.pending_inputs.get(hero.id)
        exists = extra_card is not None
    elif zone == "ULTIMATE":
        exists = hero.ultimate_card is not None and hero.level >= 8
    else:
        index = target.get("index")
        if type(index) is not int or index < 0:
            raise ValueError("PING card target zone requires a non-negative index")
        normalized["index"] = index
        if zone == "PLAYED":
            exists = index < len(hero.played_cards) and hero.played_cards[index] is not None
        elif zone == "DISCARD":
            exists = index < len(hero.discard_pile)
        else:  # CAST
            exists = index < len(hero.cast_spells)

    if not exists:
        raise ValueError("PING card target is not currently on the table")
    return normalized


def _capture_ping(game: ManagedGame, hero_id: str, target: dict[str, Any]) -> CapturedBroadcast:
    """Capture one immutable ping broadcast for every current connection."""
    message = {
        "type": "PING",
        "ping_id": uuid4().hex,
        "hero_id": hero_id,
        "target": target,
    }
    messages: CapturedBroadcast = [
        (token, ws, dict(message)) for token, ws in list(game.ws_connections.items())
    ]
    messages.extend(
        (None, ws, dict(message)) for ws in list(game.spectator_ws_connections.values())
    )
    return messages


def _action_result_message(
    game: ManagedGame, result: SessionResult, hero_id: str | None
) -> dict[str, Any]:
    """Build the ACTION_RESULT reply sent to the player who acted."""
    return {
        "type": "ACTION_RESULT",
        "result_type": result.result_type.value,
        "current_phase": result.current_phase.value,
        "events": [ev.model_dump() for ev in result.events],
        "input_request": input_request_for_viewer(
            result.input_request, game.session.state, hero_id
        ),
        "awaiting_input": awaiting_input_hero_ids(result.input_request, game.session.state),
        "winner": result.winner,
    }


def _build_state_update(
    game: ManagedGame,
    hero_id: str | None,
    *,
    board_view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a STATE_UPDATE message for a specific player."""
    hero_id_typed = HeroID(hero_id) if hero_id else None
    view = build_view(
        game.session.state,
        for_hero_id=hero_id_typed,
        prebuilt_board_view=board_view,
    )
    ir = game.last_result.input_request if game.last_result else None
    winner = game.last_result.winner if game.last_result else None
    msg: dict[str, Any] = {
        "type": "STATE_UPDATE",
        "view": view,
    }
    input_request = input_request_for_viewer(ir, game.session.state, hero_id)
    if input_request:
        msg["input_request"] = input_request
    msg["awaiting_input"] = awaiting_input_hero_ids(ir, game.session.state)
    if winner:
        msg["winner"] = winner
    # Omitted when empty, matching this builder's convention for falsy keys.
    # The client retains the last non-empty map it saw.
    if game.hero_names:
        msg["hero_names"] = game.hero_names
    return msg


async def _send_json(
    ws: WebSocket,
    data: dict[str, Any],
    *,
    encoded: str | None = None,
) -> bool:
    """Send JSON to a websocket, returning False if the connection is dead."""
    try:
        if encoded is None:
            await ws.send_json(data)
        else:
            await ws.send_text(encoded)
        return True
    except Exception:
        return False


def _capture_broadcast(
    game: ManagedGame,
    events: list[dict[str, Any]] | None = None,
    *,
    priority_token: str | None = None,
    client_action_id: str | None = None,
    board_view: dict[str, Any] | None = None,
    exclude_token: str | None = None,
    timing: dict[str, float] | None = None,
) -> CapturedBroadcast:
    """Capture every recipient's scoped payload from one state snapshot.

    Callers must hold ``game.lock`` so no mutation can interleave while the
    player-specific views and event projections are being materialized.
    """
    player_connections = [
        connection for connection in game.ws_connections.items() if connection[0] != exclude_token
    ]
    spectator_connections = list(game.spectator_ws_connections.values())
    if not player_connections and not spectator_connections:
        if timing is not None:
            timing["board_ms"] = 0.0
            timing["recipient_views_ms"] = 0.0
        return []

    if priority_token is not None:
        player_connections.sort(key=lambda connection: connection[0] != priority_token)

    capture_started = time.perf_counter()
    board_ms = 0.0
    if board_view is None:
        board_started = time.perf_counter()
        board_view = _build_board_view(game.session.state)
        board_ms = (time.perf_counter() - board_started) * 1000

    messages: CapturedBroadcast = []
    for token, ws in player_connections:
        hero_id = game.player_tokens.get(token)
        msg = _build_state_update(game, hero_id, board_view=board_view)
        if events:
            msg["events"] = events_for_viewer(events, game.session.state, hero_id)
        if token == priority_token and client_action_id is not None:
            msg["client_action_id"] = client_action_id
        messages.append((token, ws, msg))
    if spectator_connections:
        msg = _build_state_update(game, None, board_view=board_view)
        if events:
            msg["events"] = events_for_viewer(events, game.session.state, None)
        messages.extend((None, ws, msg) for ws in spectator_connections)
    if timing is not None:
        timing["board_ms"] = board_ms
        timing["recipient_views_ms"] = (time.perf_counter() - capture_started) * 1000 - board_ms
    return messages


def _capture_player_update(
    game: ManagedGame,
    token: str,
    events: list[dict[str, Any]] | None = None,
    *,
    board_view: dict[str, Any],
    client_action_id: str | None = None,
) -> CapturedMessage | None:
    """Capture one player's scoped update from the caller's locked snapshot."""
    ws = game.ws_connections.get(token)
    if ws is None:
        return None
    hero_id = game.player_tokens.get(token)
    msg = _build_state_update(game, hero_id, board_view=board_view)
    if events:
        msg["events"] = events_for_viewer(events, game.session.state, hero_id)
    if client_action_id is not None:
        msg["client_action_id"] = client_action_id
    return token, ws, msg


async def _send_captured_broadcast(
    game: ManagedGame,
    messages: CapturedBroadcast,
) -> None:
    """Send already-materialized payloads and prune failed connections."""
    payload_counts: dict[int, int] = {}
    for _, _, payload in messages:
        payload_id = id(payload)
        payload_counts[payload_id] = payload_counts.get(payload_id, 0) + 1

    encoded_payloads: dict[int, str | None] = {}
    dead_connections: list[tuple[str | None, WebSocket]] = []
    for token, ws, msg in messages:
        encoded = None
        payload_id = id(msg)
        if payload_counts[payload_id] > 1:
            if payload_id not in encoded_payloads:
                try:
                    encoded_payloads[payload_id] = json.dumps(
                        msg, separators=(",", ":"), ensure_ascii=False
                    )
                except Exception:
                    encoded_payloads[payload_id] = None
            encoded = encoded_payloads[payload_id]
            if encoded is None:
                dead_connections.append((token, ws))
                continue
        if not await _send_json(ws, msg, encoded=encoded):
            dead_connections.append((token, ws))
    for token, ws in dead_connections:
        if token is None:
            connection_id = id(ws)
            if game.spectator_ws_connections.get(connection_id) is ws:
                game.spectator_ws_connections.pop(connection_id, None)
        else:
            # A reconnect may have replaced this failed player socket while the
            # broadcast was awaiting I/O. Never remove the newer connection by
            # token alone.
            if game.ws_connections.get(token) is ws:
                game.ws_connections.pop(token, None)


async def broadcast(
    game: ManagedGame,
    registry: GameRegistry,
    events: list[dict[str, Any]] | None = None,
) -> None:
    """Send player-scoped state updates to all connected websockets.

    ``events`` is the internal event list from the mutation that triggered this
    broadcast. Each recipient receives a visibility-filtered projection so
    animations cannot expose hidden cards or facedown token identities.
    Connect/GET_VIEW updates pass nothing and stay event-free.
    """
    async with game.outbound_lock:
        async with game.lock:
            messages = _capture_broadcast(game, events)
        await _send_captured_broadcast(game, messages)


async def _handle_submit_input(
    game: ManagedGame, hero_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Handle SUBMIT_INPUT message."""
    # Turn validation (skip for simultaneous phases like UPGRADE_PHASE)
    if game.last_result and game.last_result.input_request:
        expected = game.last_result.input_request.player_id
        validate_input_turn(expected, hero_id, game.session.state)
        # For a per-hero simultaneous phase (UPGRADE_PHASE), a player may only
        # submit for their own hero.
        validate_simultaneous_input_scope(
            game.last_result.input_request, data.get("selection"), hero_id
        )

    response = InputResponse(
        request_id=data.get("request_id", ""),
        selection=data.get("selection"),
    )
    if game.game_logger:
        game.game_logger.log_input_response(hero_id, data.get("selection"))
    # Record only after the engine accepts the input, tagged with the pre-advance
    # round/turn, so a rejected selection leaves no phantom decision in the log.
    rec_round, rec_turn = game.session.state.round, game.session.state.turn
    stop_clock_for_accepted_decision(
        game,
        hero_id=hero_id,
        request_id=response.request_id,
    )
    result = game.session.advance(response)
    mark_human_action(game)
    if game.replay_recorder:
        game.replay_recorder.record_input(hero_id, data.get("selection"), rec_round, rec_turn)
    game.last_result = result
    log_session_result(game, result)
    return _action_result_message(game, result, hero_id)


async def _handle_commit_card(
    game: ManagedGame, hero_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Handle COMMIT_CARD message."""
    session = game.session
    if session.current_phase != GamePhase.PLANNING:
        raise InvalidPhaseError("PLANNING", session.current_phase.value)

    card_id = data.get("card_id", "")
    hero = session.state.get_hero(HeroID(hero_id))
    if hero is None:
        return {"type": "ERROR", "detail": "Hero not found"}

    card = next((c for c in hero.hand if c.id == card_id), None)
    if card is None:
        raise CardNotInHandError(card_id, hero_id)

    rec_round, rec_turn = session.state.round, session.state.turn
    stop_clock_for_accepted_decision(
        game,
        hero_id=hero_id,
        completes_planning=True,
    )
    result = session.commit_card(HeroID(hero_id), card)
    mark_human_action(game)
    if game.replay_recorder:
        game.replay_recorder.record_commit(hero_id, card_id, rec_round, rec_turn)
    game.last_result = result
    if game.game_logger:
        game.game_logger.log_card_commit(hero_id, card_id)
    log_session_result(game, result)
    return _action_result_message(game, result, hero_id)


async def _handle_uncommit_card(game: ManagedGame, hero_id: str) -> dict[str, Any]:
    """Handle UNCOMMIT_CARD (Planning take-back; LIFO for a two-card hero)."""
    session = game.session
    if session.current_phase != GamePhase.PLANNING:
        raise InvalidPhaseError("PLANNING", session.current_phase.value)

    # NOTE: validate+mutate stays synchronous under game.lock — that is what
    # makes an uncommit racing the final commit resolve cleanly (one side
    # simply loses; state never corrupts). Don't add awaits before the
    # session call.
    state = session.state
    hid = HeroID(hero_id)
    if state.clock and state.clock.players[hid].planning_locked_by_timeout:
        raise ValueError("Planning is locked after an automatic timeout decision")
    card = state.pending_second_cards.get(hid) or state.pending_inputs.get(hid)
    rec_round, rec_turn = state.round, state.turn
    stop_clock_for_accepted_decision(
        game,
        hero_id=hero_id,
        completes_planning=True,
    )
    result = session.uncommit_card(hid)
    mark_human_action(game)
    # Record only after success so a failed attempt never lands in the replay.
    if game.replay_recorder:
        game.replay_recorder.record_uncommit(hero_id, rec_round, rec_turn)
    game.last_result = result
    if game.game_logger and card is not None:
        game.game_logger.log_card_uncommit(hero_id, card.id)
    log_session_result(game, result)
    return _action_result_message(game, result, hero_id)


async def _handle_starting_position(
    game: ManagedGame, hero_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    from goa2.domain.hex import Hex
    from goa2.engine.starting_positions import (
        apply_position,
        clear_requests,
        request_swap,
        respond_swap,
    )

    state = game.session.state
    op = data.get("op")
    recorded = None
    if op == "move":
        destination = Hex.model_validate(data.get("destination"))
        apply_position(state, hero_id, destination=destination)
        recorded = {"hero": hero_id, "sel": {"destination": destination.model_dump()}}
    elif op == "request_swap":
        target = data.get("target")
        if not isinstance(target, str):
            raise ValueError("A teammate is required")
        request_swap(state, hero_id, target)
    elif op == "respond_swap":
        if not isinstance(data.get("accept"), bool) or not isinstance(data.get("request_id"), str):
            raise ValueError("A request ID and boolean acceptance are required")
        requester = respond_swap(state, hero_id, data["request_id"], data["accept"])
        if requester is not None:
            recorded = {"hero": requester, "sel": {"swap_with": hero_id}}
    elif op == "cancel":
        clear_requests(state, hero_id)
    else:
        raise ValueError("Unknown starting-position operation")
    if recorded is not None and game.replay_recorder:
        game.replay_recorder.record_starting_position(recorded)
    mark_human_action(game)
    return {"type": "STARTING_POSITION_UPDATED"}


async def _handle_set_ready(
    game: ManagedGame, hero_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    ready = data.get("ready")
    if not isinstance(ready, bool):
        raise ValueError("ready must be a boolean")
    timestamp = now_ms()
    set_player_ready(game, hero_id, ready, timestamp)
    if ready:
        # Local import avoids the bots -> ws broadcast dependency at import
        # time while sharing the same ready transition as REST.
        from goa2.server.bots import auto_ready_bot_heroes

        auto_ready_bot_heroes(game, at_ms=timestamp)
    return {
        "type": "READY_UPDATED",
        "hero_id": hero_id,
        "ready": ready,
    }


async def _handle_finish_planning(game: ManagedGame, hero_id: str) -> dict[str, Any]:
    """Handle FINISH_PLANNING message (done-signal for a two-card-capable
    hero — Emmitt's Alternative Timelines — playing only one card)."""
    session = game.session
    if session.current_phase != GamePhase.PLANNING:
        raise InvalidPhaseError("PLANNING", session.current_phase.value)

    rec_round, rec_turn = session.state.round, session.state.turn
    stop_clock_for_accepted_decision(
        game,
        hero_id=hero_id,
        completes_planning=True,
    )
    result = session.finish_planning(HeroID(hero_id))
    mark_human_action(game)
    if game.replay_recorder:
        game.replay_recorder.record_finish_planning(hero_id, rec_round, rec_turn)
    game.last_result = result
    log_session_result(game, result)
    return _action_result_message(game, result, hero_id)


async def _handle_pass_turn(game: ManagedGame, hero_id: str) -> dict[str, Any]:
    """Handle PASS_TURN message."""
    session = game.session
    if session.current_phase != GamePhase.PLANNING:
        raise InvalidPhaseError("PLANNING", session.current_phase.value)

    rec_round, rec_turn = session.state.round, session.state.turn
    stop_clock_for_accepted_decision(
        game,
        hero_id=hero_id,
        completes_planning=True,
    )
    result = session.pass_turn(HeroID(hero_id))
    mark_human_action(game)
    if game.replay_recorder:
        game.replay_recorder.record_pass(hero_id, rec_round, rec_turn)
    game.last_result = result
    if game.game_logger:
        game.game_logger.log_pass_turn(hero_id)
    log_session_result(game, result)
    return _action_result_message(game, result, hero_id)


async def _handle_rollback(game: ManagedGame, hero_id: str) -> dict[str, Any]:
    """Handle ROLLBACK message."""
    session = game.session
    if session.state.current_actor_id is None:
        raise NotYourTurnError(hero_id, "(no active actor)")
    # Authorize against whoever the pending input is addressed to — under Hanu's
    # ultimate that is the controller, not the controlled actor.
    responder = game.current_responder
    if responder != hero_id:
        raise NotYourTurnError(hero_id, responder or "(no active actor)")
    rec_round, rec_turn = session.state.round, session.state.turn
    request = game.last_result.input_request if game.last_result else None
    stop_clock_for_accepted_decision(
        game,
        hero_id=hero_id,
        request_id=request.id if request else None,
    )
    result = session.rollback()
    mark_human_action(game)
    if game.replay_recorder:
        game.replay_recorder.record_rollback(hero_id, rec_round, rec_turn)
    game.last_result = result
    log_session_result(game, result)
    return _action_result_message(game, result, hero_id)


async def _handle_cheats_gold(
    game: ManagedGame, hero_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Handle CHEATS_GOLD message."""
    session = game.session

    if not session.state.cheats_enabled:
        return {"type": "ERROR", "detail": "Cheats are not enabled for this game"}

    if session.current_phase != GamePhase.PLANNING:
        raise InvalidPhaseError("PLANNING", session.current_phase.value)

    target_hero_id = data.get("hero_id", "")
    hero = session.state.get_hero(target_hero_id)
    if hero is None:
        return {"type": "ERROR", "detail": f"Hero '{target_hero_id}' not found"}

    amount = data.get("amount", 0)
    if amount <= 0:
        return {"type": "ERROR", "detail": "Amount must be a positive integer"}

    stop_clock_for_accepted_decision(
        game,
        hero_id=hero_id,
        completes_planning=True,
    )
    hero.gold += amount
    mark_human_action(game)
    if game.replay_recorder:
        game.replay_recorder.record_cheat_gold(
            target_hero_id, amount, session.state.round, session.state.turn
        )

    event = GameEvent(
        event_type=GameEventType.GOLD_GAINED,
        actor_id=hero.id,
        metadata={"amount": amount, "reason": "cheat"},
    )

    if game.game_logger:
        game.game_logger.log_events([event.model_dump()])

    return {
        "type": "ACTION_RESULT",
        "result_type": "ACTION_COMPLETE",
        "current_phase": session.current_phase.value,
        "events": [event.model_dump()],
        "input_request": None,
        "winner": None,
    }


def _override_broadcast_to_all(game: ManagedGame, msg: dict[str, Any]) -> CapturedBroadcast:
    """Capture one override-protocol message for every connection (spectators
    included — they may watch the negotiation, never vote in it)."""
    messages: CapturedBroadcast = [
        (token, ws_conn, dict(msg)) for token, ws_conn in list(game.ws_connections.items())
    ]
    messages.extend(
        (None, ws_conn, dict(msg)) for ws_conn in list(game.spectator_ws_connections.values())
    )
    return messages


def _player_link_messages(game: ManagedGame, hero_id: str) -> CapturedBroadcast:
    """Hand the approved player token to the players only.

    Spectators watch the vote like any other, but the token is a seat at the
    table: it never goes out on a spectator connection.
    """
    token = game.hero_to_token.get(hero_id)
    if token is None:
        return []
    msg = {"type": "PLAYER_LINK_REVEALED", "hero_id": hero_id, "token": token}
    return [(tok, ws_conn, dict(msg)) for tok, ws_conn in list(game.ws_connections.items())]


async def _resolve_override(
    game: ManagedGame,
    registry: GameRegistry,
    outcome: str,
    reason: dict[str, str] | None = None,
) -> CapturedBroadcast:
    """Resolve the open proposal (caller holds game.lock). Returns broadcasts.

    On approval, spec ordering applies: apply via the op registry -> append
    the replay record -> save -> broadcast. A patch that fails validation at
    apply time downgrades the outcome to ``rejected`` with a structured reason.
    """
    proposal = game.pending_override
    assert proposal is not None
    game.pending_override = None
    if game.override_expiry_task is not None:
        game.override_expiry_task.cancel()
        game.override_expiry_task = None

    messages: CapturedBroadcast = []
    events: list[dict[str, Any]] | None = None

    if outcome == "applied" and proposal.family == "reveal_player":
        # A reveal mutates no GameState and charges nobody's clock, so it skips
        # the timed-mutation dance — which also lets the table recover a seat
        # while the match is paused for the very player who left.
        reconcile_game_clock(game, now_ms())
        messages.extend(
            _override_broadcast_to_all(game, ov.resolved_msg(proposal, outcome, reason))
        )
        messages.extend(_player_link_messages(game, str(proposal.args["hero_id"])))
        return messages

    if outcome == "applied":
        prepare_timed_mutation(game, registry=registry)
        rec_round, rec_turn = game.session.state.round, game.session.state.turn
        try:
            if proposal.family == "pause":
                pause_game_for_consensus(game, proposal.proposer_hero_id)
            elif proposal.family == "rewind":
                if game.replay_recorder is None:
                    raise OverrideRejectedError(
                        "This game has no replay log to rewind", code="no_replay"
                    )
                # Replaying a match from its log takes seconds; run it in a
                # worker process so the other tables keep their own core.
                new_session = await run_heavy(
                    rebuild_session_for_rewind, str(game.replay_recorder.path), proposal.to
                )
                game.session = new_session
                if new_session.state.phase in (GamePhase.PLANNING, GamePhase.GAME_OVER):
                    game.last_result = None
                else:
                    game.last_result = new_session.advance(None)
            else:
                result = apply_override_decision(game.session, proposal.op, proposal.args)
                if result is not None:
                    game.last_result = result
                    events = [ev.model_dump() for ev in result.events]
        except (OverrideRejectedError, ValueError, BrokenProcessPool) as exc:
            outcome = "rejected"
            code = getattr(exc, "code", "invalid_op")
            reason = {"code": code, "message": str(exc)}
        else:
            record: dict[str, Any] = {
                "type": "ov_rewind" if proposal.family == "rewind" else f"ov_{proposal.family}",
                "r": rec_round,
                "t": rec_turn,
                "hero": proposal.proposer_hero_id,
                "voters": proposal.tally()["yes"],
            }
            if proposal.family == "rewind":
                record["to"] = proposal.to
            else:
                record["op"] = proposal.op
                record["args"] = proposal.args
            # A pause mutates no GameState, so it is clock telemetry (a PAUSED
            # clock event) rather than a replayable override decision.
            if game.replay_recorder and proposal.family != "pause":
                game.replay_recorder.record_override(record)
        finally:
            # Clocks are paused by prepare_timed_mutation above; leaving them
            # that way strands the match, so reconcile even on an unforeseen
            # failure.
            finalize_timed_mutation(game, registry)  # reconcile + save + reschedule
        # Outcome first, then the fresh state, in one flush.
        messages.extend(
            _override_broadcast_to_all(game, ov.resolved_msg(proposal, outcome, reason))
        )
        if outcome == "applied":
            messages.extend(_capture_broadcast(game, events))
    else:
        # No state mutation; just un-pause the clocks.
        reconcile_game_clock(game, now_ms())
        registry.save_game(game.game_id)
        messages.extend(
            _override_broadcast_to_all(game, ov.resolved_msg(proposal, outcome, reason))
        )
    return messages


async def _expire_override(game: ManagedGame, registry: GameRegistry, proposal_id: str) -> None:
    """Background task: expiry is a rejection (nobody actively agreed)."""
    proposal = game.pending_override
    if proposal is None:
        return
    delay = max(0.0, proposal.expires_at - time.time())
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return
    async with game.outbound_lock:
        async with game.lock:
            current = game.pending_override
            if current is None or current.id != proposal_id:
                return
            messages = await _resolve_override(game, registry, "expired")
        await _send_captured_broadcast(game, messages)


async def _handle_override_message(
    game: ManagedGame,
    registry: GameRegistry,
    hero_id: str,
    msg_type: str,
    data: dict[str, Any],
) -> CapturedBroadcast:
    """Handle PROPOSE/VOTE/CANCEL under game.lock; returns captured broadcasts.

    These are deliberately NOT in MUTATION_MESSAGE_TYPES: a proposal or vote
    alone mutates nothing. The apply step runs its own prepare/finalize.
    """
    if msg_type == "PROPOSE_OVERRIDE":
        clock = game.session.state.clock
        family = data.get("family")
        if clock is not None and clock.status == ClockStatus.PAUSED and family != "reveal_player":
            raise ValueError("The match is paused; every player must ready up to resume")
        if family == "pause":
            if clock is None:
                raise ValueError("This match does not use time control")
            if clock.status != ClockStatus.RUNNING:
                raise ValueError("Only a running timed match can be paused")
        proposal = ov.create_proposal(game, hero_id, data)
        if proposal.family == "rewind" and game.replay_recorder is not None:
            target = proposal.to if proposal.to is not None else -1
            _, decisions = load_replay(str(game.replay_recorder.path))
            if not 0 <= target <= len(decisions):
                raise ValueError(f"Rewind target {proposal.to} out of range 0..{len(decisions)}")
        game.pending_override = proposal
        reconcile_game_clock(game, now_ms())  # pause the turn clock
        game.override_expiry_task = asyncio.create_task(
            _expire_override(game, registry, proposal.id)
        )
        messages = _override_broadcast_to_all(game, ov.proposed_msg(proposal))
        # A single-connected-player game reaches its majority on the
        # proposer's auto-yes alone.
        if proposal.outcome() == "applied":
            messages.extend(await _resolve_override(game, registry, "applied"))
        return messages

    proposal = game.pending_override  # type: ignore[assignment]  # Any on ManagedGame
    if proposal is None or proposal.id != data.get("proposal_id"):
        raise ValueError("No matching open proposal")

    if msg_type == "VOTE_OVERRIDE":
        ov.register_vote(proposal, hero_id, bool(data.get("approve")))
        outcome = proposal.outcome()
        if outcome is None:
            return _override_broadcast_to_all(game, ov.updated_msg(proposal))
        return await _resolve_override(game, registry, outcome)

    # CANCEL_OVERRIDE
    if hero_id != proposal.proposer_hero_id:
        raise ValueError("Only the proposer may cancel an override proposal")
    return await _resolve_override(game, registry, "cancelled")


@router.websocket("/games/{game_id}/ws")
async def game_ws(websocket: WebSocket, game_id: str) -> None:
    """WebSocket endpoint for real-time game interaction.

    Connect with ?token=<bearer_token> query parameter.
    """
    token = websocket.query_params.get("token", "")
    registry: GameRegistry = websocket.app.state.registry

    # Authenticate
    result = registry.resolve_token(token)
    if result is None:
        await websocket.close(code=4001, reason="Invalid token")
        return

    resolved_game_id, hero_id, is_spectator = result
    if resolved_game_id != game_id:
        await websocket.close(code=4003, reason="Token does not match game")
        return

    try:
        game = registry.get(game_id)
    except GameNotFoundError:
        await websocket.close(code=4004, reason="Game not found")
        return

    await websocket.accept()

    spectator_connection_id = id(websocket)

    # Player tokens own one live connection; the shared spectator token may
    # own any number of sockets. Register a replacement player socket before
    # closing its predecessor so the old handler's cleanup cannot leave a gap.
    async with game.outbound_lock:
        async with game.lock:
            previous_websocket = None
            if is_spectator:
                game.spectator_ws_connections[spectator_connection_id] = websocket
            else:
                previous_websocket = game.ws_connections.get(token)
                game.ws_connections[token] = websocket
            initial = _build_state_update(game, hero_id if not is_spectator else None)
        if previous_websocket is not None and previous_websocket is not websocket:
            await previous_websocket.close(
                code=4002,
                reason="Connection superseded by a newer session",
            )
        await websocket.send_json(initial)

    if game.game_logger:
        game.game_logger.log_ws_connect(hero_id if not is_spectator else None, is_spectator)

    last_ping_at = 0.0
    last_pointer_at = 0.0
    try:
        while True:
            raw = await websocket.receive_text()
            timer_events: list[GameEvent] = []
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                async with game.outbound_lock:
                    await websocket.send_json({"type": "ERROR", "detail": "Invalid JSON"})
                continue

            msg_type = data.get("type", "")

            if is_spectator and msg_type != "GET_VIEW":
                async with game.outbound_lock:
                    await websocket.send_json(
                        {"type": "ERROR", "detail": "Spectators can only GET_VIEW"}
                    )
                continue

            # Table pings are authenticated, ordered ephemeral messages. They
            # do not mutate the game and therefore never touch clocks, saves,
            # replays, logs, SessionResult, or STATE_UPDATE broadcasts.
            if msg_type == "PING":
                ping_at = time.monotonic()
                if ping_at - last_ping_at < PING_MIN_INTERVAL_SECONDS:
                    continue
                last_ping_at = ping_at
                try:
                    async with game.outbound_lock:
                        async with game.lock:
                            if game.session.state.phase in PING_BLOCKED_PHASES:
                                continue
                            target = _normalize_ping_target(game, data)
                            ping_messages = _capture_ping(game, hero_id, target)
                        await _send_captured_broadcast(game, ping_messages)
                except ValueError as exc:
                    async with game.outbound_lock:
                        await websocket.send_json({"type": "ERROR", "detail": str(exc)})
                continue

            # Pointer presence: the same ephemeral relay contract as pings.
            # Two payload shapes: raw 3D-table world coordinates (valid over
            # the shared map) or a semantic board-zone target — each client
            # lays hero boards out differently, so only "whose board, which
            # zone" is meaningful over seats. No game-state validation.
            if msg_type == "POINTER":
                pointer_at = time.monotonic()
                if pointer_at - last_pointer_at < POINTER_MIN_INTERVAL_SECONDS:
                    continue
                last_pointer_at = pointer_at
                message: dict[str, Any] = {"type": "POINTER", "hero_id": hero_id}
                if data.get("hidden") is True:
                    message["hidden"] = True
                else:
                    x = data.get("x")
                    z = data.get("z")
                    target_hero_id = data.get("target_hero_id")
                    zone = data.get("zone")
                    if x is not None or z is not None:
                        if type(x) not in (int, float) or type(z) not in (int, float):
                            continue
                        if not (-100.0 <= x <= 100.0 and -100.0 <= z <= 100.0):
                            continue
                        message["x"] = x
                        message["z"] = z
                    elif (
                        isinstance(target_hero_id, str)
                        and isinstance(zone, str)
                        and zone in POINTER_ZONES
                        and len(target_hero_id) <= 64
                        and target_hero_id.startswith("hero_")
                    ):
                        message["target_hero_id"] = target_hero_id
                        message["zone"] = zone
                    else:
                        continue
                pointer_messages: CapturedBroadcast = [
                    (tok, ws_conn, dict(message))
                    for tok, ws_conn in list(game.ws_connections.items())
                    if tok != token
                ]
                pointer_messages.extend(
                    (None, ws_conn, dict(message))
                    for ws_conn in list(game.spectator_ws_connections.values())
                )
                async with game.outbound_lock:
                    await _send_captured_broadcast(game, pointer_messages)
                continue

            if msg_type in ("PROPOSE_OVERRIDE", "VOTE_OVERRIDE", "CANCEL_OVERRIDE"):
                try:
                    async with game.outbound_lock:
                        async with game.lock:
                            override_messages = await _handle_override_message(
                                game, registry, hero_id, msg_type, data
                            )
                        await _send_captured_broadcast(game, override_messages)
                except ValueError as exc:
                    if game.game_logger:
                        game.game_logger.log_error(str(exc), hero_id)
                    async with game.outbound_lock:
                        await websocket.send_json({"type": "ERROR", "detail": str(exc)})
                continue

            raw_client_action_id = data.get("client_action_id")
            client_action_id = (
                raw_client_action_id
                if isinstance(raw_client_action_id, str)
                and CLIENT_ACTION_ID_RE.fullmatch(raw_client_action_id)
                else None
            )
            action_started = time.perf_counter()
            try:
                async with game.outbound_lock:
                    lock_wait_ms = (time.perf_counter() - action_started) * 1000
                    actor_update_ms: float | None = None
                    fanout_ms = 0.0
                    fanout_timing: dict[str, float] = {}
                    messages: CapturedBroadcast = []
                    remaining_messages: CapturedBroadcast = []
                    sender_reply_sent = False
                    send_ms = 0.0
                    async with game.lock:
                        timer_events = (
                            prepare_timed_mutation(game, registry=registry)
                            if msg_type in MUTATION_MESSAGE_TYPES and msg_type != "SET_READY"
                            else []
                        )
                        request_id = (
                            str(data.get("request_id", "")) if msg_type == "SUBMIT_INPUT" else None
                        )
                        lost_deadline_race = msg_type in {
                            "SUBMIT_INPUT",
                            "COMMIT_CARD",
                            "UNCOMMIT_CARD",
                            "PASS_TURN",
                            "FINISH_PLANNING",
                        } and client_decision_timed_out(
                            timer_events,
                            hero_id=hero_id,
                            request_id=request_id,
                        )
                        reply: dict[str, Any]
                        engine_started = time.perf_counter()
                        try:
                            if lost_deadline_race:
                                reply = {"type": "ERROR", "detail": "Decision already timed out"}
                            elif msg_type == "SUBMIT_INPUT":
                                reply = await _handle_submit_input(game, hero_id, data)
                            elif msg_type == "COMMIT_CARD":
                                reply = await _handle_commit_card(game, hero_id, data)
                            elif msg_type == "UNCOMMIT_CARD":
                                reply = await _handle_uncommit_card(game, hero_id)
                            elif msg_type == "PASS_TURN":
                                reply = await _handle_pass_turn(game, hero_id)
                            elif msg_type == "FINISH_PLANNING":
                                reply = await _handle_finish_planning(game, hero_id)
                            elif msg_type == "ROLLBACK":
                                reply = await _handle_rollback(game, hero_id)
                            elif msg_type == "CHEATS_GOLD":
                                reply = await _handle_cheats_gold(game, hero_id, data)
                            elif msg_type == "STARTING_POSITION":
                                reply = await _handle_starting_position(game, hero_id, data)
                            elif msg_type == "SET_READY":
                                reply = await _handle_set_ready(game, hero_id, data)
                            elif msg_type == "GET_VIEW":
                                hid = hero_id if not is_spectator else None
                                reply = _build_state_update(game, hid)
                            else:
                                reply = {
                                    "type": "ERROR",
                                    "detail": f"Unknown message type: {msg_type}",
                                }
                        except BaseException:
                            if msg_type in MUTATION_MESSAGE_TYPES:
                                # Restore any clocks paused by a handler before
                                # releasing game.lock, even for cancellation or
                                # an unexpected engine exception.
                                finalize_timed_mutation(game, registry)
                            raise
                        engine_ms = (time.perf_counter() - engine_started) * 1000

                        timer_event_dicts = [event.model_dump() for event in timer_events]
                        if timer_event_dicts and reply.get("type") == "ACTION_RESULT":
                            reply = {
                                **reply,
                                "events": [*timer_event_dicts, *reply.get("events", [])],
                            }

                        if msg_type in MUTATION_MESSAGE_TYPES and not lost_deadline_race:
                            finalize_timed_mutation(game, registry)

                        # Materialize both the direct reply and every scoped
                        # broadcast while they still describe this mutation.
                        sender_reply = reply
                        if reply.get("type") == "ACTION_RESULT":
                            sender_reply = {
                                **reply,
                                "events": events_for_viewer(
                                    reply.get("events", []), game.session.state, hero_id
                                ),
                            }
                        if client_action_id is not None:
                            sender_reply = {
                                **sender_reply,
                                "client_action_id": client_action_id,
                            }
                        if msg_type in MUTATION_MESSAGE_TYPES:
                            broadcast_events = reply.get("events") or timer_event_dicts
                            capture_started = time.perf_counter()
                            board_view = None
                            if game.ws_connections or game.spectator_ws_connections:
                                board_started = time.perf_counter()
                                board_view = _build_board_view(game.session.state)
                                fanout_timing["board_ms"] = (
                                    time.perf_counter() - board_started
                                ) * 1000

                                views_started = time.perf_counter()
                                actor_message = _capture_player_update(
                                    game,
                                    token,
                                    broadcast_events,
                                    board_view=board_view,
                                    client_action_id=client_action_id,
                                )
                                fanout_timing["recipient_views_ms"] = (
                                    time.perf_counter() - views_started
                                ) * 1000
                                actor_messages = [actor_message] if actor_message else []
                            else:
                                fanout_timing = {"board_ms": 0.0, "recipient_views_ms": 0.0}
                                actor_messages = []
                            fanout_ms += (time.perf_counter() - capture_started) * 1000

                            # Keep the state lock across the actor send. The
                            # remaining recipient views are then captured from
                            # the exact same locked mutation state.
                            actor_send_started = time.perf_counter()
                            await websocket.send_json(sender_reply)
                            sender_reply_sent = True
                            await _send_captured_broadcast(game, actor_messages)
                            send_ms += (time.perf_counter() - actor_send_started) * 1000
                            actor_update_ms = (
                                (time.perf_counter() - action_started) * 1000
                                if actor_messages
                                else None
                            )

                            remaining_started = time.perf_counter()
                            remaining_timing: dict[str, float] = {}
                            remaining_messages = _capture_broadcast(
                                game,
                                broadcast_events,
                                board_view=board_view,
                                exclude_token=token,
                                timing=remaining_timing,
                            )
                            fanout_ms += (time.perf_counter() - remaining_started) * 1000
                            fanout_timing["recipient_views_ms"] += remaining_timing.get(
                                "recipient_views_ms", 0.0
                            )
                            messages = [*actor_messages, *remaining_messages]

                    if not sender_reply_sent:
                        await websocket.send_json(sender_reply)
                    remaining_send_started = time.perf_counter()
                    await _send_captured_broadcast(game, remaining_messages)
                    send_ms += (time.perf_counter() - remaining_send_started) * 1000
                    if msg_type in MUTATION_MESSAGE_TYPES and game.game_logger:
                        game.game_logger.log_timing(
                            msg_type,
                            engine_ms=engine_ms,
                            fanout_ms=fanout_ms,
                            send_ms=send_ms,
                            clients=len(messages),
                            lock_wait_ms=lock_wait_ms,
                            board_ms=fanout_timing.get("board_ms", 0.0),
                            recipient_views_ms=fanout_timing.get("recipient_views_ms", 0.0),
                            actor_update_ms=actor_update_ms,
                            action_id=client_action_id,
                        )

            except (NotYourTurnError, InvalidPhaseError, CardNotInHandError, ValueError) as exc:
                if game.game_logger:
                    game.game_logger.log_error(str(exc), hero_id)
                async with game.outbound_lock:
                    error_messages: CapturedBroadcast = []
                    async with game.lock:
                        if timer_events:
                            error_messages = _capture_broadcast(
                                game,
                                [event.model_dump() for event in timer_events],
                            )
                    await websocket.send_json({"type": "ERROR", "detail": str(exc)})
                    if error_messages:
                        await _send_captured_broadcast(game, error_messages)

    except WebSocketDisconnect:
        pass
    finally:
        if is_spectator:
            if game.spectator_ws_connections.get(spectator_connection_id) is websocket:
                game.spectator_ws_connections.pop(spectator_connection_id, None)
        else:
            # The same player token may already belong to a newer reconnect.
            # Only remove this handler's socket, never the replacement.
            if game.ws_connections.get(token) is websocket:
                game.ws_connections.pop(token, None)
        if game.game_logger:
            game.game_logger.log_ws_disconnect(hero_id if not is_spectator else None, is_spectator)
