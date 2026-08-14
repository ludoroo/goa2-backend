"""WebSocket handler for real-time game events."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from goa2.domain.events import GameEvent, GameEventType
from goa2.domain.hex import Hex
from goa2.domain.input import InputResponse
from goa2.domain.models import GamePhase
from goa2.domain.types import HeroID
from goa2.domain.views import build_view
from goa2.engine.overrides import OverrideRejectedError, apply_override_decision
from goa2.engine.session import SessionResult
from goa2.server import overrides as ov
from goa2.server.bots import schedule_bot_drive
from goa2.server.errors import (
    CardNotInHandError,
    GameNotFoundError,
    InvalidPhaseError,
    NotYourTurnError,
    validate_input_turn,
    validate_simultaneous_input_scope,
)
from goa2.server.registry import GameRegistry, ManagedGame
from goa2.server.replay import load_replay, rebuild_session_for_rewind
from goa2.server.time_control import (
    client_decision_timed_out,
    finalize_timed_mutation,
    mark_human_action,
    now_ms,
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

router = APIRouter()

PING_MIN_INTERVAL_SECONDS = 0.45
PING_CARD_ZONES = frozenset({"CURRENT", "EXTRA", "PLAYED", "DISCARD", "ULTIMATE", "CAST"})

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
    }
)

CapturedBroadcast = list[tuple[str | None, WebSocket, dict[str, Any]]]


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


def _build_state_update(game: ManagedGame, hero_id: str | None) -> dict[str, Any]:
    """Build a STATE_UPDATE message for a specific player."""
    hero_id_typed = HeroID(hero_id) if hero_id else None
    view = build_view(game.session.state, for_hero_id=hero_id_typed)
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
    return msg


async def _send_json(ws: WebSocket, data: dict[str, Any]) -> bool:
    """Send JSON to a websocket, returning False if the connection is dead."""
    try:
        await ws.send_json(data)
        return True
    except Exception:
        return False


def _capture_broadcast(
    game: ManagedGame,
    events: list[dict[str, Any]] | None = None,
) -> CapturedBroadcast:
    """Capture every recipient's scoped payload from one state snapshot.

    Callers must hold ``game.lock`` so no mutation can interleave while the
    player-specific views and event projections are being materialized.
    """
    messages: CapturedBroadcast = []
    for token, ws in list(game.ws_connections.items()):
        hero_id = game.player_tokens.get(token)
        msg = _build_state_update(game, hero_id)
        if events:
            msg["events"] = events_for_viewer(events, game.session.state, hero_id)
        messages.append((token, ws, msg))
    for ws in list(game.spectator_ws_connections.values()):
        msg = _build_state_update(game, None)
        if events:
            msg["events"] = events_for_viewer(events, game.session.state, None)
        messages.append((None, ws, msg))
    return messages


async def _send_captured_broadcast(
    game: ManagedGame,
    messages: CapturedBroadcast,
) -> None:
    """Send already-materialized payloads and prune failed connections."""
    dead_connections: list[tuple[str | None, WebSocket]] = []
    for token, ws, msg in messages:
        if not await _send_json(ws, msg):
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


def _log_ws_result(game: ManagedGame, result) -> None:
    """Log a SessionResult from a WebSocket action."""
    gl = game.game_logger
    if not gl:
        return
    state = game.session.state
    gl.log_phase_change(result.current_phase.value, state.round, state.turn)
    events = [ev.model_dump() for ev in result.events]
    if events:
        gl.log_events(events)
    if result.input_request:
        gl.log_input_request(result.input_request.to_dict())
    if result.winner:
        gl.log_game_over(result.winner)


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
    _log_ws_result(game, result)
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
    _log_ws_result(game, result)
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
    _log_ws_result(game, result)
    return _action_result_message(game, result, hero_id)


async def _handle_set_ready(
    game: ManagedGame, hero_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    ready = data.get("ready")
    if not isinstance(ready, bool):
        raise ValueError("ready must be a boolean")
    set_player_ready(game, hero_id, ready, now_ms())
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
    _log_ws_result(game, result)
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
    _log_ws_result(game, result)
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
    _log_ws_result(game, result)
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

    if outcome == "applied":
        prepare_timed_mutation(game, registry=registry)
        rec_round, rec_turn = game.session.state.round, game.session.state.turn
        try:
            if proposal.family == "rewind":
                if game.replay_recorder is None:
                    raise OverrideRejectedError(
                        "This game has no replay log to rewind", code="no_replay"
                    )
                new_session = rebuild_session_for_rewind(
                    str(game.replay_recorder.path), proposal.to
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
        except (OverrideRejectedError, ValueError) as exc:
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
            if game.replay_recorder:
                game.replay_recorder.record_override(record)
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
                            target = _normalize_ping_target(game, data)
                            ping_messages = _capture_ping(game, hero_id, target)
                        await _send_captured_broadcast(game, ping_messages)
                except ValueError as exc:
                    async with game.outbound_lock:
                        await websocket.send_json({"type": "ERROR", "detail": str(exc)})
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

            try:
                async with game.outbound_lock:
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
                        messages = (
                            _capture_broadcast(
                                game,
                                reply.get("events") or timer_event_dicts,
                            )
                            if msg_type in MUTATION_MESSAGE_TYPES
                            else []
                        )

                    await websocket.send_json(sender_reply)
                    await _send_captured_broadcast(game, messages)
                    # After every mutation (or a lost-deadline path that
                    # already applied the automatic timeout inline), give
                    # the bot coordinator a chance to make the next move.
                    # ``schedule_bot_drive`` is idempotent so ``SET_READY``
                    # racing an already-running worker is a no-op.
                    if msg_type in MUTATION_MESSAGE_TYPES:
                        schedule_bot_drive(game, registry)

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
                # An inline timeout that ran during ``prepare_timed_mutation``
                # applied a real mutation under game.lock (persisted + logged
                # + broadcast just above) even though the outer client
                # request failed validation. A bot that owes the next move
                # must be scheduled here or the game stalls until another
                # client action lands. ``schedule_bot_drive`` is
                # idempotent + tombstone-safe.
                if timer_events and game.bot_specs:
                    schedule_bot_drive(game, registry)

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
