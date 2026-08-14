"""REST endpoints under /games."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from goa2.domain.events import GameEvent, GameEventType
from goa2.domain.input import InputResponse
from goa2.domain.models import GamePhase
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.domain.views import build_view
from goa2.engine.session import GameSession, SessionResult
from goa2.engine.setup import GameSetup
from goa2.server.auth import PlayerDep, RegistryDep
from goa2.server.errors import (
    AlreadyCommittedError,
    CardNotInHandError,
    InvalidPhaseError,
    validate_input_turn,
    validate_simultaneous_input_scope,
)
from goa2.server.map_paths import MapFileNotFoundError, resolve_map_path
from goa2.server.models import (
    ActionResultResponse,
    CommitCardRequest,
    CreateGameRequest,
    CreateGameResponse,
    GameViewResponse,
    GiveGoldRequest,
    PlayerToken,
    ReadyRequest,
    SubmitInputRequest,
)
from goa2.server.time_control import (
    client_decision_timed_out,
    finalize_timed_mutation,
    mark_human_action,
    merge_timer_events,
    now_ms,
    set_player_ready,
    stop_clock_for_accepted_decision,
    timed_rest_mutation,
)
from goa2.server.visibility import (
    awaiting_input_hero_ids,
    events_for_viewer,
    input_request_for_viewer,
)

router = APIRouter(prefix="/games", tags=["games"])


def _map_path(map_name: str) -> str:
    """Resolve a map name to its JSON file path."""
    try:
        return resolve_map_path(map_name)
    except MapFileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Map '{map_name}' not found") from None


def _log_result(
    game,
    result: SessionResult,
    hero_id: str | None = None,
    action: str | None = None,
    detail: str | None = None,
) -> None:
    """Log a SessionResult to the game's logger."""
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


def _result_to_response(
    result: SessionResult,
    state: GameState,
    for_hero_id: str,
) -> ActionResultResponse:
    return ActionResultResponse(
        result_type=result.result_type.value,
        current_phase=result.current_phase.value,
        events=events_for_viewer([ev.model_dump() for ev in result.events], state, for_hero_id),
        input_request=input_request_for_viewer(result.input_request, state, for_hero_id),
        awaiting_input=awaiting_input_hero_ids(result.input_request, state),
        winner=result.winner,
    )


# ---- Endpoints ----


@router.post("", response_model=CreateGameResponse, status_code=201)
async def create_game(body: CreateGameRequest, registry: RegistryDep) -> CreateGameResponse:
    map_path = _map_path(body.map_name)
    if body.game_type not in ("QUICK", "LONG"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid game_type '{body.game_type}'. Must be QUICK or LONG.",
        )
    game_seed_id = uuid.uuid4().hex
    game_id = game_seed_id[:12]
    seed = int(game_seed_id, 16)
    state = GameSetup.create_game(
        map_path,
        body.red_heroes,
        body.blue_heroes,
        body.cheats_enabled,
        body.game_type,
        seed=seed,
        time_control=body.time_control,
    )
    session = GameSession(state)

    hero_ids: list[str] = []
    for team in state.teams.values():
        for hero in team.heroes:
            hero_ids.append(hero.id)

    game = registry.create_game(session, hero_ids, game_id=game_id)

    if game.game_logger:
        game.game_logger.log_game_created(body.red_heroes, body.blue_heroes, body.map_name)
    if game.replay_recorder:
        game.replay_recorder.record_setup(
            map_name=body.map_name,
            red_heroes=body.red_heroes,
            blue_heroes=body.blue_heroes,
            game_type=body.game_type,
            cheats=body.cheats_enabled,
            seed=seed,
            time_control=body.time_control,
        )

    return CreateGameResponse(
        game_id=game.game_id,
        player_tokens=[
            PlayerToken(hero_id=hid, token=tok) for hid, tok in game.hero_to_token.items()
        ],
        spectator_token=game.spectator_token,
    )


@router.post("/{game_id}/ready", response_model=GameViewResponse)
async def set_ready(
    game_id: str,
    body: ReadyRequest,
    player: PlayerDep,
    registry: RegistryDep,
) -> GameViewResponse:
    if player.is_spectator:
        raise HTTPException(status_code=403, detail="Spectators cannot change ready state")
    game = registry.get(game_id)
    async with game.outbound_lock:
        async with game.lock:
            timestamp = now_ms()
            set_player_ready(game, player.hero_id, body.ready, timestamp)
            finalize_timed_mutation(game, registry, timestamp)
            state = game.session.state
            response = GameViewResponse(
                view=build_view(state, for_hero_id=HeroID(player.hero_id), now_ms=timestamp),
                input_request=input_request_for_viewer(
                    game.last_result.input_request if game.last_result else None,
                    state,
                    player.hero_id,
                ),
                awaiting_input=awaiting_input_hero_ids(
                    game.last_result.input_request if game.last_result else None,
                    state,
                ),
                winner=game.last_result.winner if game.last_result else None,
            )

            # REST and WebSocket ready actions share the same public, atomic
            # transition. Materialize recipient-scoped updates before releasing
            # the game lock so every connected client sees the exact state that
            # this response reports.
            from goa2.server.ws import _capture_broadcast

            messages = _capture_broadcast(game)

        from goa2.server.ws import _send_captured_broadcast

        await _send_captured_broadcast(game, messages)
    return response


@router.get("/{game_id}", response_model=GameViewResponse)
async def get_game_view(
    game_id: str,
    player: PlayerDep,
    registry: RegistryDep,
) -> GameViewResponse:
    game = registry.get(game_id)
    hero_id = player.hero_id if not player.is_spectator else None
    hero_id_typed = HeroID(hero_id) if hero_id else None
    view = build_view(game.session.state, for_hero_id=hero_id_typed)
    ir = game.last_result.input_request if game.last_result else None
    winner = game.last_result.winner if game.last_result else None
    return GameViewResponse(
        view=view,
        input_request=input_request_for_viewer(ir, game.session.state, hero_id),
        awaiting_input=awaiting_input_hero_ids(ir, game.session.state),
        winner=winner,
    )


@router.post("/{game_id}/cards", response_model=ActionResultResponse)
async def commit_card(
    game_id: str,
    body: CommitCardRequest,
    player: PlayerDep,
    registry: RegistryDep,
) -> ActionResultResponse:
    if player.is_spectator:
        raise HTTPException(status_code=403, detail="Spectators cannot commit cards")
    game = registry.get(game_id)

    async with timed_rest_mutation(game, registry) as timer_events:
        if client_decision_timed_out(timer_events, hero_id=player.hero_id):
            raise ValueError("Decision already timed out")
        session = game.session
        if session.current_phase != GamePhase.PLANNING:
            raise InvalidPhaseError("PLANNING", session.current_phase.value)

        hero = session.state.get_hero(HeroID(player.hero_id))
        if hero is None:
            raise HTTPException(status_code=404, detail="Hero not found")

        card = next((c for c in hero.hand if c.id == body.card_id), None)
        if card is None:
            raise CardNotInHandError(body.card_id, player.hero_id)

        if player.hero_id in session.state.pending_inputs:
            # A hero whose active ultimate allows two cards (Emmitt) may
            # commit a second one; everyone else is rejected here.
            from goa2.engine.phases import hero_can_play_two_cards

            if (
                not hero_can_play_two_cards(hero)
                or player.hero_id in session.state.pending_second_cards
                or player.hero_id in session.state.planning_done
            ):
                raise AlreadyCommittedError(player.hero_id)

        rec_round, rec_turn = session.state.round, session.state.turn
        stop_clock_for_accepted_decision(
            game,
            hero_id=player.hero_id,
            completes_planning=True,
        )
        result = session.commit_card(HeroID(player.hero_id), card)
        mark_human_action(game)
        if game.replay_recorder:
            game.replay_recorder.record_commit(player.hero_id, body.card_id, rec_round, rec_turn)
        result = merge_timer_events(result, timer_events)
        game.last_result = result
        if game.game_logger:
            game.game_logger.log_card_commit(player.hero_id, body.card_id)
        _log_result(game, result)
        return _result_to_response(result, session.state, player.hero_id)


@router.post("/{game_id}/uncommit", response_model=ActionResultResponse)
async def uncommit_card(
    game_id: str,
    player: PlayerDep,
    registry: RegistryDep,
) -> ActionResultResponse:
    """Take a committed card back into hand during Planning (board-game
    take-back; LIFO for a two-card hero). Allowed until the last player
    commits — that commit runs revelation synchronously, so a late uncommit
    gets a 409 phase error."""
    if player.is_spectator:
        raise HTTPException(status_code=403, detail="Spectators cannot uncommit cards")
    game = registry.get(game_id)

    async with timed_rest_mutation(game, registry) as timer_events:
        if client_decision_timed_out(timer_events, hero_id=player.hero_id):
            raise ValueError("Decision already timed out")
        session = game.session
        if session.current_phase != GamePhase.PLANNING:
            raise InvalidPhaseError("PLANNING", session.current_phase.value)

        # NOTE: validate+mutate stays synchronous under game.lock — that is
        # what makes an uncommit racing the final commit resolve cleanly
        # (one side simply loses; state never corrupts). Don't add awaits
        # between the phase check and the session call.
        state = session.state
        hid = HeroID(player.hero_id)
        if state.clock and state.clock.players[hid].planning_locked_by_timeout:
            raise ValueError("Planning is locked after an automatic timeout decision")
        card = state.pending_second_cards.get(hid) or state.pending_inputs.get(hid)
        rec_round, rec_turn = state.round, state.turn
        stop_clock_for_accepted_decision(
            game,
            hero_id=player.hero_id,
            completes_planning=True,
        )
        result = session.uncommit_card(hid)
        mark_human_action(game)
        # Record only after success so a failed attempt never lands in the replay.
        if game.replay_recorder:
            game.replay_recorder.record_uncommit(hid, rec_round, rec_turn)
        result = merge_timer_events(result, timer_events)
        game.last_result = result
        if game.game_logger and card is not None:
            game.game_logger.log_card_uncommit(hid, card.id)
        _log_result(game, result)
        return _result_to_response(result, session.state, player.hero_id)


@router.post("/{game_id}/pass", response_model=ActionResultResponse)
async def pass_turn(
    game_id: str,
    player: PlayerDep,
    registry: RegistryDep,
) -> ActionResultResponse:
    if player.is_spectator:
        raise HTTPException(status_code=403, detail="Spectators cannot pass")
    game = registry.get(game_id)

    async with timed_rest_mutation(game, registry) as timer_events:
        if client_decision_timed_out(timer_events, hero_id=player.hero_id):
            raise ValueError("Decision already timed out")
        session = game.session
        if session.current_phase != GamePhase.PLANNING:
            raise InvalidPhaseError("PLANNING", session.current_phase.value)

        rec_round, rec_turn = session.state.round, session.state.turn
        stop_clock_for_accepted_decision(
            game,
            hero_id=player.hero_id,
            completes_planning=True,
        )
        result = session.pass_turn(HeroID(player.hero_id))
        mark_human_action(game)
        if game.replay_recorder:
            game.replay_recorder.record_pass(player.hero_id, rec_round, rec_turn)
        result = merge_timer_events(result, timer_events)
        game.last_result = result
        if game.game_logger:
            game.game_logger.log_pass_turn(player.hero_id)
        _log_result(game, result)
        return _result_to_response(result, session.state, player.hero_id)


@router.post("/{game_id}/planning-done", response_model=ActionResultResponse)
async def planning_done(
    game_id: str,
    player: PlayerDep,
    registry: RegistryDep,
) -> ActionResultResponse:
    """Done-signal for a hero who may play two cards (Emmitt's ultimate) but
    chooses to play only one this turn. Requires a committed card."""
    if player.is_spectator:
        raise HTTPException(status_code=403, detail="Spectators cannot finish planning")
    game = registry.get(game_id)

    async with timed_rest_mutation(game, registry) as timer_events:
        if client_decision_timed_out(timer_events, hero_id=player.hero_id):
            raise ValueError("Decision already timed out")
        session = game.session
        if session.current_phase != GamePhase.PLANNING:
            raise InvalidPhaseError("PLANNING", session.current_phase.value)

        rec_round, rec_turn = session.state.round, session.state.turn
        stop_clock_for_accepted_decision(
            game,
            hero_id=player.hero_id,
            completes_planning=True,
        )
        result = session.finish_planning(HeroID(player.hero_id))
        mark_human_action(game)
        if game.replay_recorder:
            game.replay_recorder.record_finish_planning(player.hero_id, rec_round, rec_turn)
        result = merge_timer_events(result, timer_events)
        game.last_result = result
        _log_result(game, result)
        return _result_to_response(result, session.state, player.hero_id)


@router.post("/{game_id}/input", response_model=ActionResultResponse)
async def submit_input(
    game_id: str,
    body: SubmitInputRequest,
    player: PlayerDep,
    registry: RegistryDep,
) -> ActionResultResponse:
    if player.is_spectator:
        raise HTTPException(status_code=403, detail="Spectators cannot submit input")
    game = registry.get(game_id)

    async with timed_rest_mutation(game, registry) as timer_events:
        if client_decision_timed_out(
            timer_events,
            hero_id=player.hero_id,
            request_id=body.request_id,
        ):
            raise ValueError("Decision already timed out")
        # Turn validation (skip for simultaneous phases like UPGRADE_PHASE)
        if game.last_result and game.last_result.input_request:
            expected = game.last_result.input_request.player_id
            validate_input_turn(expected, player.hero_id, game.session.state)
            # For a per-hero simultaneous phase (UPGRADE_PHASE), a player may
            # only submit for their own hero.
            validate_simultaneous_input_scope(
                game.last_result.input_request, body.selection, player.hero_id
            )

        response = InputResponse(
            request_id=body.request_id,
            selection=body.selection,
        )
        if game.game_logger:
            game.game_logger.log_input_response(player.hero_id, body.selection)
        # Record only after the engine accepts the input, tagged with the
        # pre-advance round/turn, so a rejected selection leaves no phantom decision.
        rec_round, rec_turn = game.session.state.round, game.session.state.turn
        stop_clock_for_accepted_decision(
            game,
            hero_id=player.hero_id,
            request_id=body.request_id,
        )
        result = game.session.advance(response)
        mark_human_action(game)
        if game.replay_recorder:
            game.replay_recorder.record_input(player.hero_id, body.selection, rec_round, rec_turn)
        result = merge_timer_events(result, timer_events)
        game.last_result = result
        _log_result(game, result)
        return _result_to_response(result, game.session.state, player.hero_id)


@router.post("/{game_id}/advance", response_model=ActionResultResponse)
async def advance(
    game_id: str,
    player: PlayerDep,
    registry: RegistryDep,
) -> ActionResultResponse:
    if player.is_spectator:
        raise HTTPException(status_code=403, detail="Spectators cannot advance")
    game = registry.get(game_id)

    async with timed_rest_mutation(game, registry) as timer_events:
        result = game.session.advance()
        mark_human_action(game)
        result = merge_timer_events(result, timer_events)
        game.last_result = result
        _log_result(game, result)
        return _result_to_response(result, game.session.state, player.hero_id)


@router.post("/{game_id}/rollback", response_model=ActionResultResponse)
async def rollback_action(
    game_id: str,
    player: PlayerDep,
    registry: RegistryDep,
) -> ActionResultResponse:
    if player.is_spectator:
        raise HTTPException(status_code=403, detail="Spectators cannot rollback")
    game = registry.get(game_id)

    async with timed_rest_mutation(game, registry) as timer_events:
        session = game.session
        if session.state.current_actor_id is None:
            raise HTTPException(status_code=400, detail="No active resolution to rollback")
        # Authorize against whoever the pending input is addressed to — under
        # Hanu's ultimate that is the controller, not the controlled actor.
        if game.current_responder != player.hero_id:
            raise HTTPException(status_code=403, detail="Only the current actor can rollback")
        rec_round, rec_turn = session.state.round, session.state.turn
        request = game.last_result.input_request if game.last_result else None
        stop_clock_for_accepted_decision(
            game,
            hero_id=player.hero_id,
            request_id=request.id if request else None,
        )
        try:
            result = session.rollback()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        mark_human_action(game)
        if game.replay_recorder:
            game.replay_recorder.record_rollback(player.hero_id, rec_round, rec_turn)
        result = merge_timer_events(result, timer_events)
        game.last_result = result
        _log_result(game, result)
        return _result_to_response(result, session.state, player.hero_id)


@router.post("/{game_id}/cheats/gold", response_model=ActionResultResponse)
async def give_gold_cheat(
    game_id: str,
    body: GiveGoldRequest,
    player: PlayerDep,
    registry: RegistryDep,
) -> ActionResultResponse:
    if player.is_spectator:
        raise HTTPException(status_code=403, detail="Spectators cannot use cheats")
    game = registry.get(game_id)

    async with timed_rest_mutation(game, registry) as timer_events:
        session = game.session

        if not session.state.cheats_enabled:
            raise HTTPException(status_code=403, detail="Cheats are not enabled for this game")

        if session.current_phase != GamePhase.PLANNING:
            raise InvalidPhaseError("PLANNING", session.current_phase.value)

        hero = session.state.get_hero(HeroID(body.hero_id))
        if hero is None:
            raise HTTPException(status_code=404, detail=f"Hero '{body.hero_id}' not found")

        if body.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be a positive integer")

        stop_clock_for_accepted_decision(
            game,
            hero_id=player.hero_id,
            completes_planning=True,
        )
        hero.gold += body.amount
        mark_human_action(game)
        if game.replay_recorder:
            game.replay_recorder.record_cheat_gold(
                body.hero_id, body.amount, session.state.round, session.state.turn
            )

        event = GameEvent(
            event_type=GameEventType.GOLD_GAINED,
            actor_id=hero.id,
            metadata={"amount": body.amount, "reason": "cheat"},
        )

        result = ActionResultResponse(
            result_type="ACTION_COMPLETE",
            current_phase=session.current_phase.value,
            events=[*[ev.model_dump() for ev in timer_events], event.model_dump()],
            input_request=None,
            winner=None,
        )

        if game.game_logger:
            game.game_logger.log_events([event.model_dump()])

        return result
