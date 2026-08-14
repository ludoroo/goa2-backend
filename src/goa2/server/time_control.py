"""Authoritative server coordinator for optional match clocks."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any

from goa2.domain.events import GameEvent, GameEventType
from goa2.domain.input import SKIP, InputRequest, InputRequestType, InputResponse, selection_value
from goa2.domain.models import GamePhase
from goa2.domain.time_control import (
    ClockKind,
    ClockStatus,
    activate_clocks,
    begin_level_up,
    begin_shared_turn,
    exhausted_active_hero_ids,
    finish_game_clock,
    grant_initiative_bonus,
    grant_response_time,
    milliseconds_until_next_exhaustion,
    settle_clock,
    start_game_clock,
    suspend_game_clock_for_inactivity,
    usable_time_ms,
)
from goa2.domain.types import HeroID
from goa2.engine.phases import planning_open_for_second_card
from goa2.engine.session import SessionResult
from goa2.server.registry import GameRegistry, ManagedGame

logger = logging.getLogger(__name__)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def _team_hero_ids(game: ManagedGame, player_id: str) -> list[str]:
    if not player_id.startswith("team:"):
        return []
    team_name = player_id.removeprefix("team:")
    return [
        str(hero.id)
        for team in game.session.state.teams.values()
        if team.color.value == team_name
        for hero in team.heroes
    ]


def _planning_complete(game: ManagedGame, hero_id: str) -> bool:
    state = game.session.state
    hid = HeroID(hero_id)
    return hid in state.pending_inputs and not planning_open_for_second_card(state, hid)


def set_player_ready(game: ManagedGame, hero_id: str, ready: bool, at_ms: int) -> bool:
    """Update the pre-match ready check and start exactly once when all are ready."""
    clock = game.session.state.clock
    if clock is None:
        raise ValueError("This match does not use time controls")
    if hero_id not in clock.players:
        raise ValueError(f"Unknown clock player {hero_id!r}")
    if clock.status not in {
        ClockStatus.WAITING_FOR_PLAYERS,
        ClockStatus.SUSPENDED_FOR_INACTIVITY,
    }:
        raise ValueError("The timed match is already running")
    resuming = clock.status == ClockStatus.SUSPENDED_FOR_INACTIVITY

    ready_ids = set(clock.ready_hero_ids)
    if ready:
        ready_ids.add(hero_id)
    else:
        ready_ids.discard(hero_id)
    clock.ready_hero_ids = [hid for hid in clock.players if hid in ready_ids]
    clock.revision += 1

    if ready_ids == set(clock.players):
        start_game_clock(clock, at_ms)
        if resuming:
            # Suspension happens after the engine has entered the next shared
            # turn but before its fresh pools are initialized. A completed
            # ready check starts that boundary exactly once.
            clock.consecutive_automatic_turns = 0
            config = game.session.state.time_control
            if config is None:
                raise ValueError("Suspended clock is missing its time-control configuration")
            begin_shared_turn(
                clock,
                config,
                round_number=game.session.state.round,
                turn_number=game.session.state.turn,
                now_ms=at_ms,
            )
        reconcile_game_clock(game, at_ms)
        return True
    return False


def reconcile_game_clock(game: ManagedGame, at_ms: int) -> None:
    """Set active personal clocks from the current phase and pending request."""
    state = game.session.state
    config = state.time_control
    clock = state.clock
    if config is None or clock is None:
        return
    if clock.status in {
        ClockStatus.WAITING_FOR_PLAYERS,
        ClockStatus.SUSPENDED_FOR_INACTIVITY,
    }:
        activate_clocks(clock, None, request_id=None, now_ms=at_ms)
        return
    if clock.status == ClockStatus.FINISHED:
        return

    if state.phase == GamePhase.GAME_OVER:
        finish_game_clock(clock, at_ms)
        return

    if game.pending_override is not None:
        # A 120s override negotiation is not the active player's doing:
        # pause every personal clock while the proposal is open. Deliberate
        # departure from the rollback() precedent of never refunding time.
        activate_clocks(clock, None, request_id=None, now_ms=at_ms)
        return

    if (clock.turn_round, clock.turn_number) != (state.round, state.turn):
        completed_automatically = not clock.human_action_seen_this_turn
        clock.consecutive_automatic_turns = (
            clock.consecutive_automatic_turns + 1 if completed_automatically else 0
        )
        if (
            config.automatic_turn_limit > 0
            and clock.consecutive_automatic_turns >= config.automatic_turn_limit
        ):
            suspend_game_clock_for_inactivity(clock, at_ms)
            return

    begin_shared_turn(
        clock,
        config,
        round_number=state.round,
        turn_number=state.turn,
        now_ms=at_ms,
    )

    if state.phase == GamePhase.PLANNING:
        active: list[str] = []
        for hero_id, player_clock in clock.players.items():
            complete = _planning_complete(game, hero_id)
            player_clock.planning_complete = complete
            if not complete and not player_clock.planning_locked_by_timeout:
                active.append(hero_id)
        activate_clocks(
            clock,
            ClockKind.PLANNING,
            active,
            request_id=None,
            now_ms=at_ms,
        )
        return

    request = game.last_result.input_request if game.last_result else None
    if state.phase == GamePhase.LEVEL_UP:
        begin_level_up(clock, config, round_number=state.round)
        pending = [
            str(hero_id) for hero_id in state.pending_upgrades if str(hero_id) in clock.players
        ]
        activate_clocks(
            clock,
            ClockKind.LEVEL_UP,
            pending,
            request_id=request.id if request else None,
            now_ms=at_ms,
        )
        return

    if state.phase != GamePhase.RESOLUTION or request is None:
        activate_clocks(clock, None, request_id=None, now_ms=at_ms)
        return

    if request.player_id.startswith("team:"):
        eligible = _team_hero_ids(game, request.player_id)
        grant_response_time(clock, config, request.id, eligible)
        running = [
            hero_id
            for hero_id in eligible
            if hero_id in clock.players
            and usable_time_ms(clock.players[hero_id], ClockKind.RESPONSE) > 0
        ]
        # Keep an exhausted team request active once nobody has time left so
        # the coordinator can apply its single shared fallback immediately.
        active = running if running else eligible
        activate_clocks(
            clock,
            ClockKind.RESPONSE,
            active,
            request_id=request.id,
            now_ms=at_ms,
            decision_hero_ids=eligible,
        )
        return

    target = request.player_id
    if target not in clock.players:
        activate_clocks(clock, None, request_id=request.id, now_ms=at_ms)
        return
    owner = str(state.resolution_owner_id) if state.resolution_owner_id is not None else None
    is_response = owner is not None and owner != target
    kind = ClockKind.RESPONSE if is_response else ClockKind.RESOLUTION
    if is_response:
        grant_response_time(clock, config, request.id, [target])
    elif owner == target:
        # Only a hero prompted during their own turn may claim the one-shot
        # bonus. Turn-boundary prompts (expiring-effect finishing steps) run
        # with no resolution owner and must not consume it.
        grant_initiative_bonus(clock, config, target)
    activate_clocks(clock, kind, [target], request_id=request.id, now_ms=at_ms)


def _timeout_selection(request: InputRequest, rng: random.Random) -> Any:
    if request.request_type in (
        InputRequestType.CHOOSE_ACTION,
        InputRequestType.ACTION_CHOICE,
    ):
        hold = next((option for option in request.options if option.id.upper() == "HOLD"), None)
        if hold is not None:
            return selection_value(hold)
    if request.can_skip:
        return SKIP
    explicit_pass = next(
        (option for option in request.options if option.id.upper() == "PASS"),
        None,
    )
    if explicit_pass is not None:
        return selection_value(explicit_pass)
    if not request.options:
        raise ValueError(f"Timed request {request.id} has no legal timeout fallback")
    return selection_value(rng.choice(request.options))


def _timer_event(
    *,
    hero_id: str,
    kind: ClockKind,
    action: str,
    request_id: str | None = None,
    selection: Any = None,
    card_id: str | None = None,
    team_id: str | None = None,
    eligible_hero_ids: Iterable[str] = (),
) -> GameEvent:
    metadata: dict[str, Any] = {"clock_kind": kind.value, "automatic_action": action}
    if request_id is not None:
        metadata["request_id"] = request_id
    if selection is not None:
        metadata["selection"] = selection
    if card_id is not None:
        metadata["card_id"] = card_id
    if team_id is not None:
        metadata["team"] = team_id
        metadata["eligible_hero_ids"] = list(eligible_hero_ids)
    return GameEvent(
        event_type=GameEventType.TIMER_EXPIRED,
        actor_id=hero_id,
        metadata=metadata,
    )


def _record_timeout(
    game: ManagedGame,
    *,
    action: str,
    hero_id: str,
    card_id: str | None = None,
    selection: Any = None,
    request_id: str | None = None,
    team_id: str | None = None,
    eligible_hero_ids: Iterable[str] = (),
) -> None:
    recorder = game.replay_recorder
    if recorder is None:
        return
    state = game.session.state
    recorder.record_timer_timeout(
        action=action,
        hero_id=hero_id,
        round_num=state.round,
        turn=state.turn,
        card_id=card_id,
        selection=selection,
        request_id=request_id,
        team_id=team_id,
        eligible_hero_ids=list(eligible_hero_ids),
    )


def _store_result(game: ManagedGame, result: SessionResult) -> None:
    game.last_result = result
    if not game.game_logger:
        return
    state = game.session.state
    game.game_logger.log_phase_change(result.current_phase.value, state.round, state.turn)
    if result.events:
        game.game_logger.log_events([event.model_dump() for event in result.events])
    if result.input_request:
        game.game_logger.log_input_request(result.input_request.to_dict())
    if result.winner:
        game.game_logger.log_game_over(result.winner)


def _apply_planning_timeout(
    game: ManagedGame,
    hero_id: str,
    rng: random.Random,
) -> list[GameEvent]:
    state = game.session.state
    clock = state.clock
    assert clock is not None
    clock.players[hero_id].planning_locked_by_timeout = True
    hid = HeroID(hero_id)
    events: list[GameEvent] = []

    if hid in state.pending_inputs:
        action = "finish_planning"
        _record_timeout(game, action=action, hero_id=hero_id)
        events.append(_timer_event(hero_id=hero_id, kind=ClockKind.PLANNING, action=action))
        result = game.session.finish_planning(hid)
    else:
        hero = state.get_hero(hid)
        if hero is None:
            raise ValueError(f"Timed hero {hero_id!r} not found")
        if hero.hand:
            card = rng.choice(hero.hand)
            action = "commit"
            _record_timeout(game, action=action, hero_id=hero_id, card_id=card.id)
            events.append(
                _timer_event(
                    hero_id=hero_id,
                    kind=ClockKind.PLANNING,
                    action=action,
                    card_id=card.id,
                )
            )
            result = game.session.commit_card(hid, card)
            if state.phase == GamePhase.PLANNING and planning_open_for_second_card(state, hid):
                _record_timeout(game, action="finish_planning", hero_id=hero_id)
                result = game.session.finish_planning(hid)
        else:
            action = "pass"
            _record_timeout(game, action=action, hero_id=hero_id)
            events.append(_timer_event(hero_id=hero_id, kind=ClockKind.PLANNING, action=action))
            result = game.session.pass_turn(hid)

    _store_result(game, result)
    events.extend(result.events)
    return events


def _apply_input_timeout(
    game: ManagedGame,
    request: InputRequest,
    kind: ClockKind,
    actor_id: str,
    rng: random.Random,
) -> list[GameEvent]:
    selection = _timeout_selection(request, rng)
    team_id = request.player_id if request.player_id.startswith("team:") else None
    eligible_hero_ids = _team_hero_ids(game, request.player_id) if team_id else []
    _record_timeout(
        game,
        action="input",
        hero_id=actor_id,
        selection=selection,
        request_id=request.id,
        team_id=team_id,
        eligible_hero_ids=eligible_hero_ids,
    )
    event = _timer_event(
        hero_id=actor_id,
        kind=kind,
        action="input",
        request_id=request.id,
        selection=selection,
        team_id=team_id,
        eligible_hero_ids=eligible_hero_ids,
    )
    # An automatic Resolution/Response choice is final for this action. Reuse
    # the engine's existing freeze signal so chained steps cannot expose a
    # rollback that would reopen or reroll an already timed-out decision.
    game.session.state.execution_context["rollback_frozen"] = True
    game.session._rollback_snapshot = None
    game.session._rollback_actor_id = None
    result = game.session.advance(InputResponse(request_id=request.id, selection=selection))
    _store_result(game, result)
    return [event, *result.events]


def _upgrade_card_ids(request: InputRequest, hero_id: str) -> list[str]:
    players = request.context.get("players", {})
    hero_data = players.get(hero_id, {}) if isinstance(players, dict) else {}
    options = hero_data.get("options", []) if isinstance(hero_data, dict) else []
    return [
        str(card_id)
        for option in options
        if isinstance(option, dict)
        for card_id in option.get("pair", [])
    ]


def _apply_level_up_timeout(
    game: ManagedGame,
    hero_id: str,
    rng: random.Random,
) -> list[GameEvent]:
    events: list[GameEvent] = []
    # One allowance covers every mandatory upgrade this phase. Once it expires,
    # finish all remaining choices for this player immediately.
    while HeroID(hero_id) in game.session.state.pending_upgrades:
        request = game.last_result.input_request if game.last_result else None
        if request is None:
            raise ValueError("Level Up timeout has no pending upgrade request")
        legal = _upgrade_card_ids(request, hero_id)
        if not legal:
            raise ValueError(f"Level Up timeout has no legal upgrade for {hero_id}")
        selection = {"hero_id": hero_id, "card_id": rng.choice(legal)}
        _record_timeout(game, action="input", hero_id=hero_id, selection=selection)
        events.append(
            _timer_event(
                hero_id=hero_id,
                kind=ClockKind.LEVEL_UP,
                action="input",
                request_id=request.id,
                selection=selection,
            )
        )
        result = game.session.advance(InputResponse(request_id=request.id, selection=selection))
        _store_result(game, result)
        events.extend(result.events)
    return events


def apply_due_timeouts(
    game: ManagedGame,
    at_ms: int,
    *,
    rng: random.Random | None = None,
) -> list[GameEvent]:
    """Settle elapsed time and apply every decision due at ``at_ms``."""
    clock = game.session.state.clock
    if clock is None or clock.status != ClockStatus.RUNNING:
        return []
    chooser = rng or random.SystemRandom()
    emitted: list[GameEvent] = []

    for _ in range(100):
        settle_clock(clock, at_ms)
        kind = clock.active_kind
        exhausted = exhausted_active_hero_ids(clock)
        if kind is None or not exhausted:
            break

        if kind == ClockKind.PLANNING:
            for hero_id in exhausted:
                if game.session.state.phase != GamePhase.PLANNING:
                    break
                emitted.extend(_apply_planning_timeout(game, hero_id, chooser))
                reconcile_game_clock(game, at_ms)
            if clock.active_kind != ClockKind.PLANNING:
                break
            continue

        if kind == ClockKind.LEVEL_UP:
            for hero_id in exhausted:
                if HeroID(hero_id) in game.session.state.pending_upgrades:
                    emitted.extend(_apply_level_up_timeout(game, hero_id, chooser))
                    reconcile_game_clock(game, at_ms)
            if clock.active_kind != ClockKind.LEVEL_UP:
                break
            continue

        request = game.last_result.input_request if game.last_result else None
        if request is None or request.id != clock.active_request_id:
            reconcile_game_clock(game, at_ms)
            continue

        if kind == ClockKind.RESPONSE and request.player_id.startswith("team:"):
            eligible = _team_hero_ids(game, request.player_id)
            all_exhausted = all(
                hero_id not in clock.players
                or usable_time_ms(clock.players[hero_id], ClockKind.RESPONSE) == 0
                for hero_id in eligible
            )
            if not all_exhausted:
                reconcile_game_clock(game, at_ms)
                break
            actor_id = str(game.session.state.resolution_owner_id or request.player_id)
        else:
            actor_id = request.player_id

        emitted.extend(_apply_input_timeout(game, request, kind, actor_id, chooser))
        reconcile_game_clock(game, at_ms)
        next_request = game.last_result.input_request if game.last_result else None
        if (
            game.session.state.phase != GamePhase.RESOLUTION
            or next_request is None
            or next_request.player_id != request.player_id
        ):
            # Yield when ownership changes. A zero-valued match schedules the
            # next target at delay 0 instead of monopolizing one reconciliation.
            break
    else:
        raise RuntimeError("Timer timeout processing exceeded its safety limit")

    return emitted


def prepare_timed_mutation(
    game: ManagedGame,
    at_ms: int | None = None,
    *,
    registry: GameRegistry | None = None,
) -> list[GameEvent]:
    """Charge elapsed time before accepting a client mutation."""
    clock = game.session.state.clock
    if clock is not None and clock.status in {
        ClockStatus.WAITING_FOR_PLAYERS,
        ClockStatus.SUSPENDED_FOR_INACTIVITY,
    }:
        if clock.status == ClockStatus.SUSPENDED_FOR_INACTIVITY:
            raise ValueError("Match suspended for inactivity; every player must ready again")
        raise ValueError("Waiting for every player to be ready before starting the timed match")
    timestamp = now_ms() if at_ms is None else at_ms
    events = apply_due_timeouts(game, timestamp)
    reconcile_game_clock(game, timestamp)
    # A subsequent client validation error must not leave an already-applied
    # timeout unpersisted or without a replacement deadline task.
    if events and registry is not None:
        finalize_timed_mutation(game, registry, timestamp)
    return events


def client_decision_timed_out(
    timer_events: Iterable[GameEvent],
    *,
    hero_id: str,
    request_id: str | None = None,
) -> bool:
    """Whether this exact client decision lost an atomic deadline race."""
    for event in timer_events:
        if event.event_type != GameEventType.TIMER_EXPIRED:
            continue
        timed_request_id = event.metadata.get("request_id")
        if request_id is not None and timed_request_id == request_id:
            return True
        if (
            request_id is None
            and event.actor_id == hero_id
            and event.metadata.get("clock_kind") == ClockKind.PLANNING.value
        ):
            return True
    return False


def stop_clock_for_accepted_decision(
    game: ManagedGame,
    *,
    hero_id: str,
    request_id: str | None = None,
    completes_planning: bool = False,
) -> None:
    """Pause active clocks at receipt while authoritative server work runs.

    The resulting phase/request is reconciled at completion, so backend compute
    time is never charged to a player who could not acquire the game lock.
    """
    clock = game.session.state.clock
    if clock is None or clock.status != ClockStatus.RUNNING or clock.active_kind is None:
        return
    timestamp = clock.last_settled_at_ms if clock.last_settled_at_ms is not None else now_ms()

    if clock.active_kind == ClockKind.PLANNING:
        if not completes_planning:
            return
    else:
        if request_id is None or request_id != clock.active_request_id:
            return
        if hero_id not in clock.active_decision_hero_ids:
            return

    activate_clocks(
        clock,
        clock.active_kind,
        [],
        request_id=clock.active_request_id,
        now_ms=timestamp,
        decision_hero_ids=clock.active_decision_hero_ids,
    )


def mark_human_action(game: ManagedGame) -> None:
    """Record one accepted non-automatic gameplay decision for this turn."""
    clock = game.session.state.clock
    if (
        clock is not None
        and clock.status == ClockStatus.RUNNING
        and not clock.human_action_seen_this_turn
    ):
        clock.human_action_seen_this_turn = True
        clock.revision += 1


def finalize_timed_mutation(
    game: ManagedGame,
    registry: GameRegistry,
    at_ms: int | None = None,
) -> None:
    timestamp = now_ms() if at_ms is None else at_ms
    reconcile_game_clock(game, timestamp)
    registry.save_game(game.game_id)
    schedule_deadline(game, registry)


@asynccontextmanager
async def timed_rest_mutation(
    game: ManagedGame,
    registry: GameRegistry,
) -> AsyncIterator[list[GameEvent]]:
    """Run one REST mutation with failure-safe clock and timeout delivery.

    REST requests can win the game-lock race after a deadline has elapsed but
    before the deadline worker runs. In that case the request applies the
    automatic decision itself and must broadcast the resulting state. The
    ``finally`` also guarantees that a clock paused for backend processing is
    reconciled and rescheduled when the requested game action is rejected.
    """
    messages = []
    async with game.outbound_lock:
        try:
            async with game.lock:
                timer_events = prepare_timed_mutation(game)
                try:
                    yield timer_events
                finally:
                    finalize_timed_mutation(game, registry)
                    if timer_events:
                        from goa2.server.ws import _capture_broadcast

                        messages = _capture_broadcast(
                            game,
                            [event.model_dump() for event in timer_events],
                        )
        except Exception:
            if messages:
                from goa2.server.ws import _send_captured_broadcast

                await _send_captured_broadcast(game, messages)
            raise

        if messages:
            from goa2.server.ws import _send_captured_broadcast

            await _send_captured_broadcast(game, messages)


def schedule_deadline(game: ManagedGame, registry: GameRegistry) -> None:
    """Keep one cancellable task for the next authoritative exhaustion."""
    current = asyncio.current_task()
    previous = game.timer_task
    if previous is not None and previous is not current and not previous.done():
        previous.cancel()
    game.timer_task = None

    clock = game.session.state.clock
    if clock is None:
        return
    delay_ms = milliseconds_until_next_exhaustion(clock)
    if delay_ms is None:
        return
    revision = clock.revision
    game.timer_task = asyncio.create_task(
        _deadline_worker(game, registry, revision, delay_ms),
        name=f"game-clock-{game.game_id}",
    )


async def _deadline_worker(
    game: ManagedGame,
    registry: GameRegistry,
    revision: int,
    delay_ms: int,
) -> None:
    try:
        await asyncio.sleep(delay_ms / 1000)
        async with game.outbound_lock:
            async with game.lock:
                clock = game.session.state.clock
                if clock is None or clock.revision != revision:
                    return
                deadline_at = now_ms()
                emitted = apply_due_timeouts(game, deadline_at)
                # Deadline processing is backend work during which no client
                # can acquire the game lock. Re-anchor without spending it.
                clock = game.session.state.clock
                if clock is not None:
                    clock.last_settled_at_ms = now_ms()
                registry.save_game(game.game_id)

                # Import lazily to avoid the WebSocket module importing this
                # coordinator while defining its route handlers.
                from goa2.server.ws import _capture_broadcast

                messages = _capture_broadcast(
                    game,
                    [event.model_dump() for event in emitted] if emitted else None,
                )
                schedule_deadline(game, registry)

            from goa2.server.ws import _send_captured_broadcast

            await _send_captured_broadcast(game, messages)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Clock deadline failed for game %s", game.game_id)


async def resume_timers(registry: GameRegistry) -> None:
    """Resume persisted matches without charging backend downtime."""
    timestamp = now_ms()
    for game in registry.all_games():
        async with game.lock:
            clock = game.session.state.clock
            if clock is None:
                continue
            request = game.last_result.input_request if game.last_result else None
            if (
                request is not None
                and clock.active_kind == ClockKind.RESPONSE
                and clock.active_request_id is not None
                and request.id != clock.active_request_id
                and request.id not in clock.credited_response_request_ids
            ):
                # Input request UUIDs are transport correlation and are rebuilt
                # when a save is restored. Preserve the already-granted logical
                # response window instead of crediting it a second time.
                clock.credited_response_request_ids.append(request.id)
            clock.last_settled_at_ms = timestamp
            reconcile_game_clock(game, timestamp)
            registry.save_game(game.game_id)
            schedule_deadline(game, registry)


async def stop_timers(registry: GameRegistry) -> None:
    """Gracefully settle, persist, and cancel every deadline task."""
    timestamp = now_ms()
    tasks: list[asyncio.Task[None]] = []
    for game in registry.all_games():
        async with game.lock:
            if game.session.state.clock is not None:
                settle_clock(game.session.state.clock, timestamp)
                registry.save_game(game.game_id)
            if game.timer_task is not None:
                game.timer_task.cancel()
                tasks.append(game.timer_task)
                game.timer_task = None
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def merge_timer_events(result: SessionResult, timer_events: Iterable[GameEvent]) -> SessionResult:
    """Attach pre-mutation automatic events to the response/broadcast result."""
    events = list(timer_events)
    if not events:
        return result
    return result.model_copy(update={"events": [*events, *result.events]})
