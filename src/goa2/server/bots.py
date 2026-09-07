"""Server-side bot coordinator.

One idempotent asynchronous worker per game drives every server-managed AI
hero's decisions through the same engine, persistence, replay, clock, and
broadcast paths as human players.

Invariants (the reason the coordinator exists rather than reusing headless
harness code):

- **One task per game.** Stored in ``ManagedGame.bot_task``.
  :func:`schedule_bot_drive` is safe from any lifecycle site — it is a no-op
  while the previous task is alive, and spawns a fresh one after it exits.
- **Never hold locks during compute.** The worker snapshots the state under
  ``game.lock``, **deep-copies** the pending :class:`SessionResult` /
  :class:`InputRequest` and clones the state, then hands the isolated
  objects to :func:`asyncio.to_thread`. A misbehaving or malicious agent
  cannot mutate live objects, and no lock is held while the CPU work runs.
- **Established lock order on apply.** ``outbound_lock`` → ``game.lock``,
  matching every REST/WS mutation.
- **One decision per locked mutation.** Applied through :class:`GameSession`;
  the engine keeps authority. Stale results are silently dropped after a
  live revalidation that recomputes eligible responders — no half-applied
  mutations, no persistence writes for stale outputs, no phantom broadcasts.
- **Plain ``advance()`` when the bot owns the next work but the engine is
  between requests.** A bot-vs-bot game must be able to resolve mid-turn
  actions without a human/timer nudge. When
  :func:`~automata.runtime.driver.inspect_next_decision` returns ``None``
  during RESOLUTION, the coordinator issues one plain ``session.advance()``
  through the same locked mutation → finalize → save → broadcast path a
  human's REST/WS advance would take. It only exits to a human when the
  live pending request is addressed to a hero/team the game has no bot
  agent for.
- **Recoverable on any failure.** Agent exceptions, illegal engine outputs,
  broadcast errors — the worker logs and exits with the live state
  untouched. Every clock stop is paired with a ``finalize_timed_mutation``
  in ``finally`` so time-control state never drifts.
- **Agent instances live on the ManagedGame.** They are runtime-only (never
  persisted), created lazily with stable game-specific entropy, and cleared
  on ``registry.remove()`` and after every game restore.
- **Live ISMCTS is bounded.** Search runs off-loop with owner-scoped timeouts,
  process-wide concurrency limits, and a per-hero Heuristic fallback.

REST, WebSocket, timer, and restore lifecycle sites call
:func:`schedule_bot_drive` through the shared server mutation seams.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from automata.agents.contracts import Agent
from automata.runtime.clone import clone_state
from automata.runtime.driver import (
    BotDecision,
    DecisionKind,
    IllegalBotDecisionError,
    apply_decision,
    eligible_hero_ids_for_request,
)
from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models import GamePhase
from goa2.domain.state import GameState
from goa2.domain.time_control import ClockStatus
from goa2.domain.types import HeroID
from goa2.engine.phases import planning_open_for_second_card
from goa2.engine.session import GameSession, SessionResult, SessionResultType
from goa2.server import bot_factory, bounded_compute
from goa2.server.registry import GameRegistry, ManagedGame
from goa2.server.time_control import (
    finalize_timed_mutation,
    now_ms,
    set_player_ready,
    stop_clock_for_accepted_decision,
)

logger = logging.getLogger(__name__)

BOT_ACTION_PACING_SECONDS = 0.5

__all__ = [
    "auto_ready_bot_heroes",
    "cancel_all_bot_tasks",
    "schedule_bot_drive",
    "start_bot_lifecycle",
]


# --------------------------------------------------------------------------- #
# Snapshot helpers                                                             #
# --------------------------------------------------------------------------- #


def _snapshot_last_result(
    last_result: SessionResult | None,
) -> SessionResult | None:
    """Deep-copy a :class:`SessionResult` for the background thread.

    ``SessionResult`` is a Pydantic model; ``model_copy(deep=True)`` walks
    the whole graph, including the embedded :class:`InputRequest`, its
    :class:`InputOption` list, and its ``context`` dict. That means any
    agent that (accidentally or maliciously) mutates the received request
    can only mutate its snapshot — the live copy carried on
    ``game.last_result`` and referenced by ``state.input_stack`` is
    untouched.

    ``None`` (game start / non-INPUT_NEEDED) is passed through.
    """
    if last_result is None:
        return None
    return last_result.model_copy(deep=True)


def _snapshot_pending_request(
    last_result: SessionResult | None,
) -> InputRequest | None:
    """Return the pending input request from ``last_result``, if any."""
    if last_result is None:
        return None
    if last_result.result_type is SessionResultType.INPUT_NEEDED:
        return last_result.input_request
    return None


# --------------------------------------------------------------------------- #
# Stale validation                                                             #
# --------------------------------------------------------------------------- #


def _upgrade_still_pending(request: InputRequest, hero_id: str) -> bool:
    """UPGRADE_PHASE: this hero still owes an answer.

    UPGRADE_PHASE puts every player's remaining choice count in
    ``context['players'][hero_id]['remaining']``. A hero that has already
    picked all their upgrades is still listed but with ``remaining == 0``;
    they must not be counted as an eligible responder.
    """
    if request.request_type is not InputRequestType.UPGRADE_PHASE:
        return True
    players_ctx = request.context.get("players") or {}
    info = players_ctx.get(hero_id) or {}
    return int(info.get("remaining", 0)) > 0


def _is_decision_still_valid(
    state: GameState,
    live_last_result: SessionResult | None,
    decision: BotDecision,
    agents: dict[str, Agent],
) -> bool:
    """Whether ``decision`` may still be applied against the live state.

    The coordinator computes on a snapshot; anything can happen between
    snapshot and apply — a human commit, a timer expiry, another bot's
    turn. We must reject any decision that no longer matches the live
    world, without touching persistence.

    PLANNING decisions require:

    - ``state.phase == PLANNING``,
    - the hero still exists,
    - and either an open Emmitt second-card window or the hero is not yet
      in ``pending_inputs`` (a first commit is still valid).

    INPUT decisions require:

    - ``live_last_result`` is INPUT_NEEDED,
    - the live request's ``id`` and ``player_id`` match the decision,
    - ``decision.hero_id`` is *still* an eligible responder to the live
      request (recomputed from live state — team memberships, upgrade
      remaining counts can all have shifted), and
    - ``decision.hero_id`` is still bot-owned in ``agents`` (a mid-flight
      configuration change would otherwise let the coordinator apply a
      decision on behalf of a hero no client currently controls).
    """
    if state.phase == GamePhase.GAME_OVER:
        return False

    if decision.kind is DecisionKind.PLANNING:
        if state.phase != GamePhase.PLANNING:
            return False
        hid = HeroID(decision.hero_id)
        hero = state.get_hero(hid)
        if hero is None:
            return False
        # Bot ownership must still hold (bot_specs cannot legally change
        # mid-game today, but a defensive check keeps the coordinator honest
        # against a future reconfiguration shape change).
        if decision.hero_id not in agents:
            return False
        # Emmitt's second-card commit/finish window?
        if planning_open_for_second_card(state, hid):
            return True
        # First commit: must not already be committed.
        return hid not in state.pending_inputs

    # INPUT — recompute live eligibility.
    request = decision.request
    if request is None:
        return False
    live_request = _snapshot_pending_request(live_last_result)
    if live_request is None:
        return False
    if live_request.id != request.id:
        return False
    if live_request.player_id != request.player_id:
        return False
    # Live eligible responders. Ownership resolution runs against live state
    # so a team-scoped request whose teammate composition shifted (defeats,
    # reshuffles, etc.) is caught here.
    eligible = eligible_hero_ids_for_request(state, live_request)
    if decision.hero_id not in eligible:
        return False
    if decision.hero_id not in agents:
        return False
    # UPGRADE_PHASE: hero must still owe an upgrade.
    return _upgrade_still_pending(live_request, decision.hero_id)


# --------------------------------------------------------------------------- #
# Broadcast / logging / replay helpers                                         #
# --------------------------------------------------------------------------- #


CapturedBroadcast = list[tuple[str | None, Any, dict[str, Any]]]


def _capture_broadcast_for_result(game: ManagedGame, result: SessionResult) -> CapturedBroadcast:
    """Materialize scoped broadcasts for ``result`` while holding the lock."""
    from goa2.server.ws import _capture_broadcast

    events = [ev.model_dump() for ev in result.events] if result.events else None
    return _capture_broadcast(game, events)


async def _send_broadcast(game: ManagedGame, messages: CapturedBroadcast) -> None:
    """Send captured broadcast payloads (already outside ``game.lock``)."""
    if not messages:
        return
    from goa2.server.ws import _send_captured_broadcast

    await _send_captured_broadcast(game, messages)


def _log_result(game: ManagedGame, result: SessionResult) -> None:
    """Same shape :func:`goa2.server.routes_games._log_result` uses."""
    gl = game.game_logger
    if gl is None:
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
        if game.replay_recorder is not None:
            # Keep this import at the lifecycle edge: replay imports the
            # server's heavy-worker machinery, while bots are imported by the
            # application factory during startup.
            from goa2.server.replay import verify_replay_in_background

            verify_replay_in_background(str(game.replay_recorder.path), game.game_id)


def _record_replay(
    game: ManagedGame,
    decision: BotDecision,
    rec_round: int,
    rec_turn: int,
) -> None:
    """Append the applied bot decision to the replay recorder.

    The replay actor is always the decision-making hero (``decision.hero_id``),
    never ``request.player_id``. That distinction matters for team-scoped
    (``"team:RED"``) and simultaneous (``"simultaneous"``, UPGRADE_PHASE)
    requests: the replay must record *which* hero of the team actually
    answered so a rebuilt replay drives the identical hero through the
    same seam. This mirrors ``ws._handle_submit_input`` / REST
    ``submit_input`` where ``hero_id`` is the authenticated submitter, not
    the raw request routing address.
    """
    rec = game.replay_recorder
    if rec is None:
        return
    hero_id_str = str(decision.hero_id)
    if decision.kind is DecisionKind.PLANNING:
        plan = decision.planning
        assert plan is not None
        if plan.card is not None:
            rec.record_commit(hero_id_str, plan.card.id, rec_round, rec_turn)
        elif plan.kind.value == "FINISH":
            rec.record_finish_planning(hero_id_str, rec_round, rec_turn)
        else:  # PASS
            rec.record_pass(hero_id_str, rec_round, rec_turn)
        return
    # INPUT: use the decision maker, not request.player_id.
    rec.record_input(hero_id_str, decision.selection, rec_round, rec_turn)


def _log_action_specific(game: ManagedGame, decision: BotDecision) -> None:
    """Mirror the action-specific logger calls REST/WS mutation handlers make.

    Each REST/WS action handler emits an action-specific log line *before*
    the generic :func:`_log_result` produces the phase / events / winner
    entries. Bots must do the same so a game's log stream is identical
    regardless of whether the mutation came from a human client or the
    coordinator. The exact vocabulary mirrors the handlers:

    - COMMIT  → ``log_card_commit(hero_id, card_id)`` (see
      ``ws._handle_commit_card``, ``routes_games.commit_card``).
    - PASS    → ``log_pass_turn(hero_id)`` (see ``ws._handle_pass_turn``,
      ``routes_games.pass_turn``).
    - FINISH  → no dedicated logger call (matching ``_handle_finish_planning``
      / ``routes_games.planning_done``); only ``_log_result`` fires.
    - INPUT   → ``log_input_response(hero_id, selection)`` (see
      ``ws._handle_submit_input``).

    The ``hero_id`` passed to every logger call is the decision maker
    (``decision.hero_id``) — same actor identity used for the replay entry.
    """
    gl = game.game_logger
    if gl is None:
        return
    hero_id_str = str(decision.hero_id)
    if decision.kind is DecisionKind.PLANNING:
        plan = decision.planning
        assert plan is not None
        if plan.card is not None:
            gl.log_card_commit(hero_id_str, plan.card.id)
        elif plan.kind.value == "PASS":
            gl.log_pass_turn(hero_id_str)
        # FINISH has no dedicated log method by design.
        return
    # INPUT: log the response with the submitter identity.
    gl.log_input_response(hero_id_str, decision.selection)


def _freeze_rollback_for_bot_input(game: ManagedGame, decision: BotDecision) -> None:
    """Freeze rollback before applying a bot ``INPUT`` decision.

    An automatic Resolution/Response answer produced by a server-managed bot
    is externally revealed the same way a timer-driven answer is (see
    :func:`time_control._apply_input_timeout`, which sets exactly the same
    flags before its ``session.advance``). Reusing the engine's existing
    freeze signal ensures that chained resolution steps cannot expose a
    rollback that would reopen or reroll an already externally-decided input.

    Only RESOLUTION-phase INPUT decisions freeze — this mirrors the
    time-control coordinator's own policy: :func:`_apply_input_timeout`
    freezes rollback but :func:`_apply_level_up_timeout` and
    :func:`_apply_planning_timeout` do not. PLANNING decisions are
    committed facedown and UPGRADE_PHASE is between turns, so neither has
    a live resolution snapshot to invalidate.
    """
    if decision.kind is not DecisionKind.INPUT:
        return
    if game.session.state.phase != GamePhase.RESOLUTION:
        return
    game.session.state.execution_context["rollback_frozen"] = True
    game.session._rollback_snapshot = None
    game.session._rollback_actor_id = None


def _stop_clock_for_decision(game: ManagedGame, decision: BotDecision) -> None:
    """Pause the appropriate clock at decision acceptance."""
    if decision.kind is DecisionKind.PLANNING:
        stop_clock_for_accepted_decision(
            game,
            hero_id=str(decision.hero_id),
            completes_planning=True,
        )
        return
    request = decision.request
    assert request is not None
    stop_clock_for_accepted_decision(
        game,
        hero_id=str(decision.hero_id),
        request_id=request.id,
    )


# --------------------------------------------------------------------------- #
# Idle progression: plain advance when the bot owes work but no request yet    #
# --------------------------------------------------------------------------- #


def _current_pending_request(
    last_result: SessionResult | None,
) -> InputRequest | None:
    return _snapshot_pending_request(last_result)


def _bot_pacing_delay_seconds(game: ManagedGame) -> float:
    """Presentation delay between bot updates in games with human seats."""
    has_human = any(hero_id not in game.bot_specs for hero_id in game.hero_to_token)
    return BOT_ACTION_PACING_SECONDS if has_human else 0.0


async def _pace_before_next_bot_mutation(game: ManagedGame) -> None:
    # Called only after mutation locks and the preceding broadcast are complete.
    await asyncio.sleep(_bot_pacing_delay_seconds(game))


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #


async def _bot_drive_worker(game: ManagedGame, registry: GameRegistry) -> None:
    """Drive bot decisions until the next work belongs to a human or the game ends.

    See module docstring for invariants. The loop is:

    1. Snapshot under ``game.lock``: clone state, deep-copy last_result,
       cache the current agents mapping. Release the lock immediately.
    2. Off-loop compute via ``asyncio.to_thread``. Isolated inputs; no live
       state visible.
    3. If the driver returned a decision → apply under ``outbound_lock`` →
       ``game.lock`` with live revalidation + finally-clause finalize.
    4. If the driver returned ``None``: if a bot still owes the next
       decision (i.e. the engine is between input requests during
       RESOLUTION and we own it), issue one plain ``session.advance()``
       through the same locked-mutation path. Otherwise exit.
    5. After broadcasting and releasing locks, apply presentation pacing,
       then continue.
    """
    if not game.bot_specs:
        return

    # Safety cap: a runaway coordinator (livelocking engine bug, misbehaving
    # bot pair) must not monopolize the event loop even with ``to_thread``.
    # Real games use <10k iterations end-to-end; 100k is a wide margin that
    # still fails loud rather than silently spinning forever.
    max_iterations = 100_000
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Tombstone check: if the game was removed while we were suspended,
        # exit cleanly. Every subsequent locked section re-checks this so a
        # remove landing mid-iteration halts progress before any side
        # effect leaks past the tombstone.
        if game.removed:
            return
        # Defense-in-depth: even though :func:`schedule_bot_drive` gates
        # spawning on the runnable-state invariant, the state may have
        # transitioned out (SUSPENDED, FINISHED) between spawn and this
        # iteration. Re-check here so a worker started while RUNNING but
        # transitioned mid-flight exits cleanly instead of racing a
        # suspended clock.
        if not _is_runnable_for_bots(game):
            return

        # ------------------------------------------------------------------ #
        # 1. Snapshot under game.lock.
        # ------------------------------------------------------------------ #
        try:
            async with game.lock:
                if game.removed:
                    return
                if not _is_runnable_for_bots(game):
                    return
                if game.session.state.phase == GamePhase.GAME_OVER:
                    return
                cloned_state = clone_state(game.session.state)
                cloned_last_result = _snapshot_last_result(game.last_result)
                # Snapshot the agents mapping too — a mid-drive reconfig of
                # ``bot_specs`` would rebuild ``_bot_agents`` on
                # the next call, but this iteration operates on the map we
                # saw under the lock so the stale-check has a fixed target.
                agents = dict(bot_factory.get_or_build_agents(game))
        except asyncio.CancelledError:
            raise

        # ------------------------------------------------------------------ #
        # 2. Compute decision outside locks, on isolated snapshot objects.
        #    :func:`bounded_compute.bounded_inspect_next_decision` runs on a background
        #    thread (never on the event loop). When any ISMCTS bot is in
        #    ``agents`` it additionally enforces the process-wide semaphore,
        #    queue timeout, and per-decision search timeout — falling back
        #    to a cached HeuristicAgent on any bound violation. Random /
        #    Heuristic bots take the plain ``to_thread`` fast path.
        # ------------------------------------------------------------------ #
        try:
            decision = await bounded_compute.bounded_inspect_next_decision(
                game, cloned_state, agents, cloned_last_result
            )
        except asyncio.CancelledError:
            raise
        except IllegalBotDecisionError as exc:
            logger.error(
                "Bot for game %s produced illegal decision (%s); halting drive",
                game.game_id,
                exc,
            )
            return
        except Exception:
            logger.exception("Bot compute failed for game %s; halting drive", game.game_id)
            return

        if decision is None:
            # No bot-owned decision. Two cases:
            #
            # (a) Live pending input is addressed to a human / unmapped
            #     hero: exit and let the eventual human mutation reschedule
            #     us. Do NOT advance() blindly — that could consume an
            #     engine step that belongs to a person.
            #
            # (b) No live pending input but the phase is not PLANNING and
            #     not GAME_OVER: the engine has more stack work but hasn't
            #     surfaced the next request. A bot must nudge the engine
            #     with a plain ``session.advance()`` so a bot-vs-bot game
            #     can resolve mid-turn actions without an external caller.
            progressed = await _maybe_plain_advance(game, registry, agents)
            if not progressed:
                return
            await _pace_before_next_bot_mutation(game)
            continue

        # ------------------------------------------------------------------ #
        # 3. Apply under outbound_lock → game.lock.
        # ------------------------------------------------------------------ #
        applied = await _apply_bot_decision(game, registry, decision, agents)
        if applied is None:
            # Stale or failed apply: no side effects landed. Try again on
            # the next scheduled iteration if the caller reschedules; the
            # current worker exits so we don't loop tightly on a broken
            # agent.
            return

        # Terminal state: exit immediately, no further work.
        if applied.result_type is SessionResultType.GAME_OVER:
            return

        # ------------------------------------------------------------------ #
        # 4. Pace after broadcast and lock release before the next decision.
        # ------------------------------------------------------------------ #
        await _pace_before_next_bot_mutation(game)

    logger.error(
        "Bot drive worker for game %s exceeded iteration safety limit",
        game.game_id,
    )


async def _apply_bot_decision(
    game: ManagedGame,
    registry: GameRegistry,
    decision: BotDecision,
    agents: dict[str, Agent],
) -> SessionResult | None:
    """Apply one :class:`BotDecision` through the standard locked mutation.

    Returns the fresh :class:`SessionResult` on success, or ``None`` if the
    decision was stale, the engine rejected it, or something in the
    log/save/broadcast chain raised. On any failure path the clock is
    guaranteed to be reconciled (``finalize_timed_mutation`` in ``finally``),
    matching the guarantee REST/WS mutations give.
    """
    result: SessionResult | None = None
    messages: CapturedBroadcast = []

    try:
        async with game.outbound_lock:
            async with game.lock:
                if game.removed:
                    # Tombstone landed while we were computing. Do NOT
                    # persist, log, replay, or broadcast — the game is
                    # being torn down and any side effect after remove is
                    # a defect (stale save file resurrection, phantom
                    # STATE_UPDATE to reconnecting clients, etc.).
                    return None
                if not _is_runnable_for_bots(game):
                    # State transitioned out of RUNNING (SUSPENDED /
                    # FINISHED / GAME_OVER) while we were computing.
                    # Drop the decision silently — the coordinator must
                    # not resume a suspended clock or bypass GAME_OVER.
                    return None
                live_state = game.session.state
                if not _is_decision_still_valid(live_state, game.last_result, decision, agents):
                    logger.debug(
                        "Bot decision for game %s stale on apply; dropping",
                        game.game_id,
                    )
                    return None

                rec_round = live_state.round
                rec_turn = live_state.turn
                session: GameSession = game.session

                # Once we call stop_clock_for_accepted_decision, every exit
                # path from here must run finalize_timed_mutation so the
                # paused clock is reconciled + rescheduled. That mirrors the
                # ws._handle_* exception-restore pattern.
                clock_started = False
                try:
                    _stop_clock_for_decision(game, decision)
                    clock_started = True
                    # Externally-revealed automatic response: freeze rollback
                    # before applying, matching time_control's timeout path.
                    _freeze_rollback_for_bot_input(game, decision)
                    try:
                        result = apply_decision(session, decision)
                    except Exception:
                        logger.exception(
                            "Applying bot decision on game %s raised; halting drive",
                            game.game_id,
                        )
                        return None

                    game.last_result = result
                    try:
                        _record_replay(game, decision, rec_round, rec_turn)
                        _log_action_specific(game, decision)
                        _log_result(game, result)
                    except Exception:
                        logger.exception(
                            "Bot decision recorded but log/replay raised for game %s",
                            game.game_id,
                        )
                        # The engine already mutated; we still want to
                        # broadcast and finalize so clients see the truth.

                    try:
                        messages = _capture_broadcast_for_result(game, result)
                    except Exception:
                        logger.exception(
                            "Bot decision applied but broadcast capture failed for game %s",
                            game.game_id,
                        )
                        messages = []
                finally:
                    if clock_started:
                        # save_game + reconcile_game_clock + schedule_deadline.
                        # Must run even if apply/log/broadcast raised.
                        try:
                            finalize_timed_mutation(game, registry)
                        except Exception:
                            logger.exception(
                                "finalize_timed_mutation failed for bot mutation on game %s",
                                game.game_id,
                            )

            # game.lock released here — outbound_lock still held for send.
            await _send_broadcast(game, messages)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Unexpected error applying bot decision for game %s; halting drive",
            game.game_id,
        )
        return None

    return result


async def _maybe_plain_advance(
    game: ManagedGame,
    registry: GameRegistry,
    agents: dict[str, Agent],
) -> bool:
    """Nudge the engine with one plain ``session.advance()`` if a bot owes work.

    Called only when :func:`inspect_next_decision` returned ``None`` — i.e.
    the driver saw nothing decision-shaped to do. Two shapes:

    - Live pending input is addressed to a hero/team where **no** bot is
      eligible → exit. The next lifecycle event (human mutation, timer,
      restore) will reschedule us.
    - Otherwise (RESOLUTION mid-turn with no pending input, or PLANNING
      already exhausted) → issue one plain ``advance()`` under
      ``outbound_lock`` → ``game.lock``, save + broadcast + reschedule the
      deadline, and return ``True`` so the caller loops to compute the
      next decision.

    PLANNING with ``None`` from the driver means every remaining planning
    slot is human-owned; we exit rather than spin.

    Returns ``True`` if an ``advance()`` was applied, ``False`` if the
    worker should exit.
    """
    result: SessionResult | None = None
    messages: CapturedBroadcast = []

    try:
        async with game.outbound_lock:
            async with game.lock:
                if game.removed:
                    return False
                if not _is_runnable_for_bots(game):
                    # State transitioned out of RUNNING mid-flight; do
                    # not nudge the engine forward on a suspended or
                    # finished game.
                    return False
                state = game.session.state
                phase = state.phase

                if phase == GamePhase.GAME_OVER:
                    return False

                # If there's a live pending input, the driver's None means
                # nobody bot-owned answers it. Exit — a human owes the reply.
                pending = _current_pending_request(game.last_result)
                if pending is not None:
                    return False

                # PLANNING with no pending input: the driver has determined
                # every remaining planning slot is human. Exit.
                if phase == GamePhase.PLANNING:
                    return False

                # RESOLUTION (or another non-planning, non-terminal phase)
                # with no pending input — the engine has more stack work to
                # do. Nudge it exactly one step so the next iteration sees
                # a fresh request / phase.
                #
                # We only nudge when it plausibly leads to bot work. In a
                # mixed human/bot game, an advance() might expose an input
                # for a human — that's still fine: on the *next* loop
                # iteration inspect_next_decision returns None with a
                # pending input, and this function exits (case above).
                _ = agents  # agents set is used indirectly by the caller
                try:
                    result = game.session.advance()
                except Exception:
                    logger.exception(
                        "session.advance() failed during bot idle progression for game %s",
                        game.game_id,
                    )
                    return False

                game.last_result = result
                try:
                    _log_result(game, result)
                except Exception:
                    logger.exception("Bot idle-advance log failed for game %s", game.game_id)
                try:
                    messages = _capture_broadcast_for_result(game, result)
                except Exception:
                    logger.exception(
                        "Bot idle-advance broadcast capture failed for game %s",
                        game.game_id,
                    )
                    messages = []

                try:
                    finalize_timed_mutation(game, registry)
                except Exception:
                    logger.exception(
                        "finalize_timed_mutation failed for idle-advance on game %s",
                        game.game_id,
                    )

            await _send_broadcast(game, messages)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Unexpected error during bot idle-advance for game %s", game.game_id)
        return False

    if result is None:
        return False
    # A GAME_OVER result means "no further work" — caller checks and exits.
    return result.result_type is not SessionResultType.GAME_OVER


# --------------------------------------------------------------------------- #
# Scheduler                                                                    #
# --------------------------------------------------------------------------- #


def _is_runnable_for_bots(game: ManagedGame) -> bool:
    """Whether ``game`` is in a state where bot actions may safely run.

    The invariant: bots may only compute/apply against a game that has
    reached ``clock.status == RUNNING`` if it uses time controls, and any
    non-terminal engine phase otherwise. Every early-exit predicate below
    is a *hard* gate — a violation means the coordinator would either race
    an un-anchored clock (F1 class of bug) or apply an action on a game
    the registry has explicitly ended or removed.

    Rules:

    - **Removed** → ``False``. The registry has evicted the game;
      side effects after removal are always defects.
    - **GAME_OVER** → ``False``. The engine has finalized; further
      mutations are illegal.
    - **No clock (un-timed)** → ``True``. Un-timed matches have no
      readiness handshake; bots may act freely.
    - **Clock ``WAITING_FOR_PLAYERS``** → ``False``. The match has not
      begun; even bots whose ready flag is set must wait for every
      player (typically a human) to ready-up. Any bot compute here
      would race the eventual clock-start reconciliation.
    - **Clock ``SUSPENDED_FOR_INACTIVITY``** → ``False``. The match
      voluntarily paused; bots must not resume it unilaterally.
    - **Clock ``FINISHED``** → ``False``. The clock has ended.
    - **Clock ``RUNNING``** → ``True``.
    """
    if game.removed:
        return False
    if game.session.state.phase == GamePhase.GAME_OVER:
        return False
    clock = game.session.state.clock
    if clock is None:
        return True
    return clock.status == ClockStatus.RUNNING


def schedule_bot_drive(game: ManagedGame, registry: GameRegistry) -> None:
    """Ensure a single bot worker is running (or about to run) for ``game``.

    Idempotent: a call while a previous worker is still alive is a no-op.
    Once the previous worker finishes, the next call spawns a fresh task.

    Callers: game creation, REST/WS mutation
    completion, timer completion, and application restore. Every seam that
    can hand control to a bot should call this — it costs nothing when
    already scheduled and is the only correct way to start bot progress.

    Safe to call outside an active asyncio task if a loop is running (uses
    :func:`asyncio.create_task`). The caller must be inside an event loop.

    **Runnable-state gate.** A single :func:`_is_runnable_for_bots`
    predicate short-circuits the schedule for timed games that have not
    yet reached ``RUNNING`` (``WAITING_FOR_PLAYERS`` /
    ``SUSPENDED_FOR_INACTIVITY``), for finished/terminated games, and for
    removed games. Un-timed games and running timed games proceed. This
    is the central invariant: every REST/WS/timer/lifecycle scheduling
    seam inherits the gate automatically because they all go through
    this function.
    """
    if not game.bot_specs:
        return
    # Central runnable-state gate. See :func:`_is_runnable_for_bots`.
    # Never schedule for a game that has been removed, has ended, or
    # (for timed matches) has not yet reached ``RUNNING``.
    if not _is_runnable_for_bots(game):
        return

    existing = game.bot_task
    if existing is not None and not existing.done():
        return

    task = asyncio.create_task(
        _bot_drive_worker(game, registry),
        name=f"bot-drive-{game.game_id}",
    )
    game.bot_task = task

    def _clear_reference(t: asyncio.Task[None]) -> None:
        if game.bot_task is t:
            game.bot_task = None
        if not t.cancelled():
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "Bot drive task for game %s ended with exception: %r",
                    game.game_id,
                    exc,
                )

    task.add_done_callback(_clear_reference)


# --------------------------------------------------------------------------- #
# Lifecycle hooks                                                              #
# --------------------------------------------------------------------------- #


async def start_bot_lifecycle(
    game: ManagedGame,
    registry: GameRegistry,
    *,
    at_ms: int | None = None,
) -> bool:
    """One-call lifecycle seam for creation and restore.

    Auto-readies every bot hero on a timed match under ``outbound_lock`` →
    ``game.lock``, then — if the ready transition actually started the
    clock — runs the standard finalize (persist + reconcile + schedule the
    initial deadline task) and broadcasts a scoped state update before
    scheduling the bot coordinator. This closes the window where a bot
    could compute against a not-yet-persisted / not-yet-anchored clock.

    Returns whether the ready transition started the clock. Non-timed and
    already-running matches return ``False``. Un-bot games short-circuit
    to ``False`` before acquiring any lock.

    Callers (``create_game`` and the ``lifespan`` restore hook) must not
    call :func:`auto_ready_bot_heroes` + :func:`schedule_bot_drive`
    separately — that pattern skips persistence and broadcast between the
    two calls, which is exactly what led to bots computing against an
    un-anchored clock. This helper is the single blessed seam.
    """
    if not game.bot_specs:
        return False
    if game.removed:
        return False

    started = False
    messages: list[Any] = []
    async with game.outbound_lock:
        async with game.lock:
            if game.removed:
                return False
            started = auto_ready_bot_heroes(game, at_ms=at_ms)
            if started:
                # A ready transition started the clock: persist + reconcile
                # + schedule the initial deadline task before any bot code
                # runs. Otherwise the bot would race a not-yet-scheduled
                # authoritative deadline.
                finalize_timed_mutation(game, registry, at_ms)
                # Materialize a scoped broadcast so every connected client
                # observes the clock transitioning to RUNNING before the
                # first bot mutation lands. Timer events are not attached
                # because no ``TIMER_EXPIRED`` fired — this is a ready
                # transition, not a timeout.
                from goa2.server.ws import _capture_broadcast

                messages = _capture_broadcast(game)

        if messages:
            from goa2.server.ws import _send_captured_broadcast

            await _send_captured_broadcast(game, messages)

    # Schedule the bot task after the ready transition has been persisted
    # and broadcast (or after auto-ready was a no-op on an un-timed game).
    # ``schedule_bot_drive`` still guards against ``game.removed``.
    schedule_bot_drive(game, registry)
    return started


def auto_ready_bot_heroes(game: ManagedGame, at_ms: int | None = None) -> bool:
    """Mark every bot-owned hero as ready on a timed match.

    Timed matches start in :class:`ClockStatus.WAITING_FOR_PLAYERS` and only
    leave it once :func:`set_player_ready` has been called for every hero in
    ``clock.players``. A game full of humans + bots would otherwise stall
    forever because no client submits a ready flag on behalf of a bot. This
    helper is idempotent (a hero already listed in ``ready_hero_ids`` is
    left alone) and returns whether the match transitioned to RUNNING as a
    side effect (i.e. this call was the last-required ready signal).

    Safe to call on non-timed games (returns ``False`` — no clock) and on
    games without any bots (returns ``False``).

    Callers that need the ready transition to be *durable* (persisted and
    reconciled with an initial deadline task) must use
    :func:`start_bot_lifecycle` instead, which wraps this helper with the
    proper finalize + broadcast + scheduler chain.
    """
    if not game.bot_specs:
        return False
    clock = game.session.state.clock
    if clock is None:
        return False
    if clock.status not in {
        ClockStatus.WAITING_FOR_PLAYERS,
        ClockStatus.SUSPENDED_FOR_INACTIVITY,
    }:
        return False
    timestamp = at_ms if at_ms is not None else now_ms()
    started = False
    already_ready = set(clock.ready_hero_ids)
    for hero_id in game.bot_specs:
        if hero_id not in clock.players:
            # A bot hero not part of the clock roster cannot be readied;
            # persistence/roster drift protection — log rather than raise so
            # startup does not fail on a stale save.
            logger.warning(
                "auto_ready_bot_heroes: game %s bot hero %s not in clock roster",
                game.game_id,
                hero_id,
            )
            continue
        if hero_id in already_ready:
            continue
        try:
            if set_player_ready(game, hero_id, True, timestamp):
                started = True
        except ValueError:
            logger.exception(
                "auto_ready_bot_heroes: game %s failed to ready bot hero %s",
                game.game_id,
                hero_id,
            )
    return started


async def cancel_all_bot_tasks(
    registry: GameRegistry, *, drain_timeout_seconds: float = 5.0
) -> None:
    """Cancel every game's bot worker task and drain in-flight searches.

    Called from the FastAPI ``lifespan`` shutdown seam so we never leave an
    orphan bot task running after the app stops. Behavior in order:

    1. Cancel every :attr:`ManagedGame.bot_task` and gather them with
       ``return_exceptions=True`` — the coordinator raising
       :class:`asyncio.CancelledError` at shutdown is the expected flow.
    2. **Drain every tracked in-flight bounded-search future WITHOUT
       cancelling them on timeout.** We use :func:`asyncio.wait` (which
       does not cancel its arguments on timeout, unlike
       :func:`asyncio.wait_for` / :func:`asyncio.gather` in a
       ``wait_for``) so a slow executor thread keeps running to its
       natural completion. That preserves the coordinator's core
       invariant: a semaphore slot is released **only** by the
       done-callback that fires when the underlying thread actually
       finishes — never by shutdown cancelling the future out from
       under it.
    3. On drain timeout, log a warning that some futures are still
       pending. They remain in the trackers (:data:`_in_flight_search_futures`
       and per-game :attr:`ManagedGame._bot_search_futures`) so a later
       reader can still observe them; the done-callback will eventually
       remove them when the executor thread completes on its own.
    4. Post-drain, log any leftover tracked futures as a warning — this
       is expected on shutdown-timeout paths and informational otherwise.

    Cancelling an executor future is safe from asyncio's perspective
    (Python sets the "cancelled" flag), but the underlying thread does
    not stop; more importantly, cancellation prevents the
    done-callback from running with the normal ``result()`` path,
    which is exactly the path that releases the semaphore. We must
    not touch cancellation here — the done-callback is authoritative.
    """
    tasks: list[asyncio.Task[None]] = []
    for game in registry.all_games():
        task = game.bot_task
        if task is None or task.done():
            continue
        task.cancel()
        tasks.append(task)
        game.bot_task = None
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Snapshot the in-flight set before awaiting: the done-callback
    # removes entries from ``_in_flight_search_futures`` and the per-game
    # set, so iterating over the live set while it mutates would risk
    # skipping entries. A snapshot handles that safely.
    pending = bounded_compute.pending_compute_futures()
    if pending:
        # ``asyncio.wait`` does NOT cancel its arguments on timeout —
        # this is the critical difference from ``asyncio.wait_for`` /
        # ``asyncio.gather`` wrapped in ``wait_for``. Any future still
        # in ``still_pending`` after the timeout keeps running on its
        # executor thread; its semaphore slot stays held until the
        # thread finishes and the done-callback fires.
        done, still_pending = await asyncio.wait(
            pending,
            timeout=drain_timeout_seconds,
            return_when=asyncio.ALL_COMPLETED,
        )
        if still_pending:
            logger.warning(
                "cancel_all_bot_tasks: %d bounded-search future(s) did not "
                "drain within %.1fs; leaving them running to natural "
                "completion (tracked, not cancelled)",
                len(still_pending),
                drain_timeout_seconds,
            )
        # Surface any exceptions the drained futures raised so a shutdown
        # doesn't silently swallow an agent bug. We only inspect ``done``
        # here — ``still_pending`` futures cannot have a result yet.
        for fut in done:
            if fut.cancelled():
                # Should not happen: we never cancel these futures. Log
                # defensively rather than let a leaked cancel propagate.
                logger.warning(
                    "cancel_all_bot_tasks: drained future was cancelled "
                    "(unexpected — did another caller cancel it?)"
                )
                continue
            exc = fut.exception()
            if exc is not None:
                logger.warning(
                    "cancel_all_bot_tasks: drained future ended with " "exception: %r",
                    exc,
                )

    # Post-drain leftovers: on the timeout path this is expected (the
    # thread is still running). On the happy path it means a
    # done-callback failed to clear the tracker, which is worth a
    # warning but is not a crash.
    remaining = bounded_compute.pending_compute_futures()
    if remaining:
        logger.warning(
            "cancel_all_bot_tasks: %d future(s) still tracked module-wide " "after drain",
            len(remaining),
        )
    for game in registry.all_games():
        if game._bot_search_futures:
            logger.warning(
                "cancel_all_bot_tasks: game %s still tracks %d in-flight "
                "search future(s) after drain",
                game.game_id,
                len(game._bot_search_futures),
            )
