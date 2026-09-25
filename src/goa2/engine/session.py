"""
Game Session - Phase 2 Engine Self-Containment.

Provides a clean interface for clients to interact with the engine
without touching execution_stack or calling internal functions.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from goa2.domain.events import GameEvent
from goa2.domain.input import InputRequest, InputResponse
from goa2.domain.models import Card, GamePhase, TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID


class SessionResultType(StrEnum):
    INPUT_NEEDED = "INPUT_NEEDED"
    ACTION_COMPLETE = "ACTION_COMPLETE"
    PHASE_CHANGED = "PHASE_CHANGED"
    GAME_OVER = "GAME_OVER"


class SessionResult(BaseModel):
    result_type: SessionResultType
    input_request: InputRequest | None = None
    current_phase: GamePhase
    winner: str | None = None
    events: list[GameEvent] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GameSession:
    """
    High-level orchestrator for game interactions.

    Clients interact exclusively through this class:
    - commit_card() / uncommit_card() / pass_turn() during PLANNING
    - advance() during RESOLUTION and other phases
    """

    def __init__(
        self,
        state: GameState,
        *,
        create_rollback_snapshots: bool = True,
    ) -> None:
        self.state = state
        self._last_phase = state.phase
        self._create_rollback_snapshots = create_rollback_snapshots
        self._rollback_snapshot: dict | None = None
        self._rollback_actor_id: str | None = None

    @property
    def current_phase(self) -> GamePhase:
        return self.state.phase

    def commit_card(
        self,
        hero_id: HeroID,
        card: Card,
        *,
        stop_before_step: Callable[[GameState, Any], bool] | None = None,
    ) -> SessionResult:
        if self.state.phase != GamePhase.PLANNING:
            raise ValueError(f"Cannot commit card in {self.state.phase} phase")
        from goa2.engine.phases import commit_card as _commit_card

        _commit_card(self.state, hero_id, card)
        return self._check_after_planning(stop_before_step=stop_before_step)

    def pass_turn(
        self,
        hero_id: HeroID,
        *,
        stop_before_step: Callable[[GameState, Any], bool] | None = None,
    ) -> SessionResult:
        if self.state.phase != GamePhase.PLANNING:
            raise ValueError(f"Cannot pass in {self.state.phase} phase")
        from goa2.engine.phases import pass_turn as _pass_turn

        _pass_turn(self.state, hero_id)
        return self._check_after_planning(stop_before_step=stop_before_step)

    def uncommit_card(self, hero_id: HeroID) -> SessionResult:
        """Take a committed card back into hand during Planning (LIFO for a
        two-card hero). Rejected once the last commit has fired revelation."""
        if self.state.phase != GamePhase.PLANNING:
            raise ValueError(f"Cannot uncommit card in {self.state.phase} phase")
        from goa2.engine.phases import uncommit_card as _uncommit_card

        _uncommit_card(self.state, hero_id)
        return self._check_after_planning()

    def finish_planning(
        self,
        hero_id: HeroID,
        *,
        stop_before_step: Callable[[GameState, Any], bool] | None = None,
    ) -> SessionResult:
        """Done-signal for a hero who may play two cards (Emmitt's ultimate)
        but chooses to play only one this turn."""
        if self.state.phase != GamePhase.PLANNING:
            raise ValueError(f"Cannot finish planning in {self.state.phase} phase")
        from goa2.engine.phases import finish_planning as _finish_planning

        _finish_planning(self.state, hero_id)
        return self._check_after_planning(stop_before_step=stop_before_step)

    def advance(
        self,
        response: InputResponse | dict[str, Any] | None = None,
        *,
        stop_before_step: Callable[[GameState, Any], bool] | None = None,
    ) -> SessionResult:
        if self.state.phase == GamePhase.PLANNING:
            raise ValueError("Cannot advance() during PLANNING. Use commit_card() or pass_turn().")
        from goa2.engine.handler import process_stack, submit_input

        if response is not None:
            submit_input(self.state, response)

        stack_result = (
            process_stack(self.state)
            if stop_before_step is None
            else process_stack(self.state, stop_before_step=stop_before_step)
        )

        if (
            stack_result.input_request is None
            and self.state.phase == GamePhase.RESOLUTION
            and self.state.current_actor_id is None
            and not self.state.execution_stack
        ):
            raise RuntimeError(
                "RESOLUTION invariant violated: execution stack drained without "
                "selecting an actor or transitioning phase"
            )

        # Snapshot & rollback flag management
        self._manage_rollback(stack_result)

        return self._build_result(stack_result.input_request, events=stack_result.events)

    def rollback(self) -> SessionResult:
        """Rollback to the snapshot anchor set at the owner's most recent
        actionable prompt. Never restores past a foreign player's committed
        decision or a hidden-info reveal — segment boundaries drop the
        pre-boundary snapshot.
        """
        # Defense-in-depth: hidden-event freeze rejects even with a stale
        # snapshot; a pending re-anchor (boundary just fired inside a step)
        # means any surviving snapshot pre-dates the boundary; a mismatched
        # persisted snapshot (belongs to a prior owner) is scrubbed before
        # rejection. All three cases scrub then raise.
        if self.state.execution_context.get("rollback_frozen"):
            raise ValueError("No rollback snapshot available")
        if self.state.execution_context.get("rollback_reanchor_pending"):
            self._rollback_snapshot = None
            self._rollback_actor_id = None
            raise ValueError("No rollback snapshot available")
        if self._rollback_snapshot is None:
            raise ValueError("No rollback snapshot available")
        if not self._snapshot_belongs_to_owner():
            self._rollback_snapshot = None
            self._rollback_actor_id = None
            raise ValueError("No rollback snapshot available")

        live_time_control = self.state.time_control
        live_clock = self.state.clock
        self.state = GameState.model_validate(self._restore_snapshot(self._rollback_snapshot))
        # A rules rollback never refunds elapsed clock time.
        self.state.time_control = live_time_control
        self.state.clock = live_clock
        # Don't clear snapshot — player may rollback again after re-choosing

        from goa2.engine.handler import process_stack

        stack_result = process_stack(self.state)
        # After restore, the first input is the action choice again
        if stack_result.input_request:
            stack_result.input_request.can_rollback = True
        return self._build_result(stack_result.input_request, events=stack_result.events)

    # -- internals --

    def _make_snapshot(self) -> dict:
        """Dump the current state for rollback, excluding the static board.

        The board (tiles, zones, spawn points, lane) is fixed after setup and
        makes up the bulk of a state dump; the only board field that changes in
        play is ``tile.occupant_id``, which is derived from ``entity_locations``
        and rebuilt on load. Excluding it keeps the snapshot small in memory and
        on disk. NOTE: this assumes board geometry/terrain never mutates mid-turn.
        """
        snap = self.state.model_dump(mode="json")
        snap.pop("board", None)
        snap.pop("time_control", None)
        snap.pop("clock", None)
        return snap

    def _restore_snapshot(self, snapshot: dict) -> dict:
        """Re-attach the live (static) board so the snapshot can be validated.

        ``GameState`` requires a board, and validation re-derives tile occupancy
        from the restored ``entity_locations`` via ``rebuild_occupancy_cache``.
        """
        data = dict(snapshot)
        data["board"] = self.state.board.model_dump(mode="json")
        data["time_control"] = (
            self.state.time_control.model_dump(mode="json") if self.state.time_control else None
        )
        data["clock"] = self.state.clock.model_dump(mode="json") if self.state.clock else None
        return data

    def _manage_rollback(self, stack_result) -> None:
        """Take snapshots and set ``can_rollback`` on input requests.

        The "resolution owner" is ``state.resolution_owner_id`` (falling back
        to ``state.current_actor_id``). Behavior on each input request:

        - **Hidden-event freeze** (``execution_context['rollback_frozen']``,
          set by the timer timeout / replay path): drops the snapshot and
          rejects rollback for the rest of the resolution. Auto-skips
          ``ConfirmResolutionStep``.
        - **Boundary re-anchor pending**
          (``execution_context['rollback_reanchor_pending']``, set by a
          foreign prompt or a hidden-info reveal such as mine detonation or
          card-color guess reveal): invalidates any pre-boundary snapshot
          before deciding this pass, so a stale anchor can never be
          restored. An owner actionable prompt will then re-anchor at
          current (post-boundary) state and clear the marker; a
          ``ConfirmResolutionStep`` auto-completes if no re-anchor happened.
        - **Foreign input** (not addressed to owner, not a Hanu remap):
          clears the pre-foreign snapshot at a segment boundary and sets
          the re-anchor pending marker.
        - **Owner actionable input** (addressed to owner, including
          Hanu-remapped ``context['controlled_hero_id'] == owner``): clears
          any pending re-anchor marker and takes a fresh snapshot when none
          exists; advertises ``can_rollback``.
        - **``ConfirmResolutionStep``**: never *creates* a new snapshot,
          only inherits an existing one from a prior actionable prompt.

        The re-anchor pending flag lives in ``execution_context`` and is
        naturally cleared per turn by ``FinalizeHeroTurnStep.context.clear()``.
        """
        from goa2.domain.models.enums import StepType

        if self.state.current_actor_id is None:
            self._rollback_snapshot = None
            self._rollback_actor_id = None
            return

        if stack_result.input_request is None:
            return

        owner_id = self._resolution_owner_id()
        if owner_id is None:
            return

        if self.state.execution_context.get("rollback_frozen"):
            self._rollback_snapshot = None
            self._rollback_actor_id = None
            return

        # A boundary set inside a step (foreign prompt on a prior pass, mine
        # trigger, or card-color reveal) invalidates any pre-boundary snapshot
        # before we consider this pass's anchor. Scrub first, then decide.
        if self.state.execution_context.get("rollback_reanchor_pending"):
            self._rollback_snapshot = None
            self._rollback_actor_id = None

        request = stack_result.input_request
        is_owner_input = self._is_owner_input(request, owner_id)

        # Foreign input: segment boundary. Drop the snapshot and mark
        # re-anchor pending so a trailing confirm-only run auto-completes.
        if not is_owner_input:
            self._rollback_snapshot = None
            self._rollback_actor_id = None
            self.state.execution_context["rollback_reanchor_pending"] = True
            return

        # Owner changed (turn boundary): drop the prior owner's snapshot.
        if self._rollback_actor_id is not None and self._rollback_actor_id != owner_id:
            self._rollback_snapshot = None
            self._rollback_actor_id = None

        # ConfirmResolutionStep never *creates* a snapshot; identify it by
        # the waiting step type rather than by prompt text.
        waiting_step = self.state.execution_stack[-1] if self.state.execution_stack else None
        is_confirm_only = (
            waiting_step is not None and waiting_step.type == StepType.CONFIRM_RESOLUTION
        )

        if self._rollback_snapshot is None:
            if is_confirm_only:
                return
            # Owner actionable prompt: satisfy any pending re-anchor. Search
            # simulations preserve this boundary/control-flow behavior but
            # disable snapshot creation because their sessions are disposable.
            self.state.execution_context.pop("rollback_reanchor_pending", None)
            if not self._create_rollback_snapshots:
                return
            self._rollback_snapshot = self._make_snapshot()
            self._rollback_actor_id = owner_id

        request.can_rollback = True

    # -- shared rollback predicates --

    def _resolution_owner_id(self) -> str | None:
        """Canonical owner id: ``resolution_owner_id`` with
        ``current_actor_id`` as fallback; ``None`` when no resolution is open.
        """
        owner = (
            self.state.resolution_owner_id
            if self.state.resolution_owner_id is not None
            else self.state.current_actor_id
        )
        return str(owner) if owner is not None else None

    @staticmethod
    def _is_owner_input(request: InputRequest, owner_id: str) -> bool:
        """Whether ``request`` is native to the owner — either directly
        addressed, or Hanu-remapped via ``context['controlled_hero_id']``.
        """
        return request.player_id == owner_id or (
            request.context.get("controlled_hero_id") == owner_id
        )

    def reapply_rollback_flag(self, request: InputRequest) -> None:
        """Re-assert ``can_rollback`` on a request re-emitted after reload.

        Shares the eligibility predicate with ``_manage_rollback`` so
        persistence can't drift from live behavior. A persisted snapshot is
        stale if it was taken before an in-step boundary (mine reveal, guess
        reveal, foreign prompt); the ``rollback_reanchor_pending`` marker
        signals that, so we scrub the snapshot and refuse to advertise
        rollback until the next live pass anchors a fresh one.
        """
        if self.state.execution_context.get("rollback_frozen"):
            return
        if self.state.execution_context.get("rollback_reanchor_pending"):
            self._rollback_snapshot = None
            self._rollback_actor_id = None
            return
        if self._rollback_snapshot is None:
            return
        if not self._snapshot_belongs_to_owner():
            return
        owner_id = self._resolution_owner_id()
        if owner_id is None:
            return
        if self._is_owner_input(request, owner_id):
            request.can_rollback = True

    def _snapshot_belongs_to_owner(self) -> bool:
        """Whether the in-memory snapshot's ``_rollback_actor_id`` matches
        the canonical owner. Callers treat a mismatch as "no snapshot"
        (never restore stale state, never advertise rollback).
        """
        if self._rollback_snapshot is None or self._rollback_actor_id is None:
            return False
        owner_id = self._resolution_owner_id()
        return owner_id is not None and self._rollback_actor_id == owner_id

    def _check_after_planning(
        self,
        *,
        stop_before_step: Callable[[GameState, Any], bool] | None = None,
    ) -> SessionResult:
        """After a planning action, check if phase transitioned."""
        if self.state.phase != self._last_phase:
            from goa2.engine.handler import process_stack

            stack_result = (
                process_stack(self.state)
                if stop_before_step is None
                else process_stack(self.state, stop_before_step=stop_before_step)
            )
            self._manage_rollback(stack_result)
            result = self._build_result(stack_result.input_request, events=stack_result.events)
            self._last_phase = self.state.phase
            return result
        return SessionResult(
            result_type=SessionResultType.ACTION_COMPLETE,
            current_phase=self.state.phase,
        )

    def _build_result(
        self,
        request: InputRequest | None = None,
        events: list[GameEvent] | None = None,
    ) -> SessionResult:
        ev = events or []
        if self.state.phase == GamePhase.GAME_OVER:
            return SessionResult(
                result_type=SessionResultType.GAME_OVER,
                current_phase=GamePhase.GAME_OVER,
                winner=self._determine_winner(),
                events=ev,
            )
        if request is not None:
            return SessionResult(
                result_type=SessionResultType.INPUT_NEEDED,
                input_request=request,
                current_phase=self.state.phase,
                events=ev,
            )
        if self.state.phase != self._last_phase:
            result = SessionResult(
                result_type=SessionResultType.PHASE_CHANGED,
                current_phase=self.state.phase,
                events=ev,
            )
            self._last_phase = self.state.phase
            return result
        return SessionResult(
            result_type=SessionResultType.ACTION_COMPLETE,
            current_phase=self.state.phase,
            events=ev,
        )

    def _determine_winner(self) -> str | None:
        # Authoritative: TriggerGameOverStep records an individual or team winner.
        if self.state.individual_winner_id is not None:
            return str(self.state.individual_winner_id)
        if self.state.winner is not None:
            return self.state.winner.value

        # Fallback: life counter check (annihilation)
        red = self.state.teams.get(TeamColor.RED)
        blue = self.state.teams.get(TeamColor.BLUE)
        if not red or not blue:
            return None
        if red.life_counters <= 0:
            return "BLUE"
        if blue.life_counters <= 0:
            return "RED"
        return None
