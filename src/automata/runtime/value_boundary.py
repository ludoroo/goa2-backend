"""Stable engine boundaries used by bounded value evaluation.

A stable value boundary is deliberately narrower than an arbitrary pause in
``process_stack``.  It is either the instant immediately before a selected
hero starts resolving, or a fully-settled planning phase.  Prompts, cleanup,
ties, upgrades, reactions, and terminal states are not value boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from goa2.domain.models import GamePhase, StepType
from goa2.domain.state import GameState
from goa2.engine.steps import GameStep

__all__ = [
    "StableTransitionAnchor",
    "StableValueBoundary",
    "StableValueBoundaryKind",
    "capture_transition_anchor",
    "detect_stable_value_boundary",
    "should_stop_before_stable_boundary",
    "transition_reached",
]


class StableValueBoundaryKind(StrEnum):
    """The two settled states at which a position may be valued."""

    ACTOR_READY = "ACTOR_READY"
    PLANNING_READY = "PLANNING_READY"


@dataclass(frozen=True)
class StableValueBoundary:
    """A stable position and the actor, if resolution is about to begin."""

    kind: StableValueBoundaryKind
    round: int
    turn: int
    actor_id: str | None


@dataclass(frozen=True)
class StableTransitionAnchor:
    """Identity of the transition from which a caller is progressing."""

    start_phase: GamePhase
    start_round: int
    start_turn: int
    resolution_owner_id: str | None


def capture_transition_anchor(state: GameState) -> StableTransitionAnchor:
    """Capture the minimum state needed to reject the starting boundary."""
    return StableTransitionAnchor(
        start_phase=state.phase,
        start_round=state.round,
        start_turn=state.turn,
        resolution_owner_id=(
            str(state.resolution_owner_id) if state.resolution_owner_id is not None else None
        ),
    )


def detect_stable_value_boundary(state: GameState) -> StableValueBoundary | None:
    """Describe ``state`` when it is at a stable value boundary.

    ``ACTOR_READY`` is recognized only while the next untouched stack step is
    the selected actor's respawn or card-resolution entry step.  In
    particular, a respawn/card step left on the stack while waiting for input
    (or after input submission) is already *inside* that actor's resolution.

    ``PLANNING_READY`` requires a drained stack and clean planning buffers.
    Engine-created empty-hand auto-passes are allowed; card commitments,
    second cards, and Emmitt's planning-done marker are not.  ``current_actor``
    is intentionally ignored here because end-of-turn/end-of-round finishing
    effects can leave that advisory field stale after genuine planning starts.
    """
    if state.phase is GamePhase.RESOLUTION and state.execution_stack:
        step = state.execution_stack[-1]
        if step.type not in {StepType.RESPAWN_HERO, StepType.RESOLVE_CARD}:
            return None
        if step.pending_request_id is not None or step.pending_input is not None:
            return None

        hero_id = getattr(step, "hero_id", None)
        if hero_id is None:
            return None
        actor_id = str(hero_id)
        if (
            state.current_actor_id is None
            or state.resolution_owner_id is None
            or str(state.current_actor_id) != actor_id
            or str(state.resolution_owner_id) != actor_id
        ):
            return None
        return StableValueBoundary(
            kind=StableValueBoundaryKind.ACTOR_READY,
            round=state.round,
            turn=state.turn,
            actor_id=actor_id,
        )

    if state.phase is not GamePhase.PLANNING or state.execution_stack:
        return None
    if state.pending_second_cards or state.planning_done:
        return None
    for hero_id, card in state.pending_inputs.items():
        hero = state.get_hero(hero_id)
        if card is not None or hero is None or hero.hand:
            return None

    return StableValueBoundary(
        kind=StableValueBoundaryKind.PLANNING_READY,
        round=state.round,
        turn=state.turn,
        actor_id=None,
    )


def transition_reached(
    anchor: StableTransitionAnchor,
    state: GameState,
) -> StableValueBoundary | None:
    """Return the stable boundary beyond ``anchor``, if one was reached.

    Planning decisions must progress to an actor or a later planning turn
    (everyone may pass), not rediscover the same clean planning state.
    An open resolution must hand off to another actor/turn or settle into a
    later planning turn. Actorless system/cleanup
    anchors accept the next genuine actor or planning boundary.
    """
    boundary = detect_stable_value_boundary(state)
    if boundary is None:
        return None

    if anchor.start_phase is GamePhase.PLANNING:
        if boundary.kind is StableValueBoundaryKind.ACTOR_READY or (
            (boundary.round, boundary.turn) > (anchor.start_round, anchor.start_turn)
        ):
            return boundary
        return None

    if anchor.resolution_owner_id is not None:
        if boundary.kind is StableValueBoundaryKind.ACTOR_READY:
            if boundary.actor_id != anchor.resolution_owner_id or (
                (boundary.round, boundary.turn) > (anchor.start_round, anchor.start_turn)
            ):
                return boundary
            return None
        if (boundary.round, boundary.turn) > (anchor.start_round, anchor.start_turn):
            return boundary
        return None

    if boundary.kind is StableValueBoundaryKind.ACTOR_READY:
        return boundary

    if anchor.start_phase is not GamePhase.PLANNING and (
        (boundary.round, boundary.turn) > (anchor.start_round, anchor.start_turn)
        or state.phase is not anchor.start_phase
    ):
        return boundary
    return None


def should_stop_before_stable_boundary(
    anchor: StableTransitionAnchor,
    state: GameState,
    step: GameStep,
) -> bool:
    """``process_stack`` predicate that stops before a reached actor boundary."""
    if not state.execution_stack or state.execution_stack[-1] is not step:
        return False
    return transition_reached(anchor, state) is not None
