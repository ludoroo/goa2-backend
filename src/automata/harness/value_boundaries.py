"""Observe stable transitions on the actual played trajectory, without pausing it.

This seam deliberately has no target labels or search dependency. Observations
are provisional until the game ends normally; a dataset recorder must discard
its entire game spool when the outcome is censored or execution raises.
"""

from __future__ import annotations

from typing import Any, Protocol

from automata.runtime.value_boundary import (
    StableValueBoundary,
    capture_transition_anchor,
    transition_reached,
)
from goa2.domain.state import GameState


class StableBoundaryObserver(Protocol):
    """Synchronous sink for actual boundary states and the final game status.

    ``state`` is the live mutable game: encode or copy it during the callback.
    ``viewer_hero_ids`` contains the distinct real decision owners since the
    previous boundary. Each viewer gets their own information-safe observation;
    their entitlement must not be replaced by the next actor's private view.
    Observer exceptions intentionally abort execution; discard the game spool
    rather than resuming a state with incomplete session bookkeeping.
    """

    def record_boundary(
        self,
        state: GameState,
        boundary: StableValueBoundary,
        *,
        viewer_hero_ids: tuple[str, ...],
    ) -> None: ...

    def record_outcome(self, *, winner: str | None, rounds: int, reason: str) -> None: ...


class StableBoundaryTracker:
    """Track completed transitions, including boundaries inside one engine call."""

    def __init__(self, state: GameState, observer: StableBoundaryObserver) -> None:
        self._anchor = capture_transition_anchor(state)
        self._observer = observer
        self._viewers: set[str] = set()

    def record_viewer(self, hero_id: str) -> None:
        self._viewers.add(hero_id)

    def observe(self, state: GameState) -> None:
        boundary = transition_reached(self._anchor, state)
        if boundary is None:
            return
        viewers = tuple(sorted(self._viewers))
        self._anchor = capture_transition_anchor(state)
        self._viewers.clear()
        if viewers:
            self._observer.record_boundary(state, boundary, viewer_hero_ids=viewers)

    def visit_step(self, state: GameState, _step: Any) -> bool:
        """Inspect the engine's stop hook, but never stop or add harness ticks."""
        self.observe(state)
        return False
