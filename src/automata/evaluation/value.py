"""Value function seam for the search leaf estimate.

``ValueFn`` is the interface the search calls at a rollout cutoff: given a
non-terminal state and the perspective team, return a *normalized* scalar in
``[-1, 1]`` (higher = better for that team). The search maps this into the
``[0, 1]`` reward range exactly once, via ``(v + 1) / 2``.

Normalizing the ValueFn's own output range (rather than leaving it unbounded
and squashing inside the search) means a learned value model — a tanh head,
say, or an equivalent bounded output — is a byte-for-byte drop-in for the
current :class:`HeuristicValue`. All range/finiteness validation happens once,
in the search, over the normalized output.

Today the only implementation is :class:`HeuristicValue`, wrapping the
hand-weighted :func:`evaluate_state` through a configurable ``tanh(score /
scale)``. A learned value model (Rung 2) becomes a new implementation of this
same protocol — trained on recorded trajectories (Seam 4) over the extracted
features (Seam 2) — and drops in without touching the search loop.
"""

from __future__ import annotations

import math
from typing import Protocol

from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

from .features import evaluate_state


class ValueFn(Protocol):
    """Estimate a state's value from ``team``'s perspective.

    Implementations MUST return a finite scalar in the inclusive range
    ``[-1, 1]``. The search validates this at every leaf and raises
    :class:`ValueError` on a non-finite or out-of-range output.
    """

    def __call__(self, state: GameState, team: TeamColor) -> float: ...


class HeuristicValue:
    """Hand-weighted linear value squashed into ``[-1, 1]`` via ``tanh``.

    Delegates to :func:`evaluate_state` (unbounded score) and applies
    ``tanh(score / scale)`` to normalize it. ``scale`` is the
    order-of-magnitude of a meaningful positional edge — the default
    (``300.0``) preserves the previous ``SearchConfig.value_scale`` semantics.
    """

    def __init__(self, scale: float = 300.0) -> None:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"HeuristicValue scale must be a positive finite float, got {scale!r}")
        self._scale = scale

    def __call__(self, state: GameState, team: TeamColor) -> float:
        return math.tanh(evaluate_state(state, team) / self._scale)
