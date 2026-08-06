"""Expansion policy (prior) for ISMCTS.

Progressive widening reveals children a few at a time; with a tight iteration
budget the *order* in which legal actions are revealed is decisive. A policy
ranks a decision's legal keys best-first so promising moves are searched before
junk, and may additionally attach per-key *weights* (prior probabilities /
scores) for a future PUCT-style selection term (Rung 1) or a learned policy
(Rung 3).

The default policy, :class:`HeuristicPrior`, reuses the ``HeuristicAgent`` static
scorers (:meth:`score_card` / :meth:`score_option`), keeping search and the
baseline policy consistent. A policy only affects *ordering/scoring*, never
legality or value — so it can never make the search unsound, only faster to find
good moves. When no policy is supplied, ``ismcts`` falls back to random
expansion order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from goa2.domain.input import selection_value
from goa2.domain.models.card import Card
from goa2.domain.state import GameState

from ..agents.heuristic_agent import HeuristicAgent
from .ismcts import Decision
from .node import Key, action_key


@dataclass(frozen=True)
class PolicyResult:
    """A policy's ranking of a decision's legal keys.

    ``order`` is a best-first permutation of the legal keys (what progressive
    widening consumes). ``weights`` optionally maps each legal key to a prior
    score/probability for selection biasing; ``None`` means "ordering only"
    (today's heuristic prior). The weights slot lets a learned policy or PUCT
    plug in later without changing this signature.
    """

    order: list[Key]
    weights: dict[Key, float] | None = None


class Policy(Protocol):
    """Rank a decision's legal keys (best-first), optionally with weights."""

    def __call__(
        self, state: GameState, decision: Decision, legal: list[Key]
    ) -> PolicyResult: ...


class HeuristicPrior:
    """Order decision keys by the heuristic agent's static action scores."""

    def __init__(self, heuristic: HeuristicAgent | None = None) -> None:
        # Seed is irrelevant: the scoring methods used here are RNG-free.
        self._h = heuristic or HeuristicAgent(0)

    def __call__(
        self, state: GameState, decision: Decision, legal: list[Key]
    ) -> PolicyResult:
        scored = self._scores(state, decision, legal)
        if scored is None:
            return PolicyResult(order=list(legal), weights=None)
        # Stable sort, highest score first; preserves input order on ties so the
        # result stays deterministic.
        order = sorted(legal, key=lambda k: scored.get(k, 0.0), reverse=True)
        return PolicyResult(order=order, weights=scored)

    def _scores(
        self, state: GameState, decision: Decision, legal: list[Key]
    ) -> dict[Key, float] | None:
        if decision.kind == "CARD" and decision.hero is not None:
            by_id: dict[Key, Card] = {c.id: c for c in decision.hero.hand}
            scores: dict[Key, float] = {}
            for k in legal:
                card = by_id.get(k)
                if card is not None:
                    scores[k] = self._h.score_card(state, decision.hero, card)
            return scores
        if decision.kind == "INPUT" and decision.request is not None:
            request = decision.request
            scores = {}
            for opt in request.options:
                key = action_key(selection_value(opt))
                if key in scores:
                    continue
                scores[key] = self._h.score_option(state, request, opt)
            # SKIP (if legal) has no option object; leave it at the default 0.0
            # so concrete positive-scoring actions are revealed ahead of it.
            return scores
        return None
