"""Classic heuristic implementation of the strict search-policy seam.

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

import math
from collections.abc import Sequence

from goa2.domain.input import selection_value
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.models.card import Card
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

from ..agents.heuristic_agent import HeuristicAgent
from .contracts import LeafEvaluation, PolicyScores, ScoreSemantics, SearchContext
from .node import Key, action_key


class HeuristicPrior:
    """Order decision keys by the heuristic agent's static action scores."""

    def __init__(self, heuristic: HeuristicAgent | None = None) -> None:
        # Seed is irrelevant: the scoring methods used here are RNG-free.
        self._h = heuristic or HeuristicAgent(0)

    def score(
        self, context: SearchContext, state: GameState, legal_actions: Sequence[Key]
    ) -> PolicyScores:
        legal = list(legal_actions)
        scores: dict[Key, float] = {}
        hero = state.get_hero(HeroID(context.current_owner_id))
        if state.phase == GamePhase.PLANNING and hero is not None:
            by_id: dict[Key, Card] = {c.id: c for c in hero.hand}
            scores = {
                key: self._h.score_card(state, hero, card)
                for key in legal
                if (card := by_id.get(key)) is not None
            }
        elif state.input_stack:
            request = state.input_stack[-1]
            for option in request.options:
                key = action_key(selection_value(option))
                if key in legal and key not in scores:
                    scores[key] = self._h.score_option(state, request, option)
        return PolicyScores(
            tuple(legal), tuple(scores.get(key, 0.0) for key in legal), ScoreSemantics.LOGITS
        )


class HeuristicLeafEvaluator:
    """Small, dependency-free material estimate suitable for classic search."""

    def evaluate(self, context: SearchContext, state: GameState) -> LeafEvaluation:
        if state.winner is not None:
            return LeafEvaluation(value=1.0 if state.winner == context.perspective_team else -1.0)
        ours = state.teams[context.perspective_team]
        enemy_color = TeamColor.BLUE if context.perspective_team == TeamColor.RED else TeamColor.RED
        enemy = state.teams[enemy_color]
        material = float(ours.life_counters - enemy.life_counters)
        material += 0.1 * (sum(h.gold for h in ours.heroes) - sum(h.gold for h in enemy.heroes))
        return LeafEvaluation(value=math.tanh(material / 5.0))


__all__ = ["HeuristicLeafEvaluator", "HeuristicPrior"]
