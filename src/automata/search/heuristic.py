"""Classic heuristic policy ordering and deterministic value evaluation.

Progressive widening reveals only some legal actions at tight budgets, so
:class:`HeuristicPrior` ranks the engine-supplied legal keys best-first and
provides optional PUCT scores. It never adds, removes, or rewrites legality and
it never supplies leaf value; those remain the engine and evaluator's jobs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from automata.runtime.value_boundary import detect_stable_value_boundary
from goa2.domain.input import InputRequestType, selection_value
from goa2.domain.models import TeamColor
from goa2.domain.models.card import Card
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

from ..agents.heuristic_agent import HeuristicAgent
from ..decision import DecisionDescriptor
from .contextual_noop import ContextualNoopKind, contextual_noop_shape, request_action_keys
from .contracts import (
    LeafEvaluation,
    PolicyScores,
    ScoreSemantics,
    SearchContext,
    StableValueContext,
)
from .node import Key, action_key
from .public_consequence import PublicConsequenceSnapshot, public_material


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
        decision = context.decision
        if decision.kind == "CARD":
            hero = state.get_hero(HeroID(context.current_owner_id))
            if hero is not None:
                by_id: dict[Key, Card] = {c.id: c for c in hero.hand}
                scores = {
                    key: self._h.score_card(state, hero, card)
                    for key in legal
                    if (card := by_id.get(key)) is not None
                }
        elif decision.kind == "INPUT" and decision.request is not None:
            request = decision.request

            for option in request.options:
                key = action_key(selection_value(option))
                if key in legal and key not in scores:
                    scores[key] = self._h.score_option(state, request, option)
        return PolicyScores(
            tuple(legal), tuple(scores.get(key, 0.0) for key in legal), ScoreSemantics.LOGITS
        )


@dataclass(frozen=True, slots=True)
class _ImmediateEdge:
    """Compact immutable preparation for one privacy-safe public edge."""

    consequence: PublicConsequenceSnapshot
    owner_id: str
    request_type: InputRequestType | None
    action: Any
    executable_respawn: bool = False
    owner_had_board_presence: bool = False


class HeuristicLeafEvaluator:
    """Material estimate with bounded privacy-safe public consequences."""

    recipe_id = "public-consequence-v4"
    immediate_edge_enabled = True
    contextual_root_coverage_enabled = True
    _EDGE_LIMIT = 5.0
    _EPSILON = 1e-6

    def evaluate(self, context: SearchContext, state: GameState) -> LeafEvaluation:
        if state.winner is not None:
            return LeafEvaluation(value=1.0 if state.winner == context.perspective_team else -1.0)
        return self._evaluate_public_material(state, context.perspective_team)

    def evaluate_stable_value(
        self, context: StableValueContext, state: GameState
    ) -> LeafEvaluation:
        """Evaluate public material only at the boundary named by the caller."""
        actual = detect_stable_value_boundary(state)
        if actual is None or actual != context.boundary:
            raise ValueError("stable value boundary does not match the actual game state")
        if state.winner is not None or state.individual_winner_id is not None:
            raise ValueError("terminal states must bypass stable value evaluation")
        return self._evaluate_public_material(state, context.perspective_team)

    def _evaluate_public_material(self, state: GameState, perspective: TeamColor) -> LeafEvaluation:
        return LeafEvaluation(value=math.tanh(self._public_material(state, perspective) / 5.0))

    @staticmethod
    def _public_material(state: GameState, perspective: TeamColor) -> float:
        return public_material(state, perspective)

    def prepare_immediate_edge(
        self,
        context: SearchContext,
        state: GameState,
        decision: DecisionDescriptor,
        action: Any,
    ) -> object:
        request = decision.request
        request_type = request.request_type if request is not None else None
        shape = (
            contextual_noop_shape(decision, request_action_keys(request))
            if request is not None
            else None
        )
        executable_respawn = bool(
            shape is not None
            and shape.kind is ContextualNoopKind.RESPAWN_PASS
            and action != shape.noop_key
        )
        return _ImmediateEdge(
            consequence=PublicConsequenceSnapshot.capture(
                state,
                context.root_viewer_id,
                context.perspective_team,
            ),
            owner_id=context.current_owner_id,
            request_type=request_type,
            action=action,
            executable_respawn=executable_respawn,
            owner_had_board_presence=state.has_board_presence(context.current_owner_id),
        )

    def evaluate_immediate_edge(
        self,
        context: SearchContext,
        state: GameState,
        prepared: object,
    ) -> LeafEvaluation:
        if not isinstance(prepared, _ImmediateEdge):
            raise TypeError("invalid heuristic immediate-edge preparation")
        if prepared.consequence.perspective_team != context.perspective_team:
            raise ValueError("immediate-edge perspective does not match its preparation")
        if prepared.consequence.root_viewer_id != context.root_viewer_id:
            raise ValueError("immediate-edge root viewer does not match its preparation")

        ordinary = self.evaluate(context, state)
        if state.winner is not None:
            return ordinary
        edge = prepared.consequence.edge_units(state)
        if prepared.executable_respawn and self._respawn_completed(context, state, prepared):
            edge += self._EDGE_LIMIT
        edge = max(-self._EDGE_LIMIT, min(self._EDGE_LIMIT, edge))
        if edge == 0.0:
            return ordinary
        base = max(-1.0 + self._EPSILON, min(1.0 - self._EPSILON, ordinary.value))
        return LeafEvaluation(value=math.tanh(math.atanh(base) + edge / self._EDGE_LIMIT))

    @staticmethod
    def _respawn_completed(
        context: SearchContext, state: GameState, prepared: _ImmediateEdge
    ) -> bool:
        request = context.decision.request
        reached_destination_choice = bool(
            request is not None
            and request.request_type is InputRequestType.CHOOSE_RESPAWN_HEX
            and request.player_id == prepared.owner_id
            and request.options
        )
        return bool(
            prepared.request_type is InputRequestType.CHOOSE_RESPAWN
            and prepared.action == "RESPAWN"
            and (
                reached_destination_choice
                or (
                    not prepared.owner_had_board_presence
                    and state.has_board_presence(prepared.owner_id)
                )
            )
        )


__all__ = ["HeuristicLeafEvaluator", "HeuristicPrior"]
