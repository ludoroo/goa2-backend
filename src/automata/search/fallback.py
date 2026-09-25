"""Fallback implementations for optional search components."""

from __future__ import annotations

from goa2.domain.state import GameState

from .contracts import (
    ComponentInferenceError,
    ComponentUnavailableError,
    LeafEvaluation,
    LeafEvaluator,
    PolicyScores,
    SearchContext,
    SearchPolicy,
)

_FALLBACK_ERRORS = (ComponentUnavailableError, ComponentInferenceError)


class FallbackSearchPolicy:
    def __init__(self, primary: SearchPolicy, fallback: SearchPolicy) -> None:
        self._primary = primary
        self._fallback = fallback

    def score(self, context, state, legal_actions) -> PolicyScores:
        try:
            return self._primary.score(context, state, legal_actions)
        except _FALLBACK_ERRORS:
            return self._fallback.score(context, state, legal_actions)


class FallbackLeafEvaluator:
    def __init__(self, primary: LeafEvaluator, fallback: LeafEvaluator) -> None:
        self._primary = primary
        self._fallback = fallback

    def evaluate(self, context: SearchContext, state: GameState) -> LeafEvaluation:
        try:
            return self._primary.evaluate(context, state)
        except _FALLBACK_ERRORS:
            return self._fallback.evaluate(context, state)


__all__ = ["FallbackLeafEvaluator", "FallbackSearchPolicy"]
