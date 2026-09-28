"""Fallback implementations for optional search components."""

from __future__ import annotations

from dataclasses import dataclass, replace

from goa2.domain.state import GameState

from .contracts import (
    ComponentInferenceError,
    ComponentUnavailableError,
    LeafEvaluation,
    LeafEvaluator,
    PolicyScores,
    PolicyScoreSource,
    SearchContext,
    SearchPolicy,
    supports_contextual_root_coverage,
    supports_immediate_edge,
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
            return replace(
                self._fallback.score(context, state, legal_actions),
                source=PolicyScoreSource.FALLBACK,
            )


@dataclass(frozen=True, slots=True)
class _FallbackImmediateEdge:
    primary: object | None
    fallback: object | None
    primary_preparation_failed: bool = False


class FallbackLeafEvaluator:
    def __init__(self, primary: LeafEvaluator, fallback: LeafEvaluator) -> None:
        self._primary = primary
        self._fallback = fallback

    def evaluate(self, context: SearchContext, state: GameState) -> LeafEvaluation:
        try:
            return self._primary.evaluate(context, state)
        except _FALLBACK_ERRORS:
            return self._fallback.evaluate(context, state)

    @property
    def immediate_edge_enabled(self) -> bool:
        return supports_immediate_edge(self._primary) or supports_immediate_edge(self._fallback)

    @property
    def contextual_root_coverage_enabled(self) -> bool:
        # A fallback-only contextual recipe must not alter a successful learned
        # primary's root schedule. It still shapes an expanded edge if the
        # primary later fails recoverably.
        return supports_contextual_root_coverage(self._primary)

    def prepare_immediate_edge(self, context, state, decision, action) -> object:
        primary_preparation_failed = False
        primary_prepared: object | None = None
        if supports_immediate_edge(self._primary):
            try:
                primary_prepared = self._primary.prepare_immediate_edge(
                    context, state, decision, action
                )
            except _FALLBACK_ERRORS:
                primary_preparation_failed = True

        fallback_prepared = (
            self._fallback.prepare_immediate_edge(context, state, decision, action)
            if supports_immediate_edge(self._fallback)
            else None
        )
        return _FallbackImmediateEdge(
            primary=primary_prepared,
            fallback=fallback_prepared,
            primary_preparation_failed=primary_preparation_failed,
        )

    def evaluate_immediate_edge(
        self, context: SearchContext, state: GameState, prepared: object
    ) -> LeafEvaluation:
        if not isinstance(prepared, _FallbackImmediateEdge):
            raise TypeError("invalid fallback immediate-edge preparation")
        try:
            if prepared.primary_preparation_failed:
                raise ComponentInferenceError("primary immediate-edge preparation failed")
            if supports_immediate_edge(self._primary):
                return self._primary.evaluate_immediate_edge(context, state, prepared.primary)
            return self._primary.evaluate(context, state)
        except _FALLBACK_ERRORS:
            if supports_immediate_edge(self._fallback):
                return self._fallback.evaluate_immediate_edge(context, state, prepared.fallback)
            return self._fallback.evaluate(context, state)


__all__ = ["FallbackLeafEvaluator", "FallbackSearchPolicy"]
