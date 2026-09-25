"""Learned-model-neutral component contracts for search.

The contracts describe score meaning and preserve candidate order explicitly;
they do not import an inference runtime or observation encoder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol, TypeGuard, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

from automata.decision import ActionBoundaryKind, DecisionDescriptor
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

ActionT = TypeVar("ActionT")


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Stable root perspective plus the exact current decision and its owner."""

    root_viewer_id: str
    perspective_team: TeamColor
    current_owner_id: str
    decision: DecisionDescriptor
    action_boundary_kind: ActionBoundaryKind | None = None

    @property
    def is_action_boundary(self) -> bool:
        return self.action_boundary_kind is not None

    def for_decision(self, decision: DecisionDescriptor, *, owner_id: str) -> SearchContext:
        return replace(
            self,
            current_owner_id=owner_id,
            decision=decision,
            action_boundary_kind=None,
        )

    def for_action_boundary(
        self, kind: ActionBoundaryKind = ActionBoundaryKind.COMPLETE
    ) -> SearchContext:
        """Retain the latest encodable decision while marking an action cutoff."""
        return replace(self, action_boundary_kind=kind)


class ScoreSemantics(StrEnum):
    LOGITS = "LOGITS"
    PROBABILITIES = "PROBABILITIES"


class PolicyScoreSource(StrEnum):
    """How the policy scores used for this decision were produced."""

    PRIMARY = "PRIMARY"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True, slots=True)
class PolicyScores:
    actions: tuple[Any, ...]
    scores: tuple[float, ...]
    semantics: ScoreSemantics
    source: PolicyScoreSource = PolicyScoreSource.PRIMARY

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.scores):
            raise ValueError("policy scores must align one-to-one with actions")
        if not all(math.isfinite(score) for score in self.scores):
            raise ValueError("policy scores must be finite")
        if self.semantics is ScoreSemantics.PROBABILITIES and (
            any(score < 0.0 for score in self.scores)
            or not math.isclose(sum(self.scores), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ValueError("probability scores must be non-negative and sum to one")


class SearchPolicy(Protocol):
    def score(
        self,
        context: SearchContext,
        state: GameState,
        legal_actions: Sequence[ActionT],
    ) -> PolicyScores: ...


class ContinuationPolicy(Protocol):
    """Choose one canonical key at a controlled rollout decision."""

    def choose(
        self,
        context: SearchContext,
        state: GameState,
        decision: DecisionDescriptor,
        legal_actions: Sequence[ActionT],
    ) -> ActionT: ...


def score_policy(
    policy: SearchPolicy,
    context: SearchContext,
    state: GameState,
    legal_actions: Sequence[ActionT],
) -> PolicyScores:
    """Invoke and strictly validate the policy's candidate alignment."""
    legal = tuple(legal_actions)
    result = policy.score(context, state, legal)
    if not isinstance(result, PolicyScores):
        raise TypeError("search policy must return PolicyScores")
    if result.actions != legal:
        raise ValueError("policy actions must preserve the exact legal action order")
    return result


class LeafEvaluation(BaseModel):
    """Validated normalized value from the root team's perspective."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    value: float

    @field_validator("value")
    @classmethod
    def normalized_value(cls, value: float) -> float:
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("leaf evaluation must be a finite value in [-1, 1]")
        return value


class LeafEvaluator(Protocol):
    """Evaluate a leaf using the validated normalized-value contract."""

    def evaluate(self, context: SearchContext, state: GameState) -> LeafEvaluation: ...


@runtime_checkable
class ImmediateEdgeLeafEvaluator(LeafEvaluator, Protocol):
    """Optionally shape a nonterminal immediate-cutoff edge.

    IMMEDIATE prepares only newly expanded edges; IMMEDIATE_ACTION prepares the
    selected INPUT-root edge in each determinization before continuing it to an
    action boundary. Preparation happens against the parent simulation state.
    Implementations must return compact immutable data rather than retaining the
    state itself.
    """

    @property
    def immediate_edge_enabled(self) -> bool: ...

    def prepare_immediate_edge(
        self,
        context: SearchContext,
        state: GameState,
        decision: DecisionDescriptor,
        action: Any,
    ) -> object: ...

    def evaluate_immediate_edge(
        self,
        context: SearchContext,
        state: GameState,
        prepared: object,
    ) -> LeafEvaluation: ...


@runtime_checkable
class ContextualRootCoverageLeafEvaluator(ImmediateEdgeLeafEvaluator, Protocol):
    """An edge evaluator eligible to force narrow root/no-op comparisons."""

    @property
    def contextual_root_coverage_enabled(self) -> bool: ...


def supports_immediate_edge(evaluator: LeafEvaluator) -> TypeGuard[ImmediateEdgeLeafEvaluator]:
    """Return whether an evaluator's optional edge capability is active."""
    return isinstance(evaluator, ImmediateEdgeLeafEvaluator) and evaluator.immediate_edge_enabled


def supports_contextual_root_coverage(
    evaluator: LeafEvaluator,
) -> TypeGuard[ContextualRootCoverageLeafEvaluator]:
    """Return whether contextual value is primary enough to force root coverage."""
    return (
        isinstance(evaluator, ContextualRootCoverageLeafEvaluator)
        and evaluator.contextual_root_coverage_enabled
        and evaluator.immediate_edge_enabled
    )


class LeafMode(StrEnum):
    IMMEDIATE = "IMMEDIATE"
    IMMEDIATE_ACTION = "IMMEDIATE_ACTION"
    STABLE_TURN = "STABLE_TURN"
    BOUNDED_CONTINUATION = "BOUNDED_CONTINUATION"


class CutoffUnit(StrEnum):
    ROUNDS = "ROUNDS"
    DECISIONS = "DECISIONS"


class ComponentUnavailableError(RuntimeError):
    """A configured optional component is unavailable in this process."""


class ComponentInferenceError(RuntimeError):
    """A component was available but could not score this request."""


__all__ = [
    "ComponentInferenceError",
    "ComponentUnavailableError",
    "ContextualRootCoverageLeafEvaluator",
    "ContinuationPolicy",
    "CutoffUnit",
    "ImmediateEdgeLeafEvaluator",
    "LeafEvaluation",
    "LeafEvaluator",
    "LeafMode",
    "PolicyScoreSource",
    "PolicyScores",
    "ScoreSemantics",
    "SearchContext",
    "SearchPolicy",
    "score_policy",
    "supports_contextual_root_coverage",
    "supports_immediate_edge",
]
