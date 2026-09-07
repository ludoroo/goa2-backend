"""Learned-model-neutral component contracts for search.

The contracts describe score meaning and preserve candidate order explicitly;
they do not import an inference runtime or observation encoder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

ActionT = TypeVar("ActionT")


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Stable root perspective plus the owner of the current decision."""

    root_viewer_id: str
    perspective_team: TeamColor
    current_owner_id: str

    def for_owner(self, owner_id: str) -> SearchContext:
        return replace(self, current_owner_id=owner_id)


class ScoreSemantics(StrEnum):
    LOGITS = "LOGITS"
    PROBABILITIES = "PROBABILITIES"


@dataclass(frozen=True, slots=True)
class PolicyScores:
    actions: tuple[Any, ...]
    scores: tuple[float, ...]
    semantics: ScoreSemantics

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


def score_policy(
    policy: SearchPolicy,
    context: SearchContext,
    state: GameState,
    legal_actions: Sequence[ActionT],
) -> PolicyScores:
    """Invoke and strictly validate the policy's candidate alignment."""
    legal = tuple(legal_actions)
    result = policy.score(context, state, legal)
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


class LeafMode(StrEnum):
    IMMEDIATE = "IMMEDIATE"
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
    "CutoffUnit",
    "LeafEvaluation",
    "LeafEvaluator",
    "LeafMode",
    "PolicyScores",
    "ScoreSemantics",
    "SearchContext",
    "SearchPolicy",
    "score_policy",
]
