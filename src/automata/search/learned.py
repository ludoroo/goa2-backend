"""Learned-model runtime adapters for the model-neutral classic search contracts."""

from __future__ import annotations

from typing import Never

from automata.models.contracts import (
    ArtifactError,
    DecisionObservation,
    LearnedModelOutput,
    LearnedModelRuntime,
    LearnedPolicyOutput,
    LearnedPolicyRuntime,
    LearnedStableValueOutput,
    LearnedStableValueRuntime,
    StableValueObservation,
)
from automata.observation import encode_search_context, legal_keys_for_decision
from automata.observation.value_encoder import encode_stable_value
from goa2.domain.state import GameState

from .contracts import (
    ComponentInferenceError,
    ComponentUnavailableError,
    LeafEvaluation,
    PolicyScores,
    ScoreSemantics,
    SearchContext,
    StableValueContext,
)


def _translate_runtime_error(exc: Exception) -> Never:
    """Preserve the search component's availability/inference error boundary."""
    if isinstance(exc, ArtifactError):
        raise ComponentUnavailableError(str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise ComponentInferenceError(str(exc)) from exc
    raise exc


def _evaluate_policy(
    runtime: LearnedPolicyRuntime | LearnedModelRuntime,
    observation: DecisionObservation,
) -> LearnedPolicyOutput | LearnedModelOutput:
    try:
        if isinstance(runtime, LearnedPolicyRuntime):
            return runtime.evaluate_policy(observation)
        return runtime.evaluate(observation)
    except (ComponentUnavailableError, ComponentInferenceError):
        raise
    except Exception as exc:
        _translate_runtime_error(exc)


def _evaluate(runtime: LearnedModelRuntime, observation: DecisionObservation) -> LearnedModelOutput:
    try:
        return runtime.evaluate(observation)
    except (ComponentUnavailableError, ComponentInferenceError):
        raise
    except Exception as exc:
        _translate_runtime_error(exc)


def _evaluate_stable_value(
    runtime: LearnedStableValueRuntime,
    observation: StableValueObservation,
) -> LearnedStableValueOutput:
    try:
        return runtime.evaluate_stable_value(observation)
    except (ComponentUnavailableError, ComponentInferenceError):
        raise
    except Exception as exc:
        _translate_runtime_error(exc)


class LearnedSearchPolicy:
    """Score a canonical observation and return logits in the caller's legal order."""

    def __init__(self, runtime: LearnedPolicyRuntime | LearnedModelRuntime) -> None:
        self.runtime = runtime

    def score(self, context: SearchContext, state: GameState, legal_actions) -> PolicyScores:
        legal = tuple(legal_actions)
        canonical = tuple(legal_keys_for_decision(context.decision))
        if len(legal) != len(canonical) or any(action not in legal for action in canonical):
            raise ValueError("caller actions must contain the exact canonical legal candidates")
        observation = encode_search_context(context, state, canonical)
        output = _evaluate_policy(self.runtime, observation)
        expected = tuple(candidate.candidate_id for candidate in observation.candidates)
        if output.candidate_ids != expected:
            raise ValueError("runtime candidates must preserve exact legal action order")
        canonical_logits = tuple(float(value) for value in output.policy_logits)
        logits = tuple(canonical_logits[canonical.index(action)] for action in legal)
        return PolicyScores(legal, logits, ScoreSemantics.LOGITS)


class LearnedLeafEvaluator:
    """Evaluate a nonterminal leaf from the fixed root perspective."""

    recipe_id = "learned-value-head-v1"

    def __init__(self, runtime: LearnedModelRuntime) -> None:
        self.runtime = runtime

    def evaluate(self, context: SearchContext, state: GameState) -> LeafEvaluation:
        legal = legal_keys_for_decision(context.decision)

        if not legal:
            raise ValueError("leaf decision has no encodable legal candidates")
        output = _evaluate(self.runtime, encode_search_context(context, state, legal))
        return LeafEvaluation(value=output.value)


class LearnedStableValueEvaluator:
    """Evaluate an authoritative stable boundary with native value inference."""

    recipe_id = "learned-stable-boundary-value-v1"

    def __init__(self, runtime: LearnedStableValueRuntime) -> None:
        if not isinstance(runtime, LearnedStableValueRuntime):
            raise TypeError("runtime must implement LearnedStableValueRuntime")
        self.runtime = runtime

    def evaluate_stable_value(
        self, context: StableValueContext, state: GameState
    ) -> LeafEvaluation:
        observation = encode_stable_value(
            state,
            context.boundary,
            viewer_hero_id=context.root_viewer_id,
            perspective_team=context.perspective_team,
        )
        output = _evaluate_stable_value(self.runtime, observation)
        return LeafEvaluation(value=output.value)


__all__ = [
    "LearnedLeafEvaluator",
    "LearnedSearchPolicy",
    "LearnedStableValueEvaluator",
]
