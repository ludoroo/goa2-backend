"""Learned-model runtime adapters for the model-neutral classic search contracts."""

from __future__ import annotations

from automata.decision import DecisionDescriptor
from automata.models.contracts import (
    ArtifactError,
    DecisionObservation,
    LearnedModelOutput,
    LearnedModelRuntime,
)
from automata.observation import encode_search_context, legal_keys_for_decision
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

from .contracts import (
    ComponentInferenceError,
    ComponentUnavailableError,
    LeafEvaluation,
    PolicyScores,
    ScoreSemantics,
    SearchContext,
)


def _evaluate(runtime: LearnedModelRuntime, observation: DecisionObservation) -> LearnedModelOutput:
    try:
        return runtime.evaluate(observation)
    except (ComponentUnavailableError, ComponentInferenceError):
        raise
    except Exception as exc:
        # Artifact compatibility is an availability failure. Keep this import
        # lazy so classic H/H processes never load Torch.
        if isinstance(exc, ArtifactError):
            raise ComponentUnavailableError(str(exc)) from exc
        if isinstance(exc, RuntimeError):
            raise ComponentInferenceError(str(exc)) from exc
        raise


def _decision_for_context(context: SearchContext, state: GameState) -> DecisionDescriptor:
    decision = context.current_decision
    if decision is not None:
        return decision
    if state.input_stack:
        return DecisionDescriptor("INPUT", request=state.input_stack[-1])
    hero = state.get_hero(HeroID(context.current_owner_id))
    if hero is None:
        raise ValueError("leaf has no current decision owner")
    from goa2.engine.phases import planning_open_for_second_card

    return DecisionDescriptor(
        "CARD",
        hero=hero,
        can_finish_planning=planning_open_for_second_card(state, hero.id),
    )


class LearnedSearchPolicy:
    """Score a canonical observation and return logits in the caller's legal order."""

    def __init__(self, runtime: LearnedModelRuntime) -> None:
        self.runtime = runtime

    def score(self, context: SearchContext, state: GameState, legal_actions) -> PolicyScores:
        legal = tuple(legal_actions)
        decision = _decision_for_context(context, state)
        canonical = tuple(legal_keys_for_decision(decision))
        if len(legal) != len(canonical) or any(action not in legal for action in canonical):
            raise ValueError("caller actions must contain the exact canonical legal candidates")
        observation = encode_search_context(context, state, canonical)
        output = _evaluate(self.runtime, observation)
        expected = tuple(candidate.candidate_id for candidate in observation.candidates)
        if output.candidate_ids != expected:
            raise ValueError("runtime candidates must preserve exact legal action order")
        canonical_logits = tuple(float(value) for value in output.policy_logits)
        logits = tuple(canonical_logits[canonical.index(action)] for action in legal)
        return PolicyScores(legal, logits, ScoreSemantics.LOGITS)


class LearnedLeafEvaluator:
    """Evaluate a nonterminal leaf from the fixed root perspective."""

    def __init__(self, runtime: LearnedModelRuntime) -> None:
        self.runtime = runtime

    def evaluate(self, context: SearchContext, state: GameState) -> LeafEvaluation:
        decision = _decision_for_context(context, state)
        legal = legal_keys_for_decision(decision)
        if not legal:
            raise ValueError("leaf decision has no encodable legal candidates")
        output = _evaluate(self.runtime, encode_search_context(context, state, legal))
        return LeafEvaluation(value=output.value)


__all__ = ["LearnedLeafEvaluator", "LearnedSearchPolicy"]
