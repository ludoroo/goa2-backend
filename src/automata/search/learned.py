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


class LearnedSearchPolicy:
    """Return unnormalized model logits in the caller's exact legal order."""

    def __init__(self, runtime: LearnedModelRuntime) -> None:
        self.runtime = runtime

    def score(self, context: SearchContext, state: GameState, legal_actions) -> PolicyScores:
        legal = tuple(legal_actions)
        observation = encode_search_context(context, state, legal)
        output = _evaluate(self.runtime, observation)
        expected = tuple(candidate.candidate_id for candidate in observation.candidates)
        if output.candidate_ids != expected:
            raise ValueError("runtime candidates must preserve exact legal action order")
        logits = tuple(float(value) for value in output.policy_logits)
        return PolicyScores(legal, logits, ScoreSemantics.LOGITS)


class LearnedLeafEvaluator:
    """Evaluate a nonterminal leaf from the fixed root perspective."""

    def __init__(self, runtime: LearnedModelRuntime) -> None:
        self.runtime = runtime

    def evaluate(self, context: SearchContext, state: GameState) -> LeafEvaluation:
        if state.input_stack:
            request = state.input_stack[-1]
            decision = DecisionDescriptor("INPUT", request=request)
        else:
            hero = state.get_hero(HeroID(context.current_owner_id))
            if hero is None:
                raise ValueError("leaf has no current decision owner")
            from goa2.engine.phases import planning_open_for_second_card

            decision = DecisionDescriptor(
                "CARD",
                hero=hero,
                can_finish_planning=planning_open_for_second_card(state, hero.id),
            )
        legal = legal_keys_for_decision(decision)
        if not legal:
            raise ValueError("leaf decision has no encodable legal candidates")
        output = _evaluate(self.runtime, encode_search_context(context, state, legal))
        return LeafEvaluation(value=output.value)


__all__ = ["LearnedLeafEvaluator", "LearnedSearchPolicy"]
