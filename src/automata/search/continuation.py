"""Policies for controlled follow-up decisions inside search rollouts."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from typing import Any, TypeVar

from automata.agents.contracts import Agent, PlanningKind
from automata.decision import DecisionDescriptor
from goa2.domain.state import GameState

from .contracts import (
    ContinuationPolicy,
    ScoreSemantics,
    SearchContext,
    SearchPolicy,
    score_policy,
)
from .node import action_key

ActionT = TypeVar("ActionT")

LEARNED_ARGMAX_CONTINUATION_POLICY_ID = "learned-argmax-v1"
LEARNED_PRIOR_SAMPLING_CONTINUATION_POLICY_ID = "learned-prior-sampling-v1"


class AgentContinuationPolicy:
    """Adapt the historical Agent-driven rollout behavior to canonical keys."""

    policy_id = "agent-v1"

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def choose(
        self,
        context: SearchContext,
        state: GameState,
        decision: DecisionDescriptor,
        legal_actions: Sequence[ActionT],
    ) -> ActionT:
        del context
        legal = tuple(legal_actions)
        if not legal:
            raise ValueError("continuation policy requires at least one legal action")
        if decision.kind == "CARD":
            hero = decision.hero
            if hero is None:
                raise ValueError("CARD continuation is missing its hero")
            planning = self.agent.choose_planning(state, hero)
            selected: Any = (
                planning.card.id
                if planning.kind is PlanningKind.COMMIT and planning.card is not None
                else None
            )
        elif decision.kind == "INPUT":
            request = decision.request
            if request is None:
                raise ValueError("INPUT continuation is missing its request")
            selected = action_key(self.agent.choose_input(state, request))
        else:
            raise ValueError(f"unsupported continuation decision kind: {decision.kind!r}")
        if selected not in legal:
            raise ValueError("agent continuation selected an action outside canonical legality")
        return selected


class _BoundPriorSamplingContinuationPolicy:
    """Search-local categorical sampler with an isolated random stream."""

    policy_id = LEARNED_PRIOR_SAMPLING_CONTINUATION_POLICY_ID

    def __init__(self, policy: SearchPolicy, rng: random.Random) -> None:
        self.policy = policy
        self._rng = rng

    def for_search(self, seed: int) -> ContinuationPolicy:
        # Even an explicitly pre-bound policy gets a fresh stream when reused
        # as the input to another search; never share its partially used RNG.
        return PriorSamplingContinuationPolicy(self.policy).for_search(seed)

    def choose(
        self,
        context: SearchContext,
        state: GameState,
        decision: DecisionDescriptor,
        legal_actions: Sequence[ActionT],
    ) -> ActionT:
        if context.decision is not decision:
            raise ValueError("continuation context must describe the exact current decision")
        legal = tuple(legal_actions)
        if not legal:
            raise ValueError("continuation policy requires at least one legal action")
        if len(legal) == 1:
            return legal[0]

        scores = score_policy(self.policy, context, state, legal)
        if scores.semantics is ScoreSemantics.LOGITS:
            maximum = max(scores.scores)
            weights = tuple(math.exp(score - maximum) for score in scores.scores)
        elif scores.semantics is ScoreSemantics.PROBABILITIES:
            weights = scores.scores
        else:
            raise ValueError(f"unsupported policy score semantics: {scores.semantics!r}")

        # Sampling only positive entries makes the zero-mass exclusion exact,
        # including logits that underflow to zero after stable softmax.
        positive = tuple((index, weight) for index, weight in enumerate(weights) if weight > 0.0)
        total = math.fsum(weight for _, weight in positive)
        draw = self._rng.random() * total
        cumulative = 0.0
        for index, weight in positive:
            cumulative += weight
            if draw < cumulative:
                return legal[index]
        return legal[positive[-1][0]]


class PriorSamplingContinuationPolicy:
    """Reusable prior template that must be bound once for each search.

    The template owns no mutable RNG state, so cached strategies may safely
    reuse it across overlapping searches. ``for_search`` domain-separates the
    continuation stream from the other search RNGs.
    """

    policy_id = LEARNED_PRIOR_SAMPLING_CONTINUATION_POLICY_ID

    def __init__(self, policy: SearchPolicy) -> None:
        self.policy = policy

    def for_search(self, seed: int) -> ContinuationPolicy:
        payload = f"{self.policy_id}\0{seed}".encode()
        derived_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")
        return _BoundPriorSamplingContinuationPolicy(self.policy, random.Random(derived_seed))

    def choose(
        self,
        context: SearchContext,
        state: GameState,
        decision: DecisionDescriptor,
        legal_actions: Sequence[ActionT],
    ) -> ActionT:
        del context, state, decision, legal_actions
        raise RuntimeError(
            "PriorSamplingContinuationPolicy must be bound with for_search(seed) before use"
        )


class ArgmaxContinuationPolicy:
    """Choose the stable first argmax from a strictly validated SearchPolicy."""

    policy_id = LEARNED_ARGMAX_CONTINUATION_POLICY_ID

    def __init__(self, policy: SearchPolicy) -> None:
        self.policy = policy

    def choose(
        self,
        context: SearchContext,
        state: GameState,
        decision: DecisionDescriptor,
        legal_actions: Sequence[ActionT],
    ) -> ActionT:
        if context.decision is not decision:
            raise ValueError("continuation context must describe the exact current decision")
        legal = tuple(legal_actions)
        if not legal:
            raise ValueError("continuation policy requires at least one legal action")
        if len(legal) == 1:
            return legal[0]
        scores = score_policy(self.policy, context, state, legal)
        best_index = max(range(len(legal)), key=scores.scores.__getitem__)
        return legal[best_index]


def as_continuation_policy(policy: ContinuationPolicy | Agent) -> ContinuationPolicy:
    """Preserve Agent callers while making rollout selection policy-only."""
    if callable(getattr(policy, "choose", None)):
        return policy  # type: ignore[return-value]
    return AgentContinuationPolicy(policy)  # type: ignore[arg-type]


def prepare_continuation_policy_for_search(
    policy: ContinuationPolicy | Agent,
    seed: int,
) -> ContinuationPolicy:
    """Create search-local state only for continuation policies that request it."""
    continuation = as_continuation_policy(policy)
    binder = getattr(continuation, "for_search", None)
    if callable(binder):
        return binder(seed)
    return continuation


__all__ = [
    "LEARNED_ARGMAX_CONTINUATION_POLICY_ID",
    "LEARNED_PRIOR_SAMPLING_CONTINUATION_POLICY_ID",
    "AgentContinuationPolicy",
    "ArgmaxContinuationPolicy",
    "PriorSamplingContinuationPolicy",
    "as_continuation_policy",
    "prepare_continuation_policy_for_search",
]
