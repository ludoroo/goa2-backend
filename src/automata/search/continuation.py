"""Policies for controlled follow-up decisions inside search rollouts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from automata.agents.contracts import Agent, PlanningKind
from automata.decision import DecisionDescriptor
from goa2.domain.state import GameState

from .contracts import ContinuationPolicy, SearchContext, SearchPolicy, score_policy
from .node import action_key

ActionT = TypeVar("ActionT")

LEARNED_ARGMAX_CONTINUATION_POLICY_ID = "learned-argmax-v1"


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


__all__ = [
    "LEARNED_ARGMAX_CONTINUATION_POLICY_ID",
    "AgentContinuationPolicy",
    "ArgmaxContinuationPolicy",
    "as_continuation_policy",
]
