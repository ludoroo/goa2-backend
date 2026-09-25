from __future__ import annotations

from types import SimpleNamespace

import pytest

from automata.agents.contracts import PlanningDecision
from automata.decision import DecisionDescriptor
from automata.search.config import SearchConfig
from automata.search.continuation import (
    AgentContinuationPolicy,
    ArgmaxContinuationPolicy,
    as_continuation_policy,
)
from automata.search.contracts import (
    ComponentInferenceError,
    LeafMode,
    PolicyScores,
    ScoreSemantics,
    SearchContext,
)
from automata.search.fallback import FallbackSearchPolicy
from automata.search.ismcts.engine import _rollout
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import TeamColor


def _context(decision: DecisionDescriptor | None = None) -> SearchContext:
    return SearchContext(
        root_viewer_id="hero_wasp",
        perspective_team=TeamColor.RED,
        current_owner_id="hero_wasp",
        decision=decision or DecisionDescriptor("CARD"),
    )


class _Scores:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores
        self.calls = []

    def score(self, context, state, legal_actions):
        self.calls.append((context, state, tuple(legal_actions)))
        return PolicyScores(tuple(legal_actions), self.scores, ScoreSemantics.LOGITS)


def test_argmax_continuation_uses_canonical_order_and_stable_first_tie() -> None:
    policy = _Scores((1.0, 4.0, 4.0))
    continuation = ArgmaxContinuationPolicy(policy)
    decision = DecisionDescriptor("CARD")
    state = object()

    selected = continuation.choose(_context(decision), state, decision, ("z", "a", "m"))

    assert selected == "a"
    assert policy.calls == [(_context(decision), state, ("z", "a", "m"))]


def test_argmax_continuation_accepts_probability_scores_and_skips_singleton_inference() -> None:
    class Probabilities:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, context, state, legal_actions):
            self.calls += 1
            return PolicyScores(tuple(legal_actions), (0.2, 0.8), ScoreSemantics.PROBABILITIES)

    policy = Probabilities()
    continuation = ArgmaxContinuationPolicy(policy)
    decision = DecisionDescriptor("CARD")

    assert continuation.choose(_context(decision), object(), decision, ("a", "b")) == "b"
    assert continuation.choose(_context(decision), object(), decision, ("forced",)) == "forced"
    assert policy.calls == 1


def test_argmax_continuation_requires_context_for_exact_decision() -> None:
    decision = DecisionDescriptor("CARD")
    with pytest.raises(ValueError, match="exact current decision"):
        ArgmaxContinuationPolicy(_Scores((1.0, 0.0))).choose(
            _context(DecisionDescriptor("CARD")), object(), decision, ("a", "b")
        )


def test_argmax_continuation_rejects_invalid_output_reordering_and_nonfinite_scores() -> None:
    class Reordered:
        def score(self, context, state, legal_actions):
            return PolicyScores(("a", "b"), (2.0, 1.0), ScoreSemantics.LOGITS)

    reordered_decision = DecisionDescriptor("CARD")
    with pytest.raises(ValueError, match="exact legal action order"):
        ArgmaxContinuationPolicy(Reordered()).choose(
            _context(reordered_decision), object(), reordered_decision, ("b", "a")
        )

    class InvalidOutput:
        def score(self, context, state, legal_actions):
            return SimpleNamespace(actions=tuple(legal_actions), scores=(0.0, 1.0))

    invalid_decision = DecisionDescriptor("CARD")
    with pytest.raises(TypeError, match="PolicyScores"):
        ArgmaxContinuationPolicy(InvalidOutput()).choose(
            _context(invalid_decision), object(), invalid_decision, ("a", "b")
        )

    class Nonfinite:
        def score(self, context, state, legal_actions):
            return PolicyScores(tuple(legal_actions), (0.0, float("nan")), ScoreSemantics.LOGITS)

    nonfinite_decision = DecisionDescriptor("CARD")
    with pytest.raises(ValueError, match="finite"):
        ArgmaxContinuationPolicy(Nonfinite()).choose(
            _context(nonfinite_decision), object(), nonfinite_decision, ("a", "b")
        )


def test_argmax_continuation_falls_back_only_for_declared_component_failures() -> None:
    fallback = _Scores((1.0, 3.0))

    class Recoverable:
        def score(self, context, state, legal_actions):
            raise ComponentInferenceError("runtime failed")

    continuation = ArgmaxContinuationPolicy(FallbackSearchPolicy(Recoverable(), fallback))
    recoverable_decision = DecisionDescriptor("CARD")
    assert (
        continuation.choose(
            _context(recoverable_decision),
            object(),
            recoverable_decision,
            ("first", "second"),
        )
        == "second"
    )

    class Broken:
        def score(self, context, state, legal_actions):
            raise RuntimeError("bug")

    broken_decision = DecisionDescriptor("CARD")
    with pytest.raises(RuntimeError, match="bug"):
        ArgmaxContinuationPolicy(FallbackSearchPolicy(Broken(), fallback)).choose(
            _context(broken_decision), object(), broken_decision, ("first", "second")
        )


class _ScriptedAgent:
    def __init__(self, *, planning=None, selection=None) -> None:
        self.planning = planning
        self.selection = selection

    def choose_planning(self, state, hero):
        return self.planning

    def choose_input(self, state, request):
        return self.selection


def test_agent_continuation_maps_planning_and_raw_input_to_canonical_keys() -> None:
    card = SimpleNamespace(id="card-1")
    hero = SimpleNamespace(id="hero_wasp")
    card_decision = DecisionDescriptor("CARD", hero=hero)
    commit = AgentContinuationPolicy(
        _ScriptedAgent(planning=PlanningDecision.commit(card))  # type: ignore[arg-type]
    )
    assert (
        commit.choose(_context(card_decision), object(), card_decision, ("card-1", None))
        == "card-1"
    )

    for planning in (PlanningDecision.finish(), PlanningDecision.pass_()):
        continuation = AgentContinuationPolicy(_ScriptedAgent(planning=planning))
        assert (
            continuation.choose(_context(card_decision), object(), card_decision, ("card-1", None))
            is None
        )

    request = InputRequest(
        request_type=InputRequestType.SELECT_HEX,
        player_id="hero_wasp",
    )
    input_decision = DecisionDescriptor("INPUT", request=request)
    hex_policy = AgentContinuationPolicy(_ScriptedAgent(selection={"q": 1, "r": -1, "s": 0}))
    assert hex_policy.choose(
        _context(input_decision),
        object(),
        input_decision,
        (("hex", 1, -1, 0), "SKIP"),
    ) == ("hex", 1, -1, 0)
    skip_policy = AgentContinuationPolicy(_ScriptedAgent(selection="SKIP"))
    assert (
        skip_policy.choose(
            _context(input_decision), object(), input_decision, (("hex", 1, -1, 0), "SKIP")
        )
        == "SKIP"
    )


def test_agent_continuation_fails_closed_and_adapter_wraps_only_agents() -> None:
    decision = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            request_type=InputRequestType.SELECT_OPTION,
            player_id="hero_wasp",
        ),
    )
    agent = _ScriptedAgent(selection="stale")
    wrapped = as_continuation_policy(agent)
    assert isinstance(wrapped, AgentContinuationPolicy)
    with pytest.raises(ValueError, match="outside canonical legality"):
        wrapped.choose(_context(decision), object(), decision, ("legal",))

    direct = ArgmaxContinuationPolicy(_Scores((1.0,)))
    assert as_continuation_policy(direct) is direct
    with pytest.raises(ValueError, match="unsupported continuation"):
        AgentContinuationPolicy(agent).choose(
            _context(), object(), DecisionDescriptor("BOUNDARY"), ("legal",)
        )


def test_rollout_uses_policy_key_through_apply_ours_for_synthetic_skip() -> None:
    request = InputRequest(
        id="optional",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_arien",
        prompt="Optional",
        options=[InputOption(id="apply", text="Apply")],
        can_skip=True,
    )
    decision = DecisionDescriptor("INPUT", request=request)
    policy = _Scores((0.0, 1.0))

    class Simulator:
        def __init__(self) -> None:
            self.state = SimpleNamespace(round=1)
            self.our_team = TeamColor.RED
            self.applied = []

        def apply_ours(self, current, key, *, action_boundary=None):
            self.applied.append((current, key, action_boundary))
            return DecisionDescriptor("OVER", winner="RED")

    sim = Simulator()
    root_context = _context()
    reward = _rollout(
        sim,  # type: ignore[arg-type]
        decision,
        SearchConfig(
            leaf_mode=LeafMode.BOUNDED_CONTINUATION,
            cutoff_limit=1,
        ),
        ArgmaxContinuationPolicy(policy),
        SimpleNamespace(),  # terminal rollout does not evaluate a leaf
        root_context,
    )

    assert reward == 1.0
    assert sim.applied == [(decision, "SKIP", None)]
    called_context = policy.calls[0][0]
    assert called_context.root_viewer_id == "hero_wasp"
    assert called_context.current_owner_id == "hero_arien"
    assert called_context.decision is decision
