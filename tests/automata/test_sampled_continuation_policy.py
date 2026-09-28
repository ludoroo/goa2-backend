from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.decision import DecisionDescriptor
from automata.runtime.effects import register_all_effects
from automata.search import (
    LEARNED_PRIOR_SAMPLING_CONTINUATION_POLICY_ID,
    PriorSamplingContinuationPolicy,
)
from automata.search.config import SearchConfig
from automata.search.contracts import (
    ComponentInferenceError,
    CutoffUnit,
    PolicyScores,
    ScoreSemantics,
    SearchContext,
)
from automata.search.fallback import FallbackSearchPolicy
from automata.search.ismcts import engine
from automata.search.ismcts.engine import RootTarget, search
from automata.search.node import Node
from goa2.domain.models import TeamColor
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def _context(decision: DecisionDescriptor) -> SearchContext:
    return SearchContext(
        root_viewer_id="hero_wasp",
        perspective_team=TeamColor.RED,
        current_owner_id="hero_wasp",
        decision=decision,
    )


class _Scores:
    def __init__(
        self,
        scores: tuple[float, ...],
        semantics: ScoreSemantics = ScoreSemantics.PROBABILITIES,
    ) -> None:
        self.scores = scores
        self.semantics = semantics
        self.calls = 0

    def score(self, context, state, legal_actions):
        self.calls += 1
        return PolicyScores(tuple(legal_actions), self.scores, self.semantics)


def _draws(policy, *, seed: int, count: int = 24) -> tuple[str, ...]:
    decision = DecisionDescriptor("CARD")
    bound = PriorSamplingContinuationPolicy(policy).for_search(seed)
    return tuple(
        bound.choose(_context(decision), object(), decision, ("zero", "lower", "argmax"))
        for _ in range(count)
    )


def test_prior_sampler_uses_stable_softmax_and_can_select_non_argmax_actions() -> None:
    probability_draws = _draws(_Scores((0.0, 0.25, 0.75)), seed=19)
    logit_draws = _draws(
        _Scores(
            (-1000.0, 1000.0, 1000.0 + math.log(3.0)),
            ScoreSemantics.LOGITS,
        ),
        seed=19,
    )

    assert probability_draws == logit_draws
    assert set(probability_draws) == {"lower", "argmax"}

    # Verify the distribution, not a particular RNG/hash implementation's trace.
    draws = _draws(_Scores((0.0, 0.25, 0.75)), seed=19, count=4000)
    assert "zero" not in draws
    assert 0.22 < draws.count("lower") / len(draws) < 0.28


def test_prior_sampler_skips_inference_and_rng_for_a_single_legal_choice() -> None:
    policy = _Scores((0.5, 0.5))
    template = PriorSamplingContinuationPolicy(policy)
    decision = DecisionDescriptor("CARD")
    after_singleton = template.for_search(31)
    fresh = template.for_search(31)

    assert after_singleton.choose(_context(decision), object(), decision, ("forced",)) == "forced"
    after_draw = after_singleton.choose(_context(decision), object(), decision, ("first", "second"))
    fresh_draw = fresh.choose(_context(decision), object(), decision, ("first", "second"))

    assert after_draw == fresh_draw
    assert policy.calls == 2


def test_prior_sampler_direct_use_requires_an_explicit_search_binding() -> None:
    decision = DecisionDescriptor("CARD")
    template = PriorSamplingContinuationPolicy(_Scores((0.5, 0.5)))

    with pytest.raises(RuntimeError, match="for_search"):
        template.choose(_context(decision), object(), decision, ("a", "b"))


@pytest.mark.parametrize(
    "malformation",
    ["reordered", "wrong_type", "nonfinite", "negative", "unnormalized", "unknown_semantics"],
)
def test_prior_sampler_fails_closed_on_malformed_or_reordered_scores(malformation: str) -> None:
    class Broken:
        def score(self, context, state, legal_actions):
            if malformation == "reordered":
                return PolicyScores(("b", "a"), (0.5, 0.5), ScoreSemantics.PROBABILITIES)
            if malformation == "nonfinite":
                return PolicyScores(tuple(legal_actions), (0.0, math.nan), ScoreSemantics.LOGITS)
            if malformation == "negative":
                return PolicyScores(tuple(legal_actions), (-0.5, 1.5), ScoreSemantics.PROBABILITIES)
            if malformation == "unnormalized":
                return PolicyScores(tuple(legal_actions), (0.5, 0.6), ScoreSemantics.PROBABILITIES)
            if malformation == "unknown_semantics":
                return PolicyScores(tuple(legal_actions), (0.5, 0.5), "UNKNOWN")
            return SimpleNamespace(
                actions=tuple(legal_actions),
                scores=(0.5, 0.5),
                semantics=ScoreSemantics.PROBABILITIES,
            )

    decision = DecisionDescriptor("CARD")
    continuation = PriorSamplingContinuationPolicy(Broken()).for_search(1)
    error = TypeError if malformation == "wrong_type" else ValueError
    with pytest.raises(error):
        continuation.choose(_context(decision), object(), decision, ("a", "b"))


def test_prior_sampler_preserves_declared_fallback_and_strict_failure_semantics() -> None:
    class Unavailable:
        def score(self, context, state, legal_actions):
            raise ComponentInferenceError("inference unavailable")

    class Broken:
        def score(self, context, state, legal_actions):
            raise RuntimeError("policy bug")

    decision = DecisionDescriptor("CARD")
    fallback = _Scores((0.0, 1.0))
    recovered = PriorSamplingContinuationPolicy(
        FallbackSearchPolicy(Unavailable(), fallback)
    ).for_search(9)
    assert recovered.choose(_context(decision), object(), decision, ("a", "b")) == "b"

    strict = PriorSamplingContinuationPolicy(FallbackSearchPolicy(Broken(), fallback)).for_search(9)
    with pytest.raises(RuntimeError, match="policy bug"):
        strict.choose(_context(decision), object(), decision, ("a", "b"))


def test_prior_sampler_bindings_are_independent_and_safe_for_concurrent_reuse() -> None:
    policy = _Scores((0.0, 0.4, 0.6))
    template = PriorSamplingContinuationPolicy(policy)

    def run() -> tuple[str, ...]:
        decision = DecisionDescriptor("CARD")
        bound = template.for_search(77)
        return tuple(
            bound.choose(_context(decision), object(), decision, ("zero", "a", "b"))
            for _ in range(50)
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(run)
        second_future = executor.submit(run)
    first = first_future.result()
    second = second_future.result()

    assert first == second == run()
    assert "zero" not in first
    assert set(first) == {"a", "b"}


def test_prior_sampler_does_not_advance_global_or_unrelated_rng() -> None:
    decision = DecisionDescriptor("CARD")
    continuation = PriorSamplingContinuationPolicy(_Scores((0.5, 0.5))).for_search(5)
    engine_rng = random.Random(5)
    engine_state = engine_rng.getstate()
    global_state = random.getstate()

    continuation.choose(_context(decision), object(), decision, ("a", "b"))

    assert engine_rng.getstate() == engine_state
    assert random.getstate() == global_state


@pytest.mark.parametrize("prebound", [False, True], ids=["template", "previously-bound"])
def test_search_binds_one_fresh_sampler_per_call_and_shares_it_across_iterations(
    monkeypatch, prebound
) -> None:
    register_all_effects()
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=7,
    )
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    root_legal = tuple(card.id for card in hero.hand)
    target = RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id}))
    template = PriorSamplingContinuationPolicy(_Scores((0.5, 0.5)))
    supplied = template.for_search(99) if prebound else template
    if prebound:
        prior_decision = DecisionDescriptor("CARD")
        supplied.choose(_context(prior_decision), object(), prior_decision, ("x", "y"))
    trace: list[str] = []
    rng_isolation: list[bool] = []

    def simulate(*args, **kwargs):
        root = args[0]
        engine_rng = args[12]
        continuation = args[13]
        decision = DecisionDescriptor("CARD")
        context = _context(decision)
        before_engine = engine_rng.getstate()
        before_global = random.getstate()
        trace.append(continuation.choose(context, object(), decision, ("follow-a", "follow-b")))
        rng_isolation.append(
            engine_rng.getstate() == before_engine and random.getstate() == before_global
        )
        key = args[3][0]
        child = root.children.setdefault(key, Node())
        child.update(0.5)
        root.update(0.5)

    monkeypatch.setattr(engine, "_simulate", simulate)

    def run(seed: int) -> tuple[str, ...]:
        start = len(trace)
        result = search(
            state,
            TeamColor.RED,
            root_legal,
            HeuristicAgent(3),
            SearchConfig(iterations=16, seed=seed, use_prior=False),
            root_target=target,
            continuation_policy=supplied,
        )
        assert result.root.visits == 16
        return tuple(trace[start:])

    # Prior direct binding/use must not perturb the cached template.
    preused = template.for_search(41)
    decision = DecisionDescriptor("CARD")
    for _ in range(7):
        preused.choose(_context(decision), object(), decision, ("follow-a", "follow-b"))

    first = run(41)
    repeated = run(41)
    different_seed = run(42)

    assert first == repeated
    assert set(first) == {"follow-a", "follow-b"}
    assert different_seed != first
    assert all(rng_isolation)
    assert template.policy_id == LEARNED_PRIOR_SAMPLING_CONTINUATION_POLICY_ID


def test_sampled_continuation_repeats_within_real_search_without_mutating_live_state() -> None:
    register_all_effects()
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=7,
    )
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    legal = tuple(card.id for card in hero.hand)
    target = RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id}))
    traces = []

    class UniformPolicy:
        def score(self, context, state, legal_actions):
            assert context.root_viewer_id == context.current_owner_id == hero.id
            assert context.perspective_team == TeamColor.RED
            return PolicyScores(
                tuple(legal_actions),
                tuple(1.0 / len(legal_actions) for _ in legal_actions),
                ScoreSemantics.PROBABILITIES,
            )

    class RecordingTemplate(PriorSamplingContinuationPolicy):
        def for_search(self, seed):
            sampler = super().for_search(seed)
            trace = []
            traces.append(trace)

            class RecordingBound:
                def choose(self, context, state, decision, legal_actions):
                    choice = sampler.choose(context, state, decision, legal_actions)
                    trace.append((tuple(legal_actions), choice))
                    return choice

            return RecordingBound()

    template = RecordingTemplate(UniformPolicy())
    config = SearchConfig(
        iterations=12, seed=31, use_prior=False, cutoff_unit=CutoffUnit.DECISIONS, cutoff_limit=3
    )
    original = state.model_dump(mode="json")
    global_state = random.getstate()
    results = [
        search(
            state,
            TeamColor.RED,
            legal,
            HeuristicAgent(3),
            config,
            root_target=target,
            continuation_policy=template,
        )
        for _ in range(2)
    ]

    assert traces[0] and traces[0] == traces[1]
    assert results[0].best_key == results[1].best_key
    assert [(key, child.visits, child.q) for key, child in results[0].root.children.items()] == [
        (key, child.visits, child.q) for key, child in results[1].root.children.items()
    ]
    assert state.model_dump(mode="json") == original
    assert random.getstate() == global_state
