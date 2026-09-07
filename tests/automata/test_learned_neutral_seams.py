from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from automata.search.contracts import (
    ComponentInferenceError,
    ComponentUnavailableError,
    CutoffUnit,
    LeafEvaluation,
    LeafMode,
    PolicyScores,
    ScoreSemantics,
    SearchContext,
    score_policy,
)
from automata.search.fallback import FallbackLeafEvaluator, FallbackSearchPolicy
from automata.search.statistics import RunningStatistics
from goa2.domain.models import TeamColor


class _Policy:
    def __init__(self, label: str, failure: Exception | None = None) -> None:
        self.label = label
        self.failure = failure

    def score(self, context, state, legal_actions):
        if self.failure:
            raise self.failure
        return PolicyScores(tuple(legal_actions), (2.0, 1.0), ScoreSemantics.LOGITS)


class _Leaf:
    def __init__(self, value: float, failure: Exception | None = None) -> None:
        self.value = value
        self.failure = failure

    def evaluate(self, context, state):
        if self.failure:
            raise self.failure
        return LeafEvaluation(value=self.value)


def _context() -> SearchContext:
    return SearchContext(
        root_viewer_id="hero_wasp",
        perspective_team=TeamColor.RED,
        current_owner_id="hero_wasp",
    )


def test_search_context_keeps_root_identity_when_owner_changes() -> None:
    context = _context()
    moved = context.for_owner("hero_arien")
    assert moved.root_viewer_id == "hero_wasp"
    assert moved.perspective_team is TeamColor.RED
    assert moved.current_owner_id == "hero_arien"
    with pytest.raises(FrozenInstanceError):
        context.current_owner_id = "hero_arien"  # type: ignore[misc]


def test_policy_scores_are_explicit_and_exactly_aligned_to_legal_order() -> None:
    scores = score_policy(_Policy("H"), _context(), object(), ["b", "a"])
    assert scores.actions == ("b", "a")
    assert scores.semantics is ScoreSemantics.LOGITS

    class Reordered(_Policy):
        def score(self, context, state, legal_actions):
            return PolicyScores(("a", "b"), (2.0, 1.0), ScoreSemantics.LOGITS)

    with pytest.raises(ValueError, match="exact legal action order"):
        score_policy(Reordered("bad"), _context(), object(), ["b", "a"])


@pytest.mark.parametrize(
    ("environment", "continuation", "leaf", "expected"),
    [
        ("H", "H", "H", ("H", "H", 0.25)),
        ("H", "L", "H", ("H", "L", 0.25)),
        ("H", "H", "L", ("H", "H", 0.75)),
        ("H", "L", "L", ("H", "L", 0.75)),
    ],
)
def test_h_l_components_compose_independently(environment, continuation, leaf, expected) -> None:
    policies = {"H": _Policy("H"), "L": _Policy("L")}
    leaves = {"H": _Leaf(0.25), "L": _Leaf(0.75)}
    env_result = score_policy(policies[environment], _context(), object(), ["x", "y"])
    cont_result = score_policy(policies[continuation], _context(), object(), ["x", "y"])
    assert (
        environment,
        continuation,
        leaves[leaf].evaluate(_context(), object()).value,
    ) == expected
    assert env_result.actions == cont_result.actions == ("x", "y")


@pytest.mark.parametrize(
    "failure", [ComponentUnavailableError("off"), ComponentInferenceError("bad")]
)
def test_component_fallbacks_only_catch_declared_failures(failure: Exception) -> None:
    policy = FallbackSearchPolicy(_Policy("L", failure), _Policy("H"))
    leaf = FallbackLeafEvaluator(_Leaf(0.0, failure), _Leaf(0.5))
    assert score_policy(policy, _context(), object(), ["x", "y"]).scores == (2.0, 1.0)
    assert leaf.evaluate(_context(), object()).value == 0.5

    with pytest.raises(RuntimeError):
        score_policy(
            FallbackSearchPolicy(_Policy("L", RuntimeError("bug")), _Policy("H")),
            _context(),
            object(),
            ["x", "y"],
        )


def test_leaf_evaluation_and_modes_have_exact_score_semantics() -> None:
    assert LeafEvaluation(value=-0.4).value == -0.4
    assert {LeafMode.IMMEDIATE, LeafMode.BOUNDED_CONTINUATION} == set(LeafMode)
    assert {CutoffUnit.ROUNDS, CutoffUnit.DECISIONS} == set(CutoffUnit)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        LeafEvaluation(value=2.0)


def test_running_statistics_are_generic_and_report_population_variance() -> None:
    stats = RunningStatistics()
    for value in (1.0, 2.0, 3.0):
        stats = stats.add(value)
    assert stats.count == 3
    assert stats.mean == 2.0
    assert stats.variance == pytest.approx(2 / 3)
