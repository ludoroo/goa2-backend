from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.decision import DecisionDescriptor
from automata.models.contracts.inference import LearnedModelOutput
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import (
    ComponentInferenceError,
    CutoffUnit,
    LeafEvaluation,
    LeafMode,
    PolicyScores,
    ScoreSemantics,
    SearchContext,
)
from automata.search.fallback import FallbackSearchPolicy
from automata.search.ismcts import RootActionDiagnostic, RootTarget, search
from automata.search.learned import LearnedSearchPolicy
from automata.search.node import Key, Node
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def _root() -> tuple[GameState, RootTarget, tuple[Key, ...]]:
    register_all_effects()
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp", "Xargatha"],
        ["Arien"],
        game_type="QUICK",
        seed=7,
    )
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    target = RootTarget.card(
        hero_id=wasp.id,
        owned_hero_ids=frozenset({"hero_wasp", "hero_xargatha"}),
    )
    return state, target, tuple(reversed([card.id for card in wasp.hand]))


class _CountingPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[SearchContext, GameState, tuple[Key, ...]]] = []

    def score(
        self,
        context: SearchContext,
        state: GameState,
        legal_actions: Sequence[Key],
    ) -> PolicyScores:
        legal = tuple(legal_actions)
        self.calls.append((context, state, legal))
        return PolicyScores(
            legal,
            tuple(float(len(legal) - index) for index in range(len(legal))),
            ScoreSemantics.LOGITS,
        )


def test_search_caches_one_aligned_root_policy_but_scores_each_descendant_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from automata.search.ismcts import engine

    state, target, legal = _root()
    policy = _CountingPolicy()
    worlds: list[GameState] = []
    real_determinize = engine.determinize

    def recording_determinize(*args: Any, **kwargs: Any) -> GameState:
        world = real_determinize(*args, **kwargs)
        worlds.append(world)
        return world

    monkeypatch.setattr(engine, "determinize", recording_determinize)
    search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(2),
        SearchConfig(
            iterations=3,
            cutoff_limit=0,
            leaf_mode=LeafMode.IMMEDIATE,
            seed=11,
            root_puct_c=1.5,
            root_widening_c=0.1,
            root_widening_alpha=0.5,
        ),
        policy,
        root_target=target,
    )

    root_calls = [call for call in policy.calls if call[0].current_owner_id == "hero_wasp"]
    descendant_calls = [
        call for call in policy.calls if call[0].current_owner_id == "hero_xargatha"
    ]
    assert len(root_calls) == 1
    assert root_calls[0][2] == legal
    assert len(descendant_calls) >= 2
    assert len({id(call[1]) for call in descendant_calls}) == len(descendant_calls)
    assert len(worlds) == 3
    assert len({id(world) for world in worlds}) == 3


def test_noncanonical_root_scores_canonical_learned_observation_then_realigns() -> None:
    state, target, caller_legal = _root()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    canonical = tuple(card.id for card in wasp.hand)

    class Runtime:
        def __init__(self) -> None:
            self.candidate_selections: tuple[object, ...] = ()

        def evaluate(self, observation):
            self.candidate_selections = tuple(
                candidate.selection for candidate in observation.candidates
            )
            candidate_ids = tuple(candidate.candidate_id for candidate in observation.candidates)
            size = len(candidate_ids)
            return LearnedModelOutput(
                candidate_ids=candidate_ids,
                policy_logits=tuple(float(index) for index in range(size)),
                probabilities=tuple(1.0 / size for _ in range(size)),
                value=0.0,
            )

    runtime = Runtime()
    result = search(
        state,
        TeamColor.RED,
        caller_legal,
        HeuristicAgent(2),
        SearchConfig(
            iterations=1,
            cutoff_limit=0,
            leaf_mode=LeafMode.IMMEDIATE,
            seed=11,
        ),
        LearnedSearchPolicy(runtime),
        root_target=target,
    )

    assert runtime.candidate_selections == canonical
    assert tuple(item.action for item in result.root_action_diagnostics) == caller_legal
    canonical_logits = {key: float(index) for index, key in enumerate(canonical)}
    expected = tuple(
        math.exp(canonical_logits[key] - max(canonical_logits.values()))
        / sum(
            math.exp(canonical_logits[other] - max(canonical_logits.values()))
            for other in caller_legal
        )
        for key in caller_legal
    )
    assert tuple(
        item.prior_probability for item in result.root_action_diagnostics
    ) == pytest.approx(expected)


def test_fallback_probability_policy_uses_classic_root_schedule_without_resoftmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, target, legal = _root()
    root_widening: list[tuple[Node, float, float]] = []
    root_puct: list[tuple[Node, float]] = []
    real_should_expand = Node.should_expand
    real_select = Node.select

    def recording_should_expand(
        self: Node, legal: list[Key], widen_c: float, widen_alpha: float
    ) -> bool:
        root_widening.append((self, widen_c, widen_alpha))
        return real_should_expand(self, legal, widen_c, widen_alpha)

    def recording_select(self: Node, *args: Any, **kwargs: Any) -> Key:
        root_puct.append((self, float(kwargs.get("puct_c", args[4] if len(args) > 4 else 0.0))))
        return real_select(self, *args, **kwargs)

    class FailedPolicy:
        def score(self, context, state, legal_actions):
            raise ComponentInferenceError("learned inference failed")

    class ProbabilityFallback:
        def score(self, context, state, legal_actions):
            actions = tuple(legal_actions)
            scores = (0.65, *(0.35 / (len(actions) - 1) for _ in actions[1:]))
            return PolicyScores(actions, scores, ScoreSemantics.PROBABILITIES)

    monkeypatch.setattr(Node, "should_expand", recording_should_expand)
    monkeypatch.setattr(Node, "select", recording_select)
    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(2),
        SearchConfig(
            iterations=5,
            cutoff_limit=0,
            leaf_mode=LeafMode.IMMEDIATE,
            seed=11,
            root_puct_c=1.5,
            root_widening_c=0.1,
            root_widening_alpha=0.25,
        ),
        FallbackSearchPolicy(FailedPolicy(), ProbabilityFallback()),
        root_target=target,
    )

    root_widening_args = [args for node, *args in root_widening if node is result.root]
    root_puct_args = [puct for node, puct in root_puct if node is result.root]
    assert root_widening_args
    assert all(args == [2.0, 0.5] for args in root_widening_args)
    assert root_puct_args == [0.0]
    assert tuple(
        item.prior_probability for item in result.root_action_diagnostics
    ) == pytest.approx((0.65, *(0.35 / (len(legal) - 1) for _ in legal[1:])))


def test_rollout_carries_concrete_card_owner_into_following_team_request() -> None:
    from automata.search.ismcts.engine import _rollout

    state, _, _ = _root()
    xargatha = state.get_hero(HeroID("hero_xargatha"))
    assert xargatha is not None
    team_request = InputRequest(
        id="team-descendant",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="team:RED",
        options=[InputOption.from_value("a"), InputOption.from_value("b")],
    )
    leaf_decision = DecisionDescriptor("INPUT", request=team_request)

    class Sim:
        our_team = TeamColor.RED

        def __init__(self) -> None:
            self.state = state

        def apply_ours(self, decision, key):
            return leaf_decision

    class Evaluator:
        def __init__(self) -> None:
            self.contexts: list[SearchContext] = []

        def evaluate(self, context, state):
            self.contexts.append(context)
            return LeafEvaluation(value=0.0)

    evaluator = Evaluator()
    _rollout(
        Sim(),  # type: ignore[arg-type]
        DecisionDescriptor("CARD", hero=xargatha),
        SearchConfig(
            cutoff_limit=1,
            cutoff_unit=CutoffUnit.DECISIONS,
            leaf_mode=LeafMode.BOUNDED_CONTINUATION,
        ),
        HeuristicAgent(2),
        evaluator,
        SearchContext("hero_wasp", TeamColor.RED, "hero_wasp"),
    )

    assert evaluator.contexts[0].current_owner_id == "hero_xargatha"
    assert evaluator.contexts[0].current_decision is leaf_decision


def test_root_puct_override_does_not_change_deeper_node_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, target, legal = _root()
    observed_puct: list[float] = []
    real_select = Node.select

    def recording_select(self: Node, *args: Any, **kwargs: Any) -> Key:
        observed_puct.append(float(kwargs.get("puct_c", args[4] if len(args) > 4 else 0.0)))
        return real_select(self, *args, **kwargs)

    monkeypatch.setattr(Node, "select", recording_select)
    search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(2),
        SearchConfig(
            iterations=3,
            cutoff_limit=0,
            leaf_mode=LeafMode.IMMEDIATE,
            seed=11,
            puct_c=0.25,
            root_puct_c=1.75,
            root_widening_c=0.1,
            root_widening_alpha=0.5,
            widening_c=0.1,
            widening_alpha=0.5,
        ),
        _CountingPolicy(),
        root_target=target,
    )

    assert observed_puct[0] == 1.75
    assert 0.25 in observed_puct[1:]


def test_root_diagnostics_preserve_caller_order_and_include_unexpanded_actions() -> None:
    state, target, legal = _root()
    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(2),
        SearchConfig(
            iterations=2,
            cutoff_limit=0,
            leaf_mode=LeafMode.IMMEDIATE,
            seed=11,
            root_puct_c=1.5,
            root_widening_c=0.1,
            root_widening_alpha=0.5,
        ),
        _CountingPolicy(),
        root_target=target,
    )

    diagnostics = result.root_action_diagnostics
    assert tuple(item.action for item in diagnostics) == legal
    assert len(diagnostics) == len(legal)
    assert sum(item.visits for item in diagnostics) == 2
    assert any(item.visits == 0 for item in diagnostics)
    expected = tuple(
        math.exp(float(len(legal) - index))
        / sum(math.exp(float(len(legal) - other)) for other in range(len(legal)))
        for index in range(len(legal))
    )
    assert tuple(item.prior_probability for item in diagnostics) == pytest.approx(expected)
    assert all(item.mean_value == 0.0 for item in diagnostics if item.visits == 0)
    assert all(item.value_variance == 0.0 for item in diagnostics if item.visits == 0)


def test_singleton_root_stays_validated_without_policy_or_simulation() -> None:
    state, _, _ = _root()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    wasp.hand = wasp.hand[:1]
    legal = (wasp.hand[0].id,)
    target = RootTarget.card(hero_id=wasp.id, owned_hero_ids=frozenset({wasp.id}))
    policy = _CountingPolicy()

    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(2),
        SearchConfig(iterations=3, seed=11),
        policy,
        root_target=target,
    )

    assert result.best_key == legal[0]
    assert policy.calls == []
    assert result.root.visits == 0
    assert result.root_action_diagnostics == (RootActionDiagnostic(legal[0], None, 0, 0.0, 0.0),)


def test_learned_root_puct_can_follow_prior_instead_of_ucb_force_trying() -> None:
    root = Node(
        visits=10,
        children={
            "favored": Node(visits=10, total_value=4.0, total_squared_value=2.0),
            "unvisited": Node(),
        },
    )
    legal = ["favored", "unvisited"]

    assert root.select(legal, 1.4, random.Random(0)) == "unvisited"
    assert (
        root.select(
            legal,
            1.4,
            random.Random(0),
            priors={"favored": 1.0, "unvisited": 0.0},
            puct_c=1.5,
        )
        == "favored"
    )


def test_slower_root_widening_concentrates_early_but_eventually_exposes_every_action() -> None:
    legal = ["high", "middle", "low", "lowest"]
    classic = Node(visits=1, children={"high": Node()})
    learned = Node(visits=1, children={"high": Node()})

    assert classic.should_expand(legal, 2.0, 0.5) is True
    assert learned.should_expand(legal, 1.0, 0.5) is False

    for visits in range(2, 20):
        learned.visits = visits
        if learned.should_expand(legal, 1.0, 0.5):
            learned.expand(legal, random.Random(0), legal)

    assert tuple(learned.children) == tuple(legal)


def test_classic_search_defaults_leave_root_overrides_disabled() -> None:
    config = SearchConfig()
    assert config.puct_c == 0.0
    assert config.root_puct_c is None
    assert config.root_widening_c is None
    assert config.root_widening_alpha is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root_puct_c", -0.01),
        ("root_puct_c", math.inf),
        ("root_widening_c", 0.0),
        ("root_widening_c", math.nan),
        ("root_widening_alpha", -0.01),
        ("root_widening_alpha", 1.01),
        ("root_widening_alpha", math.inf),
    ],
)
def test_root_schedule_overrides_reject_invalid_ranges(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        SearchConfig(**{field: value})


class _InputRootSimulator:
    request: InputRequest

    def __init__(
        self,
        state: GameState,
        our_team: TeamColor,
        environment_policy: object,
        *,
        owned_hero_ids: frozenset[str],
        cfg: SearchConfig | None = None,
    ) -> None:
        self.state = state
        self.our_team = our_team

    def advance(self) -> DecisionDescriptor:
        return DecisionDescriptor("INPUT", request=self.request)

    def advance_to_root(self, target: RootTarget) -> DecisionDescriptor:
        return self.advance()

    def apply_ours(self, decision: DecisionDescriptor, key: Key) -> DecisionDescriptor:
        return DecisionDescriptor("OVER", winner=self.our_team.value)


def _search_input_root(
    monkeypatch: pytest.MonkeyPatch,
    request: InputRequest,
    legal: tuple[Key, ...],
    cfg: SearchConfig,
    prior: object | None = None,
):
    from automata.search.ismcts import engine

    state, _, _ = _root()
    _InputRootSimulator.request = request
    target = RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({"hero_wasp", "hero_xargatha"}),
        decision_owner_hero_id="hero_wasp",
        request=request,
    )
    monkeypatch.setattr(engine, "_Simulator", _InputRootSimulator)
    monkeypatch.setattr(engine, "determinize", lambda state, viewer_id, rng: state)
    return search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(2),
        cfg,
        prior,
        root_target=target,
    )


def _search_broad_hex_root(
    monkeypatch: pytest.MonkeyPatch,
    candidate_count: int,
    cfg: SearchConfig,
    prior: object | None = None,
    *,
    can_skip: bool = False,
):
    hex_count = candidate_count - int(can_skip)
    options = [
        InputOption.from_value({"q": index, "r": 0, "s": -index}) for index in range(hex_count)
    ]
    request = InputRequest(
        id="broad-hex-root",
        request_type=InputRequestType.SELECT_HEX,
        player_id="hero_wasp",
        options=options,
        can_skip=can_skip,
    )
    legal: tuple[Key, ...] = tuple(
        [("hex", index, 0, -index) for index in range(hex_count)] + (["SKIP"] if can_skip else [])
    )
    return _search_input_root(monkeypatch, request, legal, cfg, prior)


@pytest.mark.parametrize(
    ("candidate_count", "coverage_target", "effective_iterations"),
    [(15, 4, 8), (25, 5, 10), (51, 8, 16), (100, 10, 20), (130, 12, 24)],
)
def test_adaptive_hex_root_schedule_uses_architect_coverage_formula(
    monkeypatch: pytest.MonkeyPatch,
    candidate_count: int,
    coverage_target: int,
    effective_iterations: int,
) -> None:
    result = _search_broad_hex_root(
        monkeypatch,
        candidate_count,
        SearchConfig(iterations=2, adaptive_hex_root_schedule_version=1),
    )

    assert result.requested_iterations == 2
    assert result.effective_iterations == effective_iterations
    assert result.root_coverage_target == coverage_target
    assert result.root.visits == effective_iterations
    assert len(result.root_action_diagnostics) == candidate_count
    assert sum(item.visits > 0 for item in result.root_action_diagnostics) >= coverage_target
    assert tuple(result.root.children)[:coverage_target] == tuple(
        item.action for item in result.root_action_diagnostics[:coverage_target]
    )


def test_adaptive_schedule_is_opt_in_and_ignores_narrow_or_non_hex_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = _search_broad_hex_root(monkeypatch, 15, SearchConfig(iterations=3))
    narrow = _search_broad_hex_root(
        monkeypatch,
        8,
        SearchConfig(iterations=3, adaptive_hex_root_schedule_version=1),
    )
    non_hex_request = InputRequest(
        id="broad-option-root",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value(f"option-{index}") for index in range(15)],
    )
    non_hex = _search_input_root(
        monkeypatch,
        non_hex_request,
        tuple(f"option-{index}" for index in range(15)),
        SearchConfig(iterations=3, adaptive_hex_root_schedule_version=1),
    )

    assert (default.root.visits, default.effective_iterations, default.root_coverage_target) == (
        3,
        3,
        None,
    )
    assert (narrow.root.visits, narrow.effective_iterations, narrow.root_coverage_target) == (
        3,
        3,
        None,
    )
    assert (non_hex.root.visits, non_hex.effective_iterations, non_hex.root_coverage_target) == (
        3,
        3,
        None,
    )


def test_adaptive_schedule_accepts_hex_plus_skip_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _search_broad_hex_root(
        monkeypatch,
        15,
        SearchConfig(iterations=2, adaptive_hex_root_schedule_version=1),
        can_skip=True,
    )

    assert result.effective_iterations == 8
    assert result.root_coverage_target == 4
    assert result.root_action_diagnostics[-1].action == "SKIP"


def test_adaptive_hex_coverage_uses_fallback_prior_order_without_pruning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedPolicy:
        def score(self, context, state, legal_actions):
            raise ComponentInferenceError("learned inference failed")

    class ReverseFallback:
        def score(self, context, state, legal_actions):
            actions = tuple(legal_actions)
            total = sum(range(1, len(actions) + 1))
            return PolicyScores(
                actions,
                tuple(index / total for index in range(1, len(actions) + 1)),
                ScoreSemantics.PROBABILITIES,
            )

    result = _search_broad_hex_root(
        monkeypatch,
        15,
        SearchConfig(iterations=2, adaptive_hex_root_schedule_version=1),
        FallbackSearchPolicy(FailedPolicy(), ReverseFallback()),
    )

    diagnostics = result.root_action_diagnostics
    expected_first = tuple(item.action for item in reversed(diagnostics[-4:]))
    assert tuple(result.root.children)[:4] == expected_first
    assert tuple(item.action for item in diagnostics) == tuple(
        ("hex", index, 0, -index) for index in range(15)
    )
    assert len(diagnostics) == 15


def test_adaptive_hex_root_schedule_version_is_validated() -> None:
    assert SearchConfig().adaptive_hex_root_schedule_version is None
    for invalid in (True, 1.0, 0, 2):
        with pytest.raises(ValueError, match="adaptive_hex_root_schedule_version"):
            SearchConfig(adaptive_hex_root_schedule_version=invalid)  # type: ignore[arg-type]
