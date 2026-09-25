from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any, ClassVar

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.decision import DecisionDescriptor
from automata.models.contracts.inference import LearnedModelOutput
from automata.runtime.effects import register_all_effects
from automata.search import LEGACY_SCHEDULE_ID, REQUEST_AWARE_SCHEDULE_V1_ID
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
from automata.search.fallback import FallbackLeafEvaluator, FallbackSearchPolicy
from automata.search.heuristic import HeuristicLeafEvaluator
from automata.search.ismcts import RootActionDiagnostic, RootTarget, search
from automata.search.ismcts.engine import _ActionBoundary
from automata.search.learned import LearnedSearchPolicy
from automata.search.node import Key, Node
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import ActionType, TeamColor
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
        SearchContext(
            "hero_wasp",
            TeamColor.RED,
            "hero_wasp",
            DecisionDescriptor("CARD", hero=xargatha),
        ),
    )

    assert evaluator.contexts[0].current_owner_id == "hero_xargatha"
    assert evaluator.contexts[0].root_viewer_id == "hero_wasp"
    assert evaluator.contexts[0].perspective_team is TeamColor.RED
    assert evaluator.contexts[0].decision is leaf_decision


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

    def apply_ours(
        self,
        decision: DecisionDescriptor,
        key: Key,
        *,
        action_boundary: _ActionBoundary | None = None,
    ) -> DecisionDescriptor:
        return DecisionDescriptor("OVER", winner=self.our_team.value)


def _search_input_root(
    monkeypatch: pytest.MonkeyPatch,
    request: InputRequest,
    legal: tuple[Key, ...],
    cfg: SearchConfig,
    prior: object | None = None,
    *,
    leaf_evaluator: object | None = None,
    simulator_type: type[_InputRootSimulator] = _InputRootSimulator,
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
    monkeypatch.setattr(engine, "_Simulator", simulator_type)
    monkeypatch.setattr(engine, "determinize", lambda state, viewer_id, rng: state)
    return search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(2),
        cfg,
        prior,
        root_target=target,
        leaf_evaluator=leaf_evaluator,
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


def test_request_schedule_none_is_behavior_equivalent_to_legacy_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implicit = _search_broad_hex_root(
        monkeypatch,
        15,
        SearchConfig(iterations=5, seed=17),
    )
    explicit = _search_broad_hex_root(
        monkeypatch,
        15,
        SearchConfig(iterations=5, seed=17, request_schedule_version=None),
    )

    assert implicit.best_key == explicit.best_key
    assert implicit.root_action_diagnostics == explicit.root_action_diagnostics
    assert implicit.root.visits == explicit.root.visits
    assert tuple(implicit.root.children) == tuple(explicit.root.children)
    assert implicit.schedule_id == explicit.schedule_id == LEGACY_SCHEDULE_ID
    assert implicit.effective_leaf_mode is explicit.effective_leaf_mode
    assert implicit.request_type == explicit.request_type
    assert implicit.semantic_role is explicit.semantic_role
    assert implicit.requested_iterations == explicit.requested_iterations
    assert implicit.effective_iterations == explicit.effective_iterations
    assert implicit.root_coverage_target == explicit.root_coverage_target


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


def test_hex_plus_skip_contextual_comparison_requires_adaptive_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _search_broad_hex_root(
        monkeypatch,
        15,
        SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE),
        can_skip=True,
    )

    assert result.effective_iterations == 1
    assert result.root_coverage_target is None
    assert result.root.visits == 1
    assert len(result.root.children) == 1


def test_adaptive_schedule_accepts_hex_plus_skip_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _search_broad_hex_root(
        monkeypatch,
        15,
        SearchConfig(
            iterations=2,
            adaptive_hex_root_schedule_version=1,
            leaf_mode=LeafMode.IMMEDIATE,
        ),
        can_skip=True,
    )

    assert result.effective_iterations == 8
    assert result.root_coverage_target == 4
    assert result.root_action_diagnostics[-1].action == "SKIP"
    assert tuple(result.root.children)[:2] == (("hex", 0, 0, 0), "SKIP")


def test_contextual_hold_coverage_uses_real_visits_without_rewriting_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = InputRequest(
        id="action-root",
        request_type=InputRequestType.CHOOSE_ACTION,
        player_id="hero_wasp",
        options=[
            InputOption(id="ATTACK", text="Attack", metadata={"type": ActionType.ATTACK}),
            InputOption(id="HOLD", text="Hold", metadata={"type": ActionType.HOLD}),
        ],
    )

    class ProbabilityPolicy:
        def score(self, context, state, legal_actions):
            actions = tuple(legal_actions)
            return PolicyScores(actions, (0.1, 0.9), ScoreSemantics.PROBABILITIES)

    result = _search_input_root(
        monkeypatch,
        request,
        ("ATTACK", "HOLD"),
        SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE),
        ProbabilityPolicy(),
    )

    assert result.requested_iterations == 1
    assert result.effective_iterations == 2
    assert result.root_coverage_target == 2
    assert tuple(result.root.children) == ("ATTACK", "HOLD")
    assert tuple(item.visits for item in result.root_action_diagnostics) == (1, 1)
    assert tuple(item.prior_probability for item in result.root_action_diagnostics) == (0.1, 0.9)


def test_adaptive_hex_skip_coverage_preserves_nonuniform_prior_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProbabilityPolicy:
        def score(self, context, state, legal_actions):
            actions = tuple(legal_actions)
            weights = tuple(float(index + 1) for index in range(len(actions)))
            total = sum(weights)
            return PolicyScores(
                actions,
                tuple(weight / total for weight in weights),
                ScoreSemantics.PROBABILITIES,
            )

    result = _search_broad_hex_root(
        monkeypatch,
        15,
        SearchConfig(
            iterations=2,
            adaptive_hex_root_schedule_version=1,
            leaf_mode=LeafMode.IMMEDIATE,
        ),
        ProbabilityPolicy(),
        can_skip=True,
    )

    diagnostics = result.root_action_diagnostics
    expected = tuple(float(index + 1) / sum(range(1, 16)) for index in range(15))
    assert tuple(item.prior_probability for item in diagnostics) == expected
    assert tuple(result.root.children)[:2] == (("hex", 13, 0, -13), "SKIP")
    assert diagnostics[13].visits > 0
    assert diagnostics[-1].visits > 0
    assert sum(item.visits > 0 for item in diagnostics) >= result.root_coverage_target
    assert sum(item.visits for item in diagnostics) == result.effective_iterations


class _PlainLeafEvaluator:
    def evaluate(self, context, state):
        return LeafEvaluation(value=0.0)


@pytest.mark.parametrize(
    ("root_request", "legal", "cfg", "leaf_evaluator"),
    [
        (
            InputRequest(
                id="unsupported",
                request_type=InputRequestType.CHOOSE_RESPAWN,
                player_id="hero_wasp",
                options=[InputOption.from_value("RESPAWN"), InputOption.from_value("PASS")],
            ),
            ("RESPAWN", "PASS"),
            SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE),
            _PlainLeafEvaluator(),
        ),
        (
            InputRequest(
                id="bounded",
                request_type=InputRequestType.CHOOSE_RESPAWN,
                player_id="hero_wasp",
                options=[InputOption.from_value("RESPAWN"), InputOption.from_value("PASS")],
            ),
            ("RESPAWN", "PASS"),
            SearchConfig(iterations=1, leaf_mode=LeafMode.BOUNDED_CONTINUATION),
            None,
        ),
        (
            InputRequest(
                id="malformed-respawn",
                request_type=InputRequestType.CHOOSE_RESPAWN,
                player_id="hero_wasp",
                options=[
                    InputOption.from_value("RESPAWN"),
                    InputOption.from_value("PASS"),
                    InputOption.from_value("WAIT"),
                ],
            ),
            ("RESPAWN", "PASS", "WAIT"),
            SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE),
            None,
        ),
        (
            InputRequest(
                id="malformed-hold",
                request_type=InputRequestType.CHOOSE_ACTION,
                player_id="hero_wasp",
                options=[
                    InputOption(id="ATTACK", text="Attack", metadata={"type": ActionType.ATTACK}),
                    InputOption(id="HOLD", text="Hold", metadata={"type": ActionType.HOLD}),
                    InputOption(id="WAIT", text="Wait", metadata={"type": ActionType.HOLD}),
                ],
            ),
            ("ATTACK", "HOLD", "WAIT"),
            SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE),
            None,
        ),
    ],
    ids=("unsupported-evaluator", "bounded-continuation", "respawn-shape", "hold-shape"),
)
def test_contextual_root_coverage_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    root_request: InputRequest,
    legal: tuple[Key, ...],
    cfg: SearchConfig,
    leaf_evaluator: object | None,
) -> None:
    result = _search_input_root(
        monkeypatch,
        root_request,
        legal,
        cfg,
        leaf_evaluator=leaf_evaluator,
    )

    assert result.effective_iterations == 1
    assert result.root_coverage_target is None
    assert result.root.visits == 1
    assert len(result.root.children) == 1


def test_successful_learned_primary_does_not_force_contextual_root_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = InputRequest(
        id="learned-respawn",
        request_type=InputRequestType.CHOOSE_RESPAWN,
        player_id="hero_wasp",
        options=[InputOption.from_value("RESPAWN"), InputOption.from_value("PASS")],
    )

    class LearnedLeaf:
        def evaluate(self, context, state):
            return LeafEvaluation(value=0.25)

    result = _search_input_root(
        monkeypatch,
        request,
        ("RESPAWN", "PASS"),
        SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE),
        leaf_evaluator=FallbackLeafEvaluator(LearnedLeaf(), HeuristicLeafEvaluator()),
    )

    assert result.effective_iterations == 1
    assert result.root_coverage_target is None
    assert len(result.root.children) == 1


def test_immediate_edge_hooks_run_once_for_each_expanded_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = InputRequest(
        id="counted-action",
        request_type=InputRequestType.CHOOSE_ACTION,
        player_id="hero_wasp",
        options=[
            InputOption(id="ATTACK", text="Attack", metadata={"type": ActionType.ATTACK}),
            InputOption(id="HOLD", text="Hold", metadata={"type": ActionType.HOLD}),
        ],
    )

    class NonterminalSimulator(_InputRootSimulator):
        def apply_ours(self, decision: DecisionDescriptor, key: Key) -> DecisionDescriptor:
            return DecisionDescriptor(
                "INPUT",
                request=InputRequest(
                    id=f"after-{key}",
                    request_type=InputRequestType.SELECT_OPTION,
                    player_id="hero_wasp",
                    options=[InputOption.from_value("continue")],
                ),
            )

    class CountingEvaluator:
        immediate_edge_enabled = True
        contextual_root_coverage_enabled = True

        def __init__(self) -> None:
            self.prepared: list[Key] = []
            self.evaluated = 0

        def evaluate(self, context, state):
            raise AssertionError("expanded IMMEDIATE edges must use their prepared evaluation")

        def prepare_immediate_edge(self, context, state, decision, action):
            self.prepared.append(action)
            return None

        def evaluate_immediate_edge(self, context, state, prepared):
            self.evaluated += 1
            return LeafEvaluation(value=0.0)

    evaluator = CountingEvaluator()
    result = _search_input_root(
        monkeypatch,
        request,
        ("ATTACK", "HOLD"),
        SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE),
        leaf_evaluator=evaluator,
        simulator_type=NonterminalSimulator,
    )

    assert tuple(result.root.children) == ("ATTACK", "HOLD")
    assert evaluator.prepared == ["ATTACK", "HOLD"]
    assert evaluator.evaluated == len(evaluator.prepared)


def test_immediate_action_card_root_keeps_compound_tree_descent_and_signal() -> None:
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Brogan"],
        ["Arien"],
        game_type="QUICK",
        seed=19,
    )
    brogan = state.get_hero(HeroID("hero_brogan"))
    assert brogan is not None
    legal = tuple(card.id for card in brogan.hand)
    target = RootTarget.card(
        hero_id=brogan.id,
        owned_hero_ids=frozenset({str(brogan.id)}),
    )

    class CardSignalLeaf:
        def __init__(self) -> None:
            self.decision_kinds: list[str] = []

        def evaluate(self, context, simulated):
            decision = context.decision
            assert decision is not None
            self.decision_kinds.append(decision.kind)
            current = simulated.get_hero(HeroID("hero_brogan")).current_turn_card
            assert current is not None
            index = legal.index(current.id)
            return LeafEvaluation(value=-0.75 + index * 1.5 / max(1, len(legal) - 1))

    evaluator = CardSignalLeaf()
    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(3),
        SearchConfig(
            iterations=len(legal) + 1,
            leaf_mode=LeafMode.IMMEDIATE_ACTION,
            root_widening_c=100.0,
            seed=13,
        ),
        root_target=target,
        leaf_evaluator=evaluator,
    )

    assert set(evaluator.decision_kinds) == {"INPUT"}
    assert all(item.visits > 0 for item in result.root_action_diagnostics)
    assert len({item.mean_value for item in result.root_action_diagnostics}) > 1
    assert any(child.children for child in result.root.children.values())


def test_request_schedule_v1_hex_skip_uses_value_after_prior_ordered_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = InputRequest(
        id="narrow-hex",
        request_type=InputRequestType.SELECT_HEX,
        player_id="hero_wasp",
        options=[
            InputOption.from_value({"q": 1, "r": 0, "s": -1}),
            InputOption.from_value({"q": 2, "r": 0, "s": -2}),
        ],
        can_skip=True,
    )
    first_hex: Key = ("hex", 1, 0, -1)
    preferred_hex: Key = ("hex", 2, 0, -2)

    class SkipLockedPolicy:
        def score(self, context, state, legal_actions):
            actions = tuple(legal_actions)
            return PolicyScores(actions, (0.01, 0.09, 0.90), ScoreSemantics.PROBABILITIES)

    class OutcomeSimulator(_InputRootSimulator):
        selections: ClassVar[list[Key]] = []

        def apply_ours(
            self,
            decision: DecisionDescriptor,
            key: Key,
            *,
            action_boundary: _ActionBoundary | None = None,
        ) -> DecisionDescriptor:
            assert action_boundary == _ActionBoundary("hero_wasp", "hero_wasp", self.state.round)
            self.selections.append(key)
            return DecisionDescriptor("OVER", winner="BLUE" if key == "SKIP" else "RED")

    OutcomeSimulator.selections = []
    result = _search_input_root(
        monkeypatch,
        request,
        (first_hex, preferred_hex, "SKIP"),
        SearchConfig(
            iterations=2,
            leaf_mode=LeafMode.BOUNDED_CONTINUATION,
            request_schedule_version=1,
            root_puct_c=1.5,
            seed=5,
        ),
        SkipLockedPolicy(),
        simulator_type=OutcomeSimulator,
    )

    diagnostics = {item.action: item for item in result.root_action_diagnostics}
    assert tuple(item.prior_probability for item in result.root_action_diagnostics) == (
        0.01,
        0.09,
        0.90,
    )
    assert result.schedule_id == REQUEST_AWARE_SCHEDULE_V1_ID
    assert result.effective_leaf_mode is LeafMode.IMMEDIATE_ACTION
    assert result.effective_iterations == 16
    assert result.root_coverage_target == 3
    assert result.effective_root_puct_c == 0.0
    assert OutcomeSimulator.selections[:3] == [preferred_hex, "SKIP", first_hex]
    assert diagnostics[preferred_hex].mean_value == 1.0
    assert diagnostics["SKIP"].mean_value == 0.0
    assert result.best_key in {first_hex, preferred_hex}


def test_immediate_narrow_hex_keeps_prior_weighted_classic_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = InputRequest(
        id="legacy-narrow-hex",
        request_type=InputRequestType.SELECT_HEX,
        player_id="hero_wasp",
        options=[InputOption.from_value({"q": 1, "r": 0, "s": -1})],
        can_skip=True,
    )

    class SkipLockedPolicy:
        def score(self, context, state, legal_actions):
            actions = tuple(legal_actions)
            return PolicyScores(actions, (0.01, 0.99), ScoreSemantics.PROBABILITIES)

    result = _search_input_root(
        monkeypatch,
        request,
        (("hex", 1, 0, -1), "SKIP"),
        SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE, root_puct_c=1.5),
        SkipLockedPolicy(),
    )

    assert result.root_coverage_target is None
    assert result.effective_iterations == 1
    assert result.effective_root_puct_c == 1.5
    assert tuple(result.root.children) == ("SKIP",)


def test_request_schedule_v1_action_hold_retains_configured_puct_and_noop_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = InputRequest(
        id="action-root",
        request_type=InputRequestType.CHOOSE_ACTION,
        player_id="hero_wasp",
        options=[
            InputOption(id="ATTACK", text="Attack", metadata={"type": ActionType.ATTACK}),
            InputOption(id="HOLD", text="Hold", metadata={"type": ActionType.HOLD}),
        ],
    )
    result = _search_input_root(
        monkeypatch,
        request,
        ("ATTACK", "HOLD"),
        SearchConfig(
            iterations=2,
            leaf_mode=LeafMode.BOUNDED_CONTINUATION,
            request_schedule_version=1,
            root_puct_c=1.5,
        ),
    )
    assert result.schedule_id == REQUEST_AWARE_SCHEDULE_V1_ID
    assert result.effective_leaf_mode is LeafMode.IMMEDIATE_ACTION
    assert result.effective_iterations == 8
    assert result.root_coverage_target == 2
    assert result.effective_root_puct_c == 1.5


def test_immediate_action_non_noop_root_retains_configured_puct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = InputRequest(
        id="ordinary-options",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value("A"), InputOption.from_value("B")],
    )
    result = _search_input_root(
        monkeypatch,
        request,
        ("A", "B"),
        SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE_ACTION, root_puct_c=1.5),
    )
    assert result.root_coverage_target is None
    assert result.effective_root_puct_c == 1.5


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


@pytest.mark.parametrize(("requested_iterations", "effective_iterations"), [(1, 2), (20, 4)])
def test_request_schedule_v1_visits_both_binary_reactions_with_complete_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    requested_iterations: int,
    effective_iterations: int,
) -> None:
    request = InputRequest(
        id="defense-reaction",
        request_type=InputRequestType.DEFENSE_CARD,
        player_id="hero_wasp",
        options=[InputOption.from_value("DEFEND"), InputOption.from_value("PASS")],
    )

    result = _search_input_root(
        monkeypatch,
        request,
        ("DEFEND", "PASS"),
        SearchConfig(
            iterations=requested_iterations,
            leaf_mode=LeafMode.BOUNDED_CONTINUATION,
            request_schedule_version=1,
        ),
    )

    assert result.schedule_id == REQUEST_AWARE_SCHEDULE_V1_ID
    assert result.effective_leaf_mode is LeafMode.IMMEDIATE_ACTION
    assert result.request_type == InputRequestType.DEFENSE_CARD.value
    assert result.semantic_role == "DEFENSE_REACTION"
    assert result.requested_iterations == requested_iterations
    assert result.effective_iterations == effective_iterations
    assert result.root_coverage_target == 2
    assert tuple(item.action for item in result.root_action_diagnostics) == ("DEFEND", "PASS")
    visits = tuple(item.visits for item in result.root_action_diagnostics)
    assert sum(visits) == effective_iterations
    assert all(count > 0 for count in visits)


def test_request_schedule_v1_keeps_every_spatial_candidate_and_reaches_action_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_count = 15
    request = InputRequest(
        id="movement-root",
        request_type=InputRequestType.MOVEMENT_HEX,
        player_id="hero_wasp",
        options=[
            InputOption.from_value({"q": index, "r": 0, "s": -index})
            for index in range(candidate_count)
        ],
    )
    legal: tuple[Key, ...] = tuple(("hex", index, 0, -index) for index in range(candidate_count))

    class BoundaryCheckingSimulator(_InputRootSimulator):
        boundaries: ClassVar[list[_ActionBoundary | None]] = []

        def apply_ours(
            self,
            decision: DecisionDescriptor,
            key: Key,
            *,
            action_boundary: _ActionBoundary | None = None,
        ) -> DecisionDescriptor:
            self.boundaries.append(action_boundary)
            return DecisionDescriptor("OVER", winner=self.our_team.value)

    BoundaryCheckingSimulator.boundaries = []
    result = _search_input_root(
        monkeypatch,
        request,
        legal,
        SearchConfig(
            iterations=1,
            leaf_mode=LeafMode.BOUNDED_CONTINUATION,
            request_schedule_version=1,
            adaptive_hex_root_schedule_version=1,
        ),
        simulator_type=BoundaryCheckingSimulator,
    )

    assert result.effective_iterations == 16
    assert result.root_coverage_target == 4
    assert len(result.root_action_diagnostics) == candidate_count
    assert tuple(item.action for item in result.root_action_diagnostics) == legal
    assert len(BoundaryCheckingSimulator.boundaries) == 16
    assert all(boundary is not None for boundary in BoundaryCheckingSimulator.boundaries)
