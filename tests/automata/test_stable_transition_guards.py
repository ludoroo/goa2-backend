"""Fail-closed and ownership contracts for complete-transition search.

Raw stack fixtures isolate routing and bounds, not character card effects.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.decision import DecisionDescriptor
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import detect_stable_value_boundary
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode, StableValueContext
from automata.search.heuristic import HeuristicLeafEvaluator
from automata.search.ismcts import legal_keys, search
from automata.search.ismcts.engine import (
    SearchAdvanceLimitExceeded,
    SearchDeadlineExceeded,
    SearchProgressionError,
)
from automata.search.learned import LearnedLeafEvaluator
from automata.search.root import RootTarget
from goa2.domain.models import CardState, GamePhase, TargetType, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.setup import GameSetup
from goa2.engine.steps import FinalizeHeroTurnStep, SelectStep, TriggerGameOverStep


def _game(red=None):
    register_all_effects()
    return GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        red or ["Wasp", "Xargatha"],
        ["Arien"],
        game_type="QUICK",
        seed=97,
    )


def _number(player_id):
    return SelectStep(
        target_type=TargetType.NUMBER,
        prompt="Routing probe",
        number_options=[1, 2],
        override_player_id_key=player_id,
    )


def _root(*followups, ending=None):
    state = _game()
    state.phase = GamePhase.RESOLUTION
    state.pending_inputs.clear()
    state.execution_stack.clear()
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.unresolved_hero_ids = [HeroID("hero_arien")]
    for hero_id in ("hero_wasp", "hero_arien"):
        hero = state.get_hero(HeroID(hero_id))
        card = hero.hand.pop()
        card.state = CardState.UNRESOLVED
        card.is_facedown = False
        hero.current_turn_card = card
    for player_id in ("hero_wasp", "hero_xargatha", "hero_arien", "team:RED"):
        state.execution_context[player_id] = player_id
    push_steps(
        state,
        [
            _number("hero_wasp"),
            *followups,
            ending if ending is not None else FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    result = GameSession(state).advance()
    assert result.result_type is SessionResultType.INPUT_NEEDED
    request = result.input_request
    assert request is not None
    target = RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({"hero_wasp", "hero_xargatha"}),
        decision_owner_hero_id="hero_wasp",
        request=request,
    )
    legal = tuple(legal_keys(DecisionDescriptor("INPUT", request=request)))
    return state, target, legal


class _NoValue(HeuristicLeafEvaluator):
    def evaluate(self, *_args):
        pytest.fail("terminal/failure was evaluated through a decision context")

    def evaluate_stable_value(self, *_args):
        pytest.fail("terminal/failure became a stable value")


class _FirstContinuation:
    def choose(self, _context, _state, _decision, legal):
        return legal[0]


def test_team_continuation_retains_latest_owner_without_changing_private_viewer():
    state, target, legal = _root(_number("hero_xargatha"), _number("team:RED"))
    contexts = []

    class Continuation:
        def choose(self, context, _state, _decision, actions):
            contexts.append(context)
            return actions[0]

    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(0),
        SearchConfig(iterations=1, leaf_mode=LeafMode.STABLE_TRANSITION),
        root_target=target,
        continuation_policy=Continuation(),
    )

    assert result.root.visits == 1
    assert [context.decision.request.player_id for context in contexts] == [
        "hero_xargatha",
        "team:RED",
    ]
    assert all(context.current_owner_id == "hero_xargatha" for context in contexts)
    assert all(context.root_viewer_id == "hero_wasp" for context in contexts)
    assert all(context.perspective_team is TeamColor.RED for context in contexts)


def test_foreign_and_unowned_teammate_inputs_use_environment_not_continuation():
    state, target, legal = _root(
        _number("hero_arien"),
        _number("hero_xargatha"),
        _number("team:RED"),
        _number("hero_wasp"),
    )
    target = replace(target, owned_hero_ids=frozenset({"hero_wasp"}))
    foreign = []
    owned = []

    class Environment(HeuristicAgent):
        def choose_input(self, current, request, **kwargs):
            foreign.append(request.player_id)
            return super().choose_input(current, request, **kwargs)

    class Continuation:
        def choose(self, context, _state, decision, actions):
            assert context.root_viewer_id == context.current_owner_id == "hero_wasp"
            assert context.perspective_team is TeamColor.RED
            owned.append(decision.request.player_id)
            return actions[0]

    result = search(
        state,
        TeamColor.RED,
        legal,
        Environment(0),
        SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TRANSITION),
        root_target=target,
        continuation_policy=Continuation(),
    )

    assert result.root.visits == 2
    assert foreign == ["hero_arien", "hero_xargatha"] * 2
    assert owned == ["team:RED", "hero_wasp"] * 2


@pytest.mark.parametrize("own_teammate", [False, True])
def test_teammate_planning_routes_by_ownership_not_team_membership(own_teammate):
    state = _game()
    wasp = state.get_hero(HeroID("hero_wasp"))
    foreign_planning = []
    owned_planning = []

    class Environment(HeuristicAgent):
        def choose_planning(self, current, hero):
            foreign_planning.append(hero.id)
            return super().choose_planning(current, hero)

    class Continuation:
        def choose(self, context, _state, decision, actions):
            assert context.root_viewer_id == "hero_wasp"
            assert context.perspective_team is TeamColor.RED
            if decision.kind == "CARD":
                owned_planning.append(context.current_owner_id)
                assert decision.hero.id == context.current_owner_id
            return actions[0]

    owned_ids = {"hero_wasp", "hero_xargatha"} if own_teammate else {"hero_wasp"}
    result = search(
        state,
        TeamColor.RED,
        [card.id for card in wasp.hand],
        Environment(0),
        SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TRANSITION),
        root_target=RootTarget.card(hero_id=wasp.id, owned_hero_ids=frozenset(owned_ids)),
        continuation_policy=Continuation(),
    )

    assert result.root.visits == 2
    assert foreign_planning.count("hero_arien") == 2
    assert foreign_planning.count("hero_xargatha") == (0 if own_teammate else 2)
    assert owned_planning == (["hero_xargatha"] * 2 if own_teammate else [])


def test_unknown_simultaneous_input_is_not_silently_routed_as_foreign():
    unsupported = _number("simultaneous")
    unsupported.override_player_id = "simultaneous"
    state, target, legal = _root(unsupported)

    class NoEnvironment(HeuristicAgent):
        def choose_input(self, *_args, **_kwargs):
            pytest.fail("unsupported simultaneous input reached environment policy")

    with pytest.raises(SearchProgressionError, match="unsupported simultaneous"):
        search(
            state,
            TeamColor.RED,
            legal,
            NoEnvironment(0),
            SearchConfig(iterations=1, leaf_mode=LeafMode.STABLE_TRANSITION),
            root_target=target,
            leaf_evaluator=_NoValue(),
        )


@pytest.mark.parametrize(
    ("limits", "error", "message"),
    [
        ({"max_advance_steps": 1}, SearchAdvanceLimitExceeded, "advance-step limit"),
        ({"max_advance_transitions": 1}, SearchProgressionError, "transition limit"),
        ({"max_forced_decisions": 1}, SearchProgressionError, "decision limit"),
    ],
)
def test_progression_bounds_never_publish_a_partial_leaf(limits, error, message):
    state, target, legal = _root(_number("hero_xargatha"), _number("hero_arien"))
    with pytest.raises(error, match=message):
        search(
            state,
            TeamColor.RED,
            legal,
            HeuristicAgent(0),
            SearchConfig(iterations=1, leaf_mode=LeafMode.STABLE_TRANSITION, **limits),
            root_target=target,
            continuation_policy=_FirstContinuation(),
            leaf_evaluator=_NoValue(),
        )


def test_deadline_during_continuation_is_not_a_value_boundary(monkeypatch):
    import automata.search.ismcts.engine as engine

    now = [0.0]
    monkeypatch.setattr(engine, "time", SimpleNamespace(monotonic=lambda: now[0]))
    state, target, legal = _root(_number("hero_xargatha"))

    class ExpiringContinuation:
        def choose(self, _context, _state, _decision, actions):
            now[0] = 2.0
            return actions[0]

    with pytest.raises(SearchDeadlineExceeded):
        search(
            state,
            TeamColor.RED,
            legal,
            HeuristicAgent(0),
            SearchConfig(
                iterations=1,
                leaf_mode=LeafMode.STABLE_TRANSITION,
                decision_timeout_seconds=1.0,
            ),
            root_target=target,
            continuation_policy=ExpiringContinuation(),
            leaf_evaluator=_NoValue(),
        )


@pytest.mark.parametrize(
    ("winner", "expected"),
    [({"winner": TeamColor.RED}, 1.0), ({"individual_winner_id": "hero_arien"}, 0.0)],
)
def test_terminal_transition_bypasses_candidate_free_value_on_every_visit(winner, expected):
    state, target, legal = _root(ending=TriggerGameOverStep(condition="TEST", **winner))

    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(0),
        SearchConfig(iterations=3, leaf_mode=LeafMode.STABLE_TRANSITION),
        root_target=target,
        leaf_evaluator=_NoValue(),
    )

    assert result.root.visits == 3
    assert result.root.q == expected


@pytest.mark.parametrize("singleton", [False, True])
def test_actual_learned_evaluator_is_rejected_without_invoking_the_runtime(singleton):
    state = _game()
    hero = state.get_hero(HeroID("hero_wasp"))
    if singleton:
        hero.hand[:] = hero.hand[:1]

    class UnreadRuntime:
        def evaluate(self, *_args):
            pytest.fail("unsupported learned-value runtime was invoked")

    with pytest.raises(TypeError, match="StableValueEvaluator"):
        search(
            state,
            TeamColor.RED,
            [card.id for card in hero.hand],
            HeuristicAgent(0),
            SearchConfig(iterations=1, leaf_mode=LeafMode.STABLE_TRANSITION),
            root_target=RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id})),
            leaf_evaluator=LearnedLeafEvaluator(UnreadRuntime()),
        )


def test_invalid_owned_planning_key_does_not_silently_finish_emmitt_planning():
    state = _game(red=["Emmitt"])
    hero = state.get_hero(HeroID("hero_emmitt"))
    hero.level = 8

    class InvalidContinuation:
        def choose(self, _context, _state, _decision, actions):
            assert "not-a-legal-card" not in actions
            return "not-a-legal-card"

    with pytest.raises(ValueError, match="legal"):
        search(
            state,
            TeamColor.RED,
            [card.id for card in hero.hand],
            HeuristicAgent(0),
            SearchConfig(iterations=1, leaf_mode=LeafMode.STABLE_TRANSITION),
            root_target=RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id})),
            continuation_policy=InvalidContinuation(),
        )


@pytest.mark.parametrize("owned", [True, False])
def test_noncanonical_input_selection_is_rejected_before_applying_it(owned):
    follow_up = _number("hero_xargatha" if owned else "hero_arien")
    follow_up.is_mandatory = False
    state, target, legal = _root(follow_up)

    class InvalidContinuation:
        def choose(self, _context, _state, _decision, actions):
            assert "SKIP" in actions and None not in actions
            return None

    class InvalidEnvironment(HeuristicAgent):
        def choose_input(self, _state, request, **_kwargs):
            assert request.can_skip
            return None

    with pytest.raises(ValueError, match="legal"):
        search(
            state,
            TeamColor.RED,
            legal,
            InvalidEnvironment(0),
            SearchConfig(iterations=1, leaf_mode=LeafMode.STABLE_TRANSITION),
            root_target=target,
            continuation_policy=InvalidContinuation(),
        )


@pytest.mark.parametrize("terminal", [False, True])
def test_stable_value_rejects_stale_boundaries_and_terminal_states(terminal):
    state = _game()
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    if terminal:
        state.phase = GamePhase.GAME_OVER
        state.individual_winner_id = HeroID("hero_wasp")
    else:
        boundary = replace(boundary, round=boundary.round + 1)
    context = StableValueContext("hero_wasp", TeamColor.RED, boundary)

    with pytest.raises(ValueError, match=r"boundary|terminal"):
        HeuristicLeafEvaluator().evaluate_stable_value(context, state)


def test_missing_boundary_is_rejected_even_when_the_state_has_no_boundary():
    state, _target, _legal = _root()
    context = StableValueContext("hero_wasp", TeamColor.RED, None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="boundary"):
        HeuristicLeafEvaluator().evaluate_stable_value(context, state)
