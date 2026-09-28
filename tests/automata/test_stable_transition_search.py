from __future__ import annotations

from typing import Any

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import detect_stable_value_boundary
from automata.search.config import SearchConfig
from automata.search.contracts import (
    LeafEvaluation,
    LeafMode,
    StableValueContext,
)
from automata.search.fallback import FallbackLeafEvaluator
from automata.search.heuristic import HeuristicLeafEvaluator
from automata.search.ismcts import RootTarget, SearchProgressionError, search
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import CardState, GamePhase, TargetType, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.session import GameSession, SessionResult, SessionResultType
from goa2.engine.setup import GameSetup
from goa2.engine.steps import FinalizeHeroTurnStep, SelectStep


def _state(*, red: list[str] | None = None):
    register_all_effects()
    return GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        red or ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=73,
    )


class _RecordingStableLeaf(HeuristicLeafEvaluator):
    def __init__(self) -> None:
        self.records: list[tuple[StableValueContext, Any]] = []

    def evaluate_stable_value(self, context, state) -> LeafEvaluation:
        self.records.append((context, state.model_copy(deep=True)))
        return super().evaluate_stable_value(context, state)


def test_stable_transition_card_edges_reach_real_boundary_on_every_visit() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    legal = tuple(card.id for card in hero.hand)
    leaf = _RecordingStableLeaf()

    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(4),
        SearchConfig(iterations=3, leaf_mode=LeafMode.STABLE_TRANSITION, seed=9),
        root_target=RootTarget.card(
            hero_id=hero.id,
            owned_hero_ids=frozenset({hero.id}),
        ),
        leaf_evaluator=leaf,
    )

    assert result.root.visits == 3
    assert len(leaf.records) == 3
    for context, stable in leaf.records:
        assert context.root_viewer_id == "hero_wasp"
        assert context.perspective_team is TeamColor.RED
        assert detect_stable_value_boundary(stable) == context.boundary
        assert context.boundary.actor_id is not None


def test_stable_transition_requires_explicit_value_capability_even_for_singleton() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    hero.hand[:] = hero.hand[:1]

    class OrdinaryLeaf:
        def evaluate(self, _context, _state):
            pytest.fail("unsupported evaluator was invoked")

    with pytest.raises(TypeError, match="StableValueEvaluator"):
        search(
            state,
            TeamColor.RED,
            [hero.hand[0].id],
            HeuristicAgent(4),
            SearchConfig(iterations=1, leaf_mode=LeafMode.STABLE_TRANSITION),
            root_target=RootTarget.card(
                hero_id=hero.id,
                owned_hero_ids=frozenset({hero.id}),
            ),
            leaf_evaluator=OrdinaryLeaf(),  # type: ignore[arg-type]
        )


def test_stable_transition_rejects_fallback_evaluator_before_policy_inference() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None

    class FailingPolicy:
        def score(self, *_args):
            pytest.fail("policy inference ran before stable evaluator validation")

    unsupported = FallbackLeafEvaluator(HeuristicLeafEvaluator(), HeuristicLeafEvaluator())
    with pytest.raises(TypeError, match="StableValueEvaluator"):
        search(
            state,
            TeamColor.RED,
            [card.id for card in hero.hand],
            HeuristicAgent(4),
            SearchConfig(iterations=1, leaf_mode=LeafMode.STABLE_TRANSITION),
            FailingPolicy(),  # type: ignore[arg-type]
            root_target=RootTarget.card(
                hero_id=hero.id,
                owned_hero_ids=frozenset({hero.id}),
            ),
            leaf_evaluator=unsupported,
        )


def test_stable_transition_finishes_owned_emmitt_second_card_window() -> None:
    state = _state(red=["Emmitt"])
    emmitt = state.get_hero(HeroID("hero_emmitt"))
    assert emmitt is not None
    emmitt.level = 8

    class FinishContinuation:
        def __init__(self) -> None:
            self.owners: list[str] = []

        def choose(self, context, _state, decision, legal_actions):
            assert decision.kind == "CARD"
            self.owners.append(context.current_owner_id)
            assert None in legal_actions
            return None

    continuation = FinishContinuation()
    leaf = _RecordingStableLeaf()
    result = search(
        state,
        TeamColor.RED,
        [card.id for card in emmitt.hand],
        HeuristicAgent(4),
        SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TRANSITION),
        root_target=RootTarget.card(
            hero_id=emmitt.id,
            owned_hero_ids=frozenset({emmitt.id}),
        ),
        continuation_policy=continuation,
        leaf_evaluator=leaf,
    )

    assert result.root.visits == 2
    assert continuation.owners == [emmitt.id, emmitt.id]
    assert all(record[0].boundary.actor_id is not None for record in leaf.records)


def test_stable_transition_completes_real_upgrades_before_round_reset_value() -> None:
    state = _state()
    state.phase = GamePhase.RESOLUTION
    state.execution_stack.clear()
    state.pending_inputs.clear()
    state.turn = 4
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.unresolved_hero_ids = []
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    card = hero.hand.pop()
    card.state = CardState.UNRESOLVED
    card.is_facedown = False
    hero.current_turn_card = card
    state.pending_upgrades = {HeroID("hero_wasp"): 1, HeroID("hero_arien"): 1}
    push_steps(
        state,
        [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Last-actor choice",
                number_options=[1, 2],
                override_player_id="hero_wasp",
            ),
            FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    result = GameSession(state).advance()
    request = result.input_request
    assert request is not None

    class Environment(HeuristicAgent):
        def __init__(self) -> None:
            super().__init__(0)
            self.upgraded: list[str] = []

        def choose_input(self, current, request, **kwargs):
            selection = super().choose_input(current, request, **kwargs)
            if request.request_type is InputRequestType.UPGRADE_PHASE:
                assert request.player_id == "simultaneous"
                assert not request.options
                self.upgraded.append(selection["hero_id"])
            return selection

    class NoContinuation:
        def choose(self, *_args):
            pytest.fail("simultaneous upgrade used the continuation policy")

    environment = Environment()
    leaf = _RecordingStableLeaf()
    result = search(
        state,
        TeamColor.RED,
        (1, 2),
        environment,
        SearchConfig(iterations=3, leaf_mode=LeafMode.STABLE_TRANSITION),
        root_target=RootTarget.input(
            request_id=request.id,
            player_id=request.player_id,
            owned_hero_ids=frozenset({"hero_wasp"}),
            decision_owner_hero_id="hero_wasp",
            request=request,
        ),
        continuation_policy=NoContinuation(),
        leaf_evaluator=leaf,
    )

    assert result.root.visits == 3
    assert environment.upgraded.count("hero_wasp") == 3
    assert environment.upgraded.count("hero_arien") == 3
    assert len(leaf.records) == 3
    for context, stable in leaf.records:
        assert stable.pending_upgrades == {}
        assert stable.phase is GamePhase.PLANNING
        assert stable.round == state.round + 1
        assert stable.turn == 1
        assert context.boundary.actor_id is None


def test_emmitt_second_commit_and_retrieval_use_owned_planning_continuation() -> None:
    # This exercises the multi-card planning protocol, not a scripted card
    # effect; use the real GameSetup/session route rather than an effect runner.
    state = _state(red=["Emmitt"])
    emmitt = state.get_hero(HeroID("hero_emmitt"))
    assert emmitt is not None
    emmitt.level = 8
    decisions = []

    class CommitContinuation:
        def choose(self, context, _state, decision, legal):
            assert context.root_viewer_id == context.current_owner_id == emmitt.id
            assert context.perspective_team is TeamColor.RED
            decisions.append(decision)
            return next((key for key in legal if key is not None), None)

    leaf = _RecordingStableLeaf()
    result = search(
        state,
        TeamColor.RED,
        [card.id for card in emmitt.hand],
        HeuristicAgent(0),
        SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TRANSITION, seed=5),
        root_target=RootTarget.card(hero_id=emmitt.id, owned_hero_ids=frozenset({emmitt.id})),
        continuation_policy=CommitContinuation(),
        leaf_evaluator=leaf,
    )

    assert result.root.visits == len(leaf.records) == 2
    assert sum(decision.kind == "CARD" for decision in decisions) == 2
    assert [decision.request.request_type for decision in decisions if decision.request] == [
        InputRequestType.SELECT_CARD,
        InputRequestType.SELECT_CARD,
    ]
    assert all(context.boundary.actor_id is not None for context, _state in leaf.records)


def test_repeated_stable_transition_decision_fails_without_evaluating(monkeypatch) -> None:
    import automata.search.ismcts.engine as engine

    request = InputRequest(
        id="repeating-owned-request",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value("a"), InputOption.from_value("b")],
    )

    class RepeatingSession(GameSession):
        def advance(self, response=None, *, stop_before_step=None):
            return SessionResult(
                result_type=SessionResultType.INPUT_NEEDED,
                input_request=request,
                current_phase=self.state.phase,
            )

    class FirstContinuation:
        def choose(self, _context, _state, _decision, legal_actions):
            return legal_actions[0]

    class FailingStableLeaf(HeuristicLeafEvaluator):
        def evaluate_stable_value(self, _context, _state):
            pytest.fail("watchdog failure became a stable value")

    state = _state()
    state.phase = GamePhase.RESOLUTION
    target = RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({"hero_wasp"}),
        decision_owner_hero_id="hero_wasp",
        request=request,
    )
    monkeypatch.setattr(engine, "GameSession", RepeatingSession)

    with pytest.raises(SearchProgressionError, match="repeated stable-transition input"):
        search(
            state,
            TeamColor.RED,
            ("a", "b"),
            HeuristicAgent(4),
            SearchConfig(
                iterations=1,
                leaf_mode=LeafMode.STABLE_TRANSITION,
                max_forced_decisions=4,
            ),
            root_target=target,
            continuation_policy=FirstContinuation(),
            leaf_evaluator=FailingStableLeaf(),
        )


@pytest.mark.parametrize("version", [1, 2])
def test_stable_transition_rejects_historical_request_schedules(version: int) -> None:
    with pytest.raises(ValueError, match=r"STABLE_TRANSITION.*request_schedule_version"):
        SearchConfig(
            leaf_mode=LeafMode.STABLE_TRANSITION,
            request_schedule_version=version,
        )
