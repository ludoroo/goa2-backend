from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from automata.decision import DecisionDescriptor
from automata.runtime.effects import register_all_effects
from automata.search import REQUEST_AWARE_SCHEDULE_V2_ID
from automata.search.config import SearchConfig
from automata.search.contracts import LeafEvaluation, LeafMode, SearchContext
from automata.search.ismcts import legal_keys, search
from automata.search.ismcts.engine import SearchProgressionError
from automata.search.root import RootTarget
from goa2.domain.input import InputRequest, InputRequestType, selection_value
from goa2.domain.models import CardState, GamePhase, StepType, TargetType, TeamColor
from goa2.domain.models.effect import (
    ActiveEffect,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.setup import GameSetup
from goa2.engine.steps import FinalizeHeroTurnStep, SelectStep, TriggerGameOverStep


@pytest.fixture(autouse=True)
def _effects() -> None:
    register_all_effects()


def _game(red: Sequence[str], blue: Sequence[str]):
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        list(red),
        list(blue),
        game_type="QUICK",
        seed=61,
    )
    state.phase = GamePhase.RESOLUTION
    state.pending_inputs.clear()
    state.execution_stack.clear()
    return state


def _give_turn_card(state, hero_id: str, *, initiative: int | None = None) -> None:
    hero = state.get_hero(HeroID(hero_id))
    assert hero is not None and hero.hand
    card = hero.hand.pop()
    card.state = CardState.UNRESOLVED
    card.is_facedown = False
    if initiative is not None:
        card.initiative = initiative
    hero.current_turn_card = card


def _number_step(owner_key: str, prompt: str) -> SelectStep:
    return SelectStep(
        target_type=TargetType.NUMBER,
        prompt=prompt,
        number_options=[1, 2],
        override_player_id_key=owner_key,
    )


def _first_request(state) -> InputRequest:
    result = GameSession(state).advance()
    assert result.result_type is SessionResultType.INPUT_NEEDED
    assert result.input_request is not None
    return result.input_request


def _target(request: InputRequest, *, owned: frozenset[str]) -> RootTarget:
    return RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=owned,
        decision_owner_hero_id=request.player_id,
        request=request,
    )


class _FirstOptionEnvironment:
    def __init__(self) -> None:
        self.requests: list[tuple[str, InputRequestType]] = []

    def choose_input(self, _state, request: InputRequest) -> Any:
        self.requests.append((request.player_id, request.request_type))
        return selection_value(request.options[0])


class _Continuation:
    def __init__(self) -> None:
        self.requests: list[tuple[str, InputRequestType]] = []

    def choose(self, _context, _state, decision: DecisionDescriptor, legal_actions):
        request = decision.request
        assert request is not None
        self.requests.append((request.player_id, request.request_type))
        return legal_actions[0]


class _RecordingLeaf:
    def __init__(self) -> None:
        self.records: list[tuple[SearchContext, Any]] = []

    def evaluate(self, context: SearchContext, state) -> LeafEvaluation:
        self.records.append((context, state))
        return LeafEvaluation(value=0.0)


@pytest.mark.parametrize(
    ("remove_next_actor", "expected_step"),
    [(False, StepType.RESOLVE_CARD), (True, StepType.RESPAWN_HERO)],
)
def test_stable_turn_finalizes_cleanup_and_stops_before_next_actor_card(
    remove_next_actor: bool,
    expected_step: StepType,
) -> None:
    state = _game(["Wasp"], ["Arien"])
    _give_turn_card(state, "hero_wasp")
    _give_turn_card(state, "hero_arien")
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.unresolved_hero_ids = [HeroID("hero_arien")]
    if remove_next_actor:
        state.remove_entity("hero_arien")
    state.execution_context["root_owner"] = "hero_wasp"
    minion = state.teams[TeamColor.RED].minions[0]
    zone_id = state.battle_zone_for_lane(minion.lane_id)
    assert zone_id is not None
    zone = state.board.zones[zone_id]
    outside = next(
        point
        for point, tile in state.board.tiles.items()
        if point not in zone.hexes and not tile.is_terrain and tile.occupant_id is None
    )
    state.place_entity(minion.id, outside)
    push_steps(
        state,
        [
            _number_step("root_owner", "Root choice"),
            FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    request = _first_request(state)
    decision = DecisionDescriptor("INPUT", request=request)
    legal = tuple(legal_keys(decision))
    leaf = _RecordingLeaf()

    result = search(
        state,
        TeamColor.RED,
        legal,
        _FirstOptionEnvironment(),  # type: ignore[arg-type]
        SearchConfig(
            iterations=2,
            leaf_mode=LeafMode.BOUNDED_CONTINUATION,
            request_schedule_version=2,
            seed=3,
        ),
        root_target=_target(request, owned=frozenset({"hero_wasp"})),
        continuation_policy=_Continuation(),
        leaf_evaluator=leaf,
    )

    assert result.root.visits == 2
    assert result.schedule_id == REQUEST_AWARE_SCHEDULE_V2_ID
    assert result.effective_leaf_mode is LeafMode.STABLE_TURN
    assert len(leaf.records) == 2
    for context, stable in leaf.records:
        assert context.root_viewer_id == "hero_wasp"
        assert context.current_owner_id == "hero_wasp"
        assert context.decision is not None
        assert context.decision.request is not None
        assert [selection_value(option) for option in context.decision.request.options] == [
            "CONFIRM"
        ]
        assert stable.resolution_owner_id == HeroID("hero_arien")
        assert stable.current_actor_id == HeroID("hero_arien")
        assert stable.execution_stack[-1].type is expected_step
        assert stable.get_position(minion.id) in zone.hexes
        wasp = stable.get_hero(HeroID("hero_wasp"))
        assert wasp is not None and wasp.current_turn_card is None


def test_stable_turn_preserves_reaction_viewer_and_routes_prompts_by_finalization() -> None:
    state = _game(["Wasp"], ["Arien"])
    _give_turn_card(state, "hero_wasp")
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.unresolved_hero_ids = []
    state.execution_context.update(
        root_owner="hero_arien",
        owned_follow_up="hero_arien",
        foreign_follow_up="hero_wasp",
    )
    push_steps(
        state,
        [
            _number_step("root_owner", "Reaction root"),
            _number_step("owned_follow_up", "Owned continuation"),
            _number_step("foreign_follow_up", "Foreign environment"),
            FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    request = _first_request(state)
    legal = tuple(legal_keys(DecisionDescriptor("INPUT", request=request)))
    continuation = _Continuation()
    environment = _FirstOptionEnvironment()
    leaf = _RecordingLeaf()

    search(
        state,
        TeamColor.BLUE,
        legal,
        environment,  # type: ignore[arg-type]
        SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TURN, seed=5),
        root_target=_target(request, owned=frozenset({"hero_arien"})),
        continuation_policy=continuation,
        leaf_evaluator=leaf,
    )

    assert continuation.requests == [
        ("hero_arien", InputRequestType.SELECT_NUMBER),
        ("hero_arien", InputRequestType.SELECT_NUMBER),
    ]
    assert environment.requests == [
        ("hero_wasp", InputRequestType.SELECT_NUMBER),
        ("hero_wasp", InputRequestType.SELECT_NUMBER),
    ]
    for context, stable in leaf.records:
        assert context.root_viewer_id == "hero_arien"
        assert context.perspective_team is TeamColor.BLUE
        assert context.current_owner_id == "hero_wasp"
        assert context.decision is not None
        assert context.decision.request is not None
        assert context.decision.request.player_id == "hero_wasp"
        assert stable.phase in {GamePhase.PLANNING, GamePhase.CLEANUP}


def test_stable_turn_final_actor_runs_finishing_prompts_before_phase_boundary() -> None:
    state = _game(["Wasp"], ["Arien"])
    _give_turn_card(state, "hero_wasp")
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.unresolved_hero_ids = []
    state.execution_context["root_owner"] = "hero_wasp"
    state.active_effects.append(
        ActiveEffect(
            id="stable-turn-finisher",
            source_id="hero_wasp",
            effect_type=EffectType.DELAYED_TRIGGER,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_TURN,
            created_at_turn=state.turn,
            created_at_round=state.round,
            finishing_steps=[_number_step("finishing_owner", "Finishing prompt")],
        )
    )
    push_steps(
        state,
        [
            _number_step("root_owner", "Root choice"),
            FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    request = _first_request(state)
    legal = tuple(legal_keys(DecisionDescriptor("INPUT", request=request)))
    continuation = _Continuation()
    environment = _FirstOptionEnvironment()
    leaf = _RecordingLeaf()

    search(
        state,
        TeamColor.RED,
        legal,
        environment,  # type: ignore[arg-type]
        SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TURN, seed=6),
        root_target=_target(request, owned=frozenset({"hero_wasp"})),
        continuation_policy=continuation,
        leaf_evaluator=leaf,
    )

    assert continuation.requests == []
    assert environment.requests == [
        ("hero_wasp", InputRequestType.SELECT_NUMBER),
        ("hero_wasp", InputRequestType.SELECT_NUMBER),
    ]
    assert len(leaf.records) == 2
    assert all(stable.phase is GamePhase.PLANNING for _, stable in leaf.records)
    assert all(stable.turn == state.turn + 1 for _, stable in leaf.records)


def test_stable_turn_terminal_outcome_bypasses_leaf_evaluator() -> None:
    state = _game(["Wasp"], ["Arien"])
    _give_turn_card(state, "hero_wasp")
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.execution_context["root_owner"] = "hero_wasp"
    push_steps(
        state,
        [
            _number_step("root_owner", "Root choice"),
            TriggerGameOverStep(winner=TeamColor.RED, condition="TEST"),
        ],
    )
    request = _first_request(state)
    legal = tuple(legal_keys(DecisionDescriptor("INPUT", request=request)))

    class FailingLeaf:
        def evaluate(self, _context, _state):
            pytest.fail("terminal stable-turn outcome reached the leaf evaluator")

    result = search(
        state,
        TeamColor.RED,
        legal,
        _FirstOptionEnvironment(),  # type: ignore[arg-type]
        SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TURN, seed=6),
        root_target=_target(request, owned=frozenset({"hero_wasp"})),
        continuation_policy=_Continuation(),
        leaf_evaluator=FailingLeaf(),  # type: ignore[arg-type]
    )

    assert result.root.visits == 2
    assert all(item.mean_value == 1.0 for item in result.root_action_diagnostics if item.visits)


def test_stable_turn_continuation_limit_reports_owned_and_environment_inputs_separately() -> None:
    state = _game(["Wasp"], ["Arien"])
    _give_turn_card(state, "hero_wasp")
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.execution_context.update(root_owner="hero_wasp", follow_up="hero_wasp")
    push_steps(
        state,
        [
            _number_step("root_owner", "Root choice"),
            _number_step("follow_up", "First continuation"),
            _number_step("follow_up", "Second continuation"),
            FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    request = _first_request(state)
    legal = tuple(legal_keys(DecisionDescriptor("INPUT", request=request)))

    with pytest.raises(SearchProgressionError, match="stable-turn decision limit") as raised:
        search(
            state,
            TeamColor.RED,
            legal,
            _FirstOptionEnvironment(),  # type: ignore[arg-type]
            SearchConfig(
                iterations=2,
                leaf_mode=LeafMode.STABLE_TURN,
                max_forced_decisions=1,
                seed=8,
            ),
            root_target=_target(request, owned=frozenset({"hero_wasp"})),
            continuation_policy=_Continuation(),
            leaf_evaluator=_RecordingLeaf(),
        )

    assert raised.value.transition_counts["continuation_inputs"] == 1
    assert raised.value.transition_counts["environment_inputs"] == 0
    assert raised.value.root_search_plan is not None
    assert raised.value.root_search_plan.effective_leaf_mode is LeafMode.STABLE_TURN


def test_stable_turn_routes_same_team_tie_to_environment_after_finalize() -> None:
    state = _game(["Wasp", "Xargatha", "Brogan"], ["Arien"])
    _give_turn_card(state, "hero_wasp")
    _give_turn_card(state, "hero_xargatha", initiative=7)
    _give_turn_card(state, "hero_brogan", initiative=7)
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.unresolved_hero_ids = [HeroID("hero_xargatha"), HeroID("hero_brogan")]
    state.execution_context["root_owner"] = "hero_wasp"
    push_steps(
        state,
        [
            _number_step("root_owner", "Root choice"),
            FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    request = _first_request(state)
    legal = tuple(legal_keys(DecisionDescriptor("INPUT", request=request)))
    continuation = _Continuation()
    environment = _FirstOptionEnvironment()
    leaf = _RecordingLeaf()

    search(
        state,
        TeamColor.RED,
        legal,
        environment,  # type: ignore[arg-type]
        SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TURN, seed=7),
        root_target=_target(request, owned=frozenset({"hero_wasp"})),
        continuation_policy=continuation,
        leaf_evaluator=leaf,
    )

    assert continuation.requests == []
    assert environment.requests == [
        ("team:RED", InputRequestType.CHOOSE_ACTOR),
        ("team:RED", InputRequestType.CHOOSE_ACTOR),
    ]
    assert all(stable.resolution_owner_id is not None for _, stable in leaf.records)
    assert all(
        stable.execution_stack[-1].type is StepType.RESOLVE_CARD for _, stable in leaf.records
    )
