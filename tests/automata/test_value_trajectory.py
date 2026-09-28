"""Actual-play boundary recording is behavior-neutral and never sees search leaves.

Raw stack fixtures isolate transition behavior; these are not card-effect tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.harness.game_runner import DEFAULT_MAP, continue_game
from automata.harness.trajectory import InMemoryRecorder
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import (
    StableValueBoundary,
    StableValueBoundaryKind,
    detect_stable_value_boundary,
)
from goa2.domain.input import InputRequestType
from goa2.domain.models import CardState, GamePhase, StepType, TargetType, TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.setup import GameSetup
from goa2.engine.steps import (
    FinalizeHeroTurnStep,
    ResolveTieBreakerStep,
    SelectStep,
    TriggerGameOverStep,
)


@dataclass
class _Record:
    boundary: StableValueBoundary
    viewers: tuple[str, ...]
    next_step: StepType | None
    observations: tuple[str, ...]


class _Observer:
    def __init__(self) -> None:
        self.records: list[_Record] = []
        self.outcomes: list[tuple[str | None, int, str]] = []
        self.state_ids: set[int] = set()

    def record_boundary(self, state, boundary, *, viewer_hero_ids):
        assert boundary == detect_stable_value_boundary(state)
        self.state_ids.add(id(state))
        observations = []
        for viewer_id in viewer_hero_ids:
            hero = state.get_hero(HeroID(viewer_id))
            assert hero is not None and hero.team is not None
            observations.append(
                encode_stable_value(
                    state,
                    boundary,
                    viewer_hero_id=viewer_id,
                    perspective_team=hero.team,
                ).model_dump_json()
            )
        self.records.append(
            _Record(
                boundary,
                viewer_hero_ids,
                state.execution_stack[-1].type if state.execution_stack else None,
                tuple(observations),
            )
        )

    def record_outcome(self, *, winner_side, rounds, reason):
        self.outcomes.append((winner_side, rounds, reason))


def _game() -> GameState:
    register_all_effects()
    return GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=61)


def _agents():
    return {"hero_wasp": HeuristicAgent(1), "hero_arien": HeuristicAgent(2)}


def _turn(state: GameState, *, next_actor: bool) -> None:
    state.phase = GamePhase.RESOLUTION
    state.pending_inputs.clear()
    state.execution_stack.clear()
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.unresolved_hero_ids = [HeroID("hero_arien")] if next_actor else []
    for hero_id in ("hero_wasp", "hero_arien") if next_actor else ("hero_wasp",):
        hero = state.get_hero(HeroID(hero_id))
        assert hero is not None
        card = hero.hand.pop()
        card.state = CardState.UNRESOLVED
        card.is_facedown = False
        hero.current_turn_card = card


def _number(owner_key: str) -> SelectStep:
    return SelectStep(
        target_type=TargetType.NUMBER,
        prompt="Transition probe",
        number_options=[1, 2],
        override_player_id_key=owner_key,
    )


def _decision_trace(recorder: InMemoryRecorder):
    return [
        (
            row["player_id"],
            row["chosen_key"],
            row["state"]["round"],
            row["state"]["turn"],
            row["state"]["phase"],
        )
        for row in recorder.decisions
    ]


def test_observing_boundaries_does_not_change_choices_ticks_or_game_state() -> None:
    plain = _game()
    observed = _game()
    plain_trace = InMemoryRecorder()
    observed_trace = InMemoryRecorder()
    observer = _Observer()

    expected = continue_game(plain, _agents(), max_steps=120, recorder=plain_trace)
    actual = continue_game(
        observed,
        _agents(),
        max_steps=120,
        recorder=observed_trace,
        boundary_observer=observer,
    )

    assert actual == expected
    assert _decision_trace(observed_trace) == _decision_trace(plain_trace)
    assert observed.entity_locations == plain.entity_locations
    assert {team: item.life_counters for team, item in observed.teams.items()} == {
        team: item.life_counters for team, item in plain.teams.items()
    }
    assert len(observer.records) > 3
    assert observer.state_ids == {id(observed)}
    assert observer.outcomes == [(actual.winner_side, actual.rounds, actual.reason)]
    first = observer.records[0]
    assert first.boundary.kind is StableValueBoundaryKind.ACTOR_READY
    assert first.viewers == ("hero_arien", "hero_wasp")
    assert first.next_step in (StepType.RESPAWN_HERO, StepType.RESOLVE_CARD)


def test_partial_planning_produces_no_value_observation() -> None:
    observer = _Observer()
    result = continue_game(_game(), _agents(), max_steps=1, boundary_observer=observer)
    assert result.reason == "max_steps"
    assert observer.records == []
    assert observer.outcomes == [(None, result.rounds, "max_steps")]


def test_attacker_and_reaction_viewers_are_deduplicated_at_real_next_actor() -> None:
    state = _game()
    _turn(state, next_actor=True)
    state.execution_context.update(attacker="hero_wasp", defender="hero_arien")
    push_steps(
        state,
        [
            _number("defender"),
            _number("defender"),
            _number("attacker"),
            FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    observer = _Observer()

    result = continue_game(state, _agents(), max_steps=4, boundary_observer=observer)

    assert result.reason == "max_steps"
    assert len(observer.records) == 1
    record = observer.records[0]
    assert record.viewers == ("hero_arien", "hero_wasp")
    assert record.boundary.actor_id == "hero_arien"
    assert record.next_step is StepType.RESOLVE_CARD
    assert len(record.observations) == 2


@pytest.mark.parametrize("turn", [1, 4])
@pytest.mark.parametrize("empty_opponent_hand", [False, True])
def test_last_actor_observations_follow_all_turn_or_round_cleanup(
    turn: int, empty_opponent_hand: bool
) -> None:
    state = _game()
    _turn(state, next_actor=False)
    if empty_opponent_hand:
        opponent = state.get_hero(HeroID("hero_arien"))
        assert opponent is not None
        opponent.discard_pile.extend(opponent.hand)
        opponent.hand.clear()
    state.turn = turn
    initial_round = state.round
    state.execution_context["attacker"] = "hero_wasp"
    push_steps(state, [_number("attacker"), FinalizeHeroTurnStep(hero_id="hero_wasp")])
    observer = _Observer()

    continue_game(state, _agents(), max_steps=30, boundary_observer=observer)

    assert observer.records
    first = observer.records[0]
    assert first.boundary.kind is StableValueBoundaryKind.PLANNING_READY
    assert first.boundary.round == initial_round + (1 if turn == 4 else 0)
    assert first.boundary.turn == (1 if turn == 4 else 2)
    assert first.boundary.actor_id is None
    assert first.next_step is None


def test_tie_choice_records_its_real_viewer_only_after_an_actor_is_selected() -> None:
    register_all_effects()
    state = GameSetup.create_game(
        DEFAULT_MAP, ["Wasp", "Xargatha"], ["Arien"], game_type="QUICK", seed=61
    )
    state.phase = GamePhase.RESOLUTION
    state.execution_stack.clear()
    state.unresolved_hero_ids = [HeroID("hero_wasp"), HeroID("hero_xargatha")]
    for hero_id in state.unresolved_hero_ids:
        hero = state.get_hero(hero_id)
        assert hero is not None
        card = hero.hand.pop()
        card.state = CardState.UNRESOLVED
        card.is_facedown = False
        hero.current_turn_card = card
    push_steps(state, [ResolveTieBreakerStep(tied_hero_ids=list(state.unresolved_hero_ids))])
    observer = _Observer()
    chosen: list[str] = []

    class Decisions:
        def record_decision(self, current, decision):
            assert decision.request.request_type is InputRequestType.CHOOSE_ACTOR
            assert current.execution_stack[-1].type is StepType.RESOLVE_TIE_BREAKER
            assert observer.records == []
            assert detect_stable_value_boundary(current) is None
            chosen.append(decision.selection)

        def record_outcome(self, **_):
            pass

    agents = {**_agents(), "hero_xargatha": HeuristicAgent(3)}
    continue_game(
        state, agents, max_steps=2, boundary_observer=observer, decision_observer=Decisions()
    )

    assert len(chosen) == len(observer.records) == 1
    record = observer.records[0]
    assert record.boundary.kind is StableValueBoundaryKind.ACTOR_READY
    assert record.boundary.actor_id == chosen[0]
    assert record.viewers == ("hero_wasp",)


def test_all_passed_planning_can_complete_without_selecting_an_actor() -> None:
    state = _game()
    for team in state.teams.values():
        for hero in team.heroes:
            for card in hero.hand:
                card.state = CardState.DISCARD
            hero.discard_pile.extend(hero.hand)
            hero.hand.clear()
    observer = _Observer()

    continue_game(state, _agents(), max_steps=2, boundary_observer=observer)

    assert len(observer.records) == 1
    record = observer.records[0]
    assert record.boundary.kind is StableValueBoundaryKind.PLANNING_READY
    assert record.boundary.round == 2
    assert record.boundary.turn == 1
    assert record.viewers == ("hero_arien", "hero_wasp")


def test_terminal_does_not_masquerade_as_a_neural_value_boundary() -> None:
    state = _game()
    _turn(state, next_actor=False)
    state.execution_context["attacker"] = "hero_wasp"
    push_steps(
        state,
        [
            _number("attacker"),
            TriggerGameOverStep(winner=TeamColor.RED, condition="boundary_test"),
        ],
    )
    observer = _Observer()

    result = continue_game(state, _agents(), max_steps=5, boundary_observer=observer)

    assert result.reason == "game_over"
    assert observer.records == []
    assert observer.outcomes == [("RED", result.rounds, "game_over")]


def test_individual_victory_keeps_raw_id_but_labels_learning_with_team_side() -> None:
    state = _game()
    _turn(state, next_actor=False)
    state.execution_stack.clear()
    push_steps(
        state,
        [
            TriggerGameOverStep(
                individual_winner_id=HeroID("hero_wasp"),
                condition="individual_victory_test",
            )
        ],
    )
    trajectory = InMemoryRecorder()
    boundary_observer = _Observer()
    decision_outcomes: list[tuple[str | None, int, str]] = []

    class Decisions:
        def record_decision(self, _state, _decision):
            pass

        def record_outcome(self, *, winner_side, rounds, reason):
            decision_outcomes.append((winner_side, rounds, reason))

    result = continue_game(
        state,
        _agents(),
        max_steps=2,
        recorder=trajectory,
        decision_observer=Decisions(),
        boundary_observer=boundary_observer,
    )

    assert result.winner == "hero_wasp"
    assert result.winner_side == "RED"
    assert trajectory.outcome == {
        "winner": "hero_wasp",
        "rounds": result.rounds,
        "reason": "game_over",
    }
    expected = [("RED", result.rounds, "game_over")]
    assert decision_outcomes == expected
    assert boundary_observer.outcomes == expected
