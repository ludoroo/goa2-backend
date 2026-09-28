from __future__ import annotations

from functools import partial

import pytest

from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import (
    StableTransitionAnchor,
    StableValueBoundaryKind,
    capture_transition_anchor,
    detect_stable_value_boundary,
    should_stop_before_stable_boundary,
    transition_reached,
)
from goa2.domain.models import CardState, GamePhase, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.phases import commit_card
from goa2.engine.setup import GameSetup
from goa2.engine.steps import FinalizeHeroTurnStep


@pytest.fixture(autouse=True)
def _effects() -> None:
    register_all_effects()


def _game(red: list[str] | None = None, blue: list[str] | None = None):
    return GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        red or ["Wasp"],
        blue or ["Arien"],
        game_type="QUICK",
        seed=71,
    )


def _unique_initiative_cards(state):
    red = state.teams[TeamColor.RED].heroes[0]
    blue = state.teams[TeamColor.BLUE].heroes[0]
    for red_card in red.hand:
        for blue_card in blue.hand:
            if red_card.initiative != blue_card.initiative:
                actor = red if red_card.initiative > blue_card.initiative else blue
                return red, red_card, blue, blue_card, actor
    raise AssertionError("test heroes need cards with different initiatives")


def _give_turn_card(state, hero_id: str) -> None:
    hero = state.get_hero(HeroID(hero_id))
    assert hero is not None and hero.hand
    card = hero.hand.pop()
    card.state = CardState.UNRESOLVED
    card.is_facedown = False
    hero.current_turn_card = card


def test_clean_planning_is_detectable_but_not_a_transition_from_same_planning() -> None:
    state = _game()
    anchor = capture_transition_anchor(state)

    boundary = detect_stable_value_boundary(state)

    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.PLANNING_READY
    assert boundary.round == state.round
    assert boundary.turn == state.turn
    assert boundary.actor_id is None
    assert transition_reached(anchor, state) is None


def test_partial_commit_and_planning_done_are_not_planning_boundaries() -> None:
    state = _game(["Emmitt"], ["Wasp"])
    emmitt = state.teams[TeamColor.RED].heroes[0]
    emmitt.level = 8

    commit_card(state, HeroID(emmitt.id), emmitt.hand[0])
    assert detect_stable_value_boundary(state) is None

    state.planning_done.append(HeroID(emmitt.id))
    assert detect_stable_value_boundary(state) is None


def test_final_commit_reaches_actor_ready_before_resolve_card() -> None:
    state = _game()
    red, red_card, blue, blue_card, expected_actor = _unique_initiative_cards(state)
    commit_card(state, HeroID(red.id), red_card)
    anchor = capture_transition_anchor(state)

    commit_card(state, HeroID(blue.id), blue_card)
    result = process_stack(
        state,
        stop_before_step=partial(should_stop_before_stable_boundary, anchor),
    )

    assert result.input_request is None
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.ACTOR_READY
    assert boundary.actor_id == expected_actor.id
    assert transition_reached(anchor, state) == boundary
    assert state.execution_stack[-1].type.value == "resolve_card"


def test_defeated_actor_is_ready_before_clean_respawn_step() -> None:
    state = _game()
    red, red_card, blue, blue_card, expected_actor = _unique_initiative_cards(state)
    commit_card(state, HeroID(red.id), red_card)
    state.remove_entity(expected_actor.id)
    anchor = capture_transition_anchor(state)

    commit_card(state, HeroID(blue.id), blue_card)
    process_stack(
        state,
        stop_before_step=partial(should_stop_before_stable_boundary, anchor),
    )

    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.ACTOR_READY
    assert boundary.actor_id == expected_actor.id
    assert state.execution_stack[-1].type.value == "respawn_hero"

    # A respawn step that has already requested or received input is inside the
    # actor's decision, not the stable pre-actor boundary.
    respawn = state.execution_stack[-1]
    respawn.pending_request_id = "stale-request"
    assert detect_stable_value_boundary(state) is None
    respawn.pending_request_id = None
    respawn.pending_input = {"selection": "PASS"}
    assert detect_stable_value_boundary(state) is None


def test_resolution_anchor_skips_same_actor_and_stops_at_next_actor() -> None:
    state = _game()
    red = state.teams[TeamColor.RED].heroes[0]
    blue = state.teams[TeamColor.BLUE].heroes[0]
    _give_turn_card(state, red.id)
    _give_turn_card(state, blue.id)
    state.phase = GamePhase.RESOLUTION
    state.pending_inputs.clear()
    state.execution_stack.clear()
    state.current_actor_id = HeroID(red.id)
    state.resolution_owner_id = HeroID(red.id)
    state.unresolved_hero_ids = [HeroID(blue.id)]
    push_steps(state, [FinalizeHeroTurnStep(hero_id=red.id)])
    anchor = capture_transition_anchor(state)

    same_actor_state = state.model_copy(deep=True)
    same_actor_state.execution_stack.clear()
    from goa2.engine.steps import ResolveCardStep

    same_actor_state.execution_stack.append(ResolveCardStep(hero_id=red.id))
    assert detect_stable_value_boundary(same_actor_state) is not None
    assert transition_reached(anchor, same_actor_state) is None

    process_stack(
        state,
        stop_before_step=partial(should_stop_before_stable_boundary, anchor),
    )
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.ACTOR_READY
    assert boundary.actor_id == blue.id
    assert transition_reached(anchor, state) == boundary


@pytest.mark.parametrize("next_round", [False, True])
def test_same_actor_in_a_later_turn_is_a_new_boundary(next_round: bool) -> None:
    state = _game()
    red, red_card, blue, blue_card, _ = _unique_initiative_cards(state)
    planning = capture_transition_anchor(state)
    commit_card(state, HeroID(red.id), red_card)
    commit_card(state, HeroID(blue.id), blue_card)
    process_stack(state, stop_before_step=partial(should_stop_before_stable_boundary, planning))
    anchor = capture_transition_anchor(state)
    assert transition_reached(anchor, state) is None

    if next_round:
        state.round += 1
        state.turn = 1
    else:
        state.turn += 1

    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.actor_id == anchor.resolution_owner_id
    assert transition_reached(anchor, state) == boundary


def test_final_actor_reaches_clean_next_turn_planning_not_cleanup() -> None:
    state = _game()
    actor = state.teams[TeamColor.RED].heroes[0]
    _give_turn_card(state, actor.id)
    state.phase = GamePhase.RESOLUTION
    state.pending_inputs.clear()
    state.execution_stack.clear()
    state.current_actor_id = HeroID(actor.id)
    state.resolution_owner_id = HeroID(actor.id)
    state.unresolved_hero_ids = []
    push_steps(state, [FinalizeHeroTurnStep(hero_id=actor.id)])
    anchor = capture_transition_anchor(state)

    process_stack(
        state,
        stop_before_step=partial(should_stop_before_stable_boundary, anchor),
    )

    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.PLANNING_READY
    assert (boundary.round, boundary.turn) > (anchor.start_round, anchor.start_turn)
    assert transition_reached(anchor, state) == boundary

    state.phase = GamePhase.CLEANUP
    assert detect_stable_value_boundary(state) is None
    state.phase = GamePhase.LEVEL_UP
    assert detect_stable_value_boundary(state) is None
    state.phase = GamePhase.GAME_OVER
    assert detect_stable_value_boundary(state) is None


def test_turn_four_runs_through_cleanup_before_next_round_planning() -> None:
    state = _game()
    actor = state.teams[TeamColor.RED].heroes[0]
    _give_turn_card(state, actor.id)
    state.phase = GamePhase.RESOLUTION
    state.turn = 4
    state.pending_inputs.clear()
    state.execution_stack.clear()
    state.current_actor_id = HeroID(actor.id)
    state.resolution_owner_id = HeroID(actor.id)
    state.unresolved_hero_ids = []
    push_steps(state, [FinalizeHeroTurnStep(hero_id=actor.id)])
    anchor = capture_transition_anchor(state)
    visited_phases: list[GamePhase] = []

    def stop_at_boundary(current, step) -> bool:
        visited_phases.append(current.phase)
        return should_stop_before_stable_boundary(anchor, current, step)

    process_stack(state, stop_before_step=stop_at_boundary)

    assert GamePhase.CLEANUP in visited_phases
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.PLANNING_READY
    assert (boundary.round, boundary.turn) == (anchor.start_round + 1, 1)
    assert transition_reached(anchor, state) == boundary


def test_actorless_cleanup_anchor_accepts_next_genuine_planning_boundary() -> None:
    state = _game()
    state.phase = GamePhase.CLEANUP
    anchor = capture_transition_anchor(state)
    state.phase = GamePhase.PLANNING
    state.round += 1
    state.turn = 1

    assert transition_reached(anchor, state) == detect_stable_value_boundary(state)


def test_anchor_is_frozen_and_captures_only_transition_identity() -> None:
    state = _game()
    anchor = capture_transition_anchor(state)

    assert anchor == StableTransitionAnchor(
        start_phase=GamePhase.PLANNING,
        start_round=1,
        start_turn=1,
        resolution_owner_id=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        anchor.start_turn = 2  # type: ignore[misc]
