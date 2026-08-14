"""Tests for fast state cloning (the MCTS substrate)."""

from __future__ import annotations

from automata.agents.random_agent import RandomAgent
from automata.runtime.clone import clone_state
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from goa2.domain.input import InputResponse
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.setup import GameSetup


def _fresh() -> GameState:
    register_all_effects()
    return GameSetup.create_game(
        DEFAULT_MAP, ["Wasp", "Xargatha"], ["Arien", "Brogan"], game_type="QUICK", seed=1
    )


def _play_one_round(state: GameState) -> None:
    agent = RandomAgent(3)
    session = GameSession(state)
    resp = None
    for _ in range(2000):
        if session.current_phase == GamePhase.PLANNING:
            for team in state.teams.values():
                for hero in team.heroes:
                    if session.current_phase != GamePhase.PLANNING:
                        break
                    if hero.id in state.pending_inputs:
                        continue
                    card = agent.choose_card(state, hero)
                    if card is None or not hero.hand:
                        session.pass_turn(HeroID(hero.id))
                    else:
                        session.commit_card(HeroID(hero.id), card)
            resp = None
            if state.round > 1:
                return
            continue
        result = session.advance(resp)
        resp = None
        if result.result_type == SessionResultType.GAME_OVER:
            return
        if result.result_type == SessionResultType.INPUT_NEEDED:
            req = result.input_request
            assert req is not None
            resp = InputResponse(request_id=req.id, selection=agent.choose_input(state, req))


def _positions(state: GameState) -> dict[str, tuple[int, int, int]]:
    return {k: (v.q, v.r, v.s) for k, v in state.unit_locations.items()}


def _occupancy(state: GameState) -> dict[str, str]:
    return {str(tid): str(t.occupant_id) for tid, t in state.board.tiles.items() if t.occupant_id}


def test_clone_shares_static_geometry_but_not_tiles() -> None:
    state = _fresh()
    clone = clone_state(state)
    # Static geometry shared for speed...
    assert clone.board is not state.board
    assert clone.board.tiles is not state.board.tiles
    # ...but zone geometry objects are shared (not re-copied).
    zid = next(iter(state.board.zones))
    assert clone.board.zones[zid] is state.board.zones[zid]


def test_playing_clone_does_not_mutate_original() -> None:
    state = _fresh()
    pos0, occ0 = _positions(state), _occupancy(state)
    round0, turn0 = state.round, state.turn

    clone = clone_state(state)
    _play_one_round(clone)

    # The clone advanced; the original is untouched (positions AND occupancy).
    assert clone.round > round0 or clone.winner is not None
    assert _positions(state) == pos0
    assert _occupancy(state) == occ0
    assert (state.round, state.turn) == (round0, turn0)


def test_mutating_clone_state_is_independent() -> None:
    state = _fresh()
    clone = clone_state(state)
    before = state.teams[TeamColor.BLUE].life_counters
    clone.teams[TeamColor.BLUE].life_counters -= 3
    assert state.teams[TeamColor.BLUE].life_counters == before
