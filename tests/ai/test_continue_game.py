"""Behavioral contract for continuing an existing headless game."""

from __future__ import annotations

from typing import Any

import automata.runtime.harness as harness
from automata.agents.random_agent import RandomAgent
from automata.runtime.clone import clone_state
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup


def _state(*, seed: int = 9) -> GameState:
    return GameSetup.create_game(
        map_path=harness.DEFAULT_MAP,
        red_heroes=["Wasp"],
        blue_heroes=["Arien"],
        game_type="QUICK",
        seed=seed,
    )


def _agents(seed: int) -> dict[str, RandomAgent]:
    return {
        "hero_wasp": RandomAgent(seed=seed),
        "hero_arien": RandomAgent(seed=seed),
    }


def test_continue_game_mutates_fresh_state_but_not_clone_source() -> None:
    original = _state()
    before = original.model_dump(mode="json")
    continuation = clone_state(original)

    result = harness.continue_game(continuation, _agents(3), max_steps=1)

    assert result.reason == "max_steps"
    assert result.winner is None
    assert continuation.model_dump(mode="json") != before
    assert original.model_dump(mode="json") == before


def test_continue_game_is_deterministic_from_same_cloned_state() -> None:
    original = _state(seed=11)
    left = clone_state(original)
    right = clone_state(original)

    left_result = harness.continue_game(left, _agents(17), max_steps=40)
    right_result = harness.continue_game(right, _agents(17), max_steps=40)

    assert left_result == right_result
    assert left.model_dump(mode="json") == right.model_dump(mode="json")


def test_continue_game_resumes_partially_completed_planning() -> None:
    state = _state()
    wasp = state.teams[TeamColor.RED].heroes[0]
    GameSession(state).commit_card(HeroID(wasp.id), wasp.hand[0])
    assert HeroID("hero_arien") not in state.pending_inputs

    result = harness.continue_game(state, _agents(5), max_steps=1)

    assert result.reason == "max_steps"
    assert HeroID("hero_wasp") in state.pending_inputs
    assert HeroID("hero_arien") in state.pending_inputs


def test_continue_game_round_cap_is_relative_to_starting_round() -> None:
    state = _state()
    state.round = 6

    result = harness.continue_game(state, _agents(7), max_steps=10_000, max_rounds=1)

    assert result.reason == "max_rounds"
    assert result.winner is None
    assert result.rounds == 7


def test_continue_game_records_exactly_one_capped_outcome() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.outcomes: list[dict[str, Any]] = []

        def record_decision(self, **_: Any) -> None:
            pass

        def record_outcome(self, **outcome: Any) -> None:
            self.outcomes.append(outcome)

    recorder = Recorder()
    result = harness.continue_game(_state(), _agents(2), max_steps=2, recorder=recorder)

    assert result.reason == "max_steps"
    assert recorder.outcomes == [{"winner": None, "rounds": result.rounds, "reason": result.reason}]


def test_run_game_delegates_created_state_to_continue_game(monkeypatch: Any) -> None:
    expected = harness.RunResult(None, 1, 0, 1, "max_steps")
    seen: dict[str, Any] = {}

    def fake_continue(state: GameState, agents: Any, **kwargs: Any) -> harness.RunResult:
        seen.update(state=state, agents=agents, kwargs=kwargs)
        return expected

    monkeypatch.setattr(harness, "continue_game", fake_continue, raising=False)
    agents = _agents(4)

    result = harness.run_game(["Wasp"], ["Arien"], agents, seed=21, max_steps=1)

    assert result is expected
    assert isinstance(seen["state"], GameState)
    assert seen["agents"] is agents
    assert seen["kwargs"]["max_steps"] == 1
