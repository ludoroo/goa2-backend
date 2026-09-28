import logging
from typing import Any, cast
from unittest.mock import Mock

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.runtime.effects import register_all_effects
from automata.search import SearchConfig
from automata.search.contracts import LeafEvaluation, LeafMode
from automata.search.ismcts import engine
from automata.search.ismcts.engine import (
    RootTarget,
    SearchAdvanceLimitExceeded,
    SearchDeadlineExceeded,
    search,
)
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.session import SessionResult, SessionResultType
from goa2.engine.setup import GameSetup


def _state():
    register_all_effects()
    return GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=7,
    )


def test_classic_ismcts_returns_a_legal_root_card() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    planning = ISMCTSAgent(
        SearchConfig(iterations=2, cutoff_limit=0, leaf_mode=LeafMode.IMMEDIATE, seed=4)
    ).choose_planning(state, hero)
    assert planning.card in hero.hand


def test_search_suppresses_hypothetical_engine_info_logs(caplog) -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    caplog.clear()
    caplog.set_level(logging.INFO)

    ISMCTSAgent(
        SearchConfig(iterations=1, cutoff_limit=0, leaf_mode=LeafMode.IMMEDIATE, seed=4)
    ).choose_planning(state, hero)

    assert not [
        record
        for record in caplog.records
        if record.name.startswith("goa2.engine") and record.levelno == logging.INFO
    ]

    logging.getLogger("goa2.engine.phases").info("actual gameplay")
    assert any(record.message == "actual gameplay" for record in caplog.records)


def test_search_does_not_hide_hypothetical_engine_errors(caplog) -> None:
    class ErrorLoggingPolicy(HeuristicAgent):
        def choose_planning(self, state, hero):
            logging.getLogger("goa2.engine.phases").error("hypothetical engine failure")
            raise RuntimeError("search failed")

    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    caplog.clear()
    caplog.set_level(logging.INFO)
    agent = ISMCTSAgent(
        SearchConfig(iterations=1, cutoff_limit=0, leaf_mode=LeafMode.IMMEDIATE, seed=4),
        environment_policy=ErrorLoggingPolicy(seed=4),
    )

    with pytest.raises(RuntimeError, match="search failed"):
        agent.choose_planning(state, hero)

    assert any(
        record.name == "goa2.engine.phases"
        and record.levelno == logging.ERROR
        and record.message == "hypothetical engine failure"
        for record in caplog.records
    )


def test_classic_ismcts_is_deterministic_for_a_fixed_budget_and_seed() -> None:
    chosen = []
    for _ in range(2):
        state = _state()
        hero = state.get_hero(HeroID("hero_wasp"))
        assert hero is not None
        planning = ISMCTSAgent(
            SearchConfig(iterations=3, leaf_mode=LeafMode.IMMEDIATE, seed=11)
        ).choose_planning(state, hero)
        chosen.append(planning.card.id if planning.card else None)
    assert chosen[0] == chosen[1]


def test_search_checks_its_internal_deadline_between_iterations(monkeypatch) -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    target = RootTarget.card(hero_id="hero_wasp", owned_hero_ids=frozenset({"hero_wasp"}))
    simulate = Mock()
    monkeypatch.setattr(engine, "_simulate", simulate)
    ticks = iter((10.0, 10.0, 10.0, 10.0, 10.2))
    monkeypatch.setattr(engine.time, "monotonic", lambda: next(ticks))

    with pytest.raises(SearchDeadlineExceeded, match="deadline"):
        search(
            state,
            TeamColor.RED,
            [card.id for card in hero.hand],
            HeuristicAgent(seed=1),
            SearchConfig(iterations=3, decision_timeout_seconds=0.1),
            root_target=target,
        )

    simulate.assert_called_once()


def test_deadline_returns_completed_search_visits(monkeypatch) -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    clock = [0.0]
    monkeypatch.setattr(engine.time, "monotonic", lambda: clock[0])

    class SlowValue:
        def evaluate(self, context, state):
            # Finish one full evaluation, then hit the next deadline check.
            clock[0] = 2.0
            return LeafEvaluation(value=0.25)

    result = search(
        state,
        TeamColor.RED,
        [card.id for card in hero.hand],
        HeuristicAgent(seed=1),
        SearchConfig(iterations=10, decision_timeout_seconds=2.0, leaf_mode=LeafMode.IMMEDIATE),
        root_target=RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id})),
        leaf_evaluator=SlowValue(),
    )

    assert result.root.visits == 1
    assert result.root.children[result.best_key].visits == 1


@pytest.mark.parametrize(
    "failure", [SearchDeadlineExceeded, SearchAdvanceLimitExceeded, ValueError]
)
def test_interrupted_iteration_never_hides_non_deadline_failures(failure) -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None

    class InterruptedValue:
        calls = 0

        def evaluate(self, context, state):
            self.calls += 1
            if self.calls == 2:
                raise failure("interrupted")
            return LeafEvaluation(value=0.25)

    def run():
        return search(
            state,
            TeamColor.RED,
            [card.id for card in hero.hand],
            HeuristicAgent(seed=1),
            SearchConfig(iterations=10, leaf_mode=LeafMode.IMMEDIATE),
            root_target=RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id})),
            leaf_evaluator=InterruptedValue(),
        )

    if failure is SearchDeadlineExceeded:
        result = run()
        assert result.root.visits == 1
        assert result.root.children[result.best_key].visits == 1
    else:
        with pytest.raises(failure, match="interrupted"):
            run()


def test_simulator_advance_has_a_deterministic_step_cap() -> None:
    state = _state()
    state.phase = GamePhase.RESOLUTION
    simulator = engine._Simulator(
        state,
        TeamColor.RED,
        HeuristicAgent(seed=1),
        owned_hero_ids=frozenset({"hero_wasp"}),
        max_advance_steps=3,
    )

    def advancing_result(*_args, **_kwargs):
        state.turn += 1
        return SessionResult(
            result_type=SessionResultType.ACTION_COMPLETE,
            current_phase=GamePhase.RESOLUTION,
        )

    simulator.session.advance = cast(Any, Mock(side_effect=advancing_result))

    with pytest.raises(SearchAdvanceLimitExceeded, match="advance-step limit"):
        simulator.advance()

    assert cast(Mock, simulator.session.advance).call_count == 3
