from types import SimpleNamespace
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
    target = RootTarget.card(
        hero_id="hero_wasp", owned_hero_ids=frozenset({"hero_wasp"})
    )
    validated = SimpleNamespace(legal_candidates=("card-a", "card-b"))
    monkeypatch.setattr(engine, "validate_search_root", lambda *_args, **_kwargs: validated)
    simulate = Mock()
    monkeypatch.setattr(engine, "_simulate", simulate)
    ticks = iter((10.0, 10.0, 10.0, 10.2))
    monkeypatch.setattr(engine.time, "monotonic", lambda: next(ticks))

    with pytest.raises(SearchDeadlineExceeded, match="deadline"):
        search(
            state,
            TeamColor.RED,
            ["card-a", "card-b"],
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


@pytest.mark.parametrize("failure", [SearchDeadlineExceeded, SearchAdvanceLimitExceeded, ValueError])
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
    simulator.session.advance = cast(
        Any,
        Mock(
            return_value=SessionResult(
                result_type=SessionResultType.ACTION_COMPLETE,
                current_phase=GamePhase.RESOLUTION,
            )
        ),
    )

    with pytest.raises(SearchAdvanceLimitExceeded, match="advance-step limit"):
        simulator.advance()

    assert cast(Mock, simulator.session.advance).call_count == 3
