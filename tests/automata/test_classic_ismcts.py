import logging

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.runtime.effects import register_all_effects
from automata.search import SearchConfig
from automata.search.contracts import LeafMode
from goa2.domain.types import HeroID
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
