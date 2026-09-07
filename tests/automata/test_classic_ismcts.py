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
