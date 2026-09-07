from __future__ import annotations

import pytest

from automata.agents.ismcts_agent import ISMCTSAgent
from automata.search import StrategyResult
from automata.search.config import SearchConfig
from automata.search.root import RootTarget
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def _root():
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp", "Xargatha"],
        ["Arien"],
        game_type="QUICK",
        seed=4,
    )
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    target = RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id}))
    return state, hero, target


def test_strategy_result_distinguishes_a_selected_none_candidate() -> None:
    result = StrategyResult(candidates=("card", None), selected_index=1)
    assert result.selected_candidate is None
    with pytest.raises(ValueError, match="selected_index"):
        StrategyResult(candidates=("only",), selected_index=1)


class _RecordingStrategy:
    strategy_id = "recording"

    def __init__(self) -> None:
        self.target = None

    def select(self, state, team, target, candidates):
        self.target = target
        return StrategyResult(tuple(candidates), 1)


def test_ismcts_agent_delegates_to_configured_strategy() -> None:
    state, hero, target = _root()
    strategy = _RecordingStrategy()
    selected = ISMCTSAgent(SearchConfig(iterations=1), strategy=strategy).choose_planning(
        state, hero
    )
    assert strategy.target == target
    assert selected.card is hero.hand[1]
