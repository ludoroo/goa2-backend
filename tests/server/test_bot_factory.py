from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

from automata.agents.ismcts_agent import ISMCTSAgent
from automata.agents.random_agent import RandomAgent
from goa2.server.bot_factory import agent_for_spec, game_entropy, get_or_build_agents
from goa2.server.bot_models import BotSpec, SearchSettings


@dataclass
class _Game:
    game_id: str = "game-123"
    bot_specs: dict[str, BotSpec] = field(default_factory=dict)
    _bot_agents: dict[str, Any] | None = None


def test_classic_factory_builds_agents_and_has_stable_entropy() -> None:
    assert isinstance(agent_for_spec(BotSpec(kind="random"), seed=7), RandomAgent)
    assert game_entropy("game-123") == 6537217396537846519


def test_ismcts_factory_passes_the_internal_decision_timeout() -> None:
    agent = agent_for_spec(
        BotSpec(
            kind="ismcts",
            search=SearchSettings(iterations=17, decision_timeout_seconds=0.75),
        ),
        seed=7,
    )

    assert isinstance(agent, ISMCTSAgent)
    assert agent._cfg.iterations == 17
    assert agent._cfg.decision_timeout_seconds == 0.75


def test_factory_cache_builds_in_stable_hero_order() -> None:
    game = _Game(bot_specs={"hero_z": BotSpec(kind="random"), "hero_a": BotSpec(kind="heuristic")})
    built = [object(), object()]
    factory = Mock(side_effect=built)
    first = get_or_build_agents(game, factory=factory)
    second = get_or_build_agents(game, factory=Mock())
    assert first is second
    assert first == {"hero_a": built[0], "hero_z": built[1]}
