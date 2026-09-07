"""Classic server bot construction and runtime caching."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Protocol

from automata.agents.contracts import Agent
from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.agents.random_agent import RandomAgent
from automata.search.config import SearchConfig
from goa2.server.bot_models import BotSpec, SearchSettings


class AgentGame(Protocol):
    game_id: str
    bot_specs: dict[str, BotSpec]
    _bot_agents: dict[str, Agent] | None


AgentFactory = Callable[..., Agent]


def agent_for_spec(spec: BotSpec, seed: int = 0) -> Agent:
    """Build one of the classic random, heuristic, or ISMCTS agents."""
    if spec.kind == "random":
        return RandomAgent(seed=seed)
    if spec.kind == "heuristic":
        return HeuristicAgent(seed=seed)
    if spec.kind == "ismcts":
        settings = spec.search or SearchSettings()
        return ISMCTSAgent(SearchConfig(iterations=settings.iterations, seed=seed))
    raise ValueError(f"unsupported bot kind: {spec.kind!r}")


def game_entropy(game_id: str) -> int:
    """Derive stable, non-negative int64 entropy from a game id."""
    digest = hashlib.sha1(game_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def get_or_build_agents(
    game: AgentGame, *, factory: AgentFactory = agent_for_spec
) -> dict[str, Agent]:
    """Build a deterministic per-hero agent map once per managed game."""
    if game._bot_agents is not None:
        return game._bot_agents
    base = game_entropy(game.game_id)
    agents = {
        hero_id: factory(spec, seed=base ^ index)
        for index, (hero_id, spec) in enumerate(sorted(game.bot_specs.items()))
    }
    game._bot_agents = agents
    return agents


__all__ = ["agent_for_spec", "game_entropy", "get_or_build_agents"]
