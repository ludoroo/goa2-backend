"""Generic process-wide bounded execution for expensive bot compute."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from automata.agents.capabilities import (
    BoundedComputeCapability,
    bounded_compute_capability,
    heuristic_fallback,
)
from automata.agents.contracts import Agent
from automata.runtime.driver import (
    BotDecision,
    IllegalBotDecisionError,
    inspect_next_decision,
    inspect_next_owner,
)
from automata.search.config import PROD_QUEUE_TIMEOUT_SECONDS, PROD_SEARCH_CONCURRENCY
from goa2.domain.state import GameState
from goa2.engine.session import SessionResult
from goa2.server import bot_factory
from goa2.server.bot_models import BotSpec, SearchSettings

if TYPE_CHECKING:
    from goa2.server.registry import ManagedGame

# Preserve the operational logger category used before bounded execution was
# extracted from ``bots``; dashboards and filters can continue matching it.
logger = logging.getLogger("goa2.server.bots")

AgentPredicate = Callable[[Agent], bool]
TimeoutResolver = Callable[[str, dict[str, Agent], dict[str, BotSpec]], float]
InspectFunction = Callable[[GameState, dict[str, Agent], SessionResult | None], BotDecision | None]
OwnerResolver = Callable[[GameState, dict[str, Agent], SessionResult | None], str | None]


@dataclass
class BoundedComputeMetrics:
    total_calls: int = 0
    fallback_queue_timeout: int = 0
    fallback_search_timeout: int = 0
    fallback_error: int = 0
    fallback_invalid_decision: int = 0
    total_queue_wait_seconds: float = 0.0
    total_search_seconds: float = 0.0
    current_queue_depth: int = 0
    peak_queue_depth: int = 0
    late_completions: int = 0


bounded_compute_metrics = BoundedComputeMetrics()


def reset_bounded_compute_metrics() -> None:
    """Reset the canonical metrics object in place."""
    defaults = BoundedComputeMetrics()
    for metric in fields(BoundedComputeMetrics):
        setattr(bounded_compute_metrics, metric.name, getattr(defaults, metric.name))


_compute_semaphore: asyncio.Semaphore | None = None
_compute_semaphore_loop: asyncio.AbstractEventLoop | None = None
_in_flight_compute_futures: set[asyncio.Future[Any]] = set()


def pending_compute_futures() -> tuple[asyncio.Future[Any], ...]:
    """Return an immutable snapshot of process-wide pending work."""
    return tuple(_in_flight_compute_futures)


def get_compute_semaphore() -> asyncio.Semaphore:
    global _compute_semaphore, _compute_semaphore_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if _compute_semaphore is None:
            _compute_semaphore = asyncio.Semaphore(PROD_SEARCH_CONCURRENCY)
            _compute_semaphore_loop = None
        return _compute_semaphore
    if _compute_semaphore is None or _compute_semaphore_loop is not loop:
        _compute_semaphore = asyncio.Semaphore(PROD_SEARCH_CONCURRENCY)
        _compute_semaphore_loop = loop
    return _compute_semaphore


def track_future(future: asyncio.Future[Any], game: ManagedGame | None = None) -> None:
    _in_flight_compute_futures.add(future)
    if game is not None:
        game._bot_search_futures.add(future)


def untrack_future(future: asyncio.Future[Any], game: ManagedGame | None = None) -> None:
    _in_flight_compute_futures.discard(future)
    if game is not None:
        game._bot_search_futures.discard(future)


def is_bounded_compute_agent(agent: Agent) -> bool:
    return bounded_compute_capability(agent) is not None


def _timeout_for_owner(owner: str, agents: dict[str, Agent], specs: dict[str, BotSpec]) -> float:
    capability = bounded_compute_capability(agents.get(owner))
    if capability is None:
        return SearchSettings().decision_timeout_seconds
    spec = specs.get(owner)
    return (
        spec.search.decision_timeout_seconds
        if spec and spec.search
        else capability.decision_timeout_seconds
    )


def fallback_agent_for_hero(
    game: ManagedGame,
    hero_id: str,
    capability: BoundedComputeCapability | None = None,
) -> Agent:
    cache = game._bot_fallback_agents
    if cache is None:
        cache = {}
        game._bot_fallback_agents = cache
    if hero_id not in cache:
        salt = int.from_bytes(hashlib.sha1(hero_id.encode()).digest()[:8], "big") & ((1 << 63) - 1)
        seed = bot_factory.game_entropy(game.game_id) ^ salt
        cache[hero_id] = (
            capability.create_fallback(seed=seed) if capability else heuristic_fallback(seed)
        )
    return cache[hero_id]


def fallback_agents(
    agents: dict[str, Agent],
    game: ManagedGame,
    *,
    agent_predicate: AgentPredicate = is_bounded_compute_agent,
) -> dict[str, Agent]:
    return {
        hero_id: (
            fallback_agent_for_hero(game, hero_id, bounded_compute_capability(agent))
            if agent_predicate(agent)
            else agent
        )
        for hero_id, agent in agents.items()
    }


def _run_inspect_next_decision(
    state: GameState, agents: dict[str, Agent], result: SessionResult | None
) -> BotDecision | None:
    return inspect_next_decision(state, agents, result)


async def _fallback_inspect(
    game: ManagedGame,
    state: GameState,
    agents: dict[str, Agent],
    result: SessionResult | None,
    *,
    agent_predicate: AgentPredicate,
    inspect_fn: InspectFunction,
    owner: str,
    reason: str,
) -> BotDecision | None:
    try:
        return await asyncio.to_thread(
            inspect_fn,
            state,
            fallback_agents(agents, game, agent_predicate=agent_predicate),
            result,
        )
    except Exception:
        logger.exception(
            "ismcts: heuristic fallback also failed game=%s owner=%s reason=%s",
            game.game_id,
            owner,
            reason,
        )
        return None


async def bounded_inspect_next_decision(
    game: ManagedGame,
    cloned_state: GameState,
    agents: dict[str, Agent],
    cloned_last_result: SessionResult | None,
    *,
    agent_predicate: AgentPredicate = is_bounded_compute_agent,
    timeout_resolver: TimeoutResolver = _timeout_for_owner,
    inspect_fn: InspectFunction = _run_inspect_next_decision,
    owner_resolver: OwnerResolver = inspect_next_owner,
    queue_timeout_seconds: float = PROD_QUEUE_TIMEOUT_SECONDS,
) -> BotDecision | None:
    """Inspect one decision without sharing injectable behavior between calls."""
    owner = owner_resolver(cloned_state, agents, cloned_last_result)
    agent = agents.get(owner) if owner is not None else None
    if agent is None or not agent_predicate(agent):
        return await asyncio.to_thread(inspect_fn, cloned_state, agents, cloned_last_result)

    assert owner is not None
    metrics = bounded_compute_metrics
    metrics.total_calls += 1
    search_timeout = timeout_resolver(owner, agents, game.bot_specs)
    semaphore = get_compute_semaphore()
    metrics.current_queue_depth += 1
    metrics.peak_queue_depth = max(metrics.peak_queue_depth, metrics.current_queue_depth)
    queue_start = time.monotonic()
    try:
        try:
            await asyncio.wait_for(semaphore.acquire(), queue_timeout_seconds)
        except TimeoutError:
            metrics.fallback_queue_timeout += 1
            queue_wait = time.monotonic() - queue_start
            logger.info(
                "ismcts: fallback=queue_timeout game=%s owner=%s queue_wait=%.3fs",
                game.game_id,
                owner,
                queue_wait,
            )
            return await _fallback_inspect(
                game,
                cloned_state,
                agents,
                cloned_last_result,
                agent_predicate=agent_predicate,
                inspect_fn=inspect_fn,
                owner=owner,
                reason="queue_timeout",
            )
    finally:
        metrics.current_queue_depth -= 1
        metrics.total_queue_wait_seconds += time.monotonic() - queue_start

    queue_wait = time.monotonic() - queue_start
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, inspect_fn, cloned_state, agents, cloned_last_result)
    track_future(future, game)
    started = time.monotonic()
    abandoned = False

    def release(done: asyncio.Future[Any]) -> None:
        try:
            semaphore.release()
        except Exception:
            logger.exception("ismcts: semaphore release raised (game=%s)", game.game_id)
        untrack_future(done, game)
        if abandoned:
            metrics.late_completions += 1
            logger.info(
                "ismcts: late_completion (dropped) game=%s owner=%s search_wall=%.3fs",
                game.game_id,
                owner,
                time.monotonic() - started,
            )

    future.add_done_callback(release)
    try:
        decision = await asyncio.wait_for(asyncio.shield(future), search_timeout)
    except TimeoutError:
        abandoned = True
        search_wait = time.monotonic() - started
        metrics.fallback_search_timeout += 1
        logger.info(
            "ismcts: fallback=search_timeout game=%s owner=%s " "search_wait=%.3fs timeout=%.3fs",
            game.game_id,
            owner,
            search_wait,
            search_timeout,
        )
        return await _fallback_inspect(
            game,
            cloned_state,
            agents,
            cloned_last_result,
            agent_predicate=agent_predicate,
            inspect_fn=inspect_fn,
            owner=owner,
            reason="search_timeout",
        )
    except asyncio.CancelledError:
        abandoned = True
        raise
    except IllegalBotDecisionError as exc:
        search_wait = time.monotonic() - started
        metrics.total_search_seconds += search_wait
        metrics.fallback_invalid_decision += 1
        logger.warning(
            "ismcts: fallback=invalid_decision game=%s owner=%s reason=%s " "search_wall=%.3fs",
            game.game_id,
            owner,
            exc.reason,
            search_wait,
        )
        return await _fallback_inspect(
            game,
            cloned_state,
            agents,
            cloned_last_result,
            agent_predicate=agent_predicate,
            inspect_fn=inspect_fn,
            owner=owner,
            reason="invalid_decision",
        )
    except Exception:
        search_wait = time.monotonic() - started
        metrics.total_search_seconds += search_wait
        metrics.fallback_error += 1
        logger.exception(
            "ismcts: fallback=error game=%s owner=%s search_wall=%.3fs",
            game.game_id,
            owner,
            search_wait,
        )
        return await _fallback_inspect(
            game,
            cloned_state,
            agents,
            cloned_last_result,
            agent_predicate=agent_predicate,
            inspect_fn=inspect_fn,
            owner=owner,
            reason="error",
        )
    search_wait = time.monotonic() - started
    metrics.total_search_seconds += search_wait
    logger.debug(
        "ismcts: ok game=%s owner=%s queue_wait=%.3fs search_wall=%.3fs",
        game.game_id,
        owner,
        queue_wait,
        search_wait,
    )
    return decision


__all__ = [
    "BoundedComputeMetrics",
    "bounded_compute_metrics",
    "bounded_inspect_next_decision",
    "fallback_agent_for_hero",
    "fallback_agents",
    "get_compute_semaphore",
    "is_bounded_compute_agent",
    "pending_compute_futures",
    "reset_bounded_compute_metrics",
    "track_future",
    "untrack_future",
]
