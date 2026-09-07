from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import cast

import pytest

from automata.agents.contracts import Agent
from automata.runtime.driver import BotDecision, IllegalBotDecisionError
from goa2.domain.state import GameState
from goa2.server import bots, bounded_compute
from goa2.server.registry import ManagedGame


def test_metrics_reset_preserves_canonical_identity() -> None:
    metrics = bounded_compute.bounded_compute_metrics
    metrics.total_calls = 3
    bounded_compute.reset_bounded_compute_metrics()
    assert bounded_compute.bounded_compute_metrics is metrics
    assert not hasattr(bots, "ismcts_metrics")
    assert metrics.total_calls == 0


def test_pending_compute_accessor_is_an_immutable_snapshot() -> None:
    async def scenario() -> None:
        future = asyncio.get_running_loop().create_future()
        bounded_compute.track_future(future)
        snapshot = bounded_compute.pending_compute_futures()
        bounded_compute.untrack_future(future)
        assert snapshot == (future,)
        assert bounded_compute.pending_compute_futures() == ()
        future.cancel()

    asyncio.run(scenario())


def test_injected_compute_seams_are_isolated_between_concurrent_calls() -> None:
    async def scenario() -> tuple[BotDecision | None, BotDecision | None]:
        game = cast(
            ManagedGame,
            cast(
                object,
                SimpleNamespace(
                    game_id="concurrent",
                    bot_specs={},
                    _bot_search_futures=set(),
                    _bot_fallback_agents=None,
                ),
            ),
        )
        agents = {"hero_wasp": cast(Agent, object())}
        first, second = cast(BotDecision, object()), cast(BotDecision, object())

        async def invoke(marker: BotDecision, timeout: float) -> BotDecision | None:
            return await bounded_compute.bounded_inspect_next_decision(
                game,
                cast(GameState, object()),
                agents,
                None,
                agent_predicate=lambda _agent: True,
                timeout_resolver=lambda *_args: timeout,
                inspect_fn=lambda *_args: marker,
                owner_resolver=lambda *_args: "hero_wasp",
                queue_timeout_seconds=timeout,
            )

        return await asyncio.gather(invoke(first, 1.0), invoke(second, 2.0))

    results = asyncio.run(scenario())

    assert results[0] is not results[1]


def test_invalid_decision_records_search_time_and_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bounded_compute.reset_bounded_compute_metrics()
    fallback_decision = cast(BotDecision, object())
    calls = 0

    def inspect(*_args: object) -> BotDecision:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IllegalBotDecisionError("hero_wasp", "card is not in hand")
        return fallback_decision

    async def scenario() -> BotDecision | None:
        game = cast(
            ManagedGame,
            cast(
                object,
                SimpleNamespace(
                    game_id="invalid-decision",
                    bot_specs={},
                    _bot_search_futures=set(),
                    _bot_fallback_agents=None,
                ),
            ),
        )
        return await bounded_compute.bounded_inspect_next_decision(
            game,
            cast(GameState, object()),
            {"hero_wasp": cast(Agent, object())},
            None,
            agent_predicate=lambda _agent: True,
            timeout_resolver=lambda *_args: 1.0,
            inspect_fn=inspect,
            owner_resolver=lambda *_args: "hero_wasp",
        )

    with caplog.at_level(logging.INFO, logger=bounded_compute.logger.name):
        result = asyncio.run(scenario())

    metrics = bounded_compute.bounded_compute_metrics
    assert result is fallback_decision
    assert metrics.fallback_invalid_decision == 1
    assert metrics.total_search_seconds > 0
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "invalid_decision" in message
        and "hero_wasp" in message
        and "card is not in hand" in message
        for message in messages
    )


def test_compute_exception_records_search_time_and_error_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bounded_compute.reset_bounded_compute_metrics()
    fallback_decision = cast(BotDecision, object())
    calls = 0

    def inspect(*_args: object) -> BotDecision:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("search exploded")
        return fallback_decision

    async def scenario() -> BotDecision | None:
        game = cast(
            ManagedGame,
            cast(
                object,
                SimpleNamespace(
                    game_id="compute-error",
                    bot_specs={},
                    _bot_search_futures=set(),
                    _bot_fallback_agents=None,
                ),
            ),
        )
        return await bounded_compute.bounded_inspect_next_decision(
            game,
            cast(GameState, object()),
            {"hero_wasp": cast(Agent, object())},
            None,
            agent_predicate=lambda _agent: True,
            timeout_resolver=lambda *_args: 1.0,
            inspect_fn=inspect,
            owner_resolver=lambda *_args: "hero_wasp",
        )

    with caplog.at_level(logging.ERROR, logger=bounded_compute.logger.name):
        result = asyncio.run(scenario())

    metrics = bounded_compute.bounded_compute_metrics
    assert result is fallback_decision
    assert metrics.fallback_error == 1
    assert metrics.total_search_seconds > 0
    assert any(
        "fallback=error" in record.getMessage() and "hero_wasp" in record.getMessage()
        for record in caplog.records
    )
