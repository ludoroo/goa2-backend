from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from goa2.domain.models import GamePhase
from goa2.engine.session import SessionResult, SessionResultType
from goa2.server import app as app_module
from goa2.server import bots, replay
from goa2.server.registry import GameRegistry


def _registry_for(game):
    return SimpleNamespace(get=lambda _game_id: game)


def test_bot_winner_result_verifies_the_recorded_replay_once(monkeypatch, tmp_path) -> None:
    verify = Mock()
    monkeypatch.setattr(replay, "verify_replay_in_background", verify)
    game = SimpleNamespace(
        game_id="bot-game",
        game_logger=Mock(),
        replay_recorder=SimpleNamespace(path=tmp_path / "bot-game.jsonl"),
        session=SimpleNamespace(state=SimpleNamespace(round=1, turn=2)),
    )
    result = SessionResult(
        result_type=SessionResultType.GAME_OVER,
        current_phase=GamePhase.GAME_OVER,
        winner="RED",
    )

    bots._log_result(cast(Any, game), result)

    verify.assert_called_once_with(str(Path(tmp_path / "bot-game.jsonl")), "bot-game")


def test_stale_apply_resnapshots_instead_of_losing_the_bot_wakeup(monkeypatch) -> None:
    async def scenario() -> None:
        state = SimpleNamespace(phase=GamePhase.PLANNING, clock=None)
        game = SimpleNamespace(
            game_id="stale-apply",
            bot_specs={"hero_wasp": object()},
            removed=False,
            lock=asyncio.Lock(),
            session=SimpleNamespace(state=state),
            last_result=None,
            _bot_agents=None,
        )
        agents = {"hero_wasp": object()}
        decision = object()
        terminal = SessionResult(
            result_type=SessionResultType.GAME_OVER,
            current_phase=GamePhase.GAME_OVER,
            winner="RED",
        )
        inspect = AsyncMock(return_value=decision)
        apply = AsyncMock(
            side_effect=[
                bots._ApplyDecisionOutcome.stale(),
                bots._ApplyDecisionOutcome.success(terminal),
            ]
        )
        monkeypatch.setattr(bots, "clone_state", lambda value: value)
        monkeypatch.setattr(bots.bot_factory, "get_or_build_agents", lambda _game: agents)
        monkeypatch.setattr(
            bots.bounded_compute,
            "bounded_inspect_next_decision",
            inspect,
        )
        monkeypatch.setattr(bots, "_apply_bot_decision", apply)
        monkeypatch.setattr(bots, "_pace_before_next_bot_mutation", AsyncMock())

        await bots._bot_drive_worker(cast(Any, game), cast(Any, _registry_for(game)))

        assert inspect.await_count == 2
        assert apply.await_count == 2

    asyncio.run(scenario())


def test_failed_apply_still_stops_instead_of_tight_looping(monkeypatch) -> None:
    async def scenario() -> None:
        state = SimpleNamespace(phase=GamePhase.PLANNING, clock=None)
        game = SimpleNamespace(
            game_id="failed-apply",
            bot_specs={"hero_wasp": object()},
            removed=False,
            lock=asyncio.Lock(),
            session=SimpleNamespace(state=state),
            last_result=None,
            _bot_agents=None,
        )
        inspect = AsyncMock(return_value=object())
        apply = AsyncMock(return_value=bots._ApplyDecisionOutcome.failed())
        monkeypatch.setattr(bots, "clone_state", lambda value: value)
        monkeypatch.setattr(
            bots.bot_factory,
            "get_or_build_agents",
            lambda _game: {"hero_wasp": object()},
        )
        monkeypatch.setattr(
            bots.bounded_compute,
            "bounded_inspect_next_decision",
            inspect,
        )
        monkeypatch.setattr(bots, "_apply_bot_decision", apply)

        await bots._bot_drive_worker(cast(Any, game), cast(Any, _registry_for(game)))

        assert inspect.await_count == 1
        assert apply.await_count == 1

    asyncio.run(scenario())


def test_slow_agent_loader_does_not_block_the_event_loop_or_game_lock(monkeypatch) -> None:
    async def scenario() -> None:
        state = SimpleNamespace(phase=GamePhase.PLANNING, clock=None)
        game = SimpleNamespace(
            game_id="slow-loader",
            bot_specs={"hero_wasp": object()},
            removed=False,
            lock=asyncio.Lock(),
            session=SimpleNamespace(state=state),
            last_result=None,
            _bot_agents=None,
        )
        agents = {"hero_wasp": object()}
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        timed_out = False

        def slow_loader(build_game):
            nonlocal timed_out
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=5):
                timed_out = True
            build_game._bot_agents = agents
            return agents

        terminal = SessionResult(
            result_type=SessionResultType.GAME_OVER,
            current_phase=GamePhase.GAME_OVER,
            winner="RED",
        )
        monkeypatch.setattr(bots, "clone_state", lambda value: value)
        monkeypatch.setattr(bots.bot_factory, "get_or_build_agents", slow_loader)
        monkeypatch.setattr(
            bots.bounded_compute,
            "bounded_inspect_next_decision",
            AsyncMock(return_value=object()),
        )
        monkeypatch.setattr(
            bots,
            "_apply_bot_decision",
            AsyncMock(return_value=bots._ApplyDecisionOutcome.success(terminal)),
        )
        monkeypatch.setattr(bots, "_pace_before_next_bot_mutation", AsyncMock())

        worker = asyncio.create_task(
            bots._bot_drive_worker(cast(Any, game), cast(Any, _registry_for(game)))
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            assert not game.lock.locked()
        finally:
            release.set()
        await worker

        assert not timed_out
        assert game._bot_agents is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("registry_state", ["missing", "replaced"])
def test_actual_registry_rejects_apply_for_orphaned_game(monkeypatch, registry_state: str) -> None:
    async def scenario() -> None:
        state = SimpleNamespace(
            phase=GamePhase.PLANNING,
            clock=None,
            round=1,
            turn=1,
        )
        game = SimpleNamespace(
            game_id="orphaned-during-compute",
            bot_specs={"hero_wasp": object()},
            removed=False,
            lock=asyncio.Lock(),
            outbound_lock=asyncio.Lock(),
            session=SimpleNamespace(state=state),
            last_result=None,
        )
        registry = GameRegistry()
        if registry_state == "replaced":
            registry._games[game.game_id] = cast(Any, object())

        apply = Mock(
            return_value=SessionResult(
                result_type=SessionResultType.ACTION_COMPLETE,
                current_phase=GamePhase.PLANNING,
            )
        )
        monkeypatch.setattr(bots, "_is_decision_still_valid", lambda *_args: True)
        monkeypatch.setattr(bots, "_stop_clock_for_decision", lambda *_args: None)
        monkeypatch.setattr(bots, "_freeze_rollback_for_bot_input", lambda *_args: None)
        monkeypatch.setattr(bots, "apply_decision", apply)
        monkeypatch.setattr(bots, "_record_replay", lambda *_args: None)
        monkeypatch.setattr(bots, "_log_action_specific", lambda *_args: None)
        monkeypatch.setattr(bots, "_log_result", lambda *_args: None)
        monkeypatch.setattr(bots, "_capture_broadcast_for_result", lambda *_args: [])
        monkeypatch.setattr(bots, "finalize_timed_mutation", lambda *_args: None)

        outcome = await bots._apply_bot_decision(
            cast(Any, game),
            registry,
            cast(Any, object()),
            {"hero_wasp": cast(Any, object())},
        )

        assert outcome == bots._ApplyDecisionOutcome.failed()
        apply.assert_not_called()

    asyncio.run(scenario())


@pytest.mark.parametrize("registry_state", ["missing", "replaced"])
def test_actual_registry_rejects_idle_advance_for_orphaned_game(
    registry_state: str,
) -> None:
    async def scenario() -> None:
        advance = Mock()
        game = SimpleNamespace(
            game_id="orphaned-idle-progression",
            removed=False,
            lock=asyncio.Lock(),
            outbound_lock=asyncio.Lock(),
            session=SimpleNamespace(
                state=SimpleNamespace(phase=GamePhase.RESOLUTION, clock=None),
                advance=advance,
            ),
            last_result=None,
        )
        registry = GameRegistry()
        if registry_state == "replaced":
            registry._games[game.game_id] = cast(Any, object())

        progressed = await bots._maybe_plain_advance(
            cast(Any, game),
            registry,
            {"hero_wasp": cast(Any, object())},
        )

        assert progressed is False
        advance.assert_not_called()

    asyncio.run(scenario())


@pytest.mark.parametrize("registry_state", ["missing", "replaced"])
def test_actual_registry_rejects_agent_publication_for_orphaned_game(
    monkeypatch, registry_state: str
) -> None:
    async def scenario() -> None:
        state = SimpleNamespace(phase=GamePhase.PLANNING, clock=None)
        game = SimpleNamespace(
            game_id="orphaned-during-load",
            bot_specs={"hero_wasp": object()},
            removed=False,
            lock=asyncio.Lock(),
            session=SimpleNamespace(state=state),
            last_result=None,
            _bot_agents=None,
        )
        registry = GameRegistry()
        registry._games[game.game_id] = cast(Any, game)
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        built = {"hero_wasp": object()}

        def slow_loader(build_game):
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=5)
            build_game._bot_agents = built
            return built

        inspect = AsyncMock()
        monkeypatch.setattr(bots, "clone_state", lambda value: value)
        monkeypatch.setattr(bots.bot_factory, "get_or_build_agents", slow_loader)
        monkeypatch.setattr(
            bots.bounded_compute,
            "bounded_inspect_next_decision",
            inspect,
        )

        worker = asyncio.create_task(bots._bot_drive_worker(cast(Any, game), registry))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            if registry_state == "missing":
                registry._games.pop(game.game_id)
            else:
                registry._games[game.game_id] = cast(Any, object())
        finally:
            release.set()
        await worker

        assert game.removed is False
        assert game._bot_agents is None
        inspect.assert_not_awaited()

    asyncio.run(scenario())


def test_removed_game_does_not_publish_agents_built_from_orphan_snapshot(monkeypatch) -> None:
    async def scenario() -> None:
        state = SimpleNamespace(phase=GamePhase.PLANNING, clock=None)
        game = SimpleNamespace(
            game_id="removed-during-load",
            bot_specs={"hero_wasp": object()},
            removed=False,
            lock=asyncio.Lock(),
            session=SimpleNamespace(state=state),
            last_result=None,
            _bot_agents=None,
        )
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        built = {"hero_wasp": object()}

        def slow_loader(build_game):
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=5)
            build_game._bot_agents = built
            return built

        inspect = AsyncMock()
        monkeypatch.setattr(bots, "clone_state", lambda value: value)
        monkeypatch.setattr(bots.bot_factory, "get_or_build_agents", slow_loader)
        monkeypatch.setattr(
            bots.bounded_compute,
            "bounded_inspect_next_decision",
            inspect,
        )

        worker = asyncio.create_task(
            bots._bot_drive_worker(cast(Any, game), cast(Any, _registry_for(game)))
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            game.removed = True
        finally:
            release.set()
        await worker

        assert game._bot_agents is None
        inspect.assert_not_awaited()

    asyncio.run(scenario())


def test_reconfigured_game_discards_stale_agent_build_before_publication(monkeypatch) -> None:
    async def scenario() -> None:
        state = SimpleNamespace(phase=GamePhase.PLANNING, clock=None)
        old_spec = object()
        new_spec = object()
        old_agents = {"hero_wasp": object()}
        new_agents = {"hero_wasp": object()}
        game = SimpleNamespace(
            game_id="reconfigured-during-load",
            bot_specs={"hero_wasp": old_spec},
            removed=False,
            lock=asyncio.Lock(),
            session=SimpleNamespace(state=state),
            last_result=None,
            _bot_agents=None,
        )
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def loader(build_game):
            if build_game.bot_specs["hero_wasp"] is old_spec:
                loop.call_soon_threadsafe(started.set)
                release.wait(timeout=5)
                built = old_agents
            else:
                built = new_agents
            build_game._bot_agents = built
            return built

        terminal = SessionResult(
            result_type=SessionResultType.GAME_OVER,
            current_phase=GamePhase.GAME_OVER,
            winner="RED",
        )
        inspect = AsyncMock(return_value=object())
        monkeypatch.setattr(bots, "clone_state", lambda value: value)
        monkeypatch.setattr(bots.bot_factory, "get_or_build_agents", loader)
        monkeypatch.setattr(
            bots.bounded_compute,
            "bounded_inspect_next_decision",
            inspect,
        )
        monkeypatch.setattr(
            bots,
            "_apply_bot_decision",
            AsyncMock(return_value=bots._ApplyDecisionOutcome.success(terminal)),
        )
        monkeypatch.setattr(bots, "_pace_before_next_bot_mutation", AsyncMock())

        worker = asyncio.create_task(
            bots._bot_drive_worker(cast(Any, game), cast(Any, _registry_for(game)))
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            game.bot_specs = {"hero_wasp": new_spec}
        finally:
            release.set()
        await worker

        assert game._bot_agents is not old_agents
        assert game._bot_agents == new_agents
        assert inspect.await_args.args[2] == new_agents

    asyncio.run(scenario())


def test_app_shutdown_drains_bots_then_timers_before_heavy_pool(monkeypatch) -> None:
    order: list[str] = []

    class FakeRegistry:
        def __init__(self, save_dir: str) -> None:
            self.save_dir = save_dir

        def restore_all(self) -> int:
            return 0

        def all_games(self) -> list[object]:
            return []

    async def no_op_registry_call(registry: object) -> None:
        del registry

    async def cancel_bots(registry: object) -> None:
        del registry
        order.append("bots")

    async def stop_timers(registry: object) -> None:
        del registry
        order.append("timers")

    async def wait_forever(*args: object) -> None:
        del args
        try:
            await asyncio.Event().wait()
        finally:
            order.append("prewarm")

    async def cleanup_forever(*args: object) -> None:
        del args
        await asyncio.Event().wait()

    monkeypatch.setattr(app_module, "GameRegistry", FakeRegistry)
    monkeypatch.setattr(app_module, "resume_timers", no_op_registry_call)
    monkeypatch.setattr(app_module, "cancel_all_bot_tasks", cancel_bots)
    monkeypatch.setattr(app_module, "stop_timers", stop_timers)
    monkeypatch.setattr(app_module, "prewarm_heavy_pool", wait_forever)
    monkeypatch.setattr(app_module, "_cleanup_loop", cleanup_forever)
    monkeypatch.setattr(app_module, "shutdown_heavy_pool", lambda: order.append("heavy_pool"))
    monkeypatch.setenv("GOA2_PREWARM_WORKERS", "1")

    async def run_lifespan() -> None:
        async with app_module.lifespan(FastAPI()):
            await asyncio.sleep(0)

    asyncio.run(run_lifespan())

    assert order.index("bots") < order.index("timers") < order.index("heavy_pool")
    assert order.index("prewarm") < order.index("heavy_pool")
