from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI

from goa2.domain.models import GamePhase
from goa2.engine.session import SessionResult, SessionResultType
from goa2.server import app as app_module
from goa2.server import bots, replay


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

        await bots._bot_drive_worker(cast(Any, game), cast(Any, object()))

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

        await bots._bot_drive_worker(cast(Any, game), cast(Any, object()))

        assert inspect.await_count == 1
        assert apply.await_count == 1

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
