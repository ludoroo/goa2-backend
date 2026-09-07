from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

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
