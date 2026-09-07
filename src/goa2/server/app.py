"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from goa2.draft.errors import DraftError
from goa2.server.bots import cancel_all_bot_tasks, start_bot_lifecycle
from goa2.server.draft_registry import DraftRegistry
from goa2.server.draft_ws import router as draft_ws_router
from goa2.server.errors import (
    AlreadyCommittedError,
    CardNotInHandError,
    GameNotFoundError,
    InvalidPhaseError,
    NotYourTurnError,
)
from goa2.server.registry import GameRegistry
from goa2.server.routes_bug_reports import public_router as bug_reports_router
from goa2.server.routes_draft import router as draft_router
from goa2.server.routes_games import router as games_router
from goa2.server.routes_heroes import router as heroes_router
from goa2.server.routes_overrides import router as overrides_router
from goa2.server.time_control import resume_timers, stop_timers
from goa2.server.workers import prewarm_heavy_pool, shutdown_heavy_pool
from goa2.server.ws import router as ws_router

logger = logging.getLogger(__name__)

# Async play routinely goes days between turns, and the staleness clock counts
# from the last persisted mutation — connecting, watching, and reconnecting do
# not reset it.
DEFAULT_GAME_TTL_DAYS = 5

load_dotenv()


def register_all_effects():
    """Auto-discover and import all hero effect modules."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    for script_path in scripts_dir.glob("*_effects.py"):
        module_name = f"goa2.scripts.{script_path.stem}"
        try:
            importlib.import_module(module_name)
        except Exception as e:
            logger.warning(f"Failed to load effect module {module_name}: {e}")


register_all_effects()


def _game_ttl_days() -> int:
    try:
        return int(os.environ.get("GOA2_GAME_TTL_DAYS", DEFAULT_GAME_TTL_DAYS))
    except ValueError:
        return DEFAULT_GAME_TTL_DAYS


async def _cleanup_loop(registry: GameRegistry):
    """Periodically remove stale game saves and old replay logs (30d)."""
    from goa2.server.replay import cleanup_old_replays

    while True:
        await asyncio.sleep(3600)  # Check every hour
        removed = registry.cleanup_stale_games(max_age_seconds=_game_ttl_days() * 86400)
        if removed:
            logger.info("Cleanup: removed %d stale game(s)", removed)
        # Replays outlive game saves: retained for their own TTL (default 30d)
        # so bugs reported after a game ends can still be reproduced.
        purged = cleanup_old_replays()
        if purged:
            logger.info("Cleanup: removed %d old replay log(s)", purged)


@asynccontextmanager
async def lifespan(app: FastAPI):
    save_dir = os.environ.get("GOA2_SAVE_DIR", "data/games")
    registry = GameRegistry(save_dir=save_dir)
    count = registry.restore_all()
    if count:
        logger.info("Restored %d game(s) from %s", count, save_dir)
    app.state.registry = registry
    app.state.draft_registry = DraftRegistry()
    await resume_timers(registry)
    for game in registry.all_games():
        await start_bot_lifecycle(game, registry)

    cleanup_task = asyncio.create_task(_cleanup_loop(registry))
    # Background, never awaited: a worker takes seconds to spawn, and blocking
    # startup on it would delay every restart to spare the first rewind.
    # Off by default under tests, which build hundreds of apps.
    warmup_task = (
        asyncio.create_task(prewarm_heavy_pool())
        if os.environ.get("GOA2_PREWARM_WORKERS", "1") != "0"
        else None
    )
    try:
        yield
    finally:
        # Bot workers can own timed mutations and heavy-pool work. Drain them
        # before tearing down either dependency.
        await cancel_all_bot_tasks(registry)
        await stop_timers(registry)
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        if warmup_task is not None:
            warmup_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await warmup_task
        # Pool teardown stays last so neither bot draining nor worker prewarm
        # can race a closed executor.
        shutdown_heavy_pool()


def _configure_logging() -> None:
    """Give goa2's own records a handler.

    Uvicorn configures only the ``uvicorn*`` loggers, so without this every
    logger.info here reaches a root logger with no handler and is dropped —
    including the reaper reporting which games it deleted. basicConfig is a
    no-op once any handler exists, so this defers to pytest's capture and to
    an embedding application instead of fighting either.
    """
    logging.basicConfig(format="%(levelname)s:     %(message)s")
    logging.getLogger("goa2").setLevel(os.environ.get("GOA2_LOG_LEVEL", "INFO").upper())


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(title="GoA2 API", version="0.1.0", lifespan=lifespan)

    # Routers
    app.include_router(heroes_router)
    app.include_router(games_router)
    app.include_router(overrides_router)
    app.include_router(bug_reports_router)
    app.include_router(draft_router)
    app.include_router(ws_router)
    app.include_router(draft_ws_router)

    # Admin routers (replay debugger's omniscient view + bug-report triage):
    # only mounted when the dev flag or an admin token is configured, so
    # reveal-all data is never reachable from a default server. With only the
    # token configured, every request is bearer-authenticated (server/admin.py).
    from goa2.server.admin import admin_api_enabled
    from goa2.server.routes_bug_reports import admin_router as bug_reports_admin_router
    from goa2.server.routes_replays import router as replays_router
    from goa2.server.routes_shares import admin_router as shares_admin_router
    from goa2.server.routes_shares import router as shared_router

    if admin_api_enabled():
        app.include_router(replays_router)
        app.include_router(bug_reports_admin_router)
        app.include_router(shares_admin_router)

    # Shared replays are mounted unconditionally: the share token is itself the
    # credential and recipients are not admins. Minting is available to seated
    # players through /games/{game_id}/share and to admins through /replays.
    app.include_router(shared_router)

    # CORS
    allowed_origins = os.environ.get("GOA2_CORS_ORIGINS", "").split(",")
    allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
    allowed_origin_regex = os.environ.get("GOA2_CORS_ORIGIN_REGEX", "").strip()
    if allowed_origins or allowed_origin_regex:
        cors_kwargs: dict = {
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        if allowed_origins:
            cors_kwargs["allow_origins"] = allowed_origins
        if allowed_origin_regex:
            cors_kwargs["allow_origin_regex"] = allowed_origin_regex
        app.add_middleware(CORSMiddleware, **cors_kwargs)

    # Exception handlers
    @app.exception_handler(GameNotFoundError)
    async def _game_not_found(request: Request, exc: GameNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(AlreadyCommittedError)
    async def _already_committed(request: Request, exc: AlreadyCommittedError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(CardNotInHandError)
    async def _card_not_in_hand(request: Request, exc: CardNotInHandError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidPhaseError)
    async def _invalid_phase(request: Request, exc: InvalidPhaseError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NotYourTurnError)
    async def _not_your_turn(request: Request, exc: NotYourTurnError):
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(DraftError)
    async def _draft_error(request: Request, exc: DraftError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app
