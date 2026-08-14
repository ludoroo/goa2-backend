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
from goa2.server.bots import (
    cancel_all_bot_tasks,
    start_bot_lifecycle,
)
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
from goa2.server.ws import router as ws_router

logger = logging.getLogger(__name__)

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


async def _cleanup_loop(registry: GameRegistry):
    """Periodically remove stale game saves (24h) and old replay logs (30d)."""
    from goa2.server.replay import cleanup_old_replays

    while True:
        await asyncio.sleep(3600)  # Check every hour
        removed = registry.cleanup_stale_games(max_age_seconds=86400)
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

    # After timers are resumed and the event loop is fully live, run the
    # single lifecycle seam for every restored game that carries bot
    # metadata. ``start_bot_lifecycle`` acquires ``outbound_lock`` →
    # ``game.lock``, auto-readies bot heroes, and — critically — persists
    # + reconciles the clock and schedules the initial deadline task
    # *before* handing control to the bot coordinator. This closes the
    # race where a restored bot would previously compute against a clock
    # that was still ``WAITING_FOR_PLAYERS`` on disk. It must run after
    # ``resume_timers`` (which reconciles surviving deadlines) and after
    # ``restore_all`` populates the registry.
    for game in registry.all_games():
        if not game.bot_specs:
            continue
        await start_bot_lifecycle(game, registry)

    cleanup_task = asyncio.create_task(_cleanup_loop(registry))
    try:
        yield
    finally:
        await stop_timers(registry)
        # Cancel any live bot workers before we return. This runs before the
        # cleanup task is cancelled so a shutting-down coordinator does not
        # race the cleanup loop for a game removal.
        await cancel_all_bot_tasks(registry)
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


def create_app() -> FastAPI:
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
    # credential and recipients are not admins. Only minting (on the admin
    # replays router) can create one, so a server with no admin token configured
    # can serve existing shares but never make new ones.
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
