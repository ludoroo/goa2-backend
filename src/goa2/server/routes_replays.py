"""Read-only replay-debugger endpoints (omniscient view).

These endpoints reconstruct *recorded* games from their on-disk `.jsonl` logs and
serve an omniscient view (`build_view(reveal_all=True)`) so a developer can step
through a game decision-by-decision to investigate a bug. They never touch the
live games in the registry.

SECURITY: because the view reveals every hand and facedown card, this router
is only registered when admin access is configured (see server/admin.py and
server/app.py): either the `GOA2_REPLAY_API` dev flag (no auth) or a
`GOA2_ADMIN_TOKEN` (bearer auth on every request, enforced by the router-level
`require_admin` dependency). When neither is set — the default — the endpoints
simply do not exist (404). No request value ever reaches `reveal_all`, which
is hard-coded `True` here.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from goa2.server import shares
from goa2.server.admin import replay_api_enabled, require_admin
from goa2.server.replay import (
    ReplayCursor,
    _replay_dir,
    index_for_round_turn,
    load_replay,
    state_body,
)
from goa2.server.share_bake import bake_in_subprocess
from goa2.server.shares import _share_dir

__all__ = ["replay_api_enabled", "router"]

router = APIRouter(prefix="/replays", tags=["replays"], dependencies=[Depends(require_admin)])


# Tiny in-process LRU of reconstructed cursors, a pure optimization: identical
# inputs always yield identical views.
#
# Each entry is keyed by game_id and stamped with the log file's (mtime_ns,
# size). A replay of a *live* game keeps growing on disk and a cursor holds the
# decision list it was built from, so a stale stamp must not be served — that is
# what froze `total_decisions` at whatever the file held when first viewed.
#
# Replay logs are append-only, so a grown file usually still starts with the
# decisions already applied. In that case the cursor is kept and its decision
# list extended in place (a ~3 ms re-parse) rather than discarded (a rebuild from
# the seed, seconds for a long game). Only a genuinely divergent prefix rebuilds.
#
# `_LOCKS` guards seeking: `ReplayCursor.seek` mutates one shared GameSession, so
# two concurrent requests for the same game would interleave and corrupt it.
# Endpoints are sync `def`, so FastAPI runs them in a threadpool and this really
# can happen.
_CACHE: OrderedDict[str, tuple[tuple[int, int], ReplayCursor]] = OrderedDict()
_CACHE_MAX = 8
_LOCKS: OrderedDict[str, threading.Lock] = OrderedDict()
_LOCKS_GUARD = threading.Lock()


def _replay_path(game_id: str) -> Path:
    """Resolve a game_id to its replay file, rejecting path traversal."""
    if not game_id or "/" in game_id or "\\" in game_id or game_id.startswith("."):
        raise HTTPException(status_code=404, detail="Replay not found")
    path = Path(_replay_dir()) / f"{game_id}.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Replay '{game_id}' not found")
    return path


def _game_lock(game_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(game_id)
        if lock is None:
            lock = _LOCKS[game_id] = threading.Lock()
        _LOCKS.move_to_end(game_id)
        while len(_LOCKS) > _CACHE_MAX * 4:
            # Evicting a lock only loses mutual exclusion for a game nobody has
            # touched in a long while; the cursor it guarded is long gone too.
            _LOCKS.popitem(last=False)
    return lock


def _get_cursor(game_id: str) -> ReplayCursor:
    """Return a ReplayCursor for the game, reusing it across appends when possible.

    Call under `_game_lock(game_id)`: the returned cursor is shared mutable state.
    """
    path = _replay_path(game_id)
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)

    cached = _CACHE.get(game_id)
    if cached is not None and cached[0] == stamp:
        _CACHE.move_to_end(game_id)
        return cached[1]

    setup, decisions = load_replay(str(path))
    cursor: ReplayCursor | None = None
    if cached is not None:
        previous = cached[1]
        applied = previous.decisions
        if decisions[: len(applied)] == applied:
            # Pure append: everything already applied to the session still holds.
            previous.decisions = decisions
            cursor = previous
    if cursor is None:
        cursor = ReplayCursor(setup, decisions)

    _CACHE[game_id] = (stamp, cursor)
    _CACHE.move_to_end(game_id)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return cursor


@router.get("")
def list_replays() -> list[dict[str, Any]]:
    """List available replays (newest first) from their setup headers."""
    directory = Path(_replay_dir())
    out: list[dict[str, Any]] = []
    if not directory.is_dir():
        return out
    for f in directory.glob("*.jsonl"):
        try:
            with open(f) as fh:
                first = fh.readline()
                num_decisions = sum(1 for line in fh if line.strip())
            header = json.loads(first)
        except (OSError, json.JSONDecodeError):
            continue
        if header.get("type") != "setup":
            continue
        out.append(
            {
                "game_id": header.get("game_id", f.stem),
                "map": header.get("map"),
                "red": header.get("red", []),
                "blue": header.get("blue", []),
                "game_type": header.get("game_type"),
                "cheats": header.get("cheats", False),
                "seed": header.get("seed"),
                "engine": header.get("engine"),
                "created_at": header.get("created_at"),
                "num_decisions": num_decisions,
            }
        )
    out.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return out


@router.get("/{game_id}")
def get_replay_meta(game_id: str) -> dict[str, Any]:
    """Return the setup header plus the full ordered decision list for a replay."""
    path = _replay_path(game_id)
    try:
        setup, decisions = load_replay(str(path))
    except ValueError as e:  # malformed log (e.g. no setup header)
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "setup": setup,
        "decisions": [
            {
                "index": i,
                "type": d.get("type"),
                "r": d.get("r"),
                "t": d.get("t"),
                "hero": d.get("hero"),
                "card": d.get("card"),
                "sel": d.get("sel"),
            }
            for i, d in enumerate(decisions)
        ],
    }


@router.post("/{game_id}/share", status_code=201)
def create_share(game_id: str) -> dict[str, Any]:
    """Bake a finished game into a shareable artifact and return its token.

    The bake runs in a child process (see server/share_bake.py): it re-simulates
    the whole game, which is seconds of GIL-holding Python, and doing that in
    this process would stall live games. The request still waits for the result,
    so the API stays synchronous — no pending shares, no polling — while the
    interpreter serving live games stays responsive.
    """
    path = _replay_path(game_id)
    try:
        load_replay(str(path))  # fail fast on a malformed log, before spawning
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Idempotent: a finished game's artifact can never change, so re-sharing
    # returns the existing link rather than baking a second copy of it.
    existing = shares.share_for_game(game_id)
    if existing is not None:
        return {"token": existing["token"], "url": f"/shared/{existing['token']}"}

    result = bake_in_subprocess(str(path), game_id, _share_dir())

    if not result["ok"]:
        reason = result["reason"]
        if reason == "unfinished":
            raise HTTPException(
                status_code=409,
                detail="Only finished games can be shared; this game has no winner yet",
            )
        if reason == "rewind":
            raise HTTPException(status_code=422, detail="Cannot share a replay containing a rewind")
        if reason == "crashed":
            raise HTTPException(status_code=500, detail="The bake process died; check server logs")
        raise HTTPException(
            status_code=422,
            detail=f"Replay reconstruction failed at decision {result['at']}: {result['error']}",
        )

    token = result["token"]
    return {"token": token, "url": f"/shared/{token}"}


@router.get("/{game_id}/state")
def get_replay_state(
    game_id: str,
    decision: int | None = Query(None, description="Apply exactly N decisions"),
    round: int | None = Query(None, description="Position at the start of round R"),
    turn: int | None = Query(None, description="With round: position at turn T"),
) -> dict[str, Any]:
    """Reconstruct the game to a position and return the omniscient view.

    Position is chosen by, in priority order: `decision` (apply N decisions),
    else `round`/`turn` (start of that moment), else the end of the game.
    """
    with _game_lock(game_id):
        cursor = _get_cursor(game_id)
        if decision is not None:
            target = decision
        elif round is not None:
            target = index_for_round_turn(cursor.decisions, round, turn)
        else:
            target = cursor.total
        target = max(0, min(target, cursor.total))

        try:
            session = cursor.seek(target)
        except ValueError as e:
            # Engine/replay drift (e.g. a recorded card no longer in hand). Surface
            # it: compare the header's `engine` sha to the current engine.
            raise HTTPException(
                status_code=422,
                detail=f"Replay reconstruction failed at decision {cursor.cursor}: {e}",
            ) from e

        return state_body(session, cursor_index=cursor.cursor, total=cursor.total)
