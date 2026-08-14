"""Out-of-process bake driver for replay shares.

Baking a share re-simulates an entire game (measured on the deployment target:
~18 ms per decision, so ~13 s for a 700-decision game). That work is pure Python
and holds the GIL, so running it in the server process — whether inline or on a
background thread, which shares the same GIL — stalls the event loop that serves
live games over WebSocket.

So the bake runs in a **separate process**. The request still waits for it, which
keeps the API simple (no pending state, no polling, and "is this game finished?"
is still answered before responding), but the interpreter serving live games is
free the whole time.

``bake_replay_share`` is the child entry point: it must be a module-level
function taking only picklable arguments, because the pool uses the ``spawn``
start method. Spawn is chosen deliberately over fork — forking a process that
already has a threadpool running is a well-known source of deadlocks.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from typing import Any

__all__ = ["BakeResult", "bake_in_subprocess", "bake_replay_share"]

# What the child hands back. Plain data so it survives pickling, and so engine
# exceptions never have to cross the process boundary as objects.
BakeResult = dict[str, Any]


def bake_replay_share(replay_path: str, game_id: str, share_dir: str) -> BakeResult:
    """Reconstruct a replay and bake every position. Runs in a child process.

    Returns one of:
      {"ok": True, "token": "..."}                     baked and published
      {"ok": False, "reason": "unfinished"}            game has no winner yet
      {"ok": False, "reason": "drift", "at": N, "error": "..."}   reconstruction failed
      {"ok": False, "reason": "rewind"}                log contains an ov_rewind

    ``bake_in_subprocess`` adds {"ok": False, "reason": "crashed"} when the child
    dies without returning at all.
    """
    import os

    # A spawned child is a fresh interpreter: effects are registered as an import
    # side effect and must be re-registered here or every card resolves to nothing.
    from goa2.server.app import register_all_effects

    register_all_effects()

    from goa2.server import shares
    from goa2.server.replay import (
        _apply_decision,
        build_session_from_setup,
        load_replay,
        state_body,
        winner_of,
    )

    os.environ["GOA2_SHARE_DIR"] = share_dir

    setup, decisions = load_replay(replay_path)
    session = build_session_from_setup(setup)
    applied = 0

    if any(d.get("type") == "ov_rewind" for d in decisions):
        return {"ok": False, "reason": "rewind"}

    def render(index: int) -> dict[str, Any]:
        nonlocal applied
        # bake_share renders 0..len(decisions) in order, so a single forward walk
        # suffices; no position is ever rebuilt from the seed.
        while applied < index:
            _apply_decision(session, decisions[applied])
            applied += 1
        return state_body(session, cursor_index=index, total=len(decisions))

    try:
        token = shares.bake_share(
            game_id=game_id,
            setup=setup,
            decisions=decisions,
            render=render,
            # Only known once every decision is applied, so it gates publication
            # rather than gating the walk.
            validate=lambda: winner_of(session.state) is not None,
        )
    except Exception as e:
        # Any failure to reconstruct is the same answer to the caller: this log
        # cannot be baked, and here is how far it got. Catching broadly matters
        # because a malformed record raises KeyError rather than the ValueError
        # engine drift produces, and an uncaught one would surface as a 500.
        return {
            "ok": False,
            "reason": "drift",
            "at": applied,
            "error": f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e),
        }

    if token is None:
        return {"ok": False, "reason": "unfinished"}
    return {"ok": True, "token": token}


def bake_in_subprocess(replay_path: str, game_id: str, share_dir: str) -> BakeResult:
    """Run ``bake_replay_share`` in a one-shot child process and return its result.

    A fresh pool per mint costs a spawn (~1-2 s of interpreter startup) but keeps
    no idle worker resident between the rare, deliberate share operations, and
    leaves no executor lifecycle to manage across server restarts.
    """
    context = multiprocessing.get_context("spawn")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            return pool.submit(bake_replay_share, replay_path, game_id, share_dir).result()
    except concurrent.futures.process.BrokenProcessPool:
        # The child died outright — OOM killer, segfault, a hard interpreter
        # crash. Nothing was returned, so report it as a bake failure rather
        # than letting it surface as an unhandled 500.
        return {"ok": False, "reason": "crashed"}
