"""Shareable, pre-baked replay artifacts.

A share is a capability token that grants read-only access to exactly one
*finished* game's replay, without an admin token. It exists so a tester can be
shown a game for bug triage without being handed `GOA2_ADMIN_TOKEN`, which also
grants bug-report mutation.

Storage mirrors ``bug_reports.py``: one directory per share under
``GOA2_SHARE_DIR`` (default ``data/shares``)::

    data/shares/<token>/meta.json      setup, decision list, total, engine, game_id
    data/shares/<token>/000.json.gz    baked view for decision 0
    data/shares/<token>/249.json.gz    ...

Why bake instead of reconstructing per request: a finished game's log never
changes, so its positions are immutable. Reconstruction is *re-simulation* —
``ReplayCursor.seek`` has no un-apply, so every backward seek rebuilds from the
seed (measured on the deployment target: 0.82 s to build the empty session plus
18.4 ms per decision, i.e. ~5.4 s for a 249-decision game). That work is pure
Python holding the GIL, so it competes with the event loop serving live games.
Baking once at mint time (~6.6 s, ~3 MB gzipped) turns every subsequent read
into a 12 KB file read with no engine work at all.

A baked share is fully self-contained: it does not read the ``.jsonl`` log, so
it cannot fail on engine drift the way live reconstruction can.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SHARE_DIR = "data/shares"

# Tokens are secrets.token_urlsafe output: URL-safe base64 alphabet.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _share_dir() -> str:
    return os.environ.get("GOA2_SHARE_DIR", DEFAULT_SHARE_DIR)


def _share_path(token: str) -> Path:
    """Resolve a token to its share directory, rejecting path traversal.

    Raises FileNotFoundError for anything that is not a plausible token, so a
    malformed token is indistinguishable from a revoked one to the caller.
    """
    if not token or not _TOKEN_RE.match(token):
        raise FileNotFoundError(f"Share not found: {token!r}")
    return Path(_share_dir()) / token


def _position_name(index: int) -> str:
    return f"{index:03d}.json.gz"


def bake_share(
    *,
    game_id: str,
    setup: dict[str, Any],
    decisions: list[dict[str, Any]],
    render: Any,
    validate: Any = None,
) -> str | None:
    """Bake every position of a finished game and return the new share token.

    ``render`` is called as ``render(index) -> dict`` for index 0..len(decisions)
    and must return the same body ``GET /replays/{id}/state`` produces at that
    index. Keeping reconstruction in the caller leaves this module free of engine
    imports and makes the bake trivially testable with a stub.

    ``validate`` (optional) is called with no arguments after the last render and
    before the artifact is published. Returning False discards it and yields
    None — that is how "only finished games" is enforced without a second
    reconstruction pass, since whether the game finished is only known once every
    decision has been applied.

    The artifact is built in a temp directory and moved into place atomically, so
    a crash or full disk never leaves a half-written share readable.
    """
    token = secrets.token_urlsafe(32)
    root = Path(_share_dir())
    root.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=f".{token}.", dir=root))
    try:
        size_bytes = 0
        for index in range(len(decisions) + 1):
            body = render(index)
            payload = json.dumps(body, separators=(",", ":")).encode()
            blob = gzip.compress(payload, 6)
            size_bytes += len(blob)
            (staging / _position_name(index)).write_bytes(blob)

        if validate is not None and not validate():
            shutil.rmtree(staging, ignore_errors=True)
            return None

        meta = {
            "token": token,
            "game_id": game_id,
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
            "total_decisions": len(decisions),
            "engine": setup.get("engine"),
            "created_at": time.time(),
            # Recorded at bake time so listing shares never stats hundreds of files.
            "size_bytes": size_bytes,
        }
        (staging / "meta.json").write_text(json.dumps(meta, indent=2))
        os.replace(staging, root / token)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return token


def load_meta(token: str) -> dict[str, Any] | None:
    """The share's meta.json, or None if the token is unknown or revoked."""
    try:
        path = _share_path(token) / "meta.json"
    except FileNotFoundError:
        return None
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read share meta for %s", token)
        return None


def position_path(token: str, index: int) -> Path | None:
    """Path to a baked position's gzip file, or None if absent."""
    try:
        path = _share_path(token) / _position_name(index)
    except FileNotFoundError:
        return None
    return path if path.is_file() else None


def revoke_share(token: str) -> bool:
    """Delete a share directory. Returns False if it did not exist."""
    try:
        path = _share_path(token)
    except FileNotFoundError:
        return False
    if not path.is_dir():
        return False
    shutil.rmtree(path)
    return True


def list_shares() -> list[dict[str, Any]]:
    """All shares, newest first. Unreadable directories are skipped."""
    directory = Path(_share_dir())
    if not directory.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in directory.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        meta = load_meta(child.name)
        if meta is not None:
            out.append(meta)
    out.sort(key=lambda m: m.get("created_at") or 0, reverse=True)
    return out


def share_for_game(game_id: str) -> dict[str, Any] | None:
    """The newest live share for a game, if any."""
    return next((m for m in list_shares() if m.get("game_id") == game_id), None)


def shared_game_ids() -> set[str]:
    """Game ids with at least one live share (their replays are pinned).

    The baked artifact does not need the log, so this pin is belt-and-braces —
    it keeps the original available for re-baking and debugging.
    """
    return {m["game_id"] for m in list_shares() if m.get("game_id")}
