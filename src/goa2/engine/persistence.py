"""
Game state persistence: save/load GameState to/from JSON files.

Uses atomic writes (tmp + rename) to prevent corruption.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

# Ensure step_types patching is applied before any serialization
import goa2.engine.step_types as _step_types  # noqa: F401
from goa2.domain.models import GamePhase
from goa2.domain.state import GameState
from goa2.engine.handler import process_stack
from goa2.engine.session import GameSession

logger = logging.getLogger(__name__)

# Bump when the save schema changes in a non-backward-compatible way. New
# optional fields are read with .get(...) fallbacks and do NOT require a bump.
SAVE_VERSION = 1


def save_game(
    game_id: str,
    state: GameState,
    player_tokens: dict[str, str],
    spectator_token: str,
    hero_to_token: dict[str, str],
    created_at: float,
    save_dir: str,
    rollback_snapshot: dict[str, Any] | None = None,
    rollback_actor_id: str | None = None,
    bot_specs: dict[str, Any] | None = None,
) -> Path:
    """Serialize game data to a JSON file with atomic write.

    ``bot_specs`` accepts either a mapping of ``hero_id -> BotSpec`` (Pydantic
    model) or already-serialized JSON dicts. Either way, we normalize to plain
    dicts so this module doesn't take a runtime dependency on the ``server``
    package. Passing ``None`` (the default) writes an empty mapping and remains
    load-compatible with older saves that lacked the key entirely.
    """
    serialized_specs: dict[str, Any] = {}
    if bot_specs:
        for hero_id, spec in bot_specs.items():
            if hasattr(spec, "model_dump"):
                serialized_specs[hero_id] = spec.model_dump(mode="json")
            else:
                serialized_specs[hero_id] = spec

    payload: dict[str, Any] = {
        "version": SAVE_VERSION,
        "game_id": game_id,
        "player_tokens": player_tokens,
        "spectator_token": spectator_token,
        "hero_to_token": hero_to_token,
        "created_at": created_at,
        "state": state.model_dump(mode="json"),
        "rollback_snapshot": rollback_snapshot,
        "rollback_actor_id": rollback_actor_id,
        "bot_specs": serialized_specs,
    }

    os.makedirs(save_dir, exist_ok=True)
    target = Path(save_dir) / f"{game_id}.json"

    # Atomic write: write to temp file then rename
    fd, tmp_path = tempfile.mkstemp(dir=save_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, target)
    except BaseException:
        # Clean up temp file on failure
        with suppress(OSError):
            os.unlink(tmp_path)
        raise

    logger.info("Saved game %s to %s", game_id, target)
    return target


def load_game(file_path: str) -> dict[str, Any]:
    """Load game data from a JSON file.

    Returns a dict with keys: game_id, state (GameState), player_tokens,
    spectator_token, hero_to_token, created_at, last_result.

    The last_result is re-derived by calling process_stack() if the
    execution stack is non-empty.
    """
    with open(file_path) as f:
        payload = json.load(f)

    state = GameState.model_validate(payload["state"])
    session = GameSession(state)

    # Restore the in-memory rollback snapshot (not part of GameState) so a
    # mid-action rollback survives a restart.
    session._rollback_snapshot = payload.get("rollback_snapshot")
    session._rollback_actor_id = payload.get("rollback_actor_id")

    # Re-derive last_result by processing the stack. In the server, saves
    # happen after mutations, so the stack either has a step waiting for input
    # or is empty. process_stack will re-emit any pending input request.
    last_result = None
    if state.execution_stack:
        stack_result = process_stack(state)
        if stack_result.input_request:
            # Delegate can_rollback re-assertion to the session so this path
            # can't drift from GameSession._manage_rollback.
            session.reapply_rollback_flag(stack_result.input_request)
        last_result = session._build_result(stack_result.input_request, events=stack_result.events)
    elif state.phase == GamePhase.GAME_OVER:
        # Completed games have an empty execution stack, but clients still
        # obtain the winner from the transient SessionResult. Reconstruct it
        # from the authoritative persisted state after a restart.
        last_result = session._build_result()

    return {
        "game_id": payload["game_id"],
        "session": session,
        "player_tokens": payload["player_tokens"],
        "spectator_token": payload["spectator_token"],
        "hero_to_token": payload["hero_to_token"],
        "created_at": payload["created_at"],
        "last_result": last_result,
        # Raw dicts (not BotSpec models) — the registry layer knows how to
        # rebuild them into Pydantic instances. Absent in legacy saves; return
        # an empty mapping so callers can rely on the key.
        "bot_specs": payload.get("bot_specs", {}) or {},
    }


def load_all_games(save_dir: str) -> list[dict[str, Any]]:
    """Load all saved games from a directory, skipping failures."""
    results: list[dict[str, Any]] = []
    save_path = Path(save_dir)
    if not save_path.is_dir():
        return results

    for file_path in sorted(save_path.glob("*.json")):
        try:
            data = load_game(str(file_path))
            results.append(data)
            logger.info("Loaded game %s from %s", data["game_id"], file_path)
        except Exception:
            logger.exception("Failed to load game from %s", file_path)

    return results


def delete_game_save(game_id: str, save_dir: str) -> None:
    """Remove a game's save file if it exists."""
    target = Path(save_dir) / f"{game_id}.json"
    try:
        target.unlink(missing_ok=True)
        logger.info("Deleted save for game %s", game_id)
    except OSError:
        logger.exception("Failed to delete save for game %s", game_id)
