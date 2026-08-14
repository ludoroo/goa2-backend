"""In-memory game registry mapping game_id -> ManagedGame."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from goa2.engine.session import GameSession, SessionResult
from goa2.server.errors import GameNotFoundError
from goa2.server.game_logger import GameLogger, create_game_logger
from goa2.server.replay import ReplayRecorder, create_replay_recorder

logger = logging.getLogger(__name__)


@dataclass
class ManagedGame:
    game_id: str
    session: GameSession
    player_tokens: dict[str, str]  # token -> hero_id
    spectator_token: str
    hero_to_token: dict[str, str]  # hero_id -> token (reverse)
    created_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    outbound_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_result: SessionResult | None = None
    ws_connections: dict[str, WebSocket] = field(default_factory=dict)
    spectator_ws_connections: dict[int, WebSocket] = field(default_factory=dict)
    game_logger: GameLogger | None = None
    replay_recorder: ReplayRecorder | None = None
    timer_task: asyncio.Task[None] | None = None
    # Consensus-override negotiation state. Coordination, not game state:
    # never saved, never in views; unresolved proposals die on restart.
    # Typed loosely to avoid a registry <-> overrides import cycle
    # (OverrideProposal lives in server/overrides.py).
    pending_override: Any | None = None
    override_expiry_task: asyncio.Task[None] | None = None

    @property
    def current_responder(self) -> str | None:
        """The single source of truth for *who may act on the current input*.

        This is the ``player_id`` of the pending ``InputRequest`` — which
        already accounts for control remaps (under Hanu's ultimate the
        controlled action's inputs are addressed to the controller). Falls back
        to the current actor when there is no pending input. The submit-input
        and rollback paths (REST and WebSocket) authorize against this so their
        derivations cannot drift apart.
        """
        ir = self.last_result.input_request if self.last_result else None
        if ir is not None:
            return ir.player_id
        actor = self.session.state.current_actor_id
        return str(actor) if actor is not None else None


class GameRegistry:
    """Thread-safe in-memory store for active games with optional file persistence."""

    def __init__(self, save_dir: str | None = None) -> None:
        self._games: dict[str, ManagedGame] = {}
        self._save_dir = save_dir

    def create_game(
        self, session: GameSession, hero_ids: list[str], game_id: str | None = None
    ) -> ManagedGame:
        """Register a new game and generate tokens for each hero + spectator."""
        game_id = game_id or uuid.uuid4().hex[:12]
        player_tokens: dict[str, str] = {}
        hero_to_token: dict[str, str] = {}
        for hero_id in hero_ids:
            token = uuid.uuid4().hex
            player_tokens[token] = hero_id
            hero_to_token[hero_id] = token

        spectator_token = uuid.uuid4().hex

        game = ManagedGame(
            game_id=game_id,
            session=session,
            player_tokens=player_tokens,
            spectator_token=spectator_token,
            hero_to_token=hero_to_token,
            game_logger=create_game_logger(game_id),
            replay_recorder=create_replay_recorder(game_id),
        )
        self._games[game_id] = game

        if self._save_dir:
            self.save_game(game_id)

        return game

    def get(self, game_id: str) -> ManagedGame:
        """Get a game by ID or raise GameNotFoundError."""
        game = self._games.get(game_id)
        if game is None:
            raise GameNotFoundError(game_id)
        return game

    def resolve_token(self, token: str) -> tuple[str, str, bool] | None:
        """Resolve a bearer token to (game_id, hero_id, is_spectator).

        Returns None if the token is unknown.
        """
        for game_id, game in self._games.items():
            if token in game.player_tokens:
                return (game_id, game.player_tokens[token], False)
            if token == game.spectator_token:
                return (game_id, "", True)
        return None

    def remove(self, game_id: str) -> None:
        game = self._games.pop(game_id, None)
        if game is not None and game.timer_task is not None:
            game.timer_task.cancel()
        if game is not None and game.override_expiry_task is not None:
            game.override_expiry_task.cancel()
        if self._save_dir:
            from goa2.engine.persistence import delete_game_save

            delete_game_save(game_id, self._save_dir)

    def save_game(self, game_id: str) -> None:
        """Persist a game to disk. No-op if save_dir is not configured."""
        if not self._save_dir:
            return
        game = self._games.get(game_id)
        if game is None:
            return
        from goa2.engine.persistence import save_game

        try:
            save_game(
                game_id=game.game_id,
                state=game.session.state,
                player_tokens=game.player_tokens,
                spectator_token=game.spectator_token,
                hero_to_token=game.hero_to_token,
                created_at=game.created_at,
                save_dir=self._save_dir,
                rollback_snapshot=game.session._rollback_snapshot,
                rollback_actor_id=game.session._rollback_actor_id,
            )
        except Exception:
            logger.exception("Failed to save game %s", game_id)

    def restore_all(self) -> int:
        """Load all saved games from disk into the registry.

        Returns the number of games restored.
        """
        if not self._save_dir:
            return 0
        from goa2.engine.persistence import load_all_games

        games_data = load_all_games(self._save_dir)
        count = 0
        for data in games_data:
            game = ManagedGame(
                game_id=data["game_id"],
                session=data["session"],
                player_tokens=data["player_tokens"],
                spectator_token=data["spectator_token"],
                hero_to_token=data["hero_to_token"],
                created_at=data["created_at"],
                last_result=data["last_result"],
                game_logger=create_game_logger(data["game_id"]),
                replay_recorder=create_replay_recorder(data["game_id"]),
            )
            self._games[game.game_id] = game
            count += 1
            logger.info("Restored game %s", game.game_id)

        self._cleanup_orphaned_logs()
        return count

    def cleanup_stale_games(self, max_age_seconds: int = 86400) -> int:
        """Remove games whose save file hasn't been updated in max_age_seconds.

        Returns the number of games removed.
        """
        if not self._save_dir:
            return 0

        save_path = Path(self._save_dir)
        now = time.time()
        removed = 0

        for game_id in list(self._games):
            file_path = save_path / f"{game_id}.json"
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                if now - mtime > max_age_seconds:
                    logger.info(
                        "Removing stale game %s (last updated %.0f hours ago)",
                        game_id,
                        (now - mtime) / 3600,
                    )
                    self.remove(game_id)
                    removed += 1
            else:
                # No save file — remove from memory
                self.remove(game_id)
                removed += 1

        return removed

    def _cleanup_orphaned_logs(self) -> None:
        """Remove log files for games that no longer have a save file."""
        from goa2.server.game_logger import delete_game_logs

        log_dir = os.environ.get("GOA2_LOG_DIR", "logs/games")
        log_path = Path(log_dir)
        if not log_path.is_dir():
            return
        seen_ids: set[str] = set()
        for f in log_path.iterdir():
            if not f.is_file():
                continue
            # Extract game_id: either "{game_id}.log" or legacy "{game_id}_{ts}.log"
            game_id = f.stem.split("_")[0]
            if game_id not in self._games and game_id not in seen_ids:
                seen_ids.add(game_id)
                delete_game_logs(game_id, log_dir)
                logger.info("Cleaned up orphaned logs for game %s", game_id)

    def __len__(self) -> int:
        return len(self._games)

    def all_games(self) -> list[ManagedGame]:
        """Return a stable snapshot used to resume/cancel deadline tasks."""
        return list(self._games.values())
