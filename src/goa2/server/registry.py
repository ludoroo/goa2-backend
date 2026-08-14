"""In-memory game registry mapping game_id -> ManagedGame."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket

from goa2.engine.session import GameSession, SessionResult
from goa2.server.bot_models import BotSpec
from goa2.server.errors import GameNotFoundError
from goa2.server.game_logger import GameLogger, create_game_logger
from goa2.server.replay import ReplayRecorder, create_replay_recorder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from automata.agents.base import Agent

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
    # Persisted bot metadata: hero_id -> BotSpec. Loaded/saved atomically with
    # the rest of the game payload (see engine/persistence.py). This is the
    # only bot state that round-trips through disk — the live agent instance
    # and asyncio task below stay runtime-only.
    bot_specs: dict[str, BotSpec] = field(default_factory=dict)
    # Handle of the currently-running bot decision worker for this game, if
    # any. Never serialized; the coordinator (see server/bots.py) is
    # responsible for creating/cancelling it.
    bot_task: asyncio.Task[None] | None = None
    # Runtime tombstone: set to ``True`` by :meth:`GameRegistry.remove` before
    # any task cancellation. Every asynchronous seam (bot worker, deadline
    # worker, scheduler) is required to check this flag under ``game.lock``
    # (or immediately before touching the registry) and bail without saving,
    # scheduling, or broadcasting when it is set. This closes the window
    # where a background task holds a stale ``ManagedGame`` reference after
    # the registry pops it: the in-memory game is unreachable through the
    # registry, but the task keeps a live handle from before the pop. Never
    # persisted.
    removed: bool = field(default=False, repr=False)
    # Runtime-only cache of instantiated Agent objects keyed by hero_id.
    # Built lazily by the coordinator on first use so restored games (which
    # only carry the serialized ``bot_specs``) don't pay the construction
    # cost until they actually need to drive a decision. Never persisted —
    # cleared on removal/restore so a fresh worker rebuilds seeded state
    # from a stable game-specific entropy source rather than reusing an old
    # RNG position.
    _bot_agents: dict[str, Agent] | None = field(default=None, repr=False)
    # Cached per-hero :class:`HeuristicAgent` fallbacks used whenever a
    # bounded ISMCTS search hits a queue timeout, search timeout, or
    # exception. Keyed by ``hero_id`` so each replaced ISMCTS
    # agent gets its **own** RNG stream — sharing a single fallback
    # instance across heroes would couple their subsequent Random /
    # Heuristic choices, breaking the "one hero's fallback does not
    # influence another's" invariant a mixed-team game relies on.
    # Built lazily on first fallback so games that never trigger it
    # never pay for it. Never persisted; cleared on registry.remove()
    # so a restore rebuilds seeded state against fresh entropy.
    _bot_fallback_agents: dict[str, Agent] | None = field(default=None, repr=False)
    # In-flight bounded-search futures scoped to this game.
    # Populated by the bot coordinator's dispatch and
    # cleared by the done-callback. Registry teardown and app
    # shutdown observe this set to make sure a slow search cannot
    # outlive the game object.
    _bot_search_futures: set[asyncio.Future[Any]] = field(default_factory=set, repr=False)

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
        self,
        session: GameSession,
        hero_ids: list[str],
        game_id: str | None = None,
        bot_specs: dict[str, BotSpec] | None = None,
    ) -> ManagedGame:
        """Register a new game and generate tokens for each hero + spectator.

        ``bot_specs`` optionally maps hero IDs to their persisted bot
        configuration. Every bot hero must be present in the roster
        (``hero_ids``); unknown IDs raise ``ValueError`` so we never persist a
        misaligned spec.
        """
        game_id = game_id or uuid.uuid4().hex[:12]
        player_tokens: dict[str, str] = {}
        hero_to_token: dict[str, str] = {}
        for hero_id in hero_ids:
            token = uuid.uuid4().hex
            player_tokens[token] = hero_id
            hero_to_token[hero_id] = token

        spectator_token = uuid.uuid4().hex

        validated_specs: dict[str, BotSpec] = {}
        if bot_specs:
            roster = set(hero_ids)
            for hero_id, spec in bot_specs.items():
                if hero_id not in roster:
                    raise ValueError(
                        f"bot_specs references hero {hero_id!r} which is not "
                        f"in the game roster {sorted(roster)}"
                    )
                validated_specs[hero_id] = spec

        game = ManagedGame(
            game_id=game_id,
            session=session,
            player_tokens=player_tokens,
            spectator_token=spectator_token,
            hero_to_token=hero_to_token,
            game_logger=create_game_logger(game_id),
            replay_recorder=create_replay_recorder(game_id),
            bot_specs=validated_specs,
        )
        self._games[game_id] = game

        if self._save_dir:
            self.save_game(game_id)

        return game

    @staticmethod
    def _rebuild_bot_specs(
        game_id: str,
        raw_specs: object,
        roster: set[str],
    ) -> dict[str, BotSpec]:
        """Reconstruct persisted bot metadata into validated :class:`BotSpec`s.

        Called at restore time from disk. Everything here is defensive: the
        on-disk payload could be corrupt, hand-edited, or written by an older
        schema, and we must never crash startup or drop otherwise-restorable
        games. Invalid metadata (non-mapping top-level, entries that fail
        Pydantic validation, or hero IDs not in the restored roster) is logged
        and discarded; well-formed entries are kept.
        """
        if not isinstance(raw_specs, dict):
            if raw_specs:  # None / empty dict is silently normal; other truthy values are not
                logger.error(
                    "Discarding bot_specs for game %s: expected mapping, got %s",
                    game_id,
                    type(raw_specs).__name__,
                )
            return {}

        restored: dict[str, BotSpec] = {}
        for hero_id, raw in raw_specs.items():
            if not isinstance(hero_id, str):
                logger.error(
                    "Discarding bot spec for game %s: hero key %r is not a string",
                    game_id,
                    hero_id,
                )
                continue
            if hero_id not in roster:
                logger.error(
                    "Discarding bot spec for game %s hero %s: not in restored roster",
                    game_id,
                    hero_id,
                )
                continue
            try:
                restored[hero_id] = BotSpec.model_validate(raw)
            except Exception:
                logger.exception(
                    "Failed to restore bot spec for game %s hero %s; skipping spec",
                    game_id,
                    hero_id,
                )
        return restored

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
        """Evict a game and cancel every asynchronous task attached to it.

        The order is deliberate:

        1. Pop the game from the registry map so no new caller can reach it
           via :meth:`get` / :meth:`resolve_token`.
        2. Set ``game.removed = True`` **before** cancelling anything so an
           in-flight task holding a stale reference sees the tombstone the
           moment it re-enters a checkpoint under ``game.lock``.
        3. Cancel the deadline and bot tasks. Cancellation is asynchronous;
           the tombstone is the invariant that keeps the tasks from writing
           anything after they wake.
        4. Drop the runtime agent cache.

        Kept synchronous to preserve the existing caller contract (cleanup
        loops, cheat cancellations, WebSocket teardown all call this from
        sync paths). The cancellation is fire-and-forget — background tasks
        finish on their own event-loop iteration and the tombstone prevents
        harmful side effects in the meantime.
        """
        game = self._games.pop(game_id, None)
        if game is not None:
            # Order matters: mark the tombstone before cancellation so a task
            # that is about to enter a locked section observes it. A task
            # that is still off-loop in ``asyncio.to_thread`` will see the
            # tombstone when it next reacquires the game lock.
            game.removed = True
            if game.timer_task is not None:
                game.timer_task.cancel()
            if game.override_expiry_task is not None:
                game.override_expiry_task.cancel()
            if game.bot_task is not None:
                game.bot_task.cancel()
            # Drop the runtime agent cache so any lingering references
            # cannot survive game removal — a stale reference is never
            # correct once the ManagedGame is gone.
            game._bot_agents = None
            game._bot_fallback_agents = None
            # The in-flight future set is left in place so
            # the done-callback can still discard entries as searches
            # finish on background threads. We deliberately do NOT cancel
            # the futures here: they run on cloned state and their
            # results are dropped by the coordinator's tombstone check;
            # cancelling them would burn a semaphore slot without
            # releasing it (release happens in the done-callback).
        if self._save_dir:
            from goa2.engine.persistence import delete_game_save

            delete_game_save(game_id, self._save_dir)

    def save_game(self, game_id: str) -> None:
        """Persist a game to disk. No-op if save_dir is not configured.

        Also a no-op if the game has been marked ``removed`` — the caller is
        a background task that raced :meth:`remove` and must not resurrect
        the save file after it was deliberately deleted.
        """
        if not self._save_dir:
            return
        game = self._games.get(game_id)
        if game is None:
            return
        if game.removed:
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
                bot_specs=game.bot_specs,
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
            # The restored roster is the authoritative set of hero IDs from
            # the persisted token map — bot specs referencing anything else
            # are stale and must be discarded (they'd never authorize a real
            # request anyway). Defensive: the payload may be missing/invalid.
            hero_to_token_raw = data.get("hero_to_token") or {}
            roster: set[str] = (
                set(hero_to_token_raw.keys()) if isinstance(hero_to_token_raw, dict) else set()
            )
            restored_specs = self._rebuild_bot_specs(data["game_id"], data.get("bot_specs"), roster)

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
                bot_specs=restored_specs,
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
