"""Minimal, durable game replay logs.

A replay log captures the *least* information needed to reconstruct a game
exactly: the setup parameters (including the RNG seed) plus the ordered list of
player decisions. Because gameplay is fully deterministic given the seed, and
the only state-changing client operations are card commits, passes, and input
responses, replaying those decisions in order reproduces the game byte-for-byte.

Format: one JSON object per line (JSONL).

  line 1  setup header:
    {"v":1,"type":"setup","game_id":"...","map":"forgotten_island",
     "red":["Arien"],"blue":["Wasp"],"game_type":"QUICK","cheats":false,
     "seed":1234,"engine":"<git sha>","created_at":1718900000.0}

  line N  one decision (in applied order), tagged with round/turn and a
  wall-clock receipt timestamp ``ts`` (epoch seconds, UTC):
    {"type":"commit","r":1,"t":1,"hero":"hero_arien","card":"arien_basic_1","ts":1718900001.2}
    {"type":"uncommit","r":1,"t":1,"hero":"hero_arien","ts":1718900001.3}
    {"type":"pass","r":1,"t":1,"hero":"hero_wasp","ts":1718900001.4}
    {"type":"input","r":3,"t":2,"hero":"hero_arien","sel":"minion_4","ts":1718900010.0}
    {"type":"rollback","r":3,"t":2,"hero":"hero_arien","ts":1718900012.5}
    {"type":"cheat_gold","r":1,"t":1,"hero":"hero_arien","amount":5,"ts":1718900000.9}

  consensus overrides (see engine/overrides.py; ``voters`` is audit-only):
    {"type":"ov_patch","r":3,"t":2,"hero":"hero_arien","op":"move_entity","args":{"entity":"minion_4","hex":{"q":1,"r":-2,"s":1}},"voters":["hero_arien","hero_wasp"]}
    {"type":"ov_unstick","r":3,"t":2,"hero":"hero_arien","op":"abort_action","args":{},"voters":["hero_arien"]}
    {"type":"ov_rewind","r":3,"t":2,"hero":"hero_arien","to":47,"voters":["hero_arien"]}

``ov_rewind`` never truncates the log — it is a cursor move meaning "the game
continues from the state after the first N decisions". Reconstruction resolves
it by rebuilding from the seed; the superseded segment stays in the file as
evidence of the bug that caused the rewind.

Every state-changing client operation is recorded: commits, passes, input
responses, rollbacks (restore the current actor's turn-start snapshot), and the
gold cheat. Bare ``advance`` is not recorded — it only drives deterministic
engine processing between decisions. Records are written *after* the engine
accepts the operation, so a rejected one never leaves a phantom in the log.

The ``ts`` field is wall-clock receipt time captured when the operation was
accepted and logged. It is not part of the deterministic reconstruction (the
replayer ignores it), but it lets analytics measure how long players took to
make each decision.

Replays are durable: they live in their own directory with their own
retention (default 30 days) and are NOT deleted when a game's save is removed.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from goa2.domain.models import GamePhase
from goa2.domain.time_control import TimeControlConfig
from goa2.domain.types import HeroID
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup
from goa2.server.map_paths import resolve_map_path

logger = logging.getLogger(__name__)

REPLAY_VERSION = 1
DEFAULT_REPLAY_DIR = "data/replays"
DEFAULT_REPLAY_TTL_DAYS = 30


def _replay_dir() -> str:
    return os.environ.get("GOA2_REPLAY_DIR", DEFAULT_REPLAY_DIR)


def _replay_ttl_days() -> int:
    try:
        return int(os.environ.get("GOA2_REPLAY_TTL_DAYS", DEFAULT_REPLAY_TTL_DAYS))
    except ValueError:
        return DEFAULT_REPLAY_TTL_DAYS


def _resolve_map_path(map_name: str) -> str:
    """Resolve a map name to its JSON file path (mirrors routes_games._map_path)."""
    return resolve_map_path(map_name)


def _engine_revision() -> str:
    """Best-effort git sha so a replay/engine mismatch is detectable. Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


class ReplayRecorder:
    """Append-only writer for a single game's replay log.

    Safe to re-create against an existing file (e.g. after a server restart):
    the setup header is written only when the file is new/empty; decisions are
    appended thereafter.
    """

    def __init__(self, game_id: str, replay_dir: str | None = None) -> None:
        self.game_id = game_id
        directory = Path(replay_dir or _replay_dir())
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{game_id}.jsonl"

    @property
    def has_setup(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def _append(self, record: dict[str, Any]) -> None:
        # Wall-clock receipt time for data/analytics; not part of the
        # deterministic reconstruction (the replayer ignores it).
        record["ts"] = time.time()
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            logger.exception("Failed to append replay record for game %s", self.game_id)

    def record_setup(
        self,
        *,
        map_name: str,
        red_heroes: list[str],
        blue_heroes: list[str],
        game_type: str,
        cheats: bool,
        seed: int,
        time_control: TimeControlConfig | None = None,
    ) -> None:
        """Write the setup header. No-op if the log already has a header."""
        if self.has_setup:
            return
        self._append(
            {
                "v": REPLAY_VERSION,
                "type": "setup",
                "game_id": self.game_id,
                "map": map_name,
                "red": red_heroes,
                "blue": blue_heroes,
                "game_type": game_type,
                "cheats": cheats,
                "seed": seed,
                "time_control": (
                    time_control.model_dump(mode="json") if time_control is not None else None
                ),
                "engine": _engine_revision(),
                "created_at": time.time(),
            }
        )

    def record_commit(self, hero_id: str, card_id: str, round_num: int, turn: int) -> None:
        self._append(
            {"type": "commit", "r": round_num, "t": turn, "hero": hero_id, "card": card_id}
        )

    def record_pass(self, hero_id: str, round_num: int, turn: int) -> None:
        self._append({"type": "pass", "r": round_num, "t": turn, "hero": hero_id})

    def record_finish_planning(self, hero_id: str, round_num: int, turn: int) -> None:
        self._append({"type": "finish_planning", "r": round_num, "t": turn, "hero": hero_id})

    def record_uncommit(self, hero_id: str, round_num: int, turn: int) -> None:
        self._append({"type": "uncommit", "r": round_num, "t": turn, "hero": hero_id})

    def record_input(self, hero_id: str, selection: Any, round_num: int, turn: int) -> None:
        self._append(
            {"type": "input", "r": round_num, "t": turn, "hero": hero_id, "sel": selection}
        )

    def record_rollback(self, hero_id: str, round_num: int, turn: int) -> None:
        self._append({"type": "rollback", "r": round_num, "t": turn, "hero": hero_id})

    def record_cheat_gold(self, hero_id: str, amount: int, round_num: int, turn: int) -> None:
        self._append(
            {"type": "cheat_gold", "r": round_num, "t": turn, "hero": hero_id, "amount": amount}
        )

    def record_timer_timeout(
        self,
        *,
        action: str,
        hero_id: str,
        round_num: int,
        turn: int,
        card_id: str | None = None,
        selection: Any = None,
        request_id: str | None = None,
        team_id: str | None = None,
        eligible_hero_ids: list[str] | None = None,
    ) -> None:
        """Record the exact automatic decision; replay never runs a clock."""
        record: dict[str, Any] = {
            "type": "timer_timeout",
            "action": action,
            "r": round_num,
            "t": turn,
            "hero": hero_id,
        }
        if card_id is not None:
            record["card"] = card_id
        if action == "input":
            record["sel"] = selection
        if request_id is not None:
            record["request_id"] = request_id
        if team_id is not None:
            record["team"] = team_id
            record["eligible_heroes"] = list(eligible_hero_ids or [])
        self._append(record)

    def record_override(self, record: dict[str, Any]) -> None:
        """Append a consensus-override decision (ov_patch / ov_unstick / ov_rewind).

        The caller builds the full record (type, r, t, hero, op/args or to,
        voters). ``voters`` is auditability only; reconstruction ignores it.
        """
        if record.get("type") not in {"ov_patch", "ov_unstick", "ov_rewind"}:
            raise ValueError(f"Not an override record: {record.get('type')!r}")
        self._append(record)


def create_replay_recorder(game_id: str, replay_dir: str | None = None) -> ReplayRecorder:
    """Create a ReplayRecorder using GOA2_REPLAY_DIR (or the default) when unset."""
    return ReplayRecorder(game_id, replay_dir=replay_dir)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def load_replay(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a replay file into (setup_header, decisions). Raises FileNotFoundError."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Replay file not found: {path}")

    setup: dict[str, Any] | None = None
    decisions: list[dict[str, Any]] = []
    with open(p) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if record.get("type") == "setup":
                setup = record
            else:
                decisions.append(record)

    if setup is None:
        raise ValueError(f"Replay file has no setup header: {path}")
    return setup, decisions


def build_session_from_setup(setup: dict[str, Any]) -> GameSession:
    """Create a fresh GameSession from a replay setup header (seeded, no decisions)."""
    configured = setup.get("time_control")
    time_control = TimeControlConfig.model_validate(configured) if configured else None
    state = GameSetup.create_game(
        _resolve_map_path(setup["map"]),
        setup["red"],
        setup["blue"],
        setup.get("cheats", False),
        setup.get("game_type", "LONG"),
        seed=setup["seed"],
        time_control=time_control,
    )
    if time_control is not None:
        # Replays record exact decisions, not readiness or wall-clock receipt
        # times. Preserve the configured rules for inspection but do not expose
        # a fabricated WAITING_FOR_PLAYERS clock as if it were live history.
        state.clock = None
    return GameSession(state)


def effective_indices(decisions: list[dict[str, Any]], upto: int | None = None) -> list[int]:
    """Raw indices of decisions still live after resolving ov_rewind records.

    An ov_rewind at raw index i with to=N replaces the live list with the
    effective resolution of the first N raw records (N < i, so this recursion
    terminates). Rewind records themselves are never live — they are cursor
    moves, not decisions.
    """
    end = len(decisions) if upto is None else upto
    live: list[int] = []
    for i in range(end):
        d = decisions[i]
        if d.get("type") == "ov_rewind":
            live = effective_indices(decisions, int(d["to"]))
        else:
            live.append(i)
    return live


def effective_decisions(
    decisions: list[dict[str, Any]], upto: int | None = None
) -> list[dict[str, Any]]:
    """The linear decision list actually in force after resolving rewinds."""
    return [decisions[i] for i in effective_indices(decisions, upto)]


def replay_game(
    path: str,
    *,
    until_round: int | None = None,
    until_turn: int | None = None,
    until_decision: int | None = None,
) -> GameSession:
    """Reconstruct a game from its replay log, returning a live GameSession.

    Stop points (apply none, one, or combine):
      - until_decision=N : apply only the first N decisions.
      - until_round=R [, until_turn=T] : stop before the first decision that
        occurs at/after round R (and turn T), leaving the session positioned at
        the start of that moment — the point a bug was reported.
      - none : replay the entire game.

    The returned GameSession can be inspected (`.state`, `build_view(...)`) or
    advanced further one decision at a time.
    """
    setup, decisions = load_replay(path)
    session = build_session_from_setup(setup)

    for i, decision in enumerate(decisions):
        if until_decision is not None and i >= until_decision:
            break
        # Stop before the first decision at/after the target round/turn, leaving
        # the session positioned at the start of that moment.
        if _decision_at_or_after(decision, until_round, until_turn):
            break
        if decision.get("type") == "ov_rewind":
            # A rewind is a cursor move: rebuild from the seed and re-apply the
            # effective prefix, then continue forward from the next record.
            session = build_session_from_setup(setup)
            for d in effective_decisions(decisions, int(decision["to"])):
                _apply_decision(session, d)
        else:
            _apply_decision(session, decision)

    return session


def rebuild_session_for_rewind(path: str, target_index: int) -> GameSession:
    """Reconstruct a session positioned after ``target_index`` raw decisions.

    Used by the live rewind apply path: the returned session REPLACES
    ManagedGame.session. Prior ov_rewind records inside the prefix are honored.
    """
    setup, decisions = load_replay(path)
    if not 0 <= target_index <= len(decisions):
        raise ValueError(f"Rewind target {target_index} out of range 0..{len(decisions)}")
    session = build_session_from_setup(setup)
    for d in effective_decisions(decisions, target_index):
        _apply_decision(session, d)
    return session


def index_for_round_turn(
    decisions: list[dict[str, Any]], until_round: int, until_turn: int | None = None
) -> int:
    """Decision index that positions the game at the start of round R (and turn T).

    Returns the index of the first decision at/after the target round/turn — i.e.
    the number of decisions to apply to land at that moment — or len(decisions)
    if the target is past the end of the game. Mirrors replay_game()'s stop logic.
    """
    for i, decision in enumerate(decisions):
        if _decision_at_or_after(decision, until_round, until_turn):
            return i
    return len(decisions)


def winner_of(state: Any) -> str | None:
    """Winner label for a reconstructed state, or None while the game is unfinished."""
    if state.individual_winner_id is not None:
        return str(state.individual_winner_id)
    return state.winner.value if state.winner else None


def state_body(session: GameSession, *, cursor_index: int, total: int) -> dict[str, Any]:
    """The body served for a replay position.

    Lives here rather than in the route so the dynamic endpoint and the share
    bake — which runs in a separate process with no FastAPI imported — produce
    byte-identical output for the same index.
    """
    from goa2.domain.views import build_view

    state = session.state
    return {
        "view": build_view(state, reveal_all=True),
        "position": {
            "decision_index": cursor_index,
            "round": state.round,
            "turn": state.turn,
            "total_decisions": total,
        },
        "winner": winner_of(state),
    }


class ReplayCursor:
    """A reconstructed game positioned at a decision index, advanceable forward.

    Holds a live GameSession plus a cursor = number of decisions applied. Seeking
    forward applies only the intervening decisions (so stepping N -> N+1 applies
    exactly one); seeking backward rebuilds from the seed (the engine is
    forward-only). This makes action-by-action playback cheap while keeping any
    position reproducible.
    """

    def __init__(self, setup: dict[str, Any], decisions: list[dict[str, Any]]) -> None:
        self.setup = setup
        self.decisions = decisions
        self.session = build_session_from_setup(setup)
        self.cursor = 0  # number of decisions applied

    @property
    def total(self) -> int:
        return len(self.decisions)

    def seek(self, target_index: int) -> GameSession:
        """Position the session so exactly `target_index` decisions are applied."""
        target = max(0, min(target_index, len(self.decisions)))
        if target < self.cursor:
            # Backward move: no cheap un-apply, so rebuild from the seed.
            self.session = build_session_from_setup(self.setup)
            self.cursor = 0
        while self.cursor < target:
            decision = self.decisions[self.cursor]
            if decision.get("type") == "ov_rewind":
                # Rewind = cursor move: rebuild from the seed and re-apply the
                # effective prefix, then continue forward.
                self.session = build_session_from_setup(self.setup)
                for d in effective_decisions(self.decisions, int(decision["to"])):
                    _apply_decision(self.session, d)
            else:
                _apply_decision(self.session, decision)
            self.cursor += 1
        return self.session


def _decision_at_or_after(
    decision: dict[str, Any], until_round: int | None, until_turn: int | None
) -> bool:
    if until_round is None:
        return False
    r = decision.get("r", 0)
    if r != until_round:
        return r > until_round
    if until_turn is None:
        return True
    return decision.get("t", 0) >= until_turn


def _apply_decision(session: GameSession, decision: dict[str, Any]) -> None:
    kind = decision.get("type")
    hero_id = HeroID(decision["hero"])

    if kind == "timer_timeout":
        action = decision.get("action")
        if action == "input" and session.state.phase == GamePhase.RESOLUTION:
            # Live automatic Resolution/Response choices are final: they clear
            # the rollback snapshot and make ConfirmResolutionStep auto-finish.
            # Reproduce that control-flow effect before applying the selection.
            session.state.execution_context["rollback_frozen"] = True
            session._rollback_snapshot = None
            session._rollback_actor_id = None
        translated = {**decision, "type": action}
        _apply_decision(session, translated)
        return

    if kind == "commit":
        hero = session.state.get_hero(hero_id)
        if hero is None:
            raise ValueError(f"Replay: hero {hero_id} not found for commit")
        card = next((c for c in hero.hand if c.id == decision["card"]), None)
        if card is None:
            raise ValueError(
                f"Replay: card {decision['card']} not in hand of {hero_id} "
                "(engine version mismatch?)"
            )
        session.commit_card(hero_id, card)
    elif kind == "pass":
        session.pass_turn(hero_id)
    elif kind == "finish_planning":
        session.finish_planning(hero_id)
    elif kind == "uncommit":
        session.uncommit_card(hero_id)
    elif kind == "input":
        # Replay decisions are trusted server-side data; request UUIDs are
        # intentionally not logged because they are transport correlation,
        # not deterministic game decisions.
        session.advance({"selection": decision["sel"]})
    elif kind == "rollback":
        # Restores the current actor's turn-start snapshot, exactly as live play
        # did. The reconstruction session took the same snapshot deterministically
        # while applying the preceding inputs, so this reproduces it faithfully.
        session.rollback()
    elif kind == "cheat_gold":
        # Legacy record kept so old replays load; set_gold supersedes it.
        hero = session.state.get_hero(hero_id)
        if hero is None:
            raise ValueError(f"Replay: hero {hero_id} not found for cheat_gold")
        hero.gold += int(decision["amount"])
    elif kind in ("ov_patch", "ov_unstick"):
        from goa2.engine.overrides import apply_override_decision

        apply_override_decision(session, decision["op"], decision.get("args", {}))
    elif kind == "ov_rewind":
        # A rewind changes the *cursor*, not the session; the driving loop
        # (ReplayCursor.seek / replay_game) owns the cursor and must handle it.
        raise ValueError("ov_rewind must be handled by the replay driving loop")
    else:
        raise ValueError(f"Replay: unknown decision type {kind!r}")


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def cleanup_old_replays(replay_dir: str | None = None, ttl_days: int | None = None) -> int:
    """Delete replay files older than ttl_days (by mtime). Returns count removed.

    Independent of the game-save cleanup: a finished/removed game's replay is
    retained for the full TTL so bugs reported later can still be investigated.
    Replays referenced by an *open* bug report are pinned and never deleted;
    resolving or deleting the report releases the pin. Shared replays are pinned
    the same way — the baked share does not read the log, so this only keeps the
    original available for re-baking and debugging.
    """
    from goa2.server.bug_reports import open_report_game_ids
    from goa2.server.shares import shared_game_ids

    directory = Path(replay_dir or _replay_dir())
    if not directory.is_dir():
        return 0
    ttl = (ttl_days if ttl_days is not None else _replay_ttl_days()) * 86400
    now = time.time()
    pinned = open_report_game_ids() | shared_game_ids()
    removed = 0
    for f in directory.glob("*.jsonl"):
        if f.stem in pinned:
            continue
        try:
            if now - f.stat().st_mtime > ttl:
                f.unlink()
                removed += 1
                logger.info("Removed stale replay %s", f.name)
        except OSError:
            logger.exception("Failed to remove stale replay %s", f)
    return removed
