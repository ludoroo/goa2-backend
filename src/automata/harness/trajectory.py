"""Trajectory recording for self-play.

The self-play loop (:func:`run_game`) can optionally emit, per *decision*, a full
serialized ``GameState`` snapshot plus the decision context, then a terminal
outcome row. This is the raw training substrate for learned value/policy models
join decision rows to their game's outcome to get
``(state -> outcome)`` and ``(state, legal -> chosen)`` examples.

Design choices:
- **Off by default.** ``run_game`` only records when handed a recorder, so eval
  self-play stays perf-neutral.
- **Full state snapshots** (``GameState.model_dump(mode="json")``) — maximum
  offline flexibility: any future feature set or NN encoder can be derived
  without re-running self-play. Chosen over compact feature vectors deliberately.
- **Append-only JSONL, streamed to disk.** Rows are written as they occur
  (hundreds per game, ~180 KB each) rather than buffered in RAM. Each row
  carries ``game_id`` + ``decision_index``; a terminal ``{"kind":"outcome",...}``
  row lets a loader label every decision with the eventual winner without
  rewriting rows.

The recorder is a small Protocol so a null/in-memory/file implementation can be
swapped freely; ``run_game`` depends only on the Protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from goa2.domain.state import GameState


class TrajectoryRecorder(Protocol):
    """Sink for per-decision snapshots and the final game outcome."""

    def record_decision(
        self,
        *,
        state: GameState,
        team: str,
        decision_kind: str,
        player_id: str,
        legal_keys: list[Any],
        chosen_key: Any,
    ) -> None:
        """Record one decision (a full state snapshot + the choice made)."""
        ...

    def record_outcome(self, *, winner: str | None, rounds: int, reason: str) -> None:
        """Record the terminal result; closes the trajectory."""
        ...


class NullRecorder:
    """No-op recorder (explicit 'record nothing')."""

    def record_decision(self, **_: Any) -> None:
        return None

    def record_outcome(self, **_: Any) -> None:
        return None


class JsonlRecorder:
    """Stream decisions + outcome to an append-only JSONL file.

    One decision row per call:
        {"kind":"decision","game_id":..,"decision_index":i,"team":..,
         "decision_kind":..,"player_id":..,"legal_keys":[..],"chosen_key":..,
         "state":{..full snapshot..}}
    followed by one terminal row:
        {"kind":"outcome","game_id":..,"winner":..,"rounds":..,"reason":..}
    """

    def __init__(self, path: str | Path, *, game_id: str = "game") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._game_id = game_id
        self._index = 0
        # Truncate on open so each run starts clean.
        self._fh = self._path.open("w", encoding="utf-8")

    def record_decision(
        self,
        *,
        state: GameState,
        team: str,
        decision_kind: str,
        player_id: str,
        legal_keys: list[Any],
        chosen_key: Any,
    ) -> None:
        row = {
            "kind": "decision",
            "game_id": self._game_id,
            "decision_index": self._index,
            "team": team,
            "decision_kind": decision_kind,
            "player_id": player_id,
            "legal_keys": [_jsonable(k) for k in legal_keys],
            "chosen_key": _jsonable(chosen_key),
            "state": state.model_dump(mode="json"),
        }
        self._fh.write(json.dumps(row) + "\n")
        self._index += 1

    def record_outcome(self, *, winner: str | None, rounds: int, reason: str) -> None:
        row = {
            "kind": "outcome",
            "game_id": self._game_id,
            "winner": winner,
            "rounds": rounds,
            "reason": reason,
            "decisions": self._index,
        }
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class InMemoryRecorder:
    """Collect rows in memory (handy for tests / small runs)."""

    def __init__(self, game_id: str = "game") -> None:
        self.game_id = game_id
        self.decisions: list[dict[str, Any]] = []
        self.outcome: dict[str, Any] | None = None

    def record_decision(
        self,
        *,
        state: GameState,
        team: str,
        decision_kind: str,
        player_id: str,
        legal_keys: list[Any],
        chosen_key: Any,
    ) -> None:
        self.decisions.append(
            {
                "decision_index": len(self.decisions),
                "team": team,
                "decision_kind": decision_kind,
                "player_id": player_id,
                "legal_keys": [_jsonable(k) for k in legal_keys],
                "chosen_key": _jsonable(chosen_key),
                "state": state.model_dump(mode="json"),
            }
        )

    def record_outcome(self, *, winner: str | None, rounds: int, reason: str) -> None:
        self.outcome = {"winner": winner, "rounds": rounds, "reason": reason}


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe rendering of an action key (dict/scalar/tuple)."""
    if isinstance(value, dict | list | str | int | float | bool) or value is None:
        return value
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return str(value)
