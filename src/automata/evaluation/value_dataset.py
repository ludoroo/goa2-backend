"""Compact value-training dataset recorder + loader.

Records one *value example* per bot decision — the state's feature vector
(:mod:`automata.evaluation.features`) plus the eventual winner label — for
training a learned value model (Rung 2). Deliberately compact: no full
:class:`~goa2.domain.state.GameState` snapshot, no action-space payload,
just what a value head needs.

Design choices
--------------

* **Duck-typed against** :class:`~automata.runtime.trajectory.TrajectoryRecorder`.
  The recorder implements the same ``record_decision`` / ``record_outcome``
  keyword signatures so it drops into :func:`~automata.runtime.harness.run_game`
  through the existing ``recorder=`` seam without any harness changes.

* **Buffer until game_over.** Decisions are held in memory and flushed as a
  single JSONL block only when ``record_outcome`` reports
  ``reason == "game_over"``. Games that end at ``max_steps`` (or any other
  incomplete reason) are dropped: a value example needs a real terminal
  label, and half-games would pollute the training set.

* **Skip unknown-team decisions.** A team-addressed input may occasionally
  arrive with an empty team string (e.g. ``player_id="team:???"``). Such
  rows carry no useful perspective label and are dropped from the buffer,
  but *counted* in :class:`DatasetStats` so callers can spot drift.

* **Deterministic append JSONL.** Row field order is fixed. Two runs with
  identical identity + identical decisions produce byte-for-byte identical
  files. ``append=True`` at construction time appends to an existing dataset
  (multi-game CLIs); the default truncates on flush.

The module has no CLI knowledge — the generator in
:mod:`automata.scripts.generate_value_data` wires this into ``run_game``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

from .features import FEATURE_NAMES, feature_vector

# Bumped whenever the on-disk row schema changes in a non-backwards-compatible
# way (add/remove/rename fields, change semantics). Loaders should refuse
# artifacts whose schema_version they do not understand.
SCHEMA_VERSION: int = 1


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class DatasetStats:
    """Counters exposed on :attr:`ValueDatasetRecorder.stats`.

    ``recorded_*`` counts what landed on disk; ``skipped_*`` counts what
    was consciously dropped (unknown-team decisions and incomplete games).
    Handy for CI/log assertions when running a large generation job.
    """

    recorded_decisions: int = 0
    skipped_decisions: int = 0
    recorded_games: int = 0
    skipped_games: int = 0


@dataclass
class ValueExample:
    """A single value example (dataclass mirror of one written JSONL row)."""

    schema_version: int
    game_id: str
    world_seed: int
    team: str
    features: list[float]
    winner: str | None
    red_heroes: list[str]
    blue_heroes: list[str]
    source_revision: str
    dirty_tree_hash: str
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #


class ValueDatasetRecorder:
    """Buffer per-decision feature rows and flush them on game_over.

    The recorder deliberately mirrors :class:`TrajectoryRecorder`'s
    ``record_decision`` / ``record_outcome`` keyword contract so it can
    be handed to :func:`automata.runtime.harness.run_game` via the
    ``recorder=`` seam. All state-shape assumptions live in
    :func:`automata.evaluation.features.feature_vector`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        game_id: str,
        world_seed: int,
        red_heroes: list[str],
        blue_heroes: list[str],
        source_revision: str,
        dirty_tree_hash: str,
        append: bool = False,
    ) -> None:
        self._path = Path(path)
        self._game_id = game_id
        self._world_seed = int(world_seed)
        self._red_heroes = list(red_heroes)
        self._blue_heroes = list(blue_heroes)
        self._source_revision = source_revision
        self._dirty_tree_hash = dirty_tree_hash
        self._append = bool(append)

        # Rows for the currently-open game, buffered until record_outcome.
        # Each row is a partial dict missing only ``winner``, which is
        # stamped in atomically at flush time.
        self._buffer: list[dict[str, Any]] = []
        self._stats = DatasetStats()

    # --------------------------------------------------------------------- #
    # Public: TrajectoryRecorder-compatible surface
    # --------------------------------------------------------------------- #

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
        """Buffer one value example for the currently-open game.

        Skips (and counts as skipped) any decision whose ``team`` is not
        a known :class:`~goa2.domain.models.TeamColor` — value examples need
        a well-defined perspective label to compute features and to align
        with the eventual winner.
        """
        perspective = _parse_team(team)
        if perspective is None:
            self._stats.skipped_decisions += 1
            return

        features = feature_vector(state, perspective)
        # Row is built in a fixed key order for byte-deterministic JSONL.
        # ``winner`` is added at flush time so buffered rows carry only the
        # per-decision half of the example.
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "game_id": self._game_id,
            "world_seed": self._world_seed,
            "team": perspective.value,
            "features": [float(v) for v in features],
            "feature_names": list(FEATURE_NAMES),
            "red_heroes": list(self._red_heroes),
            "blue_heroes": list(self._blue_heroes),
            "source_revision": self._source_revision,
            "dirty_tree_hash": self._dirty_tree_hash,
        }
        self._buffer.append(row)

    def record_outcome(self, *, winner: str | None, rounds: int, reason: str) -> None:
        """Close the current game; flush buffered rows iff the game completed.

        A ``reason`` other than ``"game_over"`` (e.g. ``"max_steps"``) drops
        the entire buffer and increments :attr:`DatasetStats.skipped_games`.
        A completed game with an empty buffer is *also* counted as skipped
        (no rows to flush, no meaningful example produced).
        """
        buffered = self._buffer
        # Reset first — record_outcome is a hard boundary; a subsequent
        # record_decision would belong to a new game whose caller is
        # expected to reconstruct the recorder or explicitly restart.
        self._buffer = []

        if reason != "game_over" or not buffered:
            self._stats.skipped_games += 1
            return

        for row in buffered:
            row["winner"] = winner

        self._write_rows(buffered)
        self._stats.recorded_games += 1
        self._stats.recorded_decisions += len(buffered)

    # --------------------------------------------------------------------- #
    # Stats + lifecycle
    # --------------------------------------------------------------------- #

    @property
    def stats(self) -> DatasetStats:
        return self._stats

    def close(self) -> None:
        """No-op for API symmetry; the recorder holds no open file handles."""
        return None

    def __enter__(self) -> ValueDatasetRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        """Atomically append or overwrite ``rows`` as newline-delimited JSON.

        Field order in each row is fixed by construction (see
        :meth:`record_decision` + the ``winner`` stamp in
        :meth:`record_outcome`), and ``json.dumps`` preserves dict insertion
        order, so two runs with identical inputs produce byte-identical files.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        with self._path.open(mode, encoding="utf-8") as fh:
            for row in rows:
                # Ensure ``winner`` is the last field for a stable per-row
                # layout regardless of which callsite stamped it in.
                ordered = {k: row[k] for k in _ROW_KEY_ORDER if k in row}
                fh.write(json.dumps(ordered) + "\n")
        # After the first successful write we always append; a caller who
        # constructed the recorder with append=False and then flushes a
        # second game (unusual but not disallowed) will thus preserve the
        # first game's rows.
        self._append = True


# Canonical row field order. Every written row uses this ordering so the
# JSONL output is byte-deterministic across runs.
_ROW_KEY_ORDER: tuple[str, ...] = (
    "schema_version",
    "game_id",
    "world_seed",
    "team",
    "features",
    "feature_names",
    "winner",
    "red_heroes",
    "blue_heroes",
    "source_revision",
    "dirty_tree_hash",
)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    """Load all value examples from a JSONL file into a list of dicts.

    The public shape is stable — the row keys documented on
    :class:`ValueExample` are all present in each returned dict. Callers
    that prefer typed access can construct :class:`ValueExample` instances
    from the dicts themselves; we return dicts to keep the loader lean
    and dependency-free (portable across trainer environments).
    """
    p = Path(path)
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _parse_team(team: str) -> TeamColor | None:
    """Return the :class:`TeamColor` for ``team`` or ``None`` if unknown.

    ``TeamColor(...)`` would raise on an unknown value; we catch that and
    fold it into a single sentinel so the recorder can uniformly count and
    skip such decisions. Empty strings arrive naturally when a driver could
    not resolve a request's team (see
    :func:`automata.runtime.harness._team_of_request`).
    """
    if not team:
        return None
    try:
        return TeamColor(team)
    except ValueError:
        return None
