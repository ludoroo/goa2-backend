"""Deterministic resumable evaluation protocol (Rung 1).

Public boundary:

- :class:`AgentSpec`, :class:`GameCase`, :class:`GameObservation`,
  :class:`EvaluationSummary`, :class:`EvaluationProtocol`.
- :func:`run_protocol`, :func:`load_observations`, :func:`summarize`.

The protocol schedules paired A-on-RED / A-on-BLUE games per world seed with
stable case IDs, checkpoints each completed observation to a UTF-8 JSONL file
before starting the next case, resumes exactly the missing cases, and refuses
to reuse rows when any identity field changes.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

# Sides used by the schedule; the engine uses upper-case colour names.
_SIDES: tuple[str, str] = ("RED", "BLUE")

# Wilson score interval z for 95% confidence.
_WILSON_Z = 1.96

# Sidecar suffix for the writer lock file. Kept next to the checkpoint so
# every writer for the same checkpoint contends on the same POSIX inode.
_LOCK_SUFFIX = ".lock"

# Reason string written to a synthetic ``GameObservation`` when a case is
# terminated by the wall-clock timeout. Kept as a module constant so the
# summarizer / callers can key off it without a magic string.
WALL_CLOCK_TIMEOUT_REASON = "wall_clock_timeout"

# Bounded joins after signalling the child. Kept short: the whole point of
# spawn-isolation is that if SIGTERM does not reap the child within a few
# hundred milliseconds we escalate to SIGKILL rather than block the schedule.
_CHILD_TERMINATE_JOIN_S = 0.5
_CHILD_KILL_JOIN_S = 1.0


class CheckpointBusyError(ValueError):
    """Raised when another process already holds the checkpoint writer lock.

    Subclasses :class:`ValueError` so existing callers that already catch
    ``ValueError`` for schema/caller errors also catch contention.
    """


@dataclass(frozen=True, init=False)
class AgentSpec:
    """Serializable specification of one agent under test.

    ``params`` participates in identity hashing so tweaking hyperparameters
    invalidates prior checkpoints. Artifact and telemetry paths are runtime
    locations excluded from identity; artifact content digests remain included.
    """

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        kind: str,
        params: dict[str, Any] | None = None,
        cutoff_telemetry_path: str | None = None,
    ) -> None:
        runtime_params = dict(params or {})
        if cutoff_telemetry_path is not None:
            runtime_params["cutoff_telemetry_path"] = cutoff_telemetry_path
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "params", runtime_params)

    @property
    def cutoff_telemetry_path(self) -> str | None:
        """Runtime telemetry destination, excluded from protocol identity."""
        path = self.params.get("cutoff_telemetry_path")
        return str(path) if path is not None else None

    def identity(self) -> dict[str, Any]:
        # Runtime locations are not experiment identity: identical artifact
        # content must resume across machines and checkout paths.
        runtime_only = {
            "value_model_path",
            "policy_model_path",
            "cutoff_telemetry_path",
        }
        params = {key: value for key, value in self.params.items() if key not in runtime_only}
        return {"name": self.name, "kind": self.kind, "params": params}


@dataclass(frozen=True)
class GameCase:
    """One scheduled game: (world_seed, a_side) plus a stable ``case_id``."""

    case_id: str
    world_seed: int
    a_side: str


@dataclass(frozen=True)
class GameObservation:
    """Outcome of one completed :class:`GameCase`."""

    case_id: str
    world_seed: int
    a_side: str
    winner_side: str | None
    rounds: int
    steps: int
    reason: str

    def to_json(self) -> str:
        payload = {
            "case_id": self.case_id,
            "world_seed": self.world_seed,
            "a_side": self.a_side,
            "winner_side": self.winner_side,
            "rounds": self.rounds,
            "steps": self.steps,
            "reason": self.reason,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> GameObservation:
        required = {"case_id", "world_seed", "a_side", "winner_side", "rounds", "steps", "reason"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"observation missing required keys: {sorted(missing)}")
        return cls(
            case_id=str(row["case_id"]),
            world_seed=int(row["world_seed"]),
            a_side=str(row["a_side"]),
            winner_side=None if row["winner_side"] is None else str(row["winner_side"]),
            rounds=int(row["rounds"]),
            steps=int(row["steps"]),
            reason=str(row["reason"]),
        )


@dataclass
class EvaluationSummary:
    """Aggregated view over a set of :class:`GameObservation` rows."""

    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    max_step_terminations: int = 0
    timeout_terminations: int = 0
    avg_rounds: float = 0.0
    avg_steps: float = 0.0

    @property
    def decisive(self) -> int:
        return self.a_wins + self.b_wins

    @property
    def decisive_a_rate(self) -> float:
        """A's win share among decisive games (draws excluded)."""
        return self.a_wins / self.decisive if self.decisive else 0.0

    def wilson_ci(self, z: float = _WILSON_Z) -> tuple[float, float]:
        """Wilson score interval for A's win-rate over decisive games."""
        n = self.decisive
        if n == 0:
            return (0.0, 1.0)
        p = self.a_wins / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    def screening_passes(self) -> bool:
        """Point-estimate gate: decisive A-rate > 50%, no max_step, no timeout."""
        if self.max_step_terminations > 0:
            return False
        if self.timeout_terminations > 0:
            return False
        if self.decisive == 0:
            return False
        return self.decisive_a_rate > 0.5

    def promotion_passes(self) -> bool:
        """Statistical gate: Wilson lower bound > 50%, no max_step, no timeout."""
        if self.max_step_terminations > 0:
            return False
        if self.timeout_terminations > 0:
            return False
        if self.decisive == 0:
            return False
        lo, _ = self.wilson_ci()
        return lo > 0.5


class EvaluationProtocol:
    """A concrete experiment: A vs B on a fixed schedule under one identity.

    The schedule is the Cartesian product of ``world_seeds`` and the two
    possible sides for agent A. Case IDs are derived from a canonical hash of
    every identity input so any change (agent spec, rosters, map, game type,
    max_steps, source revision, dirty tree hash) produces a disjoint set of
    IDs that cannot alias against a stale checkpoint.
    """

    def __init__(
        self,
        *,
        agent_a: AgentSpec,
        agent_b: AgentSpec,
        red_heroes: tuple[str, ...],
        blue_heroes: tuple[str, ...],
        world_seeds: tuple[int, ...],
        map_path: str,
        game_type: str,
        max_steps: int,
        source_revision: str,
        dirty_tree_hash: str,
        case_timeout_seconds: float | None = None,
    ) -> None:
        if not world_seeds:
            raise ValueError("world_seeds must be non-empty")
        if len(set(world_seeds)) != len(world_seeds):
            raise ValueError("world_seeds must not contain duplicates")
        if not red_heroes or not blue_heroes:
            raise ValueError("red_heroes and blue_heroes must be non-empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if case_timeout_seconds is not None and not (case_timeout_seconds > 0):
            raise ValueError("case_timeout_seconds must be positive when set")
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.red_heroes = tuple(red_heroes)
        self.blue_heroes = tuple(blue_heroes)
        self.world_seeds = tuple(world_seeds)
        self.map_path = map_path
        self.game_type = game_type
        self.max_steps = max_steps
        self.source_revision = source_revision
        self.dirty_tree_hash = dirty_tree_hash
        self.case_timeout_seconds = case_timeout_seconds

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "agent_a": self.agent_a.identity(),
            "agent_b": self.agent_b.identity(),
            "red_heroes": list(self.red_heroes),
            "blue_heroes": list(self.blue_heroes),
            "map_path": self.map_path,
            "game_type": self.game_type,
            "max_steps": self.max_steps,
            "case_timeout_seconds": self.case_timeout_seconds,
            "source_revision": self.source_revision,
            "dirty_tree_hash": self.dirty_tree_hash,
        }

    def identity_digest(self) -> str:
        """Stable 16-hex-char digest over every identity input."""
        blob = json.dumps(self._identity_payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _case_id(self, world_seed: int, a_side: str) -> str:
        payload = {
            "identity": self._identity_payload(),
            "world_seed": world_seed,
            "a_side": a_side,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return f"{digest}-s{world_seed}-{a_side}"

    def cases(self) -> Iterable[GameCase]:
        """Yield paired RED/BLUE cases in ascending world-seed order.

        Ordering is derived from the sorted seed set so callers cannot leak
        list-order into case identity; the two sides are always emitted in
        the fixed order defined by :data:`_SIDES`.
        """
        for seed in sorted(self.world_seeds):
            for side in _SIDES:
                yield GameCase(
                    case_id=self._case_id(seed, side),
                    world_seed=seed,
                    a_side=side,
                )


def _fsync_directory(path: Path) -> None:
    """fsync a directory so recent rename/link operations are durable.

    Opening a directory read-only and fsyncing its fd is the POSIX way to
    persist directory-entry changes (an ``os.replace`` writes a new dirent
    that only the parent directory's inode knows about). Any failure is
    propagated — we do not silently swallow durability errors.
    """
    dir_fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _atomic_rewrite(path: Path, observations: Iterable[GameObservation]) -> None:
    """Rewrite ``path`` to contain exactly ``observations`` as JSONL.

    Uses a temp file in the same directory + :func:`os.replace` so readers
    never see a partial file, then fsyncs the parent directory so the new
    directory entry survives a crash. Called once at :func:`run_protocol`
    startup to drop rows whose case IDs are not in the current schedule
    (stale identity) before any append.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for obs in observations:
                fh.write(obs.to_json())
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Best-effort cleanup; re-raise the original exception.
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise
    # Persist the new dirent. Failures here are real durability failures
    # and must not be hidden.
    _fsync_directory(parent)


def _acquire_writer_lock(checkpoint_path: Path) -> int:
    """Acquire an exclusive non-blocking flock on the sidecar lock file.

    Returns the lock file descriptor; the caller must close it in a
    ``finally`` block to release the advisory lock. Because ``flock`` locks
    are tied to the open file description, closing the fd (or process exit,
    which the kernel handles for us) releases the lock — there is no stale
    lock file to garbage-collect on the next run.

    Raises :class:`CheckpointBusyError` if another process holds the lock.
    """
    lock_path = checkpoint_path.with_name(checkpoint_path.name + _LOCK_SUFFIX)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # O_CREAT so first-run works; permissions are the process default umask.
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            raise CheckpointBusyError(
                f"another process holds the checkpoint lock at {lock_path}"
            ) from exc
        raise
    return fd


def _release_writer_lock(lock_fd: int) -> None:
    """Release the advisory flock by closing the fd (idempotent)."""
    # Closing releases the flock; explicit LOCK_UN is redundant but cheap
    # and makes intent obvious.
    with contextlib.suppress(OSError):
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        os.close(lock_fd)


# --------------------------------------------------------------------------- #
# Per-case wall-clock timeout (spawn-subprocess isolation)                     #
# --------------------------------------------------------------------------- #
#
# When ``EvaluationProtocol.case_timeout_seconds`` is set, each remaining case
# is run in a one-shot ``multiprocessing`` child using the ``spawn`` start
# method. Spawn (not fork) is deliberate: it pickles the runner + case, which
# forces the CLI to hand ``run_protocol`` a picklable top-level callable (no
# closures over per-run agent factories), and it gives us a clean process
# free of the parent's imports/state we can hard-terminate on timeout.
#
# The parent owns the checkpoint file and lock — the child never touches
# either. The child sends exactly one message on a ``Pipe`` and exits.


def _child_entrypoint(
    runner: Callable[[GameCase], GameObservation],
    case: GameCase,
    child_conn: Connection,
) -> None:  # pragma: no cover - executes in a spawned child
    """Run one case in a child process and ship the result back over ``child_conn``.

    The message is a small tuple:

    - ``("ok", GameObservation)`` on success,
    - ``("err", (type_name, message, traceback_text))`` on exception.

    We never raise out of the child: the parent depends on a deterministic
    message shape so a hung / crashed pipe read is unambiguous.
    """
    try:
        obs = runner(case)
        child_conn.send(("ok", obs))
    except BaseException as exc:
        try:
            tb = traceback.format_exc()
            child_conn.send(("err", (type(exc).__name__, str(exc), tb)))
        except BaseException:
            # If sending the diagnostic itself failed there is nothing more
            # we can do from inside the child; the parent's pipe recv will
            # observe EOF and translate it into a synthetic child-crashed
            # failure. Fall through to close/exit.
            pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            child_conn.close()


class _ChildFailure(RuntimeError):
    """Raised in the parent when a spawned child reports a runner exception.

    Carries the child's exception ``type_name`` / ``message`` / ``traceback``
    verbatim so callers can distinguish a crashed runner from a timeout.
    ``__str__`` includes the sentinel message so the class-agnostic tests
    that pattern-match on the child's message still catch it.
    """

    def __init__(self, type_name: str, message: str, tb_text: str) -> None:
        self.type_name = type_name
        self.message = message
        self.traceback_text = tb_text
        super().__init__(f"{type_name}: {message}\n{tb_text}")


def _run_case_with_timeout(
    runner: Callable[[GameCase], GameObservation],
    case: GameCase,
    timeout_seconds: float,
) -> GameObservation:
    """Execute one case in a spawned child, honouring ``timeout_seconds``.

    The wall-clock budget starts before the child is spawned so process
    startup counts against the timeout — hangs during import / spawn cannot
    hide behind a still-loading interpreter.

    Returns a :class:`GameObservation` on success. On timeout returns a
    synthetic timeout observation with ``winner_side=None``, ``rounds=0``,
    ``steps=0``, and ``reason=WALL_CLOCK_TIMEOUT_REASON`` after ensuring the
    child is gone. On child exception raises :class:`_ChildFailure`.

    Cleanup is exhaustive: pipes closed and the process object joined /
    killed on every exit path (success, exception, timeout, KeyboardInterrupt)
    so we never leave an orphan child behind.
    """
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_entrypoint,
        args=(runner, case, child_conn),
    )
    deadline = time.monotonic() + timeout_seconds
    try:
        proc.start()
        # The child owns its end of the pipe. Closing the parent's copy of
        # the *child* end is what makes ``parent_conn.recv()`` observe EOF
        # if the child dies without sending — without this close the recv
        # can hang indefinitely because the fd is still open in this
        # process even after the child exits.
        with contextlib.suppress(OSError, ValueError):
            child_conn.close()

        timed_out = False
        message: Any = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            # ``poll`` is the only way to combine "wait for data" with
            # "wake up on deadline" on a Pipe.
            if parent_conn.poll(remaining):
                try:
                    message = parent_conn.recv()
                except EOFError:
                    # Child exited without sending a message. Treat as
                    # crash so the parent surfaces a real failure rather
                    # than checkpointing a phantom timeout row.
                    message = (
                        "err",
                        (
                            "ChildExited",
                            "child process exited without reporting a result",
                            "",
                        ),
                    )
                break

        if timed_out:
            _terminate_child(proc)
            return GameObservation(
                case_id=case.case_id,
                world_seed=case.world_seed,
                a_side=case.a_side,
                winner_side=None,
                rounds=0,
                steps=0,
                reason=WALL_CLOCK_TIMEOUT_REASON,
            )

        # Message received. Wait briefly for the child to exit cleanly so
        # the OS reaps it and the ``Process`` object clears its exit code.
        proc.join(_CHILD_TERMINATE_JOIN_S)
        if proc.is_alive():
            _terminate_child(proc)

        assert isinstance(message, tuple) and message  # for mypy
        tag = message[0]
        if tag == "ok":
            return message[1]
        # tag == "err"
        type_name, err_message, tb_text = message[1]
        raise _ChildFailure(type_name, err_message, tb_text)
    except KeyboardInterrupt:
        _terminate_child(proc)
        raise
    finally:
        # Belt-and-braces: at the point we leave the function the child
        # MUST be gone. Any surviving process here indicates a supervisor
        # bug — escalate to kill and reap.
        if proc.is_alive():
            _terminate_child(proc)
        with contextlib.suppress(OSError, ValueError):
            parent_conn.close()
        with contextlib.suppress(Exception):
            proc.close()


def _terminate_child(proc: BaseProcess) -> None:
    """Best-effort hard-terminate + reap ``proc``.

    Sends SIGTERM, joins briefly, escalates to SIGKILL if still alive,
    then joins again so ``waitpid`` reaps the exit status. This must never
    raise: it's called from cleanup paths.
    """
    with contextlib.suppress(Exception):
        if proc.is_alive():
            proc.terminate()
            proc.join(_CHILD_TERMINATE_JOIN_S)
    with contextlib.suppress(Exception):
        if proc.is_alive():
            proc.kill()
            proc.join(_CHILD_KILL_JOIN_S)
    with contextlib.suppress(Exception):
        # Final safety net if kill still failed for some reason (permissions?)
        # — join with no timeout to at least try to reap. In practice the
        # bounded joins above are already enough on macOS/Linux.
        proc.join(0.1)


def _require_picklable_runner(runner: Callable[[GameCase], GameObservation]) -> None:
    """Fail clearly unless ``runner`` can be shipped to a spawn child."""
    try:
        pickle.dumps(runner)
    except Exception as exc:
        raise TypeError("timed run_case must be picklable for spawn subprocess execution") from exc


# --------------------------------------------------------------------------- #
# run_protocol                                                                 #
# --------------------------------------------------------------------------- #


def run_protocol(
    protocol: EvaluationProtocol,
    *,
    checkpoint_path: Path,
    run_case: Callable[[GameCase], GameObservation],
) -> list[GameObservation]:
    """Execute ``protocol`` with per-case JSONL checkpointing.

    Behavior:
    - Acquires a non-blocking POSIX ``flock`` on a sidecar ``<path>.lock`` file
      held for the entire load → compaction → append lifecycle. On contention
      raises :class:`CheckpointBusyError` (a :class:`ValueError` subclass).
      The lock is advisory and tied to the open file description, so it is
      released automatically on process exit — no stale lock files.
    - Reads any pre-existing rows at ``checkpoint_path``. At startup, compacts
      the file (atomic temp-file + :func:`os.replace` + parent-directory
      fsync) to keep only rows whose case IDs belong to the current schedule,
      so a reused checkpoint path never accumulates stale identity rows.
    - For each remaining case, invokes ``run_case``, validates that the
      returned :class:`GameObservation` matches the scheduled ``case_id``,
      ``world_seed``, and ``a_side``, then appends the JSON line and flushes
      before the next case starts.
    - Returns the full list of observations for the protocol (cached + freshly
      produced) in schedule order.

    Concurrent :func:`load_observations` readers do not need to lock: they
    only ever see whole-line commits produced by ``flush`` and whole-file
    swaps produced by ``os.replace``.
    """
    checkpoint_path = Path(checkpoint_path)
    timeout_seconds = protocol.case_timeout_seconds
    if timeout_seconds is not None:
        _require_picklable_runner(run_case)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    lock_fd = _acquire_writer_lock(checkpoint_path)
    try:
        all_cases = list(protocol.cases())
        valid_ids = {case.case_id for case in all_cases}

        cached: dict[str, GameObservation] = {}
        if checkpoint_path.exists():
            loaded = load_observations(checkpoint_path)
            kept = [obs for obs in loaded if obs.case_id in valid_ids]
            # Compact once up front if the file contains stale rows. This runs
            # before any append so callers reusing a path across identity
            # changes never end up with an ever-growing checkpoint.
            if len(kept) != len(loaded):
                _atomic_rewrite(checkpoint_path, kept)
            cached = {obs.case_id: obs for obs in kept}

        results: list[GameObservation] = []
        with checkpoint_path.open("a", encoding="utf-8") as fh:
            for case in all_cases:
                hit = cached.get(case.case_id)
                if hit is not None:
                    results.append(hit)
                    continue
                if timeout_seconds is not None:
                    assert timeout_seconds is not None  # for mypy
                    # Timed cases run in a spawned one-shot child. On child
                    # exception the wrapper raises ``_ChildFailure`` which
                    # bubbles up through ``run_protocol`` (no row written).
                    obs = _run_case_with_timeout(run_case, case, timeout_seconds)
                else:
                    obs = run_case(case)
                if not isinstance(obs, GameObservation):
                    raise ValueError("run_case must return a GameObservation")
                if (
                    obs.case_id != case.case_id
                    or obs.world_seed != case.world_seed
                    or obs.a_side != case.a_side
                ):
                    raise ValueError(
                        "run_case produced an observation that does not match the "
                        f"scheduled case: expected case_id={case.case_id!r} "
                        f"world_seed={case.world_seed!r} a_side={case.a_side!r}; "
                        f"got case_id={obs.case_id!r} world_seed={obs.world_seed!r} "
                        f"a_side={obs.a_side!r}"
                    )
                fh.write(obs.to_json())
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
                cached[obs.case_id] = obs
                results.append(obs)

        return results
    finally:
        _release_writer_lock(lock_fd)


def load_observations(path: Path) -> list[GameObservation]:
    """Load observations from a JSONL checkpoint.

    - Empty / blank lines are skipped.
    - A truncated *final* line — one that fails to parse as JSON at all —
      is tolerated to survive crashes mid-append.
    - Any other malformed row (interior parse failure, or a final line that
      parses to a non-object / invalid observation) raises
      :class:`ValueError`. A successfully parsed non-object value is treated
      as corruption because ``run_protocol`` only ever writes JSON objects.
    """
    path = Path(path)
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    raw_lines = text.split("\n")
    # split() on trailing newline yields a phantom empty tail; drop pure-empty
    # entries at the tail so the "final line" concept refers to the last real
    # payload we tried to write.
    lines: list[str] = [ln for ln in raw_lines if ln.strip()]

    observations: list[GameObservation] = []
    for idx, line in enumerate(lines):
        is_last = idx == len(lines) - 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if is_last:
                # Truncated final line — tolerate.
                break
            raise ValueError(f"malformed JSONL row at line {idx + 1}") from None
        if not isinstance(row, dict):
            # A syntactically valid but non-object payload is corruption
            # regardless of position — we never write scalars or arrays.
            raise ValueError(f"malformed JSONL row at line {idx + 1}: not an object")
        observations.append(GameObservation.from_mapping(row))
    return observations


def _is_max_steps(obs: GameObservation) -> bool:
    return obs.reason == "max_steps"


def _is_wall_clock_timeout(obs: GameObservation) -> bool:
    return obs.reason == WALL_CLOCK_TIMEOUT_REASON


def summarize(observations: Iterable[GameObservation]) -> EvaluationSummary:
    """Aggregate observations into an :class:`EvaluationSummary`.

    Winner mapping:
    - ``winner_side is None`` → draw.
    - ``winner_side == a_side`` → A win.
    - otherwise → B win.

    ``max_step_terminations`` counts rows with ``reason == "max_steps"``.
    ``timeout_terminations`` counts rows with ``reason == "wall_clock_timeout"``.
    Both kinds also contribute to ``draws`` (winner_side is None by contract).

    ``avg_rounds`` / ``avg_steps`` are computed over *completed* rows only —
    timeout rows carry ``rounds=0`` / ``steps=0`` synthetic values so
    including them would distort the averages. When every row is a timeout,
    both averages are 0.0. Ordering does not affect the output.
    """
    summary = EvaluationSummary()
    total_rounds = 0
    total_steps = 0
    completed = 0
    for obs in observations:
        is_timeout = _is_wall_clock_timeout(obs)
        if _is_max_steps(obs):
            summary.max_step_terminations += 1
        if is_timeout:
            summary.timeout_terminations += 1
        if obs.winner_side is None:
            summary.draws += 1
        elif obs.winner_side == obs.a_side:
            summary.a_wins += 1
        else:
            summary.b_wins += 1
        # Timeout rows contribute synthetic zero rounds/steps and must be
        # excluded from averages so they don't drag the operator's picture
        # of a normal completed game.
        if not is_timeout:
            total_rounds += obs.rounds
            total_steps += obs.steps
            completed += 1
    if completed:
        summary.avg_rounds = total_rounds / completed
        summary.avg_steps = total_steps / completed
    return summary
