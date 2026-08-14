"""RED tests for the deterministic resumable evaluation protocol (Rung 1).

Covers the public boundary of :mod:`automata.evaluation.protocol`:
B1 identity (schedule + case-id stability), B2 checkpoint/resume (durable
JSONL append, exact skip, truncated-final tolerance, malformed-interior
rejection, identity invalidation), B3 aggregation + gates, and B4 the new
per-case wall-clock timeout that isolates each timed case in its own
one-shot spawned subprocess.

A fake ``run_case`` callable stands in for the real game/ISMCTS runner so
these tests never invoke a game.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import stat
import time
from pathlib import Path
from typing import Any

import pytest

from automata.evaluation import protocol as protocol_module
from automata.evaluation.protocol import (
    AgentSpec,
    CheckpointBusyError,
    EvaluationProtocol,
    GameCase,
    GameObservation,
    load_observations,
    run_protocol,
    summarize,
)

# --- module-level helpers for spawn-picklable timed run_case tests ----------
#
# The wall-clock timeout contract requires ``run_protocol`` to execute each
# timed ``run_case`` in a one-shot spawned subprocess. The ``spawn`` start
# method pickles the callable, so the runner and any side-channel it uses
# must be defined at module top level and reachable by import — no lambdas
# or nested defs.
#
# Every helper below writes a "child alive"/"child exited" marker to a
# caller-provided file so the parent can observe child lifecycle without
# racing on ``proc.poll()``.


def _hang_runner_marker_dir(base: Path) -> Path:
    d = base / "child_markers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_marker(marker_dir: Path, name: str) -> None:
    (marker_dir / name).write_text(str(os.getpid()), encoding="utf-8")


class HangingRunner:
    """Picklable run_case that busy-spins forever after touching a marker.

    Two teeth make this genuinely un-cooperative:

    1. SIGTERM is masked to SIG_IGN, so a supervisor that only sends
       SIGTERM (and waits politely) will hang the test suite. The parent
       is forced to escalate to SIGKILL / ``Process.kill()`` / equivalent
       to reclaim the child.
    2. The busy loop is CPU-bound (a tight arithmetic spin, no
       ``time.sleep``, no I/O, no ``select``). A ``sleep``-based hang would
       return from ``sleep`` early on some signal deliveries even with
       SIGTERM ignored; a CPU spin never yields, so only genuine hard
       termination stops the child.

    Deterministic: no RNG, no wall-clock branches — the loop body only
    depends on its own state.
    """

    def __init__(self, marker_dir: Path) -> None:
        self.marker_dir = Path(marker_dir)

    def __call__(self, case: GameCase) -> GameObservation:  # pragma: no cover - child
        # Ignore cooperative signals so a broken implementation that only
        # sends SIGTERM (and waits politely) would hang the test suite.
        with contextlib.suppress(OSError, ValueError):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _write_marker(self.marker_dir, f"alive-{case.case_id}")
        # CPU-bound busy spin. ``x`` and the modulo keep the compiler /
        # interpreter honest: no ``pass`` / ``time.sleep`` that could
        # cooperate with the OS on signal delivery.
        x = 0
        while True:
            x = (x + 1) % 1_000_003


class RaisingRunner:
    """Picklable run_case that raises a distinctive exception in the child."""

    sentinel = "child-boom-marker"

    def __call__(self, case: GameCase) -> GameObservation:  # pragma: no cover - child
        raise RuntimeError(self.sentinel)


class HappyRunner:
    """Picklable run_case that returns an A-wins-on-side observation."""

    def __call__(self, case: GameCase) -> GameObservation:
        return GameObservation(
            case_id=case.case_id,
            world_seed=case.world_seed,
            a_side=case.a_side,
            winner_side=case.a_side,
            rounds=5,
            steps=100,
            reason="game_over",
        )


RED = ("Wasp", "Xargatha")
BLUE = ("Arien", "Brogan")


def _agent(name: str = "a", iterations: int = 4) -> AgentSpec:
    return AgentSpec(name=name, kind="ismcts", params={"iterations": iterations})


def _protocol(**overrides: Any) -> EvaluationProtocol:
    base: dict[str, Any] = dict(
        agent_a=_agent("agent_a"),
        agent_b=_agent("agent_b"),
        red_heroes=RED,
        blue_heroes=BLUE,
        world_seeds=(0, 1, 2, 3, 4, 5),
        map_path="src/goa2/data/maps/forgotten_island.json",
        game_type="QUICK",
        max_steps=20_000,
        source_revision="rev-abc",
        dirty_tree_hash="clean",
    )
    base.update(overrides)
    return EvaluationProtocol(**base)


def _obs(
    case_id: str = "c",
    a_side: str = "RED",
    winner_side: str | None = "RED",
    reason: str = "game_over",
    world_seed: int = 0,
) -> GameObservation:
    return GameObservation(
        case_id=case_id,
        world_seed=world_seed,
        a_side=a_side,
        winner_side=winner_side,
        rounds=10,
        steps=500,
        reason=reason,
    )


class _FakeRunner:
    """Records case invocations; returns a default A-wins-on-RED outcome."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, case: GameCase) -> GameObservation:
        self.calls.append(case.case_id)
        return _obs(
            case_id=case.case_id,
            a_side=case.a_side,
            world_seed=case.world_seed,
            winner_side="RED",
        )


# --- B1: schedule / identity ------------------------------------------------


def test_canonical_screening_yields_twelve_cases_covering_both_sides() -> None:
    cases = list(_protocol().cases())
    assert len(cases) == 12
    by_seed: dict[int, set[str]] = {}
    for c in cases:
        by_seed.setdefault(c.world_seed, set()).add(c.a_side)
    assert set(by_seed) == {0, 1, 2, 3, 4, 5}
    assert all(sides == {"RED", "BLUE"} for sides in by_seed.values())


def test_protocol_rejects_duplicate_world_seeds() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        _protocol(world_seeds=(0, 1, 0))


def test_case_ids_are_unique_and_stable_and_order_independent() -> None:
    a = {(c.world_seed, c.a_side): c.case_id for c in _protocol().cases()}
    b = {(c.world_seed, c.a_side): c.case_id for c in _protocol().cases()}
    shuffled = {
        (c.world_seed, c.a_side): c.case_id
        for c in _protocol(world_seeds=(5, 3, 1, 4, 0, 2)).cases()
    }
    assert a == b == shuffled
    assert len(set(a.values())) == 12


@pytest.mark.parametrize(
    "field,value",
    [
        ("map_path", "src/goa2/data/maps/other.json"),
        ("game_type", "TOURNAMENT"),
        ("max_steps", 10_000),
        ("source_revision", "rev-xyz"),
        ("dirty_tree_hash", "deadbeef"),
        ("red_heroes", ("Wasp", "Brogan")),
        ("blue_heroes", ("Arien", "Xargatha")),
    ],
)
def test_case_id_changes_when_identity_field_changes(field: str, value: Any) -> None:
    baseline = {c.case_id for c in _protocol().cases()}
    changed = {c.case_id for c in _protocol(**{field: value}).cases()}
    assert baseline.isdisjoint(changed), f"changing {field!r} must change every case_id"


def test_case_id_changes_when_agent_spec_changes() -> None:
    baseline = {c.case_id for c in _protocol().cases()}
    changed = {c.case_id for c in _protocol(agent_a=_agent("agent_a", iterations=8)).cases()}
    assert baseline.isdisjoint(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_seeds", ()),
        ("red_heroes", ()),
        ("blue_heroes", ()),
        ("max_steps", 0),
        ("max_steps", -1),
    ],
)
def test_protocol_rejects_invalid_configuration(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        _protocol(**{field: value})


# --- B2: checkpoint / resume ------------------------------------------------


def test_run_protocol_writes_jsonl_row_per_case(tmp_path: Path) -> None:
    checkpoint = tmp_path / "obs.jsonl"
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())
    lines = [ln for ln in checkpoint.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 12
    for ln in lines:
        row = json.loads(ln)
        assert {"case_id", "world_seed", "a_side", "winner_side", "reason"} <= set(row)


def test_each_row_is_durable_before_next_case_runs(tmp_path: Path) -> None:
    """Row N must be visible before case N+1 begins."""
    checkpoint = tmp_path / "obs.jsonl"
    counts: list[int] = []

    class _Probe(_FakeRunner):
        def __call__(self, case: GameCase) -> GameObservation:
            text = checkpoint.read_text(encoding="utf-8") if checkpoint.exists() else ""
            counts.append(sum(1 for ln in text.splitlines() if ln.strip()))
            return super().__call__(case)

    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_Probe())
    assert counts == list(range(12))


def test_each_appended_row_is_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "obs.jsonl"
    file_fsyncs = 0
    real_fsync = os.fsync

    def _spy_fsync(fd: int) -> None:
        nonlocal file_fsyncs
        if stat.S_ISREG(os.fstat(fd).st_mode):
            file_fsyncs += 1
        real_fsync(fd)

    monkeypatch.setattr(protocol_module.os, "fsync", _spy_fsync)

    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())

    assert file_fsyncs == 12


def test_completed_cases_are_skipped_on_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "obs.jsonl"
    first = _FakeRunner()
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=first)
    assert len(first.calls) == 12
    second = _FakeRunner()
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=second)
    assert second.calls == []


def test_partial_run_resumes_only_missing_cases(tmp_path: Path) -> None:
    proto = _protocol()
    checkpoint = tmp_path / "obs.jsonl"

    # Bootstrap a full checkpoint then trim to the first 5 rows to simulate a
    # crash after 5/12 completed cases.
    run_protocol(proto, checkpoint_path=checkpoint, run_case=_FakeRunner())
    lines = [ln for ln in checkpoint.read_text(encoding="utf-8").splitlines() if ln.strip()]
    checkpoint.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")
    kept = {json.loads(ln)["case_id"] for ln in lines[:5]}

    resumed = _FakeRunner()
    run_protocol(proto, checkpoint_path=checkpoint, run_case=resumed)
    assert len(resumed.calls) == 7
    assert not (set(resumed.calls) & kept)


def test_truncated_final_line_is_tolerated(tmp_path: Path) -> None:
    checkpoint = tmp_path / "obs.jsonl"
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())
    lines = [ln for ln in checkpoint.read_text(encoding="utf-8").splitlines() if ln.strip()]
    truncated = lines[-1][: len(lines[-1]) // 2]
    checkpoint.write_text("\n".join(lines[:-1]) + "\n" + truncated, encoding="utf-8")

    assert len(load_observations(checkpoint)) == 11


def test_malformed_interior_line_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "obs.jsonl"
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())
    lines = [ln for ln in checkpoint.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lines[3] = "this-is-not-json"
    checkpoint.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_observations(checkpoint)


def test_identity_change_invalidates_cached_observations(tmp_path: Path) -> None:
    checkpoint = tmp_path / "obs.jsonl"
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())
    replay = _FakeRunner()
    run_protocol(
        _protocol(source_revision="rev-xyz"),
        checkpoint_path=checkpoint,
        run_case=replay,
    )
    assert len(replay.calls) == 12


def test_stale_rows_are_compacted_on_identity_change(tmp_path: Path) -> None:
    """Reusing a checkpoint under a new identity must not grow the file.

    After running the second protocol to completion, only the rows for the
    *current* schedule may remain on disk — the stale rows from the previous
    identity must have been dropped by the startup compaction.
    """
    checkpoint = tmp_path / "obs.jsonl"
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())
    stale_ids = {c.case_id for c in _protocol().cases()}

    new_proto = _protocol(source_revision="rev-xyz")
    fresh_ids = {c.case_id for c in new_proto.cases()}
    assert stale_ids.isdisjoint(fresh_ids)

    run_protocol(new_proto, checkpoint_path=checkpoint, run_case=_FakeRunner())

    disk_ids = {obs.case_id for obs in load_observations(checkpoint)}
    assert disk_ids == fresh_ids
    assert stale_ids.isdisjoint(disk_ids)


def test_runner_mismatched_output_is_rejected_and_not_written(tmp_path: Path) -> None:
    """Runner must return an observation that matches the scheduled case."""
    checkpoint = tmp_path / "obs.jsonl"

    class _WrongCaseIdRunner:
        def __call__(self, case: GameCase) -> GameObservation:
            return _obs(
                case_id="not-the-scheduled-id",
                a_side=case.a_side,
                world_seed=case.world_seed,
                winner_side="RED",
            )

    with pytest.raises(ValueError):
        run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_WrongCaseIdRunner())

    # Nothing should have been appended for the offending case; because the
    # very first case fails, the file is either absent or empty.
    if checkpoint.exists():
        text = checkpoint.read_text(encoding="utf-8")
        assert text.strip() == ""

    class _WrongSeedRunner:
        def __call__(self, case: GameCase) -> GameObservation:
            return _obs(
                case_id=case.case_id,
                a_side=case.a_side,
                world_seed=case.world_seed + 999,
                winner_side="RED",
            )

    with pytest.raises(ValueError):
        run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_WrongSeedRunner())

    class _WrongSideRunner:
        def __call__(self, case: GameCase) -> GameObservation:
            flipped = "BLUE" if case.a_side == "RED" else "RED"
            return _obs(
                case_id=case.case_id,
                a_side=flipped,
                world_seed=case.world_seed,
                winner_side="RED",
            )

    with pytest.raises(ValueError):
        run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_WrongSideRunner())


def test_valid_non_object_final_record_is_rejected(tmp_path: Path) -> None:
    """A syntactically valid but non-object final line is corruption."""
    checkpoint = tmp_path / "obs.jsonl"
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())
    lines = [ln for ln in checkpoint.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Replace the final row with a parseable JSON scalar; we never write scalars
    # so this must be treated as corruption rather than tolerated.
    checkpoint.write_text("\n".join(lines[:-1]) + "\n" + "42" + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_observations(checkpoint)


# --- B3: aggregation & gates ------------------------------------------------


def test_winner_mapping_covers_a_b_and_draw() -> None:
    a_win = summarize([_obs(a_side="RED", winner_side="RED")])
    b_win = summarize([_obs(a_side="RED", winner_side="BLUE")])
    draw = summarize([_obs(winner_side=None)])
    b_win_blue = summarize([_obs(a_side="BLUE", winner_side="RED")])
    assert (a_win.a_wins, a_win.b_wins, a_win.draws) == (1, 0, 0)
    assert (b_win.a_wins, b_win.b_wins, b_win.draws) == (0, 1, 0)
    assert (draw.a_wins, draw.b_wins, draw.draws) == (0, 0, 1)
    assert (b_win_blue.a_wins, b_win_blue.b_wins) == (0, 1)


def test_max_step_terminations_are_counted_and_are_draws() -> None:
    rows = [
        _obs(case_id="1", winner_side="RED"),
        _obs(case_id="2", winner_side=None, reason="max_steps"),
        _obs(case_id="3", winner_side=None, reason="max_steps"),
    ]
    summary = summarize(rows)
    assert summary.max_step_terminations == 2
    assert summary.draws >= 2  # max_steps rows are non-decisive


def test_summary_is_order_independent() -> None:
    rows = [
        _obs(case_id="1", a_side="RED", winner_side="RED"),
        _obs(case_id="2", a_side="BLUE", winner_side="RED"),
        _obs(case_id="3", winner_side=None),
        _obs(case_id="4", winner_side=None, reason="max_steps"),
    ]
    fwd, rev = summarize(rows), summarize(list(reversed(rows)))
    assert (fwd.a_wins, fwd.b_wins, fwd.draws, fwd.max_step_terminations) == (
        rev.a_wins,
        rev.b_wins,
        rev.draws,
        rev.max_step_terminations,
    )


def _mix(a_wins: int, b_wins: int, max_steps: int = 0) -> list[GameObservation]:
    rows: list[GameObservation] = []
    n = 0
    for _ in range(a_wins):
        rows.append(_obs(case_id=f"a{n}", a_side="RED", winner_side="RED"))
        n += 1
    for _ in range(b_wins):
        rows.append(_obs(case_id=f"b{n}", a_side="RED", winner_side="BLUE"))
        n += 1
    for _ in range(max_steps):
        rows.append(_obs(case_id=f"m{n}", winner_side=None, reason="max_steps"))
        n += 1
    return rows


def test_screening_passes_only_when_point_above_fifty_and_no_max_steps() -> None:
    assert summarize(_mix(8, 4)).screening_passes() is True
    assert summarize(_mix(6, 6)).screening_passes() is False
    assert summarize(_mix(4, 8)).screening_passes() is False
    assert summarize(_mix(11, 0, max_steps=1)).screening_passes() is False


def test_promotion_requires_wilson_lower_above_fifty_and_no_max_steps() -> None:
    assert summarize(_mix(6, 6)).promotion_passes() is False
    sweep = summarize(_mix(12, 0))
    lo, _ = sweep.wilson_ci()
    assert lo > 0.5
    assert sweep.promotion_passes() is True
    assert summarize(_mix(11, 0, max_steps=1)).promotion_passes() is False


def test_screening_can_pass_while_promotion_fails() -> None:
    # 7-5 point estimate ~58% but Wilson lower bound is below 0.5.
    marginal = summarize(_mix(7, 5))
    assert marginal.screening_passes() is True
    assert marginal.promotion_passes() is False


# --- writer-lock lifecycle & directory-fsync durability ----------------------


def _hold_sidecar_lock(checkpoint: Path) -> int:
    """Take an exclusive non-blocking flock on the sidecar file.

    Uses the same sidecar path convention run_protocol uses (``<name>.lock``
    next to the checkpoint) so we simulate a concurrent writer without
    reaching into private helpers.
    """
    lock_path = checkpoint.with_name(checkpoint.name + ".lock")
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_run_protocol_rejects_concurrent_writer(tmp_path: Path) -> None:
    """A second writer must fail fast with CheckpointBusyError (a ValueError)."""
    checkpoint = tmp_path / "obs.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    holder_fd = _hold_sidecar_lock(checkpoint)
    try:
        runner = _FakeRunner()
        with pytest.raises(CheckpointBusyError):
            run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=runner)
        # Subclass of ValueError so legacy callers still catch it.
        assert issubclass(CheckpointBusyError, ValueError)
        # Contended writer must not have called run_case at all.
        assert runner.calls == []
    finally:
        fcntl.flock(holder_fd, fcntl.LOCK_UN)
        os.close(holder_fd)


def test_writer_lock_is_released_after_run(tmp_path: Path) -> None:
    """After a successful run, the lock must be re-acquirable."""
    checkpoint = tmp_path / "obs.jsonl"
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())

    # Should not raise: previous run released the flock.
    fd = _hold_sidecar_lock(checkpoint)
    try:
        pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    # And a second run_protocol invocation also succeeds (no leaked lock,
    # observations resumed exactly).
    replay = _FakeRunner()
    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=replay)
    assert replay.calls == []


def test_writer_lock_released_after_runtime_error(tmp_path: Path) -> None:
    """A crashing run_case must not leak the writer lock."""
    checkpoint = tmp_path / "obs.jsonl"

    class _Boom:
        def __call__(self, case: GameCase) -> GameObservation:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_Boom())

    fd = _hold_sidecar_lock(checkpoint)
    try:
        pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_compaction_fsyncs_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup compaction must fsync a directory fd, not just the temp file.

    We wrap ``os.fsync`` inside the protocol module and record whether it
    was called on a directory fd. Compaction is triggered by seeding the
    checkpoint with a row whose case_id is not in the current schedule.
    """
    checkpoint = tmp_path / "obs.jsonl"

    # Seed with a valid JSONL row that will be dropped as stale.
    stale = {
        "case_id": "stale-id-not-in-schedule",
        "world_seed": 999,
        "a_side": "RED",
        "winner_side": "RED",
        "rounds": 1,
        "steps": 1,
        "reason": "game_over",
    }
    checkpoint.write_text(json.dumps(stale) + "\n", encoding="utf-8")

    dir_fsyncs: list[int] = []
    real_fsync = os.fsync

    def _spy_fsync(fd: int) -> None:
        try:
            mode = os.fstat(fd).st_mode
            if stat.S_ISDIR(mode):
                dir_fsyncs.append(fd)
        except OSError:
            pass
        real_fsync(fd)

    monkeypatch.setattr(protocol_module.os, "fsync", _spy_fsync)

    run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())

    assert dir_fsyncs, "expected compaction to fsync the parent directory fd"


def test_compaction_does_not_swallow_directory_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory-fsync OSError during compaction must surface, not be hidden."""
    checkpoint = tmp_path / "obs.jsonl"
    stale = {
        "case_id": "stale-id-not-in-schedule",
        "world_seed": 999,
        "a_side": "RED",
        "winner_side": "RED",
        "rounds": 1,
        "steps": 1,
        "reason": "game_over",
    }
    checkpoint.write_text(json.dumps(stale) + "\n", encoding="utf-8")

    real_fsync = os.fsync

    def _fail_on_dir_fsync(fd: int) -> None:
        try:
            mode = os.fstat(fd).st_mode
        except OSError:
            mode = 0
        if stat.S_ISDIR(mode):
            raise OSError("simulated fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(protocol_module.os, "fsync", _fail_on_dir_fsync)

    with pytest.raises(OSError, match="simulated fsync failure"):
        run_protocol(_protocol(), checkpoint_path=checkpoint, run_case=_FakeRunner())


# --- B4: per-case wall-clock timeout (spawned subprocess isolation) --------
#
# Contract: ``EvaluationProtocol`` gains an optional ``case_timeout_seconds``
# (positive float). When set, ``run_protocol`` must execute each case in a
# one-shot spawned subprocess so a hung case is genuinely terminable — the
# child process is gone, and the parent checkpoints a synthetic
# ``GameObservation(reason='wall_clock_timeout', winner_side=None, rounds=0,
# steps=0)`` for that case. Timeout rows resume as completed; a retry
# requires the timeout identity to change.


def test_protocol_defaults_case_timeout_seconds_to_none() -> None:
    """Backward compat: existing constructions omit case_timeout_seconds."""
    proto = _protocol()
    # Attribute must exist and default to None so legacy callers get the
    # in-process runner behavior unchanged.
    assert getattr(proto, "case_timeout_seconds", "MISSING") is None


def test_protocol_accepts_positive_case_timeout_seconds() -> None:
    proto = _protocol(case_timeout_seconds=12.5)
    assert proto.case_timeout_seconds == pytest.approx(12.5)


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.001])
def test_protocol_rejects_non_positive_case_timeout_seconds(bad: float) -> None:
    with pytest.raises(ValueError):
        _protocol(case_timeout_seconds=bad)


def test_case_timeout_seconds_participates_in_identity() -> None:
    """Changing timeout must invalidate every case_id (and the digest)."""
    baseline = _protocol()
    changed = _protocol(case_timeout_seconds=30.0)
    base_ids = {c.case_id for c in baseline.cases()}
    new_ids = {c.case_id for c in changed.cases()}
    assert base_ids.isdisjoint(
        new_ids
    ), "adding case_timeout_seconds must invalidate cached case IDs"
    assert baseline.identity_digest() != changed.identity_digest()

    # A different positive timeout must again yield a disjoint schedule.
    other = _protocol(case_timeout_seconds=60.0)
    other_ids = {c.case_id for c in other.cases()}
    assert other_ids.isdisjoint(new_ids)
    assert other.identity_digest() != changed.identity_digest()


def test_timed_protocol_rejects_non_picklable_runner_before_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "obs.jsonl"
    calls: list[str] = []

    def _local_runner(case: GameCase) -> GameObservation:
        calls.append(case.case_id)
        return HappyRunner()(case)

    with pytest.raises(TypeError, match="picklable"):
        run_protocol(
            _protocol(world_seeds=(0,), case_timeout_seconds=1.0),
            checkpoint_path=checkpoint,
            run_case=_local_runner,
        )

    assert calls == []
    assert not checkpoint.exists()


def test_untimed_protocol_accepts_local_runner(tmp_path: Path) -> None:
    calls: list[str] = []

    def _local_runner(case: GameCase) -> GameObservation:
        calls.append(case.case_id)
        return HappyRunner()(case)

    results = run_protocol(
        _protocol(world_seeds=(0,)),
        checkpoint_path=tmp_path / "obs.jsonl",
        run_case=_local_runner,
    )

    assert calls == [case.case_id for case in _protocol(world_seeds=(0,)).cases()]
    assert len(results) == 2


def _timeout_protocol(tmp_path: Path, **overrides: Any) -> EvaluationProtocol:
    """A tiny protocol with a short case timeout for spawn-subprocess tests."""
    base: dict[str, Any] = dict(
        world_seeds=(0,),
        case_timeout_seconds=1.5,
    )
    base.update(overrides)
    return _protocol(**base)


def _pid_is_gone(pid: int, *, poll_budget_s: float = 2.0) -> bool:
    """True if ``pid`` is not visible to us as a live process.

    Uses ``os.kill(pid, 0)`` which is portable across macOS/Linux. Polls
    briefly because a hard-killed child may not have been reaped by its
    parent yet at the instant :func:`run_protocol` returns — that
    reaping is asynchronous and can lag by a scheduler quantum.

    Documented PID-reuse tradeoff: after a hard kill the OS is free to
    hand ``pid`` to some *other* process, in which case ``os.kill(pid, 0)``
    succeeds and this function reports "still alive". Two mitigations make
    this acceptable for this test without coupling to the implementation:

    - Cannot run child cleanup after SIGKILL, so we cannot record an
      "exit observed" flag from inside the child. We rely on parent
      observations (marker file written before the busy spin) plus the
      prompt-return elapsed bound above to catch a broken supervisor.
    - The reuse window on macOS/Linux for a freshly killed pid is on the
      order of many seconds under normal load, and our timeout budget
      keeps the whole test <20 s, so a reuse false-positive would require
      the host to churn through the entire pid space during the test.

    A false negative here (we say "gone" but the OS actually has our
    child still around) is what we care about most, and that requires
    both the kernel to fail to deliver a signal AND our polling window
    to expire — the polling window is long enough to make that unlikely
    on real CI hardware.
    """
    deadline = time.monotonic() + poll_budget_s
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # PID reused by an unrelated process we cannot signal. We
            # cannot distinguish this from "our child is still there but
            # we lost the right to signal it" without racing on
            # ``waitpid``; treat as "gone" and rely on prompt-return +
            # marker observability elsewhere in the test.
            return True
        if time.monotonic() >= deadline:
            return False
        # Short poll — the OS reaper typically runs within a few ms of
        # SIGKILL delivery, so 25 ms is enough headroom without ballooning
        # the test wall-clock.
        time.sleep(0.025)


def test_run_protocol_wall_clock_timeout_terminates_child_and_checkpoints_row(
    tmp_path: Path,
) -> None:
    """A hung child must be terminated and produce a synthetic timeout row.

    Behavior asserted (avoiding overspecification of the process supervisor):
      1. run_protocol returns within a small multiple of the timeout.
      2. Every case yields a GameObservation with reason='wall_clock_timeout',
         winner_side=None, rounds=0, steps=0.
      3. The child process the parent spawned is no longer alive after
         run_protocol returns (i.e. it was actually terminated). Because
         hard process termination cannot run child cleanup, the child
         cannot record its own exit — so we observe termination from the
         parent side via a briefly polled ``os.kill(pid, 0)``. See
         :func:`_pid_is_gone` for the documented PID-reuse tradeoff.
      4. The synthetic rows are appended to the JSONL checkpoint.
    """
    checkpoint = tmp_path / "obs.jsonl"
    marker_dir = _hang_runner_marker_dir(tmp_path)
    runner = HangingRunner(marker_dir)
    proto = _timeout_protocol(tmp_path)

    start = time.monotonic()
    results = run_protocol(proto, checkpoint_path=checkpoint, run_case=runner)
    elapsed = time.monotonic() - start

    # A single-seed protocol produces two cases (RED + BLUE). Wall-clock
    # elapsed must be a small multiple of the timeout — the exact upper
    # bound is deliberately generous so we do not spec pool warm-up.
    assert 0 < elapsed < proto.case_timeout_seconds * 12

    assert len(results) == 2
    for obs in results:
        assert obs.reason == "wall_clock_timeout"
        assert obs.winner_side is None
        assert obs.rounds == 0
        assert obs.steps == 0

    # Rows are checkpointed exactly as returned.
    on_disk = load_observations(checkpoint)
    assert {(o.case_id, o.reason) for o in on_disk} == {
        (o.case_id, "wall_clock_timeout") for o in results
    }

    # Each case saw its child actually start (marker exists) and no
    # child process holding the marker PID is still alive when observed
    # by the parent. This is the strongest check we can make without
    # coupling to the implementation's specific supervisor mechanism.
    for obs in results:
        marker = marker_dir / f"alive-{obs.case_id}"
        assert marker.exists(), f"child for {obs.case_id} never wrote its alive marker"
        pid = int(marker.read_text(encoding="utf-8").strip())
        assert _pid_is_gone(pid), (
            f"child pid {pid} for case {obs.case_id} still alive after "
            f"run_protocol returned; supervisor did not hard-terminate the "
            f"busy-spinning child"
        )


def test_timeout_rows_resume_as_completed(tmp_path: Path) -> None:
    """A checkpointed timeout row must be treated as done on resume."""
    checkpoint = tmp_path / "obs.jsonl"
    marker_dir = _hang_runner_marker_dir(tmp_path)
    proto = _timeout_protocol(tmp_path)

    run_protocol(proto, checkpoint_path=checkpoint, run_case=HangingRunner(marker_dir))

    # RaisingRunner is picklable but must not be invoked because both timeout
    # rows are already complete in the checkpoint.
    results = run_protocol(proto, checkpoint_path=checkpoint, run_case=RaisingRunner())
    assert all(o.reason == "wall_clock_timeout" for o in results)


def test_changing_case_timeout_forces_retry(tmp_path: Path) -> None:
    """Retry only happens when case_timeout_seconds identity changes."""
    checkpoint = tmp_path / "obs.jsonl"
    marker_dir = _hang_runner_marker_dir(tmp_path)
    first = _timeout_protocol(tmp_path, case_timeout_seconds=1.0)
    run_protocol(first, checkpoint_path=checkpoint, run_case=HangingRunner(marker_dir))

    # Identical protocol → no retry. RaisingRunner would fail if invoked.
    run_protocol(first, checkpoint_path=checkpoint, run_case=RaisingRunner())

    # A different timeout produces disjoint case IDs; the schedule must
    # then require the runner to execute again.
    retried = _timeout_protocol(tmp_path, case_timeout_seconds=2.0)
    results = run_protocol(retried, checkpoint_path=checkpoint, run_case=HappyRunner())
    assert [o.case_id for o in results] == [c.case_id for c in retried.cases()]
    assert all(o.reason == "game_over" for o in results)


def test_child_exception_propagates_and_no_row_written(tmp_path: Path) -> None:
    """A child raising must become a parent failure — no timeout stand-in."""
    checkpoint = tmp_path / "obs.jsonl"
    proto = _timeout_protocol(tmp_path)

    with pytest.raises(Exception) as excinfo:
        run_protocol(proto, checkpoint_path=checkpoint, run_case=RaisingRunner())

    # The RaisingRunner's sentinel message should thread through so callers
    # can distinguish "child crashed" from "timeout expired". We do not pin
    # the exception class (implementations differ across multiprocessing
    # start methods) but we do pin that the child's message survives.
    assert RaisingRunner.sentinel in str(excinfo.value)

    # And critically: no synthetic observation was checkpointed.
    if checkpoint.exists():
        assert load_observations(checkpoint) == []


# --- B5: summary treatment of timeout rows ---------------------------------


def _timeout_obs(case_id: str = "t", a_side: str = "RED") -> GameObservation:
    return GameObservation(
        case_id=case_id,
        world_seed=0,
        a_side=a_side,
        winner_side=None,
        rounds=0,
        steps=0,
        reason="wall_clock_timeout",
    )


def test_summary_counts_timeout_terminations() -> None:
    rows = [
        _obs(case_id="1", winner_side="RED"),
        _timeout_obs(case_id="t1"),
        _timeout_obs(case_id="t2", a_side="BLUE"),
    ]
    summary = summarize(rows)
    assert getattr(summary, "timeout_terminations", 0) == 2


def test_timeout_rows_block_screening_and_promotion() -> None:
    """Even a dominant A record must fail both gates when any row timed out."""
    rows = [_obs(case_id=f"a{i}", winner_side="RED") for i in range(11)]
    rows.append(_timeout_obs())
    summary = summarize(rows)
    assert summary.screening_passes() is False
    assert summary.promotion_passes() is False


def test_avg_rounds_and_steps_ignore_timeout_rows() -> None:
    """Timeout rows must not distort avg_rounds / avg_steps.

    Two normally completed observations at rounds=10, steps=500 plus a
    timeout row (rounds=0, steps=0) must yield averages of 10 and 500,
    not the naive (10+10+0)/3 = 6.67 / (500+500+0)/3 = 333.3.
    """
    rows = [
        _obs(case_id="ok1", winner_side="RED"),
        _obs(case_id="ok2", winner_side="RED"),
        _timeout_obs(case_id="t1"),
    ]
    summary = summarize(rows)
    assert summary.avg_rounds == pytest.approx(10.0)
    assert summary.avg_steps == pytest.approx(500.0)


def test_summary_all_timeouts_gives_zero_averages_and_blocks_gates() -> None:
    """All-timeout summary: averages are 0 (no completed rows) and gates fail."""
    rows = [_timeout_obs(case_id=f"t{i}") for i in range(3)]
    summary = summarize(rows)
    assert getattr(summary, "timeout_terminations", 0) == 3
    assert summary.avg_rounds == pytest.approx(0.0)
    assert summary.avg_steps == pytest.approx(0.0)
    assert summary.screening_passes() is False
    assert summary.promotion_passes() is False
