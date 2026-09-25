"""Run generation, self-play, or evaluation across independent processes."""

from __future__ import annotations

import argparse
import contextlib
import heapq
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, TextIO

from tqdm import tqdm

from automata.evaluation.provenance import (
    GATE_FAILURE_EXIT_STATUS,
    repository_root,
    source_identity,
)
from automata.scripts.generate_joint_bootstrap import (
    CheckpointRow,
    _read_checkpoint,
    build_generator_provenance,
    generator_config_id,
    generator_provenance_path,
    parse_generator_args,
    write_generator_provenance,
)
from automata.scripts.reconcile_self_play import (
    reconcile_self_play,
    validate_self_play_run_directory,
)
from automata.scripts.run_self_play import _parser as _self_play_parser
from automata.training.dataset import JointDatasetRow, iter_joint_dataset, write_joint_dataset
from automata.training.generation import WORKER_COUNT
from automata.training.io import (
    TQDM_BAR_FORMAT,
    atomic_write_bytes,
    canonical_json_bytes,
)
from automata.training.io import (
    atomic_write_bytes as _atomic_publish,
)

OPERATIONAL_FAILURE_EXIT_STATUS = 1
DEFAULT_PROGRESS_INTERVAL = 2.0


@dataclass(frozen=True)
class Shard:
    index: int
    start: int
    count: int


@dataclass(frozen=True)
class ProgressCounts:
    success: int = 0
    failed: int = 0
    timed_out: int = 0

    @property
    def completed(self) -> int:
        return self.success + self.failed + self.timed_out


def partition_range(start: int, count: int, workers: int) -> list[Shard]:
    """Split a contiguous range into balanced, non-empty shards."""
    if count <= 0:
        raise ValueError("count must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    shard_count = min(count, workers)
    quotient, remainder = divmod(count, shard_count)
    shards: list[Shard] = []
    cursor = start
    for index in range(shard_count):
        size = quotient + int(index < remainder)
        shards.append(Shard(index=index, start=cursor, count=size))
        cursor += size
    return shards


def merge_jsonl(parts: list[Path], output: Path, *, identity_field: str) -> int:
    """Atomically merge JSONL parts, deduplicating identical identity rows."""
    rows: dict[str, dict[str, Any]] = {}
    for part in parts:
        if not part.exists():
            raise ValueError(f"missing shard output: {part}")
        with part.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = row.get(identity_field)
                if not isinstance(identity, str) or not identity:
                    raise ValueError(f"{part}:{line_number} has no valid {identity_field!r}")
                previous = rows.get(identity)
                if previous is not None and previous != row:
                    raise ValueError(f"conflicting rows for {identity_field}={identity!r}")
                rows[identity] = row

    atomic_write_bytes(
        output,
        b"".join(canonical_json_bytes(rows[identity]) + b"\n" for identity in sorted(rows)),
    )
    return len(rows)


def merge_joint_bootstrap(
    parts: Sequence[Path],
    checkpoints: Sequence[Path],
    output: Path,
    *,
    expected_seeds: range,
    show_progress: bool = False,
) -> int:
    """Validate and atomically k-way merge complete games in deterministic order."""
    if len(parts) != len(checkpoints):
        raise ValueError("joint parts and checkpoints must have equal lengths")

    completed: set[tuple[str, str, int]] = set()
    for checkpoint in checkpoints:
        checkpoint_rows, _ = _read_checkpoint(checkpoint)
        completed.update(
            (row.game_id, row.config_id, row.world_seed)
            for row in checkpoint_rows
            if row.completed and row.reason == "game_over"
        )

    def ordered_rows(part: Path) -> Iterator[JointDatasetRow]:
        previous_key: tuple[int, str, int] | None = None
        for row in iter_joint_dataset(part):
            key = (row.world_seed, row.game_id, row.decision_index)
            if previous_key is not None and key < previous_key:
                raise ValueError(f"joint shard is not ordered: {part}")
            previous_key = key
            yield row

    iterators: list[Iterator[JointDatasetRow]] = []
    for part in parts:
        if not part.exists():
            raise ValueError(f"missing shard output: {part}")
        iterators.append(ordered_rows(part))

    row_count = 0

    def merged_rows() -> Iterator[JointDatasetRow]:
        nonlocal row_count
        heap: list[tuple[tuple[int, str, int], int, JointDatasetRow]] = []
        for source_index, iterator in enumerate(iterators):
            try:
                row = next(iterator)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                ((row.world_seed, row.game_id, row.decision_index), source_index, row),
            )

        seen_decisions: set[str] = set()
        actual_seeds: set[int] = set()
        seed_games: dict[int, str] = {}
        next_indexes: dict[str, int] = {}
        previous_output: JointDatasetRow | None = None
        current_game_id: str | None = None
        with tqdm(
            total=len(expected_seeds),
            desc="Merging bootstrap",
            unit="game",
            bar_format=TQDM_BAR_FORMAT,
            disable=not show_progress,
        ) as progress:
            while heap:
                _, source_index, row = heapq.heappop(heap)
                backing = (row.game_id, row.generator_config_id, row.world_seed)
                if backing not in completed:
                    raise ValueError(
                        f"orphan decision {row.decision_id!r} has no completed checkpoint"
                    )
                if row.decision_id in seen_decisions:
                    # Sorted equal decisions are adjacent, allowing exact conflict
                    # detection while retaining at most one observation.
                    if previous_output != row:
                        raise ValueError(f"conflicting rows for decision_id={row.decision_id!r}")
                else:
                    seen_decisions.add(row.decision_id)
                    expected_index = next_indexes.get(row.game_id, 0)
                    if row.decision_index != expected_index:
                        raise ValueError(
                            f"game {row.game_id!r} decision indexes are not contiguous from zero"
                        )
                    next_indexes[row.game_id] = expected_index + 1
                    seed_game = seed_games.setdefault(row.world_seed, row.game_id)
                    if seed_game != row.game_id:
                        raise ValueError(f"world seed {row.world_seed} belongs to multiple games")
                    if current_game_id is None:
                        current_game_id = row.game_id
                    elif current_game_id != row.game_id:
                        progress.update()
                        current_game_id = row.game_id
                    actual_seeds.add(row.world_seed)
                    row_count += 1
                    previous_output = row
                    yield row

                iterator = iterators[source_index]
                try:
                    following = next(iterator)
                except StopIteration:
                    continue
                heapq.heappush(
                    heap,
                    (
                        (following.world_seed, following.game_id, following.decision_index),
                        source_index,
                        following,
                    ),
                )

            if current_game_id is not None:
                progress.update()
            required_seeds = set(expected_seeds)
            if actual_seeds != required_seeds:
                missing = sorted(required_seeds - actual_seeds)
                unexpected = sorted(actual_seeds - required_seeds)
                raise ValueError(
                    f"joint merge seed coverage mismatch: missing={missing} unexpected={unexpected}"
                )

    write_joint_dataset(output, merged_rows())
    return row_count


def _extra_args(values: list[str]) -> list[str]:
    return values[1:] if values[:1] == ["--"] else values


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _run_workers(
    commands: list[tuple[Shard, list[str], Path]], *, accepted_return_codes: set[int]
) -> None:
    running: list[tuple[Shard, subprocess.Popen[bytes], Any, Path]] = []
    try:
        for shard, command, log_path in commands:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("wb")
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running.append((shard, process, log_handle, log_path))
            print(
                f"Started shard {shard.index}: start={shard.start} "
                f"count={shard.count} pid={process.pid} log={log_path}",
                flush=True,
            )

        failures: list[str] = []
        for shard, process, log_handle, log_path in running:
            return_code = process.wait()
            log_handle.close()
            if return_code not in accepted_return_codes:
                failures.append(f"shard {shard.index} exited {return_code}; inspect {log_path}")
        if failures:
            raise RuntimeError("; ".join(failures))
    finally:
        for _, process, log_handle, _ in running:
            if process.poll() is None:
                process.terminate()
            if not log_handle.closed:
                log_handle.close()


def checkpoint_progress(
    paths: Sequence[Path], *, config_id: str, expected_seeds: set[int] | None = None
) -> ProgressCounts:
    """Summarize the latest terminal checkpoint record for each game."""
    games: dict[int, CheckpointRow] = {}
    for path in paths:
        try:
            rows, _ = _read_checkpoint(path)
        except (OSError, ValueError):
            # Reporting must never stop healthy workers. The mandatory merge
            # validation surfaces interior corruption after every worker exits.
            continue
        for row in rows:
            if row.config_id == config_id and (
                expected_seeds is None or row.world_seed in expected_seeds
            ):
                games[row.world_seed] = row
    success = sum(row.completed and row.reason == "game_over" for row in games.values())
    timed_out = sum(row.reason == "wall_clock_timeout" for row in games.values())
    return ProgressCounts(
        success=success,
        failed=len(games) - success - timed_out,
        timed_out=timed_out,
    )


def _run_joint_workers(
    commands: list[tuple[Shard, list[str], Path]],
    *,
    checkpoints: Sequence[Path],
    config_id: str,
    total: int,
    workers: int,
    show_progress: bool = True,
    stream: TextIO | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
) -> None:
    """Run every shard to completion and report checkpoint-derived progress."""
    output_stream = sys.stderr if stream is None else stream
    running: list[tuple[Shard, subprocess.Popen[bytes], Any, Path]] = []
    expected_seeds = {
        seed for shard, _, _ in commands for seed in range(shard.start, shard.start + shard.count)
    }
    initial_counts = checkpoint_progress(
        checkpoints, expected_seeds=expected_seeds, config_id=config_id
    )
    latest_counts = initial_counts
    progress = None
    try:
        for shard, command, log_path in commands:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("wb")
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running.append((shard, process, log_handle, log_path))
            print(
                f"Started joint shard {shard.index}: start={shard.start} "
                f"count={shard.count} pid={process.pid} log={log_path}",
                file=output_stream,
                flush=True,
            )

        progress = tqdm(
            total=total,
            initial=initial_counts.completed,
            unit="game",
            bar_format=TQDM_BAR_FORMAT,
            disable=not show_progress,
            file=output_stream,
            mininterval=progress_interval,
        )
        unfinished = set(range(len(running)))
        failures: list[str] = []
        progress.set_postfix(
            {
                "workers": f"{len(unfinished)}/{workers}",
                "ok": initial_counts.success,
                "failed": initial_counts.failed,
                "timeout": initial_counts.timed_out,
            },
            refresh=False,
        )
        next_report = clock() + progress_interval
        while unfinished:
            worker_exited = False
            for index in tuple(unfinished):
                shard, process, log_handle, log_path = running[index]
                return_code = process.poll()
                if return_code is None:
                    continue
                unfinished.remove(index)
                worker_exited = True
                log_handle.close()
                if return_code != 0:
                    failures.append(f"shard {shard.index} exited {return_code}; inspect {log_path}")
            now = clock()
            if worker_exited or now >= next_report or not unfinished:
                counts = checkpoint_progress(
                    checkpoints, expected_seeds=expected_seeds, config_id=config_id
                )
                delta = counts.completed - latest_counts.completed
                progress.set_postfix(
                    {
                        "workers": f"{len(unfinished)}/{workers}",
                        "ok": counts.success,
                        "failed": counts.failed,
                        "timeout": counts.timed_out,
                    },
                    refresh=False,
                )
                if delta:
                    progress.update(delta)
                latest_counts = counts
                next_report = now + progress_interval
            if unfinished:
                sleep(min(progress_interval, 0.25))
        if failures:
            raise RuntimeError("; ".join(failures))
    finally:
        for _, process, log_handle, _ in running:
            if process.poll() is None:
                process.terminate()
            if not log_handle.closed:
                log_handle.close()
        if progress is not None:
            counts = checkpoint_progress(
                checkpoints, expected_seeds=expected_seeds, config_id=config_id
            )
            delta = counts.completed - latest_counts.completed
            progress.set_postfix(
                {
                    "workers": f"0/{workers}",
                    "ok": counts.success,
                    "failed": counts.failed,
                    "timeout": counts.timed_out,
                },
                refresh=False,
            )
            if delta:
                progress.update(delta)
            progress.close()


def _joint_bootstrap(args: argparse.Namespace) -> int:
    if args.seed_end <= args.seed_start:
        raise ValueError("--seed-end must be greater than --seed-start")
    shards = partition_range(args.seed_start, args.seed_end - args.seed_start, args.workers)
    output = Path(args.out)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(f"{output}.shards")
    checkpoint_dir = Path(f"{checkpoint}.shards")
    if bool(args.source_revision) != bool(args.dirty_tree_hash):
        raise ValueError("--source-revision and --dirty-tree-hash must be supplied together")
    if args.source_revision:
        revision, dirty_hash = args.source_revision, args.dirty_tree_hash
    else:
        revision, dirty_hash = source_identity(
            exclude_paths=(
                repository_root() / "runs",
                output,
                checkpoint,
                output_dir,
                checkpoint_dir,
                generator_provenance_path(output),
            )
        )
    extra = _extra_args(args.generator_options)
    reserved = {"--out", "--checkpoint", "--seed-start", "--seed-end"}
    if any(option in extra for option in reserved):
        raise ValueError(
            "worker-specific --out, --checkpoint, --seed-start, and --seed-end "
            "cannot be forwarded"
        )

    commands: list[tuple[Shard, list[str], Path]] = []
    parts: list[Path] = []
    checkpoints: list[Path] = []
    first_worker_argv: list[str] | None = None
    for shard in shards:
        part = output_dir / f"part-{shard.index:02d}.jsonl.zst"
        shard_checkpoint = checkpoint_dir / f"part-{shard.index:02d}.games.jsonl"
        parts.append(part)
        checkpoints.append(shard_checkpoint)
        worker_argv = [
            *extra,
            *([] if "--no-progress" in extra else ["--no-progress"]),
            "--out",
            str(part),
            "--checkpoint",
            str(shard_checkpoint),
            "--seed-start",
            str(shard.start),
            "--seed-end",
            str(shard.start + shard.count),
            "--source-revision",
            revision,
            "--dirty-tree-hash",
            dirty_hash,
        ]
        if first_worker_argv is None:
            first_worker_argv = worker_argv
        command = [
            sys.executable,
            "-m",
            "automata.scripts.generate_joint_bootstrap",
            *worker_argv,
        ]
        commands.append((shard, command, Path(f"{part}.log")))

    assert first_worker_argv is not None
    worker_args = parse_generator_args(first_worker_argv)
    config_id = generator_config_id(
        worker_args,
        source_revision=revision,
        dirty_tree_hash=dirty_hash,
    )
    _run_joint_workers(
        commands,
        checkpoints=checkpoints,
        config_id=config_id,
        total=args.seed_end - args.seed_start,
        workers=len(shards),
        show_progress=args.progress,
        progress_interval=args.progress_interval,
    )
    row_count = merge_joint_bootstrap(
        parts,
        checkpoints,
        output,
        expected_seeds=range(args.seed_start, args.seed_end),
        show_progress=args.progress,
    )
    # The consolidated checkpoint is informational; shard checkpoints remain
    # authoritative for resume and retain timeout/incomplete attempts.
    checkpoint_rows: list[CheckpointRow] = []
    for path in checkpoints:
        rows, _ = _read_checkpoint(path)
        checkpoint_rows.extend(rows)
    checkpoint_rows.sort(key=lambda row: (row.world_seed, row.game_id))
    _atomic_publish(
        checkpoint,
        b"".join(canonical_json_bytes(row) + b"\n" for row in checkpoint_rows),
    )
    write_generator_provenance(
        output,
        build_generator_provenance(
            worker_args,
            source_revision=revision,
            dirty_tree_hash=dirty_hash,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
        ),
    )
    print(f"Merged {row_count} joint decisions into {output}", file=sys.stderr)
    return 0


def _checkpoint_case_count(path: Path, expected: int) -> None:
    if not path.exists():
        raise RuntimeError(f"evaluation shard did not create {path}")
    case_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("case_id")
            if isinstance(case_id, str):
                case_ids.add(case_id)
    if len(case_ids) != expected:
        raise RuntimeError(f"{path} contains {len(case_ids)}/{expected} expected cases")


def _remove_checkpoints(paths: list[Path]) -> None:
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


class _WorkerArgumentParser(argparse.ArgumentParser):
    """Turn nested worker usage errors into wrapper operational errors."""

    def error(self, message: str) -> Never:
        raise ValueError(f"invalid run_self_play options: {message}")


def self_play_completed_seeds(
    checkpoint_dir: Path, *, seed_start: int, seed_end: int
) -> tuple[int, ...]:
    """Return valid completed seeds for low-cost parent progress reporting."""
    completed: set[int] = set()
    for worker_id in range(WORKER_COUNT):
        path = checkpoint_dir / f"worker-{worker_id}.jsonl"
        try:
            with path.open("rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        continue
                    try:
                        row = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    seed = row.get("world_seed")
                    if (
                        row.get("worker_id") == worker_id
                        and row.get("reason") == "game_over"
                        and isinstance(seed, int)
                        and not isinstance(seed, bool)
                        and seed_start <= seed < seed_end
                        and (seed - seed_start) % WORKER_COUNT == worker_id
                    ):
                        completed.add(seed)
        except OSError:
            continue
    return tuple(sorted(completed))


def _run_self_play_workers(
    commands: Sequence[tuple[int, list[str], Path, Path]],
    *,
    show_progress: bool,
    checkpoint_dir: Path | None = None,
    seed_start: int = 0,
    seed_end: int = 0,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
) -> None:
    """Run persistent self-play workers with isolated logs and aggregate progress."""
    running: list[tuple[int, subprocess.Popen[bytes], Any, Any, Path, Path]] = []
    completed = (
        self_play_completed_seeds(checkpoint_dir, seed_start=seed_start, seed_end=seed_end)
        if checkpoint_dir is not None
        else ()
    )
    latest_completed = len(completed)
    progress = None
    try:
        for worker_id, command, stdout_path, stderr_path in commands:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = stdout_path.open("ab")
            try:
                stderr_handle = stderr_path.open("ab")
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                    )
                except BaseException:
                    stderr_handle.close()
                    raise
            except BaseException:
                stdout_handle.close()
                raise
            running.append(
                (
                    worker_id,
                    process,
                    stdout_handle,
                    stderr_handle,
                    stdout_path,
                    stderr_path,
                )
            )
        progress = tqdm(
            total=seed_end - seed_start if checkpoint_dir is not None else len(commands),
            initial=latest_completed,
            desc="Self-play",
            unit="game",
            bar_format=TQDM_BAR_FORMAT,
            disable=not show_progress,
            file=sys.stderr,
            mininterval=progress_interval,
        )
        unfinished = set(range(len(running)))
        failures: list[str] = []
        next_report = time.monotonic() + progress_interval
        while unfinished and not failures:
            for index in tuple(unfinished):
                worker_id, process, stdout_handle, stderr_handle, stdout_path, stderr_path = (
                    running[index]
                )
                return_code = process.poll()
                if return_code is None:
                    continue
                unfinished.remove(index)
                stdout_handle.close()
                stderr_handle.close()
                if return_code != 0:
                    failures.append(
                        f"worker {worker_id} exited {return_code}; inspect "
                        f"{stdout_path} and {stderr_path}"
                    )
            now = time.monotonic()
            if checkpoint_dir is not None and (now >= next_report or not unfinished):
                current_completed = len(
                    self_play_completed_seeds(
                        checkpoint_dir, seed_start=seed_start, seed_end=seed_end
                    )
                )
                # Do not force a redraw while the completed-game count is unchanged.
                # ``set_postfix`` defaults to ``refresh=True``, which otherwise emits
                # duplicate terminal history on every polling interval.
                progress.set_postfix(workers=f"{len(unfinished)}/{WORKER_COUNT}", refresh=False)
                if current_completed > latest_completed:
                    progress.update(current_completed - latest_completed)
                    latest_completed = current_completed
                next_report = now + progress_interval
            if unfinished and not failures:
                time.sleep(0.25)
        if failures:
            raise RuntimeError("; ".join(failures))
    finally:
        for _, process, stdout_handle, stderr_handle, _, _ in running:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not stdout_handle.closed:
                stdout_handle.close()
            if not stderr_handle.closed:
                stderr_handle.close()
        if progress is not None:
            if checkpoint_dir is not None:
                current_completed = len(
                    self_play_completed_seeds(
                        checkpoint_dir, seed_start=seed_start, seed_end=seed_end
                    )
                )
                progress.set_postfix(workers=f"0/{WORKER_COUNT}", refresh=False)
                if current_completed > latest_completed:
                    progress.update(current_completed - latest_completed)
            progress.close()


def _self_play(args: argparse.Namespace) -> int:
    if args.seed_end <= args.seed_start:
        raise ValueError("--seed-end must be greater than --seed-start")
    if args.seed_end - args.seed_start < WORKER_COUNT:
        raise ValueError(f"self-play requires at least {WORKER_COUNT} games")
    if bool(args.source_revision) != bool(args.dirty_tree_hash):
        raise ValueError("--source-revision and --dirty-tree-hash must be supplied together")

    run_dir = validate_self_play_run_directory(args.run_dir)
    output_dir = run_dir / "output"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    destination = run_dir / "generation.jsonl.zst"

    if args.source_revision:
        revision, dirty_hash = args.source_revision, args.dirty_tree_hash
    else:
        revision, dirty_hash = source_identity(exclude_paths=(repository_root() / "runs", run_dir))

    extra = _extra_args(args.self_play_options)
    reserved = {
        "--seed-start",
        "--seed-end",
        "--worker-id",
        "--output-dir",
        "--checkpoint-dir",
        "--source-revision",
        "--dirty-tree-hash",
        "--no-progress",
        "--help",
        "-h",
    }
    conflicting = sorted(
        option
        for option in reserved
        if any(token == option or token.startswith(option + "=") for token in extra)
    )
    if conflicting:
        raise ValueError(
            "wrapper-owned self-play options cannot be forwarded: " + ", ".join(conflicting)
        )

    commands: list[tuple[int, list[str], Path, Path]] = []
    for worker_id in range(WORKER_COUNT):
        worker_argv = [
            *extra,
            "--seed-start",
            str(args.seed_start),
            "--seed-end",
            str(args.seed_end),
            "--worker-id",
            str(worker_id),
            "--output-dir",
            str(output_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--source-revision",
            revision,
            "--dirty-tree-hash",
            dirty_hash,
            "--no-progress",
        ]
        # Fail before launching any process if forwarded worker arguments are
        # missing, conflicting, or malformed.
        worker_parser = _WorkerArgumentParser(
            prog="run_self_play",
            description="Validate parallel self-play worker options",
            allow_abbrev=False,
        )
        _self_play_parser(worker_parser).parse_args(worker_argv)
        commands.append(
            (
                worker_id,
                [sys.executable, "-m", "automata.scripts.run_self_play", *worker_argv],
                log_dir / f"worker-{worker_id}.stdout",
                log_dir / f"worker-{worker_id}.stderr",
            )
        )

    for path in (output_dir, checkpoint_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    _run_self_play_workers(
        commands,
        show_progress=args.progress,
        checkpoint_dir=checkpoint_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )
    result = reconcile_self_play(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        destination=destination,
        worker_count=WORKER_COUNT,
    )
    print(
        f"Published {result.game_count} games / {result.row_count} decisions to "
        f"{result.destination}",
        file=sys.stderr,
    )
    return 0


def _evaluation(args: argparse.Namespace) -> int:
    shards = partition_range(args.seed, args.paired_seeds, args.workers)
    checkpoint = Path(args.checkpoint)
    part_dir = Path(f"{checkpoint}.shards")
    extra = _extra_args(args.evaluation_options)
    commands: list[tuple[Shard, list[str], Path]] = []
    part_checkpoints: list[Path] = []
    for shard in shards:
        part = part_dir / f"part-{shard.index:02d}.jsonl"
        part_checkpoints.append(part)
        command = [
            sys.executable,
            "-m",
            "automata.evaluation.cli",
            *extra,
            "--seed",
            str(shard.start),
            "--paired-seeds",
            str(shard.count),
            "--checkpoint",
            str(part),
        ]
        commands.append((shard, command, Path(f"{part}.log")))

    try:
        _run_workers(
            commands,
            accepted_return_codes={0, GATE_FAILURE_EXIT_STATUS},
        )
        for shard, part in zip(shards, part_checkpoints, strict=True):
            _checkpoint_case_count(part, shard.count * 2)
        case_count = merge_jsonl(part_checkpoints, checkpoint, identity_field="case_id")
    except BaseException:
        _remove_checkpoints([*part_checkpoints, checkpoint])
        raise
    print(f"Merged {case_count} cases into {checkpoint}")

    summary_command = [
        sys.executable,
        "-m",
        "automata.evaluation.cli",
        *extra,
        "--seed",
        str(args.seed),
        "--paired-seeds",
        str(args.paired_seeds),
        "--checkpoint",
        str(checkpoint),
    ]
    result = subprocess.run(summary_command, check=False)
    if result.returncode not in {0, GATE_FAILURE_EXIT_STATUS}:
        _remove_checkpoints([*part_checkpoints, checkpoint])
        raise RuntimeError(f"evaluation summary exited {result.returncode}")
    return result.returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    joint = subparsers.add_parser(
        "joint-bootstrap", help="Shard heuristic joint policy/value generation"
    )
    joint.add_argument("--out", required=True)
    joint.add_argument("--checkpoint", required=True)
    joint.add_argument("--seed-start", required=True, type=int)
    joint.add_argument("--seed-end", required=True, type=int)
    joint.add_argument("--workers", type=int, default=4)
    joint.add_argument(
        "--source-revision",
        help="pin the source revision used for generation identity and resume",
    )
    joint.add_argument(
        "--dirty-tree-hash",
        help="pin the dirty-tree hash used for generation identity and resume",
    )
    joint.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="disable parent-process checkpoint progress reporting",
    )
    joint.add_argument(
        "--progress-interval",
        type=_positive_float,
        default=DEFAULT_PROGRESS_INTERVAL,
        metavar="SECONDS",
        help=f"tqdm refresh interval (default: {DEFAULT_PROGRESS_INTERVAL:g})",
    )
    joint.add_argument("generator_options", nargs=argparse.REMAINDER)
    joint.set_defaults(handler=_joint_bootstrap)

    self_play = subparsers.add_parser(
        "self-play", help="Run and reconcile the fixed four-worker self-play protocol"
    )
    self_play.add_argument(
        "--run-dir",
        required=True,
        help="fresh or resumable run root (output/checkpoints/logs are derived)",
    )
    self_play.add_argument("--seed-start", required=True, type=int)
    self_play.add_argument("--seed-end", required=True, type=int)
    self_play.add_argument(
        "--source-revision",
        help="pin the source revision used for every worker identity",
    )
    self_play.add_argument(
        "--dirty-tree-hash",
        help="pin the dirty-tree hash used for every worker identity",
    )
    self_play.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="disable parent-process worker status lines",
    )
    self_play.add_argument(
        "self_play_options",
        nargs=argparse.REMAINDER,
        help="run_self_play options after -- (except wrapper-owned paths/seeds)",
    )
    self_play.set_defaults(handler=_self_play)

    evaluation = subparsers.add_parser("evaluation", help="Shard targeted gameplay evaluation")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--seed", required=True, type=int)
    evaluation.add_argument("--paired-seeds", required=True, type=int)
    evaluation.add_argument("--workers", required=True, type=int)
    evaluation.add_argument("evaluation_options", nargs=argparse.REMAINDER)
    evaluation.set_defaults(handler=_evaluation)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "self-play":
        try:
            return int(args.handler(args))
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"self-play failed: {exc}", file=sys.stderr)
            return OPERATIONAL_FAILURE_EXIT_STATUS
    if args.command != "joint-bootstrap":
        return int(args.handler(args))
    if args.seed_end <= args.seed_start:
        parser.error("--seed-end must be greater than --seed-start")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if bool(args.source_revision) != bool(args.dirty_tree_hash):
        parser.error("--source-revision and --dirty-tree-hash must be supplied together")
    reserved = {"--out", "--checkpoint", "--seed-start", "--seed-end"}
    if any(option in _extra_args(args.generator_options) for option in reserved):
        parser.error(
            "worker-specific --out, --checkpoint, --seed-start, and --seed-end "
            "cannot be forwarded"
        )
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"joint-bootstrap failed: {exc}", file=sys.stderr)
        return OPERATIONAL_FAILURE_EXIT_STATUS


if __name__ == "__main__":
    raise SystemExit(main())
