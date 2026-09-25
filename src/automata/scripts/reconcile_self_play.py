"""Reconcile complete persistent self-play workers into one immutable dataset.

This command is the production publication path for multi-worker self-play. It
streams one complete game fragment at a time and never materializes graph rows.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard

from automata.models.contracts import canonical_json_bytes
from automata.training.dataset import JointDatasetRow
from automata.training.generation import WORKER_COUNT, CheckpointRow
from automata.training.io import (
    canonical_json_bytes as canonical_value_bytes,
)
from automata.training.io import (
    content_digest,
    fsync_directory,
)

OPERATIONAL_FAILURE_EXIT_STATUS = 1
_READ_SIZE = 8 * 1024


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    destination: Path
    provenance_path: Path
    dataset_digest: str
    output_digest: str
    row_count: int
    game_count: int


@dataclass(frozen=True, slots=True)
class _CheckpointFile:
    worker_id: int
    name: str
    digest: str
    worker_config_id: str
    rows: tuple[CheckpointRow, ...]


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise ValueError(f"unsafe path contains '..': {candidate}")
    return Path(os.path.abspath(os.fspath(candidate)))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} must not contain symlinks: {current}")


def _require_directory(path: Path, *, label: str) -> None:
    _reject_symlink_components(path, label=label)
    if not path.is_dir():
        raise ValueError(f"{label} must be an existing directory: {path}")


def _require_regular_file(path: Path, *, label: str) -> None:
    _reject_symlink_components(path, label=label)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file: {path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_self_play_run_directory(path: str | Path) -> Path:
    """Validate a fresh or resumable run root before workers are launched."""
    run_dir = _absolute(path)
    _reject_symlink_components(run_dir, label="self-play run directory")
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(f"self-play run directory must be a directory: {run_dir}")
    return run_dir


def _validate_paths(
    output_dir: Path,
    checkpoint_dir: Path,
    destination: Path,
    provenance_path: Path,
) -> None:
    _require_directory(output_dir, label="self-play output directory")
    _require_directory(checkpoint_dir, label="self-play checkpoint directory")
    if (
        output_dir == checkpoint_dir
        or _is_within(output_dir, checkpoint_dir)
        or _is_within(checkpoint_dir, output_dir)
    ):
        raise ValueError("unsafe source directory overlap")
    if not destination.name.endswith(".jsonl.zst"):
        raise ValueError("aggregate destination must end in .jsonl.zst")
    if (
        destination == provenance_path
        or _is_within(destination, provenance_path)
        or _is_within(provenance_path, destination)
    ):
        raise ValueError("aggregate destination and provenance path must be safely distinct")
    for path, label in (
        (destination, "aggregate destination"),
        (provenance_path, "aggregate provenance"),
    ):
        if _is_within(path, output_dir) or _is_within(path, checkpoint_dir):
            raise ValueError(f"unsafe {label} overlaps self-play source files")
        path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(path, label=label)
        if path.exists() and not path.is_file():
            raise ValueError(f"{label} must be a regular file when it exists")


@contextmanager
def _publication_lock(destination: Path) -> Iterator[None]:
    lock_path = destination.with_name(destination.name + ".lock")
    _reject_symlink_components(lock_path, label="publication lock")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("publication lock must be a regular file")
        # Deliberately blocking: a second publisher validates the first result
        # after it acquires the same lock instead of failing spuriously.
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _stream_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checkpoint(path: Path, worker_id: int) -> _CheckpointFile:
    _require_regular_file(path, label=f"worker {worker_id} checkpoint")
    digest = hashlib.sha256()
    rows: list[CheckpointRow] = []
    worker_config_id: str | None = None
    with path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"worker {worker_id} checkpoint has a partial final row at {line_number}"
                )
            raw = line[:-1]
            if not raw:
                raise ValueError(f"worker {worker_id} checkpoint contains a blank row")
            try:
                row = CheckpointRow.model_validate_json(raw)
            except ValueError as exc:
                raise ValueError(
                    f"invalid worker {worker_id} checkpoint row {line_number}: {exc}"
                ) from exc
            if raw != canonical_json_bytes(row):
                raise ValueError(
                    f"invalid worker {worker_id} checkpoint row {line_number}: non-canonical JSON"
                )
            if row.worker_id != worker_id:
                raise ValueError("self-play checkpoint worker assignment mismatch")
            if worker_config_id is None:
                worker_config_id = row.worker_config_id
            elif worker_config_id != row.worker_config_id:
                raise ValueError("self-play checkpoint has multiple worker identities")
            rows.append(row)
    if not rows or worker_config_id is None:
        raise ValueError(f"worker {worker_id} checkpoint is empty")
    return _CheckpointFile(
        worker_id=worker_id,
        name=path.name,
        digest=digest.hexdigest(),
        worker_config_id=worker_config_id,
        rows=tuple(rows),
    )


def _checkpoint_common_identity(row: CheckpointRow) -> tuple[object, ...]:
    return (
        row.generation_id,
        row.source_config_id,
        row.generator_config_id,
        row.parent_model_digest,
        row.parent_generation,
        row.observation_schema_version,
    )


def _load_checkpoints(
    checkpoint_dir: Path, *, seed_start: int, seed_end: int, worker_count: int
) -> tuple[tuple[_CheckpointFile, ...], tuple[CheckpointRow, ...]]:
    files = tuple(
        _read_checkpoint(checkpoint_dir / f"worker-{worker_id}.jsonl", worker_id)
        for worker_id in range(worker_count)
    )
    expected_seeds = set(range(seed_start, seed_end))
    for checkpoint in files:
        expected_worker_seeds = tuple(
            seed
            for seed in range(seed_start, seed_end)
            if (seed - seed_start) % worker_count == checkpoint.worker_id
        )
        actual_worker_seeds = tuple(row.world_seed for row in checkpoint.rows)
        if actual_worker_seeds != expected_worker_seeds:
            raise ValueError(
                f"worker {checkpoint.worker_id} checkpoint assignment/order mismatch: "
                f"expected={expected_worker_seeds} actual={actual_worker_seeds}"
            )
        expected_worker_config_id = content_digest(
            {
                "worker_id": checkpoint.worker_id,
                "generator_config_id": checkpoint.rows[0].generator_config_id,
                "games": [row.game_id for row in checkpoint.rows],
            }
        )
        if checkpoint.worker_config_id != expected_worker_config_id:
            raise ValueError(f"worker {checkpoint.worker_id} checkpoint worker identity mismatch")
    by_seed: dict[int, CheckpointRow] = {}
    game_ids: set[str] = set()
    common: tuple[object, ...] | None = None
    for checkpoint in files:
        for row in checkpoint.rows:
            expected_worker = (row.world_seed - seed_start) % worker_count
            if row.world_seed not in expected_seeds or expected_worker != checkpoint.worker_id:
                raise ValueError(
                    f"self-play checkpoint seed {row.world_seed} has wrong worker assignment"
                )
            if row.world_seed in by_seed:
                raise ValueError(f"duplicate self-play checkpoint seed {row.world_seed}")
            if row.game_id in game_ids:
                raise ValueError(f"duplicate self-play checkpoint game {row.game_id}")
            identity = _checkpoint_common_identity(row)
            if common is None:
                common = identity
            elif identity != common:
                raise ValueError("self-play checkpoints do not share one generation identity")
            by_seed[row.world_seed] = row
            game_ids.add(row.game_id)
    if set(by_seed) != expected_seeds:
        missing = sorted(expected_seeds - set(by_seed))
        unexpected = sorted(set(by_seed) - expected_seeds)
        raise ValueError(
            f"self-play checkpoint seed coverage mismatch: missing={missing} unexpected={unexpected}"
        )
    return files, tuple(by_seed[seed] for seed in range(seed_start, seed_end))


def _iter_compressed_lines(path: Path, digest: Any) -> Iterator[bytes]:
    """Yield one checksummed zstd frame and hash the exact compressed bytes."""
    try:
        with path.open("rb") as source:
            header = source.read(18)
            digest.update(header)
            try:
                parameters = zstandard.get_frame_parameters(header)
            except zstandard.ZstdError as exc:
                raise ValueError(f"invalid compressed self-play fragment header: {path}") from exc
            if not parameters.has_checksum:
                raise ValueError(f"compressed self-play fragment has no checksum: {path}")
            decompressor = zstandard.ZstdDecompressor().decompressobj()
            pending = decompressor.decompress(header)
            if decompressor.unused_data:
                raise ValueError(
                    f"invalid compressed self-play fragment: trailing frame data: {path}"
                )
            while True:
                while b"\n" in pending:
                    raw, pending = pending.split(b"\n", 1)
                    yield raw
                chunk = source.read(_READ_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                pending += decompressor.decompress(chunk)
                if decompressor.unused_data:
                    raise ValueError(
                        f"invalid compressed self-play fragment: trailing frame data: {path}"
                    )
            if decompressor.unused_data:
                raise ValueError(
                    f"invalid compressed self-play fragment: trailing frame data: {path}"
                )
            pending += decompressor.flush()
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                yield raw
            if not decompressor.eof:
                raise ValueError(
                    f"invalid compressed self-play fragment: truncated zstd frame: {path}"
                )
            if pending:
                raise ValueError(f"self-play fragment has a truncated final row: {path}")
    except zstandard.ZstdError as exc:
        raise ValueError(f"invalid compressed self-play fragment {path}: {exc}") from exc


def _row_source_identity(row: JointDatasetRow) -> tuple[object, ...]:
    return (
        row.source_revision,
        row.dirty_tree_hash,
        row.search_config_id,
        row.generator_config_id,
        row.source_model_digest,
        row.observation.schema_version,
        row.schema_version,
    )


def _game_identity(row: JointDatasetRow) -> tuple[object, ...]:
    return (
        row.game_id,
        row.world_seed,
        row.map_id,
        row.game_type,
        row.red_composition,
        row.blue_composition,
        row.generation_id,
        row.source_revision,
        row.dirty_tree_hash,
        row.source_model_digest,
        row.search_config_id,
        row.generator_config_id,
    )


def _new_temp(path: Path) -> tuple[int, Path]:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return descriptor, Path(name)


def _provenance_payload(
    *,
    seed_start: int,
    seed_end: int,
    worker_count: int,
    checkpoint_files: tuple[_CheckpointFile, ...],
    fragments: list[dict[str, Any]],
    common_checkpoint: CheckpointRow,
    common_source: tuple[object, ...],
    dataset_digest: str,
    output_digest: str,
    row_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed_range": {"start": seed_start, "end": seed_end},
        "worker_count": worker_count,
        "row_count": row_count,
        "game_count": seed_end - seed_start,
        "dataset_digest": dataset_digest,
        "output_digest": output_digest,
        "common_identity": {
            "generation_id": common_checkpoint.generation_id,
            "source_config_id": common_checkpoint.source_config_id,
            "source_revision": common_source[0],
            "dirty_tree_hash": common_source[1],
            "search_config_id": common_source[2],
            "generator_config_id": common_checkpoint.generator_config_id,
            "parent_model_digest": common_checkpoint.parent_model_digest,
            "parent_generation": common_checkpoint.parent_generation,
            "observation_schema_version": common_checkpoint.observation_schema_version,
            "joint_dataset_schema_version": common_source[6],
        },
        "checkpoints": [
            {
                "worker_id": checkpoint.worker_id,
                "name": checkpoint.name,
                "worker_config_id": checkpoint.worker_config_id,
                "checkpoint_digest": checkpoint.digest,
                "game_count": len(checkpoint.rows),
                "row_count": sum(row.row_count for row in checkpoint.rows),
            }
            for checkpoint in checkpoint_files
        ],
        "fragments": fragments,
    }


def _validate_existing_publication(
    destination: Path, provenance_path: Path, expected_payload: dict[str, Any]
) -> None:
    if destination.is_symlink() or provenance_path.is_symlink():
        raise ValueError("existing publication paths must not be symlinks")
    if not destination.is_file() or not provenance_path.is_file():
        raise ValueError("existing publication is incomplete")
    raw_provenance = provenance_path.read_bytes()
    try:
        existing = json.loads(raw_provenance)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("existing publication provenance is invalid") from exc
    if raw_provenance != canonical_value_bytes(existing):
        raise ValueError("existing publication provenance is not canonical")
    if existing != expected_payload:
        raise ValueError("existing publication provenance does not match worker output")
    if _stream_file_digest(destination) != expected_payload["output_digest"]:
        raise ValueError("existing publication aggregate digest mismatch")


def _publish_new(
    destination: Path,
    provenance_path: Path,
    aggregate_temporary: Path,
    provenance_temporary: Path,
) -> None:
    if destination.exists() or provenance_path.exists():
        raise ValueError("publication appeared while the reconciliation lock was held")
    aggregate_published = False
    try:
        os.replace(aggregate_temporary, destination)
        aggregate_published = True
        fsync_directory(destination.parent)
        os.replace(provenance_temporary, provenance_path)
        fsync_directory(provenance_path.parent)
    except BaseException:
        # The destination did not exist before this operation, so rollback can
        # never remove an unrelated artifact.
        if aggregate_published and not provenance_path.exists():
            destination.unlink(missing_ok=True)
            with suppress(OSError):
                fsync_directory(destination.parent)
        raise


def reconcile_self_play(
    *,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    seed_start: int,
    seed_end: int,
    destination: str | Path,
    provenance_path: str | Path | None = None,
    worker_count: int = WORKER_COUNT,
) -> ReconciliationResult:
    """Validate and atomically publish complete worker output with bounded memory."""
    if seed_start < 0 or seed_end <= seed_start:
        raise ValueError("expected seed range must be non-empty, half-open, and non-negative")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if worker_count > seed_end - seed_start:
        raise ValueError("worker_count cannot exceed the expected game count")
    output = _absolute(output_dir)
    checkpoints = _absolute(checkpoint_dir)
    aggregate = _absolute(destination)
    provenance = _absolute(
        provenance_path if provenance_path is not None else f"{aggregate}.provenance.json"
    )
    _validate_paths(output, checkpoints, aggregate, provenance)

    with _publication_lock(aggregate):
        checkpoint_files, receipts = _load_checkpoints(
            checkpoints,
            seed_start=seed_start,
            seed_end=seed_end,
            worker_count=worker_count,
        )
        aggregate_descriptor, aggregate_path = _new_temp(aggregate)
        aggregate_temporary: Path | None = aggregate_path
        provenance_temporary: Path | None = None
        uniqueness_path: Path | None = None
        decisions: sqlite3.Connection | None = None
        try:
            uniqueness_descriptor, uniqueness_name = tempfile.mkstemp(
                prefix=f".{aggregate.name}.decisions.", suffix=".sqlite", dir=aggregate.parent
            )
            os.close(uniqueness_descriptor)
            uniqueness_path = Path(uniqueness_name)
            decisions = sqlite3.connect(uniqueness_path)
            decisions.execute("PRAGMA cache_size=-1024")
            decisions.execute("PRAGMA temp_store=FILE")
            decisions.execute("PRAGMA journal_mode=OFF")
            decisions.execute("CREATE TABLE decision_ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
            dataset_digest = hashlib.sha256()
            row_count = 0
            fragments: list[dict[str, Any]] = []
            common_source: tuple[object, ...] | None = None
            aggregate_file = os.fdopen(aggregate_descriptor, "w+b")
            aggregate_descriptor = -1
            with aggregate_file:
                compressor = zstandard.ZstdCompressor(level=10, threads=0, write_checksum=True)
                with compressor.stream_writer(aggregate_file, closefd=False) as writer:
                    for receipt in receipts:
                        fragment_name = f"{receipt.world_seed:020d}-{receipt.game_id}.jsonl.zst"
                        fragment = output / f"worker-{receipt.worker_id}" / fragment_name
                        _require_regular_file(
                            fragment, label=f"worker {receipt.worker_id} fragment"
                        )
                        compressed_digest = hashlib.sha256()
                        expected_index = 0
                        game_identity: tuple[object, ...] | None = None
                        for line_number, raw in enumerate(
                            _iter_compressed_lines(fragment, compressed_digest), 1
                        ):
                            if not raw or raw != raw.strip():
                                raise ValueError(
                                    f"invalid self-play fragment row {line_number}: whitespace"
                                )
                            try:
                                row = JointDatasetRow.model_validate_json(raw)
                            except ValueError as exc:
                                raise ValueError(
                                    f"invalid self-play fragment row {line_number}: {exc}"
                                ) from exc
                            canonical = canonical_json_bytes(row)
                            if raw != canonical:
                                raise ValueError(
                                    f"invalid self-play fragment row {line_number}: "
                                    "non-canonical JSON"
                                )
                            if (
                                row.game_id != receipt.game_id
                                or row.world_seed != receipt.world_seed
                            ):
                                raise ValueError("self-play fragment and checkpoint game disagree")
                            expected_game_id = content_digest(
                                {
                                    "generator_config_id": row.generator_config_id,
                                    "world_seed": row.world_seed,
                                    "map_id": row.map_id,
                                    "game_type": row.game_type,
                                    "red_composition": row.red_composition,
                                    "blue_composition": row.blue_composition,
                                }
                            )
                            if row.game_id != expected_game_id:
                                raise ValueError(
                                    "self-play game_id does not match its complete game identity"
                                )
                            if (
                                row.generation_id != receipt.generation_id
                                or row.generator_config_id != receipt.generator_config_id
                                or row.source_model_digest != receipt.parent_model_digest
                                or row.observation.schema_version
                                != receipt.observation_schema_version
                            ):
                                raise ValueError(
                                    "self-play row provenance disagrees with its checkpoint"
                                )
                            source_identity = _row_source_identity(row)
                            if common_source is None:
                                common_source = source_identity
                            elif source_identity != common_source:
                                raise ValueError(
                                    "self-play rows do not share one source/search/schema identity"
                                )
                            identity = _game_identity(row)
                            if game_identity is None:
                                game_identity = identity
                            elif identity != game_identity:
                                raise ValueError("self-play fragment has conflicting game identity")
                            if row.terminal_winner != receipt.winner:
                                raise ValueError(
                                    "self-play checkpoint winner and terminal rows disagree"
                                )
                            if row.decision_index != expected_index:
                                raise ValueError(
                                    f"game {row.game_id!r} decision indexes are not contiguous "
                                    "from zero"
                                )
                            expected_index += 1
                            try:
                                decisions.execute(
                                    "INSERT INTO decision_ids(id) VALUES (?)", (row.decision_id,)
                                )
                            except sqlite3.IntegrityError as exc:
                                raise ValueError(
                                    f"duplicate self-play decision_id {row.decision_id}"
                                ) from exc
                            payload = canonical + b"\n"
                            dataset_digest.update(payload)
                            writer.write(payload)
                            row_count += 1
                        actual_fragment_digest = compressed_digest.hexdigest()
                        if actual_fragment_digest != receipt.fragment_digest:
                            raise ValueError(
                                f"self-play fragment digest disagrees with checkpoint: {fragment}"
                            )
                        if expected_index != receipt.row_count:
                            raise ValueError(
                                f"self-play fragment row count disagrees with checkpoint: {fragment}"
                            )
                        fragments.append(
                            {
                                "worker_id": receipt.worker_id,
                                "world_seed": receipt.world_seed,
                                "game_id": receipt.game_id,
                                "name": f"worker-{receipt.worker_id}/{fragment_name}",
                                "fragment_digest": receipt.fragment_digest,
                                "row_count": receipt.row_count,
                                "winner": receipt.winner,
                            }
                        )
                aggregate_file.flush()
                os.fsync(aggregate_file.fileno())
            decisions.close()
            decisions = None
            if common_source is None or not row_count:
                raise ValueError("self-play reconciliation produced no rows")
            assert aggregate_temporary is not None
            output_digest = _stream_file_digest(aggregate_temporary)
            provenance_payload = _provenance_payload(
                seed_start=seed_start,
                seed_end=seed_end,
                worker_count=worker_count,
                checkpoint_files=checkpoint_files,
                fragments=fragments,
                common_checkpoint=receipts[0],
                common_source=common_source,
                dataset_digest=dataset_digest.hexdigest(),
                output_digest=output_digest,
                row_count=row_count,
            )
            provenance_descriptor, provenance_temporary = _new_temp(provenance)
            with os.fdopen(provenance_descriptor, "wb") as provenance_file:
                provenance_file.write(canonical_value_bytes(provenance_payload))
                provenance_file.flush()
                os.fsync(provenance_file.fileno())

            if aggregate.exists() or provenance.exists():
                _validate_existing_publication(aggregate, provenance, provenance_payload)
            else:
                assert provenance_temporary is not None
                _publish_new(aggregate, provenance, aggregate_temporary, provenance_temporary)
                aggregate_temporary = None
                provenance_temporary = None
            return ReconciliationResult(
                destination=aggregate,
                provenance_path=provenance,
                dataset_digest=dataset_digest.hexdigest(),
                output_digest=output_digest,
                row_count=row_count,
                game_count=seed_end - seed_start,
            )
        finally:
            if decisions is not None:
                decisions.close()
            if aggregate_descriptor >= 0:
                os.close(aggregate_descriptor)
            if aggregate_temporary is not None:
                aggregate_temporary.unlink(missing_ok=True)
            if provenance_temporary is not None:
                provenance_temporary.unlink(missing_ok=True)
            if uniqueness_path is not None:
                uniqueness_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed-start", "--expected-seed-start", required=True, type=int)
    parser.add_argument("--seed-end", "--expected-seed-end", required=True, type=int)
    parser.add_argument(
        "--out", "--destination", required=True, help="destination aggregate .jsonl.zst"
    )
    parser.add_argument(
        "--provenance",
        "--provenance-sidecar",
        help="canonical sidecar path (default: <out>.provenance.json)",
    )
    parser.add_argument("--workers", "--worker-count", type=int, default=WORKER_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.seed_start < 0 or args.seed_end <= args.seed_start:
        parser.error("--seed-end must be greater than non-negative --seed-start")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    try:
        result = reconcile_self_play(
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
            destination=args.out,
            provenance_path=args.provenance,
            worker_count=args.workers,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"self-play reconciliation failed: {exc}", file=sys.stderr)
        return OPERATIONAL_FAILURE_EXIT_STATUS
    print(
        f"Reconciled {result.game_count} games / {result.row_count} decisions into "
        f"{result.destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
