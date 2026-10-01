"""Receipt-bound, per-game index for native policy and value records.

The source receipt is an inventory attested by its caller.  Creating one validates
its explicitly named files, but does not certify that gameplay reached a terminal
state.  The native recorder remains responsible for completion before publication.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, cast
from uuid import uuid4

import zstandard
from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, model_validator

from automata.models.contracts import canonical_json_bytes
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.training.io import atomic_write_bytes, fsync_directory
from automata.training.native_batches import (
    NativePolicyTrainingBatch,
    NativeValueTrainingBatch,
    collate_native_policy_records,
    collate_native_value_records,
)
from automata.training.native_dataset import (
    NativeDatasetRecord,
    NativeGameIdentity,
    PolicyDatasetRecord,
    ValueDatasetRecord,
    iter_native_game_records,
)

INDEX_SCHEMA_VERSION: Literal[1] = 1
DEFAULT_CHUNK_SIZE = 32
_MANIFEST_NAME = "manifest.json"
_CHECKPOINT_NAME = "checkpoint.json"
_OWNERSHIP_MARKER_NAME = ".native-index-owner"
_OWNERSHIP_MARKER_CONTENT = b"automata-native-index-v1\n"
_READ_SIZE = 128 * 1024
Head = Literal["POLICY", "VALUE"]


def _safe_logical_name(value: str) -> str:
    if type(value) is not str or not value:
        raise ValueError("source logical name must be a nonempty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("source logical name must be a normalized relative POSIX path")
    if not (value.endswith(".jsonl") or value.endswith(".jsonl.zst")):
        raise ValueError("source logical name must identify a native JSONL file")
    return value


class NativeGameSourceReceipt(BaseModel):
    """Exact portable inventory entry for one caller-attested native game file."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    logical_name: str
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_size: StrictInt = Field(gt=0)
    game_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: StrictInt = Field(gt=0)
    policy_row_count: StrictInt = Field(ge=0)
    value_row_count: StrictInt = Field(ge=0)
    boundary_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _valid_entry(self) -> NativeGameSourceReceipt:
        _safe_logical_name(self.logical_name)
        if self.row_count != self.policy_row_count + self.value_row_count:
            raise ValueError("source receipt row count must equal its head counts")
        if (self.value_row_count == 0) != (self.boundary_count == 0):
            raise ValueError("source receipt boundary count must agree with value rows")
        if self.boundary_count > self.value_row_count:
            raise ValueError("source receipt boundary count cannot exceed value rows")
        return self


class NativeDatasetSourceReceipt(BaseModel):
    """Canonical ordered inventory whose digest is independent of its physical root."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    games: tuple[NativeGameSourceReceipt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_inventory(self) -> NativeDatasetSourceReceipt:
        logical_names = tuple(game.logical_name for game in self.games)
        game_ids = tuple(game.game_id for game in self.games)
        if len(set(logical_names)) != len(logical_names):
            raise ValueError("source receipt logical names must be unique")
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("source receipt game IDs must be unique")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class NativeIndexChunkMetadata(BaseModel):
    """Exact binding and offsets for one homogeneous compressed JSONL chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    head: Head
    path: str = Field(pattern=r"^chunks/(policy|value)/[0-9]{8}/[0-9]{8}\.jsonl\.zst$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_size: StrictInt = Field(gt=0)
    uncompressed_size: StrictInt = Field(gt=0)
    game_ordinal: StrictInt = Field(ge=0)
    chunk_ordinal: StrictInt = Field(ge=0)
    global_head_start: StrictInt = Field(ge=0)
    game_head_start: StrictInt = Field(ge=0)
    row_count: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def _valid_path(self) -> NativeIndexChunkMetadata:
        head_path = self.head.lower()
        expected = f"chunks/{head_path}/{self.game_ordinal:08d}/{self.chunk_ordinal:08d}.jsonl.zst"
        if self.path != expected:
            raise ValueError("native index chunk path does not match its head and ordinals")
        return self


class NativeIndexedGameMetadata(NativeGameIdentity):
    """Full game identity, receipt counts, digest, offsets, and head chunk inventory."""

    source_logical_name: str
    game_ordinal: StrictInt = Field(ge=0)
    row_count: StrictInt = Field(gt=0)
    policy_row_count: StrictInt = Field(ge=0)
    value_row_count: StrictInt = Field(ge=0)
    boundary_count: StrictInt = Field(ge=0)
    semantic_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_global_start: StrictInt = Field(ge=0)
    value_global_start: StrictInt = Field(ge=0)
    policy_chunks: tuple[NativeIndexChunkMetadata, ...]
    value_chunks: tuple[NativeIndexChunkMetadata, ...]

    @model_validator(mode="after")
    def _valid_counts(self) -> NativeIndexedGameMetadata:
        _safe_logical_name(self.source_logical_name)
        if self.row_count != self.policy_row_count + self.value_row_count:
            raise ValueError("indexed game row count must equal its head counts")
        if (self.value_row_count == 0) != (self.boundary_count == 0):
            raise ValueError("indexed game boundary count must agree with value rows")
        return self

    @property
    def identity(self) -> NativeGameIdentity:
        """Return only the canonical native game identity fields."""
        return NativeGameIdentity.model_validate(
            self.model_dump(include=set(NativeGameIdentity.model_fields), mode="python"),
            strict=True,
        )


def _validate_chunk_ranges(
    chunks: tuple[NativeIndexChunkMetadata, ...],
    *,
    head: Head,
    game: NativeIndexedGameMetadata,
    global_start: int,
    expected_count: int,
    chunk_size: int,
    all_paths: set[str],
) -> None:
    if bool(chunks) != (expected_count > 0):
        raise ValueError("indexed head chunks must exist exactly when that head has rows")
    game_start = 0
    current_global = global_start
    for ordinal, chunk in enumerate(chunks):
        if chunk.path in all_paths:
            raise ValueError("native index chunk paths must be unique")
        all_paths.add(chunk.path)
        expected_chunk_count = min(chunk_size, expected_count - game_start)
        if (
            chunk.head != head
            or chunk.game_ordinal != game.game_ordinal
            or chunk.chunk_ordinal != ordinal
            or chunk.game_head_start != game_start
            or chunk.global_head_start != current_global
            or chunk.row_count != expected_chunk_count
            or chunk.row_count > chunk_size
        ):
            raise ValueError("native index chunk ranges do not exactly cover their head")
        game_start += chunk.row_count
        current_global += chunk.row_count
    if game_start != expected_count:
        raise ValueError("native index chunks do not cover every head row")


class NativeIndexedDatasetManifest(BaseModel):
    """Canonical identity and complete inventory of a native indexed dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = INDEX_SCHEMA_VERSION
    source_receipt: NativeDatasetSourceReceipt
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_tensor_schema_id: str = Field(min_length=1)
    decision_tensor_schema_version: StrictInt = Field(gt=0)
    decision_tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stable_value_tensor_schema_id: str = Field(min_length=1)
    stable_value_tensor_schema_version: StrictInt = Field(gt=0)
    stable_value_tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_size: StrictInt = Field(gt=0)
    semantic_dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: StrictInt = Field(gt=0)
    policy_row_count: StrictInt = Field(ge=0)
    value_row_count: StrictInt = Field(ge=0)
    boundary_count: StrictInt = Field(ge=0)
    game_count: StrictInt = Field(gt=0)
    games: tuple[NativeIndexedGameMetadata, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_manifest(self) -> NativeIndexedDatasetManifest:
        if self.source_digest != self.source_receipt.digest:
            raise ValueError("manifest source digest does not match its canonical receipt")
        if self.game_count != len(self.games) or self.game_count != len(self.source_receipt.games):
            raise ValueError("manifest game count does not match its inventories")
        totals = (
            sum(game.row_count for game in self.games),
            sum(game.policy_row_count for game in self.games),
            sum(game.value_row_count for game in self.games),
            sum(game.boundary_count for game in self.games),
        )
        if totals != (
            self.row_count,
            self.policy_row_count,
            self.value_row_count,
            self.boundary_count,
        ):
            raise ValueError("manifest totals do not match per-game metadata")
        if self.row_count != self.policy_row_count + self.value_row_count:
            raise ValueError("manifest row total does not equal its head totals")

        policy_start = 0
        value_start = 0
        all_paths: set[str] = set()
        for ordinal, (game, receipt) in enumerate(
            zip(self.games, self.source_receipt.games, strict=True)
        ):
            if game.game_ordinal != ordinal:
                raise ValueError("manifest game ordinals must be contiguous in receipt order")
            if (
                game.game_id != receipt.game_id
                or game.source_logical_name != receipt.logical_name
                or game.row_count != receipt.row_count
                or game.policy_row_count != receipt.policy_row_count
                or game.value_row_count != receipt.value_row_count
                or game.boundary_count != receipt.boundary_count
            ):
                raise ValueError("indexed game metadata does not match its source receipt")
            if game.policy_global_start != policy_start or game.value_global_start != value_start:
                raise ValueError("manifest per-head game offsets must be contiguous")
            _validate_chunk_ranges(
                game.policy_chunks,
                head="POLICY",
                game=game,
                global_start=policy_start,
                expected_count=game.policy_row_count,
                chunk_size=self.chunk_size,
                all_paths=all_paths,
            )
            _validate_chunk_ranges(
                game.value_chunks,
                head="VALUE",
                game=game,
                global_start=value_start,
                expected_count=game.value_row_count,
                chunk_size=self.chunk_size,
                all_paths=all_paths,
            )
            policy_start += game.policy_row_count
            value_start += game.value_row_count
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


class _ChunkWrapper(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sample_kind: Head
    global_head_index: StrictInt = Field(ge=0)
    game_head_index: StrictInt = Field(ge=0)
    record: NativeDatasetRecord

    @model_validator(mode="after")
    def _matching_kind(self) -> _ChunkWrapper:
        if self.record.sample_kind != self.sample_kind:
            raise ValueError("chunk wrapper and native record sample kinds disagree")
        return self


_CHUNK_WRAPPER_ADAPTER: TypeAdapter[_ChunkWrapper] = TypeAdapter(_ChunkWrapper)


class _StagingCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    source_receipt: NativeDatasetSourceReceipt
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_tensor_schema_id: str
    decision_tensor_schema_version: StrictInt
    decision_tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stable_value_tensor_schema_id: str
    stable_value_tensor_schema_version: StrictInt
    stable_value_tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_size: StrictInt = Field(gt=0)
    completed_games: tuple[NativeIndexedGameMetadata, ...]

    @model_validator(mode="after")
    def _valid_checkpoint(self) -> _StagingCheckpoint:
        if self.source_digest != self.source_receipt.digest:
            raise ValueError("staging source digest does not match its receipt")
        if len(self.completed_games) > len(self.source_receipt.games):
            raise ValueError("staging has more games than its source receipt")
        receipt_prefix = self.source_receipt.games[: len(self.completed_games)]
        policy_start = 0
        value_start = 0
        all_paths: set[str] = set()
        for ordinal, (game, source) in enumerate(
            zip(self.completed_games, receipt_prefix, strict=True)
        ):
            if (
                game.game_ordinal != ordinal
                or game.game_id != source.game_id
                or game.source_logical_name != source.logical_name
                or game.row_count != source.row_count
                or game.policy_row_count != source.policy_row_count
                or game.value_row_count != source.value_row_count
                or game.boundary_count != source.boundary_count
            ):
                raise ValueError("staging completed games do not match their receipt prefix")
            if game.policy_global_start != policy_start or game.value_global_start != value_start:
                raise ValueError("staging completed game offsets must be contiguous")
            _validate_chunk_ranges(
                game.policy_chunks,
                head="POLICY",
                game=game,
                global_start=policy_start,
                expected_count=game.policy_row_count,
                chunk_size=self.chunk_size,
                all_paths=all_paths,
            )
            _validate_chunk_ranges(
                game.value_chunks,
                head="VALUE",
                game=game,
                global_start=value_start,
                expected_count=game.value_row_count,
                chunk_size=self.chunk_size,
                all_paths=all_paths,
            )
            policy_start += game.policy_row_count
            value_start += game.value_row_count
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


class _CheckpointEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    checkpoint: _StagingCheckpoint
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _valid_digest(self) -> _CheckpointEnvelope:
        expected = hashlib.sha256(self.checkpoint.canonical_bytes()).hexdigest()
        if self.checkpoint_sha256 != expected:
            raise ValueError("staging checkpoint digest mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_READ_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _safe_root(root: Path, *, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root must be a regular directory")


def _has_ownership_marker(root: Path) -> bool:
    marker = root / _OWNERSHIP_MARKER_NAME
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        return marker.read_bytes() == _OWNERSHIP_MARKER_CONTENT
    except OSError:
        return False


def _require_owned_directory(root: Path, *, label: str) -> None:
    _safe_root(root, label=label)
    if not _has_ownership_marker(root):
        raise ValueError(f"{label} must have an exact native index ownership marker")


def _require_replaceable_directory(path: Path, *, label: str) -> None:
    """Allow destructive use only for absent, empty, or deliberately owned directories."""
    if path.is_symlink():
        raise ValueError(f"{label} must be absent, empty, or an explicitly owned native index")
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"{label} must be absent, empty, or an explicitly owned native index")
    try:
        is_empty = next(path.iterdir(), None) is None
    except OSError as exc:
        raise ValueError(f"{label} could not be validated for native index ownership") from exc
    if not is_empty and not _has_ownership_marker(path):
        raise ValueError(f"{label} must be absent, empty, or an explicitly owned native index")


def _safe_relative_file(root: Path, relative: str, *, label: str) -> Path:
    _safe_root(root, label=label)
    logical = _safe_logical_name(relative) if label == "source" else relative
    pure = PurePosixPath(logical)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} path must remain inside its root")
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks")
    if not current.is_file():
        raise ValueError(f"{label} file is missing: {relative}")
    return current


def _safe_chunk_file(root: Path, metadata: NativeIndexChunkMetadata) -> Path:
    return _safe_relative_file(root, metadata.path, label="native index chunk")


def _source_path(root: Path, receipt: NativeGameSourceReceipt) -> Path:
    return _safe_relative_file(root, receipt.logical_name, label="source")


def _source_counts(path: Path) -> tuple[NativeGameIdentity, int, int, int, int]:
    game: NativeGameIdentity | None = None
    rows = policy = value = boundaries = 0
    last_boundary: int | None = None
    for record in iter_native_game_records(path):
        if game is None:
            game = record.game
        rows += 1
        if isinstance(record, PolicyDatasetRecord):
            policy += 1
        else:
            value += 1
            if record.boundary.boundary_index != last_boundary:
                boundaries += 1
                last_boundary = record.boundary.boundary_index
    if game is None:  # iter_native_game_records normally raises first
        raise ValueError("native source game is empty")
    return game, rows, policy, value, boundaries


def load_native_source_receipt(path: str | Path) -> NativeDatasetSourceReceipt:
    """Load a strict canonical source receipt; non-canonical JSON is rejected."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("native source receipt must be a regular file")
    payload = source.read_bytes()
    try:
        receipt = NativeDatasetSourceReceipt.model_validate_json(payload, strict=True)
    except ValueError as exc:
        raise ValueError(f"invalid native source receipt: {exc}") from exc
    if payload != receipt.canonical_bytes():
        raise ValueError("native source receipt is not canonical JSON")
    return receipt


def create_native_source_receipt(
    source_root: str | Path, logical_names: Sequence[str]
) -> NativeDatasetSourceReceipt:
    """Inventory only explicitly named files; this caller attestation is not gameplay certification."""
    if isinstance(logical_names, (str, bytes)) or not isinstance(logical_names, Sequence):
        raise TypeError("logical_names must be an explicit sequence of portable names")
    names = tuple(_safe_logical_name(name) for name in logical_names)
    if not names:
        raise ValueError("logical_names must contain at least one source game")
    if len(set(names)) != len(names):
        raise ValueError("logical_names must be unique")
    root = Path(source_root)
    _safe_root(root, label="source")
    games: list[NativeGameSourceReceipt] = []
    for logical_name in names:
        path = _safe_relative_file(root, logical_name, label="source")
        before_sha, before_size = _file_sha256(path)
        identity, rows, policy, value, boundaries = _source_counts(path)
        after = _file_sha256(path)
        if after != (before_sha, before_size):
            raise ValueError(f"native source changed while receipt was created: {logical_name}")
        games.append(
            NativeGameSourceReceipt(
                logical_name=logical_name,
                file_sha256=before_sha,
                file_size=before_size,
                game_id=identity.game_id,
                row_count=rows,
                policy_row_count=policy,
                value_row_count=value,
                boundary_count=boundaries,
            )
        )
    return NativeDatasetSourceReceipt(games=tuple(games))


def _validate_source_files(root: Path, receipt: NativeDatasetSourceReceipt) -> None:
    for game in receipt.games:
        path = _source_path(root, game)
        if _file_sha256(path) != (game.file_sha256, game.file_size):
            raise ValueError(
                f"native source hash or size does not match receipt: {game.logical_name}"
            )


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def _validate_cache_separation(
    source_root: Path,
    source_receipt_path: Path,
    destination: Path,
    receipt: NativeDatasetSourceReceipt,
) -> None:
    """Reject cache/staging paths that could replace an input during cleanup/publication."""
    stage = destination.parent / f".{destination.name}.staging"
    lock = destination.parent / f".{destination.name}.lock"
    protected = [source_receipt_path]
    protected.extend(_source_path(source_root, game) for game in receipt.games)
    for managed in (destination, stage):
        if any(_paths_overlap(managed, item) for item in protected):
            raise ValueError("native index cache/staging paths overlap a source or receipt")
    if any(lock.resolve(strict=False) == item.resolve(strict=False) for item in protected):
        raise ValueError("native index cache lock overlaps a source or receipt")


def _decompress_chunk(path: Path, metadata: NativeIndexChunkMetadata) -> bytes:
    # Hold at most one declared compressed chunk and its bounded output. Verify
    # the exact bytes that will be decoded, avoiding a separate hash/read race.
    with path.open("rb") as source:
        compressed = source.read(metadata.file_size + 1)
    if (
        len(compressed) != metadata.file_size
        or hashlib.sha256(compressed).hexdigest() != metadata.sha256
    ):
        raise ValueError("native index chunk exact file identity does not match its metadata")
    try:
        declared_size = zstandard.frame_content_size(compressed)
        # A known frame size takes precedence over max_output_size in zstandard;
        # reject a mismatched header before it can allocate its declared output.
        if declared_size >= 0 and declared_size != metadata.uncompressed_size:
            raise ValueError("native index chunk frame size does not match its metadata")
        payload = zstandard.ZstdDecompressor().decompress(
            compressed,
            max_output_size=metadata.uncompressed_size,
            read_across_frames=False,
            allow_extra_data=False,
        )
    except zstandard.ZstdError as exc:
        raise ValueError(
            "native index chunk is not a valid bounded single compressed frame"
        ) from exc
    if len(payload) != metadata.uncompressed_size:
        raise ValueError("native index chunk uncompressed size does not match its metadata")
    if not payload.endswith(b"\n"):
        raise ValueError("native index chunk has a truncated final JSONL wrapper")
    return payload


def _load_chunk_records(
    root: Path,
    metadata: NativeIndexChunkMetadata,
    *,
    game: NativeIndexedGameMetadata,
) -> tuple[NativeDatasetRecord, ...]:
    path = _safe_chunk_file(root, metadata)
    payload = _decompress_chunk(path, metadata)
    lines = payload.splitlines()
    if len(lines) != metadata.row_count:
        raise ValueError("native index chunk wrapper count does not match its metadata")
    records: list[NativeDatasetRecord] = []
    for offset, line in enumerate(lines):
        if not line or line != line.strip():
            raise ValueError("native index chunk contains a blank or non-canonical wrapper")
        try:
            wrapper = _CHUNK_WRAPPER_ADAPTER.validate_json(line, strict=True)
        except ValueError as exc:
            raise ValueError(f"native index chunk wrapper is malformed: {exc}") from exc
        if line != canonical_json_bytes(wrapper):
            raise ValueError("native index chunk wrapper is not canonical JSON")
        if (
            wrapper.sample_kind != metadata.head
            or wrapper.global_head_index != metadata.global_head_start + offset
            or wrapper.game_head_index != metadata.game_head_start + offset
            or wrapper.record.game != game.identity
            or wrapper.record.game.game_id != game.game_id
        ):
            raise ValueError("native index chunk wrapper identity or offsets are invalid")
        records.append(wrapper.record)
    return tuple(records)


def _collate_policy(
    records: Sequence[PolicyDatasetRecord], count: int
) -> NativePolicyTrainingBatch:
    return collate_native_policy_records(
        records,
        game_policy_counts={record.game.game_id: count for record in records},
        schema=TensorFeatureSchema.current(),
    )


def _collate_value(records: Sequence[ValueDatasetRecord], count: int) -> NativeValueTrainingBatch:
    return collate_native_value_records(
        records,
        game_value_counts={record.game.game_id: count for record in records},
        schema=StableValueTensorSchema.current(),
    )


def _write_chunk(
    root: Path,
    records: Sequence[NativeDatasetRecord],
    *,
    head: Head,
    game_ordinal: int,
    chunk_ordinal: int,
    global_head_start: int,
    game_head_start: int,
    full_game_head_count: int,
) -> NativeIndexChunkMetadata:
    if not records or len(records) > full_game_head_count:
        raise ValueError("native index chunk requires bounded records from one full game")
    if head == "POLICY":
        typed_policy = tuple(cast(PolicyDatasetRecord, record) for record in records)
        if any(not isinstance(record, PolicyDatasetRecord) for record in typed_policy):
            raise ValueError("policy chunk cannot contain value records")
        _collate_policy(typed_policy, full_game_head_count)
    else:
        typed_value = tuple(cast(ValueDatasetRecord, record) for record in records)
        if any(not isinstance(record, ValueDatasetRecord) for record in typed_value):
            raise ValueError("value chunk cannot contain policy records")
        _collate_value(typed_value, full_game_head_count)

    relative = f"chunks/{head.lower()}/{game_ordinal:08d}/{chunk_ordinal:08d}.jsonl.zst"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    uncompressed_size = 0
    with path.open("wb") as output:
        writer = zstandard.ZstdCompressor(level=3, threads=0, write_checksum=True).stream_writer(
            output, closefd=False
        )
        try:
            for offset, record in enumerate(records):
                wrapper = _ChunkWrapper(
                    sample_kind=head,
                    global_head_index=global_head_start + offset,
                    game_head_index=game_head_start + offset,
                    record=record,
                )
                line = canonical_json_bytes(wrapper) + b"\n"
                writer.write(line)
                uncompressed_size += len(line)
        finally:
            writer.close()
        output.flush()
        os.fsync(output.fileno())
    sha256, file_size = _file_sha256(path)
    return NativeIndexChunkMetadata(
        head=head,
        path=relative,
        sha256=sha256,
        file_size=file_size,
        uncompressed_size=uncompressed_size,
        game_ordinal=game_ordinal,
        chunk_ordinal=chunk_ordinal,
        global_head_start=global_head_start,
        game_head_start=game_head_start,
        row_count=len(records),
    )


def _iter_indexed_game_records(
    root: Path, game: NativeIndexedGameMetadata
) -> Iterator[NativeDatasetRecord]:
    def head_records(
        chunks: tuple[NativeIndexChunkMetadata, ...], expected_count: int
    ) -> Iterator[NativeDatasetRecord]:
        count = 0
        for chunk in chunks:
            loaded = _load_chunk_records(root, chunk, game=game)
            count += len(loaded)
            yield from loaded
        if count != expected_count:
            raise ValueError("native index head record count does not match game metadata")

    policy = iter(head_records(game.policy_chunks, game.policy_row_count))
    value = iter(head_records(game.value_chunks, game.value_row_count))
    policy_next = next(policy, None)
    value_next = next(value, None)
    for sample_index in range(game.row_count):
        candidates = tuple(
            record
            for record in (policy_next, value_next)
            if record is not None and record.sample_index == sample_index
        )
        if len(candidates) != 1:
            raise ValueError("native index cannot reconstruct a unique contiguous sample order")
        record = candidates[0]
        yield record
        if record is policy_next:
            policy_next = next(policy, None)
        else:
            value_next = next(value, None)
    if policy_next is not None or value_next is not None:
        raise ValueError("native index contains records beyond the declared game range")


def _manifest_from_games(
    receipt: NativeDatasetSourceReceipt,
    games: tuple[NativeIndexedGameMetadata, ...],
    *,
    chunk_size: int,
    semantic_dataset_digest: str,
    decision_schema: TensorFeatureSchema,
    value_schema: StableValueTensorSchema,
) -> NativeIndexedDatasetManifest:
    return NativeIndexedDatasetManifest(
        source_receipt=receipt,
        source_digest=receipt.digest,
        decision_tensor_schema_id=decision_schema.schema_id,
        decision_tensor_schema_version=decision_schema.schema_version,
        decision_tensor_schema_digest=decision_schema.digest,
        stable_value_tensor_schema_id=value_schema.schema_id,
        stable_value_tensor_schema_version=value_schema.schema_version,
        stable_value_tensor_schema_digest=value_schema.digest,
        chunk_size=chunk_size,
        semantic_dataset_digest=semantic_dataset_digest,
        row_count=sum(game.row_count for game in games),
        policy_row_count=sum(game.policy_row_count for game in games),
        value_row_count=sum(game.value_row_count for game in games),
        boundary_count=sum(game.boundary_count for game in games),
        game_count=len(games),
        games=games,
    )


class IndexedNativeDataset:
    """Opened native index with chunk-local checks and source-backed game validation."""

    def __init__(
        self,
        cache_dir: Path,
        source_root: Path,
        manifest: NativeIndexedDatasetManifest,
    ) -> None:
        self.cache_dir = cache_dir
        self.source_root = source_root
        self.manifest = manifest
        self._games_by_id: Mapping[str, NativeIndexedGameMetadata] = MappingProxyType(
            {game.game_id: game for game in manifest.games}
        )
        self._sources_by_id: Mapping[str, NativeGameSourceReceipt] = MappingProxyType(
            {game.game_id: game for game in manifest.source_receipt.games}
        )

    @property
    def digest(self) -> str:
        return self.manifest.semantic_dataset_digest

    @property
    def source_digest(self) -> str:
        return self.manifest.source_digest

    @property
    def game_ids(self) -> tuple[str, ...]:
        return tuple(game.game_id for game in self.manifest.games)

    def game(self, game_id: str) -> NativeIndexedGameMetadata:
        try:
            return self._games_by_id[game_id]
        except KeyError as exc:
            raise KeyError(f"unknown indexed native game: {game_id!r}") from exc

    def _validate_schemas(self) -> None:
        decision = TensorFeatureSchema.current()
        value = StableValueTensorSchema.current()
        if (decision.schema_id, decision.schema_version, decision.digest) != (
            self.manifest.decision_tensor_schema_id,
            self.manifest.decision_tensor_schema_version,
            self.manifest.decision_tensor_schema_digest,
        ) or (value.schema_id, value.schema_version, value.digest) != (
            self.manifest.stable_value_tensor_schema_id,
            self.manifest.stable_value_tensor_schema_version,
            self.manifest.stable_value_tensor_schema_digest,
        ):
            raise ValueError("current native tensor schemas do not match the indexed dataset")

    def iter_policy_training_chunks(self, game_id: str) -> Iterator[NativePolicyTrainingBatch]:
        self._validate_schemas()
        game = self.game(game_id)
        for chunk in game.policy_chunks:
            records = _load_chunk_records(self.cache_dir, chunk, game=game)
            typed = tuple(cast(PolicyDatasetRecord, record) for record in records)
            if any(not isinstance(record, PolicyDatasetRecord) for record in typed):
                raise ValueError("indexed policy chunk contains a non-policy record")
            yield _collate_policy(typed, game.policy_row_count)

    def iter_value_training_chunks(self, game_id: str) -> Iterator[NativeValueTrainingBatch]:
        self._validate_schemas()
        game = self.game(game_id)
        for chunk in game.value_chunks:
            records = _load_chunk_records(self.cache_dir, chunk, game=game)
            typed = tuple(cast(ValueDatasetRecord, record) for record in records)
            if any(not isinstance(record, ValueDatasetRecord) for record in typed):
                raise ValueError("indexed value chunk contains a non-value record")
            yield _collate_value(typed, game.value_row_count)

    def validate_game(self, game_id: str) -> NativeIndexedGameMetadata:
        """Reconstruct source order and compare every canonical row with its exact source."""
        self._validate_schemas()
        game = self.game(game_id)
        source_receipt = self._sources_by_id[game_id]
        source_path = _source_path(self.source_root, source_receipt)
        if _file_sha256(source_path) != (
            source_receipt.file_sha256,
            source_receipt.file_size,
        ):
            raise ValueError("native source changed since its receipt was issued")
        semantic = hashlib.sha256()
        count = policy_count = value_count = boundary_count = 0
        last_boundary: int | None = None
        source_records = iter_native_game_records(source_path)
        indexed_records = _iter_indexed_game_records(self.cache_dir, game)
        sentinel = object()
        for source_record, indexed_record in zip_longest(
            source_records, indexed_records, fillvalue=sentinel
        ):
            if source_record is sentinel or indexed_record is sentinel:
                raise ValueError("indexed native game and source have different row counts")
            source_typed = cast(NativeDatasetRecord, source_record)
            indexed_typed = cast(NativeDatasetRecord, indexed_record)
            source_bytes = canonical_json_bytes(source_typed)
            if canonical_json_bytes(indexed_typed) != source_bytes:
                raise ValueError("indexed native game record does not match its exact source")
            semantic.update(source_bytes + b"\n")
            count += 1
            if isinstance(source_typed, PolicyDatasetRecord):
                policy_count += 1
            else:
                value_count += 1
                if source_typed.boundary.boundary_index != last_boundary:
                    boundary_count += 1
                    last_boundary = source_typed.boundary.boundary_index
        if _file_sha256(source_path) != (
            source_receipt.file_sha256,
            source_receipt.file_size,
        ):
            raise ValueError("native source changed while its indexed game was validated")
        if (
            count,
            policy_count,
            value_count,
            boundary_count,
            semantic.hexdigest(),
        ) != (
            game.row_count,
            game.policy_row_count,
            game.value_row_count,
            game.boundary_count,
            game.semantic_digest,
        ):
            raise ValueError("indexed native game count or semantic digest is invalid")
        return game

    def head_game_mask(self, head: Head, game_ids: Sequence[str] | None = None) -> tuple[bool, ...]:
        if head not in {"POLICY", "VALUE"}:
            raise ValueError("native index head must be POLICY or VALUE")
        ids = self.game_ids if game_ids is None else tuple(game_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("native index game IDs must be unique when selecting a head")
        games = tuple(self.game(game_id) for game_id in ids)
        attribute = "policy_row_count" if head == "POLICY" else "value_row_count"
        return tuple(getattr(game, attribute) > 0 for game in games)

    def head_game_count(self, head: Head, game_ids: Sequence[str] | None = None) -> int:
        return sum(self.head_game_mask(head, game_ids))


def _load_manifest(root: Path) -> NativeIndexedDatasetManifest:
    _require_owned_directory(root, label="native index")
    manifest_path = root / _MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("native index manifest must be a regular file")
    payload = manifest_path.read_bytes()
    manifest = NativeIndexedDatasetManifest.model_validate_json(payload, strict=True)
    if payload != manifest.canonical_bytes():
        raise ValueError("native index manifest is not canonical JSON")
    for game in manifest.games:
        for chunk in (*game.policy_chunks, *game.value_chunks):
            path = _safe_chunk_file(root, chunk)
            if _file_sha256(path) != (chunk.sha256, chunk.file_size):
                raise ValueError("native index chunk does not match its exact manifest identity")
    return manifest


def _compatible_manifest(
    root: Path,
    source_root: Path,
    receipt: NativeDatasetSourceReceipt,
    *,
    chunk_size: int,
    decision_schema: TensorFeatureSchema,
    value_schema: StableValueTensorSchema,
) -> NativeIndexedDatasetManifest | None:
    try:
        manifest = _load_manifest(root)
    except (OSError, ValueError):
        return None
    if (
        manifest.source_receipt != receipt
        or manifest.source_digest != receipt.digest
        or manifest.chunk_size != chunk_size
        or (
            manifest.decision_tensor_schema_id,
            manifest.decision_tensor_schema_version,
            manifest.decision_tensor_schema_digest,
        )
        != (decision_schema.schema_id, decision_schema.schema_version, decision_schema.digest)
        or (
            manifest.stable_value_tensor_schema_id,
            manifest.stable_value_tensor_schema_version,
            manifest.stable_value_tensor_schema_digest,
        )
        != (value_schema.schema_id, value_schema.schema_version, value_schema.digest)
    ):
        return None
    # Exact source hashes bind the receipt, while source-backed reconstruction
    # prevents a self-consistent rewrite of both chunk and manifest metadata.
    try:
        if _semantic_dataset_digest(root, manifest.games) != manifest.semantic_dataset_digest:
            return None
        dataset = IndexedNativeDataset(root, source_root, manifest)
        for game_id in dataset.game_ids:
            dataset.validate_game(game_id)
            for _ in dataset.iter_policy_training_chunks(game_id):
                pass
            for _ in dataset.iter_value_training_chunks(game_id):
                pass
    except (OSError, ValueError):
        return None
    return manifest


def _checkpoint_for(
    receipt: NativeDatasetSourceReceipt,
    *,
    chunk_size: int,
    decision_schema: TensorFeatureSchema,
    value_schema: StableValueTensorSchema,
    completed_games: tuple[NativeIndexedGameMetadata, ...] = (),
) -> _StagingCheckpoint:
    return _StagingCheckpoint(
        source_receipt=receipt,
        source_digest=receipt.digest,
        decision_tensor_schema_id=decision_schema.schema_id,
        decision_tensor_schema_version=decision_schema.schema_version,
        decision_tensor_schema_digest=decision_schema.digest,
        stable_value_tensor_schema_id=value_schema.schema_id,
        stable_value_tensor_schema_version=value_schema.schema_version,
        stable_value_tensor_schema_digest=value_schema.digest,
        chunk_size=chunk_size,
        completed_games=completed_games,
    )


def _write_checkpoint(stage: Path, checkpoint: _StagingCheckpoint) -> None:
    payload = checkpoint.canonical_bytes()
    envelope = _CheckpointEnvelope(
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
    )
    atomic_write_bytes(stage / _CHECKPOINT_NAME, envelope.canonical_bytes())


def _discard_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _prepare_new_stage(
    stage: Path,
    receipt: NativeDatasetSourceReceipt,
    *,
    chunk_size: int,
    decision_schema: TensorFeatureSchema,
    value_schema: StableValueTensorSchema,
) -> _StagingCheckpoint:
    _require_replaceable_directory(stage, label="native index staging path")
    _discard_path(stage)
    stage.mkdir()
    atomic_write_bytes(stage / _OWNERSHIP_MARKER_NAME, _OWNERSHIP_MARKER_CONTENT)
    fsync_directory(stage)
    (stage / "chunks" / "policy").mkdir(parents=True)
    (stage / "chunks" / "value").mkdir(parents=True)
    checkpoint = _checkpoint_for(
        receipt,
        chunk_size=chunk_size,
        decision_schema=decision_schema,
        value_schema=value_schema,
    )
    _write_checkpoint(stage, checkpoint)
    fsync_directory(stage / "chunks" / "policy")
    fsync_directory(stage / "chunks" / "value")
    fsync_directory(stage / "chunks")
    fsync_directory(stage)
    return checkpoint


def _load_checkpoint(
    stage: Path,
    receipt: NativeDatasetSourceReceipt,
    *,
    chunk_size: int,
    decision_schema: TensorFeatureSchema,
    value_schema: StableValueTensorSchema,
) -> _StagingCheckpoint:
    _require_owned_directory(stage, label="native index staging")
    for path in (stage / "chunks", stage / "chunks" / "policy", stage / "chunks" / "value"):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("native index staging chunk roots must be regular directories")
    payload = (stage / _CHECKPOINT_NAME).read_bytes()
    envelope = _CheckpointEnvelope.model_validate_json(payload, strict=True)
    if payload != envelope.canonical_bytes():
        raise ValueError("native index staging checkpoint is not canonical JSON")
    expected = _checkpoint_for(
        receipt,
        chunk_size=chunk_size,
        decision_schema=decision_schema,
        value_schema=value_schema,
        completed_games=envelope.checkpoint.completed_games,
    )
    if envelope.checkpoint != expected:
        raise ValueError("native index staging identity is stale or incompatible")
    # Fully validate completed chunks, vectorization, and semantic digests before reuse.
    for game in expected.completed_games:
        for chunk in game.policy_chunks:
            records = _load_chunk_records(stage, chunk, game=game)
            _collate_policy(
                tuple(cast(PolicyDatasetRecord, record) for record in records),
                game.policy_row_count,
            )
        for chunk in game.value_chunks:
            records = _load_chunk_records(stage, chunk, game=game)
            _collate_value(
                tuple(cast(ValueDatasetRecord, record) for record in records),
                game.value_row_count,
            )
        digest = hashlib.sha256()
        count = 0
        for record in _iter_indexed_game_records(stage, game):
            digest.update(canonical_json_bytes(record) + b"\n")
            count += 1
        if count != game.row_count or digest.hexdigest() != game.semantic_digest:
            raise ValueError("native index staging completed game failed semantic validation")
    return expected


def _clean_unfinished_games(stage: Path, *, completed_count: int, source_count: int) -> None:
    for ordinal in range(completed_count, source_count):
        for head in ("policy", "value"):
            _discard_path(stage / "chunks" / head / f"{ordinal:08d}")


def _build_one_game(
    stage: Path,
    source_root: Path,
    receipt: NativeGameSourceReceipt,
    *,
    game_ordinal: int,
    chunk_size: int,
    policy_global_start: int,
    value_global_start: int,
) -> NativeIndexedGameMetadata:
    path = _source_path(source_root, receipt)
    if _file_sha256(path) != (receipt.file_sha256, receipt.file_size):
        raise ValueError(f"native source does not match receipt: {receipt.logical_name}")
    policy_buffer: list[PolicyDatasetRecord] = []
    value_buffer: list[ValueDatasetRecord] = []
    policy_chunks: list[NativeIndexChunkMetadata] = []
    value_chunks: list[NativeIndexChunkMetadata] = []
    game: NativeGameIdentity | None = None
    semantic = hashlib.sha256()
    rows = policy_count = value_count = boundary_count = 0
    last_boundary: int | None = None

    def flush_policy() -> None:
        if not policy_buffer:
            return
        start = policy_count - len(policy_buffer)
        policy_chunks.append(
            _write_chunk(
                stage,
                policy_buffer,
                head="POLICY",
                game_ordinal=game_ordinal,
                chunk_ordinal=len(policy_chunks),
                global_head_start=policy_global_start + start,
                game_head_start=start,
                full_game_head_count=receipt.policy_row_count,
            )
        )
        policy_buffer.clear()

    def flush_value() -> None:
        if not value_buffer:
            return
        start = value_count - len(value_buffer)
        value_chunks.append(
            _write_chunk(
                stage,
                value_buffer,
                head="VALUE",
                game_ordinal=game_ordinal,
                chunk_ordinal=len(value_chunks),
                global_head_start=value_global_start + start,
                game_head_start=start,
                full_game_head_count=receipt.value_row_count,
            )
        )
        value_buffer.clear()

    for record in iter_native_game_records(path):
        game = record.game if game is None else game
        semantic.update(canonical_json_bytes(record) + b"\n")
        rows += 1
        if isinstance(record, PolicyDatasetRecord):
            policy_count += 1
            policy_buffer.append(record)
            if len(policy_buffer) == chunk_size:
                flush_policy()
        else:
            value_count += 1
            if record.boundary.boundary_index != last_boundary:
                boundary_count += 1
                last_boundary = record.boundary.boundary_index
            value_buffer.append(record)
            if len(value_buffer) == chunk_size:
                flush_value()
    flush_policy()
    flush_value()
    if game is None:
        raise ValueError("native source game is empty")
    if _file_sha256(path) != (receipt.file_sha256, receipt.file_size):
        raise ValueError(f"native source changed while indexing: {receipt.logical_name}")
    if (
        game.game_id,
        rows,
        policy_count,
        value_count,
        boundary_count,
    ) != (
        receipt.game_id,
        receipt.row_count,
        receipt.policy_row_count,
        receipt.value_row_count,
        receipt.boundary_count,
    ):
        raise ValueError("native source identity or count does not match its receipt")
    for head in ("policy", "value"):
        directory = stage / "chunks" / head / f"{game_ordinal:08d}"
        if directory.exists():
            fsync_directory(directory)
            fsync_directory(directory.parent)
    return NativeIndexedGameMetadata(
        **game.model_dump(mode="python"),
        source_logical_name=receipt.logical_name,
        game_ordinal=game_ordinal,
        row_count=rows,
        policy_row_count=policy_count,
        value_row_count=value_count,
        boundary_count=boundary_count,
        semantic_digest=semantic.hexdigest(),
        policy_global_start=policy_global_start,
        value_global_start=value_global_start,
        policy_chunks=tuple(policy_chunks),
        value_chunks=tuple(value_chunks),
    )


def _semantic_dataset_digest(root: Path, games: Sequence[NativeIndexedGameMetadata]) -> str:
    digest = hashlib.sha256()
    for game in games:
        for record in _iter_indexed_game_records(root, game):
            digest.update(canonical_json_bytes(record) + b"\n")
    return digest.hexdigest()


@contextmanager
def _exclusive_build_lock(destination: Path) -> Iterator[None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _remove_backup(path: Path) -> None:
    _discard_path(path)


def _publish_directory(stage: Path, destination: Path) -> None:
    _require_owned_directory(stage, label="native index publication source")
    _require_replaceable_directory(destination, label="native index cache destination")
    backup: Path | None = None
    if destination.exists():
        backup = destination.parent / f".{destination.name}.{uuid4().hex}.old"
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
        fsync_directory(destination.parent)
    except BaseException:
        _discard_path(destination)
        if backup is not None:
            os.replace(backup, destination)
            fsync_directory(destination.parent)
        raise
    if backup is not None:
        _remove_backup(backup)
        fsync_directory(destination.parent)


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")


def build_native_indexed_dataset(
    source_root: str | Path,
    source_receipt_path: str | Path,
    cache_dir: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> IndexedNativeDataset:
    """Build or reuse an exact index, retaining validated completed-game staging."""
    _validate_chunk_size(chunk_size)
    source = Path(source_root)
    destination = Path(cache_dir)
    if destination.is_symlink():
        raise ValueError("native index cache destination must not be a symlink")
    receipt_path = Path(source_receipt_path)
    receipt = load_native_source_receipt(receipt_path)
    _validate_cache_separation(source, receipt_path, destination, receipt)
    decision_schema = TensorFeatureSchema.current()
    value_schema = StableValueTensorSchema.current()
    with _exclusive_build_lock(destination):
        _validate_source_files(source, receipt)
        compatible = _compatible_manifest(
            destination,
            source,
            receipt,
            chunk_size=chunk_size,
            decision_schema=decision_schema,
            value_schema=value_schema,
        )
        if compatible is not None:
            return IndexedNativeDataset(destination, source, compatible)
        _require_replaceable_directory(destination, label="native index cache destination")

        stage = destination.parent / f".{destination.name}.staging"
        try:
            checkpoint = _load_checkpoint(
                stage,
                receipt,
                chunk_size=chunk_size,
                decision_schema=decision_schema,
                value_schema=value_schema,
            )
        except (OSError, ValueError):
            checkpoint = _prepare_new_stage(
                stage,
                receipt,
                chunk_size=chunk_size,
                decision_schema=decision_schema,
                value_schema=value_schema,
            )
        completed = list(checkpoint.completed_games)
        _clean_unfinished_games(
            stage, completed_count=len(completed), source_count=len(receipt.games)
        )
        policy_global_start = sum(game.policy_row_count for game in completed)
        value_global_start = sum(game.value_row_count for game in completed)
        for ordinal in range(len(completed), len(receipt.games)):
            try:
                game = _build_one_game(
                    stage,
                    source,
                    receipt.games[ordinal],
                    game_ordinal=ordinal,
                    chunk_size=chunk_size,
                    policy_global_start=policy_global_start,
                    value_global_start=value_global_start,
                )
            except BaseException:
                for head in ("policy", "value"):
                    _discard_path(stage / "chunks" / head / f"{ordinal:08d}")
                raise
            completed.append(game)
            policy_global_start += game.policy_row_count
            value_global_start += game.value_row_count
            checkpoint = checkpoint.model_copy(update={"completed_games": tuple(completed)})
            checkpoint = _StagingCheckpoint.model_validate(
                checkpoint.model_dump(mode="python"), strict=True
            )
            _write_checkpoint(stage, checkpoint)

        _validate_source_files(source, receipt)
        semantic_digest = _semantic_dataset_digest(stage, completed)
        manifest = _manifest_from_games(
            receipt,
            tuple(completed),
            chunk_size=chunk_size,
            semantic_dataset_digest=semantic_digest,
            decision_schema=decision_schema,
            value_schema=value_schema,
        )
        atomic_write_bytes(stage / _MANIFEST_NAME, manifest.canonical_bytes())
        candidate = IndexedNativeDataset(stage, source, _load_manifest(stage))
        for game_id in candidate.game_ids:
            candidate.validate_game(game_id)
        (stage / _CHECKPOINT_NAME).unlink()
        fsync_directory(stage)
        _publish_directory(stage, destination)
        return IndexedNativeDataset(destination, source, manifest)


def open_native_indexed_dataset(
    source_root: str | Path,
    source_receipt_path: str | Path,
    cache_dir: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> IndexedNativeDataset:
    """Open an exact cache, or safely rebuild a stale/corrupt disposable cache."""
    _validate_chunk_size(chunk_size)
    source = Path(source_root)
    destination = Path(cache_dir)
    if destination.is_symlink():
        raise ValueError("native index cache destination must not be a symlink")
    receipt_path = Path(source_receipt_path)
    receipt = load_native_source_receipt(receipt_path)
    _validate_cache_separation(source, receipt_path, destination, receipt)
    _validate_source_files(source, receipt)
    decision_schema = TensorFeatureSchema.current()
    value_schema = StableValueTensorSchema.current()
    compatible = _compatible_manifest(
        destination,
        source,
        receipt,
        chunk_size=chunk_size,
        decision_schema=decision_schema,
        value_schema=value_schema,
    )
    if compatible is not None:
        return IndexedNativeDataset(destination, source, compatible)
    return build_native_indexed_dataset(
        source,
        source_receipt_path,
        destination,
        chunk_size=chunk_size,
    )


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "INDEX_SCHEMA_VERSION",
    "IndexedNativeDataset",
    "NativeDatasetSourceReceipt",
    "NativeGameSourceReceipt",
    "NativeIndexChunkMetadata",
    "NativeIndexedDatasetManifest",
    "NativeIndexedGameMetadata",
    "build_native_indexed_dataset",
    "create_native_source_receipt",
    "load_native_source_receipt",
    "open_native_indexed_dataset",
]
