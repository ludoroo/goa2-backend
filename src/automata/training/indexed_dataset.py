"""Disk-backed, per-game index for strict joint training datasets.

The index is a disposable cache: its manifest is bound to the exact bytes of
its source file, while validated-row byte digests bind both the complete dataset
and each independently readable game fragment.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import math
import multiprocessing
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

import torch
import zstandard
from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, model_validator
from tqdm import tqdm

from automata.decision import DecisionSemanticRole
from automata.models.contracts import CandidateID, canonical_json_bytes
from automata.models.shared_encoder.batching import (
    CandidateTable,
    DecisionBatch,
    FeatureTable,
    RelationshipTable,
    collate_decisions,
)
from automata.models.shared_encoder.schema import (
    RecordFeatureSchema,
    TensorFeatureSchema,
    TensorSchemaID,
    TensorSchemaVersion,
    expanded_numeric_width,
)
from automata.training.dataset import (
    JointDatasetRow,
    TerminalWinner,
    iter_joint_dataset_records,
    iter_joint_row_records,
)
from automata.training.io import TQDM_BAR_FORMAT, atomic_write_bytes, fsync_directory
from goa2.domain.input import InputRequestType

INDEX_SCHEMA_VERSION: Literal[2] = 2
DEFAULT_TRAINING_CHUNK_SIZE = 32
_TRAINING_CHUNK_FORMAT: Literal["torch-save-zstd"] = "torch-save-zstd"
_TRAINING_CHUNK_FORMAT_VERSION: Literal[2] = 2
_METRIC_METADATA_VERSION: Literal[2] = 2
_MANIFEST_NAME = "manifest.json"
_STAGING_CHECKPOINT_NAME = "checkpoint.json"
_STAGING_SCHEMA_VERSION: Literal[1] = 1
_READ_SIZE = 128 * 1024
_CANDIDATE_ID_ADAPTER: TypeAdapter[CandidateID] = TypeAdapter(CandidateID)
_KNOWN_INPUT_REQUEST_TYPES = frozenset(item.value for item in InputRequestType)


class IndexedTrainingChunkMetadata(BaseModel):
    """Manifest binding for one safely serialized tensor training chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str = Field(pattern=r"^training/[0-9]{8}/[0-9]{8}\.pt\.zst$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uncompressed_size: StrictInt = Field(gt=0)
    global_row_start: StrictInt = Field(ge=0)
    row_count: StrictInt = Field(gt=0)


class IndexedGameMetadata(BaseModel):
    """Manifest identity and location of one complete game's rows."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    game_id: str = Field(min_length=1)
    fragment: str = Field(pattern=r"^fragments/[0-9]{8}\.jsonl\.zst$")
    semantic_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    fragment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_row_start: StrictInt = Field(ge=0)
    row_count: StrictInt = Field(gt=0)
    world_seed: StrictInt
    map_id: str = Field(min_length=1)
    game_type: str = Field(min_length=1)
    red_composition: tuple[str, ...] = Field(min_length=1)
    blue_composition: tuple[str, ...] = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    dirty_tree_hash: str = Field(min_length=1)
    source_model_digest: str | None = Field(default=None, min_length=1)
    search_config_id: str = Field(min_length=1)
    generator_config_id: str = Field(min_length=1)
    terminal_winner: TerminalWinner
    training_chunks: tuple[IndexedTrainingChunkMetadata, ...]


class IndexedDatasetManifest(BaseModel):
    """Canonical, versioned identity of one complete disk-backed index."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[2] = 2
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size: StrictInt = Field(ge=0)
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    tensor_schema_id: TensorSchemaID
    tensor_schema_version: TensorSchemaVersion
    tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_chunk_format_version: Literal[2] = _TRAINING_CHUNK_FORMAT_VERSION
    metric_metadata_version: Literal[2]
    training_chunk_size: StrictInt = Field(gt=0)
    row_count: StrictInt = Field(gt=0)
    game_count: StrictInt = Field(gt=0)
    games: tuple[IndexedGameMetadata, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_index_totals(self) -> IndexedDatasetManifest:
        if self.game_count != len(self.games):
            raise ValueError("game_count must equal the number of game entries")
        if self.row_count != sum(game.row_count for game in self.games):
            raise ValueError("row_count must equal the sum of per-game row counts")
        game_ids = tuple(game.game_id for game in self.games)
        fragments = tuple(game.fragment for game in self.games)
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("indexed game IDs must be unique")
        if len(set(fragments)) != len(fragments):
            raise ValueError("indexed fragment paths must be unique")
        expected_start = 0
        training_paths: set[str] = set()
        for game in self.games:
            if game.global_row_start != expected_start:
                raise ValueError("per-game global row ranges must be exactly contiguous")
            chunk_start = game.global_row_start
            for index, chunk in enumerate(game.training_chunks):
                if chunk.path in training_paths:
                    raise ValueError("indexed training chunk paths must be unique")
                training_paths.add(chunk.path)
                if chunk.global_row_start != chunk_start:
                    raise ValueError("training chunk global row ranges must be exactly contiguous")
                if chunk.row_count > self.training_chunk_size:
                    raise ValueError("training chunk exceeds the manifest chunk size")
                if index < len(game.training_chunks) - 1 and (
                    chunk.row_count != self.training_chunk_size
                ):
                    raise ValueError("only a game's final training chunk may be short")
                chunk_start += chunk.row_count
            expected_start += game.row_count
            if chunk_start != expected_start:
                raise ValueError("training chunks must cover every indexed game row")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the path-independent canonical manifest representation."""
        return canonical_json_bytes(self)


class _TensorizationCheckpoint(BaseModel):
    """Durable scan identity plus atomically updated per-game completions."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    staging_schema_version: Literal[1] = _STAGING_SCHEMA_VERSION
    scan_complete: Literal[True] = True
    index_schema_version: Literal[2] = INDEX_SCHEMA_VERSION
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size: StrictInt = Field(ge=0)
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    tensor_schema_id: TensorSchemaID
    tensor_schema_version: TensorSchemaVersion
    tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_chunk_format: Literal["torch-save-zstd"] = _TRAINING_CHUNK_FORMAT
    training_chunk_format_version: Literal[2] = _TRAINING_CHUNK_FORMAT_VERSION
    metric_metadata_version: Literal[2]
    training_chunk_size: StrictInt = Field(gt=0)
    row_count: StrictInt = Field(gt=0)
    game_count: StrictInt = Field(gt=0)
    games: tuple[IndexedGameMetadata, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_staging_ranges(self) -> _TensorizationCheckpoint:
        if self.game_count != len(self.games):
            raise ValueError("staging game count does not match its game metadata")
        if self.row_count != sum(game.row_count for game in self.games):
            raise ValueError("staging row count does not match its game metadata")
        if len({game.game_id for game in self.games}) != len(self.games):
            raise ValueError("staging game IDs must be unique")
        expected_start = 0
        for ordinal, game in enumerate(self.games):
            if game.fragment != f"fragments/{ordinal:08d}.jsonl.zst":
                raise ValueError("staging fragment paths do not match game order")
            if game.global_row_start != expected_start:
                raise ValueError("staging game row ranges must be contiguous")
            expected_start += game.row_count
            if not game.training_chunks:
                continue
            chunk_start = game.global_row_start
            expected_chunks = (game.row_count + self.training_chunk_size - 1) // (
                self.training_chunk_size
            )
            if len(game.training_chunks) != expected_chunks:
                raise ValueError("completed staging game has incomplete chunk coverage")
            for chunk_ordinal, chunk in enumerate(game.training_chunks):
                if chunk.path != f"training/{ordinal:08d}/{chunk_ordinal:08d}.pt.zst":
                    raise ValueError("staging training chunk paths do not match game order")
                expected_count = min(
                    self.training_chunk_size,
                    game.global_row_start + game.row_count - chunk_start,
                )
                if chunk.global_row_start != chunk_start or chunk.row_count != expected_count:
                    raise ValueError("staging training chunk row ranges are invalid")
                chunk_start += chunk.row_count
            if chunk_start != game.global_row_start + game.row_count:
                raise ValueError("staging training chunks do not cover their game")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


class _TensorizationCheckpointEnvelope(BaseModel):
    """Atomically written checkpoint plus an integrity digest of its metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    checkpoint: _TensorizationCheckpoint
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_checkpoint_digest(self) -> _TensorizationCheckpointEnvelope:
        expected = hashlib.sha256(self.checkpoint.canonical_bytes()).hexdigest()
        if self.checkpoint_sha256 != expected:
            raise ValueError("tensorization staging checkpoint digest mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


@dataclass(frozen=True, slots=True)
class TrainingMetricMetadata:
    """Compact row attributes required to construct policy/value diagnostics."""

    candidate_count: int
    candidate_family: str
    target_probabilities: tuple[float, ...]
    prior_probabilities: tuple[float, ...] | None
    q_variances: tuple[float, ...] | None
    hero: str | None
    map_id: str
    composition: str
    round_bucket: str | None
    input_request_type: str | None
    semantic_role: DecisionSemanticRole
    can_skip: bool

    def __post_init__(self) -> None:
        if self.input_request_type is not None and (
            self.input_request_type not in _KNOWN_INPUT_REQUEST_TYPES
        ):
            raise ValueError("metric input request type is unknown")
        if self.semantic_role is DecisionSemanticRole.PLANNING:
            if self.input_request_type is not None or self.can_skip:
                raise ValueError("CARD planning metric context must use null request and no skip")
        elif self.input_request_type is None:
            raise ValueError("non-planning metric context requires an input request type")


@dataclass(frozen=True, slots=True)
class IndexedTrainingChunk:
    """One pre-collated, model-ready chunk loaded from the safe tensor cache."""

    batch: DecisionBatch
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    global_row_offsets: tuple[int, ...]
    game_ids: tuple[str, ...]
    metric_metadata: tuple[TrainingMetricMetadata, ...]

    @property
    def row_count(self) -> int:
        """Return the number of decisions in this chunk."""
        return len(self.global_row_offsets)


@dataclass(frozen=True, slots=True)
class IndexedJointDataset:
    """An opened index whose game fragments are validated when consumed."""

    cache_dir: Path
    source_path: Path
    manifest: IndexedDatasetManifest
    _games_by_id: Mapping[str, IndexedGameMetadata]

    @classmethod
    def _open(
        cls, cache_dir: Path, source_path: Path, manifest: IndexedDatasetManifest
    ) -> IndexedJointDataset:
        games = MappingProxyType({game.game_id: game for game in manifest.games})
        return cls(cache_dir, source_path, manifest, games)

    @property
    def digest(self) -> str:
        """Return the digest of validated uncompressed rows in source order."""
        return self.manifest.dataset_digest

    @property
    def game_ids(self) -> tuple[str, ...]:
        """Return game IDs in first-seen source order."""
        return tuple(game.game_id for game in self.manifest.games)

    def game(self, game_id: str) -> IndexedGameMetadata:
        """Return split/scope/provenance metadata for ``game_id``."""
        try:
            return self._games_by_id[game_id]
        except KeyError as exc:
            raise KeyError(f"unknown indexed game: {game_id!r}") from exc

    def iter_game_rows(self, game_id: str) -> Iterator[JointDatasetRow]:
        """Stream and validate one game, checking its digest at exhaustion.

        As with :func:`iter_joint_dataset`, end-of-stream integrity errors can
        follow already yielded rows. Call :meth:`validate_game` when validation
        must complete before rows are used.
        """
        metadata = self.game(game_id)
        fragment = _safe_fragment_path(self.cache_dir, metadata.fragment)
        fragment_digest, _ = _file_sha256(fragment)
        if fragment_digest != metadata.fragment_sha256:
            raise ValueError(f"indexed fragment digest for {game_id!r} does not match its manifest")
        count = 0
        for row, _ in iter_joint_row_records(fragment):
            if row.game_id != game_id:
                raise ValueError(f"indexed fragment for {game_id!r} contains another game")
            count += 1
            yield row
        if count != metadata.row_count:
            raise ValueError(
                f"indexed fragment row count for {game_id!r} does not match its manifest"
            )

    def iter_game_chunks(
        self, game_id: str, *, chunk_size: int = 1024
    ) -> Iterator[tuple[JointDatasetRow, ...]]:
        """Yield one validated game in tuples of at most ``chunk_size`` rows."""
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        rows = self.iter_game_rows(game_id)
        while True:
            chunk: list[JointDatasetRow] = []
            try:
                for _ in range(chunk_size):
                    chunk.append(next(rows))
            except StopIteration:
                if chunk:
                    yield tuple(chunk)
                return
            yield tuple(chunk)

    def iter_game_training_chunks(self, game_id: str) -> Iterator[IndexedTrainingChunk]:
        """Yield one game's pre-collated chunks after validating each exact file."""
        game = self.game(game_id)
        schema = TensorFeatureSchema.current()
        if (
            schema.schema_id != self.manifest.tensor_schema_id
            or schema.schema_version != self.manifest.tensor_schema_version
            or schema.digest != self.manifest.tensor_schema_digest
        ):
            raise ValueError("current tensor schema does not match the indexed training cache")
        for metadata in game.training_chunks:
            path = _safe_cache_file(self.cache_dir, metadata.path, kind="training chunk")
            digest, _ = _file_sha256(path)
            if digest != metadata.sha256:
                raise ValueError(
                    f"indexed training chunk digest for {game_id!r} does not match its manifest"
                )
            try:
                compressed = path.read_bytes()
                serialized = zstandard.ZstdDecompressor().decompress(
                    compressed, max_output_size=metadata.uncompressed_size
                )
                if len(serialized) != metadata.uncompressed_size:
                    raise ValueError("training chunk decompressed size does not match its manifest")
                payload = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ValueError(f"indexed training chunk for {game_id!r} is malformed") from exc
            yield _training_chunk_from_payload(
                payload,
                metadata=metadata,
                expected_game_id=game_id,
                schema=schema,
            )

    def validate_game(self, game_id: str, *, chunk_size: int = 1024) -> IndexedGameMetadata:
        """Exhaustively validate one fragment using bounded row batches."""
        for _ in self.iter_game_chunks(game_id, chunk_size=chunk_size):
            pass
        return self.game(game_id)


class _ActiveFragment:
    """The sole compressed fragment writer retained during a streaming build."""

    __slots__ = ("_output", "_path", "_writer", "digest", "metadata", "row_count")

    def __init__(self, root: Path, metadata: IndexedGameMetadata) -> None:
        self.metadata = metadata
        self.digest = hashlib.sha256()
        self.row_count = 0
        self._path = root / metadata.fragment
        self._output = self._path.open("wb")
        compressor = zstandard.ZstdCompressor(level=3, threads=0, write_checksum=True)
        try:
            self._writer = compressor.stream_writer(self._output, closefd=False)
        except BaseException:
            self._output.close()
            raise

    def write(self, payload: bytes) -> None:
        """Write one validated source row directly into the active zstd frame."""
        self._writer.write(payload)
        self.digest.update(payload)
        self.row_count += 1

    def finish(self) -> IndexedGameMetadata:
        """Finalize and durably close the fragment, returning bound metadata."""
        try:
            self._writer.close()
            self._output.flush()
            os.fsync(self._output.fileno())
        finally:
            self._output.close()
        fragment_sha256, _ = _file_sha256(self._path)
        return self.metadata.model_copy(
            update={
                "semantic_digest": self.digest.hexdigest(),
                "fragment_sha256": fragment_sha256,
                "row_count": self.row_count,
            }
        )

    def discard(self) -> None:
        """Best-effort close used while discarding an unpublished build."""
        with suppress(Exception):
            self._writer.close()
        with suppress(Exception):
            self._output.close()


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_READ_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _safe_cache_file(cache_dir: Path, relative: str, *, kind: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts:
        raise ValueError(f"indexed {kind} path must remain inside its cache")
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise ValueError(f"indexed {kind} cache root must be a regular directory")
    path = cache_dir.joinpath(*parts.parts)
    current = cache_dir
    for part in parts.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"indexed {kind} path must not contain symlinks")
    if not path.is_file():
        raise ValueError(f"indexed {kind} is missing: {relative}")
    return path


def _safe_fragment_path(cache_dir: Path, relative: str) -> Path:
    return _safe_cache_file(cache_dir, relative, kind="fragment")


def _metadata_from_row(
    row: JointDatasetRow, *, ordinal: int, global_row_start: int
) -> IndexedGameMetadata:
    return IndexedGameMetadata(
        game_id=row.game_id,
        fragment=f"fragments/{ordinal:08d}.jsonl.zst",
        semantic_digest="0" * 64,
        fragment_sha256="0" * 64,
        global_row_start=global_row_start,
        row_count=1,
        world_seed=row.world_seed,
        map_id=row.map_id,
        game_type=row.game_type,
        red_composition=row.red_composition,
        blue_composition=row.blue_composition,
        generation_id=row.generation_id,
        source_revision=row.source_revision,
        dirty_tree_hash=row.dirty_tree_hash,
        source_model_digest=row.source_model_digest,
        search_config_id=row.search_config_id,
        generator_config_id=row.generator_config_id,
        terminal_winner=row.terminal_winner,
        training_chunks=(),
    )


def _table_payload(table: FeatureTable) -> dict[str, torch.Tensor]:
    return {
        name: value for name, value in table.__dict__.items() if isinstance(value, torch.Tensor)
    }


def _batch_payload(batch: DecisionBatch) -> dict[str, Any]:
    return {
        "tokens": {kind: _table_payload(table) for kind, table in batch.tokens.items()},
        "relationships": {
            kind: _table_payload(table) for kind, table in batch.relationships.items()
        },
        "decision_context": (
            None if batch.decision_context is None else _table_payload(batch.decision_context)
        ),
        "candidates": _table_payload(batch.candidates),
        "candidate_ids": [
            [candidate.model_dump(mode="json") for candidate in row] for row in batch.candidate_ids
        ],
        "token_kinds": list(batch.token_kinds),
        "candidate_kinds": list(batch.candidate_kinds),
    }


def _metric_metadata(row: JointDatasetRow) -> TrainingMetricMetadata:
    global_features = next(
        token.features for token in row.observation.state.tokens if token.kind == "GLOBAL"
    )
    perspective_heroes = [
        token
        for token in row.observation.state.tokens
        if token.kind == "HERO" and token.features.get("team_id") == row.perspective_team
    ]
    hero_token = next(
        (
            token
            for token in perspective_heroes
            if token.features.get("is_decision_owner") or token.features.get("is_current_actor")
        ),
        perspective_heroes[0] if perspective_heroes else None,
    )
    priors: tuple[float, ...] | None = None
    variances: tuple[float, ...] | None = None
    if row.action_stats is not None:
        action_priors = tuple(item.prior_probability for item in row.action_stats)
        if all(value is not None for value in action_priors):
            priors = tuple(float(value) for value in action_priors if value is not None)
        variances = tuple(float(item.value_variance) for item in row.action_stats)
    return TrainingMetricMetadata(
        candidate_count=len(row.policy_target),
        candidate_family="+".join(
            sorted({candidate.candidate_id.kind for candidate in row.observation.candidates})
        ),
        target_probabilities=tuple(row.policy_target),
        prior_probabilities=priors,
        q_variances=variances,
        hero=str(hero_token.features["name"]) if hero_token is not None else None,
        map_id=row.map_id,
        composition=f"{'/'.join(row.red_composition)} vs {'/'.join(row.blue_composition)}",
        round_bucket=str(global_features["round"]) if "round" in global_features else None,
        input_request_type=row.observation.input_request_type,
        semantic_role=row.observation.semantic_role,
        can_skip=row.observation.can_skip,
    )


def _training_chunk_payload(
    rows: Sequence[JointDatasetRow], *, global_row_start: int, schema: TensorFeatureSchema
) -> dict[str, Any]:
    batch = collate_decisions([row.observation for row in rows], schema=schema, training=True)
    width = batch.candidates.mask.shape[1]
    policy_targets = torch.zeros((len(rows), width), dtype=torch.float32)
    for index, row in enumerate(rows):
        policy_targets[index, : len(row.policy_target)] = torch.tensor(
            row.policy_target, dtype=torch.float32
        )
    metadata = [_metric_metadata(row) for row in rows]
    return {
        "format_version": _TRAINING_CHUNK_FORMAT_VERSION,
        "batch": _batch_payload(batch),
        "policy_targets": policy_targets,
        "value_targets": torch.tensor([row.value_target for row in rows], dtype=torch.float32),
        "global_row_offsets": list(range(global_row_start, global_row_start + len(rows))),
        "game_ids": [row.game_id for row in rows],
        "metric_metadata": [
            {
                "candidate_count": item.candidate_count,
                "candidate_family": item.candidate_family,
                "target_probabilities": list(item.target_probabilities),
                "prior_probabilities": (
                    list(item.prior_probabilities) if item.prior_probabilities is not None else None
                ),
                "q_variances": list(item.q_variances) if item.q_variances is not None else None,
                "hero": item.hero,
                "map_id": item.map_id,
                "composition": item.composition,
                "round_bucket": item.round_bucket,
                "input_request_type": item.input_request_type,
                "semantic_role": item.semantic_role.value,
                "can_skip": item.can_skip,
            }
            for item in metadata
        ],
    }


class _CountingWriter:
    """Count uncompressed bytes passed into a streaming compressor."""

    __slots__ = ("byte_count", "writer")

    def __init__(self, writer: Any) -> None:
        self.writer = writer
        self.byte_count = 0

    def write(self, payload: bytes) -> int:
        self.byte_count += len(payload)
        return int(self.writer.write(payload))

    def flush(self) -> None:
        self.writer.flush(zstandard.FLUSH_BLOCK)


def _write_training_chunk(
    root: Path,
    rows: Sequence[JointDatasetRow],
    *,
    game_ordinal: int,
    chunk_ordinal: int,
    global_row_start: int,
    schema: TensorFeatureSchema,
) -> IndexedTrainingChunkMetadata:
    relative = f"training/{game_ordinal:08d}/{chunk_ordinal:08d}.pt.zst"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _training_chunk_payload(rows, global_row_start=global_row_start, schema=schema)
    with path.open("wb") as output:
        compressor = zstandard.ZstdCompressor(level=3, threads=0, write_checksum=True)
        compressed = compressor.stream_writer(output, closefd=False)
        counted = _CountingWriter(compressed)
        torch_destination: Any = counted
        try:
            torch.save(payload, torch_destination)
        finally:
            compressed.close()
        output.flush()
        os.fsync(output.fileno())
    digest, _ = _file_sha256(path)
    return IndexedTrainingChunkMetadata(
        path=relative,
        sha256=digest,
        uncompressed_size=counted.byte_count,
        global_row_start=global_row_start,
        row_count=len(rows),
    )


def _initialize_tensor_worker() -> None:
    """Prevent each index process from creating its own CPU thread pool."""
    torch.set_num_threads(1)
    # Some embedding environments initialize the inter-op pool at import.
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _tensorize_game_fragment(
    root_path: str,
    fragment_relative: str,
    game_id: str,
    game_ordinal: int,
    global_row_start: int,
    expected_row_count: int,
    expected_fragment_sha256: str,
    expected_semantic_digest: str,
    training_chunk_size: int,
    tensor_schema_digest: str,
) -> tuple[dict[str, Any], ...]:
    """Tensorize one complete fragment in a spawn-safe worker process."""
    root = Path(root_path)
    schema = TensorFeatureSchema.current()
    if schema.digest != tensor_schema_digest:
        raise ValueError("worker tensor schema does not match the index build schema")
    fragment = _safe_fragment_path(root, fragment_relative)
    fragment_sha256, _ = _file_sha256(fragment)
    if fragment_sha256 != expected_fragment_sha256:
        raise ValueError(f"indexed fragment digest for {game_id!r} changed before tensorization")
    semantic_digest = hashlib.sha256()
    chunk_rows: list[JointDatasetRow] = []
    chunks: list[IndexedTrainingChunkMetadata] = []
    row_count = 0

    def flush_chunk() -> None:
        if not chunk_rows:
            return
        chunks.append(
            _write_training_chunk(
                root,
                chunk_rows,
                game_ordinal=game_ordinal,
                chunk_ordinal=len(chunks),
                global_row_start=global_row_start + row_count - len(chunk_rows),
                schema=schema,
            )
        )
        chunk_rows.clear()

    for row, raw_line in iter_joint_row_records(fragment):
        if row.game_id != game_id:
            raise ValueError(f"indexed fragment for {game_id!r} contains another game")
        semantic_digest.update(raw_line + b"\n")
        chunk_rows.append(row)
        row_count += 1
        if len(chunk_rows) == training_chunk_size:
            flush_chunk()
    flush_chunk()
    if row_count != expected_row_count:
        raise ValueError(f"indexed fragment row count for {game_id!r} changed during tensorization")
    if semantic_digest.hexdigest() != expected_semantic_digest:
        raise ValueError(f"indexed fragment row bytes for {game_id!r} changed during tensorization")
    return tuple(chunk.model_dump(mode="json") for chunk in chunks)


def _require_dict(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"training chunk {label} has invalid fields")
    return value


def _require_tensor(value: Any, *, dtype: torch.dtype, ndim: int, label: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != dtype
        or value.ndim != ndim
    ):
        raise ValueError(f"training chunk {label} is not a compatible CPU tensor")
    if value.is_floating_point() and not torch.isfinite(value).all().item():
        raise ValueError(f"training chunk {label} contains non-finite values")
    return value


_FEATURE_FIELDS = {
    "numeric",
    "numeric_valid",
    "categorical",
    "references",
    "reference_valid",
    "reference_kind_indices",
    "mask",
}
_RELATIONSHIP_FIELDS = _FEATURE_FIELDS | {
    "source_indices",
    "target_indices",
    "source_kind_indices",
    "target_kind_indices",
}
_CANDIDATE_FIELDS = _FEATURE_FIELDS | {
    "kind_indices",
    "target_indices",
    "target_kind_indices",
    "target_valid",
}


def _load_feature_table(
    value: Any,
    record_schema: RecordFeatureSchema | None = None,
    *,
    label: str,
    extra_fields: set[str] | None = None,
    widths: tuple[int, int, int] | None = None,
) -> dict[str, torch.Tensor]:
    fields = _FEATURE_FIELDS | (extra_fields or set())
    payload = _require_dict(value, fields, label)
    mask = _require_tensor(payload["mask"], dtype=torch.bool, ndim=2, label=f"{label}.mask")
    shape = mask.shape
    tensors = {
        "numeric": _require_tensor(
            payload["numeric"], dtype=torch.float32, ndim=3, label=f"{label}.numeric"
        ),
        "numeric_valid": _require_tensor(
            payload["numeric_valid"],
            dtype=torch.bool,
            ndim=3,
            label=f"{label}.numeric_valid",
        ),
        "categorical": _require_tensor(
            payload["categorical"],
            dtype=torch.int64,
            ndim=3,
            label=f"{label}.categorical",
        ),
        "references": _require_tensor(
            payload["references"], dtype=torch.int64, ndim=3, label=f"{label}.references"
        ),
        "reference_valid": _require_tensor(
            payload["reference_valid"],
            dtype=torch.bool,
            ndim=3,
            label=f"{label}.reference_valid",
        ),
        "reference_kind_indices": _require_tensor(
            payload["reference_kind_indices"],
            dtype=torch.int64,
            ndim=3,
            label=f"{label}.reference_kind_indices",
        ),
        "mask": mask,
    }
    if (record_schema is None) == (widths is None):
        raise ValueError("exactly one feature schema or explicit width tuple is required")
    if record_schema is not None:
        widths = (
            expanded_numeric_width(record_schema),
            len(record_schema.categorical),
            len(record_schema.references),
        )
    assert widths is not None
    numeric_width, categorical_width, reference_width = widths
    expected_shapes = {
        "numeric": (*shape, numeric_width),
        "numeric_valid": (*shape, numeric_width),
        "categorical": (*shape, categorical_width),
        "references": (*shape, reference_width),
        "reference_valid": (*shape, reference_width),
        "reference_kind_indices": (*shape, reference_width),
    }
    if any(tensors[name].shape != expected for name, expected in expected_shapes.items()):
        raise ValueError(f"training chunk {label} feature widths disagree with its schema")
    return tensors


def _load_batch(value: Any, *, schema: TensorFeatureSchema, row_count: int) -> DecisionBatch:
    payload = _require_dict(
        value,
        {
            "tokens",
            "relationships",
            "decision_context",
            "candidates",
            "candidate_ids",
            "token_kinds",
            "candidate_kinds",
        },
        "batch",
    )
    token_kinds = tuple(item.kind for item in schema.tokens)
    candidate_kinds = tuple(item.kind for item in schema.candidates)
    if payload["token_kinds"] != list(token_kinds) or payload["candidate_kinds"] != list(
        candidate_kinds
    ):
        raise ValueError("training chunk kinds disagree with its tensor schema")
    raw_tokens = _require_dict(payload["tokens"], set(token_kinds), "batch.tokens")
    tokens: dict[str, FeatureTable] = {}
    for record_schema in schema.tokens:
        fields = _load_feature_table(
            raw_tokens[record_schema.kind], record_schema, label=f"tokens.{record_schema.kind}"
        )
        if fields["mask"].shape[0] != row_count:
            raise ValueError("training chunk token batch dimension is invalid")
        tokens[record_schema.kind] = FeatureTable(**fields)

    relationship_kinds = tuple(item.kind for item in schema.relationships)
    raw_relationships = _require_dict(
        payload["relationships"], set(relationship_kinds), "batch.relationships"
    )
    relationships: dict[str, RelationshipTable] = {}
    relationship_extra = _RELATIONSHIP_FIELDS - _FEATURE_FIELDS
    for record_schema in schema.relationships:
        label = f"relationships.{record_schema.kind}"
        fields = _load_feature_table(
            raw_relationships[record_schema.kind],
            record_schema,
            label=label,
            extra_fields=relationship_extra,
        )
        shape = fields["mask"].shape
        for name in relationship_extra:
            fields[name] = _require_tensor(
                raw_relationships[record_schema.kind][name],
                dtype=torch.int64,
                ndim=2,
                label=f"{label}.{name}",
            )
            if fields[name].shape != shape:
                raise ValueError(f"training chunk {label} endpoint shape is invalid")
        if shape[0] != row_count:
            raise ValueError("training chunk relationship batch dimension is invalid")
        relationships[record_schema.kind] = RelationshipTable(**fields)

    decision_context: FeatureTable | None = None
    if schema.decision_context is not None:
        fields = _load_feature_table(
            payload["decision_context"],
            schema.decision_context,
            label="decision_context",
        )
        if fields["mask"].shape != (row_count, 1) or not fields["mask"].all().item():
            raise ValueError("training chunk requires one decision-context row per decision")
        decision_context = FeatureTable(**fields)
    elif payload["decision_context"] is not None:
        raise ValueError("legacy training chunk cannot carry decision context")

    # Candidate columns are padded to the widest candidate schema declaration.
    candidate_widths = (
        max(expanded_numeric_width(item) for item in schema.candidates),
        max(len(item.categorical) for item in schema.candidates),
        max(len(item.references) for item in schema.candidates),
    )
    raw_candidates = _require_dict(payload["candidates"], _CANDIDATE_FIELDS, "candidates")
    candidate_extra = _CANDIDATE_FIELDS - _FEATURE_FIELDS
    candidate_fields = _load_feature_table(
        raw_candidates,
        label="candidates",
        extra_fields=candidate_extra,
        widths=candidate_widths,
    )
    candidate_shape = candidate_fields["mask"].shape
    for name in candidate_extra:
        dtype = torch.bool if name == "target_valid" else torch.int64
        candidate_fields[name] = _require_tensor(
            raw_candidates[name], dtype=dtype, ndim=2, label=f"candidates.{name}"
        )
        if candidate_fields[name].shape != candidate_shape:
            raise ValueError("training chunk candidate field shape is invalid")
    candidate_mask = candidate_fields["mask"]
    if candidate_shape[0] != row_count or not candidate_mask.any(dim=1).all().item():
        raise ValueError("training chunk must contain candidates for every row")
    expected_mask = torch.arange(candidate_shape[1]).unsqueeze(0) < candidate_mask.sum(
        dim=1, keepdim=True
    )
    if not torch.equal(candidate_mask, expected_mask):
        raise ValueError("training chunk candidate masks must be contiguous prefixes")

    raw_ids = payload["candidate_ids"]
    if type(raw_ids) is not list or len(raw_ids) != row_count:
        raise ValueError("training chunk candidate IDs have invalid batch alignment")
    candidate_ids: list[tuple[CandidateID, ...]] = []
    for index, raw_row in enumerate(raw_ids):
        if type(raw_row) is not list:
            raise ValueError("training chunk candidate IDs must be plain lists")
        try:
            ids = tuple(
                _CANDIDATE_ID_ADAPTER.validate_python(item, strict=True) for item in raw_row
            )
        except ValueError as exc:
            raise ValueError("training chunk candidate ID is invalid") from exc
        if len(ids) != int(candidate_fields["mask"][index].sum().item()):
            raise ValueError("training chunk candidate IDs do not align with candidate tensors")
        candidate_ids.append(ids)
    return DecisionBatch(
        tokens=tokens,
        relationships=relationships,
        decision_context=decision_context,
        candidates=CandidateTable(**candidate_fields),
        candidate_ids=tuple(candidate_ids),
        token_kinds=token_kinds,
        candidate_kinds=candidate_kinds,
    )


def _optional_float_tuple(value: Any, *, size: int, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if type(value) is not list or len(value) != size:
        raise ValueError(f"training chunk {label} is invalid")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        raise ValueError(f"training chunk {label} must contain only numbers")
    values = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"training chunk {label} must contain finite numbers")
    return values


def _load_metric_metadata(
    value: Any, *, candidate_count: int, policy_target: torch.Tensor
) -> TrainingMetricMetadata:
    fields = {
        "candidate_count",
        "candidate_family",
        "target_probabilities",
        "prior_probabilities",
        "q_variances",
        "hero",
        "map_id",
        "composition",
        "round_bucket",
        "input_request_type",
        "semantic_role",
        "can_skip",
    }
    payload = _require_dict(value, fields, "metric metadata")
    count = payload["candidate_count"]
    strings = ("candidate_family", "map_id", "composition")
    if type(count) is not int or count != candidate_count:
        raise ValueError("training chunk metric candidate count is invalid")
    if any(type(payload[name]) is not str or not payload[name] for name in strings):
        raise ValueError("training chunk metric metadata contains an invalid string")
    for name in ("hero", "round_bucket"):
        if payload[name] is not None and type(payload[name]) is not str:
            raise ValueError("training chunk metric metadata contains an invalid optional string")
    input_request_type = payload["input_request_type"]
    if input_request_type is not None and (
        type(input_request_type) is not str or input_request_type not in _KNOWN_INPUT_REQUEST_TYPES
    ):
        raise ValueError("training chunk metric input request type is invalid")
    try:
        semantic_role = DecisionSemanticRole(payload["semantic_role"])
    except (TypeError, ValueError) as exc:
        raise ValueError("training chunk metric semantic role is invalid") from exc
    can_skip = payload["can_skip"]
    if type(can_skip) is not bool:
        raise ValueError("training chunk metric can_skip is invalid")
    targets = _optional_float_tuple(
        payload["target_probabilities"], size=count, label="metric targets"
    )
    if (
        targets is None
        or any(item < 0 for item in targets)
        or not math.isclose(sum(targets), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError("training chunk metric targets must be a probability distribution")
    if not torch.allclose(
        torch.tensor(targets, dtype=torch.float32), policy_target, rtol=1e-6, atol=1e-7
    ):
        raise ValueError("training chunk metric targets disagree with policy targets")
    priors = _optional_float_tuple(
        payload["prior_probabilities"], size=count, label="metric priors"
    )
    variances = _optional_float_tuple(payload["q_variances"], size=count, label="metric variances")
    if priors is not None and (
        any(item < 0 for item in priors) or not math.isclose(sum(priors), 1.0)
    ):
        raise ValueError("training chunk metric priors must be a probability distribution")
    if variances is not None and any(item < 0 for item in variances):
        raise ValueError("training chunk metric variances must be non-negative")
    return TrainingMetricMetadata(
        candidate_count=count,
        candidate_family=payload["candidate_family"],
        target_probabilities=targets,
        prior_probabilities=priors,
        q_variances=variances,
        hero=payload["hero"],
        map_id=payload["map_id"],
        composition=payload["composition"],
        round_bucket=payload["round_bucket"],
        input_request_type=input_request_type,
        semantic_role=semantic_role,
        can_skip=can_skip,
    )


def _training_chunk_from_payload(
    value: Any,
    *,
    metadata: IndexedTrainingChunkMetadata,
    expected_game_id: str,
    schema: TensorFeatureSchema,
) -> IndexedTrainingChunk:
    fields = {
        "format_version",
        "batch",
        "policy_targets",
        "value_targets",
        "global_row_offsets",
        "game_ids",
        "metric_metadata",
    }
    payload = _require_dict(value, fields, "root")
    if payload["format_version"] != _TRAINING_CHUNK_FORMAT_VERSION:
        raise ValueError("training chunk format version is unsupported")
    row_count = metadata.row_count
    batch = _load_batch(payload["batch"], schema=schema, row_count=row_count)
    policy = _require_tensor(
        payload["policy_targets"], dtype=torch.float32, ndim=2, label="policy targets"
    )
    value_targets = _require_tensor(
        payload["value_targets"], dtype=torch.float32, ndim=1, label="value targets"
    )
    if policy.shape != batch.candidates.mask.shape or value_targets.shape != (row_count,):
        raise ValueError("training chunk targets do not align with its decision batch")
    if (policy < 0).any().item():
        raise ValueError("training chunk policy targets must be non-negative")
    if not torch.equal(policy != 0, (policy != 0) & batch.candidates.mask):
        raise ValueError("training chunk policy target includes a padded candidate")
    if not torch.allclose(policy.sum(dim=1), torch.ones(row_count)):
        raise ValueError("training chunk policy targets must sum to one")
    if not torch.isin(value_targets, torch.tensor([-1.0, 0.0, 1.0])).all().item():
        raise ValueError("training chunk value targets are invalid")

    offsets = payload["global_row_offsets"]
    expected_offsets = list(
        range(metadata.global_row_start, metadata.global_row_start + metadata.row_count)
    )
    if offsets != expected_offsets or any(type(item) is not int for item in offsets):
        raise ValueError("training chunk global row offsets disagree with its manifest")
    game_ids = payload["game_ids"]
    if type(game_ids) is not list or game_ids != [expected_game_id] * row_count:
        raise ValueError("training chunk game IDs disagree with its manifest")
    raw_metrics = payload["metric_metadata"]
    if type(raw_metrics) is not list or len(raw_metrics) != row_count:
        raise ValueError("training chunk metric metadata has invalid batch alignment")
    metrics = tuple(
        _load_metric_metadata(
            item,
            candidate_count=int(batch.candidates.mask[index].sum().item()),
            policy_target=policy[index, : int(batch.candidates.mask[index].sum().item())],
        )
        for index, item in enumerate(raw_metrics)
    )
    return IndexedTrainingChunk(
        batch=batch,
        policy_targets=policy,
        value_targets=value_targets,
        global_row_offsets=tuple(offsets),
        game_ids=tuple(game_ids),
        metric_metadata=metrics,
    )


def _write_manifest(path: Path, manifest: IndexedDatasetManifest) -> None:
    with path.open("wb") as output:
        output.write(manifest.canonical_bytes())
        output.flush()
        os.fsync(output.fileno())


def _load_manifest(cache_dir: Path) -> IndexedDatasetManifest:
    payload = (cache_dir / _MANIFEST_NAME).read_bytes()
    manifest = IndexedDatasetManifest.model_validate_json(payload)
    if payload != manifest.canonical_bytes():
        raise ValueError("indexed dataset manifest is not canonical JSON")
    for game in manifest.games:
        _safe_fragment_path(cache_dir, game.fragment)
        for chunk in game.training_chunks:
            _safe_cache_file(cache_dir, chunk.path, kind="training chunk")
    return manifest


def _staging_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.staging"


@contextmanager
def _exclusive_build_lock(destination: Path, *, blocking: bool = True) -> Iterator[None]:
    """Prevent builders from sharing or deleting the same resumable stage."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as exc:
            raise RuntimeError(f"another index build is already running for {destination}") from exc
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _discard_staging(stage: Path) -> None:
    """Remove only the fixed sibling stage, without following a replaced symlink."""
    if stage.is_symlink() or stage.is_file():
        with suppress(FileNotFoundError):
            stage.unlink()
    else:
        with suppress(FileNotFoundError):
            shutil.rmtree(stage)


def _remove_staged_game_path(path: Path) -> None:
    """Remove an unfinished game's output without following a symlink."""
    if path.is_symlink() or path.is_file():
        with suppress(FileNotFoundError):
            path.unlink()
    else:
        with suppress(FileNotFoundError):
            shutil.rmtree(path)


def _write_staging_checkpoint(stage: Path, checkpoint: _TensorizationCheckpoint) -> None:
    payload = checkpoint.canonical_bytes()
    envelope = _TensorizationCheckpointEnvelope(
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
    )
    atomic_write_bytes(stage / _STAGING_CHECKPOINT_NAME, envelope.canonical_bytes())


def _load_staging_checkpoint(
    stage: Path,
    *,
    source_sha256: str,
    source_size: int,
    tensor_schema_digest: str,
    training_chunk_size: int,
) -> _TensorizationCheckpoint:
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("tensorization staging root must be a regular directory")
    output = stage / "index"
    fragments = output / "fragments"
    training = output / "training"
    if any(path.is_symlink() or not path.is_dir() for path in (output, fragments, training)):
        raise ValueError("tensorization staging index roots must be regular directories")
    payload = (stage / _STAGING_CHECKPOINT_NAME).read_bytes()
    envelope = _TensorizationCheckpointEnvelope.model_validate_json(payload)
    if payload != envelope.canonical_bytes():
        raise ValueError("tensorization staging checkpoint is not canonical JSON")
    checkpoint = envelope.checkpoint
    if (
        checkpoint.source_sha256 != source_sha256
        or checkpoint.source_size != source_size
        or checkpoint.tensor_schema_digest != tensor_schema_digest
        or checkpoint.training_chunk_format_version != _TRAINING_CHUNK_FORMAT_VERSION
        or checkpoint.metric_metadata_version != _METRIC_METADATA_VERSION
        or checkpoint.training_chunk_size != training_chunk_size
    ):
        raise ValueError("tensorization staging identity is stale or incompatible")
    for game in checkpoint.games:
        fragment = _safe_fragment_path(output, game.fragment)
        fragment_sha256, _ = _file_sha256(fragment)
        if fragment_sha256 != game.fragment_sha256:
            raise ValueError("tensorization staging fragment digest mismatch")
        for chunk in game.training_chunks:
            path = _safe_cache_file(output, chunk.path, kind="staging training chunk")
            chunk_sha256, _ = _file_sha256(path)
            if chunk_sha256 != chunk.sha256:
                raise ValueError("tensorization staging chunk digest mismatch")
    return checkpoint


def _remove_backup(path: Path) -> None:
    """Remove a replaced index backup without following a symlink."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _publish_directory(temporary: Path, destination: Path) -> None:
    """Replace a cache only after a complete sibling directory is durable."""
    if temporary.is_symlink() or not temporary.is_dir():
        raise ValueError("published index source must be a regular directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if destination.exists():
        backup = destination.parent / f".{destination.name}.{uuid4().hex}.old"
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        if backup is not None:
            os.replace(backup, destination)
            fsync_directory(destination.parent)
        raise
    if backup is not None:
        _remove_backup(backup)
        fsync_directory(destination.parent)


def _open_compatible_published_index(
    source: Path,
    destination: Path,
    *,
    training_chunk_size: int,
    source_identity: tuple[str, int] | None = None,
) -> IndexedJointDataset | None:
    try:
        manifest = _load_manifest(destination)
    except (OSError, ValueError):
        return None
    source_digest, source_size = source_identity or _file_sha256(source)
    schema = TensorFeatureSchema.current()
    if (
        manifest.source_sha256 != source_digest
        or manifest.source_size != source_size
        or manifest.tensor_schema_id != schema.schema_id
        or manifest.tensor_schema_version != schema.schema_version
        or manifest.tensor_schema_digest != schema.digest
        or manifest.training_chunk_format_version != _TRAINING_CHUNK_FORMAT_VERSION
        or manifest.metric_metadata_version != _METRIC_METADATA_VERSION
        or manifest.training_chunk_size != training_chunk_size
    ):
        return None
    return IndexedJointDataset._open(destination, source, manifest)


def build_indexed_dataset(
    source_path: str | Path,
    cache_dir: str | Path,
    *,
    show_progress: bool = False,
    training_chunk_size: int = DEFAULT_TRAINING_CHUNK_SIZE,
    index_workers: int = 1,
) -> IndexedJointDataset:
    """Exclusively build an atomic index, resuming validated phase-two staging."""
    if (
        isinstance(training_chunk_size, bool)
        or not isinstance(training_chunk_size, int)
        or training_chunk_size <= 0
    ):
        raise ValueError("training_chunk_size must be a positive integer")
    if isinstance(index_workers, bool) or not isinstance(index_workers, int) or index_workers <= 0:
        raise ValueError("index_workers must be a positive integer")
    source = Path(source_path)
    destination = Path(cache_dir)
    if destination.is_symlink():
        raise ValueError("indexed dataset destination must not be a symlink")
    with _exclusive_build_lock(destination):
        source_identity = _file_sha256(source)
        existing = _open_compatible_published_index(
            source,
            destination,
            training_chunk_size=training_chunk_size,
            source_identity=source_identity,
        )
        if existing is not None:
            return existing
        return _build_indexed_dataset_locked(
            source,
            destination,
            show_progress=show_progress,
            training_chunk_size=training_chunk_size,
            index_workers=index_workers,
            source_identity=source_identity,
        )


def _build_indexed_dataset_locked(
    source_path: str | Path,
    cache_dir: str | Path,
    *,
    show_progress: bool,
    training_chunk_size: int,
    index_workers: int,
    source_identity: tuple[str, int],
) -> IndexedJointDataset:
    """Build while the caller holds the destination's exclusive lock."""
    if (
        isinstance(training_chunk_size, bool)
        or not isinstance(training_chunk_size, int)
        or training_chunk_size <= 0
    ):
        raise ValueError("training_chunk_size must be a positive integer")
    if isinstance(index_workers, bool) or not isinstance(index_workers, int) or index_workers <= 0:
        raise ValueError("index_workers must be a positive integer")
    schema = TensorFeatureSchema.current()
    source = Path(source_path)
    destination = Path(cache_dir)
    source_digest, source_size = source_identity
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = _staging_path(destination)
    scan_complete = False
    published = False

    try:
        try:
            checkpoint = _load_staging_checkpoint(
                stage,
                source_sha256=source_digest,
                source_size=source_size,
                tensor_schema_digest=schema.digest,
                training_chunk_size=training_chunk_size,
            )
            scan_complete = True
        except (OSError, ValueError):
            _discard_staging(stage)
            stage.mkdir()
            fsync_directory(stage.parent)
            output = stage / "index"
            fragments = output / "fragments"
            fragments.mkdir(parents=True)
            (output / "training").mkdir()
            seen_games: set[str] = set()
            games: list[IndexedGameMetadata] = []
            active: _ActiveFragment | None = None
            dataset_digest = hashlib.sha256()
            row_count = 0

            # The source scan is intentionally all-or-nothing. Only its complete,
            # durable fragments become reusable phase-two staging state.
            try:
                with tqdm(
                    total=source_size,
                    desc="Scanning source",
                    unit="B",
                    unit_scale=True,
                    bar_format=TQDM_BAR_FORMAT,
                    mininterval=2.0,
                    disable=not show_progress,
                ) as progress:

                    def report_bytes(count: int) -> None:
                        progress.update(count)

                    records = iter_joint_dataset_records(source, on_bytes_read=report_bytes)
                    for row, raw_line in records:
                        if active is None or row.game_id != active.metadata.game_id:
                            if active is not None:
                                games.append(active.finish())
                                active = None
                            if row.game_id in seen_games:
                                raise ValueError(
                                    f"game {row.game_id!r} reappears non-contiguously "
                                    "in joint dataset"
                                )
                            seen_games.add(row.game_id)
                            metadata = _metadata_from_row(
                                row, ordinal=len(games), global_row_start=row_count
                            )
                            active = _ActiveFragment(output, metadata)
                        payload = raw_line + b"\n"
                        active.write(payload)
                        dataset_digest.update(payload)
                        row_count += 1
                if active is not None:
                    games.append(active.finish())
                    active = None
            finally:
                if active is not None:
                    active.discard()

            after_scan_digest, after_scan_size = _file_sha256(source)
            if (after_scan_digest, after_scan_size) != (source_digest, source_size):
                raise ValueError(
                    "joint dataset source changed while it was being scanned"
                ) from None
            checkpoint = _TensorizationCheckpoint(
                source_sha256=source_digest,
                source_size=source_size,
                dataset_digest=dataset_digest.hexdigest(),
                tensor_schema_id=schema.schema_id,
                tensor_schema_version=schema.schema_version,
                tensor_schema_digest=schema.digest,
                metric_metadata_version=_METRIC_METADATA_VERSION,
                training_chunk_size=training_chunk_size,
                row_count=row_count,
                game_count=len(games),
                games=tuple(games),
            )
            fsync_directory(fragments)
            fsync_directory(output / "training")
            fsync_directory(output)
            _write_staging_checkpoint(stage, checkpoint)
            scan_complete = True

        output = stage / "index"
        training = output / "training"
        games = list(checkpoint.games)
        missing = [ordinal for ordinal, game in enumerate(games) if not game.training_chunks]
        for ordinal in missing:
            _remove_staged_game_path(training / f"{ordinal:08d}")
        if missing:
            fsync_directory(training)

        # Workers only receive immutable fragment identities. The parent durably
        # checkpoints each successful game's chunk metadata before counting it done.
        spawn_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=index_workers,
            mp_context=spawn_context,
            initializer=_initialize_tensor_worker,
        ) as executor:
            futures = {
                executor.submit(
                    _tensorize_game_fragment,
                    str(output),
                    games[ordinal].fragment,
                    games[ordinal].game_id,
                    ordinal,
                    games[ordinal].global_row_start,
                    games[ordinal].row_count,
                    games[ordinal].fragment_sha256,
                    games[ordinal].semantic_digest,
                    training_chunk_size,
                    schema.digest,
                ): ordinal
                for ordinal in missing
            }
            with tqdm(
                total=len(games),
                initial=len(games) - len(missing),
                desc="Tensorizing index",
                unit="game",
                bar_format=TQDM_BAR_FORMAT,
                mininterval=2.0,
                disable=not show_progress,
            ) as progress:
                worker_failure: Exception | None = None
                try:
                    for future in as_completed(futures):
                        ordinal = futures[future]
                        try:
                            raw_chunks = future.result()
                        except Exception as exc:
                            if worker_failure is None:
                                worker_failure = exc
                                for pending in futures:
                                    pending.cancel()
                            continue
                        chunks = tuple(
                            IndexedTrainingChunkMetadata.model_validate(chunk)
                            for chunk in raw_chunks
                        )
                        games[ordinal] = games[ordinal].model_copy(
                            update={"training_chunks": chunks}
                        )
                        checkpoint = checkpoint.model_copy(update={"games": tuple(games)})
                        # Revalidation protects the checkpoint seam from malformed
                        # worker chunk paths or row ranges before it becomes durable.
                        checkpoint = _TensorizationCheckpoint.model_validate(
                            checkpoint.model_dump(mode="python")
                        )
                        fsync_directory(training / f"{ordinal:08d}")
                        fsync_directory(training)
                        _write_staging_checkpoint(stage, checkpoint)
                        progress.update(1)
                    if worker_failure is not None:
                        raise worker_failure
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise

        if any(not game.training_chunks for game in games):
            raise RuntimeError("tensor index worker completed without a result")
        after_digest, after_size = _file_sha256(source)
        if (after_digest, after_size) != (source_digest, source_size):
            raise ValueError("joint dataset source changed while its index was being built")

        manifest = IndexedDatasetManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            source_sha256=checkpoint.source_sha256,
            source_size=checkpoint.source_size,
            dataset_digest=checkpoint.dataset_digest,
            tensor_schema_id=checkpoint.tensor_schema_id,
            tensor_schema_version=checkpoint.tensor_schema_version,
            tensor_schema_digest=checkpoint.tensor_schema_digest,
            training_chunk_format_version=checkpoint.training_chunk_format_version,
            metric_metadata_version=checkpoint.metric_metadata_version,
            training_chunk_size=checkpoint.training_chunk_size,
            row_count=checkpoint.row_count,
            game_count=checkpoint.game_count,
            games=tuple(games),
        )
        _write_manifest(output / _MANIFEST_NAME, manifest)
        fsync_directory(output / "fragments")
        for game_directory in training.iterdir():
            fsync_directory(game_directory)
        fsync_directory(training)
        fsync_directory(output)
        _publish_directory(output, destination)
        published = True
        return IndexedJointDataset._open(destination, source, manifest)
    finally:
        if published or not scan_complete:
            _discard_staging(stage)
            if stage.parent.exists():
                fsync_directory(stage.parent)


def open_indexed_dataset(
    source_path: str | Path,
    cache_dir: str | Path,
    *,
    show_progress: bool = False,
    training_chunk_size: int = DEFAULT_TRAINING_CHUNK_SIZE,
    index_workers: int = 1,
) -> IndexedJointDataset:
    """Open a byte-identical cached index, or atomically build a replacement."""
    if (
        isinstance(training_chunk_size, bool)
        or not isinstance(training_chunk_size, int)
        or training_chunk_size <= 0
    ):
        raise ValueError("training_chunk_size must be a positive integer")
    if isinstance(index_workers, bool) or not isinstance(index_workers, int) or index_workers <= 0:
        raise ValueError("index_workers must be a positive integer")
    source = Path(source_path)
    destination = Path(cache_dir)
    if destination.is_symlink():
        raise ValueError("indexed dataset destination must not be a symlink")
    existing = _open_compatible_published_index(
        source, destination, training_chunk_size=training_chunk_size
    )
    if existing is not None:
        return existing
    return build_indexed_dataset(
        source,
        destination,
        show_progress=show_progress,
        training_chunk_size=training_chunk_size,
        index_workers=index_workers,
    )


__all__ = [
    "DEFAULT_TRAINING_CHUNK_SIZE",
    "INDEX_SCHEMA_VERSION",
    "IndexedDatasetManifest",
    "IndexedGameMetadata",
    "IndexedJointDataset",
    "IndexedTrainingChunk",
    "IndexedTrainingChunkMetadata",
    "TrainingMetricMetadata",
    "build_indexed_dataset",
    "open_indexed_dataset",
]
