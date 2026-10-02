"""Strict provenance receipts for normally completed native games.

Completion receipts are issued only by the recorder's opt-in controlled path.
They bind its live counters and normal decisive callback to exact published bytes;
valid native files without sidecars are deliberately not completion-certified.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from automata.models.contracts import canonical_json_bytes
from automata.training.native_dataset import (
    NativeGameIdentity,
    PolicyDatasetRecord,
    iter_native_game_records,
)

NATIVE_COMPLETION_CONTRACT: Literal["normal-decisive-native-game-v1"] = (
    "normal-decisive-native-game-v1"
)
_READ_SIZE = 128 * 1024


@dataclass(frozen=True)
class NativeCompletionTarget:
    """Filesystem target that opts one recorder into controlled publication."""

    source_root: Path
    receipt_path: Path


def _safe_logical_name(value: str) -> str:
    if type(value) is not str or not value:
        raise ValueError("completion logical name must be a nonempty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("completion logical name must be a normalized relative POSIX path")
    if not (value.endswith(".jsonl") or value.endswith(".jsonl.zst")):
        raise ValueError("completion logical name must identify a native JSONL file")
    return value


class NativeGameCompletionReceipt(BaseModel):
    """Canonical sidecar for one recorder-controlled, normally completed game."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    receipt_kind: Literal["RECORDER_COMPLETION"] = "RECORDER_COMPLETION"
    completion_contract: Literal["normal-decisive-native-game-v1"] = NATIVE_COMPLETION_CONTRACT
    logical_name: str
    game: NativeGameIdentity
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_size: StrictInt = Field(gt=0)
    row_count: StrictInt = Field(gt=0)
    policy_row_count: StrictInt = Field(ge=0)
    value_row_count: StrictInt = Field(ge=0)
    boundary_count: StrictInt = Field(ge=0)
    reason: Literal["game_over"]
    terminal_winner: Literal["RED", "BLUE"]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _valid_completion(self) -> NativeGameCompletionReceipt:
        _safe_logical_name(self.logical_name)
        if self.row_count != self.policy_row_count + self.value_row_count:
            raise ValueError("completion row count must equal its head counts")
        if (self.value_row_count == 0) != (self.boundary_count == 0):
            raise ValueError("completion boundary count must agree with value rows")
        if self.boundary_count > self.value_row_count:
            raise ValueError("completion boundary count cannot exceed value rows")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class NativeDatasetCompletionReceipt(BaseModel):
    """Ordered, nonempty collection of exact recorder completion sidecars."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    receipt_kind: Literal["RECORDER_COMPLETION_SET"] = "RECORDER_COMPLETION_SET"
    games: tuple[NativeGameCompletionReceipt, ...] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _valid_set(self) -> NativeDatasetCompletionReceipt:
        logical_names = tuple(game.logical_name for game in self.games)
        game_ids = tuple(game.game.game_id for game in self.games)
        if len(set(logical_names)) != len(logical_names):
            raise ValueError("completion logical names must be unique")
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("completion game IDs must be unique")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _require_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks")


def _load_canonical_receipt(
    path: str | Path,
    model: type[NativeGameCompletionReceipt] | type[NativeDatasetCompletionReceipt],
    *,
    label: str,
) -> NativeGameCompletionReceipt | NativeDatasetCompletionReceipt:
    source = Path(path)
    _require_no_symlink_components(source, label=label)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file")
    payload = source.read_bytes()
    try:
        receipt = model.model_validate_json(payload, strict=True)
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if payload != receipt.canonical_bytes():
        raise ValueError(f"{label} is not canonical JSON")
    return receipt


def load_native_game_completion_receipt(path: str | Path) -> NativeGameCompletionReceipt:
    """Load one strict canonical recorder sidecar from a regular file."""
    receipt = _load_canonical_receipt(
        path, NativeGameCompletionReceipt, label="native game completion receipt"
    )
    assert isinstance(receipt, NativeGameCompletionReceipt)
    return receipt


def load_native_dataset_completion_receipt(path: str | Path) -> NativeDatasetCompletionReceipt:
    """Load one strict canonical ordered completion set from a regular file."""
    receipt = _load_canonical_receipt(
        path, NativeDatasetCompletionReceipt, label="native dataset completion receipt"
    )
    assert isinstance(receipt, NativeDatasetCompletionReceipt)
    return receipt


def _safe_root(root: Path) -> Path:
    _require_no_symlink_components(root, label="native completion source root")
    if not root.is_dir():
        raise ValueError("native completion source root must be a regular directory")
    return root


def _source_path(root: Path, logical_name: str) -> Path:
    _safe_root(root)
    pure = PurePosixPath(_safe_logical_name(logical_name))
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("native completion source path must not contain symlinks")
    if not current.is_file():
        raise ValueError(f"native completion source file is missing: {logical_name}")
    return current


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_READ_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _strict_completion(
    completion: NativeDatasetCompletionReceipt,
) -> NativeDatasetCompletionReceipt:
    if not isinstance(completion, NativeDatasetCompletionReceipt):
        raise TypeError("completion must be a NativeDatasetCompletionReceipt")
    payload = completion.model_dump(mode="json", warnings=False)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return NativeDatasetCompletionReceipt.model_validate_json(encoded, strict=True)


def validate_native_dataset_completion(
    source_root: str | Path,
    completion: NativeDatasetCompletionReceipt,
) -> str:
    """Fully validate exact sources and return their ordered semantic row digest."""
    validated = _strict_completion(completion)
    root = _safe_root(Path(source_root))
    semantic = hashlib.sha256()
    for receipt in validated.games:
        path = _source_path(root, receipt.logical_name)
        before = _file_sha256(path)
        if before != (receipt.file_sha256, receipt.file_size):
            raise ValueError(
                f"native completion source hash or size mismatch: {receipt.logical_name}"
            )

        identity: NativeGameIdentity | None = None
        rows = policy = value = boundaries = 0
        last_boundary: int | None = None
        for record in iter_native_game_records(path):
            identity = record.game if identity is None else identity
            if record.game != receipt.game:
                raise ValueError("native completion source game identity mismatch")
            payload = canonical_json_bytes(record)
            semantic.update(payload + b"\n")
            rows += 1
            if isinstance(record, PolicyDatasetRecord):
                policy += 1
            else:
                value += 1
                if record.terminal_winner != receipt.terminal_winner:
                    raise ValueError("native completion terminal winner mismatch")
                expected_value = 1 if record.perspective_team == receipt.terminal_winner else -1
                if record.value_target != expected_value:
                    raise ValueError("native completion value label does not match its winner")
                if record.boundary.boundary_index != last_boundary:
                    boundaries += 1
                    last_boundary = record.boundary.boundary_index
        if identity != receipt.game:
            raise ValueError("native completion source game identity mismatch")
        if (rows, policy, value, boundaries) != (
            receipt.row_count,
            receipt.policy_row_count,
            receipt.value_row_count,
            receipt.boundary_count,
        ):
            raise ValueError("native completion source count mismatch")
        if _file_sha256(path) != before:
            raise ValueError(
                f"native completion source changed during validation: {receipt.logical_name}"
            )
    return semantic.hexdigest()


def create_native_dataset_completion_receipt(
    source_root: str | Path,
    receipt_paths: Sequence[str | Path],
) -> NativeDatasetCompletionReceipt:
    """Load only explicitly supplied sidecars and fully validate their exact sources."""
    if isinstance(receipt_paths, (str, bytes)) or not isinstance(receipt_paths, Sequence):
        raise TypeError("receipt_paths must be an explicit sequence")
    paths = tuple(receipt_paths)
    if not paths:
        raise ValueError("receipt_paths must contain at least one completion sidecar")
    completion = NativeDatasetCompletionReceipt(
        games=tuple(load_native_game_completion_receipt(path) for path in paths)
    )
    validate_native_dataset_completion(source_root, completion)
    return completion


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_completion_receipt(path: Path, receipt: NativeGameCompletionReceipt) -> os.stat_result:
    """Durably publish a canonical sidecar without replacing any existing entry."""
    temporary: Path | None = None
    published_identity: os.stat_result | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(receipt.canonical_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published_identity = path.stat(follow_symlinks=False)
        _fsync_directory(path.parent)
        return published_identity
    except BaseException:
        if published_identity is not None:
            _unlink_if_same_file(path, published_identity)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _unlink_if_same_file(path: Path, identity: os.stat_result) -> bool:
    """Best-effort rollback that never knowingly removes a replacement entry."""
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
        return False
    path.unlink()
    _fsync_directory(path.parent)
    return True


__all__ = [
    "NATIVE_COMPLETION_CONTRACT",
    "NativeCompletionTarget",
    "NativeDatasetCompletionReceipt",
    "NativeGameCompletionReceipt",
    "create_native_dataset_completion_receipt",
    "load_native_dataset_completion_receipt",
    "load_native_game_completion_receipt",
    "validate_native_dataset_completion",
]
