"""Strict, complete-game joint policy/value dataset contract.

Publication is intentionally a single-writer operation. A recorder owns one
new destination, buffers one game, and atomically publishes only a normal
terminal outcome. Parallel generation and append/merge locking belong at the
generation layer rather than this dataset contract.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Literal, cast

import zstandard
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, model_validator

from automata.models.contracts import CandidateID, DecisionObservation, canonical_json_bytes
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget

SCHEMA_VERSION: Literal[1] = 1
PolicySource = Literal["HEURISTIC", "ISMCTS_VISITS"]
TerminalWinner = Literal["RED", "BLUE"] | None

_IDENTITY_FIELDS = (
    "game_id",
    "world_seed",
    "map_id",
    "game_type",
    "red_composition",
    "blue_composition",
    "generation_id",
    "source_revision",
    "dirty_tree_hash",
    "source_model_digest",
    "search_config_id",
    "generator_config_id",
)


def _canonical_data(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def joint_decision_id(
    *,
    game_id: str,
    world_seed: int,
    map_id: str,
    game_type: str,
    red_composition: Sequence[str],
    blue_composition: Sequence[str],
    generation_id: str,
    source_revision: str,
    dirty_tree_hash: str,
    source_model_digest: str | None,
    search_config_id: str,
    generator_config_id: str,
    decision_index: int,
) -> str:
    """Return the stable digest of game provenance and decision position."""
    identity = {
        "game_id": game_id,
        "world_seed": world_seed,
        "map_id": map_id,
        "game_type": game_type,
        "red_composition": list(red_composition),
        "blue_composition": list(blue_composition),
        "generation_id": generation_id,
        "source_revision": source_revision,
        "dirty_tree_hash": dirty_tree_hash,
        "source_model_digest": source_model_digest,
        "search_config_id": search_config_id,
        "generator_config_id": generator_config_id,
        "decision_index": decision_index,
    }
    return hashlib.sha256(_canonical_data(identity)).hexdigest()


class JointDatasetRow(BaseModel):
    """One immutable, fully reconciled policy/value training decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    decision_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    game_id: str = Field(min_length=1)
    world_seed: StrictInt
    decision_index: StrictInt = Field(ge=0)
    perspective_team: Literal["RED", "BLUE"]
    observation: DecisionObservation
    policy_source: PolicySource
    policy_target: tuple[float, ...] = Field(min_length=1)
    selected_candidate_id: CandidateID
    selected_selection: JsonValue
    action_stats: tuple[SearchActionTarget, ...] | None = None
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
    value_target: Literal[-1, 0, 1]

    @model_validator(mode="after")
    def _validate_joint_alignment(self) -> JointDatasetRow:
        candidates = self.observation.candidates
        if len(self.policy_target) != len(candidates):
            raise ValueError("policy target must align exactly with observation candidates")
        if any(not math.isfinite(value) or value < 0.0 for value in self.policy_target):
            raise ValueError("policy target must contain finite non-negative values")
        if not math.isclose(sum(self.policy_target), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("policy target must sum to one")

        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.candidate_id == self.selected_candidate_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("selected candidate must be legal")
        if selected.selection != self.selected_selection:
            raise ValueError("selected candidate and exact engine selection must agree")

        if self.action_stats is not None:
            if tuple(action.candidate for action in self.action_stats) != candidates:
                raise ValueError("action statistics must align with candidate length and order")
            # Reuse the generic contract's duplicate and optional-distribution
            # checks while keeping the row representation as aligned stats.
            SearchPolicyTarget(schema_version=1, actions=self.action_stats)
            marked = tuple(
                action.candidate.candidate_id for action in self.action_stats if action.selected
            )
            if marked != (self.selected_candidate_id,):
                raise ValueError("action statistics must mark the exact selected candidate")

        if self.observation.state.viewer.perspective_team != self.perspective_team:
            raise ValueError("perspective team must agree with the nested observation")
        self._validate_observation_metadata()

        expected_value = (
            0
            if self.terminal_winner is None
            else (1 if self.terminal_winner == self.perspective_team else -1)
        )
        if self.value_target != expected_value:
            raise ValueError("value target must be perspective-correct for the terminal winner")

        expected_id = joint_decision_id(
            **{name: getattr(self, name) for name in _IDENTITY_FIELDS},
            decision_index=self.decision_index,
        )
        if self.decision_id != expected_id:
            raise ValueError("decision_id does not match game identity and decision index")
        return self

    def _validate_observation_metadata(self) -> None:
        global_tokens = [token for token in self.observation.state.tokens if token.kind == "GLOBAL"]
        if len(global_tokens) != 1:
            raise ValueError("observation must contain exactly one GLOBAL token")
        global_features = global_tokens[0].features
        if global_features.get("map_id") != self.map_id:
            raise ValueError("map metadata must agree with the nested observation")
        if global_features.get("game_type") != self.game_type:
            raise ValueError("game metadata must agree with the nested observation")

        compositions: dict[str, list[str]] = {"RED": [], "BLUE": []}
        for token in self.observation.state.tokens:
            if token.kind != "HERO":
                continue
            team = token.features.get("team_id")
            name = token.features.get("name")
            if team not in compositions or not isinstance(name, str) or not name:
                raise ValueError("hero tokens require public name and RED/BLUE team metadata")
            compositions[team].append(name)
        if tuple(compositions["RED"]) != self.red_composition:
            raise ValueError("red composition must agree with the nested observation")
        if tuple(compositions["BLUE"]) != self.blue_composition:
            raise ValueError("blue composition must agree with the nested observation")


class JointDatasetMetadata(BaseModel):
    """Canonical trainer-facing identity for one loaded dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: StrictInt = Field(ge=0)
    game_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JointDataset:
    """Validated rows plus deterministic identity and game grouping."""

    rows: tuple[JointDatasetRow, ...]
    digest: str
    metadata: JointDatasetMetadata
    rows_by_game: Mapping[str, tuple[JointDatasetRow, ...]]

    @property
    def game_ids(self) -> tuple[str, ...]:
        return self.metadata.game_ids

    def canonical_bytes(self) -> bytes:
        return b"".join(canonical_json_bytes(row) + b"\n" for row in self.rows)


def _is_compressed(path: Path) -> bool:
    return path.name.endswith(".jsonl.zst")


def write_joint_dataset(
    path: str | Path,
    rows: Iterable[JointDatasetRow],
    *,
    overwrite: bool = True,
) -> None:
    """Atomically write canonical rows, compressing ``.jsonl.zst`` destinations."""
    destination = Path(path)
    if not overwrite and destination.exists():
        raise FileExistsError(f"joint dataset destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            if _is_compressed(destination):
                compressor = zstandard.ZstdCompressor(
                    level=10,
                    threads=0,
                    write_checksum=True,
                )
                with compressor.stream_writer(handle, closefd=False) as writer:
                    for row in rows:
                        writer.write(canonical_json_bytes(row) + b"\n")
            else:
                for row in rows:
                    handle.write(canonical_json_bytes(row) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and destination.exists():
            raise FileExistsError(f"joint dataset destination already exists: {destination}")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    observation: DecisionObservation
    policy_source: PolicySource
    policy_target: tuple[float, ...]
    selected_candidate_id: CandidateID
    selected_selection: JsonValue
    action_stats: tuple[SearchActionTarget, ...] | None


class JointDatasetRecorder:
    """Buffer one complete game and publish it to a new JSONL destination."""

    def __init__(
        self,
        path: str | Path,
        *,
        game_id: str,
        world_seed: int,
        map_id: str,
        game_type: str,
        red_composition: Sequence[str],
        blue_composition: Sequence[str],
        generation_id: str,
        source_revision: str,
        dirty_tree_hash: str,
        source_model_digest: str | None,
        search_config_id: str,
        generator_config_id: str,
    ) -> None:
        self._path = Path(path)
        if self._path.exists():
            raise FileExistsError(f"joint dataset destination already exists: {self._path}")
        self._identity: dict[str, Any] = {
            "game_id": game_id,
            "world_seed": world_seed,
            "map_id": map_id,
            "game_type": game_type,
            "red_composition": tuple(red_composition),
            "blue_composition": tuple(blue_composition),
            "generation_id": generation_id,
            "source_revision": source_revision,
            "dirty_tree_hash": dirty_tree_hash,
            "source_model_digest": source_model_digest,
            "search_config_id": search_config_id,
            "generator_config_id": generator_config_id,
        }
        self._buffer: list[_PendingDecision] = []
        self._closed = False

    def record_decision(
        self,
        *,
        observation: DecisionObservation,
        policy_source: PolicySource,
        policy_target: Sequence[float],
        selected_candidate_id: CandidateID,
        selected_selection: JsonValue,
        action_stats: Sequence[SearchActionTarget] | None = None,
    ) -> None:
        """Buffer an immutable pre-decision observation and aligned target."""
        if self._closed:
            raise RuntimeError("joint dataset recorder is closed")
        pending = _PendingDecision(
            observation=observation.model_copy(deep=True),
            policy_source=policy_source,
            policy_target=tuple(float(value) for value in policy_target),
            selected_candidate_id=selected_candidate_id,
            selected_selection=selected_selection,
            action_stats=(
                tuple(action.model_copy(deep=True) for action in action_stats)
                if action_stats is not None
                else None
            ),
        )
        # Validate everything available before a terminal label without
        # weakening the completed-row contract.
        self._build_row(pending, len(self._buffer), terminal_winner=None)
        self._buffer.append(pending)

    def record_outcome(self, *, winner: str | None, rounds: int, reason: str) -> None:
        """Publish only a normal terminal game; all other outcomes are discarded."""
        del rounds
        if self._closed:
            raise RuntimeError("joint dataset recorder is closed")
        pending, self._buffer = self._buffer, []
        self._closed = True
        if reason != "game_over" or not pending:
            return
        if winner not in {None, "RED", "BLUE"}:
            raise ValueError("terminal winner must be RED, BLUE, or None for a draw")
        rows = tuple(
            self._build_row(item, index, terminal_winner=winner)
            for index, item in enumerate(pending)
        )
        self._publish(rows)

    def _build_row(
        self,
        pending: _PendingDecision,
        index: int,
        *,
        terminal_winner: str | None,
    ) -> JointDatasetRow:
        perspective = pending.observation.state.viewer.perspective_team
        if perspective not in {"RED", "BLUE"}:
            raise ValueError("joint decisions require a RED or BLUE perspective")
        value = 0 if terminal_winner is None else (1 if terminal_winner == perspective else -1)
        identity = self._identity
        typed_perspective = cast(Literal["RED", "BLUE"], perspective)
        typed_winner = cast(TerminalWinner, terminal_winner)
        typed_value = cast(Literal[-1, 0, 1], value)
        return JointDatasetRow(
            schema_version=SCHEMA_VERSION,
            decision_id=joint_decision_id(**identity, decision_index=index),
            **identity,
            decision_index=index,
            perspective_team=typed_perspective,
            observation=pending.observation,
            policy_source=pending.policy_source,
            policy_target=pending.policy_target,
            selected_candidate_id=pending.selected_candidate_id,
            selected_selection=pending.selected_selection,
            action_stats=pending.action_stats,
            terminal_winner=typed_winner,
            value_target=typed_value,
        )

    def _publish(self, rows: tuple[JointDatasetRow, ...]) -> None:
        write_joint_dataset(self._path, rows, overwrite=False)

    def close(self) -> None:
        """Discard an unfinished buffered game."""
        self._buffer.clear()
        self._closed = True

    def __enter__(self) -> JointDatasetRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()


def load_joint_dataset(path: str | Path) -> JointDataset:
    """Load strict canonical JSONL, rejecting truncation and cross-row conflicts."""
    source = Path(path)
    payload = source.read_bytes()
    if _is_compressed(source):
        try:
            with zstandard.ZstdDecompressor().stream_reader(
                io.BytesIO(payload), read_across_frames=True
            ) as reader:
                payload = reader.read()
        except zstandard.ZstdError as exc:
            raise ValueError(f"invalid compressed joint dataset: {exc}") from exc
    if payload and not payload.endswith(b"\n"):
        raise ValueError("joint dataset has a truncated final line")
    if not payload:
        raise ValueError("joint dataset is empty")

    rows: list[JointDatasetRow] = []
    seen_decisions: set[str] = set()
    identities: dict[str, tuple[object, ...]] = {}
    terminals: dict[str, tuple[TerminalWinner, int]] = {}
    seeds: dict[int, str] = {}
    grouped: dict[str, list[JointDatasetRow]] = {}
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line:
            raise ValueError(f"invalid joint row {line_number}: blank rows are prohibited")
        try:
            row = JointDatasetRow.model_validate_json(raw_line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid joint row {line_number}: {exc}") from exc
        if raw_line != canonical_json_bytes(row):
            raise ValueError(f"invalid joint row {line_number}: row is not canonical JSON")
        if row.decision_id in seen_decisions:
            raise ValueError(f"invalid joint row {line_number}: duplicate decision_id")
        seen_decisions.add(row.decision_id)

        identity = tuple(getattr(row, name) for name in _IDENTITY_FIELDS)
        if row.game_id in identities and identities[row.game_id] != identity:
            raise ValueError(f"invalid joint row {line_number}: conflicting game identity")
        identities[row.game_id] = identity
        terminal = (
            row.terminal_winner,
            row.value_target * (1 if row.perspective_team == "RED" else -1),
        )
        if row.game_id in terminals and terminals[row.game_id] != terminal:
            raise ValueError(f"invalid joint row {line_number}: conflicting terminal metadata")
        terminals[row.game_id] = terminal
        if row.world_seed in seeds and seeds[row.world_seed] != row.game_id:
            raise ValueError(
                f"invalid joint row {line_number}: world seed belongs to multiple games"
            )
        seeds[row.world_seed] = row.game_id
        rows.append(row)
        grouped.setdefault(row.game_id, []).append(row)

    for game_id, game_rows in grouped.items():
        if [row.decision_index for row in game_rows] != list(range(len(game_rows))):
            raise ValueError(f"game {game_id!r} decision indexes are not contiguous from zero")

    canonical = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    digest = hashlib.sha256(canonical).hexdigest()
    game_ids = tuple(grouped)
    immutable_groups = MappingProxyType(
        {game_id: tuple(game_rows) for game_id, game_rows in grouped.items()}
    )
    metadata = JointDatasetMetadata(
        schema_version=SCHEMA_VERSION,
        dataset_digest=digest,
        row_count=len(rows),
        game_ids=game_ids,
    )
    return JointDataset(tuple(rows), digest, metadata, immutable_groups)


__all__ = [
    "SCHEMA_VERSION",
    "JointDataset",
    "JointDatasetMetadata",
    "JointDatasetRecorder",
    "JointDatasetRow",
    "joint_decision_id",
    "load_joint_dataset",
    "write_joint_dataset",
]
