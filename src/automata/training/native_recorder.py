"""Bounded whole-game spooling for native policy and value records.

Policy decisions and real stable boundaries are serialized immediately, but no
rows become visible at the destination until a normal terminal outcome labels
the value samples and atomically publishes the complete game.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast

import zstandard
from pydantic import BaseModel, ConfigDict

from automata.models.contracts import (
    DecisionObservation,
    StableValueObservation,
    canonical_json_bytes,
)
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.outcomes import WinnerSide
from automata.runtime.value_boundary import StableValueBoundary, StableValueBoundaryKind
from automata.training.native_dataset import (
    NativeBoundaryProvenance,
    NativeDatasetRecord,
    NativeGameIdentity,
    PolicyDatasetRecord,
    ValueDatasetRecord,
    _iter_json_lines,
    native_sample_id,
    publish_native_game,
)
from automata.training.native_receipts import (
    NativeCompletionTarget,
    NativeDatasetCompletionReceipt,
    NativeGameCompletionReceipt,
    _file_sha256,
    _publish_completion_receipt,
    _require_no_symlink_components,
    _unlink_if_same_file,
    validate_native_dataset_completion,
)
from automata.training.search_targets import SearchPolicyTarget
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID


class _PendingValue(BaseModel):
    """A boundary sample awaiting a real terminal label."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    pending_kind: Literal["VALUE"] = "VALUE"
    sample_id: str
    sample_index: int
    perspective_team: Literal["RED", "BLUE"]
    boundary: NativeBoundaryProvenance
    observation: StableValueObservation


class NativeDatasetRecorder:
    """Spool one live game and publish it only after normal completion.

    The recorder is a :class:`StableBoundaryObserver`. Policy recording is an
    explicit outer-strategy call, not a decision-observer callback, so search
    leaves and unsearched decisions cannot enter the native dataset by accident.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        game: NativeGameIdentity,
        completion_target: NativeCompletionTarget | None = None,
    ) -> None:
        self._path = Path(path)
        self._completion_target = (
            self._anchored_completion_target(completion_target)
            if completion_target is not None
            else None
        )
        if self._completion_target is not None:
            # Raw mode intentionally retains its historical relative-path behavior.
            self._path = Path(os.path.abspath(self._path))
        self._completion_receipt: NativeGameCompletionReceipt | None = None
        # Revalidate inputs before creating any filesystem state.
        self._game = NativeGameIdentity.model_validate_json(canonical_json_bytes(game))
        if not (self._path.name.endswith(".jsonl") or self._path.name.endswith(".jsonl.zst")):
            raise ValueError("native dataset destination must end in .jsonl or .jsonl.zst")
        if self._path.exists():
            raise FileExistsError(f"native dataset destination already exists: {self._path}")
        if self._completion_target is not None:
            self._validate_completion_paths(self._completion_target)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._completion_target is not None:
            self._completion_target.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        spool = tempfile.NamedTemporaryFile(  # noqa: SIM115 - recorder lifetime owns it
            mode="w+b",
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".pending.zst",
            delete=False,
        )
        self._spool_path = Path(spool.name)
        if self._completion_target is not None and self._same_path(
            self._spool_path, self._completion_target.receipt_path
        ):
            spool.close()
            self._spool_path.unlink(missing_ok=True)
            raise ValueError("native completion source, receipt, and spool paths must not collide")
        self._spool_file: Any | None = spool
        compressor = zstandard.ZstdCompressor(level=3, threads=0, write_checksum=True)
        self._spool_writer: Any | None = compressor.stream_writer(spool, closefd=False)
        self._sample_index = 0
        self._policy_index = 0
        self._boundary_index = 0
        self._closed = False

    @property
    def completion_receipt(self) -> NativeGameCompletionReceipt | None:
        """Return the issued controlled receipt, or ``None`` until it exists."""
        return self._completion_receipt

    def record_policy(
        self,
        *,
        observation: DecisionObservation,
        target: SearchPolicyTarget,
        perspective_team: Literal["RED", "BLUE"],
    ) -> None:
        """Validate and immediately spool one immutable root-search sample."""
        self._require_open()
        try:
            record = PolicyDatasetRecord(
                game=self._game,
                sample_id=native_sample_id(
                    game_id=self._game.game_id,
                    sample_kind="POLICY",
                    sample_index=self._sample_index,
                ),
                sample_index=self._sample_index,
                policy_index=self._policy_index,
                perspective_team=perspective_team,
                observation=observation,
                target=target,
            )
            # Round-tripping before append both freezes mutable nested JSON and
            # ensures the exact bytes in the spool are schema-valid now.
            payload = canonical_json_bytes(record)
            frozen = PolicyDatasetRecord.model_validate_json(payload)
            self._append_payloads((canonical_json_bytes(frozen),))
            self._sample_index += 1
            self._policy_index += 1
        except BaseException:
            self._poison()
            raise

    def record_boundary(
        self,
        state: GameState,
        boundary: StableValueBoundary,
        *,
        viewer_hero_ids: tuple[str, ...],
    ) -> None:
        """Encode every entitled viewer atomically at one real live boundary."""
        self._require_open()
        try:
            if tuple(sorted(set(viewer_hero_ids))) != viewer_hero_ids:
                raise ValueError("boundary viewers must be sorted and unique")
            if not viewer_hero_ids:
                return
            self._validate_live_game_identity(state)

            pending: list[_PendingValue] = []
            for offset, viewer_id in enumerate(viewer_hero_ids):
                hero = state.get_hero(HeroID(viewer_id))
                if hero is None or hero.team is None:
                    raise ValueError(f"unknown or teamless boundary viewer {viewer_id!r}")
                if hero.team not in {TeamColor.RED, TeamColor.BLUE}:
                    raise ValueError(f"unsupported boundary viewer team {hero.team.value!r}")
                observation = encode_stable_value(
                    state,
                    boundary,
                    viewer_hero_id=viewer_id,
                    perspective_team=hero.team,
                )
                viewer_ref, actor_ref = self._local_boundary_refs(observation, boundary)
                provenance = NativeBoundaryProvenance(
                    boundary_index=self._boundary_index,
                    kind=boundary.kind.value,
                    round=boundary.round,
                    turn=boundary.turn,
                    viewer_ref=viewer_ref,
                    actor_ref=actor_ref,
                )
                sample_index = self._sample_index + offset
                item = _PendingValue(
                    sample_id=native_sample_id(
                        game_id=self._game.game_id,
                        sample_kind="VALUE",
                        sample_index=sample_index,
                    ),
                    sample_index=sample_index,
                    perspective_team=hero.team.value,
                    boundary=provenance,
                    observation=observation,
                )
                payload = canonical_json_bytes(item)
                pending.append(_PendingValue.model_validate_json(payload))

            # Nothing is appended until every viewer has encoded and validated.
            self._append_payloads(tuple(canonical_json_bytes(item) for item in pending))
            self._sample_index += len(pending)
            self._boundary_index += 1
        except BaseException:
            self._poison()
            raise

    def record_outcome(
        self,
        *,
        winner_side: WinnerSide | None,
        rounds: int,
        reason: str,
    ) -> None:
        """Publish a normal decisive terminal game; discard all other games."""
        del rounds
        self._require_open()
        try:
            if winner_side not in {None, "RED", "BLUE"}:
                raise ValueError("winner_side must be RED, BLUE, or None")
            if reason == "game_over":
                if winner_side is None:
                    raise ValueError("game_over requires a winner; the engine has no draw rule")
            elif winner_side is not None:
                raise ValueError("nonterminal outcome must have winner_side=None")

            self._closed = True
            self._close_spool_writer()
            if reason != "game_over" or self._sample_index == 0:
                self._spool_path.unlink(missing_ok=True)
                return

            typed_winner = cast(Literal["RED", "BLUE"], winner_side)
            if self._completion_target is not None:
                self._revalidate_completion_paths_for_publication(self._completion_target)
            publish_native_game(
                self._path,
                self._completed_records(terminal_winner=typed_winner),
            )
            if self._completion_target is not None:
                self._issue_completion_receipt(terminal_winner=typed_winner)
            self._spool_path.unlink(missing_ok=True)
        except BaseException:
            self._poison()
            raise

    def _issue_completion_receipt(self, *, terminal_winner: Literal["RED", "BLUE"]) -> None:
        target = self._completion_target
        if target is None:  # pragma: no cover - caller guards the raw path
            return
        published_identity = self._path.stat(follow_symlinks=False)
        try:
            file_sha256, file_size = _file_sha256(self._path)
            receipt = NativeGameCompletionReceipt(
                logical_name=self._controlled_logical_name(target),
                game=self._game,
                file_sha256=file_sha256,
                file_size=file_size,
                row_count=self._sample_index,
                policy_row_count=self._policy_index,
                value_row_count=self._sample_index - self._policy_index,
                boundary_count=self._boundary_index,
                reason="game_over",
                terminal_winner=terminal_winner,
            )
            completion = NativeDatasetCompletionReceipt(games=(receipt,))
            validate_native_dataset_completion(target.source_root, completion)
            _publish_completion_receipt(target.receipt_path, receipt)
        except BaseException:
            _unlink_if_same_file(self._path, published_identity)
            raise
        self._completion_receipt = receipt

    def _completed_records(
        self, *, terminal_winner: Literal["RED", "BLUE"]
    ) -> Iterator[NativeDatasetRecord]:
        sample_count = policy_count = boundary_count = 0
        last_boundary_index: int | None = None
        for line_number, raw_line in enumerate(self._iter_spool_lines(), 1):
            try:
                data = json.loads(raw_line)
                if not isinstance(data, dict):
                    raise ValueError("pending row must be an object")
                if data.get("sample_kind") == "POLICY":
                    record: NativeDatasetRecord = PolicyDatasetRecord.model_validate_json(raw_line)
                    policy_count += 1
                elif data.get("pending_kind") == "VALUE":
                    pending = _PendingValue.model_validate_json(raw_line)
                    if pending.boundary.boundary_index != last_boundary_index:
                        boundary_count += 1
                        last_boundary_index = pending.boundary.boundary_index
                    value = 1 if pending.perspective_team == terminal_winner else -1
                    record = ValueDatasetRecord(
                        game=self._game,
                        sample_id=pending.sample_id,
                        sample_index=pending.sample_index,
                        perspective_team=pending.perspective_team,
                        boundary=pending.boundary,
                        observation=pending.observation,
                        terminal_winner=terminal_winner,
                        value_target=cast(Literal[-1, 1], value),
                    )
                else:
                    raise ValueError("unknown pending native row kind")
                expected = (
                    canonical_json_bytes(record)
                    if isinstance(record, PolicyDatasetRecord)
                    else canonical_json_bytes(_PendingValue.model_validate_json(raw_line))
                )
                if raw_line != expected:
                    raise ValueError("non-canonical JSON")
            except (ValueError, TypeError, UnicodeDecodeError) as exc:
                raise ValueError(f"invalid pending native row {line_number}: {exc}") from exc
            sample_count += 1
            yield record
        if (sample_count, policy_count, boundary_count) != (
            self._sample_index,
            self._policy_index,
            self._boundary_index,
        ):
            raise ValueError("pending native spool sample counts do not match the recorded game")

    def _iter_spool_lines(self) -> Iterator[bytes]:
        yield from _iter_json_lines(self._spool_path, compressed=True)

    def _append_payloads(self, payloads: Sequence[bytes]) -> None:
        writer = self._spool_writer
        if writer is None:
            raise RuntimeError("native dataset recorder spool is closed")
        for payload in payloads:
            writer.write(payload + b"\n")
            writer.flush(zstandard.FLUSH_BLOCK)

    def _validate_live_game_identity(self, state: GameState) -> None:
        game = self._game
        if state.rng_seed is not None and state.rng_seed != game.world_seed:
            raise ValueError("world seed does not match native game identity")
        if state.board.map_id != game.map_id:
            raise ValueError("map does not match native game identity")
        if state.game_type.value != game.game_type:
            raise ValueError("game type does not match native game identity")

        compositions: dict[TeamColor, tuple[str, ...]] = {}
        for team in (TeamColor.RED, TeamColor.BLUE):
            roster = state.teams.get(team)
            if roster is None:
                raise ValueError(f"game state has no {team.value} roster")
            compositions[team] = tuple(hero.name for hero in roster.heroes)
        if compositions[TeamColor.RED] != game.red_composition:
            raise ValueError("red composition does not match native game identity")
        if compositions[TeamColor.BLUE] != game.blue_composition:
            raise ValueError("blue composition does not match native game identity")

    @staticmethod
    def _local_boundary_refs(
        observation: StableValueObservation,
        boundary: StableValueBoundary,
    ) -> tuple[str, str | None]:
        heroes = tuple(token for token in observation.state.tokens if token.kind == "HERO")
        viewers = tuple(token for token in heroes if token.features.get("relation") == "SELF")
        if len(viewers) != 1:
            raise ValueError("value observation must identify exactly one SELF hero")
        viewer_ref = viewers[0].local_ref

        current = tuple(token for token in heroes if token.features.get("is_current_actor") is True)
        owners = tuple(token for token in heroes if token.features.get("is_decision_owner") is True)
        if boundary.kind is StableValueBoundaryKind.PLANNING_READY:
            if current or owners:
                raise ValueError("planning boundary must not identify an actor or decision owner")
            return viewer_ref, None
        if len(current) != 1 or len(owners) != 1 or current[0].local_ref != owners[0].local_ref:
            raise ValueError("actor boundary must identify one matching actor and decision owner")
        return viewer_ref, current[0].local_ref

    @staticmethod
    def _anchored_completion_target(target: NativeCompletionTarget) -> NativeCompletionTarget:
        if not isinstance(target, NativeCompletionTarget):
            raise TypeError("completion_target must be a NativeCompletionTarget")
        if not isinstance(target.source_root, Path) or not isinstance(target.receipt_path, Path):
            raise TypeError("native completion target paths must be pathlib.Path values")
        return NativeCompletionTarget(
            source_root=Path(os.path.abspath(target.source_root)),
            receipt_path=Path(os.path.abspath(target.receipt_path)),
        )

    @staticmethod
    def _same_path(left: Path, right: Path) -> bool:
        return Path(os.path.abspath(left)) == Path(os.path.abspath(right))

    @staticmethod
    def _paths_overlap(left: Path, right: Path) -> bool:
        left = left.resolve(strict=False)
        right = right.resolve(strict=False)
        return left == right or left in right.parents or right in left.parents

    @staticmethod
    def _safe_parent_chain(path: Path, *, label: str) -> None:
        current = path.parent
        while not current.exists():
            if current.is_symlink():
                raise ValueError(f"native completion {label} path must not contain symlinks")
            current = current.parent
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"native completion {label} parent must lead from a regular directory")
        while current != current.parent:
            if current.is_symlink():
                raise ValueError(f"native completion {label} path must not contain symlinks")
            current = current.parent

    def _controlled_logical_name(self, target: NativeCompletionTarget) -> str:
        root = Path(os.path.abspath(target.source_root))
        source = Path(os.path.abspath(self._path))
        try:
            relative = source.relative_to(root)
        except ValueError as exc:
            raise ValueError("native completion source must remain inside its source root") from exc
        return relative.as_posix()

    def _validate_completion_path_topology(self, target: NativeCompletionTarget) -> None:
        root = target.source_root
        _require_no_symlink_components(root, label="native completion source root")
        if not root.is_dir():
            raise ValueError("native completion source root must be a regular directory")
        _require_no_symlink_components(self._path, label="native completion source")
        _require_no_symlink_components(target.receipt_path, label="native completion receipt")
        if self._paths_overlap(self._path, target.receipt_path):
            raise ValueError("native completion source and receipt paths overlap")
        logical_name = self._controlled_logical_name(target)
        current = root
        for part in Path(logical_name).parts[:-1]:
            current /= part
            if current.is_symlink():
                raise ValueError("native completion source path must not contain symlinks")
            if current.exists() and not current.is_dir():
                raise ValueError("native completion source parent must be a directory")
        self._safe_parent_chain(target.receipt_path, label="receipt")

    def _validate_completion_paths(self, target: NativeCompletionTarget) -> None:
        self._validate_completion_path_topology(target)
        if target.receipt_path.exists():
            raise FileExistsError(
                f"native completion receipt already exists: {target.receipt_path}"
            )

    def _revalidate_completion_paths_for_publication(self, target: NativeCompletionTarget) -> None:
        self._validate_completion_path_topology(target)
        if not self._path.parent.is_dir() or not target.receipt_path.parent.is_dir():
            raise ValueError("native completion publication parents must remain directories")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("native dataset recorder is closed")

    def _close_spool_writer(self) -> None:
        writer, self._spool_writer = self._spool_writer, None
        spool, self._spool_file = self._spool_file, None
        try:
            if writer is not None:
                writer.close()
        finally:
            if spool is not None:
                try:
                    spool.flush()
                    os.fsync(spool.fileno())
                finally:
                    spool.close()

    def _poison(self) -> None:
        self._closed = True
        writer, self._spool_writer = self._spool_writer, None
        spool, self._spool_file = self._spool_file, None
        if writer is not None:
            with suppress(BaseException):
                writer.close()
        if spool is not None:
            with suppress(BaseException):
                spool.close()
        with suppress(BaseException):
            self._spool_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Idempotently discard an unfinished game."""
        if self._closed:
            return
        self._poison()

    def __enter__(self) -> NativeDatasetRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()


__all__ = ["NativeDatasetRecorder"]
