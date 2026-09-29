"""Native Gen1 records and strict, atomic publication of one complete game."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Annotated, Literal, cast

import zstandard
from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, model_validator

from automata.models.contracts import (
    DecisionObservation,
    LearnedObservation,
    StableValueObservation,
    canonical_json_bytes,
)
from automata.training.search_targets import SearchPolicyTarget

GameTeam = Literal["RED", "BLUE"]
TerminalWinner = GameTeam | None
BoundaryKind = Literal["ACTOR_READY", "PLANNING_READY"]
SampleKind = Literal["POLICY", "VALUE"]

_GAME_IDENTITY_FIELDS = (
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


def _namespaced_digest(namespace: str, values: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_data({"namespace": namespace, **values})).hexdigest()


def native_game_id(
    *,
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
) -> str:
    """Return the canonical ``native-game-v1`` identity digest."""
    if isinstance(world_seed, bool) or not isinstance(world_seed, int):
        raise TypeError("world_seed must be an integer")
    values: dict[str, object] = {
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
    }
    return _namespaced_digest("native-game-v1", values)


def native_sample_id(
    *, game_id: str, sample_kind: Literal["POLICY", "VALUE"], sample_index: int
) -> str:
    """Return the canonical ``native-sample-v1`` identity digest."""
    if sample_kind not in {"POLICY", "VALUE"}:
        raise ValueError("sample_kind must be POLICY or VALUE")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    return _namespaced_digest(
        "native-sample-v1",
        {
            "game_id": game_id,
            "sample_kind": sample_kind,
            "sample_index": sample_index,
        },
    )


class NativeGameIdentity(BaseModel):
    """Portable, public provenance that uniquely identifies one played game."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    game_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_seed: StrictInt
    map_id: str = Field(min_length=1)
    game_type: str = Field(min_length=1)
    red_composition: tuple[str, ...]
    blue_composition: tuple[str, ...]
    generation_id: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    dirty_tree_hash: str = Field(min_length=1)
    source_model_digest: str | None = Field(default=None, min_length=1)
    search_config_id: str = Field(min_length=1)
    generator_config_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_game_id(self) -> NativeGameIdentity:
        expected = native_game_id(**{name: getattr(self, name) for name in _GAME_IDENTITY_FIELDS})
        if self.game_id != expected:
            raise ValueError("game_id does not match the canonical native game identity")
        return self


class NativeBoundaryProvenance(BaseModel):
    """Public, observation-local identity for one retained value boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    boundary_index: StrictInt = Field(ge=0)
    kind: BoundaryKind
    round: StrictInt = Field(ge=1)
    turn: StrictInt = Field(ge=1)
    viewer_ref: str = Field(min_length=1)
    actor_ref: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _valid_actor_presence(self) -> NativeBoundaryProvenance:
        if self.kind == "ACTOR_READY" and self.actor_ref is None:
            raise ValueError("ACTOR_READY boundary requires an actor_ref")
        if self.kind == "PLANNING_READY" and self.actor_ref is not None:
            raise ValueError("PLANNING_READY boundary cannot carry an actor_ref")
        return self


def _public_hero_roster(observation: LearnedObservation) -> dict[str, tuple[str, GameTeam]]:
    """Stable public identities, independent of token order and local aliases."""
    roster: dict[str, tuple[str, GameTeam]] = {}
    for token in observation.tokens:
        if token.kind != "HERO":
            continue
        hero_id = token.features.get("hero_id")
        name = token.features.get("name")
        team = token.features.get("team_id")
        if (
            not isinstance(hero_id, str)
            or not hero_id
            or not isinstance(name, str)
            or not name
            or team not in ("RED", "BLUE")
        ):
            raise ValueError("HERO tokens require public hero_id, name and RED/BLUE team metadata")
        if hero_id in roster:
            raise ValueError("HERO tokens must have unique public hero IDs")
        roster[hero_id] = (name, cast(GameTeam, team))
    return roster


def _validate_observation_metadata(
    observation: LearnedObservation,
    *,
    game: NativeGameIdentity,
    boundary: NativeBoundaryProvenance | None = None,
) -> None:
    global_tokens = tuple(token for token in observation.tokens if token.kind == "GLOBAL")
    if len(global_tokens) != 1:
        raise ValueError("observation must contain exactly one GLOBAL token")
    global_features = global_tokens[0].features
    if global_features.get("map_id") != game.map_id:
        raise ValueError("map metadata must agree with the game identity")
    if global_features.get("game_type") != game.game_type:
        raise ValueError("game metadata must agree with the game identity")
    round_number = global_features.get("round")
    turn_number = global_features.get("turn")
    if type(round_number) is not int or round_number < 1:  # bool is not valid provenance
        raise ValueError("GLOBAL token must contain a positive integer round")
    if type(turn_number) is not int or turn_number < 1:
        raise ValueError("GLOBAL token must contain a positive integer turn")
    if boundary is not None and (round_number, turn_number) != (boundary.round, boundary.turn):
        raise ValueError("boundary round and turn must agree with the GLOBAL token")

    compositions: dict[str, list[str]] = {"RED": [], "BLUE": []}
    for name, team in _public_hero_roster(observation).values():
        compositions[team].append(name)
    # Setup order remains in the game identity: it can affect the starting world.
    # The encoder sorts tokens independently, so compare roster membership here.
    if sorted(compositions["RED"]) != sorted(game.red_composition):
        raise ValueError("red composition must agree with retained HERO rows")
    if sorted(compositions["BLUE"]) != sorted(game.blue_composition):
        raise ValueError("blue composition must agree with retained HERO rows")


def _validate_sample_identity(
    *, game: NativeGameIdentity, sample_kind: SampleKind, sample_index: int, sample_id: str
) -> None:
    expected = native_sample_id(
        game_id=game.game_id,
        sample_kind=sample_kind,
        sample_index=sample_index,
    )
    if sample_id != expected:
        raise ValueError("sample_id does not match game, kind, and sample index")


def _validate_perspective(observation: LearnedObservation, perspective_team: GameTeam) -> None:
    if observation.viewer.perspective_team != perspective_team:
        raise ValueError("perspective team must agree with the nested observation")


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


class PolicyDatasetRecord(BaseModel):
    """One actual decision paired only with its native ISMCTS policy evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    sample_kind: Literal["POLICY"] = "POLICY"
    game: NativeGameIdentity
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_index: StrictInt = Field(ge=0)
    policy_index: StrictInt = Field(ge=0)
    perspective_team: GameTeam
    observation: DecisionObservation
    policy_source: Literal["ISMCTS_VISITS"] = "ISMCTS_VISITS"
    target: SearchPolicyTarget

    @model_validator(mode="after")
    def _valid_policy_record(self) -> PolicyDatasetRecord:
        _validate_sample_identity(
            game=self.game,
            sample_kind=self.sample_kind,
            sample_index=self.sample_index,
            sample_id=self.sample_id,
        )
        _validate_perspective(self.observation.state, self.perspective_team)
        _validate_observation_metadata(self.observation.state, game=self.game)
        heroes = tuple(token for token in self.observation.state.tokens if token.kind == "HERO")
        viewers = tuple(token for token in heroes if token.features.get("relation") == "SELF")
        owners = tuple(token for token in heroes if token.features.get("is_decision_owner") is True)
        if len(viewers) != 1 or owners != viewers:
            raise ValueError("policy viewer must be the unique SELF HERO and decision owner")
        viewer = viewers[0]
        if (
            viewer.features["hero_id"] != self.observation.state.viewer.private_hero_id
            or viewer.features["team_id"] != self.perspective_team
        ):
            raise ValueError("policy viewer must agree with private hero and perspective team")

        actions = self.target.actions
        candidates = self.observation.candidates
        if not actions:
            raise ValueError("policy target must contain at least one action")
        if tuple(action.candidate for action in actions) != candidates:
            raise ValueError("policy target actions must match candidate length and exact order")
        if any(type(action.selected) is not bool for action in actions):
            raise ValueError("policy target selected markers must be booleans")
        if sum(action.selected for action in actions) != 1:
            raise ValueError("policy target must mark exactly one selected action")

        priors = tuple(action.prior_probability for action in actions)
        if not (
            all(value is None for value in priors) or all(value is not None for value in priors)
        ):
            raise ValueError("actual prior probabilities must be present for every action or none")
        if priors[0] is not None:
            typed_priors = cast(tuple[float, ...], priors)
            if any(
                not _is_finite_number(value) or not 0.0 <= value <= 1.0 for value in typed_priors
            ):
                raise ValueError("actual prior probabilities must be finite probabilities")
            if not math.isclose(sum(typed_priors), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("actual prior probabilities must sum to one")

        for action in actions:
            if (
                isinstance(action.sample_count, bool)
                or not isinstance(action.sample_count, int)
                or action.sample_count < 0
            ):
                raise ValueError("search visits must be non-negative integers")
            probability = action.improved_probability
            if probability is None:
                raise ValueError("improved probability must be present for every action")
            if not _is_finite_number(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("improved probabilities must be finite probabilities")
            if not _is_finite_number(action.mean_value) or not 0.0 <= action.mean_value <= 1.0:
                raise ValueError("mean values must remain actual search rewards in [0, 1]")
            if not _is_finite_number(action.value_variance) or action.value_variance < 0.0:
                raise ValueError("search reward variance must be finite and non-negative")
            if action.sample_count == 0 and (
                action.mean_value != 0.0 or action.value_variance != 0.0
            ):
                raise ValueError("unvisited actions must have zero mean and variance")

        total_visits = sum(action.sample_count for action in actions)
        if total_visits == 0:
            if len(actions) != 1 or actions[0].improved_probability != 1.0:
                raise ValueError("only a singleton zero-visit target may have probability one")
        else:
            for action in actions:
                expected = action.sample_count / total_visits
                if not math.isclose(
                    cast(float, action.improved_probability),
                    expected,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError("improved probabilities must match normalized visits")
        return self


class ValueDatasetRecord(BaseModel):
    """One actual stable boundary paired only with the played game's outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    sample_kind: Literal["VALUE"] = "VALUE"
    game: NativeGameIdentity
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_index: StrictInt = Field(ge=0)
    perspective_team: GameTeam
    boundary: NativeBoundaryProvenance
    observation: StableValueObservation
    terminal_winner: TerminalWinner
    value_target: Literal[-1, 0, 1]

    @model_validator(mode="after")
    def _valid_value_record(self) -> ValueDatasetRecord:
        _validate_sample_identity(
            game=self.game,
            sample_kind=self.sample_kind,
            sample_index=self.sample_index,
            sample_id=self.sample_id,
        )
        _validate_perspective(self.observation.state, self.perspective_team)
        _validate_observation_metadata(
            self.observation.state,
            game=self.game,
            boundary=self.boundary,
        )
        if self.observation.boundary_kind != self.boundary.kind:
            raise ValueError("observation and provenance boundary kinds must agree")

        hero_tokens = tuple(
            token for token in self.observation.state.tokens if token.kind == "HERO"
        )
        by_ref = {token.local_ref: token for token in hero_tokens}
        viewer = by_ref.get(self.boundary.viewer_ref)
        if viewer is None:
            raise ValueError("viewer_ref must be an observation-local HERO ref")
        self_heroes = tuple(
            token for token in hero_tokens if token.features.get("relation") == "SELF"
        )
        if self_heroes != (viewer,):
            raise ValueError("viewer_ref must identify the unique SELF HERO")
        if viewer.features.get("hero_id") != self.observation.state.viewer.private_hero_id:
            raise ValueError("viewer_ref must agree with viewer.private_hero_id")
        if viewer.features.get("team_id") != self.perspective_team:
            raise ValueError("viewer SELF HERO must agree with perspective team")

        current_actors = tuple(
            token for token in hero_tokens if token.features.get("is_current_actor") is True
        )
        decision_owners = tuple(
            token for token in hero_tokens if token.features.get("is_decision_owner") is True
        )
        if self.boundary.kind == "ACTOR_READY":
            actor = by_ref.get(cast(str, self.boundary.actor_ref))
            if actor is None:
                raise ValueError("actor_ref must be an observation-local HERO ref")
            if current_actors != (actor,) or decision_owners != (actor,):
                raise ValueError(
                    "ACTOR_READY actor_ref must be the unique current actor and decision owner"
                )
        elif current_actors or decision_owners:
            raise ValueError("PLANNING_READY cannot retain an actor or decision owner")

        expected_value = (
            0
            if self.terminal_winner is None
            else (1 if self.terminal_winner == self.perspective_team else -1)
        )
        if self.value_target != expected_value:
            raise ValueError("value target must be perspective-correct for the terminal winner")
        return self


NativeDatasetRecord = Annotated[
    PolicyDatasetRecord | ValueDatasetRecord,
    Field(discriminator="sample_kind"),
]
NATIVE_RECORD_ADAPTER: TypeAdapter[NativeDatasetRecord] = TypeAdapter(NativeDatasetRecord)


def _is_compressed(path: Path) -> bool:
    if path.name.endswith(".jsonl.zst"):
        return True
    if path.name.endswith(".jsonl"):
        return False
    raise ValueError("native game path must end in .jsonl or .jsonl.zst")


def _iter_json_lines(path: Path, *, compressed: bool | None = None) -> Iterator[bytes]:
    """Yield complete lines while accepting exactly one zstd frame, including spools."""
    if compressed is None:
        compressed = _is_compressed(path)
    try:
        with path.open("rb") as source:
            if not compressed:
                while raw_line := source.readline():
                    if not raw_line.endswith(b"\n"):
                        raise ValueError("native game stream has a truncated final JSONL row")
                    yield raw_line[:-1]
                return

            decompressor = zstandard.ZstdDecompressor().decompressobj()
            pending = b""
            while chunk := source.read(1024):
                output = decompressor.decompress(chunk)
                parts = (pending + output).split(b"\n")
                pending = parts.pop()
                yield from parts
                if decompressor.eof:
                    if decompressor.unused_data or source.read(1):
                        raise ValueError(
                            "invalid compressed native game stream: trailing data or zstd frame"
                        )
                    break
            if not decompressor.eof:
                raise ValueError("invalid compressed native game stream: truncated zstd frame")
            output = decompressor.flush()
            parts = (pending + output).split(b"\n")
            pending = parts.pop()
            yield from parts
            if pending:
                raise ValueError("native game stream has a truncated final JSONL row")
    except zstandard.ZstdError as exc:
        raise ValueError(f"invalid compressed native game stream: {exc}") from exc


class _OneGameValidator:
    """O(1)-state validation for one ordered native game stream."""

    def __init__(self) -> None:
        self.game: NativeGameIdentity | None = None
        self.roster: dict[str, tuple[str, GameTeam]] | None = None
        self.next_sample_index = 0
        self.next_policy_index = 0
        self.current_boundary_index: int | None = None
        self.current_boundary_signature: tuple[object, ...] | None = None
        self.boundary_viewers: set[str] = set()
        self.boundary_open = False
        self.terminal_winner: TerminalWinner | object = _UNSET

    def accept(self, record: NativeDatasetRecord) -> None:
        if self.game is None:
            self.game = record.game
        elif record.game != self.game:
            raise ValueError("native game stream must contain exactly one consistent game identity")
        roster = _public_hero_roster(record.observation.state)
        if self.roster is None:
            self.roster = roster
        elif roster != self.roster:
            raise ValueError("native game stream must retain one consistent public hero roster")
        if record.sample_index != self.next_sample_index:
            raise ValueError("native sample indexes must be contiguous from zero")
        self.next_sample_index += 1

        if isinstance(record, PolicyDatasetRecord):
            if record.policy_index != self.next_policy_index:
                raise ValueError("native policy indexes must be contiguous from zero")
            self.next_policy_index += 1
            self.boundary_open = False
            return

        self._accept_value(record)

    def _accept_value(self, record: ValueDatasetRecord) -> None:
        boundary = record.boundary
        hero_ids = {
            token.local_ref: cast(str, token.features["hero_id"])
            for token in record.observation.state.tokens
            if token.kind == "HERO"
        }
        actor_id = hero_ids[boundary.actor_ref] if boundary.actor_ref is not None else None
        viewer_id = hero_ids[boundary.viewer_ref]
        signature = (boundary.kind, boundary.round, boundary.turn, actor_id)
        if self.current_boundary_index is None:
            if boundary.boundary_index != 0:
                raise ValueError("native boundary indexes must be contiguous from zero")
            self._open_boundary(boundary.boundary_index, signature)
        elif self.boundary_open and boundary.boundary_index == self.current_boundary_index:
            if signature != self.current_boundary_signature:
                raise ValueError("rows in one boundary group must share boundary provenance")
        else:
            expected = self.current_boundary_index + 1
            if boundary.boundary_index != expected:
                raise ValueError("native boundary groups must be contiguous and ordered")
            self._open_boundary(boundary.boundary_index, signature)

        if viewer_id in self.boundary_viewers:
            raise ValueError("duplicate viewer in native boundary group")
        self.boundary_viewers.add(viewer_id)
        self.boundary_open = True

        if self.terminal_winner is _UNSET:
            self.terminal_winner = record.terminal_winner
        elif record.terminal_winner != self.terminal_winner:
            raise ValueError("native value rows have inconsistent terminal winner metadata")

    def _open_boundary(self, index: int, signature: tuple[object, ...]) -> None:
        self.current_boundary_index = index
        self.current_boundary_signature = signature
        self.boundary_viewers.clear()
        self.boundary_open = True

    def finish(self) -> None:
        if self.next_sample_index == 0:
            raise ValueError("native game stream is empty")


_UNSET = object()


def _strict_record_from_model(record: NativeDatasetRecord) -> NativeDatasetRecord:
    if type(record) not in {PolicyDatasetRecord, ValueDatasetRecord}:
        raise TypeError("native publication requires PolicyDatasetRecord or ValueDatasetRecord")
    # Round-trip through JSON so Pydantic revalidates nested model instances and
    # fields even when callers supplied model_copy/model_construct products.
    return NATIVE_RECORD_ADAPTER.validate_json(canonical_json_bytes(record), strict=True)


def iter_native_game_records(path: str | Path) -> Iterator[NativeDatasetRecord]:
    """Stream one canonical, nonempty, internally consistent native game file."""
    source = Path(path)
    validator = _OneGameValidator()
    for line_number, raw_line in enumerate(_iter_json_lines(source), 1):
        if not raw_line:
            raise ValueError(f"invalid native record {line_number}: blank rows are prohibited")
        if raw_line != raw_line.strip():
            raise ValueError(
                f"invalid native record {line_number}: surrounding whitespace is prohibited"
            )
        try:
            record = NATIVE_RECORD_ADAPTER.validate_json(raw_line, strict=True)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid native record {line_number}: {exc}") from exc
        if raw_line != canonical_json_bytes(record):
            raise ValueError(f"invalid native record {line_number}: non-canonical JSON")
        try:
            validator.accept(record)
        except ValueError as exc:
            raise ValueError(f"invalid native record {line_number}: {exc}") from exc
        yield record
    validator.finish()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_native_game(path: str | Path, records: Iterable[NativeDatasetRecord]) -> None:
    """Validate and atomically publish exactly one game without clobbering a peer."""
    destination = Path(path)
    compressed = _is_compressed(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            validator = _OneGameValidator()
            writer = None
            try:
                if compressed:
                    writer = zstandard.ZstdCompressor(
                        level=10,
                        threads=0,
                        write_checksum=True,
                    ).stream_writer(handle, closefd=False)
                output = writer if writer is not None else handle
                for supplied in records:
                    record = _strict_record_from_model(supplied)
                    validator.accept(record)
                    output.write(canonical_json_bytes(record) + b"\n")
                validator.finish()
            finally:
                if writer is not None:
                    writer.close()
            handle.flush()
            os.fsync(handle.fileno())

        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "NATIVE_RECORD_ADAPTER",
    "NativeBoundaryProvenance",
    "NativeDatasetRecord",
    "NativeGameIdentity",
    "PolicyDatasetRecord",
    "ValueDatasetRecord",
    "iter_native_game_records",
    "native_game_id",
    "native_sample_id",
    "publish_native_game",
]
