"""Atomic persistent catalog of validated native complete-game replay references."""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from automata.models.contracts import (
    CURRENT_MAP_SCHEMA_VERSION,
    GEN1_RUNTIME_COMPATIBILITY_VERSION,
    canonical_json_bytes,
)
from automata.models.shared_encoder.artifacts import Gen1ModelArtifactManifest
from automata.observation.hero_adapters import HeroObservationAdapterRegistry
from automata.training.io import atomic_write_bytes, fsync_directory
from automata.training.native_dataset import NativeGameIdentity
from automata.training.native_indexed_dataset import (
    INDEX_SCHEMA_VERSION,
    IndexedNativeDataset,
    NativeIndexedDatasetManifest,
    create_native_source_receipt_from_completions,
)
from automata.training.native_receipts import (
    NativeDatasetCompletionReceipt,
    validate_native_dataset_completion,
)
from automata.training.native_splits import (
    NativeSeedSplitLedger,
    NativeSplitConfig,
    create_native_split_ledger,
    extend_native_split_ledger,
)
from goa2.data.heroes import HeroRegistry
from goa2.domain.models import GameType

_SELECTION_RECIPE: Literal["uniform-train-games-v1"] = "uniform-train-games-v1"
_MODEL_ID: Literal["goa2-gen1-policy-stable-value-v1"] = "goa2-gen1-policy-stable-value-v1"
_VALUE_SEMANTICS: Literal["stable-boundary-outcome-v1"] = "stable-boundary-outcome-v1"
_LOCK_CONTENT = b"automata-native-replay-lock-v1\n"
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class NativeReplayConfig(_FrozenModel):
    """Complete immutable replay recipe; no experiment defaults are inherited."""

    schema_version: Literal[1] = 1
    selection_recipe: Literal["uniform-train-games-v1"] = _SELECTION_RECIPE
    split_config: NativeSplitConfig
    max_games: StrictInt | None = Field(default=None, gt=0)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value


class NativeReplayCompatibility(_FrozenModel):
    """Executable native record, tensor, model, and value contract identity."""

    native_record_schema_version: Literal[1] = 1
    decision_tensor_schema_id: str = Field(min_length=1)
    decision_tensor_schema_version: StrictInt = Field(gt=0)
    decision_tensor_schema_digest: str = Field(pattern=_DIGEST_PATTERN)
    stable_value_tensor_schema_id: str = Field(min_length=1)
    stable_value_tensor_schema_version: StrictInt = Field(gt=0)
    stable_value_tensor_schema_digest: str = Field(pattern=_DIGEST_PATTERN)
    model_id: Literal["goa2-gen1-policy-stable-value-v1"] = _MODEL_ID
    value_semantics: Literal["stable-boundary-outcome-v1"] = _VALUE_SEMANTICS

    @field_validator("native_record_schema_version", mode="before")
    @classmethod
    def _strict_record_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("native record schema version must be the integer 1")
        return value


class NativeReplayGameRef(_FrozenModel):
    """A retained whole-game reference; it contains no rows or tensors."""

    game: NativeGameIdentity
    dataset_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_digest: str = Field(pattern=_DIGEST_PATTERN)
    completion_receipt_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_logical_name: str = Field(min_length=1)
    policy_row_count: StrictInt = Field(ge=0)
    value_row_count: StrictInt = Field(ge=0)
    boundary_count: StrictInt = Field(ge=0)
    insertion_index: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _valid_reference(self) -> NativeReplayGameRef:
        _safe_source_logical_name(self.source_logical_name)
        if self.policy_row_count + self.value_row_count == 0:
            raise ValueError("replay game reference must contribute at least one row")
        if (self.value_row_count == 0) != (self.boundary_count == 0):
            raise ValueError("replay boundary count must agree with value rows")
        if self.boundary_count > self.value_row_count:
            raise ValueError("replay boundary count cannot exceed value rows")
        return self


class NativeReplayGeneration(_FrozenModel):
    """Immutable enrollment history, including validation-only and evicted games."""

    generation_id: str = Field(min_length=1)
    source_model_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    dataset_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_digest: str = Field(pattern=_DIGEST_PATTERN)
    completion_receipt_digest: str = Field(pattern=_DIGEST_PATTERN)
    all_game_ids: tuple[str, ...] = Field(min_length=1)
    replay_game_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _valid_game_ids(self) -> NativeReplayGeneration:
        if len(set(self.all_game_ids)) != len(self.all_game_ids):
            raise ValueError("generation game IDs must be unique")
        if len(set(self.replay_game_ids)) != len(self.replay_game_ids):
            raise ValueError("generation replay game IDs must be unique")
        if any(not _is_digest(game_id) for game_id in self.all_game_ids):
            raise ValueError("generation game IDs must be SHA256 digests")
        if not set(self.replay_game_ids) <= set(self.all_game_ids):
            raise ValueError("generation replay game IDs must belong to its complete inventory")
        return self


def _validated_split_assignments(
    ledger: NativeSeedSplitLedger,
) -> tuple[NativeSeedSplitLedger, dict[int, str]]:
    validated = NativeSeedSplitLedger.model_validate(ledger.model_dump(mode="python"), strict=True)
    return validated, {
        assignment.world_seed: assignment.split for assignment in validated.assignments
    }


class NativeReplayCatalog(_FrozenModel):
    """Canonical append-only replay history plus capacity-bounded retained references."""

    schema_version: Literal[1] = 1
    config: NativeReplayConfig
    compatibility: NativeReplayCompatibility | None
    split_ledger: NativeSeedSplitLedger
    generations: tuple[NativeReplayGeneration, ...]
    games: tuple[NativeReplayGameRef, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _valid_catalog(self) -> NativeReplayCatalog:
        validated_ledger, split_by_seed = _validated_split_assignments(self.split_ledger)
        if validated_ledger.config != self.config.split_config:
            raise ValueError("catalog split ledger does not match its replay config")
        if bool(self.generations) != (self.compatibility is not None):
            raise ValueError("catalog compatibility must exist exactly when history exists")

        generation_ids = tuple(item.generation_id for item in self.generations)
        dataset_digests = tuple(item.dataset_digest for item in self.generations)
        if len(set(generation_ids)) != len(generation_ids):
            raise ValueError("catalog generation IDs must remain unique across history")
        if len(set(dataset_digests)) != len(dataset_digests):
            raise ValueError("catalog dataset digests must remain unique across history")

        all_game_ids = tuple(
            game_id for generation in self.generations for game_id in generation.all_game_ids
        )
        if len(set(all_game_ids)) != len(all_game_ids):
            raise ValueError("catalog game IDs must remain unique across complete history")
        replay_game_ids = tuple(
            game_id for generation in self.generations for game_id in generation.replay_game_ids
        )
        by_game_id: dict[str, tuple[NativeReplayGeneration, int]] = {}
        insertion_index = 0
        for generation in self.generations:
            for game_id in generation.replay_game_ids:
                by_game_id[game_id] = (generation, insertion_index)
                insertion_index += 1
        retained_count = len(replay_game_ids)
        if self.config.max_games is not None:
            retained_count = min(retained_count, self.config.max_games)
        expected_retained_ids = replay_game_ids[len(replay_game_ids) - retained_count :]
        actual_retained_ids = tuple(item.game.game_id for item in self.games)
        if actual_retained_ids != expected_retained_ids:
            raise ValueError(
                "retained replay references must be the capacity-bounded history suffix"
            )

        if len(set(actual_retained_ids)) != len(actual_retained_ids):
            raise ValueError("retained replay game references must be unique")
        for item in self.games:
            generation, insertion_index = by_game_id[item.game.game_id]
            if (
                item.game.generation_id != generation.generation_id
                or item.game.source_model_digest != generation.source_model_digest
                or item.dataset_digest != generation.dataset_digest
                or item.source_digest != generation.source_digest
                or item.completion_receipt_digest != generation.completion_receipt_digest
                or item.insertion_index != insertion_index
            ):
                raise ValueError("retained replay reference does not match generation history")
            if split_by_seed.get(item.game.world_seed) != "train":
                raise ValueError("retained replay reference is not assigned to the train split")
        return self

    def _revalidated(self) -> NativeReplayCatalog:
        return NativeReplayCatalog.model_validate(self.model_dump(mode="python"), strict=True)

    def canonical_bytes(self) -> bytes:
        """Return strict canonical bytes suitable for atomic persistent publication."""
        return canonical_json_bytes(self._revalidated())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class NativeReplaySample(_FrozenModel):
    """Deterministically selected game references with separate head normalizers."""

    games: tuple[NativeReplayGameRef, ...]
    policy_contributing_game_count: StrictInt = Field(ge=0)
    value_contributing_game_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _valid_counts(self) -> NativeReplaySample:
        if len({game.game.game_id for game in self.games}) != len(self.games):
            raise ValueError("native replay sample cannot contain duplicate games")
        if self.policy_contributing_game_count != sum(
            game.policy_row_count > 0 for game in self.games
        ):
            raise ValueError("policy contributing-game normalizer is invalid")
        if self.value_contributing_game_count != sum(
            game.value_row_count > 0 for game in self.games
        ):
            raise ValueError("value contributing-game normalizer is invalid")
        return self

    @property
    def game_ids(self) -> tuple[str, ...]:
        return tuple(game.game.game_id for game in self.games)


class _SelectionRankInput(_FrozenModel):
    recipe: Literal["uniform-train-games-v1"] = _SELECTION_RECIPE
    sampling_seed: StrictInt
    game_id: str = Field(pattern=_DIGEST_PATTERN)


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_source_logical_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or not (value.endswith(".jsonl") or value.endswith(".jsonl.zst"))
    ):
        raise ValueError("source logical name must be a normalized relative native JSONL path")
    return value


def _revalidate_config(config: NativeReplayConfig) -> NativeReplayConfig:
    return NativeReplayConfig.model_validate(config.model_dump(mode="python"), strict=True)


def _revalidate_completion(
    completion: NativeDatasetCompletionReceipt,
) -> NativeDatasetCompletionReceipt:
    if not isinstance(completion, NativeDatasetCompletionReceipt):
        raise TypeError("completion_receipt must be a NativeDatasetCompletionReceipt")
    return NativeDatasetCompletionReceipt.model_validate(
        completion.model_dump(mode="python"), strict=True
    )


def _revalidate_manifest(dataset: IndexedNativeDataset) -> NativeIndexedDatasetManifest:
    if not isinstance(dataset, IndexedNativeDataset):
        raise TypeError("dataset must be an IndexedNativeDataset")
    return NativeIndexedDatasetManifest.model_validate(
        dataset.manifest.model_dump(mode="python"), strict=True
    )


def _current_scope() -> tuple[set[str], set[str], dict[str, int]]:
    maps_root = Path(__file__).parents[2] / "goa2" / "data" / "maps"
    maps = {path.stem for path in maps_root.glob("*.json") if path.is_file()}
    heroes = set(HeroRegistry.list_heroes())
    game_types = {game_type.value for game_type in GameType}
    if not maps or not heroes or not game_types:  # pragma: no cover - packaging guard
        raise ValueError("current native runtime scope could not be enumerated")
    registry = HeroObservationAdapterRegistry()
    registered = registry.registered_versions
    adapters = {
        "generic": registry.generic_version,
        **{hero: registered.get(hero, registry.generic_version) for hero in heroes},
    }
    return maps, game_types, adapters


def _validate_native_scope(games: tuple[NativeGameIdentity, ...]) -> None:
    maps, game_types, adapters = _current_scope()
    heroes = set(adapters) - {"generic"}
    for game in games:
        if (
            game.map_id not in maps
            or game.game_type not in game_types
            or not set((*game.red_composition, *game.blue_composition)) <= heroes
        ):
            raise ValueError("native dataset game is outside the complete current runtime scope")


def _validate_parent(
    parent: Gen1ModelArtifactManifest,
    *,
    manifest: NativeIndexedDatasetManifest,
    source_model_digest: str,
) -> Gen1ModelArtifactManifest:
    if not isinstance(parent, Gen1ModelArtifactManifest):
        raise TypeError("learned native replay requires an actual Gen1 model artifact manifest")
    validated = Gen1ModelArtifactManifest.model_validate(
        parent.model_dump(mode="python"), strict=True
    )
    if validated.model_digest != source_model_digest:
        raise ValueError("native generation source model digest does not match its Gen1 parent")
    if validated.runtime_compatibility_version != GEN1_RUNTIME_COMPATIBILITY_VERSION:
        raise ValueError("Gen1 parent runtime compatibility identity is invalid")
    if (
        validated.decision_observation_schema_version != 4
        or validated.stable_value_observation_schema_version != 1
        or validated.graph_observation_schema_version != 2
        or validated.map_schema_version != CURRENT_MAP_SCHEMA_VERSION
        or validated.model_id != _MODEL_ID
        or validated.value_semantics != _VALUE_SEMANTICS
    ):
        raise ValueError("Gen1 parent executable schema or semantics are incompatible")
    if (
        validated.decision_tensor_schema_id,
        validated.decision_tensor_schema_version,
        validated.decision_tensor_schema_digest,
        validated.stable_value_tensor_schema_id,
        validated.stable_value_tensor_schema_version,
        validated.stable_value_tensor_schema_digest,
    ) != (
        manifest.decision_tensor_schema_id,
        manifest.decision_tensor_schema_version,
        manifest.decision_tensor_schema_digest,
        manifest.stable_value_tensor_schema_id,
        manifest.stable_value_tensor_schema_version,
        manifest.stable_value_tensor_schema_digest,
    ):
        raise ValueError("Gen1 parent and native dataset tensor schemas are incompatible")

    maps, game_types, adapters = _current_scope()
    heroes = set(adapters) - {"generic"}
    if (
        set(validated.supported_maps) != maps
        or set(validated.supported_game_types) != game_types
        or set(validated.supported_heroes) != heroes
        or validated.hero_adapter_versions != adapters
    ):
        raise ValueError("Gen1 parent must have the exact complete current runtime scope")
    return validated


def _compatibility_from_manifest(
    manifest: NativeIndexedDatasetManifest,
) -> NativeReplayCompatibility:
    if manifest.schema_version != INDEX_SCHEMA_VERSION:
        raise ValueError("native indexed dataset schema is incompatible with replay")
    return NativeReplayCompatibility(
        decision_tensor_schema_id=manifest.decision_tensor_schema_id,
        decision_tensor_schema_version=manifest.decision_tensor_schema_version,
        decision_tensor_schema_digest=manifest.decision_tensor_schema_digest,
        stable_value_tensor_schema_id=manifest.stable_value_tensor_schema_id,
        stable_value_tensor_schema_version=manifest.stable_value_tensor_schema_version,
        stable_value_tensor_schema_digest=manifest.stable_value_tensor_schema_digest,
    )


def load_native_replay_catalog(path: str | Path) -> NativeReplayCatalog:
    """Load a strict canonical catalog from a regular file without following symlinks."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("native replay catalog must be a regular file")
    payload = source.read_bytes()
    try:
        catalog = NativeReplayCatalog.model_validate_json(payload, strict=True)
    except ValueError as exc:
        raise ValueError(f"invalid native replay catalog: {exc}") from exc
    if payload != catalog.canonical_bytes():
        raise ValueError("native replay catalog is not canonical JSON")
    return catalog


def _resolved(path: Path) -> Path:
    return path.resolve(strict=False)


def _is_under(path: Path, root: Path) -> bool:
    resolved_path = _resolved(path)
    resolved_root = _resolved(root)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _source_paths(
    dataset: IndexedNativeDataset, completion: NativeDatasetCompletionReceipt
) -> tuple[Path, ...]:
    return tuple(
        Path(dataset.source_root) / _safe_source_logical_name(game.logical_name)
        for game in completion.games
    )


def _validate_managed_paths(
    path: Path,
    lock_path: Path,
    dataset: IndexedNativeDataset,
    completion: NativeDatasetCompletionReceipt,
) -> None:
    if ".." in path.parts:
        raise ValueError("native replay catalog path must not contain parent traversal")
    parent = path.parent
    absolute_parent = parent.absolute()
    current = Path(absolute_parent.anchor)
    for part in absolute_parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("native replay catalog path must not contain symlinks")
    if not parent.is_dir():
        raise ValueError("native replay catalog parent must be a regular existing directory")
    if not path.name or path.name in {".", ".."}:
        raise ValueError("native replay catalog path must name a file")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("native replay catalog path must be absent or a regular file")
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise ValueError("native replay lock path must be absent or a regular file")
    if _is_under(path, Path(dataset.cache_dir)) or _is_under(lock_path, Path(dataset.cache_dir)):
        raise ValueError("native replay catalog and lock cannot live under disposable index cache")
    sources = _source_paths(dataset, completion)
    if any(
        _resolved(managed) == _resolved(source)
        for managed in (path, lock_path)
        for source in sources
    ):
        raise ValueError("native replay catalog or lock overlaps a source file")


def _open_lock(lock_path: Path) -> int:
    with suppress(FileExistsError):
        _publish_new_without_clobber(lock_path, _LOCK_CONTENT)
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("native replay lock must be a regular owned lock file")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        if os.read(descriptor, len(_LOCK_CONTENT) + 1) != _LOCK_CONTENT:
            raise ValueError("native replay lock path contains unrelated data")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def _catalog_lock(lock_path: Path) -> Iterator[None]:
    descriptor = _open_lock(lock_path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(_LOCK_CONTENT) + 1) != _LOCK_CONTENT:
            raise ValueError("native replay lock ownership marker changed")
        yield
    finally:
        os.close(descriptor)


def _publish_new_without_clobber(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError("native replay catalog appeared during publication") from exc
        fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _validate_dataset_completion(
    dataset: IndexedNativeDataset,
    manifest: NativeIndexedDatasetManifest,
    completion: NativeDatasetCompletionReceipt,
) -> tuple[NativeGameIdentity, ...]:
    inventory = create_native_source_receipt_from_completions(dataset.source_root, completion)
    if inventory != manifest.source_receipt or inventory.digest != dataset.source_digest:
        raise ValueError("completion receipt does not match indexed source inventory and order")
    if tuple(game.game.game_id for game in completion.games) != dataset.game_ids:
        raise ValueError("completion receipt and native index game identities differ")

    identities: list[NativeGameIdentity] = []
    for indexed, completed in zip(manifest.games, completion.games, strict=True):
        if (
            indexed.identity != completed.game
            or indexed.source_logical_name != completed.logical_name
            or indexed.row_count != completed.row_count
            or indexed.policy_row_count != completed.policy_row_count
            or indexed.value_row_count != completed.value_row_count
            or indexed.boundary_count != completed.boundary_count
        ):
            raise ValueError("completion receipt and indexed game metadata differ")
        validated_game = dataset.validate_game(indexed.game_id)
        if validated_game != indexed:
            raise ValueError("native index returned inconsistent validated game metadata")
        identities.append(indexed.identity)

    semantic_digest = validate_native_dataset_completion(dataset.source_root, completion)
    if semantic_digest != dataset.digest or semantic_digest != manifest.semantic_dataset_digest:
        raise ValueError("completed source semantic digest does not match native index")
    return tuple(identities)


def update_native_replay_catalog(
    path: str | Path,
    *,
    config: NativeReplayConfig,
    dataset: IndexedNativeDataset,
    completion_receipt: NativeDatasetCompletionReceipt,
    parent_artifact: Gen1ModelArtifactManifest | None,
) -> NativeReplayCatalog:
    """Validate and atomically enroll exactly one complete native generation."""
    target = Path(path)
    lock_path = target.parent / f".{target.name}.lock"
    validated_config = _revalidate_config(config)
    manifest = _revalidate_manifest(dataset)
    completion = _revalidate_completion(completion_receipt)
    _validate_managed_paths(target, lock_path, dataset, completion)

    with _catalog_lock(lock_path):
        existing = load_native_replay_catalog(target) if target.exists() else None
        if existing is not None and existing.config != validated_config:
            raise ValueError("native replay config is immutable once a catalog exists")

        identities = _validate_dataset_completion(dataset, manifest, completion)
        _validate_native_scope(identities)
        generation_ids = {game.generation_id for game in identities}
        source_model_digests = {game.source_model_digest for game in identities}
        if len(generation_ids) != 1:
            raise ValueError("native replay enrollment requires exactly one generation ID")
        if len(source_model_digests) != 1:
            raise ValueError("native replay enrollment requires exactly one source model digest")
        generation_id = next(iter(generation_ids))
        source_model_digest = next(iter(source_model_digests))

        if source_model_digest is None:
            if parent_artifact is not None:
                raise ValueError("heuristic bootstrap data must not declare a parent artifact")
        else:
            if parent_artifact is None:
                raise ValueError("learned native data requires its exact Gen1 parent artifact")
            _validate_parent(
                parent_artifact,
                manifest=manifest,
                source_model_digest=source_model_digest,
            )

        compatibility = _compatibility_from_manifest(manifest)
        if existing is not None:
            if existing.compatibility != compatibility:
                raise ValueError("native generation is incompatible with the replay catalog")
            historical_generations = existing.generations
            historical_games = existing.games
            ledger = existing.split_ledger
            if generation_id in {item.generation_id for item in historical_generations}:
                raise ValueError(f"native replay generation already exists: {generation_id!r}")
            if manifest.semantic_dataset_digest in {
                item.dataset_digest for item in historical_generations
            }:
                raise ValueError("native replay dataset digest already exists")
            duplicate_ids = set(dataset.game_ids) & {
                game_id
                for generation in historical_generations
                for game_id in generation.all_game_ids
            }
            if duplicate_ids:
                raise ValueError(f"native replay game IDs already exist: {sorted(duplicate_ids)!r}")
        else:
            historical_generations = ()
            historical_games = ()
            ledger = create_native_split_ledger(validated_config.split_config)

        ledger = extend_native_split_ledger(ledger, (game.world_seed for game in identities))
        split_by_seed = {
            assignment.world_seed: assignment.split for assignment in ledger.assignments
        }
        replay_ids = tuple(
            game.game_id for game in identities if split_by_seed[game.world_seed] == "train"
        )
        generation = NativeReplayGeneration(
            generation_id=generation_id,
            source_model_digest=source_model_digest,
            dataset_digest=manifest.semantic_dataset_digest,
            source_digest=manifest.source_digest,
            completion_receipt_digest=completion.digest,
            all_game_ids=tuple(game.game_id for game in identities),
            replay_game_ids=replay_ids,
        )
        first_insertion_index = sum(len(item.replay_game_ids) for item in historical_generations)
        completed_by_id = {game.game.game_id: game for game in completion.games}
        indexed_by_id = {game.game_id: game for game in manifest.games}
        replay_id_set = set(replay_ids)
        additions = tuple(
            NativeReplayGameRef(
                game=game,
                dataset_digest=generation.dataset_digest,
                source_digest=generation.source_digest,
                completion_receipt_digest=generation.completion_receipt_digest,
                source_logical_name=completed_by_id[game.game_id].logical_name,
                policy_row_count=indexed_by_id[game.game_id].policy_row_count,
                value_row_count=indexed_by_id[game.game_id].value_row_count,
                boundary_count=indexed_by_id[game.game_id].boundary_count,
                insertion_index=first_insertion_index + offset,
            )
            for offset, game in enumerate(
                game for game in identities if game.game_id in replay_id_set
            )
        )
        retained = (*historical_games, *additions)
        if validated_config.max_games is not None:
            retained = retained[-validated_config.max_games :]
        catalog = NativeReplayCatalog(
            config=validated_config,
            compatibility=compatibility,
            split_ledger=ledger,
            generations=(*historical_generations, generation),
            games=retained,
        )
        payload = catalog.canonical_bytes()
        if existing is None:
            _publish_new_without_clobber(target, payload)
        else:
            atomic_write_bytes(target, payload)
        return catalog


def _strict_positive_count(game_count: int) -> int:
    if isinstance(game_count, bool) or not isinstance(game_count, int):
        raise TypeError("game_count must be a positive integer")
    if game_count <= 0:
        raise ValueError("game_count must be a positive integer")
    return game_count


def _strict_sampling_seed(sampling_seed: int) -> int:
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
        raise TypeError("sampling_seed must be an integer")
    return sampling_seed


def sample_native_replay(
    catalog: NativeReplayCatalog,
    *,
    game_count: int,
    sampling_seed: int,
) -> NativeReplaySample:
    """Select uniform whole train games by deterministic domain-separated hash ranking."""
    validated = NativeReplayCatalog.model_validate(catalog.model_dump(mode="python"), strict=True)
    count = _strict_positive_count(game_count)
    seed = _strict_sampling_seed(sampling_seed)
    if count > len(validated.games):
        raise ValueError(
            f"native replay catalog has {len(validated.games)} train games, requires {count}"
        )

    def rank(game: NativeReplayGameRef) -> tuple[bytes, str]:
        payload = canonical_json_bytes(
            _SelectionRankInput(sampling_seed=seed, game_id=game.game.game_id)
        )
        return hashlib.sha256(payload).digest(), game.game.game_id

    selected = tuple(sorted(validated.games, key=rank)[:count])
    return NativeReplaySample(
        games=selected,
        policy_contributing_game_count=sum(game.policy_row_count > 0 for game in selected),
        value_contributing_game_count=sum(game.value_row_count > 0 for game in selected),
    )


__all__ = [
    "NativeReplayCatalog",
    "NativeReplayCompatibility",
    "NativeReplayConfig",
    "NativeReplayGameRef",
    "NativeReplayGeneration",
    "NativeReplaySample",
    "load_native_replay_catalog",
    "sample_native_replay",
    "update_native_replay_catalog",
]
