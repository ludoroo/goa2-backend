"""Deterministic, game-grouped splits for joint policy/value datasets."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from automata.models.contracts import canonical_json_bytes
from automata.training.dataset import JointDataset, JointDatasetRow
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from automata.training.indexed_dataset import IndexedJointDataset

SplitName = Literal[
    "train",
    "validation",
    "map_holdout",
    "composition_holdout",
    "game_mode_holdout",
]


class JointSplitConfig(BaseModel):
    """Immutable declarations controlling one reproducible split."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    seed: StrictInt = 0
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed_purpose: str = Field(default="training", min_length=1)
    holdout_map_ids: tuple[str, ...] = ()
    holdout_compositions: tuple[tuple[str, ...], ...] = ()
    holdout_game_modes: tuple[str, ...] = ()


class JointSplitMembership(BaseModel):
    """The sole split assignment for one complete game."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    game_id: str = Field(min_length=1)
    world_seed: StrictInt
    split: SplitName


class JointSplitManifest(BaseModel):
    """Canonical immutable split membership tied to exact dataset bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: JointSplitConfig
    memberships: tuple[JointSplitMembership, ...]

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class JointSplits:
    """A manifest and immutable row views for each declared split."""

    manifest: JointSplitManifest
    rows_by_split: Mapping[SplitName, tuple[JointDatasetRow, ...]]

    @property
    def digest(self) -> str:
        return self.manifest.digest

    def game_ids(self, split: SplitName) -> tuple[str, ...]:
        return tuple(item.game_id for item in self.manifest.memberships if item.split == split)

    def rows(self, split: SplitName) -> tuple[JointDatasetRow, ...]:
        return self.rows_by_split.get(split, ())


class _GameMetadata(Protocol):
    game_id: str
    world_seed: int
    map_id: str
    game_type: str
    red_composition: tuple[str, ...]
    blue_composition: tuple[str, ...]


def _score(config: JointSplitConfig, game_id: str, world_seed: int) -> bytes:
    material = f"{config.seed}\0{game_id}\0{world_seed}".encode()
    return hashlib.sha256(material).digest()


def _stratum(row: _GameMetadata) -> tuple[object, ...]:
    return (row.map_id, row.game_type, row.red_composition, row.blue_composition)


def _holdout_split(row: _GameMetadata, config: JointSplitConfig) -> SplitName | None:
    if row.map_id in config.holdout_map_ids:
        return "map_holdout"
    if (
        row.red_composition in config.holdout_compositions
        or row.blue_composition in config.holdout_compositions
    ):
        return "composition_holdout"
    if row.game_type in config.holdout_game_modes:
        return "game_mode_holdout"
    return None


def _representatives(dataset: JointDataset) -> dict[str, JointDatasetRow]:
    representatives: dict[str, JointDatasetRow] = {}
    seed_owners: dict[int, str] = {}
    for game_id, rows in dataset.rows_by_game.items():
        if not rows:
            raise ValueError(f"game {game_id!r} has no rows")
        representative = rows[0]
        identity = (
            representative.world_seed,
            representative.map_id,
            representative.game_type,
            representative.red_composition,
            representative.blue_composition,
        )
        if representative.game_id != game_id or any(
            row.game_id != game_id
            or (
                row.world_seed,
                row.map_id,
                row.game_type,
                row.red_composition,
                row.blue_composition,
            )
            != identity
            for row in rows
        ):
            raise ValueError(f"game {game_id!r} has conflicting split metadata")
        owner = seed_owners.get(representative.world_seed)
        if owner is not None and owner != game_id:
            raise ValueError(
                f"world seed {representative.world_seed} belongs to multiple games: "
                f"{owner!r} and {game_id!r}"
            )
        seed_owners[representative.world_seed] = game_id
        representatives[game_id] = representative
    if set(representatives) != set(dataset.game_ids):
        raise ValueError("dataset game index does not exactly match dataset metadata")
    if any(row.game_id not in representatives for row in dataset.rows):
        raise ValueError("dataset rows contain a game absent from the game index")
    return representatives


def _materialize(
    dataset: JointDataset,
    manifest: JointSplitManifest,
    assignments: Mapping[str, SplitName],
) -> JointSplits:
    grouped: dict[SplitName, list[JointDatasetRow]] = defaultdict(list)
    for row in dataset.rows:
        grouped[assignments[row.game_id]].append(row)
    frozen_groups: dict[SplitName, tuple[JointDatasetRow, ...]] = {
        name: tuple(rows) for name, rows in grouped.items()
    }
    return JointSplits(manifest=manifest, rows_by_split=MappingProxyType(frozen_groups))


def _split_manifest(
    games: Sequence[_GameMetadata],
    *,
    dataset_digest: str,
    config: JointSplitConfig,
) -> JointSplitManifest:
    representatives: dict[str, _GameMetadata] = {}
    seed_owners: dict[int, str] = {}
    strata: dict[tuple[object, ...], list[_GameMetadata]] = defaultdict(list)
    assignments: dict[str, SplitName] = {}
    for game in games:
        if game.game_id in representatives:
            raise ValueError(f"duplicate game metadata for {game.game_id!r}")
        owner = seed_owners.get(game.world_seed)
        if owner is not None and owner != game.game_id:
            raise ValueError(
                f"world seed {game.world_seed} belongs to multiple games: "
                f"{owner!r} and {game.game_id!r}"
            )
        seed_owners[game.world_seed] = game.game_id
        representatives[game.game_id] = game
        PHASE0_EXPERIMENT.seed_registry.require_seed(game.world_seed, purpose=config.seed_purpose)
        holdout = _holdout_split(game, config)
        if holdout is None:
            strata[_stratum(game)].append(game)
        else:
            assignments[game.game_id] = holdout

    for stratum_games in strata.values():
        ordered = sorted(
            stratum_games,
            key=lambda game: (_score(config, game.game_id, game.world_seed), game.game_id),
        )
        validation_count = min(
            len(ordered) - 1, max(1, round(len(ordered) * config.validation_fraction))
        )
        for index, game in enumerate(ordered):
            assignments[game.game_id] = "validation" if index < validation_count else "train"

    memberships = tuple(
        JointSplitMembership(
            game_id=game_id,
            world_seed=representatives[game_id].world_seed,
            split=assignments[game_id],
        )
        for game_id in sorted(assignments)
    )
    return JointSplitManifest(
        schema_version=1,
        dataset_digest=dataset_digest,
        config=config,
        memberships=memberships,
    )


def split_joint_dataset(
    dataset: JointDataset, *, config: JointSplitConfig | None = None
) -> JointSplits:
    """Create deterministic stratified assignments without splitting a game."""
    config = config or JointSplitConfig()
    representatives = _representatives(dataset)
    manifest = _split_manifest(
        tuple(representatives.values()), dataset_digest=dataset.digest, config=config
    )
    assignments = {item.game_id: item.split for item in manifest.memberships}
    return _materialize(dataset, manifest, assignments)


def split_indexed_joint_dataset(
    dataset: IndexedJointDataset, *, config: JointSplitConfig | None = None
) -> JointSplitManifest:
    """Create a whole-game split from compact index metadata only."""
    return _split_manifest(
        dataset.manifest.games,
        dataset_digest=dataset.digest,
        config=config or JointSplitConfig(),
    )


def _validate_manifest(
    games: Sequence[_GameMetadata],
    *,
    dataset_digest: str,
    manifest: JointSplitManifest,
) -> dict[str, SplitName]:
    if dataset_digest != manifest.dataset_digest:
        raise ValueError("split manifest dataset digest does not match the dataset")
    representatives = {game.game_id: game for game in games}
    if len(representatives) != len(games):
        raise ValueError("dataset game metadata contains duplicate game IDs")
    membership_ids = tuple(item.game_id for item in manifest.memberships)
    if len(set(membership_ids)) != len(membership_ids) or set(membership_ids) != set(
        representatives
    ):
        raise ValueError("split manifest membership must exactly match dataset games")

    assignments: dict[str, SplitName] = {}
    for item in manifest.memberships:
        game = representatives[item.game_id]
        if item.world_seed != game.world_seed:
            raise ValueError(f"split manifest seed does not match game {item.game_id!r}")
        PHASE0_EXPERIMENT.seed_registry.require_seed(
            game.world_seed, purpose=manifest.config.seed_purpose
        )
        assignments[item.game_id] = item.split
    return assignments


def apply_joint_split_manifest(dataset: JointDataset, manifest: JointSplitManifest) -> JointSplits:
    """Reuse recorded membership only when the exact source dataset still matches."""
    representatives = _representatives(dataset)
    assignments = _validate_manifest(
        tuple(representatives.values()), dataset_digest=dataset.digest, manifest=manifest
    )
    return _materialize(dataset, manifest, assignments)


def apply_indexed_joint_split_manifest(
    dataset: IndexedJointDataset, manifest: JointSplitManifest
) -> JointSplitManifest:
    """Validate a recorded manifest against compact index metadata."""
    _validate_manifest(dataset.manifest.games, dataset_digest=dataset.digest, manifest=manifest)
    return manifest


def reuse_joint_split_manifest(dataset: JointDataset, manifest: JointSplitManifest) -> JointSplits:
    """Alias emphasizing that applying a manifest performs no re-splitting."""
    return apply_joint_split_manifest(dataset, manifest)


__all__ = [
    "JointSplitConfig",
    "JointSplitManifest",
    "JointSplitMembership",
    "JointSplits",
    "SplitName",
    "apply_indexed_joint_split_manifest",
    "apply_joint_split_manifest",
    "reuse_joint_split_manifest",
    "split_indexed_joint_dataset",
    "split_joint_dataset",
]
