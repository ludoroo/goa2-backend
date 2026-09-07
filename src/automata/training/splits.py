"""Deterministic, game-grouped splits for joint policy/value datasets."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from automata.models.contracts import canonical_json_bytes
from automata.training.dataset import JointDataset, JointDatasetRow
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT

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


def _score(config: JointSplitConfig, game_id: str, world_seed: int) -> bytes:
    material = f"{config.seed}\0{game_id}\0{world_seed}".encode()
    return hashlib.sha256(material).digest()


def _stratum(row: JointDatasetRow) -> tuple[object, ...]:
    return (row.map_id, row.game_type, row.red_composition, row.blue_composition)


def _holdout_split(row: JointDatasetRow, config: JointSplitConfig) -> SplitName | None:
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


def split_joint_dataset(
    dataset: JointDataset, *, config: JointSplitConfig | None = None
) -> JointSplits:
    """Create deterministic stratified assignments without splitting a game."""
    config = config or JointSplitConfig()
    representatives = _representatives(dataset)

    strata: dict[tuple[object, ...], list[JointDatasetRow]] = defaultdict(list)
    assignments: dict[str, SplitName] = {}
    for row in representatives.values():
        PHASE0_EXPERIMENT.seed_registry.require_seed(row.world_seed, purpose=config.seed_purpose)
        holdout = _holdout_split(row, config)
        if holdout is None:
            strata[_stratum(row)].append(row)
        else:
            assignments[row.game_id] = holdout

    for stratum_rows in strata.values():
        ordered = sorted(
            stratum_rows,
            key=lambda row: (_score(config, row.game_id, row.world_seed), row.game_id),
        )
        validation_count = min(
            len(ordered) - 1, max(1, round(len(ordered) * config.validation_fraction))
        )
        for index, row in enumerate(ordered):
            assignments[row.game_id] = "validation" if index < validation_count else "train"

    memberships = tuple(
        JointSplitMembership(
            game_id=game_id,
            world_seed=representatives[game_id].world_seed,
            split=assignments[game_id],
        )
        for game_id in sorted(assignments)
    )
    manifest = JointSplitManifest(
        schema_version=1,
        dataset_digest=dataset.digest,
        config=config,
        memberships=memberships,
    )
    return _materialize(dataset, manifest, assignments)


def apply_joint_split_manifest(dataset: JointDataset, manifest: JointSplitManifest) -> JointSplits:
    """Reuse recorded membership only when the exact source dataset still matches."""
    if dataset.digest != manifest.dataset_digest:
        raise ValueError("split manifest dataset digest does not match the dataset")
    representatives = _representatives(dataset)
    membership_ids = tuple(item.game_id for item in manifest.memberships)
    if len(set(membership_ids)) != len(membership_ids) or set(membership_ids) != set(
        representatives
    ):
        raise ValueError("split manifest membership must exactly match dataset games")

    assignments: dict[str, SplitName] = {}
    for item in manifest.memberships:
        row = representatives[item.game_id]
        if item.world_seed != row.world_seed:
            raise ValueError(f"split manifest seed does not match game {item.game_id!r}")
        PHASE0_EXPERIMENT.seed_registry.require_seed(
            row.world_seed, purpose=manifest.config.seed_purpose
        )
        assignments[item.game_id] = item.split
    return _materialize(dataset, manifest, assignments)


def reuse_joint_split_manifest(dataset: JointDataset, manifest: JointSplitManifest) -> JointSplits:
    """Alias emphasizing that applying a manifest performs no re-splitting."""
    return apply_joint_split_manifest(dataset, manifest)


__all__ = [
    "JointSplitConfig",
    "JointSplitManifest",
    "JointSplitMembership",
    "JointSplits",
    "SplitName",
    "apply_joint_split_manifest",
    "reuse_joint_split_manifest",
    "split_joint_dataset",
]
