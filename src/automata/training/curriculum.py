"""Declarative, leakage-safe hero and map curriculum utilities.

This module schedules reproducible cases and validates compatibility metadata.
It deliberately does not run games, train models, or make promotion claims.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

T = TypeVar("T")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class AdapterRequirement(_FrozenModel):
    """Exact observation adapter identity required by a hero declaration."""

    adapter_id: str = Field(min_length=1)
    schema_version: StrictInt = Field(ge=1)


class HeroPoolEntry(_FrozenModel):
    """One hero available to the curriculum, including its adapter contract."""

    name: str = Field(min_length=1)
    adapter: AdapterRequirement


class MapTopologyMetadata(_FrozenModel):
    """Auditable graph-shape facts used to report map coverage."""

    lane_count: StrictInt = Field(ge=1)
    playable_tile_count: StrictInt = Field(ge=1)
    graph_node_count: StrictInt = Field(ge=1)
    graph_edge_count: StrictInt = Field(ge=0)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_metadata(self) -> MapTopologyMetadata:
        _require_unique_nonempty(self.tags, "topology tags")
        if self.graph_node_count < self.playable_tile_count:
            raise ValueError("graph node count cannot be smaller than playable tile count")
        return self


class MapPoolEntry(_FrozenModel):
    """One map and the map-schema/topology metadata needed before use."""

    map_id: str = Field(min_length=1)
    map_schema_version: StrictInt = Field(ge=1)
    topology: MapTopologyMetadata


class CurriculumComposition(_FrozenModel):
    """An explicit matchup; no implicit Cartesian product is sampled."""

    composition_id: str = Field(min_length=1)
    red_heroes: tuple[str, ...]
    blue_heroes: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_heroes(self) -> CurriculumComposition:
        if not self.red_heroes or not self.blue_heroes:
            raise ValueError("composition teams must be non-empty")
        all_heroes = (*self.red_heroes, *self.blue_heroes)
        _require_unique_nonempty(all_heroes, "composition heroes")
        return self

    @property
    def hero_ids(self) -> tuple[str, ...]:
        return (*self.red_heroes, *self.blue_heroes)


class CurriculumStage(_FrozenModel):
    """A cumulative snapshot of the pools unlocked at one stage."""

    name: str = Field(min_length=1)
    hero_ids: tuple[str, ...]
    map_ids: tuple[str, ...]
    composition_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_pools(self) -> CurriculumStage:
        if not self.hero_ids or not self.map_ids or not self.composition_ids:
            raise ValueError("stage hero, map, and composition pools must be non-empty")
        _require_unique_nonempty(self.hero_ids, "stage heroes")
        _require_unique_nonempty(self.map_ids, "stage maps")
        _require_unique_nonempty(self.composition_ids, "stage compositions")
        return self


class SchemaRequirement(_FrozenModel):
    """Exact shared tensor/observation schema required by this curriculum."""

    observation_schema_version: StrictInt = Field(ge=1)
    tensor_schema_id: str = Field(min_length=1)
    tensor_schema_version: StrictInt = Field(ge=1)
    tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class CurriculumCase(_FrozenModel):
    """One deterministic training game declaration."""

    stage_name: str
    sample_index: StrictInt = Field(ge=0)
    world_seed: StrictInt = Field(ge=0)
    map_id: str
    composition_id: str
    red_heroes: tuple[str, ...]
    blue_heroes: tuple[str, ...]


class TopologyCoverage(_FrozenModel):
    """Aggregate graph-shape coverage for one unlocked map pool."""

    map_ids: tuple[str, ...]
    lane_counts: tuple[int, ...]
    playable_tile_count_range: tuple[int, int]
    graph_node_count_range: tuple[int, int]
    graph_edge_count_range: tuple[int, int]
    tags: tuple[str, ...]


class CurriculumConfig(_FrozenModel):
    """Complete explicit pools, holdouts, cumulative stages, and seed ownership."""

    schema_version: Literal[1] = 1
    schema_requirement: SchemaRequirement
    heroes: tuple[HeroPoolEntry, ...]
    maps: tuple[MapPoolEntry, ...]
    compositions: tuple[CurriculumComposition, ...]
    holdout_hero_ids: tuple[str, ...] = ()
    holdout_map_ids: tuple[str, ...] = ()
    holdout_composition_ids: tuple[str, ...] = ()
    stages: tuple[CurriculumStage, ...]
    training_world_seeds: tuple[StrictInt, ...]
    arena_world_seeds: tuple[StrictInt, ...]

    @property
    def heroes_by_name(self) -> Mapping[str, HeroPoolEntry]:
        return {hero.name: hero for hero in self.heroes}

    @property
    def maps_by_id(self) -> Mapping[str, MapPoolEntry]:
        return {map_entry.map_id: map_entry for map_entry in self.maps}

    @property
    def compositions_by_id(self) -> Mapping[str, CurriculumComposition]:
        return {composition.composition_id: composition for composition in self.compositions}

    def stage(self, name: str) -> CurriculumStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(f"unknown curriculum stage {name!r}")

    @model_validator(mode="after")
    def _validate_model(self) -> CurriculumConfig:
        return self.validate_declaration()

    def validate_declaration(self) -> CurriculumConfig:
        """Validate references, cumulative unlocks, holdouts, and seed isolation."""

        if not self.heroes or not self.maps or not self.compositions or not self.stages:
            raise ValueError("curriculum pools and stages must be non-empty")
        _require_unique_nonempty(tuple(hero.name for hero in self.heroes), "hero pool")
        _require_unique_nonempty(tuple(item.map_id for item in self.maps), "map pool")
        _require_unique_nonempty(
            tuple(item.composition_id for item in self.compositions), "composition pool"
        )
        _require_unique_nonempty(tuple(stage.name for stage in self.stages), "stage names")
        _require_unique_nonempty(self.holdout_hero_ids, "hero holdouts")
        _require_unique_nonempty(self.holdout_map_ids, "map holdouts")
        _require_unique_nonempty(self.holdout_composition_ids, "composition holdouts")
        _validate_seed_pool(self.training_world_seeds, "training")
        _validate_seed_pool(self.arena_world_seeds, "arena")

        overlap = set(self.training_world_seeds) & set(self.arena_world_seeds)
        if overlap:
            raise ValueError(
                f"training and arena seed pools overlap at {min(overlap)}; seed leakage is forbidden"
            )

        hero_ids = set(self.heroes_by_name)
        map_ids = set(self.maps_by_id)
        composition_ids = set(self.compositions_by_id)
        _require_subset(self.holdout_hero_ids, hero_ids, "hero holdouts")
        _require_subset(self.holdout_map_ids, map_ids, "map holdouts")
        _require_subset(self.holdout_composition_ids, composition_ids, "composition holdouts")
        for composition in self.compositions:
            _require_subset(
                composition.hero_ids, hero_ids, f"composition {composition.composition_id}"
            )

        previous: CurriculumStage | None = None
        for stage in self.stages:
            _require_subset(stage.hero_ids, hero_ids, f"stage {stage.name} heroes")
            _require_subset(stage.map_ids, map_ids, f"stage {stage.name} maps")
            _require_subset(
                stage.composition_ids, composition_ids, f"stage {stage.name} compositions"
            )
            _reject_overlap(stage.hero_ids, self.holdout_hero_ids, "hero", stage.name)
            _reject_overlap(stage.map_ids, self.holdout_map_ids, "map", stage.name)
            _reject_overlap(
                stage.composition_ids, self.holdout_composition_ids, "composition", stage.name
            )
            for composition_id in stage.composition_ids:
                composition = self.compositions_by_id[composition_id]
                if not set(composition.hero_ids) <= set(stage.hero_ids):
                    raise ValueError(
                        f"stage {stage.name!r} composition {composition_id!r} uses locked heroes"
                    )
            if previous is not None and not (
                set(previous.hero_ids) <= set(stage.hero_ids)
                and set(previous.map_ids) <= set(stage.map_ids)
                and set(previous.composition_ids) <= set(stage.composition_ids)
            ):
                raise ValueError("curriculum stage pools must be incremental and cumulative")
            previous = stage
        return self


class _Adapter(Protocol):
    adapter_id: str
    schema_version: int


class _AdapterRegistry(Protocol):
    def resolve(self, hero_definition_name: str) -> _Adapter: ...


class SchemaIdentity(Protocol):
    """Narrow schema identity required by curriculum validation."""

    @property
    def observation_schema_version(self) -> int: ...

    @property
    def schema_id(self) -> str: ...

    @property
    def schema_version(self) -> int: ...

    @property
    def digest(self) -> str: ...


def validate_curriculum_requirements(
    config: CurriculumConfig,
    *,
    tensor_schema: SchemaIdentity,
    adapter_registry: _AdapterRegistry,
    available_map_schema_versions: Mapping[str, int],
) -> None:
    """Fail closed unless all declared schemas and adapter identities are available."""

    required = config.schema_requirement
    actual_schema = (
        tensor_schema.observation_schema_version,
        tensor_schema.schema_id,
        tensor_schema.schema_version,
        tensor_schema.digest,
    )
    required_schema = (
        required.observation_schema_version,
        required.tensor_schema_id,
        required.tensor_schema_version,
        required.tensor_schema_digest,
    )
    if actual_schema != required_schema:
        raise ValueError("tensor schema does not satisfy curriculum schema requirement")

    for hero in config.heroes:
        adapter = adapter_registry.resolve(hero.name)
        if (
            adapter.adapter_id != hero.adapter.adapter_id
            or adapter.schema_version != hero.adapter.schema_version
        ):
            raise ValueError(f"hero {hero.name!r} adapter does not satisfy curriculum requirement")

    for map_entry in config.maps:
        actual_version = available_map_schema_versions.get(map_entry.map_id)
        if actual_version != map_entry.map_schema_version:
            raise ValueError(
                f"map {map_entry.map_id!r} schema version does not satisfy curriculum requirement"
            )


def sample_training_cases(
    config: CurriculumConfig, *, stage_name: str, count: int, seed: int
) -> tuple[CurriculumCase, ...]:
    """Sample unique training seeds while balancing games across unlocked maps."""

    if isinstance(count, bool) or count < 0:
        raise ValueError("sample count must be non-negative")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("sampling seed must be non-negative")
    if count > len(config.training_world_seeds):
        raise ValueError("sample count exceeds the unique training world-seed pool")
    stage = config.stage(stage_name)
    world_seeds = _ordered(seed, "world-seed", config.training_world_seeds)[:count]
    compositions = tuple(config.compositions_by_id[item] for item in stage.composition_ids)

    cases: list[CurriculumCase] = []
    for sample_index, world_seed in enumerate(world_seeds):
        map_id = stage.map_ids[sample_index % len(stage.map_ids)]
        composition = _ordered(
            seed,
            f"composition:{sample_index}:{world_seed}",
            compositions,
            identity=lambda item: item.composition_id,
        )[0]
        cases.append(
            CurriculumCase(
                stage_name=stage.name,
                sample_index=sample_index,
                world_seed=world_seed,
                map_id=map_id,
                composition_id=composition.composition_id,
                red_heroes=composition.red_heroes,
                blue_heroes=composition.blue_heroes,
            )
        )
    return tuple(cases)


def topology_coverage(config: CurriculumConfig, *, stage_name: str) -> TopologyCoverage:
    """Summarize declared graph variation for the maps unlocked at a stage."""

    stage = config.stage(stage_name)
    metadata = tuple(config.maps_by_id[map_id].topology for map_id in stage.map_ids)
    return TopologyCoverage(
        map_ids=stage.map_ids,
        lane_counts=tuple(sorted({item.lane_count for item in metadata})),
        playable_tile_count_range=_range(item.playable_tile_count for item in metadata),
        graph_node_count_range=_range(item.graph_node_count for item in metadata),
        graph_edge_count_range=_range(item.graph_edge_count for item in metadata),
        tags=tuple(sorted({tag for item in metadata for tag in item.tags})),
    )


def _require_unique_nonempty(values: tuple[str, ...], label: str) -> None:
    if any(not value or value != value.strip() for value in values):
        raise ValueError(f"{label} must contain non-empty normalized strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _validate_seed_pool(seeds: tuple[int, ...], label: str) -> None:
    if not seeds:
        raise ValueError(f"{label} seed pool must be non-empty")
    if any(isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError(f"{label} seed pool must contain non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{label} seed pool must contain unique seeds")


def _require_subset(values: tuple[str, ...], available: set[str], label: str) -> None:
    missing = set(values) - available
    if missing:
        raise ValueError(f"{label} references unknown values: {sorted(missing)}")


def _reject_overlap(
    active: tuple[str, ...], holdouts: tuple[str, ...], kind: str, stage_name: str
) -> None:
    overlap = set(active) & set(holdouts)
    if overlap:
        raise ValueError(f"stage {stage_name!r} unlocks {kind} holdout {sorted(overlap)!r}")


def _ordered(
    seed: int,
    domain: str,
    values: tuple[T, ...],
    *,
    identity: Callable[[T], object] = str,
) -> tuple[T, ...]:
    def key(value: T) -> tuple[bytes, str]:
        value_identity = str(identity(value))
        material = f"{seed}\0{domain}\0{value_identity}".encode()
        return hashlib.sha256(material).digest(), value_identity

    return tuple(sorted(values, key=key))


def _range(values: Iterable[int]) -> tuple[int, int]:
    materialized = tuple(values)
    return min(materialized), max(materialized)


__all__ = [
    "AdapterRequirement",
    "CurriculumCase",
    "CurriculumComposition",
    "CurriculumConfig",
    "CurriculumStage",
    "HeroPoolEntry",
    "MapPoolEntry",
    "MapTopologyMetadata",
    "SchemaIdentity",
    "SchemaRequirement",
    "TopologyCoverage",
    "sample_training_cases",
    "topology_coverage",
    "validate_curriculum_requirements",
]
