"""Behavioral contract for explicit, reproducible curriculum declarations."""

from __future__ import annotations

import pytest

from automata.models.shared_encoder.schema import TensorFeatureSchema
from automata.observation import HeroObservationAdapterRegistry
from automata.training.curriculum import (
    AdapterRequirement,
    CurriculumComposition,
    CurriculumConfig,
    CurriculumStage,
    HeroPoolEntry,
    MapPoolEntry,
    MapTopologyMetadata,
    SchemaRequirement,
    sample_training_cases,
    topology_coverage,
    validate_curriculum_requirements,
)


def _config() -> CurriculumConfig:
    schema = TensorFeatureSchema.current()
    return CurriculumConfig(
        schema_requirement=SchemaRequirement(
            observation_schema_version=schema.observation_schema_version,
            tensor_schema_id=schema.schema_id,
            tensor_schema_version=schema.schema_version,
            tensor_schema_digest=schema.digest,
        ),
        heroes=tuple(
            HeroPoolEntry(
                name=name, adapter=AdapterRequirement(adapter_id="generic", schema_version=1)
            )
            for name in ("Wasp", "Arien", "Brogan", "Razzle", "Xargatha")
        ),
        maps=(
            MapPoolEntry(
                map_id="forgotten_island",
                map_schema_version=1,
                topology=MapTopologyMetadata(
                    lane_count=1,
                    playable_tile_count=121,
                    graph_node_count=128,
                    graph_edge_count=334,
                    tags=("single_lane",),
                ),
            ),
            MapPoolEntry(
                map_id="across_the_river",
                map_schema_version=1,
                topology=MapTopologyMetadata(
                    lane_count=2,
                    playable_tile_count=181,
                    graph_node_count=194,
                    graph_edge_count=502,
                    tags=("double_lane", "river"),
                ),
            ),
            MapPoolEntry(
                map_id="vexing_cliffs",
                map_schema_version=1,
                topology=MapTopologyMetadata(
                    lane_count=2,
                    playable_tile_count=117,
                    graph_node_count=128,
                    graph_edge_count=290,
                    tags=("double_lane", "cliffs"),
                ),
            ),
        ),
        compositions=(
            CurriculumComposition(
                composition_id="benchmark",
                red_heroes=("Wasp", "Xargatha"),
                blue_heroes=("Arien", "Brogan"),
            ),
            CurriculumComposition(
                composition_id="razzle-recombination",
                red_heroes=("Razzle", "Arien"),
                blue_heroes=("Wasp", "Brogan"),
            ),
        ),
        holdout_hero_ids=("Razzle",),
        holdout_map_ids=("vexing_cliffs",),
        holdout_composition_ids=("razzle-recombination",),
        stages=(
            CurriculumStage(
                name="benchmark",
                hero_ids=("Wasp", "Xargatha", "Arien", "Brogan"),
                map_ids=("forgotten_island",),
                composition_ids=("benchmark",),
            ),
            CurriculumStage(
                name="map-expansion",
                hero_ids=("Wasp", "Xargatha", "Arien", "Brogan"),
                map_ids=("forgotten_island", "across_the_river"),
                composition_ids=("benchmark",),
            ),
        ),
        training_world_seeds=tuple(range(20_000, 20_012)),
        arena_world_seeds=tuple(range(1_010_000, 1_010_006)),
    )


def test_explicit_holdouts_and_incremental_unlocks_are_enforced() -> None:
    config = _config()

    assert config.stage("benchmark").map_ids == ("forgotten_island",)
    assert config.stage("map-expansion").map_ids == (
        "forgotten_island",
        "across_the_river",
    )

    with pytest.raises(ValueError, match="incremental"):
        config.model_copy(
            update={
                "stages": (
                    config.stages[1],
                    config.stages[0],
                )
            }
        ).validate_declaration()

    with pytest.raises(ValueError, match="holdout"):
        config.model_copy(
            update={
                "stages": (
                    config.stages[0].model_copy(
                        update={"map_ids": ("forgotten_island", "vexing_cliffs")}
                    ),
                )
            }
        ).validate_declaration()


def test_seeded_sampling_is_reproducible_map_balanced_and_training_only() -> None:
    config = _config()

    first = sample_training_cases(config, stage_name="map-expansion", count=7, seed=47)
    second = sample_training_cases(config, stage_name="map-expansion", count=7, seed=47)

    assert first == second
    assert first != sample_training_cases(config, stage_name="map-expansion", count=7, seed=48)
    map_counts = {
        map_id: sum(case.map_id == map_id for case in first) for map_id in config.maps_by_id
    }
    assert map_counts == {
        "forgotten_island": 4,
        "across_the_river": 3,
        "vexing_cliffs": 0,
    }
    assert {case.world_seed for case in first} <= set(config.training_world_seeds)
    assert not ({case.world_seed for case in first} & set(config.arena_world_seeds))


def test_training_and_arena_seed_pools_must_be_disjoint() -> None:
    config = _config()

    with pytest.raises(ValueError, match=r"training.*arena.*overlap"):
        config.model_copy(
            update={"arena_world_seeds": (*config.arena_world_seeds, 20_003)}
        ).validate_declaration()


def test_schema_map_and_adapter_requirements_are_checked_without_adding_adapters() -> None:
    config = _config()
    schema = TensorFeatureSchema.current()
    adapters = HeroObservationAdapterRegistry()

    validate_curriculum_requirements(
        config,
        tensor_schema=schema,
        adapter_registry=adapters,
        available_map_schema_versions={map_id: 1 for map_id in config.maps_by_id},
    )

    with pytest.raises(ValueError, match="tensor schema"):
        validate_curriculum_requirements(
            config,
            tensor_schema=schema.model_copy(update={"schema_id": "wrong"}),
            adapter_registry=adapters,
            available_map_schema_versions={map_id: 1 for map_id in config.maps_by_id},
        )

    bad_adapter = config.heroes[0].model_copy(
        update={"adapter": AdapterRequirement(adapter_id="wasp-special", schema_version=1)}
    )
    with pytest.raises(ValueError, match=r"Wasp.*adapter"):
        validate_curriculum_requirements(
            config.model_copy(update={"heroes": (bad_adapter, *config.heroes[1:])}),
            tensor_schema=schema,
            adapter_registry=adapters,
            available_map_schema_versions={map_id: 1 for map_id in config.maps_by_id},
        )


def test_topology_coverage_reports_shape_variation_for_each_unlocked_map() -> None:
    coverage = topology_coverage(_config(), stage_name="map-expansion")

    assert coverage.map_ids == ("forgotten_island", "across_the_river")
    assert coverage.lane_counts == (1, 2)
    assert coverage.playable_tile_count_range == (121, 181)
    assert coverage.graph_node_count_range == (128, 194)
    assert coverage.graph_edge_count_range == (334, 502)
    assert coverage.tags == ("double_lane", "river", "single_lane")
