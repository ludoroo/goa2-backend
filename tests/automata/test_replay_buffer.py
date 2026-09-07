"""Behavioral contract for complete-game learned-model replay sampling."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

from automata.models.shared_encoder.artifacts import ModelArtifactManifest
from automata.training.contracts.experiment import SeedRange, SeedRegistry
from automata.training.dataset import (
    JointDataset,
    JointDatasetMetadata,
    JointDatasetRow,
)
from automata.training.replay_buffer import ReplayBuffer, ReplayCompatibilityError


def _manifest(digest: str, *, tensor_digest: str = "f" * 64) -> ModelArtifactManifest:
    return ModelArtifactManifest.model_construct(
        schema_version=2,
        model_digest=digest,
        observation_schema_version=2,
        map_schema_version=1,
        runtime_compatibility_version=1,
        hero_adapter_versions={"Wasp": 1, "Arien": 1},
        supported_heroes=("Wasp", "Arien"),
        supported_maps=("forgotten_island",),
        supported_game_types=("QUICK",),
        tensor_schema_id="joint-v1",
        tensor_schema_version=1,
        tensor_schema_digest=tensor_digest,
        architecture_config={},
        tensors={},
        files={},
        non_executable_files=(),
    )


def _dataset(generation: str, parent: str, game_lengths: dict[str, int]) -> JointDataset:
    grouped: dict[str, tuple[JointDatasetRow, ...]] = {}
    rows: list[JointDatasetRow] = []
    for game_number, (game_id, length) in enumerate(game_lengths.items()):
        game_rows = tuple(
            JointDatasetRow.model_construct(
                schema_version=1,
                game_id=game_id,
                decision_id=f"{game_id}-{index}",
                decision_index=index,
                generation_id=generation,
                world_seed=20_000 + game_number,
                source_model_digest=parent,
                map_id="forgotten_island",
                game_type="QUICK",
                red_composition=("Wasp",),
                blue_composition=("Arien",),
                observation=SimpleNamespace(
                    schema_version=3,
                    state=SimpleNamespace(schema_version=2),
                ),
            )
            for index in range(length)
        )
        grouped[game_id] = game_rows
        rows.extend(game_rows)
    metadata = JointDatasetMetadata.model_construct(
        schema_version=1,
        dataset_digest=(generation[0] * 64),
        row_count=len(rows),
        game_ids=tuple(grouped),
    )
    return JointDataset(
        rows=tuple(rows),
        digest=metadata.dataset_digest,
        metadata=metadata,
        rows_by_game=MappingProxyType(grouped),
    )


@pytest.mark.parametrize("arena_seed", [1_000_000, 1_010_000, 1_020_000])
def test_rejects_arena_seed_games_before_replay_storage(arena_seed: int) -> None:
    parent = "a" * 64
    dataset = _dataset("generation-1", parent, {"arena-game": 1})
    arena_row = dataset.rows[0].model_copy(update={"world_seed": arena_seed})
    dataset = replace(
        dataset,
        rows=(arena_row,),
        rows_by_game=MappingProxyType({"arena-game": (arena_row,)}),
    )
    buffer = ReplayBuffer()

    with pytest.raises(ValueError, match="not eligible for replay"):
        buffer.add_generation(dataset, champion_parent=_manifest(parent))

    assert buffer.game_ids == ()


@pytest.mark.parametrize("replay_purpose_seed", [10_000, 20_000])
def test_bootstrap_and_training_seeds_are_explicit_replay_owners(
    replay_purpose_seed: int,
) -> None:
    parent = "a" * 64
    dataset = _dataset("generation-1", parent, {"eligible-game": 1})
    row = dataset.rows[0].model_copy(update={"world_seed": replay_purpose_seed})
    dataset = replace(
        dataset,
        rows=(row,),
        rows_by_game=MappingProxyType({"eligible-game": (row,)}),
    )
    buffer = ReplayBuffer()

    buffer.add_generation(dataset, champion_parent=_manifest(parent))

    assert buffer.sample(game_count=1, seed=replay_purpose_seed).game_ids == ("eligible-game",)


def test_replay_seed_ownership_policy_can_be_injected() -> None:
    parent = "a" * 64
    registry = SeedRegistry({"self_play": SeedRange(7, 9), "arena": SeedRange(9, 11)})
    dataset = _dataset("generation-1", parent, {"eligible-game": 1})
    row = dataset.rows[0].model_copy(update={"world_seed": 7})
    dataset = replace(
        dataset,
        rows=(row,),
        rows_by_game=MappingProxyType({"eligible-game": (row,)}),
    )
    buffer = ReplayBuffer(
        seed_registry=registry,
        replay_seed_purposes=("self_play",),
    )

    buffer.add_generation(dataset, champion_parent=_manifest(parent))

    assert buffer.sample(game_count=1, seed=8).game_ids == ("eligible-game",)


@pytest.mark.parametrize("arena_seed", [1_000_000, 1_010_000, 1_020_000])
def test_rejects_arena_seed_before_replay_sampling(arena_seed: int) -> None:
    parent = "a" * 64
    buffer = ReplayBuffer()
    buffer.add_generation(
        _dataset("generation-1", parent, {"eligible-game": 1}),
        champion_parent=_manifest(parent),
    )

    with pytest.raises(ValueError, match="not eligible for replay"):
        buffer.sample(game_count=1, seed=arena_seed)


def test_samples_50_30_20_by_whole_game_reproducibly_with_equal_game_weight() -> None:
    parent = "a" * 64
    buffer = ReplayBuffer()
    buffer.add_generation(
        _dataset("generation-1", parent, {f"historical-{i}": i + 1 for i in range(5)}),
        champion_parent=_manifest(parent),
        historical_champion=True,
    )
    buffer.add_generation(
        _dataset("generation-2", parent, {f"recent-{i}": i + 1 for i in range(5)}),
        champion_parent=_manifest(parent),
    )
    buffer.add_generation(
        _dataset("generation-3", parent, {f"latest-{i}": i + 1 for i in range(5)}),
        champion_parent=_manifest(parent),
    )

    first = buffer.sample(game_count=10, seed=20_712)
    second = buffer.sample(game_count=10, seed=20_712)

    assert first.game_ids == second.game_ids
    assert tuple(game.source for game in first.games).count("LATEST") == 5
    assert tuple(game.source for game in first.games).count("RECENT") == 3
    assert tuple(game.source for game in first.games).count("HISTORICAL_HARD") == 2
    assert {game.champion_parent_digest for game in first.games} == {parent}
    assert len(first.rows) == sum(len(game.rows) for game in first.games)
    assert {row.game_id for row in first.rows} == set(first.game_ids)
    assert sum(first.decision_weights) == pytest.approx(1.0)
    for game in first.games:
        indexes = [index for index, row in enumerate(first.rows) if row.game_id == game.game_id]
        assert sum(first.decision_weights[index] for index in indexes) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("game_count", "expected"),
    [
        (1, (1, 0, 0)),
        (2, (1, 1, 0)),
        (3, (1, 1, 1)),
        (4, (2, 1, 1)),
        (5, (3, 1, 1)),
        (6, (3, 2, 1)),
    ],
)
def test_small_pool_quotas_use_deterministic_largest_remainders(
    game_count: int, expected: tuple[int, int, int]
) -> None:
    parent = "a" * 64
    buffer = ReplayBuffer()
    buffer.add_generation(
        _dataset("generation-1", parent, {f"hard-{i}": 1 for i in range(6)}),
        champion_parent=_manifest(parent),
        historical_champion=True,
    )
    buffer.add_generation(
        _dataset("generation-2", parent, {f"recent-{i}": 1 for i in range(6)}),
        champion_parent=_manifest(parent),
    )
    buffer.add_generation(
        _dataset("generation-3", parent, {f"latest-{i}": 1 for i in range(6)}),
        champion_parent=_manifest(parent),
    )

    sample = buffer.sample(game_count=game_count, seed=20_100 + game_count)
    sources = tuple(game.source for game in sample.games)

    assert (
        tuple(sources.count(source) for source in ("LATEST", "RECENT", "HISTORICAL_HARD"))
        == expected
    )


def test_insufficient_strata_redistribute_deterministically_to_eligible_games() -> None:
    parent = "a" * 64
    buffer = ReplayBuffer()
    buffer.add_generation(
        _dataset("generation-1", parent, {f"hard-{i}": 1 for i in range(4)}),
        champion_parent=_manifest(parent),
        historical_champion=True,
    )
    buffer.add_generation(
        _dataset("generation-2", parent, {f"latest-{i}": 1 for i in range(6)}),
        champion_parent=_manifest(parent),
    )

    sample = buffer.sample(game_count=10, seed=20_200)
    sources = tuple(game.source for game in sample.games)

    assert sources.count("LATEST") == 6
    assert sources.count("RECENT") == 0
    assert sources.count("HISTORICAL_HARD") == 4


def test_sampling_fails_only_when_total_eligible_pool_is_too_small() -> None:
    parent = "a" * 64
    buffer = ReplayBuffer()
    buffer.add_generation(
        _dataset("generation-1", parent, {"only-game": 1}),
        champion_parent=_manifest(parent),
    )

    with pytest.raises(ValueError, match="has 1 games, requires 2"):
        buffer.sample(game_count=2, seed=20_250)


def test_latest_hard_case_is_reclassified_into_historical_hard_stratum() -> None:
    parent = "a" * 64
    buffer = ReplayBuffer()
    buffer.add_generation(
        _dataset("generation-1", parent, {"latest": 1, "latest-hard": 1}),
        champion_parent=_manifest(parent),
        hard_game_ids=("latest-hard",),
    )

    sample = buffer.sample(game_count=2, seed=20_300)

    assert {(game.game_id, game.source) for game in sample.games} == {
        ("latest", "LATEST"),
        ("latest-hard", "HISTORICAL_HARD"),
    }


def test_hard_cases_survive_capacity_eviction_permanently() -> None:
    parent = "a" * 64
    buffer = ReplayBuffer(max_games=3)
    buffer.add_generation(
        _dataset("generation-1", parent, {"hard-old": 1, "ordinary-old": 1}),
        champion_parent=_manifest(parent),
        hard_game_ids=("hard-old",),
    )
    buffer.add_generation(
        _dataset("generation-2", parent, {"new-1": 1, "new-2": 1, "new-3": 1}),
        champion_parent=_manifest(parent),
    )

    assert buffer.game_ids == ("hard-old", "new-2", "new-3")
    assert buffer.hard_game_ids == ("hard-old",)


def test_rejects_generation_parent_schema_and_artifact_incompatibility() -> None:
    parent = "a" * 64
    buffer = ReplayBuffer()
    buffer.add_generation(
        _dataset("generation-1", parent, {"baseline": 1}),
        champion_parent=_manifest(parent),
    )

    wrong_parent = _dataset("generation-2", "b" * 64, {"wrong-parent": 1})
    with pytest.raises(ReplayCompatibilityError, match="parent digest"):
        buffer.add_generation(wrong_parent, champion_parent=_manifest(parent))

    wrong_schema = _dataset("generation-2", parent, {"wrong-schema": 1})
    wrong_schema = replace(
        wrong_schema,
        metadata=JointDatasetMetadata.model_construct(
            schema_version=2,
            dataset_digest=wrong_schema.digest,
            row_count=1,
            game_ids=wrong_schema.game_ids,
        ),
    )
    with pytest.raises(ReplayCompatibilityError, match="dataset schema"):
        buffer.add_generation(wrong_schema, champion_parent=_manifest(parent))

    incompatible_artifact = _dataset("generation-2", parent, {"wrong-artifact": 1})
    with pytest.raises(ReplayCompatibilityError, match="artifact is incompatible"):
        buffer.add_generation(
            incompatible_artifact,
            champion_parent=_manifest(parent, tensor_digest="e" * 64),
        )

    mixed_generation = _dataset("generation-2", parent, {"first": 1, "second": 1})
    altered = mixed_generation.rows[1].model_copy(update={"generation_id": "generation-3"})
    mixed_generation = replace(
        mixed_generation,
        rows=(mixed_generation.rows[0], altered),
        rows_by_game=MappingProxyType({"first": (mixed_generation.rows[0],), "second": (altered,)}),
    )
    with pytest.raises(ReplayCompatibilityError, match="exactly one generation"):
        buffer.add_generation(mixed_generation, champion_parent=_manifest(parent))
