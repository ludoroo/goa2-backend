"""Behavioral contract for deterministic, leakage-safe joint-data splits."""

from __future__ import annotations

from types import MappingProxyType

import pytest
from pydantic import ValidationError

from automata.training.dataset import (
    JointDataset,
    JointDatasetMetadata,
    JointDatasetRow,
)
from automata.training.splits import (
    JointSplitConfig,
    JointSplitManifest,
    apply_joint_split_manifest,
    split_joint_dataset,
)


def _dataset(*games: tuple[str, int, str, str, tuple[str, ...], tuple[str, ...]]) -> JointDataset:
    rows = tuple(
        JointDatasetRow.model_construct(
            game_id=game_id,
            world_seed=seed,
            decision_index=decision_index,
            map_id=map_id,
            game_type=game_type,
            red_composition=red,
            blue_composition=blue,
        )
        for game_id, seed, map_id, game_type, red, blue in games
        for decision_index in range(2)
    )
    digest = f"{sum(seed for _, seed, *_ in games):064x}"
    grouped = MappingProxyType(
        {game_id: tuple(row for row in rows if row.game_id == game_id) for game_id, *_ in games}
    )
    metadata = JointDatasetMetadata(
        schema_version=1,
        dataset_digest=digest,
        row_count=len(rows),
        game_ids=tuple(game_id for game_id, *_ in games),
    )
    return JointDataset(rows=rows, digest=digest, metadata=metadata, rows_by_game=grouped)


def test_split_is_deterministic_stratified_and_keeps_whole_games_together() -> None:
    dataset = _dataset(
        ("a1", 20_001, "island", "QUICK", ("Wasp",), ("Arien",)),
        ("a2", 20_002, "island", "QUICK", ("Wasp",), ("Arien",)),
        ("b1", 20_003, "island", "LONG", ("Brogan",), ("Xargatha",)),
        ("b2", 20_004, "island", "LONG", ("Brogan",), ("Xargatha",)),
    )
    config = JointSplitConfig(seed=17, validation_fraction=0.5)

    first = split_joint_dataset(dataset, config=config)
    second = split_joint_dataset(dataset, config=config)

    assert first.manifest == second.manifest
    assert len(first.game_ids("train")) == 2
    assert len(first.game_ids("validation")) == 2
    for prefix in ("a", "b"):
        assert sum(game_id.startswith(prefix) for game_id in first.game_ids("train")) == 1
        assert sum(game_id.startswith(prefix) for game_id in first.game_ids("validation")) == 1
    assert {row.game_id for row in first.rows("train")} == set(first.game_ids("train"))
    assert {row.game_id for row in first.rows("validation")} == set(first.game_ids("validation"))


def test_declared_map_composition_and_game_mode_holdouts_are_reserved() -> None:
    dataset = _dataset(
        ("train-1", 20_011, "island", "QUICK", ("Wasp",), ("Arien",)),
        ("train-2", 20_012, "island", "QUICK", ("Wasp",), ("Arien",)),
        ("map", 20_013, "caverns", "QUICK", ("Wasp",), ("Arien",)),
        ("composition", 20_014, "island", "QUICK", ("Brogan",), ("Arien",)),
        ("mode", 20_015, "island", "LONG", ("Wasp",), ("Arien",)),
    )
    config = JointSplitConfig(
        seed=4,
        validation_fraction=0.5,
        holdout_map_ids=("caverns",),
        holdout_compositions=(("Brogan",),),
        holdout_game_modes=("LONG",),
    )

    splits = split_joint_dataset(dataset, config=config)

    assert splits.game_ids("map_holdout") == ("map",)
    assert splits.game_ids("composition_holdout") == ("composition",)
    assert splits.game_ids("game_mode_holdout") == ("mode",)
    assert set(splits.game_ids("train") + splits.game_ids("validation")) == {
        "train-1",
        "train-2",
    }


def test_seed_from_another_registered_purpose_is_rejected_as_leakage() -> None:
    dataset = _dataset(
        ("training", 20_021, "island", "QUICK", ("Wasp",), ("Arien",)),
        ("arena", 1_010_021, "island", "QUICK", ("Wasp",), ("Arien",)),
    )

    with pytest.raises(ValueError, match="belongs to 'screen'"):
        split_joint_dataset(dataset, config=JointSplitConfig(seed_purpose="training"))


def test_duplicate_world_seed_across_games_is_rejected() -> None:
    dataset = _dataset(
        ("first", 20_031, "island", "QUICK", ("Wasp",), ("Arien",)),
        ("second", 20_031, "island", "QUICK", ("Wasp",), ("Arien",)),
    )

    with pytest.raises(ValueError, match="world seed 20031 belongs to multiple games"):
        split_joint_dataset(dataset)


def test_manifest_is_immutable_canonical_and_reuses_exact_membership() -> None:
    dataset = _dataset(
        ("game-a", 20_041, "island", "QUICK", ("Wasp",), ("Arien",)),
        ("game-b", 20_042, "island", "QUICK", ("Wasp",), ("Arien",)),
    )
    created = split_joint_dataset(dataset, config=JointSplitConfig(seed=9, validation_fraction=0.5))

    restored_manifest = JointSplitManifest.model_validate_json(created.manifest.canonical_bytes())
    reused = apply_joint_split_manifest(dataset, restored_manifest)

    assert reused.manifest == created.manifest
    assert reused.digest == created.digest
    assert reused.game_ids("train") == created.game_ids("train")
    assert reused.game_ids("validation") == created.game_ids("validation")
    assert created.manifest.canonical_bytes() == (
        b'{"config":{"holdout_compositions":[],"holdout_game_modes":[],'
        b'"holdout_map_ids":[],"schema_version":1,"seed":9,'
        b'"seed_purpose":"training","validation_fraction":0.5},'
        b'"dataset_digest":"0000000000000000000000000000000000000000000000000000000000009c93",'
        b'"memberships":[{"game_id":"game-a","split":"train",'
        b'"world_seed":20041},{"game_id":"game-b","split":"validation",'
        b'"world_seed":20042}],"schema_version":1}'
    )
    assert created.digest == "6b5de31a6d7018bdea628d5bc8b3e5742befdcca2b7c4ef0dba1fc37d8952f5e"
    with pytest.raises(ValidationError):
        created.manifest.config.seed = 10


def test_manifest_reuse_rejects_changed_dataset_and_membership() -> None:
    dataset = _dataset(
        ("game-a", 20_051, "island", "QUICK", ("Wasp",), ("Arien",)),
        ("game-b", 20_052, "island", "QUICK", ("Wasp",), ("Arien",)),
    )
    manifest = split_joint_dataset(dataset).manifest
    changed = _dataset(
        ("game-a", 20_051, "island", "QUICK", ("Wasp",), ("Arien",)),
        ("game-c", 20_053, "island", "QUICK", ("Wasp",), ("Arien",)),
    )

    with pytest.raises(ValueError, match="dataset digest"):
        apply_joint_split_manifest(changed, manifest)

    missing = manifest.model_copy(update={"memberships": manifest.memberships[:-1]})
    with pytest.raises(ValueError, match="exactly match dataset games"):
        apply_joint_split_manifest(dataset, missing)
