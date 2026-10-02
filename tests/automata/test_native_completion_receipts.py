"""Controlled native completion receipts bind normal games to exact source bytes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automata.harness.game_runner import DEFAULT_MAP
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import detect_stable_value_boundary
from automata.training.native_dataset import NativeGameIdentity, native_game_id
from automata.training.native_indexed_dataset import (
    create_native_source_receipt,
    create_native_source_receipt_from_completions,
)
from automata.training.native_receipts import (
    NATIVE_COMPLETION_CONTRACT,
    NativeCompletionTarget,
    NativeDatasetCompletionReceipt,
    NativeGameCompletionReceipt,
    create_native_dataset_completion_receipt,
    load_native_dataset_completion_receipt,
    load_native_game_completion_receipt,
    validate_native_dataset_completion,
)
from automata.training.native_recorder import NativeDatasetRecorder
from goa2.engine.setup import GameSetup


def _identity(seed: int = 73) -> NativeGameIdentity:
    fields: dict[str, Any] = {
        "world_seed": seed,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": "receipt-generation",
        "source_revision": "abc123",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "search-config-1",
        "generator_config_id": "generator-config-1",
    }
    return NativeGameIdentity(game_id=native_game_id(**fields), **fields)


def _record_completed(
    root: Path,
    *,
    seed: int = 73,
    logical_name: str | None = None,
    receipt_name: str | None = None,
) -> tuple[Path, Path, NativeGameCompletionReceipt]:
    register_all_effects()
    source_root = root / "source"
    receipts_root = root / "receipts"
    source_root.mkdir(parents=True)
    receipts_root.mkdir(parents=True)
    logical_name = logical_name or f"game-{seed}.jsonl.zst"
    receipt_name = receipt_name or f"game-{seed}.completion.json"
    source = source_root / logical_name
    source.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_root / receipt_name
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    state = GameSetup.create_game(
        DEFAULT_MAP,
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=seed,
    )
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder = NativeDatasetRecorder(
        source,
        game=_identity(seed),
        completion_target=NativeCompletionTarget(
            source_root=source_root,
            receipt_path=receipt_path,
        ),
    )
    assert recorder.completion_receipt is None
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))
    recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")
    assert recorder.completion_receipt is not None
    return source, receipt_path, recorder.completion_receipt


def test_controlled_recorder_issues_canonical_receipt_from_live_counts(tmp_path: Path) -> None:
    source, receipt_path, receipt = _record_completed(
        tmp_path, logical_name="nested/game.jsonl.zst"
    )

    assert receipt.completion_contract == NATIVE_COMPLETION_CONTRACT
    assert receipt.logical_name == "nested/game.jsonl.zst"
    assert receipt.game == _identity()
    assert (
        receipt.row_count,
        receipt.policy_row_count,
        receipt.value_row_count,
        receipt.boundary_count,
        receipt.reason,
        receipt.terminal_winner,
    ) == (1, 0, 1, 1, "game_over", "RED")
    assert load_native_game_completion_receipt(receipt_path) == receipt
    assert receipt_path.read_bytes() == receipt.canonical_bytes()
    assert receipt.digest

    completion = create_native_dataset_completion_receipt(tmp_path / "source", [receipt_path])
    assert completion.games == (receipt,)
    assert validate_native_dataset_completion(tmp_path / "source", completion)
    assert source.is_file()


def test_dataset_completion_preserves_explicit_order_and_adapts_to_inventory(
    tmp_path: Path,
) -> None:
    first_source, first_path, first = _record_completed(tmp_path / "first", seed=73)
    second_source, second_path, second = _record_completed(tmp_path / "second", seed=74)
    source_root = tmp_path / "combined"
    source_root.mkdir()
    for source, receipt in ((first_source, first), (second_source, second)):
        target = source_root / receipt.logical_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    completion = create_native_dataset_completion_receipt(source_root, [second_path, first_path])
    inventory = create_native_source_receipt_from_completions(source_root, completion)

    assert completion.games == (second, first)
    assert tuple(game.logical_name for game in inventory.games) == (
        second.logical_name,
        first.logical_name,
    )
    assert tuple(game.game_id for game in inventory.games) == (
        second.game.game_id,
        first.game.game_id,
    )
    assert inventory != create_native_source_receipt(
        source_root, [first.logical_name, second.logical_name]
    )

    constructed = NativeDatasetCompletionReceipt.model_construct(
        games=[second.model_dump(mode="python")]
    )
    normalized_inventory = create_native_source_receipt_from_completions(source_root, constructed)
    assert normalized_inventory.games[0].logical_name == second.logical_name

    path = tmp_path / "dataset-completion.json"
    path.write_bytes(completion.canonical_bytes())
    assert load_native_dataset_completion_receipt(path) == completion
    path.write_bytes(json.dumps(completion.model_dump(mode="json"), indent=2).encode())
    with pytest.raises(ValueError, match="canonical"):
        load_native_dataset_completion_receipt(path)


@pytest.mark.parametrize("ending", ["censored", "empty", "exception"])
def test_controlled_recorder_never_receipts_noncompleted_games(tmp_path: Path, ending: str) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    receipt_path = tmp_path / "receipt.json"
    source = source_root / "game.jsonl"
    recorder = NativeDatasetRecorder(
        source,
        game=_identity(),
        completion_target=NativeCompletionTarget(source_root, receipt_path),
    )
    if ending != "empty":
        register_all_effects()
        state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=73)
        boundary = detect_stable_value_boundary(state)
        assert boundary is not None
        recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))

    if ending == "censored":
        recorder.record_outcome(winner_side=None, rounds=2, reason="max_steps")
    elif ending == "empty":
        recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")
    else:
        recorder.close()

    assert recorder.completion_receipt is None
    assert not source.exists()
    assert not receipt_path.exists()


def test_invalid_winner_poisoning_does_not_publish_receipt(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    receipt_path = tmp_path / "receipt.json"
    recorder = NativeDatasetRecorder(
        source_root / "game.jsonl",
        game=_identity(),
        completion_target=NativeCompletionTarget(source_root, receipt_path),
    )
    with pytest.raises(ValueError, match="winner"):
        recorder.record_outcome(winner_side=None, rounds=1, reason="game_over")
    assert recorder.completion_receipt is None
    assert not receipt_path.exists()


def test_source_tamper_count_tamper_and_winner_tamper_are_rejected(tmp_path: Path) -> None:
    source, _, receipt = _record_completed(tmp_path)
    completion = NativeDatasetCompletionReceipt(games=(receipt,))

    original = source.read_bytes()
    source.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match=r"hash|size|trailing|frame"):
        validate_native_dataset_completion(tmp_path / "source", completion)
    source.write_bytes(original)

    for update, match in (
        ({"row_count": 2, "value_row_count": 2}, "count"),
        ({"terminal_winner": "BLUE"}, "winner"),
    ):
        bad = receipt.model_copy(update=update)
        bad_set = NativeDatasetCompletionReceipt.model_construct(games=(bad,))
        with pytest.raises(ValueError, match=match):
            validate_native_dataset_completion(tmp_path / "source", bad_set)


def test_receipt_models_reject_unsafe_paths_duplicate_identity_and_invalid_counts(
    tmp_path: Path,
) -> None:
    _, _, receipt = _record_completed(tmp_path)
    with pytest.raises(ValidationError, match=r"logical|relative|normalized"):
        NativeGameCompletionReceipt(
            **{**receipt.model_dump(mode="python"), "logical_name": "../game.jsonl"}
        )
    with pytest.raises(ValidationError, match="count"):
        NativeGameCompletionReceipt(
            **{
                **receipt.model_dump(mode="python"),
                "row_count": 2,
                "policy_row_count": 0,
                "value_row_count": 1,
            }
        )
    with pytest.raises(ValidationError, match=r"logical names|game IDs"):
        NativeDatasetCompletionReceipt(games=(receipt, receipt))

    completion = NativeDatasetCompletionReceipt(games=(receipt,))
    for model, valid in (
        (NativeGameCompletionReceipt, receipt),
        (NativeDatasetCompletionReceipt, completion),
    ):
        for invalid_version in (True, 1.0):
            payload = valid.model_dump(mode="json")
            payload["schema_version"] = invalid_version
            with pytest.raises(ValidationError, match="schema_version"):
                model.model_validate_json(json.dumps(payload, separators=(",", ":")))


def test_controlled_paths_reject_symlinks_escapes_and_collisions_before_spooling(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = source_root / "linked"
    symlink.symlink_to(outside, target_is_directory=True)
    before = set(tmp_path.rglob("*"))

    targets = (
        (outside / "escape.jsonl", tmp_path / "escape.receipt.json"),
        (symlink / "game.jsonl", tmp_path / "symlink.receipt.json"),
        (source_root / "same.jsonl", source_root / "same.jsonl"),
    )
    for source, receipt in targets:
        with pytest.raises(ValueError, match=r"root|inside|symlink|overlap|collision"):
            NativeDatasetRecorder(
                source,
                game=_identity(),
                completion_target=NativeCompletionTarget(source_root, receipt),
            )
    assert set(tmp_path.rglob("*")) == before

    nested_real_root = outside / "nested-root"
    nested_real_root.mkdir()
    nested_alias_root = symlink / "nested-root"
    with pytest.raises(ValueError, match="symlink"):
        NativeDatasetRecorder(
            nested_alias_root / "game.jsonl",
            game=_identity(),
            completion_target=NativeCompletionTarget(
                nested_alias_root, nested_real_root / "game.jsonl"
            ),
        )
    assert not list(nested_real_root.iterdir())


@pytest.mark.parametrize(
    ("source_relative", "receipt_relative"),
    [
        ("game.jsonl", "game.jsonl/receipt.json"),
        ("nested/game.jsonl", "nested"),
    ],
)
def test_controlled_source_and_receipt_ancestor_collisions_have_no_side_effects(
    tmp_path: Path, source_relative: str, receipt_relative: str
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    before = set(tmp_path.rglob("*"))

    with pytest.raises(ValueError, match="overlap"):
        NativeDatasetRecorder(
            source_root / source_relative,
            game=_identity(),
            completion_target=NativeCompletionTarget(source_root, source_root / receipt_relative),
        )

    assert set(tmp_path.rglob("*")) == before


def test_controlled_paths_are_anchored_when_working_directory_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = tmp_path / "start"
    other = tmp_path / "other"
    start.mkdir()
    other.mkdir()
    monkeypatch.chdir(start)
    source_root = Path("data")
    source_root.mkdir()
    recorder = NativeDatasetRecorder(
        Path("data/game.jsonl"),
        game=_identity(),
        completion_target=NativeCompletionTarget(source_root, Path("data/game.completion.json")),
    )
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=73)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))

    monkeypatch.chdir(other)
    recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    assert (start / "data/game.jsonl").is_file()
    assert (start / "data/game.completion.json").is_file()
    assert not list(other.iterdir())


def test_path_topology_is_rechecked_before_controlled_publication(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    receipts = tmp_path / "receipts"
    outside = tmp_path / "outside"
    source_root.mkdir()
    receipts.mkdir()
    outside.mkdir()
    source = source_root / "game.jsonl"
    recorder = NativeDatasetRecorder(
        source,
        game=_identity(),
        completion_target=NativeCompletionTarget(source_root, receipts / "game.completion.json"),
    )
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=73)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))
    receipts.rename(tmp_path / "receipts-original")
    receipts.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    assert not source.exists()
    assert not list(outside.iterdir())


def test_sidecar_race_rolls_back_only_newly_published_source(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "game.jsonl"
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=73)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder = NativeDatasetRecorder(
        source,
        game=_identity(),
        completion_target=NativeCompletionTarget(source_root, receipt_path),
    )
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))
    receipt_path.write_bytes(b"competitor receipt")

    with pytest.raises(FileExistsError):
        recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    assert recorder.completion_receipt is None
    assert receipt_path.read_bytes() == b"competitor receipt"
    assert not source.exists()


def test_receipt_interrupt_rolls_back_the_newly_published_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import automata.training.native_recorder as recorder_module

    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "game.jsonl"
    receipt_path = tmp_path / "receipt.json"
    recorder = NativeDatasetRecorder(
        source,
        game=_identity(),
        completion_target=NativeCompletionTarget(source_root, receipt_path),
    )
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=73)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt("receipt interrupted")

    monkeypatch.setattr(recorder_module, "_publish_completion_receipt", interrupt)
    with pytest.raises(KeyboardInterrupt, match="receipt interrupted"):
        recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    assert recorder.completion_receipt is None
    assert not source.exists()
    assert not receipt_path.exists()


def test_existing_source_race_is_preserved_and_never_receipted(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "game.jsonl"
    receipt_path = tmp_path / "receipt.json"
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=73)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder = NativeDatasetRecorder(
        source,
        game=_identity(),
        completion_target=NativeCompletionTarget(source_root, receipt_path),
    )
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))
    source.write_bytes(b"competitor source")

    with pytest.raises(FileExistsError):
        recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    assert source.read_bytes() == b"competitor source"
    assert not receipt_path.exists()
    assert recorder.completion_receipt is None


def test_unreceipted_valid_file_cannot_be_upgraded_by_completion_api(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = raw_root / "raw.jsonl"
    # Raw recorder behavior remains valid but deliberately has no completion provenance.
    recorder = NativeDatasetRecorder(raw, game=_identity())
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=73)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))
    recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")
    assert raw.is_file() and recorder.completion_receipt is None

    with pytest.raises((TypeError, ValueError)):
        create_native_dataset_completion_receipt(raw_root, [raw])


def test_receipt_and_recorder_imports_do_not_import_torch() -> None:
    code = """
import sys
import automata.training.native_receipts
import automata.training.native_recorder
assert 'torch' not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
