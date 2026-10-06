"""Finite native-run contracts and safety behavior.

The terminal fixture exercises the real native pipeline under pytest only.  It
uses the same deliberately short raw-stack wiring as the existing native
trainer/generator flow tests; generation, recording, receipts, indexing,
replay, validation, optimization, and artifact IO remain concrete.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from test_native_trainer_generator_flow import terminal_game as terminal_game

import automata.training.native_run as native_run_module
from automata.search.contracts import CutoffUnit, LeafMode
from automata.training.native_generation import NativeGenerationGame
from automata.training.native_replay import NativeReplayConfig
from automata.training.native_run import run_native_one
from automata.training.native_run_contracts import (
    NativeRunAuthorities,
    NativeRunCompletion,
    NativeRunConfig,
    NativeRunFailure,
    NativeRunResult,
    NativeRunSearchConfig,
    create_native_run_manifest,
    load_completed_native_run,
    load_native_run_result,
)
from automata.training.native_splits import (
    NativeSeedRange,
    NativeSplitConfig,
    create_native_split_ledger,
    extend_native_split_ledger,
)
from automata.training.native_trainer import NativeTrainerConfig, NativeTrainerInitialization


def _split() -> NativeSplitConfig:
    return NativeSplitConfig(
        namespace="native-one-run-tests",
        salt="fixed",
        validation_fraction=0.2,
        seed_ranges=(
            NativeSeedRange(purpose="training", start=0, stop=100),
            NativeSeedRange(purpose="validation", start=100, stop=102),
        ),
    )


def _train_seed(split: NativeSplitConfig) -> int:
    ledger = extend_native_split_ledger(create_native_split_ledger(split), range(100))
    return next(item.world_seed for item in ledger.assignments if item.split == "train")


def _game(seed: int, purpose: str) -> NativeGenerationGame:
    return NativeGenerationGame(
        world_seed=seed,
        seed_purpose=purpose,
        map_id="forgotten_island",
        game_type="QUICK",
        red_composition=("Wasp",),
        blue_composition=("Arien",),
    )


def _search() -> NativeRunSearchConfig:
    return NativeRunSearchConfig(
        iterations=2,
        decision_timeout_seconds=None,
        max_advance_steps=100,
        uct_c=1.4,
        cutoff_limit=2,
        cutoff_unit=CutoffUnit.ROUNDS,
        max_advance_transitions=100,
        max_forced_decisions=50,
        leaf_mode=LeafMode.STABLE_TRANSITION,
        widening_c=2.0,
        widening_alpha=0.5,
        root_widening_c=None,
        root_widening_alpha=None,
        adaptive_hex_root_schedule_version=None,
        request_schedule_version=None,
        seed=0,
        use_prior=True,
        puct_c=0.0,
        root_puct_c=None,
    )


def _trainer() -> NativeTrainerConfig:
    return NativeTrainerConfig(
        seed=7,
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        policy_weight=1.0,
        value_weight=1.0,
        entropy_weight=0.01,
        l2_weight=0.0001,
        max_gradient_norm=1.0,
        token_width=8,
        state_width=12,
        candidate_width=8,
        message_passing_layers=1,
        dropout=0.0,
    )


def _config(**changes: Any) -> NativeRunConfig:
    split = _split()
    train_seed = _train_seed(split)
    values: dict[str, Any] = {
        "run_id": "bounded-native-run-test",
        "generation_id": "bounded-native-generation-test",
        "source_revision": "test-revision",
        "dirty_tree_hash": "fixture",
        "teacher_kind": "HEURISTIC_BOOTSTRAP",
        "source_model_digest": None,
        "search": _search(),
        "split_config": split,
        "random_stream_namespace": "bounded-native-run-streams",
        "visit_temperature": 1.0,
        "max_steps": 30,
        "max_rounds": None,
        "games": (
            _game(train_seed, "training"),
            _game(100, "validation"),
        ),
        "replay_config": NativeReplayConfig(split_config=split, max_games=1),
        "trainer_config": _trainer(),
        "initialization": NativeTrainerInitialization(mode="FRESH_BOOTSTRAP"),
        "optimizer_steps": 1,
        "games_per_step": 1,
        "sampling_seeds": (19,),
        "index_chunk_size": 1,
    }
    values.update(changes)
    return NativeRunConfig(**values)


def test_manifest_derives_complete_finite_plan_without_touching_root(tmp_path: Path) -> None:
    root = (tmp_path / "run").absolute()

    manifest = create_native_run_manifest(
        _config(), NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )

    assert not root.exists()
    assert tuple(item.ordinal for item in manifest.planned_games) == (0, 1)
    assert tuple(item.source_logical_name for item in manifest.planned_games) == (
        "games/00000000.jsonl.zst",
        "games/00000001.jsonl.zst",
    )
    assert tuple(item.split for item in manifest.planned_games) == ("train", "validation")
    assert tuple(item.game.world_seed for item in manifest.planned_games) == tuple(
        item.world_seed for item in manifest.planned_split_ledger.assignments
    )
    assert (
        manifest.digest
        == create_native_run_manifest(
            _config(), NativeRunAuthorities(output_root=root, parent_artifact_path=None)
        ).digest
    )


def test_config_digest_excludes_authority_but_manifest_digest_binds_it(
    tmp_path: Path,
) -> None:
    config = _config()
    first = create_native_run_manifest(
        config,
        NativeRunAuthorities(
            output_root=(tmp_path / "first").absolute(), parent_artifact_path=None
        ),
    )
    second = create_native_run_manifest(
        config,
        NativeRunAuthorities(
            output_root=(tmp_path / "second").absolute(), parent_artifact_path=None
        ),
    )

    assert first.config.digest == second.config.digest == config.digest
    assert first.planned_split_ledger.digest == second.planned_split_ledger.digest
    assert first.digest != second.digest


def test_config_rejects_unbounded_or_inconsistent_explicit_budgets() -> None:
    base = _config()
    with pytest.raises(ValidationError, match="sampling"):
        _config(optimizer_steps=2)
    with pytest.raises(ValidationError, match="max_games"):
        _config(replay_config=base.replay_config.model_copy(update={"max_games": None}))
    with pytest.raises(ValidationError, match="games_per_step"):
        _config(games_per_step=2)


def test_config_rejects_invalid_cohorts_and_replay_authority() -> None:
    base = _config()
    train = base.games[0]
    validation = base.games[1]
    second_train_seed = next(
        assignment.world_seed
        for assignment in extend_native_split_ledger(
            create_native_split_ledger(base.split_config), range(100)
        ).assignments
        if assignment.split == "train" and assignment.world_seed != train.world_seed
    )
    two_train_games = (train, _game(second_train_seed, "training"), validation)
    mismatched_split = base.split_config.model_copy(update={"salt": "other"})

    for changes, match in (
        ({"games": (train,)}, "cohorts"),
        ({"games": (validation,)}, "cohorts"),
        ({"games": (train, train, validation)}, "identities"),
        (
            {
                "games": two_train_games,
                "replay_config": NativeReplayConfig(split_config=base.split_config, max_games=1),
            },
            "retain",
        ),
        (
            {"replay_config": NativeReplayConfig(split_config=mismatched_split, max_games=1)},
            "split_config",
        ),
    ):
        with pytest.raises(ValidationError, match=match):
            _config(**changes)


def test_config_rejects_invalid_temperature_and_seed_purpose() -> None:
    base = _config()
    with pytest.raises(ValidationError, match="temperature"):
        _config(visit_temperature=float("nan"))
    with pytest.raises(ValidationError, match="temperature"):
        _config(visit_temperature=1)
    with pytest.raises(ValidationError, match="seed_purpose"):
        _config(games=(base.games[0], _game(100, "training")))


def test_authorities_reject_parent_overlap_and_non_normalized_paths(tmp_path: Path) -> None:
    root = (tmp_path / "run").absolute()
    with pytest.raises(ValidationError, match="disjoint"):
        NativeRunAuthorities(output_root=root, parent_artifact_path=root / "artifact")
    with pytest.raises(ValidationError, match="normalized"):
        NativeRunAuthorities(
            output_root=tmp_path.absolute() / "other" / ".." / "run",
            parent_artifact_path=None,
        )


def test_manifest_rejects_unsafe_nested_copy_and_illegal_mode_before_root_creation(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "run").absolute()
    config = _config()
    unsafe = config.model_copy(update={"sampling_seeds": (True,)})
    with pytest.raises(ValidationError):
        create_native_run_manifest(
            unsafe, NativeRunAuthorities(output_root=root, parent_artifact_path=None)
        )
    unsafe_search = config.search.model_copy(update={"seed": True})
    with pytest.raises(ValidationError):
        unsafe_search.to_search_config()
    with pytest.raises(ValidationError, match=r"teacher|initialization|parent"):
        _config(
            teacher_kind="GEN1_PARENT",
            source_model_digest="a" * 64,
        )
    assert not root.exists()


def test_real_bounded_pipeline_reaches_digest_bound_completion(
    tmp_path: Path, terminal_game: None
) -> None:
    root = (tmp_path / "run").absolute()
    manifest = create_native_run_manifest(
        _config(), NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )

    result = run_native_one(manifest)

    loaded_manifest, loaded_result = load_completed_native_run(root)
    assert result.status == "SUCCEEDED"
    assert result.products is not None
    assert loaded_manifest == manifest
    assert loaded_result == result
    assert result.products.generated_game_ids == tuple(
        item.expected_game_id for item in manifest.planned_games
    )
    assert result.products.validation_scope == "CURRENT_RUN_UPDATES"
    assert result.products.parent_training_exposure == "NONE_FRESH_INITIALIZATION"
    assert len(result.products.training_steps) == manifest.config.optimizer_steps

    mismatched_root = (tmp_path / "mismatched-parent-run").absolute()
    mismatched_digest = "0" * 64
    parent_config = _config().model_copy(
        update={
            "teacher_kind": "GEN1_PARENT",
            "source_model_digest": mismatched_digest,
            "initialization": NativeTrainerInitialization(
                mode="GEN1_PARENT", parent_model_digest=mismatched_digest
            ),
        }
    )
    parent_manifest = create_native_run_manifest(
        parent_config,
        NativeRunAuthorities(
            output_root=mismatched_root,
            parent_artifact_path=root / "artifact",
        ),
    )
    with pytest.raises(ValueError, match="digest"):
        run_native_one(parent_manifest)
    assert not mismatched_root.exists()

    result_path = root / "result.json"
    marker_path = root / "complete.json"
    result_payload = result_path.read_bytes()
    marker_payload = marker_path.read_bytes()

    marker_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_completed_native_run(root)
    marker_path.write_bytes(marker_payload)

    stale_marker = NativeRunCompletion(
        manifest_digest=manifest.digest,
        result_digest="0" * 64,
    )
    marker_path.write_bytes(stale_marker.canonical_bytes())
    with pytest.raises(ValueError, match="stale"):
        load_completed_native_run(root)
    marker_path.write_bytes(marker_payload)

    running = NativeRunResult(
        status="RUNNING",
        manifest_digest=manifest.digest,
        config_digest=manifest.config.digest,
        progress=result.progress,
        products=None,
        failure=None,
    )
    result_path.write_bytes(running.canonical_bytes())
    with pytest.raises(ValueError, match="not succeeded"):
        load_completed_native_run(root)

    failed = NativeRunResult(
        status="FAILED",
        manifest_digest=manifest.digest,
        config_digest=manifest.config.digest,
        progress=result.progress,
        products=None,
        failure=NativeRunFailure(
            phase=result.progress.phase,
            error_type="RuntimeError",
            message="fixture failure",
        ),
    )
    result_path.write_bytes(failed.canonical_bytes())
    with pytest.raises(ValueError, match="not succeeded"):
        load_completed_native_run(root)

    result_path.write_bytes(result_payload + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        load_completed_native_run(root)
    result_path.write_bytes(result_payload)

    provenance_path = root / "artifact" / "provenance.json"
    artifact_manifest_path = root / "artifact" / "manifest.json"
    provenance_payload = provenance_path.read_bytes()
    artifact_manifest_payload = artifact_manifest_path.read_bytes()
    provenance = json.loads(provenance_payload)
    provenance["successful_steps"][0]["total_loss"] += 1.0
    changed_provenance = json.dumps(
        provenance,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    provenance_path.write_bytes(changed_provenance)
    artifact_manifest = json.loads(artifact_manifest_payload)
    artifact_manifest["files"]["provenance.json"] = {
        "length": len(changed_provenance),
        "sha256": hashlib.sha256(changed_provenance).hexdigest(),
    }
    changed_artifact_manifest = json.dumps(
        artifact_manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    artifact_manifest_path.write_bytes(changed_artifact_manifest)
    changed_result_data = result.model_dump(mode="python")
    changed_result_data["products"]["artifact_manifest_digest"] = hashlib.sha256(
        changed_artifact_manifest
    ).hexdigest()
    changed_result = NativeRunResult.model_validate(changed_result_data, strict=True)
    result_path.write_bytes(changed_result.canonical_bytes())
    marker_path.write_bytes(
        NativeRunCompletion(
            manifest_digest=manifest.digest,
            result_digest=changed_result.digest,
        ).canonical_bytes()
    )
    with pytest.raises(ValueError, match="provenance"):
        load_completed_native_run(root)

    provenance_path.write_bytes(provenance_payload)
    artifact_manifest_path.write_bytes(artifact_manifest_payload)
    result_path.write_bytes(result_payload)
    marker_path.write_bytes(marker_payload)
    with (root / manifest.planned_games[0].source_logical_name).open("ab") as source:
        source.write(b"tamper")
    with pytest.raises(ValueError):
        load_completed_native_run(root)


def test_no_clobber_publication_never_deletes_a_racing_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "complete.json"
    competitor = b"competitor"
    real_link = native_run_module.os.link

    def replace_after_link(source: Path, destination: Path) -> None:
        real_link(source, destination)
        Path(destination).unlink()
        Path(destination).write_bytes(competitor)

    def fail_directory_sync(directory: Path) -> None:
        raise OSError("injected directory sync failure")

    monkeypatch.setattr(native_run_module.os, "link", replace_after_link)
    monkeypatch.setattr(native_run_module, "fsync_directory", fail_directory_sync)

    with pytest.raises(OSError, match="directory sync failure"):
        native_run_module._publish_new(target, b"ours")

    assert target.read_bytes() == competitor
    assert not tuple(tmp_path.glob(".complete.json.*.tmp"))


def test_interrupt_after_link_rolls_back_only_owned_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "complete.json"
    real_link = native_run_module.os.link

    def interrupt_after_link(source: Path, destination: Path) -> None:
        real_link(source, destination)
        raise KeyboardInterrupt("after link")

    monkeypatch.setattr(native_run_module.os, "link", interrupt_after_link)

    with pytest.raises(KeyboardInterrupt, match="after link"):
        native_run_module._publish_new(target, b"ours")

    assert not target.exists()
    assert not tuple(tmp_path.iterdir())


def test_root_claim_fsync_failure_persists_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "run").absolute()
    manifest = create_native_run_manifest(
        _config(), NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )
    real_fsync = native_run_module.fsync_directory

    def fail_claim_sync(directory: Path) -> None:
        if directory == root.parent:
            raise OSError("injected root claim sync failure")
        real_fsync(directory)

    monkeypatch.setattr(native_run_module, "fsync_directory", fail_claim_sync)

    with pytest.raises(OSError, match="root claim sync failure"):
        run_native_one(manifest)

    failed = load_native_run_result(root / "result.json")
    assert failed.status == "FAILED"
    assert failed.progress.phase == "CLAIMED"
    assert not (root / "complete.json").exists()


def test_post_replace_sync_failure_persists_failed_status_and_reraises_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "run").absolute()
    manifest = create_native_run_manifest(
        _config(), NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )
    real_atomic_write = native_run_module.atomic_write_bytes
    original = OSError("injected post-replace result sync failure")
    armed = True

    def replace_then_fail(path: Path, payload: bytes) -> None:
        nonlocal armed
        real_atomic_write(path, payload)
        if armed and path == root / "result.json":
            armed = False
            raise original

    monkeypatch.setattr(native_run_module, "atomic_write_bytes", replace_then_fail)

    with pytest.raises(OSError, match="post-replace result sync failure") as caught:
        run_native_one(manifest)

    assert caught.value is original
    failed = load_native_run_result(root / "result.json")
    assert failed.status == "FAILED"
    assert failed.progress.phase == "GENERATING"
    assert not (root / "complete.json").exists()


def test_failure_message_truncation_preserves_original_and_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "run").absolute()
    manifest = create_native_run_manifest(
        _config(), NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )
    original = RuntimeError("a" * 999 + " tail")

    def fail_generation(*args: Any, **kwargs: Any) -> None:
        raise original

    monkeypatch.setattr(native_run_module, "generate_native_game", fail_generation)

    with pytest.raises(RuntimeError) as caught:
        run_native_one(manifest)

    assert caught.value is original
    failed = load_native_run_result(root / "result.json")
    assert failed.status == "FAILED"
    assert failed.failure is not None
    assert failed.failure.error_type == "RuntimeError"
    assert failed.failure.message == "a" * 999
    assert not (root / "complete.json").exists()


def test_original_base_exception_survives_failed_status_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "run").absolute()
    manifest = create_native_run_manifest(
        _config(), NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )
    original = KeyboardInterrupt("injected generation interrupt")
    status_failure_armed = False
    real_replace = native_run_module._replace_owned

    def interrupted_generation(*args: Any, **kwargs: Any) -> None:
        nonlocal status_failure_armed
        status_failure_armed = True
        raise original

    def fail_only_terminal_status(*args: Any, **kwargs: Any) -> None:
        if status_failure_armed:
            raise OSError("injected status failure")
        real_replace(*args, **kwargs)

    monkeypatch.setattr(native_run_module, "generate_native_game", interrupted_generation)
    monkeypatch.setattr(native_run_module, "_replace_owned", fail_only_terminal_status)

    with pytest.raises(KeyboardInterrupt, match="generation interrupt") as caught:
        run_native_one(manifest)

    assert caught.value is original
    assert any("failed to persist" in note for note in getattr(original, "__notes__", ()))
    assert not (root / "complete.json").exists()


def test_run_preflight_does_not_mutate_existing_output_root(tmp_path: Path) -> None:
    root = (tmp_path / "run").absolute()
    root.mkdir()
    sentinel = root / "owned-by-someone-else"
    sentinel.write_text("keep", encoding="utf-8")
    manifest = create_native_run_manifest(
        _config(), NativeRunAuthorities(output_root=root, parent_artifact_path=None)
    )

    with pytest.raises(FileExistsError):
        run_native_one(manifest)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert tuple(root.iterdir()) == (sentinel,)


def test_run_preflight_rejects_symlink_root_and_missing_parent_before_output(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    symlink_root = (tmp_path / "run-link").absolute()
    symlink_root.symlink_to(target, target_is_directory=True)
    symlink_manifest = create_native_run_manifest(
        _config(),
        NativeRunAuthorities(output_root=symlink_root, parent_artifact_path=None),
    )
    with pytest.raises(ValueError, match="symlink"):
        run_native_one(symlink_manifest)
    assert not tuple(target.iterdir())

    output_root = (tmp_path / "parent-run").absolute()
    missing_parent = (tmp_path / "missing-parent").absolute()
    base = _config()
    parent_config = base.model_copy(
        update={
            "teacher_kind": "GEN1_PARENT",
            "source_model_digest": "a" * 64,
            "initialization": NativeTrainerInitialization(
                mode="GEN1_PARENT", parent_model_digest="a" * 64
            ),
        }
    )
    parent_manifest = create_native_run_manifest(
        parent_config,
        NativeRunAuthorities(
            output_root=output_root,
            parent_artifact_path=missing_parent,
        ),
    )
    with pytest.raises(FileNotFoundError, match="parent artifact"):
        run_native_one(parent_manifest)
    assert not output_root.exists()
