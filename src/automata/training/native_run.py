"""Concrete finite orchestration for exactly one native Gen1 generation and optimizer."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import NoReturn

from automata.models.shared_encoder.gen1_model import Gen1ModelConfig
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.training.io import atomic_write_bytes, fsync_directory
from automata.training.native_gen1 import load_current_gen1_parent_artifact
from automata.training.native_generation import (
    NativeGenerationConfig,
    NativeGenerationOutput,
    generate_native_game,
)
from automata.training.native_indexed_dataset import (
    build_native_indexed_dataset,
    create_native_source_receipt_from_completions,
    load_native_source_receipt,
)
from automata.training.native_receipts import (
    create_native_dataset_completion_receipt,
    load_native_dataset_completion_receipt,
)
from automata.training.native_replay import (
    load_native_replay_catalog,
    sample_native_replay,
    update_native_replay_catalog,
)
from automata.training.native_run_contracts import (
    NativeRunCompletion,
    NativeRunFailure,
    NativeRunManifest,
    NativeRunProducts,
    NativeRunProgress,
    NativeRunResult,
    NativeRunTrainingStep,
    NativeTrainingStepRecord,
    load_native_run_manifest,
)
from automata.training.native_trainer import (
    NativeReplayDatasetBinding,
    NativeTrainingStepResult,
    bind_native_replay_sample,
    create_native_trainer,
    open_bound_native_dataset,
)
from automata.training.native_validation import (
    NativeValidationLedger,
    create_native_validation_ledger,
    evaluate_native_validation,
)


def _generation_config(manifest: NativeRunManifest) -> NativeGenerationConfig:
    config = manifest.config
    return NativeGenerationConfig(
        generation_id=config.generation_id,
        source_revision=config.source_revision,
        dirty_tree_hash=config.dirty_tree_hash,
        teacher_kind=config.teacher_kind,
        source_model_digest=config.source_model_digest,
        search_config=config.search.to_search_config(),
        split_config=config.split_config,
        random_stream_namespace=config.random_stream_namespace,
        visit_temperature=config.visit_temperature,
        max_steps=config.max_steps,
        max_rounds=config.max_rounds,
    )


def _require_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks")


def _preflight(manifest: NativeRunManifest):
    root = manifest.authorities.output_root
    _require_no_symlink_components(root, label="native run output root")
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"native run output root already exists: {root}")
    if not root.parent.is_dir():
        raise FileNotFoundError("native run output root parent must be an existing directory")
    parent_path = manifest.authorities.parent_artifact_path
    loaded_parent = None
    if parent_path is not None:
        _require_no_symlink_components(parent_path, label="native run parent artifact")
        if not parent_path.is_dir():
            raise FileNotFoundError("native run parent artifact must be an existing directory")
        assert manifest.config.source_model_digest is not None
        loaded_parent = load_current_gen1_parent_artifact(
            parent_path,
            expected_model_digest=manifest.config.source_model_digest,
        )
        trainer_config = manifest.config.trainer_config
        expected_parent_config = Gen1ModelConfig(
            decision_schema_digest=TensorFeatureSchema.current().digest,
            stable_value_schema_digest=StableValueTensorSchema.current().digest,
            token_width=trainer_config.token_width,
            state_width=trainer_config.state_width,
            candidate_width=trainer_config.candidate_width,
            message_passing_layers=trainer_config.message_passing_layers,
            dropout=trainer_config.dropout,
        )
        if loaded_parent.config != expected_parent_config:
            raise ValueError("Gen1 parent architecture does not match trainer configuration")
    # Reconstructing exercises SearchConfig, generation, seed-purpose, split,
    # replay, trainer, and all derived identity validation before root claim.
    generation = _generation_config(manifest)
    return generation, loaded_parent


def _publish_new(path: Path, payload: bytes) -> os.stat_result:
    """Durably link new bytes without replacing or misidentifying a competitor."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    owned_identity: os.stat_result | None = None
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            # Capture ownership from our still-open temporary file.  Reading
            # destination metadata after link would attribute a racing
            # replacement to this invocation.
            owned_identity = os.fstat(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"native run path appeared during publication: {path}") from exc
        fsync_directory(path.parent)
        current = path.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (
            owned_identity.st_dev,
            owned_identity.st_ino,
        ) or path.read_bytes() != payload:
            raise RuntimeError("native run publication was replaced before durability completed")
        temporary.unlink()
        return owned_identity
    except BaseException as original:
        # This is safe even if os.link linked and then raised before returning:
        # only the inode captured from our temporary file can be removed.
        if owned_identity is not None:
            try:
                _unlink_if_same(path, owned_identity)
            except BaseException as rollback_error:
                original.add_note(
                    "Additionally failed to roll back owned native run publication: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            original.add_note(
                "Additionally failed to clean native run publication temporary: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise


def _unlink_if_same(path: Path, identity: os.stat_result) -> bool:
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
        return False
    path.unlink()
    fsync_directory(path.parent)
    return True


def _replace_owned(
    path: Path,
    payload: bytes,
    *,
    previous: bytes | tuple[bytes, ...],
) -> None:
    expected = (previous,) if isinstance(previous, bytes) else previous
    if not expected or path.is_symlink() or not path.is_file() or path.read_bytes() not in expected:
        raise RuntimeError("native run result ownership changed; refusing to clobber it")
    atomic_write_bytes(path, payload)


def _step_record(result) -> NativeTrainingStepRecord:
    return NativeTrainingStepRecord(
        optimizer_step=result.optimizer_step,
        policy_cross_entropy=result.policy_cross_entropy,
        policy_entropy=result.policy_entropy,
        value_bce=result.value_bce,
        regularization=result.regularization,
        total_loss=result.total_loss,
        gradient_norm_before_clip=result.gradient_norm_before_clip,
        provenance=result.provenance,
    )


def _validation_identity(metrics) -> tuple[object, ...]:
    return (
        metrics.validation_scope,
        metrics.validation_ledger_digest,
        metrics.game_ids,
        metrics.dataset_digests,
        metrics.source_digests,
        metrics.completion_receipt_digests,
        metrics.game_count,
        metrics.policy_contributing_game_count,
        metrics.policy_row_count,
        metrics.value_contributing_game_count,
        metrics.value_row_count,
    )


def _raise_original_with_status_failure(
    original: BaseException,
    traceback,
    status_error: BaseException,
) -> NoReturn:
    original.add_note(
        f"Additionally failed to persist native run FAILED status: "
        f"{type(status_error).__name__}: {status_error}"
    )
    raise original.with_traceback(traceback) from status_error


def run_native_one(manifest: NativeRunManifest) -> NativeRunResult:
    """Execute one finite native run into a new root; failures are persisted and raised."""
    if not isinstance(manifest, NativeRunManifest):
        raise TypeError("manifest must be a NativeRunManifest")
    validated = NativeRunManifest.model_validate(manifest.model_dump(mode="python"), strict=True)
    generation_config, loaded_parent = _preflight(validated)
    root = validated.authorities.output_root
    progress = NativeRunProgress(
        phase="CLAIMED",
        optimizer_steps_completed=0,
    )
    root.mkdir(exist_ok=False)

    result_path = root / "result.json"
    last_result_bytes: bytes | None = None
    attempted_result_bytes: bytes | None = None
    marker_identity: os.stat_result | None = None

    def write_running() -> None:
        nonlocal attempted_result_bytes, last_result_bytes
        running = NativeRunResult(
            status="RUNNING",
            manifest_digest=validated.digest,
            config_digest=validated.config.digest,
            progress=progress,
            products=None,
            failure=None,
        )
        payload = running.canonical_bytes()
        attempted_result_bytes = payload
        if last_result_bytes is None:
            _publish_new(result_path, payload)
        else:
            _replace_owned(result_path, payload, previous=last_result_bytes)
        last_result_bytes = payload
        attempted_result_bytes = None

    try:
        # Root creation is the ownership boundary.  Every fallible durability
        # operation after the successful claim must pass through FAILED status.
        fsync_directory(root.parent)
        for directory in (
            root / "games",
            root / "receipts",
            root / "receipts" / "games",
            root / "replay",
        ):
            directory.mkdir()
        _publish_new(root / "manifest.json", validated.canonical_bytes())
        write_running()

        attempted: list[str] = []
        uncertified: list[str] = []
        completed: list[str] = []
        sidecars: list[Path] = []
        progress = NativeRunProgress(
            phase="GENERATING",
            attempted_game_ids=(),
            uncertified_game_ids=(),
            completed_game_ids=(),
            optimizer_steps_completed=0,
        )
        write_running()
        for planned in validated.planned_games:
            attempted.append(planned.expected_game_id)
            progress = NativeRunProgress(
                phase="GENERATING",
                attempted_game_ids=tuple(attempted),
                uncertified_game_ids=tuple(uncertified),
                completed_game_ids=tuple(completed),
                optimizer_steps_completed=0,
            )
            write_running()
            receipt_path = root / planned.completion_relative_path
            result = generate_native_game(
                planned.game,
                generation_config,
                NativeGenerationOutput(
                    source_root=root,
                    logical_name=planned.source_logical_name,
                    completion_receipt_path=receipt_path,
                ),
                parent_artifact_path=validated.authorities.parent_artifact_path,
            )
            if not result.completed or result.completion_receipt is None:
                uncertified.append(planned.expected_game_id)
                progress = NativeRunProgress(
                    phase="GENERATING",
                    attempted_game_ids=tuple(attempted),
                    uncertified_game_ids=tuple(uncertified),
                    completed_game_ids=tuple(completed),
                    optimizer_steps_completed=0,
                )
                write_running()
                raise RuntimeError(
                    f"planned native game returned without decisive certification: "
                    f"{planned.expected_game_id}"
                )
            source_path = root / planned.source_logical_name
            if (
                result.game.game_id != planned.expected_game_id
                or result.completion_receipt.game != result.game
                or result.completion_receipt.logical_name != planned.source_logical_name
                or result.completion_receipt.reason != "game_over"
                or result.completion_receipt.row_count <= 0
                or not source_path.is_file()
                or not receipt_path.is_file()
            ):
                uncertified.append(planned.expected_game_id)
                progress = NativeRunProgress(
                    phase="GENERATING",
                    attempted_game_ids=tuple(attempted),
                    uncertified_game_ids=tuple(uncertified),
                    completed_game_ids=tuple(completed),
                    optimizer_steps_completed=0,
                )
                write_running()
                raise ValueError("native generation result does not match its planned authority")
            completed.append(planned.expected_game_id)
            sidecars.append(receipt_path)
            progress = NativeRunProgress(
                phase="GENERATING",
                attempted_game_ids=tuple(attempted),
                uncertified_game_ids=tuple(uncertified),
                completed_game_ids=tuple(completed),
                optimizer_steps_completed=0,
            )
            write_running()

        progress = NativeRunProgress(
            phase="INDEXING",
            attempted_game_ids=tuple(attempted),
            completed_game_ids=tuple(completed),
            optimizer_steps_completed=0,
        )
        write_running()
        completion = create_native_dataset_completion_receipt(root, tuple(sidecars))
        completion_path = root / "receipts" / "completion-set.json"
        _publish_new(completion_path, completion.canonical_bytes())
        source_receipt = create_native_source_receipt_from_completions(root, completion)
        source_receipt_path = root / "receipts" / "source-inventory.json"
        _publish_new(source_receipt_path, source_receipt.canonical_bytes())
        dataset = build_native_indexed_dataset(
            root,
            source_receipt_path,
            root / "index",
            chunk_size=validated.config.index_chunk_size,
        )
        binding = NativeReplayDatasetBinding(
            dataset_digest=dataset.digest,
            source_digest=dataset.source_digest,
            completion_receipt_digest=completion.digest,
            source_root=root,
            source_receipt_path=source_receipt_path,
            completion_receipt_path=completion_path,
            index_cache_dir=root / "index",
            chunk_size=validated.config.index_chunk_size,
        )
        reopened = open_bound_native_dataset(binding, rebuild_index=False)
        if reopened.manifest != dataset.manifest:
            raise ValueError("read-only reopened native dataset differs from built index")

        validation_ledger = create_native_validation_ledger(
            validated.planned_split_ledger,
            datasets=(reopened,),
            completion_receipts=(completion,),
        )
        expected_validation_ids = tuple(
            item.expected_game_id for item in validated.planned_games if item.split == "validation"
        )
        if tuple(item.game.game_id for item in validation_ledger.games) != (
            expected_validation_ids
        ):
            raise ValueError("validation ledger differs from the planned validation cohort")
        validation_ledger_path = root / "receipts" / "validation-ledger.json"
        _publish_new(validation_ledger_path, validation_ledger.canonical_bytes())

        progress = NativeRunProgress(
            phase="REPLAY",
            attempted_game_ids=tuple(attempted),
            completed_game_ids=tuple(completed),
            optimizer_steps_completed=0,
        )
        write_running()
        catalog_path = root / "replay" / "catalog.json"
        catalog = update_native_replay_catalog(
            catalog_path,
            config=validated.config.replay_config,
            dataset=reopened,
            completion_receipt=completion,
            parent_artifact=(None if loaded_parent is None else loaded_parent.manifest),
        )
        catalog = load_native_replay_catalog(catalog_path)
        expected_train_ids = tuple(
            item.expected_game_id for item in validated.planned_games if item.split == "train"
        )
        if (
            catalog.split_ledger != validated.planned_split_ledger
            or len(catalog.generations) != 1
            or catalog.generations[0].all_game_ids
            != tuple(item.expected_game_id for item in validated.planned_games)
            or catalog.generations[0].replay_game_ids != expected_train_ids
            or tuple(item.game.game_id for item in catalog.games) != expected_train_ids
            or any(item.game.game_id in expected_validation_ids for item in catalog.games)
        ):
            raise ValueError("run-local replay catalog differs from its complete planned cohort")

        # Materialize and physically bind every fixed sample before any model
        # update.  The retained bound selections are consumed read-only by the
        # trainer, so later training cannot repair a corrupted cache.
        bound_samples = []
        samples = []
        for sampling_seed in validated.config.sampling_seeds:
            sample = sample_native_replay(
                catalog,
                game_count=validated.config.games_per_step,
                sampling_seed=sampling_seed,
            )
            if (
                len(sample.games) != validated.config.games_per_step
                or len(set(sample.game_ids)) != len(sample.games)
                or not set(sample.game_ids) <= set(expected_train_ids)
            ):
                raise ValueError("fixed replay sample is outside the planned TRAIN cohort")
            policy_eligible = (
                validated.config.trainer_config.policy_weight > 0.0
                and sample.policy_contributing_game_count > 0
            )
            value_eligible = (
                validated.config.trainer_config.value_weight > 0.0
                and sample.value_contributing_game_count > 0
            )
            if not (policy_eligible or value_eligible):
                raise ValueError("fixed replay sample has no enabled contributing head")
            samples.append(sample)
            bound_samples.append(bind_native_replay_sample(catalog, sample, bindings=(binding,)))

        trainer = create_native_trainer(
            validated.config.trainer_config,
            initialization=validated.config.initialization,
            parent_artifact_path=validated.authorities.parent_artifact_path,
        )
        progress = NativeRunProgress(
            phase="INITIAL_VALIDATION",
            attempted_game_ids=tuple(attempted),
            completed_game_ids=tuple(completed),
            optimizer_steps_completed=0,
        )
        write_running()
        initial_validation = evaluate_native_validation(
            trainer.model,
            validation_ledger,
            bindings=(binding,),
        )

        progress_steps: list[NativeRunTrainingStep] = []
        native_results: list[NativeTrainingStepResult] = []
        progress = NativeRunProgress(
            phase="TRAINING",
            attempted_game_ids=tuple(attempted),
            completed_game_ids=tuple(completed),
            optimizer_steps_completed=0,
        )
        write_running()
        for sampling_seed, sample, bound in zip(
            validated.config.sampling_seeds, samples, bound_samples, strict=True
        ):
            native_result = trainer.train_logical_batch(bound)
            expected_step = len(native_results) + 1
            if native_result.optimizer_step != expected_step:
                raise ValueError("native trainer optimizer lineage skipped a declared step")
            native_results.append(native_result)
            progress_steps.append(
                NativeRunTrainingStep(
                    sampling_seed=sampling_seed,
                    selected_game_ids=sample.game_ids,
                    result=_step_record(native_result),
                )
            )
            progress = NativeRunProgress(
                phase="TRAINING",
                attempted_game_ids=tuple(attempted),
                completed_game_ids=tuple(completed),
                optimizer_steps_completed=len(progress_steps),
                training_steps=tuple(progress_steps),
            )
            write_running()
        if trainer.optimizer_steps != validated.config.optimizer_steps:
            raise ValueError("native trainer did not execute the exact optimizer budget")

        progress = NativeRunProgress(
            phase="FINAL_VALIDATION",
            attempted_game_ids=tuple(attempted),
            completed_game_ids=tuple(completed),
            optimizer_steps_completed=len(progress_steps),
            training_steps=tuple(progress_steps),
        )
        write_running()
        final_validation = evaluate_native_validation(
            trainer.model,
            validation_ledger,
            bindings=(binding,),
        )
        if _validation_identity(initial_validation) != _validation_identity(final_validation):
            raise ValueError("initial and final validation used different physical authorities")

        progress = NativeRunProgress(
            phase="EXPORTING",
            attempted_game_ids=tuple(attempted),
            completed_game_ids=tuple(completed),
            optimizer_steps_completed=len(progress_steps),
            training_steps=tuple(progress_steps),
        )
        write_running()
        artifact_manifest = trainer.export_artifact(
            root / "artifact",
            step=native_results[-1],
            source_revision=validated.config.source_revision,
            dirty_tree_hash=validated.config.dirty_tree_hash,
        )
        loaded_artifact = load_current_gen1_parent_artifact(
            root / "artifact",
            expected_model_digest=artifact_manifest.model_digest,
        )
        artifact_manifest_bytes = (root / "artifact" / "manifest.json").read_bytes()
        artifact_manifest_digest = hashlib.sha256(artifact_manifest_bytes).hexdigest()

        progress = NativeRunProgress(
            phase="VERIFYING",
            attempted_game_ids=tuple(attempted),
            completed_game_ids=tuple(completed),
            optimizer_steps_completed=len(progress_steps),
            training_steps=tuple(progress_steps),
        )
        write_running()
        if load_native_run_manifest(root / "manifest.json") != validated:
            raise ValueError("published native run manifest changed")
        loaded_completion = load_native_dataset_completion_receipt(completion_path)
        loaded_source = load_native_source_receipt(source_receipt_path)
        loaded_dataset = open_bound_native_dataset(binding, rebuild_index=False)
        loaded_catalog = load_native_replay_catalog(catalog_path)
        ledger_payload = validation_ledger_path.read_bytes()
        loaded_ledger = NativeValidationLedger.model_validate_json(ledger_payload, strict=True)
        if ledger_payload != loaded_ledger.canonical_bytes():
            raise ValueError("native validation ledger is not canonical JSON")
        if (
            loaded_completion != completion
            or loaded_source != source_receipt
            or loaded_dataset.manifest != dataset.manifest
            or loaded_catalog != catalog
            or loaded_ledger != validation_ledger
            or loaded_artifact.manifest != artifact_manifest
            or (root / "artifact" / "manifest.json").read_bytes() != artifact_manifest_bytes
        ):
            raise ValueError("final native run physical revalidation changed a product")
        exported_validation = evaluate_native_validation(
            loaded_artifact.model,
            loaded_ledger,
            bindings=(binding,),
        )
        if exported_validation != final_validation:
            raise ValueError("exported artifact does not reproduce final validation metrics")

        products = NativeRunProducts(
            parent_training_exposure=(
                "NONE_FRESH_INITIALIZATION"
                if validated.config.initialization.mode == "FRESH_BOOTSTRAP"
                else "UNKNOWN"
            ),
            generated_game_ids=tuple(completed),
            completion_receipt_digest=loaded_completion.digest,
            source_receipt_digest=loaded_source.digest,
            dataset_digest=loaded_dataset.digest,
            catalog_digest=loaded_catalog.digest,
            validation_ledger_digest=loaded_ledger.digest,
            initial_validation=initial_validation,
            training_steps=tuple(progress_steps),
            final_validation=final_validation,
            artifact_model_digest=loaded_artifact.manifest.model_digest,
            artifact_manifest_digest=artifact_manifest_digest,
        )
        progress = NativeRunProgress(
            phase="COMPLETE",
            attempted_game_ids=tuple(attempted),
            completed_game_ids=tuple(completed),
            optimizer_steps_completed=len(progress_steps),
            training_steps=tuple(progress_steps),
        )
        succeeded = NativeRunResult(
            status="SUCCEEDED",
            manifest_digest=validated.digest,
            config_digest=validated.config.digest,
            progress=progress,
            products=products,
            failure=None,
        )
        succeeded_bytes = succeeded.canonical_bytes()
        assert last_result_bytes is not None
        attempted_result_bytes = succeeded_bytes
        _replace_owned(result_path, succeeded_bytes, previous=last_result_bytes)
        last_result_bytes = succeeded_bytes
        attempted_result_bytes = None
        completion_marker = NativeRunCompletion(
            manifest_digest=validated.digest,
            result_digest=succeeded.digest,
        )
        marker_identity = _publish_new(root / "complete.json", completion_marker.canonical_bytes())
        return succeeded
    except BaseException as original:
        _, _, traceback = sys.exc_info()
        if marker_identity is not None:
            with suppress(BaseException):
                _unlink_if_same(root / "complete.json", marker_identity)
        try:
            failure = NativeRunResult(
                status="FAILED",
                manifest_digest=validated.digest,
                config_digest=validated.config.digest,
                progress=progress,
                products=None,
                failure=NativeRunFailure(
                    phase=progress.phase,
                    error_type=type(original).__name__,
                    message=str(original)[:1000].strip() or type(original).__name__,
                ),
            )
            payload = failure.canonical_bytes()
            known_result_bytes = tuple(
                dict.fromkeys(
                    candidate
                    for candidate in (last_result_bytes, attempted_result_bytes)
                    if candidate is not None
                )
            )
            attempted_result_bytes = payload
            if last_result_bytes is None and not result_path.exists():
                _publish_new(result_path, payload)
            else:
                _replace_owned(result_path, payload, previous=known_result_bytes)
            last_result_bytes = payload
            attempted_result_bytes = None
        except BaseException as status_error:
            _raise_original_with_status_failure(
                original,
                traceback,
                status_error,
            )
        raise


__all__ = ["run_native_one"]
