"""Completion-bound held-out validation contracts for native Gen1 data."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

import automata.training.native_validation as native_validation_module
from automata.decision import DecisionDescriptor
from automata.harness.game_runner import DEFAULT_MAP
from automata.models.shared_encoder.gen1_model import PolicyHeadOutput
from automata.observation import encode_decision, legal_keys_for_decision
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import detect_stable_value_boundary
from automata.training.native_dataset import (
    NativeBoundaryProvenance,
    NativeGameIdentity,
    PolicyDatasetRecord,
    ValueDatasetRecord,
    native_game_id,
    native_sample_id,
    publish_native_game,
)
from automata.training.native_indexed_dataset import (
    IndexedNativeDataset,
    build_native_indexed_dataset,
    create_native_source_receipt_from_completions,
)
from automata.training.native_receipts import (
    NativeDatasetCompletionReceipt,
    NativeGameCompletionReceipt,
)
from automata.training.native_splits import (
    NativeSeedRange,
    NativeSeedSplitLedger,
    NativeSplitConfig,
    create_native_split_ledger,
    extend_native_split_ledger,
)
from automata.training.native_trainer import (
    NativeReplayDatasetBinding,
    NativeTrainerConfig,
    NativeTrainerInitialization,
    create_native_trainer,
    open_bound_native_dataset,
)
from automata.training.native_validation import (
    NativeValidationDatasetRef,
    NativeValidationLedger,
    NativeValidationMetrics,
    create_native_validation_ledger,
    evaluate_native_validation,
)
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget
from goa2.domain.models import TeamColor
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def _split_config() -> NativeSplitConfig:
    return NativeSplitConfig(
        namespace="native-validation-tests",
        salt="fixed-test-salt",
        validation_fraction=0.01,
        seed_ranges=(
            NativeSeedRange(purpose="training", start=0, stop=100),
            NativeSeedRange(purpose="validation", start=100, stop=200),
        ),
    )


def _ledger(*seeds: int) -> NativeSeedSplitLedger:
    return extend_native_split_ledger(create_native_split_ledger(_split_config()), seeds)


def _identity(
    seed: int,
    *,
    generation: str = "validation-generation",
    composition: tuple[str, ...] = ("Wasp",),
) -> NativeGameIdentity:
    values: dict[str, Any] = {
        "world_seed": seed,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": composition,
        "blue_composition": ("Arien",),
        "generation_id": generation,
        "source_revision": "validation-test",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "fixture-search",
        "generator_config_id": "fixture-generator",
    }
    return NativeGameIdentity(game_id=native_game_id(**values), **values)


def _records(
    identity: NativeGameIdentity,
    *,
    policy_count: int,
    value_count: int,
) -> tuple[PolicyDatasetRecord | ValueDatasetRecord, ...]:
    register_all_effects()
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(identity.red_composition),
        list(identity.blue_composition),
        game_type=identity.game_type,
        seed=identity.world_seed,
    )
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    descriptor = DecisionDescriptor("CARD", hero=hero)
    observation = encode_decision(
        state,
        descriptor,
        legal_keys_for_decision(descriptor),
        decision_owner_hero_id=hero.id,
        perspective_team="RED",
    )
    probabilities = tuple(range(1, len(observation.candidates) + 1))
    total = sum(probabilities)
    target = SearchPolicyTarget(
        actions=tuple(
            SearchActionTarget(
                candidate=candidate,
                prior_probability=1.0 / len(observation.candidates),
                sample_count=weight,
                mean_value=0.1 * index,
                value_variance=0.0,
                improved_probability=weight / total,
                selected=index == len(observation.candidates) - 1,
            )
            for index, (candidate, weight) in enumerate(
                zip(observation.candidates, probabilities, strict=True)
            )
        )
    )
    records: list[PolicyDatasetRecord | ValueDatasetRecord] = []
    for index in range(policy_count):
        records.append(
            PolicyDatasetRecord(
                game=identity,
                sample_id=native_sample_id(
                    game_id=identity.game_id,
                    sample_kind="POLICY",
                    sample_index=index,
                ),
                sample_index=index,
                policy_index=index,
                perspective_team="RED",
                observation=observation,
                target=target,
            )
        )

    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    value_observation = encode_stable_value(
        state,
        boundary,
        viewer_hero_id=hero.id,
        perspective_team=TeamColor.RED,
    )
    viewer_ref = next(
        token.local_ref
        for token in value_observation.state.tokens
        if token.kind == "HERO" and token.features["relation"] == "SELF"
    )
    for offset in range(value_count):
        sample_index = policy_count + offset
        records.append(
            ValueDatasetRecord(
                game=identity,
                sample_id=native_sample_id(
                    game_id=identity.game_id,
                    sample_kind="VALUE",
                    sample_index=sample_index,
                ),
                sample_index=sample_index,
                perspective_team="RED",
                boundary=NativeBoundaryProvenance(
                    boundary_index=offset,
                    kind="PLANNING_READY",
                    round=state.round,
                    turn=state.turn,
                    viewer_ref=viewer_ref,
                ),
                observation=value_observation,
                terminal_winner="RED",
                value_target=1,
            )
        )
    return tuple(records)


def _dataset(
    root: Path,
    identities: tuple[NativeGameIdentity, ...],
    counts: tuple[tuple[int, int], ...],
    *,
    chunk_size: int,
) -> tuple[IndexedNativeDataset, NativeDatasetCompletionReceipt, NativeReplayDatasetBinding]:
    source_root = root / "source"
    source_root.mkdir(parents=True)
    completions: list[NativeGameCompletionReceipt] = []
    for ordinal, (identity, (policy_count, value_count)) in enumerate(
        zip(identities, counts, strict=True)
    ):
        logical_name = f"games/{ordinal:08d}.jsonl.zst"
        path = source_root / logical_name
        publish_native_game(
            path,
            _records(identity, policy_count=policy_count, value_count=value_count),
        )
        payload = path.read_bytes()
        completions.append(
            NativeGameCompletionReceipt(
                logical_name=logical_name,
                game=identity,
                file_sha256=hashlib.sha256(payload).hexdigest(),
                file_size=len(payload),
                row_count=policy_count + value_count,
                policy_row_count=policy_count,
                value_row_count=value_count,
                boundary_count=value_count,
                reason="game_over",
                terminal_winner="RED",
            )
        )
    completion = NativeDatasetCompletionReceipt(games=tuple(completions))
    completion_path = root / "completion.json"
    completion_path.write_bytes(completion.canonical_bytes())
    source_receipt = create_native_source_receipt_from_completions(source_root, completion)
    source_receipt_path = root / "source-receipt.json"
    source_receipt_path.write_bytes(source_receipt.canonical_bytes())
    cache = root / "index"
    dataset = build_native_indexed_dataset(
        source_root,
        source_receipt_path,
        cache,
        chunk_size=chunk_size,
    )
    binding = NativeReplayDatasetBinding(
        dataset_digest=dataset.digest,
        source_digest=dataset.source_digest,
        completion_receipt_digest=completion.digest,
        source_root=source_root,
        source_receipt_path=source_receipt_path,
        completion_receipt_path=completion_path,
        index_cache_dir=cache,
        chunk_size=chunk_size,
    )
    return dataset, completion, binding


def _trainer_config() -> NativeTrainerConfig:
    return NativeTrainerConfig(
        seed=17,
        learning_rate=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        policy_weight=1.0,
        value_weight=1.0,
        entropy_weight=0.0,
        l2_weight=0.0,
        max_gradient_norm=10.0,
        token_width=8,
        state_width=12,
        candidate_width=8,
        message_passing_layers=1,
        dropout=0.0,
    )


def _model():
    return create_native_trainer(
        _trainer_config(),
        initialization=NativeTrainerInitialization(mode="FRESH_BOOTSTRAP"),
    ).model


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        item.relative_to(root).as_posix(): item.read_bytes()
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def test_ledger_is_canonical_all_and_only_validation_in_physical_order(
    tmp_path: Path,
) -> None:
    train = _identity(1)
    first = _identity(100)
    repeated = _identity(100, generation="validation-generation-repeated-seed")
    dataset, completion, _ = _dataset(
        tmp_path,
        (train, first, repeated),
        ((1, 1), (2, 3), (1, 2)),
        chunk_size=2,
    )

    ledger = create_native_validation_ledger(
        _ledger(1, 100),
        datasets=(dataset,),
        completion_receipts=(completion,),
    )

    assert tuple(ref.game for ref in ledger.games) == (first, repeated)
    assert tuple(ref.source_logical_name for ref in ledger.games) == (
        "games/00000001.jsonl.zst",
        "games/00000002.jsonl.zst",
    )
    assert ledger.digest == hashlib.sha256(ledger.canonical_bytes()).hexdigest()
    assert (
        NativeValidationLedger.model_validate_json(ledger.canonical_bytes(), strict=True) == ledger
    )
    assert ledger.games[0].dataset_digest == dataset.digest
    assert ledger.games[0].completion_receipt_digest == completion.digest
    assert ledger.games[0].source_digest == dataset.source_digest


def test_ledger_rejects_omission_train_inclusion_unknown_and_duplicate_game_ids(
    tmp_path: Path,
) -> None:
    train = _identity(1)
    validation = _identity(100)
    dataset, completion, _ = _dataset(
        tmp_path,
        (train, validation),
        ((1, 1), (1, 1)),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(1, 100), datasets=(dataset,), completion_receipts=(completion,)
    )

    with pytest.raises(ValidationError, match=r"validation|nonempty"):
        NativeValidationLedger(
            split_ledger=ledger.split_ledger,
            datasets=ledger.datasets,
            games=(),
        )
    with pytest.raises(ValidationError, match=r"validation|split"):
        NativeValidationLedger(
            split_ledger=ledger.split_ledger,
            datasets=ledger.datasets,
            games=(ledger.games[0].model_copy(update={"game": train}),),
        )
    with pytest.raises(ValidationError, match=r"unique|duplicate"):
        NativeValidationLedger(
            split_ledger=ledger.split_ledger,
            datasets=ledger.datasets,
            games=(ledger.games[0], ledger.games[0]),
        )
    with pytest.raises(ValueError, match=r"not enrolled|world seed"):
        create_native_validation_ledger(
            _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
        )
    train_only_dataset, train_only_completion, _ = _dataset(
        tmp_path / "all-train",
        (_identity(1, generation="all-train"),),
        ((1, 1),),
        chunk_size=1,
    )
    with pytest.raises(ValueError, match=r"validation-assigned|at least one validation"):
        create_native_validation_ledger(
            _ledger(1),
            datasets=(train_only_dataset,),
            completion_receipts=(train_only_completion,),
        )

    with_historical_seed = create_native_validation_ledger(
        _ledger(1, 100, 101),
        datasets=(dataset,),
        completion_receipts=(completion,),
    )
    assert with_historical_seed.games == ledger.games


def test_two_dataset_ledger_binds_train_only_and_validation_authorities(
    tmp_path: Path,
) -> None:
    train = _identity(1, generation="train-only-dataset")
    validation = _identity(100, generation="validation-only-dataset")
    train_dataset, train_completion, train_binding = _dataset(
        tmp_path / "train",
        (train,),
        ((1, 1),),
        chunk_size=1,
    )
    validation_dataset, validation_completion, validation_binding = _dataset(
        tmp_path / "validation",
        (validation,),
        ((1, 1),),
        chunk_size=1,
    )

    ledger = create_native_validation_ledger(
        _ledger(1, 2, 100),
        datasets=(train_dataset, validation_dataset),
        completion_receipts=(train_completion, validation_completion),
    )

    assert ledger.datasets == (
        NativeValidationDatasetRef(
            dataset_digest=train_dataset.digest,
            source_digest=train_dataset.source_digest,
            completion_receipt_digest=train_completion.digest,
            validation_game_ids=(),
        ),
        NativeValidationDatasetRef(
            dataset_digest=validation_dataset.digest,
            source_digest=validation_dataset.source_digest,
            completion_receipt_digest=validation_completion.digest,
            validation_game_ids=(validation.game_id,),
        ),
    )
    metrics = evaluate_native_validation(
        _model(),
        ledger,
        bindings=(train_binding, validation_binding),
    )
    assert metrics.game_ids == (validation.game_id,)
    assert metrics.dataset_digests == (train_dataset.digest, validation_dataset.digest)
    assert metrics.source_digests == (
        train_dataset.source_digest,
        validation_dataset.source_digest,
    )
    assert metrics.completion_receipt_digests == (
        train_completion.digest,
        validation_completion.digest,
    )
    with pytest.raises(ValueError, match=r"exactly|bindings"):
        evaluate_native_validation(_model(), ledger, bindings=(validation_binding,))


def test_repeated_validation_seed_across_datasets_cannot_hide_an_omission(
    tmp_path: Path,
) -> None:
    first = _identity(100, generation="first-validation-dataset")
    second = _identity(100, generation="second-validation-dataset")
    first_dataset, first_completion, first_binding = _dataset(
        tmp_path / "first",
        (first,),
        ((1, 1),),
        chunk_size=1,
    )
    second_dataset, second_completion, second_binding = _dataset(
        tmp_path / "second",
        (second,),
        ((1, 1),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100),
        datasets=(first_dataset, second_dataset),
        completion_receipts=(first_completion, second_completion),
    )

    metrics = evaluate_native_validation(_model(), ledger, bindings=(first_binding, second_binding))
    assert metrics.game_ids == (first.game_id, second.game_id)

    missing_game_ref = ledger.model_copy(update={"games": ledger.games[:1]})
    with pytest.raises(ValidationError, match=r"inventory|validation game|physical order"):
        evaluate_native_validation(
            _model(),
            missing_game_ref,
            bindings=(first_binding, second_binding),
        )
    with pytest.raises(ValueError, match=r"exactly|bindings"):
        evaluate_native_validation(_model(), ledger, bindings=(first_binding,))


def test_evaluation_exact_metrics_unequal_counts_and_chunk_parity(tmp_path: Path) -> None:
    identities = (_identity(100), _identity(101))
    small, small_completion, small_binding = _dataset(
        tmp_path / "small",
        identities,
        ((1, 3), (4, 1)),
        chunk_size=1,
    )
    large_binding = small_binding.model_copy(
        update={"index_cache_dir": tmp_path / "large-index", "chunk_size": 8}
    )
    open_bound_native_dataset(large_binding, rebuild_index=True)
    ledger = create_native_validation_ledger(
        _ledger(100, 101), datasets=(small,), completion_receipts=(small_completion,)
    )
    model = _model()

    first = evaluate_native_validation(model, ledger, bindings=(small_binding,))
    second = evaluate_native_validation(model, ledger, bindings=(large_binding,))

    assert first.validation_scope == "CURRENT_RUN_UPDATES"
    assert first == second.model_copy(
        update={
            "policy_cross_entropy": first.policy_cross_entropy,
            "policy_entropy": first.policy_entropy,
            "value_bce": first.value_bce,
        }
    )
    assert first.policy_cross_entropy == pytest.approx(second.policy_cross_entropy, rel=2e-6)
    assert first.policy_entropy == pytest.approx(second.policy_entropy, rel=2e-6)
    assert first.value_bce == pytest.approx(second.value_bce, rel=2e-6)
    assert first.dataset_digests == (small.digest,)
    assert first.source_digests == (small.source_digest,)
    assert first.completion_receipt_digests == (small_completion.digest,)
    assert first.policy_contributing_game_count == 2
    assert first.policy_row_count == 5
    assert first.value_contributing_game_count == 2
    assert first.value_row_count == 4

    policy_ce = policy_entropy = value_bce = 0.0
    with torch.inference_mode():
        for ref in ledger.games:
            for batch in small.iter_policy_training_chunks(ref.game.game_id):
                logits = model.forward_policy(batch.batch).policy_logits
                active = batch.row_mask
                legal = batch.batch.candidates.mask[active]
                targets = batch.policy_targets[active]
                weights = batch.row_weights[active]
                log_probabilities = torch.log_softmax(
                    logits[active].masked_fill(~legal, -torch.inf), dim=-1
                )
                finite_log_probabilities = log_probabilities.masked_fill(~legal, 0.0)
                cross_entropy_rows = -(targets * finite_log_probabilities).sum(dim=-1)
                probabilities = log_probabilities.exp()
                entropy_rows = -(probabilities * finite_log_probabilities).sum(dim=-1)
                policy_ce += float((weights * cross_entropy_rows).sum() / 2.0)
                policy_entropy += float((weights * entropy_rows).sum() / 2.0)
            for batch in small.iter_value_training_chunks(ref.game.game_id):
                values = model.forward_stable_value(batch.batch).value[batch.row_mask]
                targets = batch.value_targets[batch.row_mask]
                weights = batch.row_weights[batch.row_mask]
                probabilities = ((values + 1.0) / 2.0).clamp(1e-7, 1.0 - 1e-7)
                binary_targets = (targets + 1.0) / 2.0
                bce_rows = -(
                    binary_targets * probabilities.log()
                    + (1.0 - binary_targets) * (1.0 - probabilities).log()
                )
                value_bce += float((weights * bce_rows).sum() / 2.0)
    assert first.policy_cross_entropy == pytest.approx(policy_ce)
    assert first.policy_entropy == pytest.approx(policy_entropy)
    assert first.value_bce == pytest.approx(value_bce)


def test_absent_heads_report_none_with_independent_counts(tmp_path: Path) -> None:
    policy_dataset, policy_completion, policy_binding = _dataset(
        tmp_path / "policy",
        (_identity(100),),
        ((2, 0),),
        chunk_size=1,
    )
    policy_ledger = create_native_validation_ledger(
        _ledger(100),
        datasets=(policy_dataset,),
        completion_receipts=(policy_completion,),
    )
    policy = evaluate_native_validation(_model(), policy_ledger, bindings=(policy_binding,))
    assert policy.policy_cross_entropy is not None
    assert policy.policy_entropy is not None
    assert policy.value_contributing_game_count == policy.value_row_count == 0
    assert policy.value_bce is None

    value_dataset, value_completion, value_binding = _dataset(
        tmp_path / "value",
        (_identity(101),),
        ((0, 2),),
        chunk_size=1,
    )
    value_ledger = create_native_validation_ledger(
        _ledger(101),
        datasets=(value_dataset,),
        completion_receipts=(value_completion,),
    )
    value = evaluate_native_validation(_model(), value_ledger, bindings=(value_binding,))
    assert value.policy_contributing_game_count == value.policy_row_count == 0
    assert value.policy_cross_entropy is None
    assert value.policy_entropy is None
    assert value.value_bce is not None

    inconsistent_absent_heads = (
        {**policy.model_dump(mode="python"), "value_row_count": 1},
        {**policy.model_dump(mode="python"), "value_contributing_game_count": 1},
        {**value.model_dump(mode="python"), "policy_row_count": 1},
        {**value.model_dump(mode="python"), "policy_contributing_game_count": 1},
    )
    for inconsistent in inconsistent_absent_heads:
        with pytest.raises(ValidationError, match=r"absent|contributing|count|rows"):
            NativeValidationMetrics.model_validate(inconsistent, strict=True)


@pytest.mark.parametrize(
    "tamper",
    (
        "completion-receipt",
        "source-inventory",
        "source-bytes",
        "binding-repointing",
        "schema-change",
    ),
)
def test_evaluation_authority_tampering_fails_closed_without_repair(
    tmp_path: Path,
    tamper: str,
) -> None:
    root = tmp_path / tamper
    dataset, completion, binding = _dataset(
        root,
        (_identity(100),),
        ((1, 1),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
    )
    evaluated_binding = binding
    if tamper == "completion-receipt":
        binding.completion_receipt_path.write_bytes(
            binding.completion_receipt_path.read_bytes() + b"\n"
        )
    elif tamper == "source-inventory":
        source_receipt = dataset.manifest.source_receipt
        changed_game = source_receipt.games[0].model_copy(
            update={"logical_name": "games/repointed.jsonl"}
        )
        changed_receipt = source_receipt.model_copy(update={"games": (changed_game,)})
        binding.source_receipt_path.write_bytes(changed_receipt.canonical_bytes())
    elif tamper == "source-bytes":
        source = binding.source_root / completion.games[0].logical_name
        source.write_bytes(source.read_bytes() + b"tampered")
    elif tamper == "binding-repointing":
        evaluated_binding = binding.model_copy(update={"index_cache_dir": root / "repointed-index"})
    elif tamper == "schema-change":
        changed_manifest = dataset.manifest.model_copy(
            update={"decision_tensor_schema_digest": "0" * 64}
        )
        (binding.index_cache_dir / "manifest.json").write_bytes(changed_manifest.canonical_bytes())
    else:  # pragma: no cover - exhaustive guard for future parametrization edits
        raise AssertionError(tamper)
    before = _tree_bytes(root)

    with pytest.raises((OSError, ValueError)):
        evaluate_native_validation(_model(), ledger, bindings=(evaluated_binding,))

    assert _tree_bytes(root) == before
    if tamper == "binding-repointing":
        assert not evaluated_binding.index_cache_dir.exists()


def test_evaluation_reopens_exact_authorities_readonly_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((2, 2),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
    )
    before = _tree_bytes(tmp_path)
    assert evaluate_native_validation(_model(), ledger, bindings=(binding,)).game_count == 1
    assert _tree_bytes(tmp_path) == before

    chunk = dataset.game(dataset.game_ids[0]).policy_chunks[0]
    (binding.index_cache_dir / chunk.path).write_bytes(b"physically-tampered")
    tampered = _tree_bytes(tmp_path)
    with pytest.raises(ValueError, match=r"cache|chunk|compatible|rebuild"):
        evaluate_native_validation(_model(), ledger, bindings=(binding,))
    assert _tree_bytes(tmp_path) == tampered


def test_ledger_creation_reads_exact_source_authorities(tmp_path: Path) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((1, 1),),
        chunk_size=1,
    )
    source = binding.source_root / completion.games[0].logical_name
    source.write_bytes(source.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match=r"source|hash|size|receipt"):
        create_native_validation_ledger(
            _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
        )


def test_public_bound_opener_revalidates_unsafe_binding_and_is_readonly_by_default(
    tmp_path: Path,
) -> None:
    dataset, _, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((1, 1),),
        chunk_size=1,
    )
    assert open_bound_native_dataset(binding).digest == dataset.digest
    broken = binding.model_copy(update={"dataset_digest": "not-a-digest"})
    with pytest.raises(ValidationError, match="dataset_digest"):
        open_bound_native_dataset(broken)
    with pytest.raises(TypeError, match="rebuild_index"):
        open_bound_native_dataset(binding, rebuild_index=1)  # type: ignore[arg-type]
    chunk = dataset.game(dataset.game_ids[0]).policy_chunks[0]
    (binding.index_cache_dir / chunk.path).write_bytes(b"corrupt")
    before = _tree_bytes(binding.index_cache_dir)
    with pytest.raises(ValueError, match=r"cache|compatible|rebuild"):
        open_bound_native_dataset(binding)
    assert _tree_bytes(binding.index_cache_dir) == before


def test_public_bound_opener_rejects_symlinked_authority_ancestors(tmp_path: Path) -> None:
    authority_root = tmp_path / "physical-authority"
    _, _, binding = _dataset(
        authority_root,
        (_identity(100),),
        ((1, 1),),
        chunk_size=1,
    )
    alias = tmp_path / "authority-alias"
    alias.symlink_to(authority_root, target_is_directory=True)
    aliased = binding.model_copy(
        update={
            "source_root": alias / binding.source_root.relative_to(authority_root),
            "source_receipt_path": alias / binding.source_receipt_path.relative_to(authority_root),
            "completion_receipt_path": alias
            / binding.completion_receipt_path.relative_to(authority_root),
            "index_cache_dir": alias / binding.index_cache_dir.relative_to(authority_root),
        }
    )

    with pytest.raises(ValueError, match=r"symlink|authority"):
        open_bound_native_dataset(aliased)


def test_evaluation_rejects_unsafe_ledger_model_and_binding_sets(tmp_path: Path) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100), _identity(101)),
        ((1, 1), (1, 1)),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100, 101), datasets=(dataset,), completion_receipts=(completion,)
    )
    broken_ref = ledger.games[0].model_copy(update={"policy_row_count": -1})
    broken = ledger.model_copy(update={"games": (broken_ref,)})
    with pytest.raises(ValidationError, match="policy_row_count"):
        evaluate_native_validation(_model(), broken, bindings=(binding,))
    omitted = ledger.model_copy(update={"games": ledger.games[:1]})
    with pytest.raises(ValueError, match=r"every and only|physical order"):
        evaluate_native_validation(_model(), omitted, bindings=(binding,))
    with pytest.raises(ValueError, match=r"exactly|bindings"):
        evaluate_native_validation(_model(), ledger, bindings=())
    with pytest.raises(ValueError, match=r"unambiguous|duplicate"):
        evaluate_native_validation(_model(), ledger, bindings=(binding, binding))


def test_evaluation_preserves_model_gradients_flags_requires_grad_and_cpu_rng_on_success(
    tmp_path: Path,
) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((2, 2),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
    )
    model = _model()
    modules = tuple(model.modules())
    for index, module in enumerate(modules):
        module.training = index % 2 == 0
    parameters = tuple(model.parameters())
    parameters[0].requires_grad_(False)
    for index, parameter in enumerate(parameters):
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, index + 0.25)
    state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    gradients_before = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )
    requires_before = tuple(parameter.requires_grad for parameter in parameters)
    flags_before = tuple(module.training for module in modules)
    torch.manual_seed(909)
    rng_before = torch.random.get_rng_state().clone()

    metrics = evaluate_native_validation(model, ledger, bindings=(binding,))

    assert metrics.game_count == 1
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert tuple(module.training for module in modules) == flags_before
    assert tuple(parameter.requires_grad for parameter in parameters) == requires_before
    assert model.state_dict().keys() == state_before.keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, state_before[name])
    for parameter, expected in zip(parameters, gradients_before, strict=True):
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.equal(parameter.grad, expected)


def test_evaluation_restores_state_and_rng_when_forward_fails_after_one_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((2, 1),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
    )
    model = _model()
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, 3.0)
    state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    grad_before = parameter.grad.clone()
    flags_before = tuple(module.training for module in model.modules())
    requires_before = tuple(item.requires_grad for item in model.parameters())
    rng_before = torch.random.get_rng_state().clone()
    original = model.forward_policy
    calls = 0

    def fail_after_one(*args: Any, **kwargs: Any) -> PolicyHeadOutput:
        nonlocal calls
        calls += 1
        if calls == 2:
            torch.rand(1)
            with torch.inference_mode():
                parameter.add_(7.0)
            parameter.grad = None
            next(model.modules()).training = not flags_before[0]
            raise RuntimeError("injected validation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward_policy", fail_after_one)
    with pytest.raises(RuntimeError, match="injected validation failure"):
        evaluate_native_validation(model, ledger, bindings=(binding,))

    assert calls == 2
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert tuple(module.training for module in model.modules()) == flags_before
    assert tuple(item.requires_grad for item in model.parameters()) == requires_before
    assert parameter.grad is not None and torch.equal(parameter.grad, grad_before)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, state_before[name])


def test_evaluation_preserves_original_failure_with_restore_failure_as_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((1, 0),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
    )
    model = _model()

    def fail_forward(*args: Any, **kwargs: Any) -> PolicyHeadOutput:
        raise RuntimeError("original inference failure")

    def fail_restore(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("snapshot restore failure")

    monkeypatch.setattr(model, "forward_policy", fail_forward)
    monkeypatch.setattr(
        native_validation_module._ModelSnapshot,
        "restore_and_report_mutation",
        fail_restore,
    )

    with pytest.raises(RuntimeError, match="original inference failure") as caught:
        evaluate_native_validation(model, ledger, bindings=(binding,))

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "snapshot restore failure"


@pytest.mark.parametrize("dropout", [0, 0.0])
def test_evaluation_shape_validation_uses_float32_meta_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dropout: float,
) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((1, 1),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
    )
    model = _model()
    # Gen1ModelConfig accepts both numeric representations of disabled dropout;
    # compatible parent artifacts may therefore carry integer zero.
    model.config = model.config.__class__(**{**asdict(model.config), "dropout": dropout})
    reset_devices: list[str] = []
    original_reset = torch.nn.Linear.reset_parameters

    def record_reset_device(linear: torch.nn.Linear) -> None:
        reset_devices.append(linear.weight.device.type)
        original_reset(linear)

    monkeypatch.setattr(torch.nn.Linear, "reset_parameters", record_reset_device)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        metrics = evaluate_native_validation(model, ledger, bindings=(binding,))
    finally:
        torch.set_default_dtype(original_dtype)

    assert metrics.game_count == 1
    # Also permits a future signature-only implementation with no construction.
    assert all(device == "meta" for device in reset_devices)


def test_evaluation_rejects_live_nonzero_dropout_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((1, 1),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
    )
    model = _model()
    live_dropout = next(
        module for module in model.modules() if isinstance(module, torch.nn.Dropout)
    )
    live_dropout.p = 0.25
    forward_called = False

    def forbidden_forward(*args: Any, **kwargs: Any) -> PolicyHeadOutput:
        nonlocal forward_called
        forward_called = True
        raise AssertionError("inference must not run with live dropout")

    monkeypatch.setattr(model, "forward_policy", forbidden_forward)

    with pytest.raises(ValueError, match=r"live.*Dropout.*p.*0"):
        evaluate_native_validation(model, ledger, bindings=(binding,))

    assert forward_called is False


def test_evaluation_rejects_nonzero_dropout_nonfinite_and_unsafe_model_config(
    tmp_path: Path,
) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((1, 1),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
    )
    model = _model()
    model.config = model.config.__class__(**{**asdict(model.config), "dropout": 0.1})
    with pytest.raises(ValueError, match="dropout"):
        evaluate_native_validation(model, ledger, bindings=(binding,))

    model = _model()
    model.config = model.config.__class__(
        **{
            **asdict(model.config),
            "message_passing_layers": model.config.message_passing_layers + 1,
        }
    )
    with pytest.raises(ValueError, match=r"structure|configuration"):
        evaluate_native_validation(model, ledger, bindings=(binding,))

    model = _model()
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_native_validation(model, ledger, bindings=(binding,))


def test_metrics_contract_is_strict_finite_and_scope_labeled(tmp_path: Path) -> None:
    dataset, completion, binding = _dataset(
        tmp_path,
        (_identity(100),),
        ((1, 1),),
        chunk_size=1,
    )
    ledger = create_native_validation_ledger(
        _ledger(100), datasets=(dataset,), completion_receipts=(completion,)
    )
    metrics = evaluate_native_validation(_model(), ledger, bindings=(binding,))
    assert (
        NativeValidationMetrics.model_validate_json(metrics.canonical_bytes(), strict=True)
        == metrics
    )
    assert metrics.digest == hashlib.sha256(metrics.canonical_bytes()).hexdigest()
    with pytest.raises(ValidationError, match=r"finite|policy_cross_entropy"):
        NativeValidationMetrics.model_validate(
            {**metrics.model_dump(mode="python"), "policy_cross_entropy": float("nan")},
            strict=True,
        )
