"""Physical replay binding and one-step native Gen1 trainer contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from automata.decision import DecisionDescriptor
from automata.harness.game_runner import DEFAULT_MAP
from automata.models.shared_encoder.artifacts import load_gen1_model_artifact
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
from automata.training.native_gen1 import (
    current_gen1_artifact_scope,
    current_gen1_runtime_requirements,
)
from automata.training.native_indexed_dataset import (
    create_native_source_receipt_from_completions,
)
from automata.training.native_losses import native_policy_loss, native_stable_value_loss
from automata.training.native_receipts import (
    NativeDatasetCompletionReceipt,
    NativeGameCompletionReceipt,
)
from automata.training.native_replay import (
    NativeReplayCatalog,
    NativeReplayConfig,
    NativeReplaySample,
    sample_native_replay,
    update_native_replay_catalog,
)
from automata.training.native_splits import NativeSeedRange, NativeSplitConfig
from automata.training.native_trainer import (
    BoundNativeReplaySample,
    NativeReplayDatasetBinding,
    NativeTrainer,
    NativeTrainerConfig,
    NativeTrainerInitialization,
    bind_native_replay_sample,
    create_native_trainer,
)
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget
from goa2.domain.models import TeamColor
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


@dataclass(frozen=True)
class ReplayFixture:
    catalog: NativeReplayCatalog
    sample: NativeReplaySample
    binding: NativeReplayDatasetBinding
    bound: BoundNativeReplaySample


def _config(**changes: object) -> NativeTrainerConfig:
    values: dict[str, object] = {
        "seed": 17,
        "learning_rate": 1e-3,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "entropy_weight": 0.01,
        "l2_weight": 1e-5,
        "max_gradient_norm": 10.0,
        "token_width": 8,
        "state_width": 12,
        "candidate_width": 8,
        "message_passing_layers": 1,
        "dropout": 0.0,
    }
    values.update(changes)
    return NativeTrainerConfig.model_validate(values, strict=True)


def _split_config() -> NativeSplitConfig:
    return NativeSplitConfig(
        namespace="native-trainer-tests",
        salt="fixed-test-salt",
        validation_fraction=0.01,
        seed_ranges=(NativeSeedRange(purpose="training", start=0, stop=10_000),),
    )


def _train_seeds(count: int) -> tuple[int, ...]:
    from automata.training.native_splits import (
        create_native_split_ledger,
        extend_native_split_ledger,
    )

    config = _split_config()
    ledger = create_native_split_ledger(config)
    result: list[int] = []
    for seed in range(10_000):
        candidate = extend_native_split_ledger(ledger, (seed,))
        if candidate.split_for_seed(seed) == "train":
            result.append(seed)
            ledger = candidate
            if len(result) == count:
                return tuple(result)
    raise AssertionError("test split did not provide enough train seeds")


def _identity(seed: int, *, generation: str = "trainer-generation") -> NativeGameIdentity:
    values: dict[str, Any] = {
        "world_seed": seed,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": generation,
        "source_revision": "trainer-test",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "fixture-search",
        "generator_config_id": "fixture-generator",
    }
    return NativeGameIdentity(game_id=native_game_id(**values), **values)


def _records(
    identity: NativeGameIdentity, *, policy_count: int, value_count: int
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


def _replay_fixture(
    root: Path,
    *,
    counts: tuple[tuple[int, int], ...] = ((2, 3), (0, 2), (4, 0)),
    chunk_size: int = 2,
) -> ReplayFixture:
    source_root = root / "source"
    source_root.mkdir(parents=True)
    completions: list[NativeGameCompletionReceipt] = []
    for ordinal, (seed, (policy_count, value_count)) in enumerate(
        zip(_train_seeds(len(counts)), counts, strict=True)
    ):
        identity = _identity(seed)
        logical_name = f"games/game-{ordinal}.jsonl"
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

    from automata.training.native_indexed_dataset import build_native_indexed_dataset

    cache = root / "index"
    dataset = build_native_indexed_dataset(
        source_root, source_receipt_path, cache, chunk_size=chunk_size
    )
    catalog = update_native_replay_catalog(
        root / "catalog.json",
        config=NativeReplayConfig(split_config=_split_config()),
        dataset=dataset,
        completion_receipt=completion,
        parent_artifact=None,
    )
    sample = sample_native_replay(catalog, game_count=len(counts), sampling_seed=91)
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
    return ReplayFixture(
        catalog=catalog,
        sample=sample,
        binding=binding,
        bound=bind_native_replay_sample(catalog, sample, bindings=(binding,)),
    )


def _fresh(config: NativeTrainerConfig | None = None) -> NativeTrainer:
    return create_native_trainer(
        config or _config(),
        initialization=NativeTrainerInitialization(mode="FRESH_BOOTSTRAP"),
    )


def _parameters(trainer: NativeTrainer) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        item.relative_to(root).as_posix(): item.read_bytes()
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def _direct_objective(
    trainer: NativeTrainer, bound: BoundNativeReplaySample
) -> tuple[float, float, float, float, float, dict[str, torch.Tensor]]:
    trainer.model.zero_grad(set_to_none=True)
    policy_ce = policy_entropy = value_bce = 0.0
    sample = bound.sample
    config = trainer.config
    for game in bound.games:
        if config.policy_weight > 0.0:
            for batch in game.dataset.iter_policy_training_chunks(game.reference.game.game_id):
                loss = native_policy_loss(
                    trainer.model.forward_policy(batch.batch),
                    legal_mask=batch.batch.candidates.mask,
                    policy_targets=batch.policy_targets,
                    row_weights=batch.row_weights,
                    row_mask=batch.row_mask,
                    head_normalizer=float(sample.policy_contributing_game_count),
                    entropy_weight=config.entropy_weight,
                )
                (config.policy_weight * loss.total).backward()
                policy_ce += float(loss.cross_entropy.detach())
                policy_entropy += float(loss.entropy.detach())
        if config.value_weight > 0.0:
            for batch in game.dataset.iter_value_training_chunks(game.reference.game.game_id):
                loss = native_stable_value_loss(
                    trainer.model.forward_stable_value(batch.batch),
                    value_targets=batch.value_targets,
                    row_weights=batch.row_weights,
                    row_mask=batch.row_mask,
                    head_normalizer=float(sample.value_contributing_game_count),
                )
                (config.value_weight * loss.total).backward()
                value_bce += float(loss.bce.detach())
    regularization = sum(
        (parameter.square().sum() for parameter in trainer.model.parameters()),
        torch.zeros((), dtype=torch.float32),
    )
    (config.l2_weight * regularization).backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }
    norm = math.sqrt(sum(float(gradient.square().sum()) for gradient in gradients.values()))
    total = (
        config.policy_weight * (policy_ce - config.entropy_weight * policy_entropy)
        + config.value_weight * value_bce
        + config.l2_weight * float(regularization.detach())
    )
    return (
        policy_ce,
        policy_entropy,
        value_bce,
        float(regularization.detach()),
        total,
        gradients | {"__norm__": torch.tensor(norm)},
    )


def test_config_is_strict_finite_canonical_and_constrains_dropout() -> None:
    first = _config()
    second = _config()

    assert first.digest == second.digest
    assert first.canonical_bytes() == second.canonical_bytes()
    with pytest.raises(ValidationError, match="dropout"):
        _config(dropout=0.1)
    with pytest.raises(ValidationError, match=r"finite|learning_rate"):
        _config(learning_rate=math.inf)
    with pytest.raises(ValidationError, match=r"policy|value|positive"):
        _config(policy_weight=0.0, value_weight=0.0)
    with pytest.raises(ValidationError):
        _config(seed=True)


def test_binding_paths_must_be_absolute() -> None:
    values: dict[str, object] = {
        "dataset_digest": "a" * 64,
        "source_digest": "b" * 64,
        "completion_receipt_digest": "c" * 64,
        "source_root": Path("relative-source"),
        "source_receipt_path": Path("relative-source-receipt.json"),
        "completion_receipt_path": Path("relative-completion.json"),
        "index_cache_dir": Path("relative-index"),
        "chunk_size": 2,
    }

    with pytest.raises(ValidationError, match=r"absolute|source_root"):
        NativeReplayDatasetBinding.model_validate(values, strict=True)


def test_binding_requires_exact_catalog_train_membership_and_exact_binding_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _replay_fixture(tmp_path)
    first = fixture.sample.games[0]
    changed = first.model_copy(update={"source_logical_name": "games/other.jsonl"})
    tampered = fixture.sample.model_copy(update={"games": (changed, *fixture.sample.games[1:])})
    with pytest.raises(ValueError, match=r"exact retained|catalog"):
        bind_native_replay_sample(fixture.catalog, tampered, bindings=(fixture.binding,))

    with pytest.raises(ValueError, match=r"exactly|selected"):
        bind_native_replay_sample(fixture.catalog, fixture.sample, bindings=())
    with pytest.raises(ValueError, match="unambiguous"):
        bind_native_replay_sample(
            fixture.catalog,
            fixture.sample,
            bindings=(fixture.binding, fixture.binding),
        )

    assert fixture.catalog.compatibility is not None
    incompatible = fixture.catalog.model_copy(
        update={
            "compatibility": fixture.catalog.compatibility.model_copy(
                update={"decision_tensor_schema_digest": "0" * 64}
            )
        }
    )
    with pytest.raises(ValueError, match="tensor compatibility"):
        bind_native_replay_sample(incompatible, fixture.sample, bindings=(fixture.binding,))

    from automata.training.native_splits import NativeSeedSplitLedger

    def forbidden_lookup(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("binding must consume the already validated assignment map")

    monkeypatch.setattr(NativeSeedSplitLedger, "split_for_seed", forbidden_lookup)
    rebound = bind_native_replay_sample(
        fixture.catalog, fixture.sample, bindings=(fixture.binding,)
    )
    assert rebound.sample == fixture.sample


def test_consumption_revalidates_catalog_models_and_physical_source_completion_and_index(
    tmp_path: Path,
) -> None:
    fixtures = tuple(_replay_fixture(tmp_path / name) for name in ("source", "receipt", "index"))
    mutations = (
        lambda item: (
            item.binding.source_root / item.sample.games[0].source_logical_name
        ).write_bytes(b"changed\n"),
        lambda item: item.binding.completion_receipt_path.write_bytes(
            item.binding.completion_receipt_path.read_bytes() + b"\n"
        ),
        lambda item: next((item.binding.index_cache_dir / "chunks").rglob("*.zst")).write_bytes(
            b"changed"
        ),
    )
    for fixture, mutate in zip(fixtures, mutations, strict=True):
        trainer = _fresh()
        before = _parameters(trainer)
        mutate(fixture)
        with pytest.raises(ValueError):
            trainer.train_logical_batch(fixture.bound)
        assert trainer.optimizer_steps == 0
        assert trainer.optimizer.state == {}
        assert all(parameter.grad is None for parameter in trainer.model.parameters())
        for name, parameter in trainer.model.named_parameters():
            assert torch.equal(parameter, before[name])

    fixture = _replay_fixture(tmp_path / "models")
    broken_catalog = fixture.catalog.model_copy(
        update={"games": tuple(reversed(fixture.catalog.games))}
    )
    tampered = replace(fixture.bound, catalog=broken_catalog)
    with pytest.raises(ValueError, match=r"retained|history|insertion"):
        _fresh().train_logical_batch(tampered)


def test_consumption_rejects_unselected_cache_corruption_without_rebuilding_or_step(
    tmp_path: Path,
) -> None:
    fixture = _replay_fixture(tmp_path)
    selected = fixture.catalog.games[0]
    sample = NativeReplaySample(
        games=(selected,),
        policy_contributing_game_count=int(selected.policy_row_count > 0),
        value_contributing_game_count=int(selected.value_row_count > 0),
    )
    bound = bind_native_replay_sample(fixture.catalog, sample, bindings=(fixture.binding,))
    unselected = next(
        reference
        for reference in fixture.catalog.games
        if reference.game.game_id != selected.game.game_id
    )
    metadata = bound.games[0].dataset.game(unselected.game.game_id)
    chunk = (*metadata.policy_chunks, *metadata.value_chunks)[0]
    chunk_path = fixture.binding.index_cache_dir / chunk.path
    chunk_path.write_bytes(b"corrupt-unselected-chunk")
    cache_before = _tree_bytes(fixture.binding.index_cache_dir)
    trainer = _fresh()
    weights_before = _parameters(trainer)

    with pytest.raises(ValueError, match=r"cache|compatible|rebuild|chunk"):
        trainer.train_logical_batch(bound)

    assert _tree_bytes(fixture.binding.index_cache_dir) == cache_before
    assert trainer.optimizer_steps == 0
    assert trainer.optimizer.state == {}
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, weights_before[name])


def test_one_step_matches_direct_loss_grad_norm_and_updates_across_chunk_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _replay_fixture(tmp_path, chunk_size=1)
    other_binding = fixture.binding.model_copy(
        update={"index_cache_dir": tmp_path / "other-index", "chunk_size": 4}
    )
    other_bound = bind_native_replay_sample(
        fixture.catalog, fixture.sample, bindings=(other_binding,)
    )
    config = _config(max_gradient_norm=0.01)
    direct = _fresh(config)
    expected = _direct_objective(direct, fixture.bound)
    expected_gradients = expected[-1]
    expected_norm = float(expected_gradients.pop("__norm__"))

    first = _fresh(config)
    counts = {"clip": 0, "step": 0, "zero": 0}
    observed_gradients: dict[str, torch.Tensor] = {}
    real_clip = torch.nn.utils.clip_grad_norm_
    real_step = first.optimizer.step
    real_zero = first.optimizer.zero_grad

    def clip(*args: Any, **kwargs: Any) -> torch.Tensor:
        counts["clip"] += 1
        observed_gradients.update(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in first.model.named_parameters()
                if parameter.grad is not None
            }
        )
        return real_clip(*args, **kwargs)

    def step(*args: Any, **kwargs: Any) -> Any:
        counts["step"] += 1
        return real_step(*args, **kwargs)

    def zero(*args: Any, **kwargs: Any) -> Any:
        counts["zero"] += 1
        return real_zero(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(first.optimizer, "step", step)
    monkeypatch.setattr(first.optimizer, "zero_grad", zero)
    first_result = first.train_logical_batch(fixture.bound)
    assert counts == {"clip": 1, "step": 1, "zero": 1}
    assert (
        first_result.policy_cross_entropy,
        first_result.policy_entropy,
        first_result.value_bce,
        first_result.regularization,
        first_result.total_loss,
    ) == pytest.approx(expected[:5], rel=2e-6, abs=2e-7)
    assert first_result.gradient_norm_before_clip == pytest.approx(
        expected_norm, rel=2e-5, abs=2e-6
    )
    assert first_result.gradient_norm_before_clip > config.max_gradient_norm
    assert observed_gradients.keys() == expected_gradients.keys()
    for name in observed_gradients:
        assert torch.allclose(
            observed_gradients[name], expected_gradients[name], rtol=2e-5, atol=2e-6
        ), name
    clipped_norm = math.sqrt(
        sum(
            float(parameter.grad.square().sum())
            for parameter in first.model.parameters()
            if parameter.grad is not None
        )
    )
    assert clipped_norm == pytest.approx(config.max_gradient_norm, rel=2e-4)

    second = _fresh(config)
    second_result = second.train_logical_batch(other_bound)
    assert (
        first_result.policy_cross_entropy,
        first_result.policy_entropy,
        first_result.value_bce,
        first_result.regularization,
        first_result.total_loss,
        first_result.gradient_norm_before_clip,
    ) == pytest.approx(
        (
            second_result.policy_cross_entropy,
            second_result.policy_entropy,
            second_result.value_bce,
            second_result.regularization,
            second_result.total_loss,
            second_result.gradient_norm_before_clip,
        ),
        rel=2e-5,
        abs=2e-6,
    )
    for (name, first_parameter), (_, second_parameter) in zip(
        first.model.named_parameters(), second.model.named_parameters(), strict=True
    ):
        assert torch.allclose(first_parameter, second_parameter, rtol=2e-5, atol=2.5e-4), name
    assert first.optimizer_steps == second.optimizer_steps == 1
    assert first.optimizer.state and second.optimizer.state
    assert first_result.provenance.policy_contributing_game_count == 2
    assert first_result.provenance.value_contributing_game_count == 2


def test_l2_disabled_leaves_disabled_head_without_grad_or_adam_state(tmp_path: Path) -> None:
    fixture = _replay_fixture(tmp_path)
    trainer = _fresh(_config(value_weight=0.0, l2_weight=0.0))
    value_parameters = trainer.model.parameter_groups()["value"]
    before = {id(parameter): parameter.detach().clone() for parameter in value_parameters}

    result = trainer.train_logical_batch(fixture.bound)

    assert result.regularization == 0.0
    assert all(parameter.grad is None for parameter in value_parameters)
    assert all(parameter not in trainer.optimizer.state for parameter in value_parameters)
    assert all(torch.equal(parameter, before[id(parameter)]) for parameter in value_parameters)
    assert trainer.optimizer.state


def test_missing_or_disabled_heads_use_independent_denominators(tmp_path: Path) -> None:
    fixture = _replay_fixture(tmp_path)
    policy_only = _fresh(_config(value_weight=0.0))
    result = policy_only.train_logical_batch(fixture.bound)
    assert result.value_bce == 0.0
    assert result.provenance.policy_contributing_game_count == 2
    assert result.provenance.value_contributing_game_count == 2

    value_only_ref = next(game for game in fixture.sample.games if game.policy_row_count == 0)
    value_only_sample = NativeReplaySample(
        games=(value_only_ref,),
        policy_contributing_game_count=0,
        value_contributing_game_count=1,
    )
    value_only_bound = bind_native_replay_sample(
        fixture.catalog, value_only_sample, bindings=(fixture.binding,)
    )
    trainer = _fresh(_config(value_weight=0.0))
    called = False
    original = trainer.optimizer.zero_grad

    def zero_grad(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return original(*args, **kwargs)

    trainer.optimizer.zero_grad = zero_grad  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="no enabled"):
        trainer.train_logical_batch(value_only_bound)
    assert called is False
    assert trainer.optimizer_steps == 0


def test_config_initialization_and_explicit_adam_contract_are_pinned(
    tmp_path: Path,
) -> None:
    fixture = _replay_fixture(tmp_path)
    trainer = _fresh()
    original_config = trainer.config
    original_initialization = trainer.initialization
    before = _parameters(trainer)
    changes = (
        {"l2_weight": 0.0},
        {"value_weight": 0.0},
        {"entropy_weight": 0.0},
        {"max_gradient_norm": 0.5},
        {"seed": 18},
    )
    for change in changes:
        trainer.config = original_config.model_copy(update=change)
        with pytest.raises(ValueError, match=r"config|pinned|changed"):
            trainer.train_logical_batch(fixture.bound)
        assert trainer.optimizer_steps == 0
        assert trainer.optimizer.state == {}
        assert all(parameter.grad is None for parameter in trainer.model.parameters())
        for name, parameter in trainer.model.named_parameters():
            assert torch.equal(parameter, before[name])
        trainer.config = original_config

    trainer.initialization = original_initialization.model_copy(
        update={"mode": "GEN1_PARENT", "parent_model_digest": "0" * 64}
    )
    with pytest.raises(ValueError, match=r"initialization|pinned|changed"):
        trainer.train_logical_batch(fixture.bound)
    trainer.initialization = original_initialization

    group = trainer.optimizer.param_groups[0]
    assert {
        "foreach": group["foreach"],
        "capturable": group["capturable"],
        "differentiable": group["differentiable"],
        "fused": group["fused"],
        "amsgrad": group["amsgrad"],
        "maximize": group["maximize"],
        "weight_decay": group["weight_decay"],
    } == {
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "amsgrad": False,
        "maximize": False,
        "weight_decay": 0.0,
    }
    group["foreach"] = True
    with pytest.raises(ValueError, match=r"Adam|contract"):
        trainer.train_logical_batch(fixture.bound)
    group["foreach"] = False

    step = trainer.train_logical_batch(fixture.bound)
    trainer.config = original_config.model_copy(update={"entropy_weight": 0.0})
    destination = tmp_path / "forbidden-export"
    with pytest.raises(ValueError, match=r"config|pinned|changed"):
        trainer.export_artifact(
            destination,
            step=step,
            source_revision="abc",
            dirty_tree_hash="clean",
        )
    assert not destination.exists()


def test_fresh_initialization_is_rng_isolated_seeded_cpu_float32_and_empty_adam() -> None:
    torch.manual_seed(909)
    before = torch.random.get_rng_state().clone()
    cuda_before = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    mps_get_rng_state = getattr(torch.mps, "get_rng_state", None)
    mps_before = (
        mps_get_rng_state().clone()
        if torch.backends.mps.is_available() and mps_get_rng_state is not None
        else None
    )
    first = _fresh()
    after = torch.random.get_rng_state().clone()
    second = _fresh()
    different = _fresh(_config(seed=18))

    assert torch.equal(before, after)
    if cuda_before:
        assert all(
            torch.equal(expected, actual)
            for expected, actual in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True)
        )
    if mps_before is not None:
        assert torch.equal(mps_before, mps_get_rng_state())
    assert first.optimizer.state == second.optimizer.state == different.optimizer.state == {}
    assert all(
        parameter.device.type == "cpu" and parameter.dtype == torch.float32
        for parameter in first.model.parameters()
    )
    for left, right in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.equal(left, right)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.model.parameters(), different.model.parameters(), strict=True)
    )
    with pytest.raises(ValueError, match="must not"):
        create_native_trainer(
            _config(),
            initialization=NativeTrainerInitialization(mode="FRESH_BOOTSTRAP"),
            parent_artifact_path="unused",
        )


def test_export_has_full_scope_lineage_rejects_stale_foreign_and_loads_as_parent(
    tmp_path: Path,
) -> None:
    fixture = _replay_fixture(tmp_path / "replay")
    trainer = _fresh()
    first = trainer.train_logical_batch(fixture.bound)
    second = trainer.train_logical_batch(fixture.bound)
    with pytest.raises(ValueError, match=r"exact latest|stale"):
        trainer.export_artifact(
            tmp_path / "stale", step=first, source_revision="abc", dirty_tree_hash="clean"
        )
    with pytest.raises(ValueError, match=r"exact latest"):
        trainer.export_artifact(
            tmp_path / "foreign",
            step=replace(second),
            source_revision="abc",
            dirty_tree_hash="clean",
        )

    destination = tmp_path / "artifact"
    manifest = trainer.export_artifact(
        destination,
        step=second,
        source_revision="abc",
        dirty_tree_hash="clean",
    )
    scope = current_gen1_artifact_scope()
    assert manifest.supported_heroes == scope.supported_heroes
    assert manifest.supported_maps == scope.supported_maps
    assert manifest.supported_game_types == scope.supported_game_types
    assert manifest.hero_adapter_versions == scope.hero_adapter_versions
    provenance = json.loads((destination / "provenance.json").read_bytes())
    assert provenance["trainer_config_digest"] == trainer.config.digest
    assert provenance["initialization"] == {
        "mode": "FRESH_BOOTSTRAP",
        "parent_model_digest": None,
    }
    assert [item["optimizer_step"] for item in provenance["successful_steps"]] == [1, 2]
    assert (
        provenance["successful_steps"][-1]["provenance"]["replay_catalog_digest"]
        == fixture.catalog.digest
    )
    assert not any(
        str(fixture.binding.source_root) in json.dumps(item)
        for item in provenance["successful_steps"]
    )

    requirements = current_gen1_runtime_requirements(
        heroes=("Wasp", "Arien"), map_id="forgotten_island", game_type="QUICK"
    )
    loaded = load_gen1_model_artifact(destination, requirements=requirements)
    parent = create_native_trainer(
        trainer.config,
        initialization=NativeTrainerInitialization(
            mode="GEN1_PARENT", parent_model_digest=manifest.model_digest
        ),
        parent_artifact_path=destination,
    )
    assert parent.optimizer.state == {}
    assert parent.optimizer_steps == 0
    for expected, actual in zip(loaded.model.parameters(), parent.model.parameters(), strict=True):
        assert torch.equal(expected, actual)
        assert actual.requires_grad

    with pytest.raises(ValueError, match=r"digest|parent"):
        create_native_trainer(
            trainer.config,
            initialization=NativeTrainerInitialization(
                mode="GEN1_PARENT", parent_model_digest="0" * 64
            ),
            parent_artifact_path=destination,
        )
    with pytest.raises(ValueError, match="architecture"):
        create_native_trainer(
            _config(token_width=9),
            initialization=NativeTrainerInitialization(
                mode="GEN1_PARENT", parent_model_digest=manifest.model_digest
            ),
            parent_artifact_path=destination,
        )


def test_nonfinite_failure_preserves_weights_and_optimizer_then_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _replay_fixture(tmp_path)
    trainer = _fresh()
    before = _parameters(trainer)
    original = trainer.model.forward_policy

    def nonfinite(*args: Any, **kwargs: Any) -> PolicyHeadOutput:
        output = original(*args, **kwargs)
        return PolicyHeadOutput(policy_logits=output.policy_logits * float("nan"))

    monkeypatch.setattr(trainer.model, "forward_policy", nonfinite)
    with pytest.raises(ValueError, match=r"finite|policy_logits"):
        trainer.train_logical_batch(fixture.bound)
    assert trainer.optimizer_steps == 0
    assert trainer.optimizer.state == {}
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, before[name])

    monkeypatch.setattr(trainer.model, "forward_policy", original)
    assert trainer.train_logical_batch(fixture.bound).optimizer_step == 1


def test_optimizer_failure_poisons_trainer_without_claiming_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _replay_fixture(tmp_path / "replay")
    trainer = _fresh()
    successful = trainer.train_logical_batch(fixture.bound)

    def partial_failure(*args: Any, **kwargs: Any) -> None:
        with torch.no_grad():
            next(trainer.model.parameters()).add_(1.0)
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(trainer.optimizer, "step", partial_failure)
    with pytest.raises(RuntimeError, match="injected"):
        trainer.train_logical_batch(fixture.bound)
    assert trainer.optimizer_steps == 1
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.train_logical_batch(fixture.bound)
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.export_artifact(
            tmp_path / "forbidden",
            step=successful,
            source_revision="abc",
            dirty_tree_hash="clean",
        )
