"""Native index chunks feed separate model/loss heads without changing game mass."""

from __future__ import annotations

import hashlib
import io
import shutil
from pathlib import Path

import pytest
import torch
import zstandard

from automata.decision import DecisionDescriptor
from automata.models.contracts import canonical_json_bytes
from automata.models.shared_encoder.gen1_model import Gen1ModelConfig, Gen1PolicyValueModel
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.observation import encode_decision, legal_keys_for_decision
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.value_boundary import detect_stable_value_boundary
from automata.training.native_dataset import (
    NativeBoundaryProvenance,
    NativeGameIdentity,
    PolicyDatasetRecord,
    ValueDatasetRecord,
    iter_native_game_records,
    native_game_id,
    native_sample_id,
    publish_native_game,
)
from automata.training.native_indexed_dataset import (
    IndexedNativeDataset,
    build_native_indexed_dataset,
    create_native_source_receipt,
    open_native_indexed_dataset,
)
from automata.training.native_losses import native_policy_loss, native_stable_value_loss
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget
from goa2.domain.models import TeamColor
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def _publish_fixture_game(
    path: Path, *, seed: int, generation: str, policy_count: int, value_count: int
) -> str:
    """Publish schema-valid fixtures, not an assertion of real gameplay completion."""
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=seed,
    )
    identity = dict(
        world_seed=seed,
        map_id="forgotten_island",
        game_type="QUICK",
        red_composition=("Wasp",),
        blue_composition=("Arien",),
        generation_id=generation,
        source_revision="index-loss-test",
        dirty_tree_hash="clean",
        source_model_digest=None,
        search_config_id="fixture-search",
        generator_config_id="fixture-generator",
    )
    game = NativeGameIdentity(game_id=native_game_id(**identity), **identity)
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    decision = DecisionDescriptor("CARD", hero=hero)
    observation = encode_decision(
        state,
        decision,
        legal_keys_for_decision(decision),
        decision_owner_hero_id=hero.id,
        perspective_team="RED",
    )
    target = SearchPolicyTarget(
        schema_version=1,
        actions=tuple(
            SearchActionTarget(
                schema_version=1,
                candidate=candidate,
                prior_probability=1.0 / len(observation.candidates),
                sample_count=1,
                mean_value=0.0,
                value_variance=0.0,
                improved_probability=1.0 / len(observation.candidates),
                selected=index == 0,
            )
            for index, candidate in enumerate(observation.candidates)
        ),
    )
    records: list[PolicyDatasetRecord | ValueDatasetRecord] = []
    for index in range(policy_count):
        records.append(
            PolicyDatasetRecord(
                game=game,
                sample_id=native_sample_id(
                    game_id=game.game_id, sample_kind="POLICY", sample_index=index
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
        state, boundary, viewer_hero_id=hero.id, perspective_team=TeamColor.RED
    )
    viewer_ref = next(
        token.local_ref
        for token in value_observation.state.tokens
        if token.kind == "HERO" and token.features["relation"] == "SELF"
    )
    for index in range(value_count):
        sample_index = policy_count + index
        records.append(
            ValueDatasetRecord(
                game=game,
                sample_id=native_sample_id(
                    game_id=game.game_id, sample_kind="VALUE", sample_index=sample_index
                ),
                sample_index=sample_index,
                perspective_team="RED",
                boundary=NativeBoundaryProvenance(
                    boundary_index=index,
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
    publish_native_game(path, records)
    return game.game_id


def test_native_index_model_and_losses_preserve_per_head_mass_and_chunk_gradients(
    tmp_path: Path,
) -> None:
    names = ("short.jsonl", "long.jsonl.zst", "value-only.jsonl", "policy-only.jsonl")
    counts = ((2, 3), (5, 1), (0, 2), (1, 0))
    game_ids = tuple(
        _publish_fixture_game(
            tmp_path / name,
            seed=81,
            generation=f"fixture-{index}",
            policy_count=policy_count,
            value_count=value_count,
        )
        for index, (name, (policy_count, value_count)) in enumerate(zip(names, counts, strict=True))
    )
    receipt = create_native_source_receipt(tmp_path, names)
    receipt_path = tmp_path / "sources.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    decision_schema = TensorFeatureSchema.current()
    value_schema = StableValueTensorSchema.current()
    torch.manual_seed(17)
    model = Gen1PolicyValueModel(
        decision_schema=decision_schema,
        stable_value_schema=value_schema,
        config=Gen1ModelConfig(
            decision_schema_digest=decision_schema.digest,
            stable_value_schema_digest=value_schema.digest,
            token_width=8,
            state_width=12,
            candidate_width=8,
            message_passing_layers=1,
        ),
    )
    model.eval()
    results = []
    for chunk_size in (1, 4):
        indexed = build_native_indexed_dataset(
            tmp_path, receipt_path, tmp_path / f"index-{chunk_size}", chunk_size=chunk_size
        )
        assert indexed.game_ids == game_ids
        assert indexed.head_game_mask("POLICY") == (True, True, False, True)
        assert indexed.head_game_mask("VALUE") == (True, True, True, False)
        assert indexed.head_game_count("POLICY") == indexed.head_game_count("VALUE") == 3
        model.zero_grad(set_to_none=True)
        policy_total = 0.0
        value_total = 0.0
        for game_id, (policy_count, value_count) in zip(game_ids, counts, strict=True):
            policy_mass = 0.0
            value_mass = 0.0
            for chunk in indexed.iter_policy_training_chunks(game_id):
                assert len(chunk.game_ids) <= chunk_size
                policy_mass += float(chunk.row_weights.sum())
                policy_loss = native_policy_loss(
                    model.forward_policy(chunk.batch),
                    legal_mask=chunk.batch.candidates.mask,
                    policy_targets=chunk.policy_targets,
                    row_weights=chunk.row_weights,
                    row_mask=chunk.row_mask,
                    head_normalizer=indexed.head_game_count("POLICY"),
                )
                policy_total += float(policy_loss.total.detach())
                policy_loss.total.backward()
            for chunk in indexed.iter_value_training_chunks(game_id):
                assert len(chunk.game_ids) <= chunk_size
                value_mass += float(chunk.row_weights.sum())
                assert not hasattr(chunk.batch, "candidates")
                value_loss = native_stable_value_loss(
                    model.forward_stable_value(chunk.batch),
                    value_targets=chunk.value_targets,
                    row_weights=chunk.row_weights,
                    row_mask=chunk.row_mask,
                    head_normalizer=indexed.head_game_count("VALUE"),
                )
                value_total += float(value_loss.total.detach())
                value_loss.total.backward()
            assert policy_mass == pytest.approx(1.0 if policy_count else 0.0)
            assert value_mass == pytest.approx(1.0 if value_count else 0.0)
        gradients = {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients.values())
        results.append((indexed.digest, policy_total, value_total, gradients))
    first, second = results
    assert first[0] == second[0]
    assert first[1:3] == pytest.approx(second[1:3], rel=1e-6, abs=1e-7)
    assert first[3].keys() == second[3].keys()
    for name in first[3]:
        assert torch.allclose(first[3][name], second[3][name], rtol=1e-5, atol=1e-6), name


@pytest.mark.parametrize(
    ("root_name", "source_name", "receipt_name", "cache_name"),
    [
        ("sources", "sources/game.jsonl", "sources/receipt.json", "sources"),
        ("sources", "sources/game.jsonl", "sources/receipt.json", "."),
        ("sources", "sources/game.jsonl", "sources/receipt.json", "sources/game.jsonl"),
        (".index.staging", ".index.staging/game.jsonl", "receipt.json", "index"),
        ("sources", "sources/game.jsonl", "index/receipt.json", "index"),
        ("sources", "sources/cache/game.jsonl", "receipt.json", "sources/cache"),
        ("sources", "sources/game.jsonl", ".index.staging/receipt.json", "index"),
    ],
)
def test_index_never_replaces_its_sources_or_receipt(
    tmp_path: Path, root_name: str, source_name: str, receipt_name: str, cache_name: str
) -> None:
    source_root = tmp_path / root_name
    source_root.mkdir(parents=True)
    source = tmp_path / source_name
    _publish_fixture_game(source, seed=83, generation="fixture", policy_count=1, value_count=1)
    receipt_path = tmp_path / receipt_name
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    logical_name = source.relative_to(source_root).as_posix()
    receipt_path.write_bytes(
        create_native_source_receipt(source_root, (logical_name,)).canonical_bytes()
    )
    source_bytes = source.read_bytes()
    receipt_bytes = receipt_path.read_bytes()

    with pytest.raises(ValueError, match=r"overlap|source|receipt|cache"):
        build_native_indexed_dataset(source_root, receipt_path, tmp_path / cache_name)

    assert source.read_bytes() == source_bytes
    assert receipt_path.read_bytes() == receipt_bytes


def test_reopening_does_not_trust_a_changed_semantic_dataset_digest(tmp_path: Path) -> None:
    source = tmp_path / "game.jsonl"
    _publish_fixture_game(source, seed=84, generation="fixture", policy_count=1, value_count=1)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(
        create_native_source_receipt(tmp_path, (source.name,)).canonical_bytes()
    )
    cache = tmp_path / "index"
    index = build_native_indexed_dataset(tmp_path, receipt_path, cache)
    expected_digest = index.digest
    altered = index.manifest.model_copy(update={"semantic_dataset_digest": "0" * 64})
    (cache / "manifest.json").write_bytes(canonical_json_bytes(altered))

    reopened = open_native_indexed_dataset(tmp_path, receipt_path, cache)

    assert reopened.digest == expected_digest


def test_native_dataset_identity_is_portable_and_independent_of_compression(
    tmp_path: Path,
) -> None:
    plain_root = tmp_path / "plain"
    compressed_root = tmp_path / "compressed"
    relocated_root = tmp_path / "relocated"
    plain = plain_root / "game.jsonl"
    _publish_fixture_game(plain, seed=85, generation="fixture", policy_count=1, value_count=1)
    compressed = compressed_root / "game.jsonl.zst"
    publish_native_game(compressed, iter_native_game_records(plain))
    relocated_root.mkdir()
    shutil.copy2(plain, relocated_root / plain.name)
    plain_receipt = create_native_source_receipt(plain_root, (plain.name,))
    assert create_native_source_receipt(relocated_root, (plain.name,)) == plain_receipt
    receipt_path = tmp_path / "plain-receipt.json"
    receipt_path.write_bytes(plain_receipt.canonical_bytes())
    compressed_receipt_path = tmp_path / "compressed-receipt.json"
    compressed_receipt_path.write_bytes(
        create_native_source_receipt(compressed_root, (compressed.name,)).canonical_bytes()
    )
    cache = tmp_path / "plain-index"
    plain_index = build_native_indexed_dataset(plain_root, receipt_path, cache)
    compressed_index = build_native_indexed_dataset(
        compressed_root, compressed_receipt_path, tmp_path / "compressed-index"
    )
    relocated_index = open_native_indexed_dataset(relocated_root, receipt_path, cache)

    assert plain_index.digest == compressed_index.digest == relocated_index.digest
    assert plain_index.digest == hashlib.sha256(plain.read_bytes()).hexdigest()
    assert plain_index.source_digest != compressed_index.source_digest
    assert plain_index.source_digest == relocated_index.source_digest


@pytest.mark.parametrize(
    "damage", ["truncated", "trailing", "second-frame", "oversized-known", "oversized-unknown"]
)
def test_chunk_loader_rejects_invalid_frames_even_with_matching_file_hashes(
    tmp_path: Path, damage: str
) -> None:
    source = tmp_path / "game.jsonl"
    game_id = _publish_fixture_game(
        source, seed=86, generation="fixture", policy_count=1, value_count=1
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(
        create_native_source_receipt(tmp_path, (source.name,)).canonical_bytes()
    )
    cache = tmp_path / "index"
    index = build_native_indexed_dataset(tmp_path, receipt_path, cache)
    game = index.game(game_id)
    metadata = game.policy_chunks[0]
    path = cache / metadata.path
    original = path.read_bytes()
    if damage == "truncated":
        damaged = original[:-1]
    elif damage == "trailing":
        damaged = original + b"garbage"
    elif damage == "second-frame":
        damaged = original + original
    elif damage == "oversized-known":
        damaged = zstandard.ZstdCompressor().compress(b"x" * (metadata.uncompressed_size * 4))
    else:
        stream = io.BytesIO()
        with zstandard.ZstdCompressor().stream_writer(stream, closefd=False) as writer:
            writer.write(b"x" * (metadata.uncompressed_size * 4))
        damaged = stream.getvalue()
    path.write_bytes(damaged)
    changed_chunk = metadata.model_copy(
        update={"sha256": hashlib.sha256(damaged).hexdigest(), "file_size": len(damaged)}
    )
    changed_game = game.model_copy(update={"policy_chunks": (changed_chunk,)})
    changed_manifest = index.manifest.model_copy(update={"games": (changed_game,)})
    opened = IndexedNativeDataset(cache, tmp_path, changed_manifest)

    with pytest.raises(ValueError, match=r"frame|size|bounded"):
        tuple(opened.iter_policy_training_chunks(game_id))
