"""Receipt-bound native policy/value index behavior and integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automata.decision import DecisionDescriptor
from automata.harness.game_runner import DEFAULT_MAP
from automata.observation import encode_decision
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
    NativeDatasetSourceReceipt,
    NativeGameSourceReceipt,
    build_native_indexed_dataset,
    create_native_source_receipt,
    load_native_source_receipt,
    open_native_indexed_dataset,
)
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget
from goa2.domain.models import TeamColor
from goa2.engine.setup import GameSetup

_OWNER_MARKER_NAME = ".native-index-owner"
_OWNER_MARKER_CONTENT = b"automata-native-index-v1\n"


def _identity(seed: int) -> NativeGameIdentity:
    values: dict[str, Any] = {
        "world_seed": seed,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": "generation-1",
        "source_revision": "abc123",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "search-1",
        "generator_config_id": "generator-1",
    }
    return NativeGameIdentity(game_id=native_game_id(**values), **values)


def _state(seed: int):
    register_all_effects()
    return GameSetup.create_game(
        DEFAULT_MAP,
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=seed,
    )


def _policy(game: NativeGameIdentity, sample_index: int, policy_index: int) -> PolicyDatasetRecord:
    state = _state(game.world_seed)
    hero = state.teams[TeamColor.RED].heroes[0]
    hero.hand = hero.hand[:1]
    observation = encode_decision(
        state,
        DecisionDescriptor("CARD", hero=hero),
        tuple(card.id for card in hero.hand),
        decision_owner_hero_id=str(hero.id),
        perspective_team=TeamColor.RED.value,
    )
    candidate = observation.candidates[0]
    return PolicyDatasetRecord(
        game=game,
        sample_id=native_sample_id(
            game_id=game.game_id,
            sample_kind="POLICY",
            sample_index=sample_index,
        ),
        sample_index=sample_index,
        policy_index=policy_index,
        perspective_team="RED",
        observation=observation,
        target=SearchPolicyTarget(
            actions=(
                SearchActionTarget(
                    candidate=candidate,
                    prior_probability=1.0,
                    sample_count=1,
                    mean_value=0.5,
                    value_variance=0.25,
                    improved_probability=1.0,
                    selected=True,
                ),
            )
        ),
    )


def _value(game: NativeGameIdentity, sample_index: int, boundary_index: int) -> ValueDatasetRecord:
    state = _state(game.world_seed)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    viewer = state.teams[TeamColor.RED].heroes[0]
    observation = encode_stable_value(
        state,
        boundary,
        viewer_hero_id=str(viewer.id),
        perspective_team=TeamColor.RED,
    )
    self_token = next(
        token
        for token in observation.state.tokens
        if token.kind == "HERO" and token.features.get("relation") == "SELF"
    )
    return ValueDatasetRecord(
        game=game,
        sample_id=native_sample_id(
            game_id=game.game_id,
            sample_kind="VALUE",
            sample_index=sample_index,
        ),
        sample_index=sample_index,
        perspective_team="RED",
        boundary=NativeBoundaryProvenance(
            boundary_index=boundary_index,
            kind=boundary.kind.value,
            round=boundary.round,
            turn=boundary.turn,
            viewer_ref=self_token.local_ref,
            actor_ref=None,
        ),
        observation=observation,
        terminal_winner="RED",
        value_target=1,
    )


def _publish_sources(root: Path) -> tuple[tuple[str, ...], tuple[NativeGameIdentity, ...]]:
    identities = (_identity(41), _identity(42), _identity(43))
    logical_names = ("nested/mixed.jsonl.zst", "policy.jsonl", "value.jsonl.zst")
    publish_native_game(
        root / logical_names[0],
        (
            _policy(identities[0], 0, 0),
            _value(identities[0], 1, 0),
            _policy(identities[0], 2, 1),
            _value(identities[0], 3, 1),
            _policy(identities[0], 4, 2),
        ),
    )
    publish_native_game(
        root / logical_names[1],
        (_policy(identities[1], 0, 0), _policy(identities[1], 1, 1)),
    )
    publish_native_game(
        root / logical_names[2],
        (_value(identities[2], 0, 0), _value(identities[2], 1, 1)),
    )
    return logical_names, identities


def _write_receipt(
    root: Path, logical_names: tuple[str, ...], path: Path
) -> NativeDatasetSourceReceipt:
    receipt = create_native_source_receipt(root, logical_names)
    path.write_bytes(receipt.canonical_bytes())
    return receipt


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        item.relative_to(root).as_posix(): item.read_bytes()
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def test_receipt_is_explicit_portable_and_not_terminal_certification(tmp_path: Path) -> None:
    source = tmp_path / "source"
    logical_names, identities = _publish_sources(source)
    (source / "unlisted.jsonl").write_bytes(b"not a native stream")

    receipt = create_native_source_receipt(source, logical_names)
    relocated = tmp_path / "relocated"
    for logical_name in logical_names:
        destination = relocated / logical_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / logical_name).read_bytes())

    assert receipt == create_native_source_receipt(relocated, logical_names)
    assert tuple(game.logical_name for game in receipt.games) == logical_names
    assert tuple(game.game_id for game in receipt.games) == tuple(
        identity.game_id for identity in identities
    )
    assert receipt.digest == hashlib.sha256(receipt.canonical_bytes()).hexdigest()
    assert "not gameplay certification" in (create_native_source_receipt.__doc__ or "")

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    assert load_native_source_receipt(receipt_path) == receipt
    receipt_path.write_bytes(json.dumps(receipt.model_dump(mode="json"), indent=2).encode())
    with pytest.raises(ValueError, match="canonical"):
        load_native_source_receipt(receipt_path)


@pytest.mark.parametrize("logical_name", ["../game.jsonl", "/game.jsonl", "nested/../game.jsonl"])
def test_receipt_rejects_unsafe_logical_paths(tmp_path: Path, logical_name: str) -> None:
    with pytest.raises((ValueError, ValidationError), match=r"logical|path|relative"):
        NativeGameSourceReceipt(
            logical_name=logical_name,
            file_sha256="0" * 64,
            file_size=1,
            game_id="1" * 64,
            row_count=1,
            policy_row_count=1,
            value_row_count=0,
            boundary_count=0,
        )


def test_index_builds_homogeneous_chunks_with_full_game_weights_and_missing_heads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    names, identities = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    receipt = _write_receipt(source, names, receipt_path)

    indexed = build_native_indexed_dataset(source, receipt_path, tmp_path / "cache", chunk_size=2)

    assert indexed.source_digest == receipt.digest
    assert indexed.game_ids == tuple(identity.game_id for identity in identities)
    assert indexed.head_game_mask("POLICY") == (True, True, False)
    assert indexed.head_game_mask("VALUE") == (True, False, True)
    assert indexed.head_game_count("POLICY") == 2
    assert indexed.head_game_count("VALUE") == 2
    with pytest.raises(ValueError, match="unique"):
        indexed.head_game_mask("POLICY", (identities[0].game_id, identities[0].game_id))
    with pytest.raises(ValueError, match="unique"):
        indexed.head_game_count("VALUE", (identities[0].game_id, identities[0].game_id))
    mixed = indexed.game(identities[0].game_id)
    assert [chunk.row_count for chunk in mixed.policy_chunks] == [2, 1]
    assert [chunk.row_count for chunk in mixed.value_chunks] == [2]

    policy_batches = tuple(indexed.iter_policy_training_chunks(identities[0].game_id))
    value_batches = tuple(indexed.iter_value_training_chunks(identities[0].game_id))
    assert [batch.sample_indexes for batch in policy_batches] == [(0, 2), (4,)]
    assert [batch.sample_indexes for batch in value_batches] == [(1, 3)]
    assert [batch.row_weights.tolist() for batch in policy_batches] == [
        pytest.approx([1 / 3, 1 / 3]),
        pytest.approx([1 / 3]),
    ]
    assert value_batches[0].row_weights.tolist() == pytest.approx([0.5, 0.5])
    assert tuple(indexed.iter_value_training_chunks(identities[1].game_id)) == ()
    assert tuple(indexed.iter_policy_training_chunks(identities[2].game_id)) == ()
    for game_id in indexed.game_ids:
        assert indexed.validate_game(game_id) == indexed.game(game_id)

    reopened = open_native_indexed_dataset(source, receipt_path, tmp_path / "cache", chunk_size=2)
    assert reopened.manifest == indexed.manifest
    assert reopened.digest == indexed.digest


def test_interrupted_game_is_discarded_and_completed_games_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automata.training import native_indexed_dataset as module

    source = tmp_path / "source"
    names, identities = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)
    cache = tmp_path / "cache"
    real_write = module._write_chunk
    failed = False

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        game_ordinal = kwargs["game_ordinal"]
        if game_ordinal == 1 and not failed:
            failed = True
            raise RuntimeError("simulated interruption")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(module, "_write_chunk", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    assert not cache.exists()
    assert (tmp_path / ".cache.staging" / "checkpoint.json").is_file()

    scanned: list[str] = []
    real_iter = module.iter_native_game_records

    def track(path: str | Path):
        scanned.append(Path(path).name)
        yield from real_iter(path)

    monkeypatch.setattr(module, "_write_chunk", real_write)
    monkeypatch.setattr(module, "iter_native_game_records", track)
    resumed = build_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)

    # The completed game is read once by final source-backed publication validation,
    # while unfinished games are read once to build and once to validate.
    assert scanned.count(Path(names[0]).name) == 1
    assert scanned.count(Path(names[1]).name) == 2
    assert resumed.validate_game(identities[0].game_id).game_id == identities[0].game_id
    assert not (tmp_path / ".cache.staging").exists()


def test_stale_receipt_or_source_and_failed_rebuild_never_clobber_valid_cache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    names, _ = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)
    cache = tmp_path / "cache"
    open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    before = _tree_bytes(cache)

    with (source / names[0]).open("ab") as output:
        output.write(b"corrupt")
    with pytest.raises(ValueError, match=r"source|receipt|hash|size"):
        open_native_indexed_dataset(source, receipt_path, cache, chunk_size=3)

    assert _tree_bytes(cache) == before


def test_failed_rebuild_preserves_valid_published_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automata.training import native_indexed_dataset as module

    source = tmp_path / "source"
    names, identities = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)
    cache = tmp_path / "cache"
    open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    before = _tree_bytes(cache)

    def fail_write(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated chunk write failure")

    monkeypatch.setattr(module, "_write_chunk", fail_write)
    with pytest.raises(RuntimeError, match="simulated chunk write failure"):
        open_native_indexed_dataset(source, receipt_path, cache, chunk_size=3)

    assert _tree_bytes(cache) == before
    reopened = open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    assert reopened.validate_game(identities[0].game_id).game_id == identities[0].game_id


def test_unrelated_cache_paths_are_never_replaced(tmp_path: Path) -> None:
    source = tmp_path / "source"
    names, _ = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)

    unrelated_directory = tmp_path / "application-cache"
    unrelated_directory.mkdir()
    (unrelated_directory / "manifest.json").write_bytes(b'{"application":"data"}')
    (unrelated_directory / _OWNER_MARKER_NAME).write_bytes(b"not the exact native marker\n")
    directory_before = _tree_bytes(unrelated_directory)
    with pytest.raises(ValueError, match=r"owned|empty"):
        build_native_indexed_dataset(source, receipt_path, unrelated_directory, chunk_size=2)
    assert _tree_bytes(unrelated_directory) == directory_before

    unrelated_file = tmp_path / "application-cache-file"
    unrelated_file.write_bytes(b"important application bytes")
    file_before = unrelated_file.read_bytes()
    with pytest.raises(ValueError, match=r"owned|empty"):
        build_native_indexed_dataset(source, receipt_path, unrelated_file, chunk_size=2)
    assert unrelated_file.read_bytes() == file_before


def test_unrelated_staging_paths_are_never_cleared(tmp_path: Path) -> None:
    source = tmp_path / "source"
    names, _ = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)

    directory_cache = tmp_path / "directory-cache"
    directory_stage = tmp_path / ".directory-cache.staging"
    directory_stage.mkdir()
    (directory_stage / "notes.txt").write_bytes(b"keep this directory")
    (directory_stage / _OWNER_MARKER_NAME).write_bytes(b"not the exact native marker\n")
    directory_before = _tree_bytes(directory_stage)
    with pytest.raises(ValueError, match=r"owned|empty"):
        build_native_indexed_dataset(source, receipt_path, directory_cache, chunk_size=2)
    assert _tree_bytes(directory_stage) == directory_before
    assert not directory_cache.exists()

    file_cache = tmp_path / "file-cache"
    file_stage = tmp_path / ".file-cache.staging"
    file_stage.write_bytes(b"keep this file")
    file_before = file_stage.read_bytes()
    with pytest.raises(ValueError, match=r"owned|empty"):
        build_native_indexed_dataset(source, receipt_path, file_cache, chunk_size=2)
    assert file_stage.read_bytes() == file_before
    assert not file_cache.exists()


def test_empty_cache_and_staging_directories_are_safe_to_use(tmp_path: Path) -> None:
    source = tmp_path / "source"
    names, identities = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)
    cache = tmp_path / "cache"
    stage = tmp_path / ".cache.staging"
    cache.mkdir()
    stage.mkdir()

    indexed = build_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)

    assert indexed.validate_game(identities[0].game_id).game_id == identities[0].game_id
    assert (cache / _OWNER_MARKER_NAME).read_bytes() == _OWNER_MARKER_CONTENT


def test_readonly_open_rejects_missing_or_corrupt_cache_without_filesystem_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    names, identities = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)
    missing = tmp_path / "missing-cache"
    before_missing = _tree_bytes(tmp_path)

    with pytest.raises(ValueError, match=r"cache|compatible|rebuild"):
        open_native_indexed_dataset(
            source,
            receipt_path,
            missing,
            chunk_size=2,
            rebuild=False,
        )

    assert _tree_bytes(tmp_path) == before_missing
    cache = tmp_path / "cache"
    indexed = open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    chunk = indexed.game(identities[0].game_id).policy_chunks[0]
    (cache / chunk.path).write_bytes(b"corrupt")
    before_corrupt = _tree_bytes(cache)

    with pytest.raises(ValueError, match=r"cache|compatible|rebuild"):
        open_native_indexed_dataset(
            source,
            receipt_path,
            cache,
            chunk_size=2,
            rebuild=False,
        )

    assert _tree_bytes(cache) == before_corrupt


@pytest.mark.parametrize("rebuild", [0, 1, None, "false"])
def test_open_requires_strict_boolean_rebuild_before_filesystem_effects(
    tmp_path: Path, rebuild: object
) -> None:
    source = tmp_path / "source"
    names, _ = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)
    before = _tree_bytes(tmp_path)

    with pytest.raises((TypeError, ValueError), match="rebuild"):
        open_native_indexed_dataset(
            source,
            receipt_path,
            tmp_path / "cache",
            rebuild=rebuild,  # type: ignore[arg-type]
        )

    assert _tree_bytes(tmp_path) == before


def test_malformed_marked_native_cache_is_safely_rebuilt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    names, identities = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)
    cache = tmp_path / "cache"
    open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    assert (cache / _OWNER_MARKER_NAME).read_bytes() == _OWNER_MARKER_CONTENT
    (cache / "manifest.json").write_bytes(b"{malformed native manifest")

    rebuilt = open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)

    assert rebuilt.validate_game(identities[0].game_id).game_id == identities[0].game_id
    assert (cache / _OWNER_MARKER_NAME).read_bytes() == _OWNER_MARKER_CONTENT


def test_corrupt_chunk_or_manifest_is_not_reused_and_is_safely_rebuilt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    names, identities = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)
    cache = tmp_path / "cache"
    first = open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    chunk = first.game(identities[0].game_id).policy_chunks[0]
    (cache / chunk.path).write_bytes(b"corrupt")

    rebuilt = open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    assert rebuilt.validate_game(identities[0].game_id).game_id == identities[0].game_id

    # Even self-consistent exact hashes cannot make trailing compressed data reusable.
    chunk = rebuilt.game(identities[0].game_id).policy_chunks[0]
    chunk_path = cache / chunk.path
    chunk_path.write_bytes(chunk_path.read_bytes() + b"trailing")
    payload = rebuilt.manifest.model_dump(mode="json")
    raw_chunk = payload["games"][0]["policy_chunks"][0]
    raw_chunk["sha256"] = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
    raw_chunk["file_size"] = chunk_path.stat().st_size
    (cache / "manifest.json").write_bytes(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    )
    rebuilt = open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    assert (cache / rebuilt.game(identities[0].game_id).policy_chunks[0].path).read_bytes()[
        -8:
    ] != b"trailing"

    payload = rebuilt.manifest.model_dump(mode="json")
    payload["games"][0]["policy_chunks"][0]["path"] = "../escape.jsonl.zst"
    (cache / "manifest.json").write_bytes(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    )
    again = open_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)
    assert again.validate_game(identities[0].game_id).game_id == identities[0].game_id


def test_internally_consistent_wrong_receipt_counts_reach_source_mismatch_without_publish(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    names, _ = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    receipt = create_native_source_receipt(source, names)
    first = receipt.games[0]
    bad_game = NativeGameSourceReceipt.model_validate(
        {
            **first.model_dump(mode="python"),
            "row_count": first.row_count + 1,
            "policy_row_count": first.policy_row_count + 1,
        },
        strict=True,
    )
    bad_receipt = NativeDatasetSourceReceipt(games=(bad_game, *receipt.games[1:]))
    receipt_path.write_bytes(bad_receipt.canonical_bytes())

    with pytest.raises(ValueError, match=r"identity|count"):
        build_native_indexed_dataset(source, receipt_path, tmp_path / "cache", chunk_size=2)
    assert not (tmp_path / "cache").exists()


def test_duplicate_receipt_game_ids_are_rejected() -> None:
    receipt_game = NativeGameSourceReceipt(
        logical_name="one.jsonl",
        file_sha256="0" * 64,
        file_size=1,
        game_id="1" * 64,
        row_count=1,
        policy_row_count=1,
        value_row_count=0,
        boundary_count=0,
    )
    duplicate_id = receipt_game.model_copy(update={"logical_name": "duplicate.jsonl"})
    with pytest.raises(ValidationError, match="game IDs"):
        NativeDatasetSourceReceipt(games=(receipt_game, duplicate_id))


def test_invalid_completed_checkpoint_counts_are_discarded_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automata.training import native_indexed_dataset as module

    source = tmp_path / "source"
    names, identities = _publish_sources(source)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(source, names, receipt_path)
    cache = tmp_path / "cache"
    real_write = module._write_chunk

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        if kwargs["game_ordinal"] == 1:
            raise RuntimeError("leave one completed game")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(module, "_write_chunk", interrupt)
    with pytest.raises(RuntimeError, match="leave one completed game"):
        build_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)

    stage = tmp_path / ".cache.staging"
    envelope = json.loads((stage / "checkpoint.json").read_bytes())
    game = envelope["checkpoint"]["completed_games"][0]
    game["row_count"] = 4
    game["policy_row_count"] = 2
    game["policy_chunks"] = game["policy_chunks"][:1]
    semantic = hashlib.sha256()
    for record in tuple(module.iter_native_game_records(source / names[0]))[:4]:
        semantic.update(module.canonical_json_bytes(record) + b"\n")
    game["semantic_digest"] = semantic.hexdigest()

    def canonical_mapping(value: dict[str, Any]) -> bytes:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()

    checkpoint_bytes = canonical_mapping(envelope["checkpoint"])
    envelope["checkpoint_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
    (stage / "checkpoint.json").write_bytes(canonical_mapping(envelope))

    monkeypatch.setattr(module, "_write_chunk", real_write)
    rebuilt = build_native_indexed_dataset(source, receipt_path, cache, chunk_size=2)

    assert rebuilt.game(identities[0].game_id).policy_row_count == 3
    assert rebuilt.validate_game(identities[0].game_id).game_id == identities[0].game_id
