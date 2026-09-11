"""Disk-backed indexing contract for strict joint datasets."""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
import torch
import zstandard

from automata.models.contracts import (
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    Viewer,
    canonical_json_bytes,
)
from automata.models.shared_encoder.batching import DecisionBatch
from automata.models.shared_encoder.schema import TensorFeatureSchema
from automata.training import indexed_dataset as indexed_dataset_module
from automata.training.dataset import (
    JointDatasetRow,
    joint_decision_id,
    load_joint_dataset,
    write_joint_dataset,
)
from automata.training.indexed_dataset import (
    IndexedDatasetManifest,
    IndexedTrainingChunk,
    open_indexed_dataset,
)


def _observation() -> DecisionObservation:
    candidates = tuple(
        EncodedCandidate(
            schema_version=1,
            candidate_id=OptionCandidateID(schema_version=1, option_id=name),
            selection=name,
        )
        for name in ("hold", "advance")
    )
    return DecisionObservation(
        schema_version=3,
        state=LearnedObservation(
            schema_version=2,
            viewer=Viewer(schema_version=2, perspective_team="RED"),
            tokens=(
                ObservationToken(
                    schema_version=1,
                    local_ref="global:0",
                    kind="GLOBAL",
                    features={"map_id": "forgotten_island", "game_type": "QUICK"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="team:red",
                    kind="TEAM",
                    features={"relation": "OWN"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="team:blue",
                    kind="TEAM",
                    features={"relation": "ENEMY"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:red:0",
                    kind="HERO",
                    features={"name": "Wasp", "team_id": "RED", "team_ref": "team:red"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:blue:0",
                    kind="HERO",
                    features={"name": "Arien", "team_id": "BLUE", "team_ref": "team:blue"},
                ),
            ),
        ),
        decision_kind="INPUT",
        candidates=candidates,
    )


def _row(
    game: int, decision_index: int, *, policy: tuple[float, float] = (1.0, 0.0)
) -> JointDatasetRow:
    observation = _observation()
    identity: dict[str, Any] = {
        "game_id": f"game-{game}",
        "world_seed": game,
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
    selected = 0 if policy[0] else 1
    return JointDatasetRow(
        schema_version=1,
        decision_id=joint_decision_id(**identity, decision_index=decision_index),
        **identity,
        decision_index=decision_index,
        perspective_team="RED",
        observation=observation,
        policy_source="HEURISTIC",
        policy_target=policy,
        selected_candidate_id=observation.candidates[selected].candidate_id,
        selected_selection=observation.candidates[selected].selection,
        action_stats=None,
        terminal_winner="RED",
        value_target=1,
    )


def test_builds_canonical_atomic_index_and_reads_one_game_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint.jsonl.zst"
    cache = tmp_path / "index"
    rows = (_row(7, 0), _row(7, 1), _row(8, 0))
    write_joint_dataset(source, rows)
    opened_for_write: list[Path] = []
    path_open = Path.open

    def track_writes(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in "wax+"):
            opened_for_write.append(path)
        return path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_writes)
    indexed = open_indexed_dataset(source, cache)

    assert opened_for_write
    assert not any(
        path.name.endswith(".jsonl") or ".spools" in path.parts for path in opened_for_write
    )
    manifest_payload = (cache / "manifest.json").read_bytes()
    restored = IndexedDatasetManifest.model_validate_json(manifest_payload)
    assert manifest_payload == canonical_json_bytes(restored)
    assert indexed.digest == load_joint_dataset(source).digest
    assert indexed.manifest.row_count == 3
    assert indexed.manifest.game_count == 2
    assert indexed.game_ids == ("game-7", "game-8")
    first = indexed.game("game-7")
    assert first.global_row_start == 0
    assert first.row_count == 2
    assert first.map_id == "forgotten_island"
    assert first.red_composition == ("Wasp",)
    assert first.generation_id == "generation-1"
    assert (cache / first.fragment).read_bytes().startswith(b"\x28\xb5\x2f\xfd")
    assert list(indexed.iter_game_chunks("game-7", chunk_size=1)) == [
        (rows[0],),
        (rows[1],),
    ]
    assert indexed.validate_game("game-7", chunk_size=1) == first
    assert not (tmp_path / ".index.staging").exists()


def test_index_preserves_validated_source_json_without_canonical_reserialization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "joint.jsonl"
    row = _row(7, 0)
    raw_line = json.dumps(row.model_dump(mode="json"), ensure_ascii=False).encode()
    assert raw_line != canonical_json_bytes(row)
    source.write_bytes(raw_line + b"\n")

    indexed = open_indexed_dataset(source, tmp_path / "index")
    fragment = tmp_path / "index" / indexed.game("game-7").fragment
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(fragment.read_bytes())) as reader:
        fragment_bytes = reader.read()

    assert fragment_bytes == raw_line + b"\n"
    assert indexed.digest == hashlib.sha256(raw_line + b"\n").hexdigest()


def _assert_batch_equal(actual: DecisionBatch, expected: DecisionBatch) -> None:
    assert actual.candidate_ids == expected.candidate_ids
    assert actual.token_kinds == expected.token_kinds
    assert actual.candidate_kinds == expected.candidate_kinds
    assert actual.tokens.keys() == expected.tokens.keys()
    assert actual.relationships.keys() == expected.relationships.keys()
    for actual_table, expected_table in [
        *((actual.tokens[kind], expected.tokens[kind]) for kind in actual.tokens),
        *(
            (actual.relationships[kind], expected.relationships[kind])
            for kind in actual.relationships
        ),
        (actual.candidates, expected.candidates),
    ]:
        assert actual_table.__dict__.keys() == expected_table.__dict__.keys()
        for field in actual_table.__dict__:
            assert torch.equal(getattr(actual_table, field), getattr(expected_table, field))


def test_tensorized_training_chunks_roundtrip_with_targets_offsets_and_metrics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    rows = (
        _row(7, 0),
        _row(7, 1, policy=(1.0 / 3.0, 2.0 / 3.0)),
        _row(7, 2, policy=(0.0, 1.0)),
    )
    write_joint_dataset(source, rows)

    indexed = open_indexed_dataset(source, cache, training_chunk_size=2)
    chunks = tuple(indexed.iter_game_training_chunks("game-7"))

    assert all(isinstance(chunk, IndexedTrainingChunk) for chunk in chunks)
    assert [chunk.row_count for chunk in chunks] == [2, 1]
    assert chunks[0].global_row_offsets == (0, 1)
    assert chunks[1].global_row_offsets == (2,)
    assert chunks[0].game_ids == ("game-7", "game-7")
    assert torch.equal(
        chunks[0].policy_targets,
        torch.tensor([[1.0, 0.0], [1.0 / 3.0, 2.0 / 3.0]]),
    )
    assert chunks[0].metric_metadata[1].target_probabilities == (
        1.0 / 3.0,
        2.0 / 3.0,
    )
    assert torch.equal(chunks[1].policy_targets, torch.tensor([[0.0, 1.0]]))
    assert torch.equal(chunks[0].value_targets, torch.tensor([1.0, 1.0]))
    assert chunks[0].metric_metadata[0].candidate_count == 2
    assert chunks[0].metric_metadata[0].candidate_family == "OPTION"
    assert chunks[0].metric_metadata[0].hero == "Wasp"
    assert chunks[0].metric_metadata[0].map_id == "forgotten_island"
    assert chunks[0].metric_metadata[0].composition == "Wasp vs Arien"
    assert chunks[0].metric_metadata[0].round_bucket is None

    schema = TensorFeatureSchema.current()
    for chunk, expected_rows in zip(chunks, (rows[:2], rows[2:]), strict=True):
        expected = indexed_dataset_module.collate_decisions(
            [row.observation for row in expected_rows], schema=schema, training=True
        )
        _assert_batch_equal(chunk.batch, expected)


def test_training_chunk_tampering_is_rejected_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0),))
    indexed = open_indexed_dataset(source, cache)
    chunk_path = cache / indexed.game("game-7").training_chunks[0].path
    chunk_path.write_bytes(chunk_path.read_bytes() + b"tampered")

    def unsafe_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("torch.load must not run before SHA-256 validation")

    monkeypatch.setattr(torch, "load", unsafe_load)
    with pytest.raises(ValueError, match="training chunk digest"):
        tuple(indexed.iter_game_training_chunks("game-7"))


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("format", "format version"),
        ("value_dtype", "value targets is not a compatible CPU tensor"),
        ("offset", "global row offsets"),
        ("metric_target", "metric targets disagree"),
        ("non_prefix_mask", "candidate masks must be contiguous prefixes"),
    ],
)
def test_digest_valid_but_invalid_training_payload_is_rejected(
    tmp_path: Path, corruption: str, message: str
) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0),))
    indexed = open_indexed_dataset(source, cache)
    chunk_metadata = indexed.game("game-7").training_chunks[0]
    chunk_path = cache / chunk_metadata.path
    raw = zstandard.ZstdDecompressor().decompress(
        chunk_path.read_bytes(), max_output_size=chunk_metadata.uncompressed_size
    )
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if corruption == "format":
        payload["format_version"] = 999
    elif corruption == "value_dtype":
        payload["value_targets"] = payload["value_targets"].to(torch.int64)
    elif corruption == "offset":
        payload["global_row_offsets"] = [999]
    elif corruption == "metric_target":
        payload["metric_metadata"][0]["target_probabilities"] = [0.5, 0.5]
    else:
        payload["batch"]["candidates"]["mask"][0] = torch.tensor([False, True])
        payload["batch"]["candidate_ids"][0] = payload["batch"]["candidate_ids"][0][:1]

    serialized = io.BytesIO()
    torch.save(payload, serialized)
    compressed = zstandard.ZstdCompressor(level=3, write_checksum=True).compress(
        serialized.getvalue()
    )
    chunk_path.write_bytes(compressed)
    manifest_payload = indexed.manifest.model_dump(mode="json")
    raw_chunk_metadata = manifest_payload["games"][0]["training_chunks"][0]
    raw_chunk_metadata["sha256"] = hashlib.sha256(compressed).hexdigest()
    raw_chunk_metadata["uncompressed_size"] = len(serialized.getvalue())
    manifest = IndexedDatasetManifest.model_validate_json(json.dumps(manifest_payload))
    (cache / "manifest.json").write_bytes(manifest.canonical_bytes())
    reopened = open_indexed_dataset(source, cache)

    with pytest.raises(ValueError, match=message):
        tuple(reopened.iter_game_training_chunks("game-7"))


def test_tensor_cache_manifest_is_bound_to_schema_and_chunk_size(tmp_path: Path) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0), _row(7, 1), _row(7, 2)))

    first = open_indexed_dataset(source, cache, training_chunk_size=2)
    first_chunk = cache / first.game("game-7").training_chunks[0].path
    first_inode = first_chunk.stat().st_ino
    wrong_schema = first.manifest.model_dump(mode="json")
    wrong_schema["tensor_schema_digest"] = "0" * 64
    wrong_manifest = IndexedDatasetManifest.model_validate_json(json.dumps(wrong_schema))
    (cache / "manifest.json").write_bytes(wrong_manifest.canonical_bytes())

    schema_rebuilt = open_indexed_dataset(source, cache, training_chunk_size=2)
    schema_rebuilt_chunk = cache / schema_rebuilt.game("game-7").training_chunks[0].path
    schema_rebuilt_inode = schema_rebuilt_chunk.stat().st_ino
    second = open_indexed_dataset(source, cache, training_chunk_size=1)

    assert first.manifest.tensor_schema_digest == TensorFeatureSchema.current().digest
    assert schema_rebuilt.manifest.tensor_schema_digest == TensorFeatureSchema.current().digest
    assert schema_rebuilt_inode != first_inode
    assert first.manifest.training_chunk_size == 2
    assert second.manifest.training_chunk_size == 1
    assert len(second.game("game-7").training_chunks) == 3
    assert (cache / second.game("game-7").training_chunks[0].path).stat().st_ino != first_inode


def test_index_build_never_accumulates_more_than_training_chunk_size_source_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, tuple(_row(7, index) for index in range(7)))

    indexed = open_indexed_dataset(source, cache, training_chunk_size=3)

    assert [item.row_count for item in indexed.game("game-7").training_chunks] == [3, 3, 1]


def test_parallel_tensorization_is_manifest_and_data_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "joint.jsonl.zst"
    sequential_cache = tmp_path / "sequential"
    parallel_cache = tmp_path / "parallel"
    write_joint_dataset(
        source,
        (
            _row(7, 0),
            _row(7, 1, policy=(0.25, 0.75)),
            _row(8, 0, policy=(0.0, 1.0)),
            _row(8, 1),
        ),
    )

    sequential = open_indexed_dataset(
        source, sequential_cache, training_chunk_size=1, index_workers=1
    )
    parallel = open_indexed_dataset(source, parallel_cache, training_chunk_size=1, index_workers=2)

    assert sequential.manifest.canonical_bytes() == parallel.manifest.canonical_bytes()
    sequential_chunk_path = sequential_cache / sequential.game("game-7").training_chunks[0].path
    sequential_chunk_inode = sequential_chunk_path.stat().st_ino
    reused_with_more_workers = open_indexed_dataset(
        source, sequential_cache, training_chunk_size=1, index_workers=2
    )
    assert reused_with_more_workers.manifest == sequential.manifest
    assert sequential_chunk_path.stat().st_ino == sequential_chunk_inode
    for game_id in sequential.game_ids:
        sequential_chunks = tuple(sequential.iter_game_training_chunks(game_id))
        parallel_chunks = tuple(parallel.iter_game_training_chunks(game_id))
        assert len(sequential_chunks) == len(parallel_chunks)
        for actual, expected in zip(parallel_chunks, sequential_chunks, strict=True):
            _assert_batch_equal(actual.batch, expected.batch)
            assert torch.equal(actual.policy_targets, expected.policy_targets)
            assert torch.equal(actual.value_targets, expected.value_targets)
            assert actual.global_row_offsets == expected.global_row_offsets
            assert actual.game_ids == expected.game_ids
            assert actual.metric_metadata == expected.metric_metadata


def test_index_workers_must_be_a_positive_integer(tmp_path: Path) -> None:
    source = tmp_path / "joint.jsonl"
    write_joint_dataset(source, (_row(7, 0),))

    for value in (True, 0, -1):
        with pytest.raises(ValueError, match="index_workers must be a positive integer"):
            open_indexed_dataset(source, tmp_path / f"index-{value}", index_workers=value)


class _InlineExecutor:
    def __init__(self, **_: Any) -> None:
        pass

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def submit(self, function: Any, *args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_failed_tensorization_resumes_without_rescan_or_redoing_completed_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    clean_cache = tmp_path / "clean"
    write_joint_dataset(source, (_row(7, 0), _row(8, 0)))
    scan_calls = 0
    tensor_calls: list[str] = []
    progress_calls: list[dict[str, Any]] = []
    failed = False
    real_scan = indexed_dataset_module.iter_joint_dataset_records
    real_tensorize = indexed_dataset_module._tensorize_game_fragment
    real_tqdm = indexed_dataset_module.tqdm

    def scan(*args: Any, **kwargs: Any) -> Any:
        nonlocal scan_calls
        scan_calls += 1
        return real_scan(*args, **kwargs)

    def tensorize(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        game_id = str(args[2])
        tensor_calls.append(game_id)
        if game_id == "game-8" and not failed:
            failed = True
            raise RuntimeError("forced tensor worker failure")
        return real_tensorize(*args, **kwargs)

    def track_progress(*args: Any, **kwargs: Any) -> Any:
        progress_calls.append(kwargs)
        return real_tqdm(*args, **kwargs)

    monkeypatch.setattr(indexed_dataset_module, "iter_joint_dataset_records", scan)
    monkeypatch.setattr(indexed_dataset_module, "tqdm", track_progress)
    monkeypatch.setattr(indexed_dataset_module, "_tensorize_game_fragment", tensorize)
    monkeypatch.setattr(indexed_dataset_module, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(indexed_dataset_module, "as_completed", lambda futures: futures)

    with pytest.raises(RuntimeError, match="forced tensor worker failure"):
        open_indexed_dataset(source, cache)

    stage = tmp_path / ".index.staging"
    assert stage.is_dir()
    orphan = stage / "index" / "training" / "00000001" / "orphan.pt.zst"
    orphan.parent.mkdir(exist_ok=True)
    orphan.write_bytes(b"must be removed before retry")
    resumed = open_indexed_dataset(source, cache)

    assert scan_calls == 1
    assert not (cache / "training" / "00000001" / "orphan.pt.zst").exists()
    assert tensor_calls == ["game-7", "game-8", "game-8"]
    resumed_progress = [call for call in progress_calls if call["desc"] == "Tensorizing index"][-1]
    assert resumed_progress["initial"] == 1
    assert resumed_progress["bar_format"] == indexed_dataset_module.TQDM_BAR_FORMAT

    clean = open_indexed_dataset(source, clean_cache)
    assert scan_calls == 2
    assert resumed.manifest.canonical_bytes() == clean.manifest.canonical_bytes()
    assert not (cache / "checkpoint.json").exists()
    assert {
        path.relative_to(cache): path.read_bytes() for path in cache.rglob("*") if path.is_file()
    } == {
        path.relative_to(clean_cache): path.read_bytes()
        for path in clean_cache.rglob("*")
        if path.is_file()
    }
    assert not stage.exists()


def test_concurrent_builder_cannot_enter_or_destroy_shared_staging(tmp_path: Path) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0),))

    with (
        indexed_dataset_module._exclusive_build_lock(cache),
        pytest.raises(RuntimeError, match="another index build is already running"),
        indexed_dataset_module._exclusive_build_lock(cache, blocking=False),
    ):
        pass

    indexed = open_indexed_dataset(source, cache)
    assert indexed.game_ids == ("game-7",)


def test_builder_rechecks_published_index_after_acquiring_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0),))
    published = open_indexed_dataset(source, cache)
    calls = 0
    real_open = indexed_dataset_module._open_compatible_published_index

    def simulate_publish_while_waiting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return None if calls == 1 else real_open(*args, **kwargs)

    def forbid_rebuild(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("compatible index must be reused after lock acquisition")

    monkeypatch.setattr(
        indexed_dataset_module,
        "_open_compatible_published_index",
        simulate_publish_while_waiting,
    )
    monkeypatch.setattr(indexed_dataset_module, "_build_indexed_dataset_locked", forbid_rebuild)

    reopened = open_indexed_dataset(source, cache)

    assert calls == 2
    assert reopened.manifest == published.manifest


def test_symlinked_staging_index_root_is_rejected_without_following_target(
    tmp_path: Path,
) -> None:
    stage = tmp_path / ".index.staging"
    target = tmp_path / "published-index"
    stage.mkdir()
    target.mkdir()
    marker = target / "must-survive"
    marker.write_text("safe")
    (stage / "index").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="staging index roots must be regular directories"):
        indexed_dataset_module._load_staging_checkpoint(
            stage,
            source_sha256="0" * 64,
            source_size=0,
            tensor_schema_digest="0" * 64,
            training_chunk_size=32,
        )

    assert marker.read_text() == "safe"


def test_symlinked_destination_fails_without_rebuilding_or_removing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "joint.jsonl"
    target = tmp_path / "target-index"
    destination = tmp_path / "index-link"
    write_joint_dataset(source, (_row(7, 0),))
    built = open_indexed_dataset(source, target)
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="destination must not be a symlink"):
        open_indexed_dataset(source, destination)

    assert destination.is_symlink()
    assert (target / "manifest.json").read_bytes() == built.manifest.canonical_bytes()


def test_cold_open_hashes_source_only_at_required_integrity_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0),))
    source_hashes = 0
    real_hash = indexed_dataset_module._file_sha256

    def count_hashes(path: Path) -> tuple[str, int]:
        nonlocal source_hashes
        if path == source:
            source_hashes += 1
        return real_hash(path)

    monkeypatch.setattr(indexed_dataset_module, "_file_sha256", count_hashes)

    open_indexed_dataset(source, cache)

    assert source_hashes == 3  # initial identity, post-scan, and pre-publication


def test_publish_refuses_a_symlinked_index_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    destination = tmp_path / "index"
    target.mkdir()
    marker = target / "must-survive"
    marker.write_text("safe")
    temporary = tmp_path / "stage-index"
    temporary.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="source must be a regular directory"):
        indexed_dataset_module._publish_directory(temporary, destination)

    assert marker.read_text() == "safe"
    assert not destination.exists()


@pytest.mark.parametrize("corruption", ["chunk", "checkpoint_metadata"])
def test_stale_tensorization_stage_is_discarded_and_source_is_rescanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0), _row(8, 0)))
    scan_calls = 0
    failed = False
    real_scan = indexed_dataset_module.iter_joint_dataset_records
    real_tensorize = indexed_dataset_module._tensorize_game_fragment

    def scan(*args: Any, **kwargs: Any) -> Any:
        nonlocal scan_calls
        scan_calls += 1
        return real_scan(*args, **kwargs)

    def tensorize(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        if args[2] == "game-8" and not failed:
            failed = True
            raise RuntimeError("forced tensor worker failure")
        return real_tensorize(*args, **kwargs)

    monkeypatch.setattr(indexed_dataset_module, "iter_joint_dataset_records", scan)
    monkeypatch.setattr(indexed_dataset_module, "_tensorize_game_fragment", tensorize)
    monkeypatch.setattr(indexed_dataset_module, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(indexed_dataset_module, "as_completed", lambda futures: futures)

    with pytest.raises(RuntimeError, match="forced tensor worker failure"):
        open_indexed_dataset(source, cache)
    stage = tmp_path / ".index.staging"
    if corruption == "chunk":
        completed_chunk = next((stage / "index" / "training").rglob("*.pt.zst"))
        completed_chunk.write_bytes(completed_chunk.read_bytes() + b"stale")
    else:
        checkpoint_path = stage / "checkpoint.json"
        payload = json.loads(checkpoint_path.read_bytes())
        payload["checkpoint"]["dataset_digest"] = "0" * 64
        checkpoint_path.write_bytes(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )

    rebuilt = open_indexed_dataset(source, cache)

    assert scan_calls == 2
    assert rebuilt.game_ids == ("game-7", "game-8")
    assert not (tmp_path / ".index.staging").exists()


def test_tensor_worker_failure_never_publishes_or_replaces_cache(tmp_path: Path) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    original = _row(7, 0)
    write_joint_dataset(source, (original,))
    indexed = open_indexed_dataset(source, cache)
    old_manifest = (cache / "manifest.json").read_bytes()

    tokens = tuple(
        (
            token.model_copy(
                update={
                    "features": {
                        key: value for key, value in token.features.items() if key != "team_ref"
                    }
                }
            )
            if token.kind == "HERO"
            else token
        )
        for token in original.observation.state.tokens
    )
    invalid_observation = original.observation.model_copy(
        update={"state": original.observation.state.model_copy(update={"tokens": tokens})}
    )
    write_joint_dataset(source, (original.model_copy(update={"observation": invalid_observation}),))

    with pytest.raises(ValueError, match="required reference 'team_ref' is missing"):
        open_indexed_dataset(source, cache, index_workers=2)

    assert (cache / "manifest.json").read_bytes() == old_manifest
    assert tuple(indexed.iter_game_rows("game-7")) == (original,)
    assert (tmp_path / ".index.staging" / "checkpoint.json").is_file()


def test_reuses_only_an_index_bound_to_the_exact_source_file_bytes(tmp_path: Path) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    first_rows = (_row(7, 0),)
    write_joint_dataset(source, first_rows)
    first = open_indexed_dataset(source, cache)
    fragment = cache / first.game("game-7").fragment
    original_fragment_stat = fragment.stat()

    reused = open_indexed_dataset(source, cache)

    assert reused.manifest == first.manifest
    assert fragment.stat().st_ino == original_fragment_stat.st_ino

    write_joint_dataset(source, (*first_rows, _row(8, 0)))
    rebuilt = open_indexed_dataset(source, cache)

    assert rebuilt.manifest.source_sha256 != first.manifest.source_sha256
    assert rebuilt.game_ids == ("game-7", "game-8")
    assert rebuilt.manifest.row_count == 2


def test_game_validation_detects_semantically_valid_fragment_tampering(tmp_path: Path) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0),))
    indexed = open_indexed_dataset(source, cache)
    metadata = indexed.game("game-7")

    write_joint_dataset(cache / metadata.fragment, (_row(7, 0, policy=(0.0, 1.0)),))

    with pytest.raises(ValueError, match="fragment digest"):
        indexed.validate_game("game-7", chunk_size=1)


def test_noncontiguous_games_are_rejected_without_publishing_a_cache(tmp_path: Path) -> None:
    source = tmp_path / "interleaved.jsonl.zst"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0), _row(8, 0), _row(7, 1)))

    with pytest.raises(ValueError, match="non-contiguously"):
        open_indexed_dataset(source, cache)

    assert not cache.exists()
    assert not (tmp_path / ".index.staging").exists()


def test_invalid_rebuild_never_replaces_a_complete_cache(tmp_path: Path) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0),))
    indexed = open_indexed_dataset(source, cache)
    old_manifest = (cache / "manifest.json").read_bytes()

    source.write_bytes(canonical_json_bytes(_row(8, 1)) + b"\n")

    with pytest.raises(ValueError, match="decision indexes"):
        open_indexed_dataset(source, cache)
    assert (cache / "manifest.json").read_bytes() == old_manifest
    assert tuple(indexed.iter_game_rows("game-7")) == (_row(7, 0),)


def test_noncanonical_or_internally_inconsistent_manifest_is_not_reused(tmp_path: Path) -> None:
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(7, 0),))
    indexed = open_indexed_dataset(source, cache)
    payload = indexed.manifest.model_dump(mode="json")
    payload["row_count"] = 99
    (cache / "manifest.json").write_text(json.dumps(payload, indent=2))

    rebuilt = open_indexed_dataset(source, cache)

    assert rebuilt.manifest.row_count == 1
    assert (cache / "manifest.json").read_bytes() == canonical_json_bytes(rebuilt.manifest)
