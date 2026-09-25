"""Production reconciliation contract for completed persistent self-play workers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from automata.decision import DecisionSemanticRole
from automata.models.contracts import (
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    Viewer,
    canonical_json_bytes,
)
from automata.scripts import reconcile_self_play
from automata.training.dataset import (
    JointDatasetRow,
    joint_decision_id,
    load_joint_dataset,
    write_joint_dataset,
)
from automata.training.generation import (
    CheckpointRow,
    GameSpec,
    GenerationConfig,
    build_worker_specs,
)
from automata.training.io import canonical_json_bytes as canonical_value_bytes


def _row(spec: Any, game: GameSpec, decision_index: int = 0) -> JointDatasetRow:
    candidate_id = OptionCandidateID(schema_version=1, option_id="advance")
    observation = DecisionObservation(
        schema_version=4,
        state=LearnedObservation(
            schema_version=2,
            viewer=Viewer(schema_version=2, perspective_team="RED"),
            tokens=(
                ObservationToken(
                    schema_version=1,
                    local_ref="global:0",
                    kind="GLOBAL",
                    features={"map_id": game.map_id, "game_type": game.game_type},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:red",
                    kind="HERO",
                    features={"name": "Wasp", "team_id": "RED"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:blue",
                    kind="HERO",
                    features={"name": "Arien", "team_id": "BLUE"},
                ),
            ),
        ),
        decision_kind="INPUT",
        input_request_type="SELECT_OPTION",
        can_skip=False,
        semantic_role=DecisionSemanticRole.OPTION_SELECTION,
        candidates=(
            EncodedCandidate(schema_version=1, candidate_id=candidate_id, selection="advance"),
        ),
    )
    identity = {
        "game_id": game.game_id(spec.config),
        "world_seed": game.world_seed,
        "map_id": game.map_id,
        "game_type": game.game_type,
        "red_composition": game.red_composition,
        "blue_composition": game.blue_composition,
        "generation_id": spec.config.generation_id,
        "source_revision": spec.config.source_revision,
        "dirty_tree_hash": spec.config.dirty_tree_hash,
        "source_model_digest": spec.config.parent_model_digest,
        "search_config_id": spec.config.search_config_id,
        "generator_config_id": spec.config.generator_config_id,
    }
    return JointDatasetRow(
        schema_version=2,
        decision_id=joint_decision_id(**identity, decision_index=decision_index),
        **identity,
        decision_index=decision_index,
        perspective_team="RED",
        observation=observation,
        policy_source="ISMCTS_VISITS",
        policy_target=(1.0,),
        selected_candidate_id=candidate_id,
        selected_selection="advance",
        terminal_winner="RED",
        value_target=1,
    )


def _completed_workers(
    root: Path, *, seed_start: int = 20_000, seed_end: int = 20_004, workers: int = 2
) -> tuple[Path, Path, tuple[Any, ...]]:
    output_dir = root / "output"
    checkpoint_dir = root / "checkpoints"
    config = GenerationConfig(
        generation_id="official-self-play",
        parent_model_digest="a" * 64,
        parent_generation=1,
        observation_schema_version=4,
        source_revision="revision",
        dirty_tree_hash="tree",
        search_config={"iterations": 8},
        source_config={"recipe": "official", "worker_count": workers},
        max_steps=10_000,
        timeout_seconds=3_600,
    )
    games = tuple(
        GameSpec(seed, "forgotten_island", "map.json", "QUICK", ("Wasp",), ("Arien",))
        for seed in range(seed_start, seed_end)
    )
    specs = build_worker_specs(config, games, worker_count=workers)
    for spec in specs:
        worker_dir = output_dir / f"worker-{spec.worker_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        receipts: list[CheckpointRow] = []
        for game in spec.games:
            fragment = worker_dir / (f"{game.world_seed:020d}-{game.game_id(config)}.jsonl.zst")
            rows = (_row(spec, game, 0), _row(spec, game, 1))
            write_joint_dataset(fragment, rows)
            receipts.append(
                CheckpointRow(
                    worker_id=spec.worker_id,
                    worker_config_id=spec.worker_config_id,
                    generator_config_id=config.generator_config_id,
                    source_config_id=config.source_config_id,
                    generation_id=config.generation_id,
                    parent_model_digest=config.parent_model_digest,
                    parent_generation=config.parent_generation,
                    observation_schema_version=config.observation_schema_version,
                    game_id=game.game_id(config),
                    world_seed=game.world_seed,
                    fragment_digest=hashlib.sha256(fragment.read_bytes()).hexdigest(),
                    row_count=len(rows),
                    winner="RED",
                    rounds=2,
                    turns=3,
                    steps=4,
                )
            )
        (checkpoint_dir / f"worker-{spec.worker_id}.jsonl").write_bytes(
            b"".join(canonical_json_bytes(row) + b"\n" for row in receipts)
        )
    return output_dir, checkpoint_dir, specs


def _reconcile(root: Path, **changes: Any) -> Any:
    output_dir = changes.pop("output_dir", root / "output")
    checkpoint_dir = changes.pop("checkpoint_dir", root / "checkpoints")
    destination = changes.pop("destination", root / "aggregate.jsonl.zst")
    return reconcile_self_play.reconcile_self_play(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        seed_start=20_000,
        seed_end=20_004,
        destination=destination,
        worker_count=2,
        **changes,
    )


def test_streams_complete_workers_in_stable_order_and_writes_bound_provenance(
    tmp_path: Path,
) -> None:
    output_dir, checkpoint_dir, _ = _completed_workers(tmp_path)
    source_bytes = {
        path: path.read_bytes() for path in (*output_dir.rglob("*.zst"), *checkpoint_dir.glob("*"))
    }

    result = _reconcile(tmp_path)

    destination = tmp_path / "aggregate.jsonl.zst"
    provenance_path = Path(f"{destination}.provenance.json")
    dataset = load_joint_dataset(destination)
    provenance = json.loads(provenance_path.read_bytes())
    assert [(row.world_seed, row.decision_index) for row in dataset.rows] == [
        (seed, index) for seed in range(20_000, 20_004) for index in range(2)
    ]
    assert result.row_count == provenance["row_count"] == 8
    assert result.game_count == provenance["game_count"] == 4
    assert result.dataset_digest == provenance["dataset_digest"] == dataset.digest
    assert result.output_digest == provenance["output_digest"]
    assert result.output_digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert len(provenance["fragments"]) == 4
    assert len(provenance["checkpoints"]) == 2
    assert provenance["seed_range"] == {"start": 20_000, "end": 20_004}
    assert provenance_path.read_bytes() == canonical_value_bytes(provenance)
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes


@pytest.mark.parametrize("fault", ["missing", "checksum", "trailing"])
def test_corrupt_or_missing_fragment_fails_without_publication(tmp_path: Path, fault: str) -> None:
    output_dir, _, specs = _completed_workers(tmp_path)
    game = specs[0].games[0]
    fragment = (
        output_dir
        / "worker-0"
        / f"{game.world_seed:020d}-{game.game_id(specs[0].config)}.jsonl.zst"
    )
    if fault == "missing":
        fragment.unlink()
    elif fault == "checksum":
        payload = bytearray(fragment.read_bytes())
        payload[-1] ^= 1
        fragment.write_bytes(payload)
    else:
        fragment.write_bytes(fragment.read_bytes() + b"trailing")

    with pytest.raises((OSError, ValueError), match=r"fragment|compressed|digest|zstd"):
        _reconcile(tmp_path)
    assert not (tmp_path / "aggregate.jsonl.zst").exists()
    assert not Path(f"{tmp_path / 'aggregate.jsonl.zst'}.provenance.json").exists()


@pytest.mark.parametrize("fault", ["missing_seed", "duplicate_seed", "wrong_worker"])
def test_checkpoint_requires_exact_unique_seed_assignment(tmp_path: Path, fault: str) -> None:
    _, checkpoint_dir, _ = _completed_workers(tmp_path)
    first = checkpoint_dir / "worker-0.jsonl"
    rows = [CheckpointRow.model_validate_json(line) for line in first.read_bytes().splitlines()]
    if fault == "missing_seed":
        rows.pop()
    elif fault == "duplicate_seed":
        rows.append(rows[0])
    else:
        rows[0] = rows[0].model_copy(update={"worker_id": 1})
    first.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))

    with pytest.raises(ValueError, match=r"coverage|duplicate|assignment|worker"):
        _reconcile(tmp_path)


def test_rejects_noncanonical_checkpoint_and_out_of_order_fragment(tmp_path: Path) -> None:
    output_dir, checkpoint_dir, specs = _completed_workers(tmp_path)
    checkpoint = checkpoint_dir / "worker-0.jsonl"
    checkpoint.write_bytes(checkpoint.read_bytes().replace(b'"game_id"', b'"game_id" '))
    with pytest.raises(ValueError, match="canonical"):
        _reconcile(tmp_path)

    _completed_workers(tmp_path)
    game = specs[0].games[0]
    fragment = (
        output_dir
        / "worker-0"
        / f"{game.world_seed:020d}-{game.game_id(specs[0].config)}.jsonl.zst"
    )
    write_joint_dataset(fragment, (_row(specs[0], game, 1), _row(specs[0], game, 0)))
    rows = [
        CheckpointRow.model_validate_json(line) for line in checkpoint.read_bytes().splitlines()
    ]
    rows[0] = rows[0].model_copy(
        update={"fragment_digest": hashlib.sha256(fragment.read_bytes()).hexdigest()}
    )
    checkpoint.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match=r"contiguous|ordered"):
        _reconcile(tmp_path)


def test_existing_matching_publication_is_idempotent_and_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    _completed_workers(tmp_path)
    first = _reconcile(tmp_path)
    destination = tmp_path / "aggregate.jsonl.zst"
    provenance = Path(f"{destination}.provenance.json")
    before = (destination.read_bytes(), provenance.read_bytes())

    assert _reconcile(tmp_path) == first
    assert (destination.read_bytes(), provenance.read_bytes()) == before

    provenance.write_bytes(provenance.read_bytes().replace(b'"row_count":8', b'"row_count":9'))
    with pytest.raises(ValueError, match=r"existing|provenance|publication"):
        _reconcile(tmp_path)
    assert destination.read_bytes() == before[0]


def test_rejects_symlink_destination_and_unsafe_output_overlap(tmp_path: Path) -> None:
    output_dir, _, _ = _completed_workers(tmp_path)
    target = tmp_path / "target"
    target.write_bytes(b"unrelated")
    destination = tmp_path / "aggregate.jsonl.zst"
    destination.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _reconcile(tmp_path, destination=destination)
    assert target.read_bytes() == b"unrelated"

    destination.unlink()
    with pytest.raises(ValueError, match=r"unsafe|overlap|source"):
        _reconcile(tmp_path, destination=output_dir / "aggregate.jsonl.zst")


@pytest.mark.parametrize("failure_call", [1, 2])
def test_atomic_failure_leaves_no_partial_publication_or_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_call: int
) -> None:
    output_dir, checkpoint_dir, _ = _completed_workers(tmp_path)
    source_bytes = {
        path: path.read_bytes() for path in (*output_dir.rglob("*.zst"), *checkpoint_dir.glob("*"))
    }

    real_replace = reconcile_self_play.os.replace
    calls = 0

    def fail_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("simulated atomic failure")
        real_replace(source, destination)

    monkeypatch.setattr(reconcile_self_play.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic failure"):
        _reconcile(tmp_path)
    assert not (tmp_path / "aggregate.jsonl.zst").exists()
    assert not Path(f"{tmp_path / 'aggregate.jsonl.zst'}.provenance.json").exists()
    assert not list(tmp_path.glob(".aggregate.jsonl.zst.*.tmp"))
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes


def test_publication_lock_is_blocking_not_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _completed_workers(tmp_path)
    real_flock = reconcile_self_play.fcntl.flock
    operations: list[int] = []

    def record_flock(descriptor: int, operation: int) -> Any:
        operations.append(operation)
        return real_flock(descriptor, operation)

    monkeypatch.setattr(reconcile_self_play.fcntl, "flock", record_flock)
    _reconcile(tmp_path)

    assert reconcile_self_play.fcntl.LOCK_EX in operations
    assert all(not (operation & reconcile_self_play.fcntl.LOCK_NB) for operation in operations)


def test_reconciliation_does_not_use_materializing_dataset_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _completed_workers(tmp_path)
    monkeypatch.setattr(
        "automata.training.dataset.load_joint_dataset",
        lambda *_args, **_kwargs: pytest.fail("reconciliation must remain streaming"),
    )

    result = _reconcile(tmp_path)

    assert result.row_count == 8
