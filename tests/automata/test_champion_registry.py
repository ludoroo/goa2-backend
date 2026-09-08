"""Behavioral contract for immutable candidate and champion management."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import stat
from multiprocessing.synchronize import Event as EventType
from pathlib import Path

import pytest
import torch

from automata.models.contracts import ArtifactScope, RuntimeRequirements
from automata.models.shared_encoder.artifacts import export_model_artifact
from automata.models.shared_encoder.model import JointModelConfig, JointPolicyValueModel
from automata.models.shared_encoder.schema import TensorFeatureSchema
from automata.training.registry import CandidateMetadata, ChampionRegistry, RegistryBusyError

HEROES = ("Arien", "Brogan", "Razzle", "Wasp")
ADAPTERS = {"generic": 1, "Arien": 1, "Brogan": 1, "Razzle": 1, "Wasp": 1}


@pytest.fixture(scope="module")
def requirements() -> RuntimeRequirements:
    return RuntimeRequirements(
        runtime_compatibility_version=1,
        observation_schema_version=3,
        map_schema_version=1,
        heroes=frozenset(HEROES),
        map_id="forgotten_island",
        game_type="QUICK",
        hero_adapter_versions=ADAPTERS,
    )


def _artifact(path: Path, *, seed: int) -> Path:
    schema = TensorFeatureSchema.current()
    config = JointModelConfig(
        model_version=1,
        schema_digest=schema.digest,
        token_width=8,
        state_width=12,
        candidate_width=8,
        message_passing_layers=1,
        dropout=0.0,
    )
    torch.manual_seed(seed)
    model = JointPolicyValueModel(schema=schema, config=config)
    export_model_artifact(
        path,
        model=model,
        schema=schema,
        scope=ArtifactScope(
            supported_heroes=HEROES,
            supported_maps=("forgotten_island",),
            supported_game_types=("QUICK",),
            hero_adapter_versions=ADAPTERS,
            map_schema_version=1,
        ),
        runtime_compatibility_version=1,
    )
    return path


def _metadata(*, generation: int, parent_digest: str | None) -> CandidateMetadata:
    return CandidateMetadata(
        generation=generation,
        parent_champion_digest=parent_digest,
        training_data_digests=("1" * 64,),
        source_revision="790b0d4",
        source_tree_digest="2" * 64,
        search_config={"simulations": 64},
        training_config={"epochs": 3},
        offline_metrics={"loss": 0.25},
        arena_results={"paired_score": 0.61},
    )


def _hold_registry_lock(lock_path: str, ready: EventType, release: EventType) -> None:
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        ready.set()
        release.wait(timeout=10)
    finally:
        os.close(descriptor)


def test_register_and_promote_resolves_a_loader_validated_immutable_champion(
    tmp_path: Path, requirements: RuntimeRequirements
) -> None:
    source = _artifact(tmp_path / "source", seed=11)
    registry = ChampionRegistry(tmp_path / "registry", requirements=requirements)

    candidate = registry.register_candidate(
        source, metadata=_metadata(generation=0, parent_digest=None)
    )
    with pytest.raises(FileExistsError):
        registry.register_candidate(source, metadata=_metadata(generation=0, parent_digest=None))
    (source / "weights.pt").write_bytes((source / "weights.pt").read_bytes() + b"corrupt")
    pointer = registry.promote(candidate.digest)
    champion = registry.load_champion()

    assert pointer.manifest_digest == candidate.digest
    assert champion.manifest == candidate
    assert champion.artifact.manifest.model_digest == candidate.artifact_digest
    assert champion.artifact_path.parent.name == "artifacts"
    assert champion.artifact_path != source


@pytest.mark.parametrize(
    ("generation", "parent"),
    [(0, "current"), (1, None), (2, "current"), (1, "3" * 64)],
)
def test_promotion_rejects_stale_or_skipped_lineage(
    tmp_path: Path,
    requirements: RuntimeRequirements,
    generation: int,
    parent: str | None,
) -> None:
    registry = ChampionRegistry(tmp_path / "registry", requirements=requirements)
    first = registry.register_candidate(
        _artifact(tmp_path / "first", seed=21),
        metadata=_metadata(generation=0, parent_digest=None),
    )
    registry.promote(first.digest)
    parent_digest = first.digest if parent == "current" else parent
    candidate = registry.register_candidate(
        _artifact(tmp_path / "next", seed=22),
        metadata=_metadata(generation=generation, parent_digest=parent_digest),
    )

    with pytest.raises(ValueError, match=r"parent|generation"):
        registry.promote(candidate.digest)

    assert registry.load_champion().manifest.digest == first.digest


def test_rejection_preserves_digest_validated_evidence_and_blocks_promotion(
    tmp_path: Path, requirements: RuntimeRequirements
) -> None:
    registry = ChampionRegistry(tmp_path / "registry", requirements=requirements)
    candidate = registry.register_candidate(
        _artifact(tmp_path / "candidate", seed=31),
        metadata=_metadata(generation=0, parent_digest=None),
    )

    rejected = registry.reject(
        candidate.digest,
        evidence={"gate": "reliability", "illegal_actions": 1, "passed": False},
    )

    assert registry.rejection(candidate.digest) == rejected
    assert rejected.candidate_digest == candidate.digest
    assert rejected.evidence["illegal_actions"] == 1
    with pytest.raises(FileExistsError):
        registry.reject(candidate.digest, evidence={"gate": "arena"})
    with pytest.raises(ValueError, match="rejected"):
        registry.promote(candidate.digest)


def test_rollback_atomically_restores_an_ancestor_and_allows_a_new_branch(
    tmp_path: Path, requirements: RuntimeRequirements
) -> None:
    registry = ChampionRegistry(tmp_path / "registry", requirements=requirements)
    first = registry.register_candidate(
        _artifact(tmp_path / "first", seed=41),
        metadata=_metadata(generation=0, parent_digest=None),
    )
    registry.promote(first.digest)
    second = registry.register_candidate(
        _artifact(tmp_path / "second", seed=42),
        metadata=_metadata(generation=1, parent_digest=first.digest),
    )
    registry.promote(second.digest)

    rolled_back = registry.rollback(first.digest)

    assert rolled_back.manifest_digest == first.digest
    assert rolled_back.previous_manifest_digest == second.digest
    assert registry.load_champion().manifest == first
    replacement = registry.register_candidate(
        _artifact(tmp_path / "replacement", seed=43),
        metadata=_metadata(generation=1, parent_digest=first.digest),
    )
    assert registry.promote(replacement.digest).manifest_digest == replacement.digest
    with pytest.raises(ValueError, match=r"ancestor|rollback"):
        registry.rollback(second.digest)


def test_interrupted_pointer_replace_leaves_the_previous_champion_complete(
    tmp_path: Path,
    requirements: RuntimeRequirements,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ChampionRegistry(tmp_path / "registry", requirements=requirements)
    first = registry.register_candidate(
        _artifact(tmp_path / "first", seed=51),
        metadata=_metadata(generation=0, parent_digest=None),
    )
    registry.promote(first.digest)
    second = registry.register_candidate(
        _artifact(tmp_path / "second", seed=52),
        metadata=_metadata(generation=1, parent_digest=first.digest),
    )
    real_replace = os.replace

    def interrupted_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name == "champion.json":
            raise OSError("simulated interruption")
        real_replace(source, destination)

    monkeypatch.setattr("automata.training.registry.os.replace", interrupted_replace)
    with pytest.raises(OSError, match="interruption"):
        registry.promote(second.digest)

    assert registry.load_champion().manifest == first
    assert not list((tmp_path / "registry").glob(".champion.json.*"))


def test_pointer_publication_fsyncs_file_and_containing_directory(
    tmp_path: Path,
    requirements: RuntimeRequirements,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ChampionRegistry(tmp_path / "registry", requirements=requirements)
    candidate = registry.register_candidate(
        _artifact(tmp_path / "candidate", seed=61),
        metadata=_metadata(generation=0, parent_digest=None),
    )
    real_fsync = os.fsync
    synced_kinds: set[str] = set()

    def observed_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        synced_kinds.add("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr("automata.training.registry.os.fsync", observed_fsync)
    registry.promote(candidate.digest)

    assert synced_kinds == {"file", "directory"}


def test_digest_mutation_fails_closed_before_loading_a_champion(
    tmp_path: Path, requirements: RuntimeRequirements
) -> None:
    registry = ChampionRegistry(tmp_path / "registry", requirements=requirements)
    candidate = registry.register_candidate(
        _artifact(tmp_path / "candidate", seed=71),
        metadata=_metadata(generation=0, parent_digest=None),
    )
    registry.promote(candidate.digest)
    manifest_path = tmp_path / "registry" / "manifests" / f"{candidate.digest}.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["offline_metrics"]["loss"] = 0.01
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"manifest|digest"):
        registry.load_champion()


def test_concurrent_process_lock_prevents_a_competing_transition(
    tmp_path: Path, requirements: RuntimeRequirements
) -> None:
    registry = ChampionRegistry(tmp_path / "registry", requirements=requirements)
    candidate = registry.register_candidate(
        _artifact(tmp_path / "candidate", seed=81),
        metadata=_metadata(generation=0, parent_digest=None),
    )
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_registry_lock,
        args=(str(tmp_path / "registry" / ".lock"), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=10)
        with pytest.raises(RegistryBusyError, match="busy"):
            registry.promote(candidate.digest)
        assert registry.champion_pointer() is None
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join()
    assert process.exitcode == 0
