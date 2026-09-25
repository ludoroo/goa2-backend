"""Durable, immutable registry for learned-model candidates and champion pointers."""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from automata.models.contracts import RuntimeRequirements
from automata.models.shared_encoder.artifacts import LoadedModelArtifact, load_model_artifact
from automata.training.io import atomic_write_bytes as _atomic_write
from automata.training.io import canonical_json_bytes as _canonical
from automata.training.io import content_digest
from automata.training.io import fsync_directory as _fsync_directory


class RegistryError(ValueError):
    """Registry contents or a requested transition are invalid."""


class RegistryBusyError(RegistryError):
    """Another process currently owns the registry mutation lock."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _content_digest(value: BaseModel) -> str:
    payload = value.model_dump(mode="json", exclude={"digest"})
    return content_digest(payload)


class CandidateMetadata(_FrozenModel):
    """Training and evaluation provenance attached to one candidate."""

    generation: int = Field(ge=0)
    parent_champion_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_data_digests: tuple[str, ...]
    source_revision: str = Field(min_length=1)
    source_tree_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_config: dict[str, JsonValue]
    training_config: dict[str, JsonValue]
    offline_metrics: dict[str, JsonValue]
    arena_results: dict[str, JsonValue]
    runtime_format: str = "pytorch_state_dict"

    @model_validator(mode="after")
    def _validate_digests(self) -> CandidateMetadata:
        if not self.training_data_digests or any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.training_data_digests
        ):
            raise ValueError("training data digests must be nonempty SHA-256 digests")
        return self


class CandidateManifest(_FrozenModel):
    """Digest-bearing immutable description of a registered model candidate."""

    schema_version: Literal[1] = 1
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(ge=0)
    parent_champion_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    observation_schema_version: int = Field(gt=0)
    hero_adapter_versions: dict[str, int]
    map_schema_version: int = Field(gt=0)
    training_data_digests: tuple[str, ...]
    source_revision: str
    source_tree_digest: str
    search_config: dict[str, JsonValue]
    training_config: dict[str, JsonValue]
    supported_heroes: tuple[str, ...]
    supported_maps: tuple[str, ...]
    supported_game_types: tuple[str, ...]
    offline_metrics: dict[str, JsonValue]
    arena_results: dict[str, JsonValue]
    runtime_format: str
    runtime_compatibility_version: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_digest(self) -> CandidateManifest:
        if self.digest != _content_digest(self):
            raise ValueError("candidate manifest digest mismatch")
        return self


class ChampionPointer(_FrozenModel):
    """The complete atomic champion reference."""

    schema_version: Literal[1] = 1
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(ge=0)
    previous_manifest_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_digest(self) -> ChampionPointer:
        if self.digest != _content_digest(self):
            raise ValueError("champion pointer digest mismatch")
        return self


class RejectionRecord(_FrozenModel):
    """Immutable gate evidence explaining why a candidate was rejected."""

    schema_version: Literal[1] = 1
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(ge=0)
    evidence: dict[str, JsonValue]

    @model_validator(mode="after")
    def _validate_digest(self) -> RejectionRecord:
        if not self.evidence:
            raise ValueError("rejection evidence cannot be empty")
        if self.digest != _content_digest(self):
            raise ValueError("rejection record digest mismatch")
        return self


class LoadedChampion(_FrozenModel):
    """A resolved champion with its fully validated executable artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    pointer: ChampionPointer
    manifest: CandidateManifest
    artifact_path: Path
    artifact: LoadedModelArtifact


def _with_digest(model_type: type[BaseModel], values: dict[str, Any]) -> Any:
    provisional = model_type.model_construct(digest="0" * 64, **values)
    return model_type(digest=_content_digest(provisional), **values)


class ChampionRegistry:
    """Content-addressed candidate store with serialized champion transitions."""

    def __init__(self, root: str | Path, *, requirements: RuntimeRequirements) -> None:
        self.root = Path(root)
        self.requirements = requirements
        self.artifacts = self.root / "artifacts"
        self.manifests = self.root / "manifests"
        self.rejections = self.root / "rejections"
        self.pointer_path = self.root / "champion.json"
        created = not self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (self.artifacts, self.manifests, self.rejections):
            directory.mkdir(exist_ok=True)
        if created:
            _fsync_directory(self.root.parent)
        _fsync_directory(self.root)

    @contextmanager
    def _mutation_lock(self) -> Iterator[None]:
        descriptor = os.open(self.root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RegistryBusyError("champion registry is busy") from exc
            yield
        finally:
            os.close(descriptor)

    def register_candidate(
        self, source: str | Path, *, metadata: CandidateMetadata
    ) -> CandidateManifest:
        """Validate and durably store a new immutable candidate."""

        loaded = load_model_artifact(source, requirements=self.requirements)
        artifact = loaded.manifest
        values = {
            "artifact_digest": artifact.model_digest,
            "generation": metadata.generation,
            "parent_champion_digest": metadata.parent_champion_digest,
            "observation_schema_version": artifact.observation_schema_version,
            "hero_adapter_versions": artifact.hero_adapter_versions,
            "map_schema_version": artifact.map_schema_version,
            "training_data_digests": metadata.training_data_digests,
            "source_revision": metadata.source_revision,
            "source_tree_digest": metadata.source_tree_digest,
            "search_config": metadata.search_config,
            "training_config": metadata.training_config,
            "supported_heroes": artifact.supported_heroes,
            "supported_maps": artifact.supported_maps,
            "supported_game_types": artifact.supported_game_types,
            "offline_metrics": metadata.offline_metrics,
            "arena_results": metadata.arena_results,
            "runtime_format": metadata.runtime_format,
            "runtime_compatibility_version": artifact.runtime_compatibility_version,
        }
        manifest: CandidateManifest = _with_digest(CandidateManifest, values)
        with self._mutation_lock():
            manifest_path = self.manifests / f"{manifest.digest}.json"
            if manifest_path.exists():
                raise FileExistsError(f"candidate manifest already exists: {manifest.digest}")
            self._store_artifact(Path(source), artifact.model_digest)
            _atomic_write(manifest_path, _canonical(manifest))
        return manifest

    def finalize_candidate(
        self, candidate_digest: str, *, arena_results: dict[str, JsonValue]
    ) -> CandidateManifest:
        """Create an immutable arena-evidenced copy of a registered candidate."""
        if not arena_results:
            raise RegistryError("finalized candidate requires arena evidence")
        with self._mutation_lock():
            candidate = self.candidate(candidate_digest)
            values = candidate.model_dump(mode="python", exclude={"digest", "arena_results"})
            values["arena_results"] = arena_results
            finalized: CandidateManifest = _with_digest(CandidateManifest, values)
            path = self.manifests / f"{finalized.digest}.json"
            if path.exists():
                existing = self.candidate(finalized.digest)
                if existing != finalized:
                    raise RegistryError("finalized candidate manifest has conflicting content")
                return existing
            _atomic_write(path, _canonical(finalized))
            return finalized

    def _store_artifact(self, source: Path, digest: str) -> None:
        target = self.artifacts / digest
        if target.exists():
            load_model_artifact(target, requirements=self.requirements)
            return
        temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=self.artifacts))
        try:
            shutil.rmtree(temporary)
            shutil.copytree(source, temporary)
            for child in temporary.iterdir():
                if child.is_file():
                    with child.open("rb") as stream:
                        os.fsync(stream.fileno())
            _fsync_directory(temporary)
            os.rename(temporary, target)
            _fsync_directory(self.artifacts)
            load_model_artifact(target, requirements=self.requirements)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _read_model(self, path: Path, model_type: type[BaseModel], label: str) -> Any:
        try:
            payload = path.read_bytes()
            model = model_type.model_validate_json(payload)
        except (OSError, ValueError) as exc:
            raise RegistryError(f"invalid {label}") from exc
        if payload != _canonical(model):
            raise RegistryError(f"{label} is not canonical JSON")
        return model

    def candidate(self, digest: str) -> CandidateManifest:
        return self._read_model(
            self.manifests / f"{digest}.json", CandidateManifest, "candidate manifest"
        )

    def champion_pointer(self) -> ChampionPointer | None:
        if not self.pointer_path.exists():
            return None
        return self._read_model(self.pointer_path, ChampionPointer, "champion pointer")

    def rejection(self, candidate_digest: str) -> RejectionRecord:
        return self._read_model(
            self.rejections / f"{candidate_digest}.json", RejectionRecord, "rejection record"
        )

    def reject(self, candidate_digest: str, *, evidence: dict[str, JsonValue]) -> RejectionRecord:
        """Durably record evidence for a candidate that must not be promoted."""

        with self._mutation_lock():
            candidate = self.candidate(candidate_digest)
            path = self.rejections / f"{candidate_digest}.json"
            if path.exists():
                raise FileExistsError(f"candidate rejection already exists: {candidate_digest}")
            values = {
                "candidate_digest": candidate.digest,
                "artifact_digest": candidate.artifact_digest,
                "generation": candidate.generation,
                "evidence": evidence,
            }
            record: RejectionRecord = _with_digest(RejectionRecord, values)
            _atomic_write(path, _canonical(record))
            return record

    def promote(self, candidate_digest: str) -> ChampionPointer:
        """Atomically make a registered candidate the champion."""

        with self._mutation_lock():
            candidate = self.candidate(candidate_digest)
            if not candidate.arena_results:
                raise RegistryError("candidate without arena evidence cannot be promoted")
            if (self.rejections / f"{candidate_digest}.json").exists():
                # Parse the record rather than trusting the path's existence.
                self.rejection(candidate_digest)
                raise RegistryError("rejected candidate cannot be promoted")
            current = self.champion_pointer()
            expected_generation = 0 if current is None else current.generation + 1
            expected_parent = None if current is None else current.manifest_digest
            if (
                candidate.generation != expected_generation
                or candidate.parent_champion_digest != expected_parent
            ):
                raise RegistryError(
                    "candidate parent or generation does not match current champion"
                )
            values = {
                "manifest_digest": candidate.digest,
                "artifact_digest": candidate.artifact_digest,
                "generation": candidate.generation,
                "previous_manifest_digest": expected_parent,
            }
            pointer: ChampionPointer = _with_digest(ChampionPointer, values)
            _atomic_write(self.pointer_path, _canonical(pointer))
            return pointer

    def rollback(self, ancestor_digest: str) -> ChampionPointer:
        """Atomically restore a prior champion on the current lineage."""

        with self._mutation_lock():
            current = self.champion_pointer()
            if current is None:
                raise RegistryError("cannot rollback an uninitialized champion")
            cursor = self.candidate(current.manifest_digest)
            is_ancestor = False
            while cursor.parent_champion_digest is not None:
                if cursor.parent_champion_digest == ancestor_digest:
                    is_ancestor = True
                    break
                cursor = self.candidate(cursor.parent_champion_digest)
            if not is_ancestor:
                raise RegistryError("rollback target is not an ancestor of the current champion")
            ancestor = self.candidate(ancestor_digest)
            values = {
                "manifest_digest": ancestor.digest,
                "artifact_digest": ancestor.artifact_digest,
                "generation": ancestor.generation,
                "previous_manifest_digest": current.manifest_digest,
            }
            pointer: ChampionPointer = _with_digest(ChampionPointer, values)
            _atomic_write(self.pointer_path, _canonical(pointer))
            return pointer

    def load_champion(self) -> LoadedChampion:
        pointer = self.champion_pointer()
        if pointer is None:
            raise RegistryError("champion is not initialized")
        manifest = self.candidate(pointer.manifest_digest)
        if (
            pointer.artifact_digest != manifest.artifact_digest
            or pointer.generation != manifest.generation
        ):
            raise RegistryError("champion pointer and manifest disagree")
        artifact_path = self.artifacts / manifest.artifact_digest
        artifact = load_model_artifact(artifact_path, requirements=self.requirements)
        return LoadedChampion(
            pointer=pointer,
            manifest=manifest,
            artifact_path=artifact_path,
            artifact=artifact,
        )


__all__ = [
    "CandidateManifest",
    "CandidateMetadata",
    "ChampionPointer",
    "ChampionRegistry",
    "LoadedChampion",
    "RegistryBusyError",
    "RegistryError",
    "RejectionRecord",
]
