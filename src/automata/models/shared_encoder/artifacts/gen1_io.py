"""Immutable artifact IO for the Gen1 policy/stable-value model."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import JsonValue, ValidationError

from ...contracts import (
    GEN1_RUNTIME_COMPATIBILITY_VERSION,
    ArtifactError,
    ArtifactScope,
    Gen1RuntimeRequirements,
    canonical_json_bytes,
)
from ..gen1_model import GEN1_ARCHITECTURE_ID, Gen1ModelConfig, Gen1PolicyValueModel
from ..schema import StableValueTensorSchema, TensorFeatureSchema
from .io import (
    _canonical_mapping_bytes,
    _file_record,
    _model_digest,
    _tensor_inventory,
    _validate_state,
)
from .manifest import Gen1ModelArtifactManifest


@dataclass(frozen=True, slots=True)
class LoadedGen1ModelArtifact:
    manifest: Gen1ModelArtifactManifest
    decision_schema: TensorFeatureSchema
    stable_value_schema: StableValueTensorSchema
    config: Gen1ModelConfig
    model: Gen1PolicyValueModel


def _executable_metadata(
    *,
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
    config: Mapping[str, JsonValue],
    scope: ArtifactScope,
    runtime_compatibility_version: int,
    serialized_decision_schema: Mapping[str, JsonValue] | None = None,
    serialized_stable_value_schema: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {
        "artifact_schema_version": 3,
        "artifact_kind": "GEN1_POLICY_STABLE_VALUE",
        "model_id": GEN1_ARCHITECTURE_ID,
        "runtime_compatibility_version": runtime_compatibility_version,
        "decision_observation_schema_version": decision_schema.observation_schema_version,
        "stable_value_observation_schema_version": stable_value_schema.observation_schema_version,
        "graph_observation_schema_version": stable_value_schema.graph_observation_schema_version,
        "map_schema_version": scope.map_schema_version,
        "value_semantics": "stable-boundary-outcome-v1",
        "hero_adapter_versions": dict(scope.hero_adapter_versions),
        "supported_heroes": list(scope.supported_heroes),
        "supported_maps": list(scope.supported_maps),
        "supported_game_types": list(scope.supported_game_types),
        "decision_tensor_schema": dict(
            serialized_decision_schema or decision_schema.model_dump(mode="json")
        ),
        "stable_value_tensor_schema": dict(
            serialized_stable_value_schema or stable_value_schema.model_dump(mode="json")
        ),
        "architecture_config": dict(config),
    }


def _validate_export_inputs(
    *,
    model: Gen1PolicyValueModel,
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
    scope: ArtifactScope,
    runtime_compatibility_version: int,
) -> None:
    if runtime_compatibility_version != GEN1_RUNTIME_COMPATIBILITY_VERSION:
        raise ArtifactError("only Gen1 runtime compatibility version 1 is supported")
    if model.config.decision_schema_digest != decision_schema.digest:
        raise ArtifactError("model config and decision tensor schema are incompatible")
    if model.config.stable_value_schema_digest != stable_value_schema.digest:
        raise ArtifactError("model config and stable-value tensor schema are incompatible")
    if decision_schema.tokens != stable_value_schema.tokens or (
        decision_schema.relationships != stable_value_schema.relationships
    ):
        raise ArtifactError("decision and stable-value schemas do not share a graph")
    adapters = scope.hero_adapter_versions
    if "generic" not in adapters or any(hero not in adapters for hero in scope.supported_heroes):
        raise ArtifactError("artifact scope lacks required hero adapter versions")


def export_gen1_model_artifact(
    destination: str | Path,
    *,
    model: Gen1PolicyValueModel,
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
    scope: ArtifactScope,
    runtime_compatibility_version: int = GEN1_RUNTIME_COMPATIBILITY_VERSION,
    provenance: Mapping[str, Any] | None = None,
) -> Gen1ModelArtifactManifest:
    """Publish a new closed Gen1 artifact directory without replacing a path."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"artifact destination already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"artifact parent directory does not exist: {target.parent}")
    _validate_export_inputs(
        model=model,
        decision_schema=decision_schema,
        stable_value_schema=stable_value_schema,
        scope=scope,
        runtime_compatibility_version=runtime_compatibility_version,
    )

    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    config = cast(dict[str, JsonValue], asdict(model.config))
    metadata = _executable_metadata(
        decision_schema=decision_schema,
        stable_value_schema=stable_value_schema,
        config=config,
        scope=scope,
        runtime_compatibility_version=runtime_compatibility_version,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    owns_target = False
    try:
        (temporary / "decision_schema.json").write_bytes(canonical_json_bytes(decision_schema))
        (temporary / "stable_value_schema.json").write_bytes(
            canonical_json_bytes(stable_value_schema)
        )
        with (temporary / "weights.pt").open("wb") as stream:
            torch.save(state, stream)
        non_executable: tuple[str, ...] = ()
        if provenance is not None:
            (temporary / "provenance.json").write_bytes(_canonical_mapping_bytes(provenance))
            non_executable = ("provenance.json",)
        payload_names = (
            "decision_schema.json",
            "stable_value_schema.json",
            "weights.pt",
            *non_executable,
        )
        files = {name: _file_record(temporary / name) for name in payload_names}
        manifest = Gen1ModelArtifactManifest(
            model_digest=_model_digest(metadata, state),
            runtime_compatibility_version=cast(Literal[1], runtime_compatibility_version),
            map_schema_version=scope.map_schema_version,
            hero_adapter_versions=dict(scope.hero_adapter_versions),
            supported_heroes=scope.supported_heroes,
            supported_maps=scope.supported_maps,
            supported_game_types=scope.supported_game_types,
            decision_tensor_schema_id=decision_schema.schema_id,
            decision_tensor_schema_version=decision_schema.schema_version,
            decision_tensor_schema_digest=decision_schema.digest,
            stable_value_tensor_schema_id=stable_value_schema.schema_id,
            stable_value_tensor_schema_version=stable_value_schema.schema_version,
            stable_value_tensor_schema_digest=stable_value_schema.digest,
            architecture_config=config,
            tensors=_tensor_inventory(state),
            files=files,
            non_executable_files=non_executable,
        )
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))

        try:
            target.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(f"artifact destination already exists: {target}") from exc
        owns_target = True
        for name in payload_names:
            (temporary / name).replace(target / name)
        (temporary / "manifest.json").replace(target / "manifest.json")
        return manifest
    except BaseException:
        if owns_target:
            shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _read_manifest(path: Path) -> Gen1ModelArtifactManifest:
    try:
        payload = path.read_bytes()
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("manifest must be an object")
        if raw.get("schema_version") != 3 or raw.get("artifact_kind") != (
            "GEN1_POLICY_STABLE_VALUE"
        ):
            raise ArtifactError("artifact is not a Gen1 policy/stable-value artifact")
        manifest = Gen1ModelArtifactManifest.model_validate(raw)
    except ArtifactError:
        raise
    except (OSError, ValueError, ValidationError) as exc:
        raise ArtifactError("invalid Gen1 artifact manifest") from exc
    if payload != canonical_json_bytes(manifest):
        raise ArtifactError("Gen1 artifact manifest is not canonical JSON")
    return manifest


def _validate_files(path: Path, manifest: Gen1ModelArtifactManifest) -> None:
    expected = {"manifest.json", *manifest.files}
    try:
        entries = tuple(path.iterdir())
    except OSError as exc:
        raise ArtifactError("artifact directory could not be inspected") from exc
    if any(entry.is_symlink() for entry in entries):
        raise ArtifactError("artifact contains a symlink")
    actual_files = {entry.name for entry in entries if entry.is_file()}
    other_entries = [entry.name for entry in entries if not entry.is_file()]
    if actual_files != expected or other_entries:
        raise ArtifactError("artifact contains missing or unexpected files outside its allowlist")
    for name, record in manifest.files.items():
        if _file_record(path / name) != record:
            raise ArtifactError(f"artifact file integrity hash or length mismatch: {name}")


def _require_compatible(
    manifest: Gen1ModelArtifactManifest, required: Gen1RuntimeRequirements
) -> None:
    versions = (
        ("runtime compatibility", required.runtime_compatibility_version, 1),
        (
            "decision observation schema",
            required.decision_observation_schema_version,
            manifest.decision_observation_schema_version,
        ),
        (
            "stable-value observation schema",
            required.stable_value_observation_schema_version,
            manifest.stable_value_observation_schema_version,
        ),
        (
            "graph observation schema",
            required.graph_observation_schema_version,
            manifest.graph_observation_schema_version,
        ),
        ("map schema", required.map_schema_version, manifest.map_schema_version),
    )
    for label, wanted, supported in versions:
        if wanted != supported:
            raise ArtifactError(
                f"incompatible {label} version: requires {wanted}, supports {supported}"
            )
    if not required.heroes:
        raise ArtifactError("required hero scope cannot be empty")
    missing_heroes = required.heroes - set(manifest.supported_heroes)
    if missing_heroes:
        raise ArtifactError(f"unsupported hero scope: {sorted(missing_heroes)!r}")
    if required.map_id not in manifest.supported_maps:
        raise ArtifactError(f"unsupported map scope: {required.map_id!r}")
    if required.game_type not in manifest.supported_game_types:
        raise ArtifactError(f"unsupported game type scope: {required.game_type!r}")
    expected_adapters = {"generic", *required.heroes}
    if expected_adapters - set(required.hero_adapter_versions):
        raise ArtifactError("runtime requirements lack hero adapter scope")
    for adapter, version in required.hero_adapter_versions.items():
        if manifest.hero_adapter_versions.get(adapter) != version:
            raise ArtifactError(f"incompatible hero adapter {adapter!r} version {version}")


def _load_schema(
    path: Path, schema_type: type[TensorFeatureSchema] | type[StableValueTensorSchema], label: str
) -> tuple[TensorFeatureSchema | StableValueTensorSchema, dict[str, JsonValue]]:
    try:
        payload = path.read_bytes()
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("schema must be an object")
        schema = schema_type.model_validate(raw)
    except (OSError, ValueError, ValidationError) as exc:
        raise ArtifactError(f"invalid {label} tensor schema") from exc
    if payload != _canonical_mapping_bytes(raw):
        raise ArtifactError(f"{label} tensor schema is not canonical")
    return schema, cast(dict[str, JsonValue], raw)


def load_gen1_model_artifact(
    source: str | Path, *, requirements: Gen1RuntimeRequirements
) -> LoadedGen1ModelArtifact:
    """Verify all Gen1 metadata and inventories before loading tensor weights."""

    path = Path(source)
    if path.is_symlink():
        raise ArtifactError("artifact path cannot be a symlink")
    if not path.is_dir():
        raise ArtifactError(f"artifact path is not a directory: {path}")
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink():
        raise ArtifactError("artifact manifest cannot be a symlink")
    manifest = _read_manifest(manifest_path)
    _validate_files(path, manifest)
    _require_compatible(manifest, requirements)

    decision_raw, raw_decision = _load_schema(
        path / "decision_schema.json", TensorFeatureSchema, "decision"
    )
    value_raw, raw_value = _load_schema(
        path / "stable_value_schema.json", StableValueTensorSchema, "stable-value"
    )
    assert isinstance(decision_raw, TensorFeatureSchema)
    assert isinstance(value_raw, StableValueTensorSchema)
    decision_schema = decision_raw
    stable_value_schema = value_raw
    if (
        decision_schema.schema_id != manifest.decision_tensor_schema_id
        or decision_schema.schema_version != manifest.decision_tensor_schema_version
        or decision_schema.digest != manifest.decision_tensor_schema_digest
        or decision_schema.observation_schema_version
        != manifest.decision_observation_schema_version
    ):
        raise ArtifactError("decision tensor schema identity or digest mismatch")
    if (
        stable_value_schema.schema_id != manifest.stable_value_tensor_schema_id
        or stable_value_schema.schema_version != manifest.stable_value_tensor_schema_version
        or stable_value_schema.digest != manifest.stable_value_tensor_schema_digest
        or stable_value_schema.observation_schema_version
        != manifest.stable_value_observation_schema_version
        or stable_value_schema.graph_observation_schema_version
        != manifest.graph_observation_schema_version
    ):
        raise ArtifactError("stable-value tensor schema identity or digest mismatch")
    if decision_schema.tokens != stable_value_schema.tokens or (
        decision_schema.relationships != stable_value_schema.relationships
    ):
        raise ArtifactError("decision and stable-value tensor schemas have different graphs")

    try:
        config = Gen1ModelConfig(**cast(dict[str, Any], manifest.architecture_config))
    except (TypeError, ValueError) as exc:
        raise ArtifactError("model architecture config is invalid") from exc
    if (
        config.decision_schema_id != decision_schema.schema_id
        or config.decision_schema_version != decision_schema.schema_version
        or config.decision_schema_digest != decision_schema.digest
    ):
        raise ArtifactError("model config and decision tensor schema are incompatible")
    if (
        config.stable_value_schema_id != stable_value_schema.schema_id
        or config.stable_value_schema_version != stable_value_schema.schema_version
        or config.stable_value_schema_digest != stable_value_schema.digest
    ):
        raise ArtifactError("model config and stable-value tensor schema are incompatible")

    try:
        raw_state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactError("weight state could not be loaded safely") from exc
    state = _validate_state(raw_state, manifest.tensors)
    scope = ArtifactScope(
        supported_heroes=manifest.supported_heroes,
        supported_maps=manifest.supported_maps,
        supported_game_types=manifest.supported_game_types,
        hero_adapter_versions=manifest.hero_adapter_versions,
        map_schema_version=manifest.map_schema_version,
    )
    metadata = _executable_metadata(
        decision_schema=decision_schema,
        stable_value_schema=stable_value_schema,
        config=manifest.architecture_config,
        scope=scope,
        runtime_compatibility_version=manifest.runtime_compatibility_version,
        serialized_decision_schema=raw_decision,
        serialized_stable_value_schema=raw_value,
    )
    if _model_digest(metadata, state) != manifest.model_digest:
        raise ArtifactError("executable model digest mismatch")
    try:
        model = Gen1PolicyValueModel(
            decision_schema=decision_schema,
            stable_value_schema=stable_value_schema,
            config=config,
        )
        model.load_state_dict(state, strict=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ArtifactError("model config or weight state is incompatible") from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedGen1ModelArtifact(
        manifest=manifest,
        decision_schema=decision_schema,
        stable_value_schema=stable_value_schema,
        config=config,
        model=model,
    )


__all__ = [
    "LoadedGen1ModelArtifact",
    "export_gen1_model_artifact",
    "load_gen1_model_artifact",
]
