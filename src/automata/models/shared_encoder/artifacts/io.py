"""Immutable, content-addressed artifacts for the explicit PyTorch model boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import JsonValue

from ...contracts import (
    ArtifactError,
    ArtifactScope,
    RuntimeRequirements,
    canonical_json_bytes,
    from_canonical_json,
)
from ..model import JointModelConfig, JointPolicyValueModel
from ..schema import TensorFeatureSchema
from .manifest import ArtifactFile, ArtifactTensor, ModelArtifactManifest


@dataclass(frozen=True, slots=True)
class LoadedModelArtifact:
    manifest: ModelArtifactManifest
    schema: TensorFeatureSchema
    config: JointModelConfig
    model: JointPolicyValueModel


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes(order="C")


def _tensor_inventory(state: Mapping[str, torch.Tensor]) -> dict[str, ArtifactTensor]:
    return {
        name: ArtifactTensor(shape=tuple(value.shape), dtype=_dtype_name(value))
        for name, value in sorted(state.items())
    }


def _executable_metadata(
    *,
    schema: TensorFeatureSchema,
    config: Mapping[str, JsonValue],
    scope: ArtifactScope,
    runtime_compatibility_version: int,
) -> dict[str, JsonValue]:
    return {
        "runtime_compatibility_version": runtime_compatibility_version,
        "observation_schema_version": schema.observation_schema_version,
        "map_schema_version": scope.map_schema_version,
        "hero_adapter_versions": dict(scope.hero_adapter_versions),
        "supported_heroes": list(scope.supported_heroes),
        "supported_maps": list(scope.supported_maps),
        "supported_game_types": list(scope.supported_game_types),
        "tensor_schema": schema.model_dump(mode="json"),
        "architecture_config": dict(config),
    }


def _model_digest(metadata: Mapping[str, JsonValue], state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    for name, tensor in sorted(state.items()):
        raw = _tensor_bytes(tensor)
        descriptor = {
            "name": name,
            "dtype": _dtype_name(tensor),
            "shape": list(tensor.shape),
            "length": len(raw),
        }
        digest.update(b"\0")
        digest.update(json.dumps(descriptor, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
    return digest.hexdigest()


def _file_record(path: Path) -> ArtifactFile:
    payload = path.read_bytes()
    return ArtifactFile(length=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _canonical_mapping_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def export_model_artifact(
    destination: str | Path,
    *,
    model: JointPolicyValueModel,
    schema: TensorFeatureSchema,
    scope: ArtifactScope,
    runtime_compatibility_version: int,
    provenance: Mapping[str, Any] | None = None,
) -> ModelArtifactManifest:
    """Write a new artifact directory without ever replacing an existing path."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"artifact destination already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"artifact parent directory does not exist: {target.parent}")
    if model.config.schema_digest != schema.digest:
        raise ArtifactError("model config and tensor schema are incompatible")

    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    config: dict[str, JsonValue] = asdict(model.config)
    metadata = _executable_metadata(
        schema=schema,
        config=config,
        scope=scope,
        runtime_compatibility_version=runtime_compatibility_version,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        (temporary / "schema.json").write_bytes(canonical_json_bytes(schema))
        with (temporary / "weights.pt").open("wb") as stream:
            torch.save(state, stream)
        non_executable: tuple[str, ...] = ()
        if provenance is not None:
            (temporary / "provenance.json").write_bytes(_canonical_mapping_bytes(provenance))
            non_executable = ("provenance.json",)
        payload_names = ("schema.json", "weights.pt", *non_executable)
        files = {name: _file_record(temporary / name) for name in payload_names}
        manifest = ModelArtifactManifest(
            model_digest=_model_digest(metadata, state),
            observation_schema_version=schema.observation_schema_version,
            map_schema_version=scope.map_schema_version,
            runtime_compatibility_version=runtime_compatibility_version,
            hero_adapter_versions=dict(scope.hero_adapter_versions),
            supported_heroes=scope.supported_heroes,
            supported_maps=scope.supported_maps,
            supported_game_types=scope.supported_game_types,
            tensor_schema_id=schema.schema_id,
            tensor_schema_version=schema.schema_version,
            tensor_schema_digest=schema.digest,
            architecture_config=config,
            tensors=_tensor_inventory(state),
            files=files,
            non_executable_files=non_executable,
        )
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        os.rename(temporary, target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_manifest(path: Path) -> ModelArtifactManifest:
    try:
        payload = path.read_bytes()
        manifest = from_canonical_json(ModelArtifactManifest, payload)
    except (OSError, ValueError) as exc:
        raise ArtifactError("invalid artifact manifest") from exc
    if payload != canonical_json_bytes(manifest):
        raise ArtifactError("artifact manifest is not canonical JSON")
    return manifest


def _validate_files(path: Path, manifest: ModelArtifactManifest) -> None:
    expected = {"manifest.json", *manifest.files}
    try:
        actual = {item.name for item in path.iterdir() if item.is_file()}
        directories = [item.name for item in path.iterdir() if not item.is_file()]
    except OSError as exc:
        raise ArtifactError("artifact directory could not be inspected") from exc
    if actual != expected or directories:
        raise ArtifactError("artifact contains missing or unexpected files outside its allowlist")
    if {"schema.json", "weights.pt"} - set(manifest.files):
        raise ArtifactError("artifact file inventory is missing executable files")
    for name, record in manifest.files.items():
        current = _file_record(path / name)
        if current != record:
            raise ArtifactError(f"artifact file integrity hash or length mismatch: {name}")


def _require_compatible(manifest: ModelArtifactManifest, required: RuntimeRequirements) -> None:
    versions = (
        (
            "runtime compatibility",
            required.runtime_compatibility_version,
            manifest.runtime_compatibility_version,
        ),
        (
            "observation schema",
            required.observation_schema_version,
            manifest.observation_schema_version,
        ),
        ("map schema", required.map_schema_version, manifest.map_schema_version),
    )
    for label, wanted, supported in versions:
        if wanted != supported:
            raise ArtifactError(
                f"incompatible {label} version: requires {wanted}, supports {supported}"
            )
    missing_heroes = required.heroes - set(manifest.supported_heroes)
    if missing_heroes:
        raise ArtifactError(f"unsupported hero scope: {sorted(missing_heroes)!r}")
    if required.map_id not in manifest.supported_maps:
        raise ArtifactError(f"unsupported map scope: {required.map_id!r}")
    if required.game_type not in manifest.supported_game_types:
        raise ArtifactError(f"unsupported game type scope: {required.game_type!r}")
    for adapter, version in required.hero_adapter_versions.items():
        if manifest.hero_adapter_versions.get(adapter) != version:
            raise ArtifactError(f"incompatible hero adapter {adapter!r} version {version}")


def _validate_state(
    state: object, declared: Mapping[str, ArtifactTensor]
) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ArtifactError("weight state must contain tensors only")
    if set(state) != set(declared):
        raise ArtifactError("tensor state names do not exactly match the manifest")
    typed: dict[str, torch.Tensor] = state
    for name, tensor in typed.items():
        record = declared[name]
        if tuple(tensor.shape) != record.shape:
            raise ArtifactError(f"tensor shape mismatch for {name!r}")
        if _dtype_name(tensor) != record.dtype:
            raise ArtifactError(f"tensor dtype mismatch for {name!r}")
    return typed


def load_model_artifact(
    source: str | Path, *, requirements: RuntimeRequirements
) -> LoadedModelArtifact:
    """Verify an artifact completely before returning an executable model."""

    path = Path(source)
    if not path.is_dir():
        raise ArtifactError(f"artifact path is not a directory: {path}")
    manifest = _read_manifest(path / "manifest.json")
    _validate_files(path, manifest)
    _require_compatible(manifest, requirements)
    try:
        schema_payload = (path / "schema.json").read_bytes()
        schema = TensorFeatureSchema.model_validate_json(schema_payload)
    except (OSError, ValueError) as exc:
        raise ArtifactError("invalid tensor schema") from exc
    if schema_payload != canonical_json_bytes(schema):
        raise ArtifactError("tensor schema is not canonical")
    if (
        schema.schema_id != manifest.tensor_schema_id
        or schema.schema_version != manifest.tensor_schema_version
        or schema.digest != manifest.tensor_schema_digest
        or schema.observation_schema_version != manifest.observation_schema_version
    ):
        raise ArtifactError("tensor schema identity or digest mismatch")
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
        schema=schema,
        config=manifest.architecture_config,
        scope=scope,
        runtime_compatibility_version=manifest.runtime_compatibility_version,
    )
    if _model_digest(metadata, state) != manifest.model_digest:
        raise ArtifactError("executable model digest mismatch")
    try:
        config = JointModelConfig(**cast(dict[str, Any], manifest.architecture_config))
        model = JointPolicyValueModel(schema=schema, config=config)
        model.load_state_dict(state, strict=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ArtifactError("model config or weight state is incompatible") from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedModelArtifact(manifest=manifest, schema=schema, config=config, model=model)


__all__ = [
    "LoadedModelArtifact",
    "export_model_artifact",
    "load_model_artifact",
]
