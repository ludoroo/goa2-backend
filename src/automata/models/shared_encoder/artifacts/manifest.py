"""Manifest schema for shared-encoder model artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ..schema import (
    StableValueTensorSchemaID,
    StableValueTensorSchemaVersion,
    TensorSchemaID,
    TensorSchemaVersion,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ArtifactFile(_FrozenModel):
    length: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactTensor(_FrozenModel):
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)


class Gen1ModelArtifactManifest(_FrozenModel):
    """Closed Gen1 artifact inventory with separate decision and value schemas."""

    schema_version: Literal[3] = 3
    artifact_kind: Literal["GEN1_POLICY_STABLE_VALUE"] = "GEN1_POLICY_STABLE_VALUE"
    model_id: Literal["goa2-gen1-policy-stable-value-v1"] = "goa2-gen1-policy-stable-value-v1"
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_compatibility_version: Literal[1] = 1
    decision_observation_schema_version: Literal[4] = 4
    stable_value_observation_schema_version: Literal[1] = 1
    graph_observation_schema_version: Literal[2] = 2
    map_schema_version: int
    value_semantics: Literal["stable-boundary-outcome-v1"] = "stable-boundary-outcome-v1"
    hero_adapter_versions: dict[str, int]
    supported_heroes: tuple[str, ...]
    supported_maps: tuple[str, ...]
    supported_game_types: tuple[str, ...]
    decision_tensor_schema_id: TensorSchemaID
    decision_tensor_schema_version: TensorSchemaVersion
    decision_tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stable_value_tensor_schema_id: StableValueTensorSchemaID
    stable_value_tensor_schema_version: StableValueTensorSchemaVersion
    stable_value_tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    architecture_config: dict[str, JsonValue]
    tensors: dict[str, ArtifactTensor]
    files: dict[str, ArtifactFile]
    non_executable_files: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_inventory_and_scope(self) -> Gen1ModelArtifactManifest:
        if self.map_schema_version <= 0:
            raise ValueError("map schema version must be positive")
        for label, values in (
            ("heroes", self.supported_heroes),
            ("maps", self.supported_maps),
            ("game types", self.supported_game_types),
        ):
            if (
                not values
                or len(values) != len(set(values))
                or any(not value or value != value.strip() for value in values)
            ):
                raise ValueError(f"artifact scope {label} must be nonempty, unique strings")
        if (
            "generic" not in self.hero_adapter_versions
            or any(hero not in self.hero_adapter_versions for hero in self.supported_heroes)
            or any(
                not name
                or name != name.strip()
                or isinstance(version, bool)
                or not isinstance(version, int)
                or version <= 0
                for name, version in self.hero_adapter_versions.items()
            )
        ):
            raise ValueError("hero adapter scope is incomplete or invalid")
        if not self.architecture_config:
            raise ValueError("architecture config cannot be empty")
        if not self.tensors or any(
            any(dimension < 0 for dimension in tensor.shape) for tensor in self.tensors.values()
        ):
            raise ValueError("artifact tensor inventory is empty or invalid")
        required = {"decision_schema.json", "stable_value_schema.json", "weights.pt"}
        allowed = required | {"provenance.json"}
        if set(self.files) not in (required, allowed):
            raise ValueError("artifact file inventory is not the Gen1 allowlist")
        expected_non_executable = ("provenance.json",) if "provenance.json" in self.files else ()
        if self.non_executable_files != expected_non_executable:
            raise ValueError("non-executable file inventory is invalid")
        return self


class ModelArtifactManifest(_FrozenModel):
    schema_version: Literal[2] = 2
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_schema_version: Literal[4]
    map_schema_version: int
    runtime_compatibility_version: Literal[2]
    hero_adapter_versions: dict[str, int]
    supported_heroes: tuple[str, ...]
    supported_maps: tuple[str, ...]
    supported_game_types: tuple[str, ...]
    tensor_schema_id: TensorSchemaID
    tensor_schema_version: TensorSchemaVersion
    tensor_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    architecture_config: dict[str, JsonValue]
    tensors: dict[str, ArtifactTensor]
    files: dict[str, ArtifactFile]
    non_executable_files: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_inventory(self) -> ModelArtifactManifest:
        if not self.tensors:
            raise ValueError("artifact tensor inventory cannot be empty")
        if set(self.non_executable_files) - set(self.files):
            raise ValueError("non-executable file is absent from file inventory")
        return self


__all__ = [
    "ArtifactFile",
    "ArtifactTensor",
    "Gen1ModelArtifactManifest",
    "ModelArtifactManifest",
]
