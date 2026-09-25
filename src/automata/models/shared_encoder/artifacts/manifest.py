"""Manifest schema for shared-encoder model artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ArtifactFile(_FrozenModel):
    length: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactTensor(_FrozenModel):
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)


class ModelArtifactManifest(_FrozenModel):
    schema_version: Literal[2] = 2
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_schema_version: int
    map_schema_version: int
    runtime_compatibility_version: int
    hero_adapter_versions: dict[str, int]
    supported_heroes: tuple[str, ...]
    supported_maps: tuple[str, ...]
    supported_game_types: tuple[str, ...]
    tensor_schema_id: str
    tensor_schema_version: int
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


__all__ = ["ArtifactFile", "ArtifactTensor", "ModelArtifactManifest"]
