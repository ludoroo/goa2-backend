"""Architecture-neutral learned-model artifact contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class ArtifactError(ValueError):
    """A learned-model artifact is malformed, corrupted, or incompatible."""


@dataclass(frozen=True, slots=True)
class ArtifactScope:
    supported_heroes: tuple[str, ...]
    supported_maps: tuple[str, ...]
    supported_game_types: tuple[str, ...]
    hero_adapter_versions: Mapping[str, int]
    map_schema_version: int

    def __post_init__(self) -> None:
        for label, values in (
            ("heroes", self.supported_heroes),
            ("maps", self.supported_maps),
            ("game types", self.supported_game_types),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"artifact scope {label} must be nonempty and unique")
        if self.map_schema_version <= 0:
            raise ValueError("map schema version must be positive")
        if not self.hero_adapter_versions or any(
            not name or not isinstance(version, int) or version <= 0
            for name, version in self.hero_adapter_versions.items()
        ):
            raise ValueError("hero adapter versions must be positive integers")


@dataclass(frozen=True, slots=True)
class RuntimeRequirements:
    runtime_compatibility_version: int
    observation_schema_version: int
    map_schema_version: int
    heroes: frozenset[str]
    map_id: str
    game_type: str
    hero_adapter_versions: Mapping[str, int]


__all__ = [
    "ArtifactError",
    "ArtifactScope",
    "RuntimeRequirements",
]
