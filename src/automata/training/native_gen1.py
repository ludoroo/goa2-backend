"""Complete-current-scope and exact-parent contracts for native Gen1 work."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from automata.models.contracts import (
    CURRENT_MAP_SCHEMA_VERSION,
    GEN1_RUNTIME_COMPATIBILITY_VERSION,
    ArtifactError,
    ArtifactScope,
    Gen1RuntimeRequirements,
)
from automata.models.shared_encoder.artifacts import (
    ArtifactTensor,
    Gen1ModelArtifactManifest,
    LoadedGen1ModelArtifact,
    load_gen1_model_artifact,
)
from automata.models.shared_encoder.gen1_model import (
    GEN1_ARCHITECTURE_ID,
    Gen1ModelConfig,
    Gen1PolicyValueModel,
)
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.observation.hero_adapters import HeroObservationAdapterRegistry
from goa2.data.heroes import HeroRegistry
from goa2.domain.models import GameType

_MODEL_DIGEST_LENGTH = 64
_VALUE_SEMANTICS = "stable-boundary-outcome-v1"


def current_gen1_artifact_scope() -> ArtifactScope:
    """Return the complete, sorted scope supported by the current Gen1 runtime."""
    maps_root = Path(__file__).parents[2] / "goa2" / "data" / "maps"
    heroes = tuple(sorted(HeroRegistry.list_heroes()))
    maps = tuple(sorted(path.stem for path in maps_root.glob("*.json") if path.is_file()))
    game_types = tuple(sorted(game_type.value for game_type in GameType))
    if not heroes or not maps or not game_types:  # pragma: no cover - packaging guard
        raise ValueError("complete current Gen1 runtime scope could not be enumerated")

    registry = HeroObservationAdapterRegistry()
    registered = registry.registered_versions
    adapter_versions = {
        "generic": registry.generic_version,
        **{hero: registered.get(hero, registry.generic_version) for hero in heroes},
    }
    return ArtifactScope(
        supported_heroes=heroes,
        supported_maps=maps,
        supported_game_types=game_types,
        hero_adapter_versions=adapter_versions,
        map_schema_version=CURRENT_MAP_SCHEMA_VERSION,
    )


def current_gen1_runtime_requirements(
    *,
    heroes: Collection[str],
    map_id: str,
    game_type: str,
) -> Gen1RuntimeRequirements:
    """Build exact current runtime requirements for one nonempty game scope."""
    if isinstance(heroes, (str, bytes)) or not isinstance(heroes, Collection):
        raise TypeError("heroes must be a collection of hero names")
    if not isinstance(map_id, str):
        raise TypeError("map_id must be a string")
    if not isinstance(game_type, str):
        raise TypeError("game_type must be a string")

    scope = current_gen1_artifact_scope()
    requested = tuple(heroes)
    if not requested:
        raise ValueError("required Gen1 hero scope cannot be empty")
    if any(not isinstance(hero, str) or not hero for hero in requested):
        raise TypeError("required Gen1 heroes must be nonempty strings")
    unknown_heroes = set(requested) - set(scope.supported_heroes)
    if unknown_heroes:
        raise ValueError(f"unknown current Gen1 hero scope: {sorted(unknown_heroes)!r}")
    if map_id not in scope.supported_maps:
        raise ValueError(f"unknown current Gen1 map scope: {map_id!r}")
    if game_type not in scope.supported_game_types:
        raise ValueError(f"unknown current Gen1 game type scope: {game_type!r}")

    decision_schema = TensorFeatureSchema.current()
    stable_value_schema = StableValueTensorSchema.current()
    requested_heroes = frozenset(requested)
    adapters = {
        "generic": scope.hero_adapter_versions["generic"],
        **{hero: scope.hero_adapter_versions[hero] for hero in sorted(requested_heroes)},
    }
    return Gen1RuntimeRequirements(
        runtime_compatibility_version=GEN1_RUNTIME_COMPATIBILITY_VERSION,
        decision_observation_schema_version=decision_schema.observation_schema_version,
        stable_value_observation_schema_version=stable_value_schema.observation_schema_version,
        graph_observation_schema_version=stable_value_schema.graph_observation_schema_version,
        map_schema_version=CURRENT_MAP_SCHEMA_VERSION,
        heroes=requested_heroes,
        map_id=map_id,
        game_type=game_type,
        hero_adapter_versions=adapters,
    )


def _validated_expected_digest(expected_model_digest: str) -> str:
    if (
        not isinstance(expected_model_digest, str)
        or len(expected_model_digest) != _MODEL_DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_model_digest)
    ):
        raise ValueError("expected Gen1 model digest must be a lowercase SHA-256 digest")
    return expected_model_digest


def _strict_manifest(manifest: Gen1ModelArtifactManifest) -> Gen1ModelArtifactManifest:
    if not isinstance(manifest, Gen1ModelArtifactManifest):
        raise TypeError("current parent must be a Gen1 model artifact manifest")
    try:
        return Gen1ModelArtifactManifest.model_validate(
            manifest.model_dump(mode="python"), strict=True
        )
    except ValidationError as exc:
        raise ArtifactError("invalid current Gen1 parent manifest schema or scope") from exc


def _reconstructed_model_inventory(
    *,
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
    architecture_config: dict[str, Any],
) -> dict[str, ArtifactTensor]:
    try:
        config = Gen1ModelConfig(**architecture_config)
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            model = Gen1PolicyValueModel(
                decision_schema=decision_schema,
                stable_value_schema=stable_value_schema,
                config=config,
            ).to(dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ArtifactError("current Gen1 parent architecture config is invalid") from exc
    return {
        name: ArtifactTensor(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
        )
        for name, tensor in model.state_dict().items()
    }


def validate_current_gen1_parent_manifest(
    manifest: Gen1ModelArtifactManifest,
    *,
    expected_model_digest: str,
) -> Gen1ModelArtifactManifest:
    """Strictly validate an exact full-current-scope Gen1 parent manifest."""
    expected_digest = _validated_expected_digest(expected_model_digest)
    validated = _strict_manifest(manifest)
    if validated.model_digest != expected_digest:
        raise ArtifactError("current Gen1 parent model digest does not match the expected digest")

    scope = current_gen1_artifact_scope()
    decision_schema = TensorFeatureSchema.current()
    stable_value_schema = StableValueTensorSchema.current()
    executable_identity = (
        validated.model_id,
        validated.runtime_compatibility_version,
        validated.decision_observation_schema_version,
        validated.stable_value_observation_schema_version,
        validated.graph_observation_schema_version,
        validated.map_schema_version,
        validated.value_semantics,
    )
    expected_identity = (
        GEN1_ARCHITECTURE_ID,
        GEN1_RUNTIME_COMPATIBILITY_VERSION,
        decision_schema.observation_schema_version,
        stable_value_schema.observation_schema_version,
        stable_value_schema.graph_observation_schema_version,
        CURRENT_MAP_SCHEMA_VERSION,
        _VALUE_SEMANTICS,
    )
    if executable_identity != expected_identity:
        raise ArtifactError("current Gen1 parent executable schema or semantics are incompatible")

    tensor_schema_identity = (
        validated.decision_tensor_schema_id,
        validated.decision_tensor_schema_version,
        validated.decision_tensor_schema_digest,
        validated.stable_value_tensor_schema_id,
        validated.stable_value_tensor_schema_version,
        validated.stable_value_tensor_schema_digest,
    )
    expected_tensor_schema_identity = (
        decision_schema.schema_id,
        decision_schema.schema_version,
        decision_schema.digest,
        stable_value_schema.schema_id,
        stable_value_schema.schema_version,
        stable_value_schema.digest,
    )
    if tensor_schema_identity != expected_tensor_schema_identity:
        raise ArtifactError("current Gen1 parent tensor schema identity is incompatible")

    if (
        validated.supported_heroes != scope.supported_heroes
        or validated.supported_maps != scope.supported_maps
        or validated.supported_game_types != scope.supported_game_types
        or validated.hero_adapter_versions != scope.hero_adapter_versions
    ):
        raise ArtifactError("current Gen1 parent must have the exact complete current scope")

    expected_inventory = _reconstructed_model_inventory(
        decision_schema=decision_schema,
        stable_value_schema=stable_value_schema,
        architecture_config=validated.architecture_config,
    )
    if validated.tensors != expected_inventory:
        if any(
            expected.dtype == "float32"
            and validated.tensors.get(name) is not None
            and validated.tensors[name].dtype != "float32"
            for name, expected in expected_inventory.items()
        ):
            raise ArtifactError("current Gen1 parent serialized weights must be float32")
        raise ArtifactError("current Gen1 parent tensor inventory does not match its model")
    return validated


def load_current_gen1_parent_artifact(
    source: str | Path,
    *,
    expected_model_digest: str,
) -> LoadedGen1ModelArtifact:
    """Safely load and validate an exact CPU-float32 full-current Gen1 parent."""
    expected_digest = _validated_expected_digest(expected_model_digest)
    with torch.random.fork_rng(devices=[]):
        scope = current_gen1_artifact_scope()
        requirements = current_gen1_runtime_requirements(
            heroes=scope.supported_heroes,
            map_id=scope.supported_maps[0],
            game_type=scope.supported_game_types[0],
        )
        loaded = load_gen1_model_artifact(source, requirements=requirements)
        validate_current_gen1_parent_manifest(
            loaded.manifest,
            expected_model_digest=expected_digest,
        )
        if any(
            parameter.device.type != "cpu" or parameter.dtype != torch.float32
            for parameter in loaded.model.parameters()
        ):
            raise ArtifactError("current Gen1 parent model parameters must be CPU float32")
        if any(
            buffer.device.type != "cpu"
            or (buffer.is_floating_point() and buffer.dtype != torch.float32)
            for buffer in loaded.model.buffers()
        ):
            raise ArtifactError("current Gen1 parent model buffers must be CPU float32")
        return loaded


__all__ = [
    "current_gen1_artifact_scope",
    "current_gen1_runtime_requirements",
    "load_current_gen1_parent_artifact",
    "validate_current_gen1_parent_manifest",
]
