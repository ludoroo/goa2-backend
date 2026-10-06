"""Complete-current-scope and exact-parent contracts for native Gen1 work."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from automata.models.contracts import CURRENT_MAP_SCHEMA_VERSION, ArtifactError
from automata.models.shared_encoder.artifacts import (
    Gen1ModelArtifactManifest,
    ModelArtifactManifest,
    export_gen1_model_artifact,
)
from automata.models.shared_encoder.gen1_model import Gen1ModelConfig, Gen1PolicyValueModel
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.observation.hero_adapters import HeroObservationAdapterRegistry
from automata.training.native_gen1 import (
    current_gen1_artifact_scope,
    current_gen1_runtime_requirements,
    load_current_gen1_parent_artifact,
    validate_current_gen1_parent_manifest,
)
from goa2.data.heroes import HeroRegistry
from goa2.domain.models import GameType


def _model(*, dtype: torch.dtype = torch.float32) -> Gen1PolicyValueModel:
    decision = TensorFeatureSchema.current()
    value = StableValueTensorSchema.current()
    model = Gen1PolicyValueModel(
        decision_schema=decision,
        stable_value_schema=value,
        config=Gen1ModelConfig(
            decision_schema_digest=decision.digest,
            stable_value_schema_digest=value.digest,
            token_width=8,
            state_width=12,
            candidate_width=8,
            message_passing_layers=1,
        ),
    )
    return model.to(dtype=dtype)


def _export(path: Path, *, dtype: torch.dtype = torch.float32) -> Gen1ModelArtifactManifest:
    return export_gen1_model_artifact(
        path,
        model=_model(dtype=dtype),
        decision_schema=TensorFeatureSchema.current(),
        stable_value_schema=StableValueTensorSchema.current(),
        scope=current_gen1_artifact_scope(),
    )


def test_current_scope_is_sorted_complete_and_uses_effective_adapter_versions() -> None:
    scope = current_gen1_artifact_scope()
    maps_root = Path(__file__).parents[2] / "src" / "goa2" / "data" / "maps"
    registry = HeroObservationAdapterRegistry()
    heroes = tuple(sorted(HeroRegistry.list_heroes()))

    assert scope.supported_heroes == heroes
    assert scope.supported_maps == tuple(
        sorted(path.stem for path in maps_root.glob("*.json") if path.is_file())
    )
    assert scope.supported_game_types == tuple(sorted(item.value for item in GameType))
    assert scope.map_schema_version == CURRENT_MAP_SCHEMA_VERSION
    assert dict(scope.hero_adapter_versions) == {
        "generic": registry.generic_version,
        **{
            hero: registry.registered_versions.get(hero, registry.generic_version)
            for hero in heroes
        },
    }


def test_current_requirements_are_exact_for_requested_scope() -> None:
    scope = current_gen1_artifact_scope()
    heroes = (scope.supported_heroes[-1], scope.supported_heroes[0])

    required = current_gen1_runtime_requirements(
        heroes=heroes,
        map_id=scope.supported_maps[0],
        game_type=scope.supported_game_types[0],
    )

    assert required.heroes == frozenset(heroes)
    assert required.hero_adapter_versions == {
        "generic": scope.hero_adapter_versions["generic"],
        **{hero: scope.hero_adapter_versions[hero] for hero in sorted(heroes)},
    }


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"supported_maps": ("forgotten_island",)}, "scope"),
        ({"supported_maps": (*current_gen1_artifact_scope().supported_maps, "extra")}, "scope"),
        ({"supported_heroes": current_gen1_artifact_scope().supported_heroes[1:]}, "scope"),
        ({"decision_observation_schema_version": 3}, "schema|manifest"),
        ({"decision_tensor_schema_digest": "0" * 64}, "schema"),
        ({"stable_value_tensor_schema_digest": "0" * 64}, "schema"),
        ({"value_semantics": "other"}, "semantics|manifest"),
        ({"architecture_config": {"architecture": "legacy"}}, "config|architecture"),
    ],
)
def test_manifest_validation_rejects_noncurrent_or_non_gen1_contracts(
    tmp_path: Path, change: dict[str, object], match: str
) -> None:
    manifest = _export(tmp_path / "parent")
    unsafe = manifest.model_copy(update=change)

    with pytest.raises((ArtifactError, TypeError, ValueError), match=match):
        validate_current_gen1_parent_manifest(
            unsafe,
            expected_model_digest=manifest.model_digest,
        )


def test_manifest_validation_requires_sorted_full_scope_tuples(tmp_path: Path) -> None:
    manifest = _export(tmp_path / "parent")
    assert len(manifest.supported_maps) > 1
    same_maps_different_order = manifest.model_copy(
        update={"supported_maps": tuple(reversed(manifest.supported_maps))}
    )

    with pytest.raises(ArtifactError, match=r"exact complete current scope"):
        validate_current_gen1_parent_manifest(
            same_maps_different_order,
            expected_model_digest=manifest.model_digest,
        )


def test_manifest_validation_reconstructs_inventory_without_cpu_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _export(tmp_path / "parent")
    parameter_devices: list[frozenset[str]] = []
    original_init = Gen1PolicyValueModel.__init__

    def tracking_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        parameter_devices.append(frozenset(item.device.type for item in self.parameters()))

    monkeypatch.setattr(Gen1PolicyValueModel, "__init__", tracking_init)

    validate_current_gen1_parent_manifest(
        manifest,
        expected_model_digest=manifest.model_digest,
    )

    assert parameter_devices == [frozenset({"meta"})]


def test_manifest_validation_rejects_legacy_manifest_type(tmp_path: Path) -> None:
    manifest = _export(tmp_path / "parent")

    with pytest.raises(TypeError, match=r"Gen1|parent"):
        validate_current_gen1_parent_manifest(
            ModelArtifactManifest.model_construct(),  # type: ignore[arg-type]
            expected_model_digest=manifest.model_digest,
        )


def test_manifest_validation_rejects_wrong_digest_and_tensor_inventory(tmp_path: Path) -> None:
    manifest = _export(tmp_path / "parent")

    with pytest.raises(ValueError, match="digest"):
        validate_current_gen1_parent_manifest(
            manifest,
            expected_model_digest="f" * 64,
        )

    name, tensor = next(iter(manifest.tensors.items()))
    malformed = manifest.model_copy(
        update={"tensors": {**manifest.tensors, name: tensor.model_copy(update={"shape": (1,)})}}
    )
    with pytest.raises(ValueError, match=r"tensor|model"):
        validate_current_gen1_parent_manifest(
            malformed,
            expected_model_digest=manifest.model_digest,
        )


def test_parent_load_is_verified_exact_cpu_float32(tmp_path: Path) -> None:
    manifest = _export(tmp_path / "parent")

    loaded = load_current_gen1_parent_artifact(
        tmp_path / "parent",
        expected_model_digest=manifest.model_digest,
    )

    assert loaded.manifest == manifest
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in loaded.model.parameters())


def test_parent_load_preserves_caller_cpu_rng_on_success_and_rejection(
    tmp_path: Path,
) -> None:
    manifest = _export(tmp_path / "parent")

    before_success = torch.random.get_rng_state().clone()
    load_current_gen1_parent_artifact(
        tmp_path / "parent",
        expected_model_digest=manifest.model_digest,
    )
    assert torch.equal(torch.random.get_rng_state(), before_success)

    wrong_digest = "0" * 64 if manifest.model_digest != "0" * 64 else "f" * 64
    before_rejection = torch.random.get_rng_state().clone()
    with pytest.raises(ArtifactError, match="digest"):
        load_current_gen1_parent_artifact(
            tmp_path / "parent",
            expected_model_digest=wrong_digest,
        )
    assert torch.equal(torch.random.get_rng_state(), before_rejection)


def test_parent_load_rejects_serialized_float64_instead_of_casting(tmp_path: Path) -> None:
    manifest = _export(tmp_path / "parent", dtype=torch.float64)

    with pytest.raises(ArtifactError, match="float32"):
        load_current_gen1_parent_artifact(
            tmp_path / "parent",
            expected_model_digest=manifest.model_digest,
        )


def test_requirements_reject_empty_and_unknown_scope() -> None:
    scope = current_gen1_artifact_scope()
    valid = {
        "heroes": (scope.supported_heroes[0],),
        "map_id": scope.supported_maps[0],
        "game_type": scope.supported_game_types[0],
    }

    for change in (
        {"heroes": ()},
        {"heroes": ("unknown",)},
        {"map_id": "unknown"},
        {"game_type": "unknown"},
    ):
        with pytest.raises((TypeError, ValueError), match=r"scope|hero|map|game type"):
            current_gen1_runtime_requirements(**(valid | change))
