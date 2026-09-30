"""Artifact contract for the Gen1 policy/stable-value model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from automata.models.contracts import (
    ArtifactError,
    ArtifactScope,
    Gen1RuntimeRequirements,
    canonical_json_bytes,
)
from automata.models.shared_encoder.artifacts import (
    Gen1ModelArtifactManifest,
    export_gen1_model_artifact,
    export_model_artifact,
    load_gen1_model_artifact,
    load_model_artifact,
)
from automata.models.shared_encoder.gen1_model import Gen1ModelConfig, Gen1PolicyValueModel
from automata.models.shared_encoder.model import JointModelConfig, JointPolicyValueModel
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema

HEROES = ("Arien", "Brogan", "Razzle", "Wasp")
ADAPTERS = {"generic": 1, **dict.fromkeys(HEROES, 1)}


def _scope() -> ArtifactScope:
    return ArtifactScope(
        supported_heroes=HEROES,
        supported_maps=("forgotten_island",),
        supported_game_types=("QUICK",),
        hero_adapter_versions=ADAPTERS,
        map_schema_version=1,
    )


def _requirements(**changes: Any) -> Gen1RuntimeRequirements:
    values: dict[str, Any] = {
        "runtime_compatibility_version": 1,
        "decision_observation_schema_version": 4,
        "stable_value_observation_schema_version": 1,
        "graph_observation_schema_version": 2,
        "map_schema_version": 1,
        "heroes": frozenset(HEROES),
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "hero_adapter_versions": ADAPTERS,
    }
    values.update(changes)
    return Gen1RuntimeRequirements(**values)


def _gen1_model(
    decision: TensorFeatureSchema, value: StableValueTensorSchema
) -> Gen1PolicyValueModel:
    torch.manual_seed(17)
    return Gen1PolicyValueModel(
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


def _legacy_model(schema: TensorFeatureSchema) -> JointPolicyValueModel:
    return JointPolicyValueModel(
        schema=schema,
        config=JointModelConfig(
            model_version=2,
            schema_digest=schema.digest,
            token_width=8,
            state_width=12,
            candidate_width=8,
            message_passing_layers=1,
        ),
    )


def _export(path: Path) -> Gen1ModelArtifactManifest:
    decision = TensorFeatureSchema.current()
    value = StableValueTensorSchema.current()
    return export_gen1_model_artifact(
        path,
        model=_gen1_model(decision, value),
        decision_schema=decision,
        stable_value_schema=value,
        scope=_scope(),
        provenance={"run": "gen1-test"},
    )


def test_gen1_artifact_round_trips_both_schemas_config_and_weights(tmp_path: Path) -> None:
    path = tmp_path / "gen1"
    manifest = _export(path)

    loaded = load_gen1_model_artifact(path, requirements=_requirements())

    assert loaded.manifest == manifest
    assert manifest.schema_version == 3
    assert manifest.artifact_kind == "GEN1_POLICY_STABLE_VALUE"
    assert manifest.runtime_compatibility_version == 1
    assert manifest.value_semantics == "stable-boundary-outcome-v1"
    assert loaded.decision_schema == TensorFeatureSchema.current()
    assert loaded.stable_value_schema == StableValueTensorSchema.current()
    assert loaded.config == loaded.model.config
    assert (path / "manifest.json").read_bytes() == canonical_json_bytes(manifest)
    assert {item.name for item in path.iterdir()} == {
        "decision_schema.json",
        "stable_value_schema.json",
        "weights.pt",
        "provenance.json",
        "manifest.json",
    }
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())


def test_gen1_export_preserves_directory_created_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gen1"
    decision = TensorFeatureSchema.current()
    value = StableValueTensorSchema.current()
    original_save = torch.save
    raced_identity: tuple[int, int] | None = None

    def save_after_racer(*args: Any, **kwargs: Any) -> None:
        nonlocal raced_identity
        path.mkdir()
        stat = path.stat()
        raced_identity = (stat.st_dev, stat.st_ino)
        original_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", save_after_racer)

    with pytest.raises(FileExistsError):
        export_gen1_model_artifact(
            path,
            model=_gen1_model(decision, value),
            decision_schema=decision,
            stable_value_schema=value,
            scope=_scope(),
        )

    stat = path.stat()
    assert raced_identity == (stat.st_dev, stat.st_ino)
    assert not tuple(path.iterdir())


def test_gen1_loader_rejects_scope_and_files_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gen1"
    _export(path)
    (path / "unexpected.pkl").write_bytes(b"unsafe")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("loaded weights"))

    with pytest.raises(ArtifactError, match=r"unexpected|allowlist|files"):
        load_gen1_model_artifact(path, requirements=_requirements())


@pytest.mark.parametrize(
    "config_digest",
    ("decision_schema_digest", "stable_value_schema_digest"),
)
def test_gen1_loader_rejects_config_schema_digest_mismatch_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_digest: str
) -> None:
    path = tmp_path / "gen1"
    _export(path)
    manifest = json.loads((path / "manifest.json").read_bytes())
    manifest["architecture_config"][config_digest] = "0" * 64
    incompatible = Gen1ModelArtifactManifest.model_validate(manifest)
    (path / "manifest.json").write_bytes(canonical_json_bytes(incompatible))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("loaded weights"))

    with pytest.raises(ArtifactError, match=r"config|schema|incompatible"):
        load_gen1_model_artifact(path, requirements=_requirements())


def test_legacy_and_gen1_loaders_reject_each_others_artifacts_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen1 = tmp_path / "gen1"
    legacy = tmp_path / "legacy"
    _export(gen1)
    decision = TensorFeatureSchema.current()
    export_model_artifact(
        legacy,
        model=_legacy_model(decision),
        schema=decision,
        scope=_scope(),
        runtime_compatibility_version=2,
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("loaded weights"))

    with pytest.raises(ArtifactError, match=r"manifest|Gen1|gen1|artifact"):
        load_gen1_model_artifact(legacy, requirements=_requirements())
    from automata.models.contracts import RuntimeRequirements

    with pytest.raises(ArtifactError, match=r"manifest|artifact"):
        load_model_artifact(
            gen1,
            requirements=RuntimeRequirements(
                runtime_compatibility_version=2,
                observation_schema_version=4,
                map_schema_version=1,
                heroes=frozenset(HEROES),
                map_id="forgotten_island",
                game_type="QUICK",
                hero_adapter_versions=ADAPTERS,
            ),
        )


def test_gen1_loader_rejects_noncanonical_manifest_and_symlinks(tmp_path: Path) -> None:
    path = tmp_path / "gen1"
    _export(path)
    manifest = json.loads((path / "manifest.json").read_bytes())
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ArtifactError, match="canonical"):
        load_gen1_model_artifact(path, requirements=_requirements())

    linked = tmp_path / "linked"
    linked.symlink_to(path, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        load_gen1_model_artifact(linked, requirements=_requirements())
