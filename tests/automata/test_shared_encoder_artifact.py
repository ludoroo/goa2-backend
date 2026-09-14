"""Behavioral contract for learned-model artifact export, loading, and CPU inference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from automata.decision import DecisionDescriptor as Decision
from automata.models.contracts import (
    DecisionObservation,
    canonical_json_bytes,
    from_canonical_json,
)
from automata.models.contracts.artifacts import (
    ArtifactScope,
    RuntimeRequirements,
)
from automata.models.contracts.inference import LearnedModelOutput
from automata.models.shared_encoder.artifacts import (
    export_model_artifact,
    load_model_artifact,
)
from automata.models.shared_encoder.artifacts.manifest import ModelArtifactManifest
from automata.models.shared_encoder.batching import collate_decisions, masked_softmax
from automata.models.shared_encoder.model import JointModelConfig, JointPolicyValueModel
from automata.models.shared_encoder.runtime import SharedEncoderRuntime
from automata.models.shared_encoder.schema import TensorFeatureSchema
from automata.observation import encode_decision
from automata.search.ismcts.engine import legal_keys
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.engine.setup import GameSetup

MAP = Path("src/goa2/data/maps/forgotten_island.json")
HEROES = ("Arien", "Brogan", "Razzle", "Wasp")
ADAPTER_VERSIONS = {"generic": 1, "Arien": 1, "Brogan": 1, "Razzle": 1, "Wasp": 1}


def _request(values: Iterable[str]) -> InputRequest:
    return InputRequest(
        id="request-id",
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_razzle",
        prompt="choose",
        options=[InputOption.from_value(value) for value in values],
    )


def _observation(*, small: bool = False) -> DecisionObservation:
    state = GameSetup.create_game(
        str(MAP),
        ["Razzle"] if small else ["Razzle", "Wasp"],
        ["Arien"] if small else ["Arien", "Brogan"],
        game_type="QUICK",
        seed=31,
    )
    candidates = ["hero_arien"] if small else ["hero_arien", "hero_razzle_piece_1"]
    decision = Decision("INPUT", request=_request(candidates))
    return encode_decision(
        state,
        decision,
        legal_keys(decision),
        decision_owner_hero_id="hero_razzle",
        perspective_team="RED",
    )


@pytest.fixture(scope="module")
def schema() -> TensorFeatureSchema:
    return TensorFeatureSchema.current()


def _config(schema: TensorFeatureSchema) -> JointModelConfig:
    return JointModelConfig(
        model_version=1,
        schema_digest=schema.digest,
        token_width=8,
        state_width=12,
        candidate_width=8,
        message_passing_layers=1,
        dropout=0.0,
    )


def _model(schema: TensorFeatureSchema) -> JointPolicyValueModel:
    torch.manual_seed(73)
    return JointPolicyValueModel(schema=schema, config=_config(schema))


def _scope(**changes: Any) -> ArtifactScope:
    values: dict[str, Any] = {
        "supported_heroes": HEROES,
        "supported_maps": ("forgotten_island",),
        "supported_game_types": ("QUICK",),
        "hero_adapter_versions": ADAPTER_VERSIONS,
        "map_schema_version": 1,
    }
    values.update(changes)
    return ArtifactScope(**values)


def _requirements(**changes: Any) -> RuntimeRequirements:
    values: dict[str, Any] = {
        "runtime_compatibility_version": 1,
        "observation_schema_version": 3,
        "map_schema_version": 1,
        "heroes": frozenset(HEROES),
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "hero_adapter_versions": ADAPTER_VERSIONS,
    }
    values.update(changes)
    return RuntimeRequirements(**values)


def _export(
    path: Path,
    schema: TensorFeatureSchema,
    *,
    scope: ArtifactScope | None = None,
    provenance: dict[str, Any] | None = None,
) -> ModelArtifactManifest:
    return export_model_artifact(
        path,
        model=_model(schema),
        schema=schema,
        scope=scope or _scope(),
        runtime_compatibility_version=1,
        provenance=provenance,
    )


def _manifest_data(path: Path) -> dict[str, Any]:
    value = json.loads((path / "manifest.json").read_bytes())
    assert isinstance(value, dict)
    return value


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    (path / "manifest.json").write_text(
        json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ),
        encoding="utf-8",
    )


def test_export_is_canonical_content_addressed_and_does_not_overwrite(
    tmp_path: Path, schema: TensorFeatureSchema
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    provenance = {"training_run": "run-7", "metrics": {"loss": 0.25}}

    first = _export(first_path, schema, provenance=provenance)
    second = _export(second_path, schema, provenance=provenance)

    assert first.schema_version == 2
    assert first.model_digest == second.model_digest
    assert (first_path / "manifest.json").read_bytes() == canonical_json_bytes(first)
    assert (second_path / "manifest.json").read_bytes() == canonical_json_bytes(second)
    assert (
        from_canonical_json(ModelArtifactManifest, (first_path / "manifest.json").read_bytes())
        == first
    )
    assert set(path.name for path in first_path.iterdir()) == {
        "manifest.json",
        "schema.json",
        "weights.pt",
        "provenance.json",
    }
    assert json.loads((first_path / "schema.json").read_bytes()) == schema.model_dump(mode="json")
    manifest_data = _manifest_data(first_path)
    assert manifest_data["architecture_config"] == asdict(_config(schema))
    assert manifest_data["tensor_schema_id"] == schema.schema_id
    assert manifest_data["tensor_schema_version"] == schema.schema_version
    assert manifest_data["tensor_schema_digest"] == schema.digest

    with pytest.raises(FileExistsError):
        _export(first_path, schema, provenance=provenance)


def test_provenance_is_non_executable_but_executable_metadata_changes_digest(
    tmp_path: Path, schema: TensorFeatureSchema
) -> None:
    baseline = _export(tmp_path / "baseline", schema, provenance={"loss": 0.8})
    changed_provenance = _export(tmp_path / "provenance", schema, provenance={"loss": 0.1})
    changed_scope = _export(
        tmp_path / "scope",
        schema,
        scope=_scope(supported_game_types=("QUICK", "STANDARD")),
        provenance={"loss": 0.8},
    )

    assert changed_provenance.model_digest == baseline.model_digest
    assert changed_scope.model_digest != baseline.model_digest
    assert _manifest_data(tmp_path / "baseline")["non_executable_files"] == ["provenance.json"]


def test_loader_returns_exact_schema_config_and_state_dict_only_model(
    tmp_path: Path,
    schema: TensorFeatureSchema,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact"
    manifest = _export(path, schema)
    real_load = torch.load
    calls: list[dict[str, Any]] = []

    def observed_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observed_load)
    loaded = load_model_artifact(path, requirements=_requirements())

    assert loaded.manifest == manifest
    assert loaded.schema == schema
    assert loaded.config == _config(schema)
    assert isinstance(loaded.model, JointPolicyValueModel)
    assert calls and all(call.get("weights_only") is True for call in calls)
    persisted = real_load(path / "weights.pt", map_location="cpu", weights_only=True)
    assert isinstance(persisted, dict)
    assert persisted and all(isinstance(name, str) for name in persisted)
    assert all(isinstance(value, torch.Tensor) for value in persisted.values())


@pytest.mark.parametrize("filename", ["schema.json", "weights.pt"])
def test_file_byte_or_length_mutation_fails_integrity_check(
    tmp_path: Path, schema: TensorFeatureSchema, filename: str
) -> None:
    path = tmp_path / filename.replace(".", "-")
    _export(path, schema)
    target = path / filename
    target.write_bytes(target.read_bytes() + b"mutation")

    with pytest.raises(ValueError, match=r"hash|length|integrity"):
        load_model_artifact(path, requirements=_requirements())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("architecture_config", {"token_width": 99}, r"digest|config"),
        ("hero_adapter_versions", {"generic": 99}, r"digest|adapter"),
        ("supported_maps", ["vexing_cliffs"], r"digest|map|scope"),
        ("tensor_schema_id", "future-schema", r"digest|schema"),
        ("tensor_schema_version", 2, r"digest|schema|version"),
        ("tensor_schema_digest", "0" * 64, r"digest|schema"),
    ],
)
def test_executable_manifest_mutation_fails_closed(
    tmp_path: Path,
    schema: TensorFeatureSchema,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    path = tmp_path / field
    _export(path, schema)
    manifest = _manifest_data(path)
    if field == "architecture_config":
        manifest[field].update(replacement)
    else:
        manifest[field] = replacement
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match=message):
        load_model_artifact(path, requirements=_requirements())


def test_schema_vocabulary_and_normalizer_declarations_are_executable(
    tmp_path: Path, schema: TensorFeatureSchema
) -> None:
    for mutation in ("vocabulary", "normalization"):
        path = tmp_path / mutation
        _export(path, schema)
        data = json.loads((path / "schema.json").read_bytes())
        if mutation == "vocabulary":
            data["tokens"][0]["categorical"][0]["vocabulary"].append("future_map")
        else:
            data["tokens"][0]["numeric"][0]["normalization"] = "SIGNED_LOG"
        (path / "schema.json").write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match=r"schema|hash|integrity|digest"):
            load_model_artifact(path, requirements=_requirements())


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype", "value"])
def test_tensor_inventory_must_exactly_match_the_declared_model(
    tmp_path: Path,
    schema: TensorFeatureSchema,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    path = tmp_path / change
    _export(path, schema)
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    name = next(iter(state))
    changed = dict(state)
    if change == "missing":
        changed.pop(name)
    elif change == "extra":
        changed["unexpected.weight"] = torch.zeros(1)
    elif change == "shape":
        changed[name] = changed[name].reshape(-1)[:1]
    elif change == "dtype":
        changed[name] = changed[name].to(torch.float64)
    else:
        changed[name] = changed[name] + 1

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError, match=r"tensor|state|shape|dtype|weight|digest"):
        load_model_artifact(path, requirements=_requirements())


def test_unexpected_files_are_rejected_before_any_weight_loading(
    tmp_path: Path,
    schema: TensorFeatureSchema,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact"
    _export(path, schema)
    (path / "model.pkl").write_bytes(b"not allowed")
    called = False

    def forbidden_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("weights must not load before file validation")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match=r"unexpected|allow|file"):
        load_model_artifact(path, requirements=_requirements())
    assert called is False


@pytest.mark.parametrize(
    "changes",
    [
        {"runtime_compatibility_version": 2},
        {"observation_schema_version": 2},
        {"map_schema_version": 2},
        {"heroes": frozenset({*HEROES, "FutureHero"})},
        {"map_id": "vexing_cliffs"},
        {"game_type": "STANDARD"},
        {"hero_adapter_versions": {**ADAPTER_VERSIONS, "Razzle": 2}},
    ],
)
def test_load_rejects_runtime_or_scope_mismatch_before_constructing_runtime(
    tmp_path: Path, schema: TensorFeatureSchema, changes: dict[str, Any]
) -> None:
    path = tmp_path / next(iter(changes))
    _export(path, schema)

    with pytest.raises(ValueError, match=r"incompatible|unsupported|scope|version|schema|adapter"):
        load_model_artifact(path, requirements=_requirements(**changes))


def test_runtime_matches_in_memory_model_and_excludes_batch_padding(
    tmp_path: Path, schema: TensorFeatureSchema
) -> None:
    path = tmp_path / "artifact"
    model = _model(schema)
    export_model_artifact(
        path,
        model=model,
        schema=schema,
        scope=_scope(),
        runtime_compatibility_version=1,
    )
    observations = (_observation(), _observation(small=True))
    batch = collate_decisions(observations, schema=schema, training=False)
    model.eval()
    with torch.inference_mode():
        expected = model(batch)
        probabilities = masked_softmax(expected.policy_logits, batch.candidates.mask)

    runtime = SharedEncoderRuntime.from_artifact(path, requirements=_requirements())
    actual = runtime.evaluate_batch(observations)

    assert isinstance(actual, tuple) and len(actual) == 2
    assert all(isinstance(item, LearnedModelOutput) for item in actual)
    for index, item in enumerate(actual):
        count = len(observations[index].candidates)
        assert item.candidate_ids == tuple(
            candidate.candidate_id for candidate in observations[index].candidates
        )
        assert len(item.policy_logits) == len(item.probabilities) == count
        assert item.policy_logits == pytest.approx(
            expected.policy_logits[index, :count].tolist(), abs=1e-6
        )
        assert item.probabilities == pytest.approx(probabilities[index, :count].tolist(), abs=1e-6)
        assert sum(item.probabilities) == pytest.approx(1.0)
        assert -1.0 <= item.value <= 1.0


@pytest.mark.parametrize("bad_logit", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_runtime_rejects_non_finite_model_policy_output(
    tmp_path: Path,
    schema: TensorFeatureSchema,
    monkeypatch: pytest.MonkeyPatch,
    bad_logit: float,
) -> None:
    path = tmp_path / "artifact"
    _export(path, schema)
    runtime = SharedEncoderRuntime.from_artifact(path, requirements=_requirements())
    original_forward = runtime.model.forward

    def invalid_forward(batch):
        output = original_forward(batch)
        logits = output.policy_logits.clone()
        logits[0, 0] = bad_logit
        return output.__class__(policy_logits=logits, value=output.value)

    monkeypatch.setattr(runtime.model, "forward", invalid_forward)

    with pytest.raises(ValueError, match="learned model scores must be finite"):
        runtime.evaluate(_observation(small=True))


def test_runtime_rejects_non_finite_model_value_output(
    tmp_path: Path,
    schema: TensorFeatureSchema,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact"
    _export(path, schema)
    runtime = SharedEncoderRuntime.from_artifact(path, requirements=_requirements())
    original_forward = runtime.model.forward

    def invalid_forward(batch):
        output = original_forward(batch)
        return output.__class__(
            policy_logits=output.policy_logits,
            value=torch.full_like(output.value, float("inf")),
        )

    monkeypatch.setattr(runtime.model, "forward", invalid_forward)

    with pytest.raises(ValueError, match=r"value must be finite and in \[-1, 1\]"):
        runtime.evaluate(_observation(small=True))


def test_runtime_single_inference_is_deterministic_eval_only_and_gradient_free(
    tmp_path: Path, schema: TensorFeatureSchema
) -> None:
    path = tmp_path / "artifact"
    _export(path, schema)
    runtime = SharedEncoderRuntime.from_artifact(path, requirements=_requirements())
    observation = _observation(small=True)
    runtime.model.train()

    first = runtime.evaluate(observation)
    second = runtime.evaluate(observation)

    assert isinstance(first, LearnedModelOutput)
    assert first == second
    assert runtime.device.type == "cpu"
    assert runtime.model.training is False
    assert all(parameter.grad is None for parameter in runtime.model.parameters())


@pytest.mark.parametrize("mismatch", ["map", "game_type", "hero", "schema"])
def test_observation_scope_mismatch_fails_before_model_forward(
    tmp_path: Path,
    schema: TensorFeatureSchema,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    path = tmp_path / mismatch
    _export(path, schema)
    runtime = SharedEncoderRuntime.from_artifact(path, requirements=_requirements())
    observation = _observation(small=True)
    if mismatch == "schema":
        observation = observation.model_copy(update={"schema_version": 999})
    else:
        token_index = next(
            index
            for index, token in enumerate(observation.state.tokens)
            if token.kind == ("HERO" if mismatch == "hero" else "GLOBAL")
        )
        token = observation.state.tokens[token_index]
        field = {"map": "map_id", "game_type": "game_type", "hero": "name"}[mismatch]
        value = {"map": "vexing_cliffs", "game_type": "STANDARD", "hero": "FutureHero"}[mismatch]
        tokens = list(observation.state.tokens)
        tokens[token_index] = token.model_copy(
            update={"features": {**token.features, field: value}}
        )
        observation = observation.model_copy(
            update={"state": observation.state.model_copy(update={"tokens": tuple(tokens)})}
        )

    called = False

    def forbidden_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("model must not run for incompatible observations")

    monkeypatch.setattr(runtime.model, "forward", forbidden_forward)
    with pytest.raises(ValueError, match=r"schema|map|game type|hero|unsupported|incompatible"):
        runtime.evaluate(observation)
    assert called is False


def test_explicit_artifact_and_runtime_modules_are_the_only_new_torch_boundaries() -> None:
    script = textwrap.dedent("""
        import sys

        import automata.models
        import automata.observation
        from goa2.server.app import create_app

        assert create_app() is not None
        assert "torch" not in sys.modules

        import automata.models.shared_encoder.artifacts
        import automata.models.shared_encoder.runtime
        assert "torch" in sys.modules
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
