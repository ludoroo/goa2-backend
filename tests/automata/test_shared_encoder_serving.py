"""Serving contracts for artifact-pinned learned-model decisions."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from automata.decision import DecisionDescriptor as Decision
from automata.models.contracts.artifacts import ArtifactScope, RuntimeRequirements
from automata.models.shared_encoder.artifacts import (
    export_model_artifact,
)
from automata.models.shared_encoder.batching import collate_decisions
from automata.models.shared_encoder.model import JointModelConfig, JointPolicyValueModel
from automata.models.shared_encoder.schema import TensorFeatureSchema
from automata.observation import encode_decision
from automata.search.ismcts.engine import legal_keys
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.engine.setup import GameSetup

MAP = "src/goa2/data/maps/forgotten_island.json"
HEROES = ("Arien", "Razzle")
ADAPTERS = {"generic": 1, "Arien": 1, "Razzle": 1}


def _requirements() -> RuntimeRequirements:
    return RuntimeRequirements(
        runtime_compatibility_version=1,
        observation_schema_version=3,
        map_schema_version=1,
        heroes=frozenset(HEROES),
        map_id="forgotten_island",
        game_type="QUICK",
        hero_adapter_versions=ADAPTERS,
    )


def _observation():
    state = GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)
    request = InputRequest(
        id="pick",
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_razzle",
        prompt="pick",
        options=[
            InputOption.from_value("hero_arien"),
            InputOption.from_value("hero_razzle_piece_1"),
        ],
    )
    decision = Decision("INPUT", request=request)
    return encode_decision(
        state,
        decision,
        legal_keys(decision),
        decision_owner_hero_id="hero_razzle",
        perspective_team="RED",
    )


def test_pytorch_serving_cache_preserves_exported_model_output_and_identity(tmp_path: Path) -> None:
    from automata.models.shared_encoder.serving import RuntimeCache

    schema = TensorFeatureSchema.current()
    torch.manual_seed(9)
    model = JointPolicyValueModel(
        schema=schema,
        config=JointModelConfig(
            model_version=1,
            schema_digest=schema.digest,
            token_width=8,
            state_width=12,
            candidate_width=8,
            message_passing_layers=1,
            dropout=0.0,
        ),
    )
    model.eval()
    artifact = tmp_path / "artifact"
    manifest = export_model_artifact(
        artifact,
        model=model,
        schema=schema,
        scope=ArtifactScope(
            supported_heroes=HEROES,
            supported_maps=("forgotten_island",),
            supported_game_types=("QUICK",),
            hero_adapter_versions=ADAPTERS,
            map_schema_version=1,
        ),
        runtime_compatibility_version=1,
    )
    observation = _observation()
    with torch.inference_mode():
        training_output = model(collate_decisions([observation], schema=schema, training=False))

    cache = RuntimeCache()
    with ThreadPoolExecutor(max_workers=4) as pool:
        runtimes = list(
            pool.map(lambda _: cache.get(artifact, requirements=_requirements()), range(8))
        )
    first = runtimes[0]
    served = first.evaluate(observation)

    assert all(runtime is first for runtime in runtimes)
    assert served.policy_logits == pytest.approx(training_output.policy_logits[0].tolist())
    assert served.value == pytest.approx(training_output.value[0].item())

    assert (
        cache.get(
            artifact,
            requirements=_requirements(),
            expected_digest=manifest.model_digest,
        )
        is first
    )
    with pytest.raises(ValueError, match="pinned digest"):
        cache.get(
            artifact,
            requirements=_requirements(),
            expected_digest="0" * 64,
        )

    with (artifact / "weights.pt").open("ab") as stream:
        stream.write(b"mutated")
    with pytest.raises(ValueError, match="mutated"):
        cache.get(artifact, requirements=_requirements())


def test_runtime_cache_does_not_poison_a_failed_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import automata.models.shared_encoder.serving as serving

    artifact = tmp_path / "retryable"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps({"model_digest": "a" * 64}))
    sentinel = object()
    calls = 0

    def flaky_loader(cls, path, *, requirements, device):
        nonlocal calls
        del cls, path, requirements, device
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return sentinel

    monkeypatch.setattr(serving.SharedEncoderRuntime, "from_artifact", classmethod(flaky_loader))
    cache = serving.RuntimeCache()

    with pytest.raises(RuntimeError, match="temporary"):
        cache.get(artifact, requirements=_requirements())

    assert cache.get(artifact, requirements=_requirements()) is sentinel
    assert calls == 2
