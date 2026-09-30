"""Neutral inference contracts for the Gen1 policy/stable-value runtime."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from automata.models.contracts import (
    GEN1_RUNTIME_COMPATIBILITY_VERSION,
    Gen1RuntimeRequirements,
    LearnedModelOutput,
    LearnedPolicyOutput,
    LearnedPolicyRuntime,
    LearnedStableValueOutput,
    LearnedStableValueRuntime,
    UnitCandidateID,
)


def _candidate(value: str) -> UnitCandidateID:
    return UnitCandidateID(schema_version=1, unit_id=value)


def test_gen1_outputs_are_minimal_strict_and_finite() -> None:
    policy = LearnedPolicyOutput(
        candidate_ids=(_candidate("hero_arien"),),
        policy_logits=(1.25,),
    )
    value = LearnedStableValueOutput(value=-0.5)

    assert policy.model_dump() == {
        "candidate_ids": ({"schema_version": 1, "kind": "UNIT", "unit_id": "hero_arien"},),
        "policy_logits": (1.25,),
    }
    assert value.model_dump() == {"value": -0.5}

    for payload in (
        {"candidate_ids": (), "policy_logits": ()},
        {
            "candidate_ids": (_candidate("hero_arien"), _candidate("hero_arien")),
            "policy_logits": (0.0, 1.0),
        },
        {"candidate_ids": (_candidate("hero_arien"),), "policy_logits": ()},
        {"candidate_ids": (_candidate("hero_arien"),), "policy_logits": (math.inf,)},
        {
            "candidate_ids": (_candidate("hero_arien"),),
            "policy_logits": (0.0,),
            "value": 0.0,
        },
    ):
        with pytest.raises(ValidationError):
            LearnedPolicyOutput.model_validate(payload)
    for payload in ({"value": math.nan}, {"value": 1.1}, {"value": 0.0, "candidate_ids": []}):
        with pytest.raises(ValidationError):
            LearnedStableValueOutput.model_validate(payload)


def test_gen1_runtime_protocols_are_runtime_checkable_and_capability_specific() -> None:
    class PolicyOnly:
        def evaluate_policy(self, observation: object) -> LearnedPolicyOutput:
            raise NotImplementedError

    class ValueOnly:
        def evaluate_stable_value(self, observation: object) -> LearnedStableValueOutput:
            raise NotImplementedError

    assert isinstance(PolicyOnly(), LearnedPolicyRuntime)
    assert not isinstance(PolicyOnly(), LearnedStableValueRuntime)
    assert isinstance(ValueOnly(), LearnedStableValueRuntime)
    assert not isinstance(ValueOnly(), LearnedPolicyRuntime)


def test_gen1_requirements_are_separate_from_legacy_runtime_version() -> None:
    requirements = Gen1RuntimeRequirements(
        runtime_compatibility_version=GEN1_RUNTIME_COMPATIBILITY_VERSION,
        decision_observation_schema_version=4,
        stable_value_observation_schema_version=1,
        graph_observation_schema_version=2,
        map_schema_version=1,
        heroes=frozenset({"Arien", "Wasp"}),
        map_id="forgotten_island",
        game_type="QUICK",
        hero_adapter_versions={"generic": 1, "Arien": 1, "Wasp": 1},
    )

    assert requirements.runtime_compatibility_version == 1


def test_legacy_shared_encoder_has_an_additive_policy_wrapper() -> None:
    from automata.models.shared_encoder.runtime import SharedEncoderRuntime

    candidate = _candidate("hero_arien")
    legacy = LearnedModelOutput(
        candidate_ids=(candidate,),
        policy_logits=(2.0,),
        probabilities=(1.0,),
        value=0.25,
    )
    runtime = object.__new__(SharedEncoderRuntime)
    runtime.evaluate = lambda observation: legacy  # type: ignore[method-assign]

    assert runtime.evaluate_policy(object()) == LearnedPolicyOutput(  # type: ignore[arg-type]
        candidate_ids=(candidate,), policy_logits=(2.0,)
    )
