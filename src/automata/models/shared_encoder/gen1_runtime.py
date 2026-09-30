"""CPU inference runtime for separate Gen1 policy and stable-value heads."""

from __future__ import annotations

from pathlib import Path

import torch

from ..contracts import (
    ArtifactError,
    DecisionObservation,
    Gen1RuntimeRequirements,
    LearnedObservation,
    LearnedPolicyOutput,
    LearnedStableValueOutput,
    StableValueObservation,
)
from .artifacts import load_gen1_model_artifact
from .batching import collate_decisions, collate_stable_values
from .gen1_model import Gen1PolicyValueModel
from .schema import StableValueTensorSchema, TensorFeatureSchema


class Gen1SharedEncoderRuntime:
    """Artifact-pinned Gen1 inference with isolated policy and value entry points."""

    def __init__(
        self,
        *,
        model: Gen1PolicyValueModel,
        decision_schema: TensorFeatureSchema,
        stable_value_schema: StableValueTensorSchema,
        requirements: Gen1RuntimeRequirements,
        supported_heroes: tuple[str, ...],
        supported_maps: tuple[str, ...],
        supported_game_types: tuple[str, ...],
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ArtifactError("only CPU Gen1 inference is supported")
        self.model = model.to(self.device)
        self.decision_schema = decision_schema
        self.stable_value_schema = stable_value_schema
        self.requirements = requirements
        self.supported_heroes = frozenset(supported_heroes)
        self.supported_maps = frozenset(supported_maps)
        self.supported_game_types = frozenset(supported_game_types)
        self._validate_runtime_identity()

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        requirements: Gen1RuntimeRequirements,
        device: str | torch.device = "cpu",
    ) -> Gen1SharedEncoderRuntime:
        loaded = load_gen1_model_artifact(path, requirements=requirements)
        return cls(
            model=loaded.model,
            decision_schema=loaded.decision_schema,
            stable_value_schema=loaded.stable_value_schema,
            requirements=requirements,
            supported_heroes=loaded.manifest.supported_heroes,
            supported_maps=loaded.manifest.supported_maps,
            supported_game_types=loaded.manifest.supported_game_types,
            device=device,
        )

    def evaluate_policy(self, observation: DecisionObservation) -> LearnedPolicyOutput:
        return self.evaluate_policy_batch((observation,))[0]

    def evaluate_policy_batch(
        self, observations: tuple[DecisionObservation, ...] | list[DecisionObservation]
    ) -> tuple[LearnedPolicyOutput, ...]:
        for observation in observations:
            self._validate_policy_observation(observation)
        if not observations:
            raise ArtifactError("Gen1 policy batch cannot be empty")
        batch = collate_decisions(observations, schema=self.decision_schema, training=False)
        self.model.eval()
        with torch.inference_mode():
            output = self.model.forward_policy(batch)
        logits = output.policy_logits
        if not isinstance(logits, torch.Tensor) or logits.shape != batch.candidates.mask.shape:
            raise ArtifactError("Gen1 policy output has an incompatible shape")
        if not bool(torch.isfinite(logits).all()):
            raise ArtifactError("Gen1 policy output must contain only finite logits")
        results: list[LearnedPolicyOutput] = []
        for index, candidate_ids in enumerate(batch.candidate_ids):
            count = len(candidate_ids)
            results.append(
                LearnedPolicyOutput(
                    candidate_ids=candidate_ids,
                    policy_logits=tuple(logits[index, :count].tolist()),
                )
            )
        return tuple(results)

    def evaluate_stable_value(
        self, observation: StableValueObservation
    ) -> LearnedStableValueOutput:
        return self.evaluate_stable_value_batch((observation,))[0]

    def evaluate_stable_value_batch(
        self,
        observations: tuple[StableValueObservation, ...] | list[StableValueObservation],
    ) -> tuple[LearnedStableValueOutput, ...]:
        for observation in observations:
            self._validate_stable_value_observation(observation)
        if not observations:
            raise ArtifactError("Gen1 stable-value batch cannot be empty")
        batch = collate_stable_values(observations, schema=self.stable_value_schema)
        self.model.eval()
        with torch.inference_mode():
            output = self.model.forward_stable_value(batch)
        values = output.value
        expected_shape = (len(observations),)
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != expected_shape:
            raise ArtifactError("Gen1 stable-value output has an incompatible shape")
        if not bool(torch.isfinite(values).all()):
            raise ArtifactError("Gen1 stable-value output must contain only finite values")
        return tuple(LearnedStableValueOutput(value=float(value.item())) for value in values)

    def _validate_runtime_identity(self) -> None:
        required = self.requirements
        versions = (
            required.runtime_compatibility_version == 1,
            required.decision_observation_schema_version
            == self.decision_schema.observation_schema_version,
            required.stable_value_observation_schema_version
            == self.stable_value_schema.observation_schema_version,
            required.graph_observation_schema_version
            == self.stable_value_schema.graph_observation_schema_version,
        )
        if not all(versions):
            raise ArtifactError("Gen1 runtime requirements and schemas are incompatible")
        if self.model.config.decision_schema_digest != self.decision_schema.digest or (
            self.model.config.stable_value_schema_digest != self.stable_value_schema.digest
        ):
            raise ArtifactError("Gen1 model and tensor schemas are incompatible")
        if self.decision_schema.tokens != self.stable_value_schema.tokens or (
            self.decision_schema.relationships != self.stable_value_schema.relationships
        ):
            raise ArtifactError("Gen1 decision and stable-value schemas have different graphs")
        if not required.heroes or not required.heroes <= self.supported_heroes:
            raise ArtifactError("Gen1 runtime has unsupported required hero scope")
        if required.map_id not in self.supported_maps:
            raise ArtifactError("Gen1 runtime has unsupported required map scope")
        if required.game_type not in self.supported_game_types:
            raise ArtifactError("Gen1 runtime has unsupported required game type scope")

    def _validate_policy_observation(self, observation: DecisionObservation) -> None:
        if observation.schema_version != self.decision_schema.observation_schema_version:
            raise ArtifactError("incompatible decision observation schema version")
        if not observation.candidates:
            raise ArtifactError("policy observation must contain at least one candidate")
        heroes = self._validate_private_graph(observation.state)
        owners = [token for token in heroes if token.features.get("is_decision_owner") is True]
        if len(owners) != 1:
            raise ArtifactError("policy observation must identify exactly one decision owner")
        owner = owners[0]
        viewer_team = observation.state.viewer.perspective_team
        if owner.features.get("team_id") != viewer_team or owner.features.get("relation") not in {
            "SELF",
            "ALLY",
        }:
            raise ArtifactError("decision owner must be SELF or an allied continuation owner")

    def _validate_stable_value_observation(self, observation: StableValueObservation) -> None:
        if observation.schema_version != self.stable_value_schema.observation_schema_version:
            raise ArtifactError("incompatible stable-value observation schema version")
        heroes = self._validate_private_graph(observation.state)
        actors = [token for token in heroes if token.features.get("is_current_actor") is True]
        owners = [token for token in heroes if token.features.get("is_decision_owner") is True]
        if observation.boundary_kind == "ACTOR_READY":
            if len(actors) != 1 or len(owners) != 1 or actors[0].local_ref != owners[0].local_ref:
                raise ArtifactError(
                    "ACTOR_READY observation must identify one matching actor and owner"
                )
        elif actors or owners:
            raise ArtifactError("PLANNING_READY observation cannot identify an actor or owner")

    def _validate_private_graph(self, state: LearnedObservation):
        if state.schema_version != self.requirements.graph_observation_schema_version:
            raise ArtifactError("incompatible graph observation schema version")
        globals_ = [token for token in state.tokens if token.kind == "GLOBAL"]
        if len(globals_) != 1:
            raise ArtifactError("observation must contain exactly one GLOBAL token")
        global_features = globals_[0].features
        if global_features.get("map_id") != self.requirements.map_id:
            raise ArtifactError("observation map does not match runtime requirements")
        if global_features.get("game_type") != self.requirements.game_type:
            raise ArtifactError("observation game type does not match runtime requirements")

        heroes = [token for token in state.tokens if token.kind == "HERO"]
        hero_ids = [token.features.get("hero_id") for token in heroes]
        hero_names = [token.features.get("name") for token in heroes]
        if (
            any(not isinstance(hero_id, str) or not hero_id for hero_id in hero_ids)
            or len(hero_ids) != len(set(hero_ids))
            or any(not isinstance(name, str) or not name for name in hero_names)
            or len(hero_names) != len(set(hero_names))
            or set(hero_names) != set(self.requirements.heroes)
        ):
            raise ArtifactError("observation hero roster does not exactly match required scope")

        viewer = state.viewer
        if (
            viewer.schema_version != 2
            or not viewer.private_hero_id
            or viewer.perspective_team not in {"RED", "BLUE"}
        ):
            raise ArtifactError("observation requires a private viewer and perspective team")
        self_tokens = [token for token in heroes if token.features.get("relation") == "SELF"]
        if len(self_tokens) != 1:
            raise ArtifactError("observation must identify exactly one SELF viewer")
        self_token = self_tokens[0]
        if (
            self_token.features.get("hero_id") != viewer.private_hero_id
            or self_token.features.get("team_id") != viewer.perspective_team
        ):
            raise ArtifactError("private viewer, SELF hero, and perspective team do not match")
        return heroes


__all__ = ["Gen1SharedEncoderRuntime"]
