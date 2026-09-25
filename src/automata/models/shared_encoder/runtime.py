"""Deterministic inference for the shared-encoder learned model."""

from __future__ import annotations

from pathlib import Path

import torch

from ..contracts import ArtifactError, DecisionObservation, LearnedModelOutput, RuntimeRequirements
from .artifacts import load_model_artifact
from .batching import collate_decisions, masked_softmax
from .model import JointPolicyValueModel
from .schema import TensorFeatureSchema


class SharedEncoderRuntime:
    """CPU-by-default, artifact-pinned policy/value inference."""

    def __init__(
        self,
        *,
        model: JointPolicyValueModel,
        schema: TensorFeatureSchema,
        requirements: RuntimeRequirements,
        supported_heroes: tuple[str, ...],
        supported_maps: tuple[str, ...],
        supported_game_types: tuple[str, ...],
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.schema = schema
        self.requirements = requirements
        self.supported_heroes = frozenset(supported_heroes)
        self.supported_maps = frozenset(supported_maps)
        self.supported_game_types = frozenset(supported_game_types)

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        requirements: RuntimeRequirements,
        device: str | torch.device = "cpu",
    ) -> SharedEncoderRuntime:
        loaded = load_model_artifact(path, requirements=requirements)
        return cls(
            model=loaded.model,
            schema=loaded.schema,
            requirements=requirements,
            supported_heroes=loaded.manifest.supported_heroes,
            supported_maps=loaded.manifest.supported_maps,
            supported_game_types=loaded.manifest.supported_game_types,
            device=device,
        )

    def evaluate(self, observation: DecisionObservation) -> LearnedModelOutput:
        return self.evaluate_batch((observation,))[0]

    def evaluate_batch(
        self, observations: tuple[DecisionObservation, ...] | list[DecisionObservation]
    ) -> tuple[LearnedModelOutput, ...]:
        for observation in observations:
            self._validate_observation(observation)
        batch = collate_decisions(observations, schema=self.schema, training=False)
        if self.device.type != "cpu":
            raise ArtifactError("only CPU learned-model inference is currently supported")
        self.model.eval()
        with torch.inference_mode():
            output = self.model(batch)
            probabilities = masked_softmax(output.policy_logits, batch.candidates.mask)
        results: list[LearnedModelOutput] = []
        for index, candidate_ids in enumerate(batch.candidate_ids):
            count = len(candidate_ids)
            results.append(
                LearnedModelOutput(
                    candidate_ids=candidate_ids,
                    policy_logits=tuple(output.policy_logits[index, :count].tolist()),
                    probabilities=tuple(probabilities[index, :count].tolist()),
                    value=float(output.value[index].item()),
                )
            )
        return tuple(results)

    def _validate_observation(self, observation: DecisionObservation) -> None:
        if observation.schema_version != self.schema.observation_schema_version:
            raise ArtifactError("incompatible observation schema version")
        map_ids: set[str] = set()
        game_types: set[str] = set()
        heroes: set[str] = set()
        for token in observation.state.tokens:
            if token.kind == "GLOBAL":
                map_id = token.features.get("map_id")
                game_type = token.features.get("game_type")
                if isinstance(map_id, str):
                    map_ids.add(map_id)
                if isinstance(game_type, str):
                    game_types.add(game_type)
            elif token.kind == "HERO":
                hero = token.features.get("name")
                if isinstance(hero, str):
                    heroes.add(hero)
        if len(map_ids) != 1 or not map_ids <= self.supported_maps:
            raise ArtifactError(f"unsupported map in observation: {sorted(map_ids)!r}")
        if len(game_types) != 1 or not game_types <= self.supported_game_types:
            raise ArtifactError(f"unsupported game type in observation: {sorted(game_types)!r}")
        unsupported_heroes = heroes - self.supported_heroes
        if unsupported_heroes:
            raise ArtifactError(f"unsupported hero in observation: {sorted(unsupported_heroes)!r}")


__all__ = ["SharedEncoderRuntime"]
