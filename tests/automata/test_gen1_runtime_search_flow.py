"""Native artifacts must supply both real policy and stable-boundary search inference."""

from pathlib import Path

import torch

from automata.agents.heuristic_agent import HeuristicAgent
from automata.models.contracts import (
    ArtifactScope,
    DecisionObservation,
    Gen1RuntimeRequirements,
    LearnedPolicyOutput,
    LearnedStableValueOutput,
    StableValueObservation,
)
from automata.models.shared_encoder.artifacts import export_gen1_model_artifact
from automata.models.shared_encoder.gen1_model import Gen1ModelConfig, Gen1PolicyValueModel
from automata.models.shared_encoder.gen1_runtime import Gen1SharedEncoderRuntime
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode
from automata.search.ismcts import RootTarget, search
from automata.search.learned import LearnedSearchPolicy, LearnedStableValueEvaluator
from goa2.domain.models import TeamColor
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def test_published_native_artifact_drives_policy_and_completed_transition_values(
    tmp_path: Path,
) -> None:
    register_all_effects()
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=73,
    )
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    hero.hand[:] = [card for card in hero.hand if card.id in {"lift_up", "shock"}]
    legal = tuple(card.id for card in hero.hand)
    decision_schema = TensorFeatureSchema.current()
    value_schema = StableValueTensorSchema.current()
    torch.manual_seed(81)
    model = Gen1PolicyValueModel(
        decision_schema=decision_schema,
        stable_value_schema=value_schema,
        config=Gen1ModelConfig(
            decision_schema_digest=decision_schema.digest,
            stable_value_schema_digest=value_schema.digest,
            token_width=8,
            state_width=12,
            candidate_width=8,
            message_passing_layers=1,
        ),
    )
    adapters = {"generic": 1, "Wasp": 1, "Arien": 1}
    artifact = tmp_path / "native-model"
    export_gen1_model_artifact(
        artifact,
        model=model,
        decision_schema=decision_schema,
        stable_value_schema=value_schema,
        scope=ArtifactScope(
            supported_heroes=("Wasp", "Arien"),
            supported_maps=("forgotten_island",),
            supported_game_types=("QUICK",),
            map_schema_version=1,
            hero_adapter_versions=adapters,
        ),
    )
    runtime = Gen1SharedEncoderRuntime.from_artifact(
        artifact,
        requirements=Gen1RuntimeRequirements(
            runtime_compatibility_version=1,
            decision_observation_schema_version=4,
            stable_value_observation_schema_version=1,
            graph_observation_schema_version=2,
            heroes=frozenset({"Wasp", "Arien"}),
            map_id="forgotten_island",
            game_type="QUICK",
            map_schema_version=1,
            hero_adapter_versions=adapters,
        ),
    )
    policy_observations: list[DecisionObservation] = []
    value_observations: list[StableValueObservation] = []

    class RecordingRuntime:
        def evaluate_policy(self, observation: DecisionObservation) -> LearnedPolicyOutput:
            policy_observations.append(observation)
            return runtime.evaluate_policy(observation)

        def evaluate_stable_value(
            self, observation: StableValueObservation
        ) -> LearnedStableValueOutput:
            value_observations.append(observation)
            return runtime.evaluate_stable_value(observation)

    recording = RecordingRuntime()
    before = state.model_dump_json()
    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(4),
        SearchConfig(iterations=3, leaf_mode=LeafMode.STABLE_TRANSITION, seed=9),
        prior=LearnedSearchPolicy(recording),
        root_target=RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id})),
        leaf_evaluator=LearnedStableValueEvaluator(recording),
    )

    assert result.root.visits == len(value_observations) == 3
    assert policy_observations
    assert 0.0 <= result.root.q <= 1.0
    assert state.model_dump_json() == before
    for observation in (*policy_observations, *value_observations):
        assert observation.state.viewer.private_hero_id == "hero_wasp"
        assert observation.state.viewer.perspective_team == "RED"
        assert not any(
            token.kind == "CARD"
            and token.features["owner_ref"] == "hero:hero_arien"
            and token.features["area"] == "hand"
            for token in observation.state.tokens
        )
    assert all(observation.boundary_kind == "ACTOR_READY" for observation in value_observations)
    assert any(
        token.kind == "HERO"
        and token.features["hero_id"] == "hero_arien"
        and token.features["is_current_actor"]
        for observation in value_observations
        for token in observation.state.tokens
    )
