"""CPU inference behavior for the Gen1 shared-encoder runtime."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from automata.decision import DecisionDescriptor
from automata.models.contracts import (
    ArtifactError,
    ArtifactScope,
    Gen1RuntimeRequirements,
    LearnedPolicyOutput,
    LearnedStableValueOutput,
)
from automata.models.shared_encoder import gen1_runtime as gen1_runtime_module
from automata.models.shared_encoder.artifacts import export_gen1_model_artifact
from automata.models.shared_encoder.gen1_model import Gen1ModelConfig, Gen1PolicyValueModel
from automata.models.shared_encoder.gen1_runtime import Gen1SharedEncoderRuntime
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.observation import encode_decision
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.value_boundary import detect_stable_value_boundary
from automata.search.contracts import SearchContext
from automata.search.ismcts.engine import legal_keys
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import CardState, GamePhase, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.phases import resolve_next_action
from goa2.engine.setup import GameSetup

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))
HEROES = ("Arien", "Brogan", "Razzle", "Wasp")
ADAPTERS = {"generic": 1, **dict.fromkeys(HEROES, 1)}


def _state() -> Any:
    return GameSetup.create_game(
        MAP, ["Razzle", "Wasp"], ["Arien", "Brogan"], game_type="QUICK", seed=31
    )


def _request(values: Iterable[str], *, player_id: str = "hero_razzle") -> InputRequest:
    return InputRequest(
        id="request-id",
        request_type=InputRequestType.SELECT_UNIT,
        player_id=player_id,
        options=[InputOption.from_value(value) for value in values],
    )


def _policy_observation(values: Iterable[str] = ("hero_arien", "hero_wasp")):
    state = _state()
    decision = DecisionDescriptor("INPUT", request=_request(values))
    return encode_decision(
        state,
        decision,
        legal_keys(decision),
        decision_owner_hero_id="hero_razzle",
        perspective_team="RED",
    )


def _allied_policy_observation():
    state = _state()
    decision = DecisionDescriptor(
        "INPUT",
        request=_request(["hero_arien", "hero_brogan"], player_id="hero_wasp"),
    )
    return encode_decision(
        state,
        decision,
        legal_keys(decision),
        decision_owner_hero_id="hero_razzle",
        perspective_team="RED",
        context=SearchContext(
            root_viewer_id="hero_razzle",
            perspective_team=TeamColor.RED,
            current_owner_id="hero_wasp",
            decision=decision,
        ),
    )


def _stable_observation(
    *, viewer_hero_id: str = "hero_razzle", perspective_team: TeamColor = TeamColor.RED
):
    state = _state()
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    return encode_stable_value(
        state,
        boundary,
        viewer_hero_id=viewer_hero_id,
        perspective_team=perspective_team,
    )


def _actor_ready_observation():
    state = _state()
    actor = state.get_hero(HeroID("hero_wasp"))
    assert actor is not None and actor.hand
    card = actor.hand.pop()
    card.state = CardState.UNRESOLVED
    card.is_facedown = False
    actor.current_turn_card = card
    state.phase = GamePhase.RESOLUTION
    state.unresolved_hero_ids = [actor.id]
    resolve_next_action(state)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None and boundary.kind.value == "ACTOR_READY"
    return encode_stable_value(
        state,
        boundary,
        viewer_hero_id="hero_razzle",
        perspective_team=TeamColor.RED,
    )


def _replace_token_feature(
    observation: Any,
    *,
    kind: str,
    feature: str,
    value: object,
    hero_id: str | None = None,
):
    tokens = list(observation.state.tokens)
    index = next(
        index
        for index, token in enumerate(tokens)
        if token.kind == kind and (hero_id is None or token.features.get("hero_id") == hero_id)
    )
    token = tokens[index]
    tokens[index] = token.model_copy(update={"features": {**token.features, feature: value}})
    return observation.model_copy(
        update={"state": observation.state.model_copy(update={"tokens": tuple(tokens)})}
    )


def _requirements() -> Gen1RuntimeRequirements:
    return Gen1RuntimeRequirements(
        runtime_compatibility_version=1,
        decision_observation_schema_version=4,
        stable_value_observation_schema_version=1,
        graph_observation_schema_version=2,
        map_schema_version=1,
        heroes=frozenset(HEROES),
        map_id="forgotten_island",
        game_type="QUICK",
        hero_adapter_versions=ADAPTERS,
    )


def _model() -> Gen1PolicyValueModel:
    decision = TensorFeatureSchema.current()
    value = StableValueTensorSchema.current()
    torch.manual_seed(11)
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


def _runtime(
    model: Gen1PolicyValueModel | None = None,
    *,
    device: str | torch.device = "cpu",
) -> Gen1SharedEncoderRuntime:
    return Gen1SharedEncoderRuntime(
        model=model or _model(),
        decision_schema=TensorFeatureSchema.current(),
        stable_value_schema=StableValueTensorSchema.current(),
        requirements=_requirements(),
        supported_heroes=HEROES,
        supported_maps=("forgotten_island",),
        supported_game_types=("QUICK",),
        device=device,
    )


def test_gen1_runtime_exposes_separate_policy_and_stable_value_inference() -> None:
    runtime = _runtime()
    policy_observation = _policy_observation()
    stable_observation = _stable_observation()

    policy = runtime.evaluate_policy(policy_observation)
    value = runtime.evaluate_stable_value(stable_observation)

    assert isinstance(policy, LearnedPolicyOutput)
    assert policy.candidate_ids == tuple(
        candidate.candidate_id for candidate in policy_observation.candidates
    )
    assert len(policy.policy_logits) == len(policy.candidate_ids)
    assert isinstance(value, LearnedStableValueOutput)
    assert -1.0 <= value.value <= 1.0
    assert not hasattr(runtime, "evaluate")
    assert runtime.model.training is False
    assert all(parameter.grad is None for parameter in runtime.model.parameters())


def test_gen1_runtime_loads_verified_artifact(tmp_path: Path) -> None:
    path = tmp_path / "gen1"
    export_gen1_model_artifact(
        path,
        model=_model(),
        decision_schema=TensorFeatureSchema.current(),
        stable_value_schema=StableValueTensorSchema.current(),
        scope=ArtifactScope(
            supported_heroes=HEROES,
            supported_maps=("forgotten_island",),
            supported_game_types=("QUICK",),
            hero_adapter_versions=ADAPTERS,
            map_schema_version=1,
        ),
    )

    runtime = Gen1SharedEncoderRuntime.from_artifact(path, requirements=_requirements())

    assert isinstance(runtime.evaluate_policy(_policy_observation()), LearnedPolicyOutput)
    assert isinstance(
        runtime.evaluate_stable_value(_stable_observation()), LearnedStableValueOutput
    )


def test_gen1_runtime_rejects_private_view_or_scope_mismatch_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    observation = _policy_observation()
    viewer = observation.state.viewer.model_copy(update={"private_hero_id": "hero_arien"})
    changed = observation.model_copy(
        update={"state": observation.state.model_copy(update={"viewer": viewer})}
    )
    monkeypatch.setattr(
        runtime.model,
        "forward_policy",
        lambda *args: pytest.fail("model forward should not run"),
    )

    with pytest.raises(ArtifactError, match=r"viewer|SELF|private|team"):
        runtime.evaluate_policy(changed)


def test_gen1_runtime_rejects_wrong_stable_boundary_flags_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    observation = _stable_observation()
    hero_index = next(
        index for index, token in enumerate(observation.state.tokens) if token.kind == "HERO"
    )
    hero = observation.state.tokens[hero_index]
    tokens = list(observation.state.tokens)
    tokens[hero_index] = hero.model_copy(
        update={"features": {**hero.features, "is_current_actor": True}}
    )
    changed = observation.model_copy(
        update={"state": observation.state.model_copy(update={"tokens": tuple(tokens)})}
    )
    monkeypatch.setattr(
        runtime.model,
        "forward_stable_value",
        lambda *args: pytest.fail("model forward should not run"),
    )

    with pytest.raises(ArtifactError, match=r"PLANNING_READY|actor|owner"):
        runtime.evaluate_stable_value(changed)


def test_gen1_runtime_single_and_batch_inference_match_with_ragged_candidates() -> None:
    runtime = _runtime()
    policy_observations = (
        _policy_observation(["hero_arien"]),
        _policy_observation(["hero_arien", "hero_wasp", "hero_brogan"]),
    )
    stable_observations = (
        _stable_observation(),
        _stable_observation(viewer_hero_id="hero_arien", perspective_team=TeamColor.BLUE),
    )

    batched_policy = runtime.evaluate_policy_batch(policy_observations)
    batched_value = runtime.evaluate_stable_value_batch(stable_observations)

    for observation, batched in zip(policy_observations, batched_policy, strict=True):
        single = runtime.evaluate_policy(observation)
        assert batched.candidate_ids == single.candidate_ids
        assert batched.policy_logits == pytest.approx(single.policy_logits, abs=1e-6)
    for observation, batched in zip(stable_observations, batched_value, strict=True):
        single = runtime.evaluate_stable_value(observation)
        assert batched.value == pytest.approx(single.value, abs=1e-6)


def test_gen1_runtime_accepts_allied_continuation_owner_distinct_from_self() -> None:
    runtime = _runtime()
    observation = _allied_policy_observation()
    heroes = {
        token.features["hero_id"]: token
        for token in observation.state.tokens
        if token.kind == "HERO"
    }

    output = runtime.evaluate_policy(observation)

    assert heroes["hero_razzle"].features["relation"] == "SELF"
    assert heroes["hero_razzle"].features["is_decision_owner"] is False
    assert heroes["hero_wasp"].features["relation"] == "ALLY"
    assert heroes["hero_wasp"].features["is_decision_owner"] is True
    assert output.candidate_ids == tuple(
        candidate.candidate_id for candidate in observation.candidates
    )


@pytest.mark.parametrize(
    ("kind", "feature", "value", "message"),
    [
        ("GLOBAL", "map_id", "other_map", "map"),
        ("GLOBAL", "game_type", "STANDARD", "game type"),
        ("HERO", "name", "Emmitt", "hero roster|required scope"),
    ],
)
def test_gen1_runtime_rejects_nonexact_observation_scope_before_forward(
    kind: str,
    feature: str,
    value: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    changed = _replace_token_feature(_policy_observation(), kind=kind, feature=feature, value=value)
    monkeypatch.setattr(
        runtime.model,
        "forward_policy",
        lambda *args: pytest.fail("model forward should not run"),
    )

    with pytest.raises(ArtifactError, match=message):
        runtime.evaluate_policy(changed)


def test_gen1_runtime_rejects_actor_ready_with_different_actor_and_owner_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    observation = _actor_ready_observation()
    changed = _replace_token_feature(
        observation,
        kind="HERO",
        hero_id="hero_wasp",
        feature="is_decision_owner",
        value=False,
    )
    changed = _replace_token_feature(
        changed,
        kind="HERO",
        hero_id="hero_razzle",
        feature="is_decision_owner",
        value=True,
    )
    monkeypatch.setattr(
        runtime.model,
        "forward_stable_value",
        lambda *args: pytest.fail("model forward should not run"),
    )

    with pytest.raises(ArtifactError, match=r"ACTOR_READY|matching actor and owner"):
        runtime.evaluate_stable_value(changed)


def test_gen1_runtime_rejects_malformed_and_nonfinite_model_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    policy_observation = _policy_observation(["hero_arien", "hero_wasp"])
    stable_observation = _stable_observation()

    monkeypatch.setattr(
        runtime.model,
        "forward_policy",
        lambda _batch: SimpleNamespace(policy_logits=torch.zeros((1, 3))),
    )
    with pytest.raises(ArtifactError, match=r"policy output.*shape"):
        runtime.evaluate_policy(policy_observation)

    monkeypatch.setattr(
        runtime.model,
        "forward_policy",
        lambda _batch: SimpleNamespace(policy_logits=torch.full((1, 2), float("nan"))),
    )
    with pytest.raises(ArtifactError, match="finite logits"):
        runtime.evaluate_policy(policy_observation)

    monkeypatch.setattr(
        runtime.model,
        "forward_stable_value",
        lambda _batch: SimpleNamespace(value=torch.zeros((1, 1))),
    )
    with pytest.raises(ArtifactError, match=r"stable-value output.*shape"):
        runtime.evaluate_stable_value(stable_observation)

    monkeypatch.setattr(
        runtime.model,
        "forward_stable_value",
        lambda _batch: SimpleNamespace(value=torch.tensor([float("inf")])),
    )
    with pytest.raises(ArtifactError, match="finite values"):
        runtime.evaluate_stable_value(stable_observation)


def test_gen1_native_stable_inference_is_independent_of_policy_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()

    def reject_policy_path(*args: object, **kwargs: object) -> None:
        raise AssertionError("native stable inference called the policy capability")

    monkeypatch.setattr(gen1_runtime_module, "collate_decisions", reject_policy_path)
    monkeypatch.setattr(runtime.model, "forward_policy", reject_policy_path)

    output = runtime.evaluate_stable_value(_stable_observation())

    assert isinstance(output, LearnedStableValueOutput)
    assert -1.0 <= output.value <= 1.0


def test_gen1_runtime_rejects_non_cpu_device() -> None:
    with pytest.raises(ArtifactError, match="only CPU"):
        _runtime(device="cuda")
