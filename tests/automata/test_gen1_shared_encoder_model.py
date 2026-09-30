"""Behavioral contract for the gen1 policy/stable-value shared encoder."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from automata.decision import DecisionDescriptor as Decision
from automata.models.contracts import DecisionObservation, StableValueObservation
from automata.models.shared_encoder.batching import (
    DecisionBatch,
    StableValueBatch,
    collate_decisions,
    collate_stable_values,
)
from automata.models.shared_encoder.gen1_model import (
    GEN1_ARCHITECTURE_ID,
    Gen1ModelConfig,
    Gen1PolicyValueModel,
    PolicyHeadOutput,
    StableValueHeadOutput,
)
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.observation import encode_decision
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.value_boundary import detect_stable_value_boundary
from automata.search.ismcts.engine import legal_keys
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import TeamColor
from goa2.engine.setup import GameSetup

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))


def _state(*, small: bool = False) -> Any:
    return GameSetup.create_game(
        MAP,
        ["Razzle"] if small else ["Razzle", "Wasp"],
        ["Arien"] if small else ["Arien", "Brogan"],
        game_type="QUICK",
        seed=31,
    )


def _request(values: Iterable[str]) -> InputRequest:
    return InputRequest(
        id="request-id",
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_razzle",
        prompt="choose",
        options=[InputOption.from_value(value) for value in values],
    )


def _decision_observation(*, small: bool = False) -> DecisionObservation:
    state = _state(small=small)
    candidates = ["hero_arien"] if small else ["hero_arien", "hero_razzle_piece_1"]
    decision = Decision("INPUT", request=_request(candidates))
    return encode_decision(
        state,
        decision,
        legal_keys(decision),
        decision_owner_hero_id="hero_razzle",
        perspective_team="RED",
    )


def _stable_observation(*, small: bool = False) -> StableValueObservation:
    state = _state(small=small)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    return encode_stable_value(
        state,
        boundary,
        viewer_hero_id="hero_razzle",
        perspective_team=TeamColor.RED,
    )


@pytest.fixture
def decision_schema() -> TensorFeatureSchema:
    return TensorFeatureSchema.current()


@pytest.fixture
def stable_value_schema() -> StableValueTensorSchema:
    return StableValueTensorSchema.current()


def _config(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
    **changes: object,
) -> Gen1ModelConfig:
    values: dict[str, object] = {
        "decision_schema_digest": decision_schema.digest,
        "stable_value_schema_digest": stable_value_schema.digest,
        "token_width": 16,
        "state_width": 24,
        "candidate_width": 16,
        "message_passing_layers": 2,
        "dropout": 0.0,
    }
    values.update(changes)
    return Gen1ModelConfig(**values)  # type: ignore[arg-type]


def _model(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
    *,
    seed: int = 7,
) -> Gen1PolicyValueModel:
    torch.manual_seed(seed)
    model = Gen1PolicyValueModel(
        decision_schema=decision_schema,
        stable_value_schema=stable_value_schema,
        config=_config(decision_schema, stable_value_schema),
    )
    model.eval()
    return model


def _decision_batch(
    schema: TensorFeatureSchema, observations: list[DecisionObservation]
) -> DecisionBatch:
    return collate_decisions(observations, schema=schema, training=True)


def _stable_batch(
    schema: StableValueTensorSchema, observations: list[StableValueObservation]
) -> StableValueBatch:
    return collate_stable_values(observations, schema=schema)


def test_config_has_pinned_identity_and_strict_validation(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    config = _config(decision_schema, stable_value_schema)

    assert config.architecture_id == GEN1_ARCHITECTURE_ID
    assert config.architecture_version == 1
    assert config.decision_schema_id == decision_schema.schema_id
    assert config.decision_schema_version == decision_schema.schema_version
    assert config.stable_value_schema_id == stable_value_schema.schema_id
    assert config.stable_value_schema_version == stable_value_schema.schema_version

    invalid_identity = {
        "architecture_id": "other-model",
        "architecture_version": 2,
        "decision_schema_id": "goa2-tensor-features-v1",
        "decision_schema_version": 1,
        "stable_value_schema_id": "other-value-schema",
        "stable_value_schema_version": 2,
    }
    for field, value in invalid_identity.items():
        with pytest.raises(ValueError, match="identity"):
            _config(decision_schema, stable_value_schema, **{field: value})
    for field in (
        "architecture_version",
        "decision_schema_version",
        "stable_value_schema_version",
    ):
        for value in (True, float(getattr(config, field))):
            with pytest.raises(ValueError, match="identity"):
                _config(decision_schema, stable_value_schema, **{field: value})
    for field in ("token_width", "state_width", "candidate_width", "message_passing_layers"):
        for value in (True, 0, -1, 1.5):
            with pytest.raises(ValueError, match=field):
                _config(decision_schema, stable_value_schema, **{field: value})
    for value in (float("nan"), float("inf"), -0.1, 1.0):
        with pytest.raises(ValueError, match="dropout"):
            _config(decision_schema, stable_value_schema, dropout=value)
    with pytest.raises(ValueError, match="digest"):
        _config(decision_schema, stable_value_schema, decision_schema_digest="0")


def test_constructor_verifies_both_schemas_and_matching_graph_declarations(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    with pytest.raises(ValueError, match="decision schema"):
        Gen1PolicyValueModel(
            decision_schema=decision_schema,
            stable_value_schema=stable_value_schema,
            config=_config(
                decision_schema,
                stable_value_schema,
                decision_schema_digest="0" * 64,
            ),
        )
    with pytest.raises(ValueError, match="stable-value schema"):
        Gen1PolicyValueModel(
            decision_schema=decision_schema,
            stable_value_schema=stable_value_schema,
            config=_config(
                decision_schema,
                stable_value_schema,
                stable_value_schema_digest="0" * 64,
            ),
        )

    mismatched_graph = stable_value_schema.model_copy(
        update={"tokens": tuple(reversed(stable_value_schema.tokens))}
    )
    with pytest.raises(ValueError, match="graph"):
        Gen1PolicyValueModel(
            decision_schema=decision_schema,
            stable_value_schema=mismatched_graph,
            config=_config(decision_schema, stable_value_schema),
        )


def test_separate_forwards_have_expected_shapes_and_bounded_value(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    decisions = [_decision_observation(), _decision_observation(small=True)]
    values = [_stable_observation(), _stable_observation(small=True)]
    decision_batch = _decision_batch(decision_schema, decisions)
    stable_batch = _stable_batch(stable_value_schema, values)
    model = _model(decision_schema, stable_value_schema)

    policy = model.forward_policy(decision_batch)
    stable_value = model.forward_stable_value(stable_batch)

    assert isinstance(policy, PolicyHeadOutput)
    assert isinstance(stable_value, StableValueHeadOutput)
    assert policy.policy_logits.shape == decision_batch.candidates.mask.shape
    assert stable_value.value.shape == (len(values),)
    assert torch.isfinite(policy.policy_logits[decision_batch.candidates.mask]).all()
    assert torch.isfinite(stable_value.value).all()
    assert ((stable_value.value >= -1.0) & (stable_value.value <= 1.0)).all()
    assert set(policy.__dict__) == {"policy_logits"}
    assert set(stable_value.__dict__) == {"value"}
    with pytest.raises(NotImplementedError):
        model(decision_batch)


def test_batched_and_single_forwards_are_identical(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    decisions = [_decision_observation(), _decision_observation(small=True)]
    values = [_stable_observation(), _stable_observation(small=True)]
    model = _model(decision_schema, stable_value_schema)
    combined_policy = model.forward_policy(_decision_batch(decision_schema, decisions))
    combined_value = model.forward_stable_value(_stable_batch(stable_value_schema, values))

    for index, observation in enumerate(decisions):
        single = model.forward_policy(_decision_batch(decision_schema, [observation]))
        assert torch.allclose(
            combined_policy.policy_logits[index, : single.policy_logits.shape[1]],
            single.policy_logits[0],
            atol=1e-6,
        )
    for index, observation in enumerate(values):
        single = model.forward_stable_value(_stable_batch(stable_value_schema, [observation]))
        assert torch.allclose(combined_value.value[index], single.value[0], atol=1e-6)


def test_candidate_and_graph_storage_permutations_preserve_semantics(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    decision = _decision_observation()
    candidate_permutation = (1, 0)
    permuted_candidates = decision.model_copy(
        update={
            "candidates": tuple(decision.candidates[index] for index in candidate_permutation),
            "state": decision.state.model_copy(
                update={
                    "tokens": tuple(reversed(decision.state.tokens)),
                    "relationships": tuple(reversed(decision.state.relationships)),
                }
            ),
        }
    )
    value = _stable_observation()
    permuted_value = value.model_copy(
        update={
            "state": value.state.model_copy(
                update={
                    "tokens": tuple(reversed(value.state.tokens)),
                    "relationships": tuple(reversed(value.state.relationships)),
                }
            )
        }
    )
    model = _model(decision_schema, stable_value_schema)

    policy = model.forward_policy(_decision_batch(decision_schema, [decision, permuted_candidates]))
    stable = model.forward_stable_value(_stable_batch(stable_value_schema, [value, permuted_value]))

    assert torch.allclose(
        policy.policy_logits[0],
        policy.policy_logits[1, list(candidate_permutation)],
        atol=1e-6,
    )
    assert torch.allclose(stable.value[0], stable.value[1], atol=1e-6)


def _poison_feature_padding(table: Any) -> Any:
    numeric = table.numeric.clone()
    categorical = table.categorical.clone()
    references = table.references.clone()
    numeric[~table.mask] = float("nan")
    categorical[~table.mask] = torch.iinfo(torch.int64).max
    references[~table.mask] = torch.iinfo(torch.int64).max
    return replace(table, numeric=numeric, categorical=categorical, references=references)


def test_padding_is_inert_for_both_forwards(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    decision_batch = _decision_batch(
        decision_schema, [_decision_observation(), _decision_observation(small=True)]
    )
    stable_batch = _stable_batch(
        stable_value_schema, [_stable_observation(), _stable_observation(small=True)]
    )
    model = _model(decision_schema, stable_value_schema)
    baseline_policy = model.forward_policy(decision_batch)
    baseline_value = model.forward_stable_value(stable_batch)

    decision_tokens = {
        kind: _poison_feature_padding(table) for kind, table in decision_batch.tokens.items()
    }
    decision_relationships = {}
    for kind, table in decision_batch.relationships.items():
        poisoned = _poison_feature_padding(table)
        source_indices = table.source_indices.clone()
        target_indices = table.target_indices.clone()
        source_indices[~table.mask] = torch.iinfo(torch.int64).max
        target_indices[~table.mask] = torch.iinfo(torch.int64).max
        decision_relationships[kind] = replace(
            poisoned, source_indices=source_indices, target_indices=target_indices
        )
    candidates = _poison_feature_padding(decision_batch.candidates)
    kind_indices = candidates.kind_indices.clone()
    kind_indices[~candidates.mask] = torch.iinfo(torch.int64).max
    poisoned_decision = replace(
        decision_batch,
        tokens=decision_tokens,
        relationships=decision_relationships,
        candidates=replace(candidates, kind_indices=kind_indices),
    )

    stable_tokens = {
        kind: _poison_feature_padding(table) for kind, table in stable_batch.graph.tokens.items()
    }
    stable_relationships = {}
    for kind, table in stable_batch.graph.relationships.items():
        poisoned = _poison_feature_padding(table)
        source_indices = table.source_indices.clone()
        target_indices = table.target_indices.clone()
        source_indices[~table.mask] = torch.iinfo(torch.int64).max
        target_indices[~table.mask] = torch.iinfo(torch.int64).max
        stable_relationships[kind] = replace(
            poisoned, source_indices=source_indices, target_indices=target_indices
        )
    poisoned_stable = replace(
        stable_batch,
        graph=replace(
            stable_batch.graph,
            tokens=stable_tokens,
            relationships=stable_relationships,
        ),
    )

    changed_policy = model.forward_policy(poisoned_decision)
    changed_value = model.forward_stable_value(poisoned_stable)
    assert torch.allclose(
        baseline_policy.policy_logits[decision_batch.candidates.mask],
        changed_policy.policy_logits[decision_batch.candidates.mask],
        atol=1e-6,
    )
    assert torch.allclose(baseline_value.value, changed_value.value, atol=1e-6)


def test_boundary_context_observably_changes_stable_value(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    batch = _stable_batch(stable_value_schema, [_stable_observation()])
    actor_context = batch.value_context.categorical.clone()
    planning_context = batch.value_context.categorical.clone()
    vocabulary = stable_value_schema.value_context.categorical[0].vocabulary
    actor_index = vocabulary.index("ACTOR_READY")
    planning_index = vocabulary.index("PLANNING_READY")
    actor_context[0, 0, 0] = actor_index
    planning_context[0, 0, 0] = planning_index
    actor_batch = replace(
        batch,
        value_context=replace(batch.value_context, categorical=actor_context),
    )
    planning_batch = replace(
        batch,
        value_context=replace(batch.value_context, categorical=planning_context),
    )
    model = _model(decision_schema, stable_value_schema)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.01)
        boundary_embedding = model.value_context_encoder.embeddings[0].weight
        boundary_embedding[actor_index].zero_()
        boundary_embedding[planning_index].fill_(1.0)

    actor_value = model.forward_stable_value(actor_batch).value
    planning_value = model.forward_stable_value(planning_batch).value

    assert not torch.allclose(actor_value, planning_value)


def _has_nonzero_finite_gradient(parameters: tuple[nn.Parameter, ...]) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool(torch.count_nonzero(parameter.grad))
        for parameter in parameters
    )


def test_parameter_groups_are_exhaustive_disjoint_and_follow_forward_ownership(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    model = _model(decision_schema, stable_value_schema)
    groups = model.parameter_groups()
    grouped_ids = [id(parameter) for parameters in groups.values() for parameter in parameters]

    assert set(groups) == {"shared", "policy", "value"}
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {id(parameter) for parameter in model.parameters()}
    assert {id(parameter) for parameter in model.decision_encoder.parameters()} <= {
        id(parameter) for parameter in groups["policy"]
    }
    assert {id(parameter) for parameter in model.value_context_encoder.parameters()} <= {
        id(parameter) for parameter in groups["value"]
    }

    decision_batch = _decision_batch(decision_schema, [_decision_observation()])
    model.forward_policy(decision_batch).policy_logits.sum().backward()
    assert _has_nonzero_finite_gradient(groups["shared"])
    assert _has_nonzero_finite_gradient(groups["policy"])
    assert all(parameter.grad is None for parameter in groups["value"])

    model.zero_grad(set_to_none=True)
    stable_batch = _stable_batch(stable_value_schema, [_stable_observation()])
    model.forward_stable_value(stable_batch).value.sum().backward()
    assert _has_nonzero_finite_gradient(groups["shared"])
    assert all(parameter.grad is None for parameter in groups["policy"])
    assert _has_nonzero_finite_gradient(groups["value"])


def test_parameter_groups_reject_unclassified_parameters(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    model = _model(decision_schema, stable_value_schema)
    model.register_parameter("extension_parameter", nn.Parameter(torch.ones(1)))

    with pytest.raises(RuntimeError, match="exhaustive and disjoint partition"):
        model.parameter_groups()


def test_stable_forward_never_touches_policy_or_candidate_components(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(decision_schema, stable_value_schema)

    def reject(*args: object, **kwargs: object) -> None:
        raise AssertionError("stable-value forward touched a policy-only component")

    monkeypatch.setattr(model, "_encode_candidates", reject)
    monkeypatch.setattr(DecisionBatch, "gather_candidate_targets", reject)
    for module in (
        model.decision_encoder,
        *model.candidate_encoders.values(),
        model.policy_head,
    ):
        module.register_forward_pre_hook(reject)

    output = model.forward_stable_value(_stable_batch(stable_value_schema, [_stable_observation()]))

    assert output.value.shape == (1,)


def test_forwards_reject_incompatible_batch_context_and_layout(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    model = _model(decision_schema, stable_value_schema)
    decision = _decision_batch(decision_schema, [_decision_observation()])
    stable = _stable_batch(stable_value_schema, [_stable_observation()])
    assert decision.decision_context is not None
    bad_policy_batches = (
        (replace(decision, candidate_kinds=()), "candidate kinds"),
        (replace(decision, decision_context=None), "decision-context"),
        (
            replace(
                decision,
                decision_context=replace(
                    decision.decision_context,
                    mask=torch.zeros_like(decision.decision_context.mask),
                ),
            ),
            "exactly one valid row",
        ),
    )
    for batch, message in bad_policy_batches:
        with pytest.raises(ValueError, match=message):
            model.forward_policy(batch)
    bad_value_batches = (
        (replace(stable, viewers=()), "viewers must align"),
        (
            replace(stable, graph=replace(stable.graph, token_kinds=())),
            "token kinds",
        ),
        (replace(stable, graph=replace(stable.graph, tokens={})), "token tables"),
        (
            replace(stable, graph=replace(stable.graph, relationships={})),
            "relationship tables",
        ),
    )
    for value_batch, message in bad_value_batches:
        with pytest.raises(ValueError, match=message):
            model.forward_stable_value(value_batch)


def test_training_value_gradients_ignore_padded_graph_rows(
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> None:
    model = Gen1PolicyValueModel(
        decision_schema=decision_schema,
        stable_value_schema=stable_value_schema,
        config=_config(decision_schema, stable_value_schema, dropout=0.2),
    )
    model.train()
    batch = _stable_batch(
        stable_value_schema, [_stable_observation(), _stable_observation(small=True)]
    )
    tables = (*batch.graph.tokens.values(), *batch.graph.relationships.values())
    assert any(bool((~table.mask).any()) for table in tables)
    for table in tables:
        table.numeric.requires_grad_(True)
    output = model.forward_stable_value(batch)
    assert torch.isfinite(output.value).all()
    output.value.sum().backward()
    for table in tables:
        if table.numeric.grad is not None:
            assert torch.isfinite(table.numeric.grad).all()
            assert torch.count_nonzero(table.numeric.grad[~table.mask]) == 0
    groups = model.parameter_groups()
    assert _has_nonzero_finite_gradient(groups["shared"])
    assert _has_nonzero_finite_gradient(groups["value"])
    assert all(parameter.grad is None for parameter in groups["policy"])
