"""Behavioral contract for the compact joint policy/value model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from automata.decision import DecisionDescriptor as Decision
from automata.models.contracts import DecisionObservation
from automata.models.shared_encoder.batching import DecisionBatch, collate_decisions, masked_softmax
from automata.models.shared_encoder.model import JointModelConfig, JointPolicyValueModel
from automata.models.shared_encoder.schema import TensorFeatureSchema
from automata.observation import encode_decision
from automata.search.ismcts.engine import legal_keys
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))
TWO_LANE_MAP = str(Path("src/goa2/data/maps/across_the_river.json"))


def _state(
    *,
    map_path: str = MAP,
    red: list[str] | None = None,
    blue: list[str] | None = None,
) -> Any:
    return GameSetup.create_game(
        map_path,
        red or ["Razzle", "Wasp"],
        blue or ["Arien", "Brogan"],
        game_type="QUICK",
        seed=31,
    )


def _request(
    kind: InputRequestType, values: Iterable[Any], *, can_skip: bool = False
) -> InputRequest:
    return InputRequest(
        id="request-id",
        request_type=kind,
        player_id="hero_razzle",
        prompt="choose",
        options=[InputOption.from_value(value) for value in values],
        can_skip=can_skip,
    )


def _encode(state: Any, decision: Decision) -> DecisionObservation:
    return encode_decision(
        state,
        decision,
        legal_keys(decision),
        decision_owner_hero_id="hero_razzle",
        perspective_team="RED",
    )


def _unit_observation(*, small: bool = False) -> DecisionObservation:
    state = _state(red=["Razzle"] if small else None, blue=["Arien"] if small else None)
    candidates = ["hero_arien"] if small else ["hero_arien", "hero_razzle_piece_1"]
    return _encode(
        state,
        Decision(
            "INPUT",
            request=_request(InputRequestType.SELECT_UNIT, candidates),
        ),
    )


def _decision_families() -> list[DecisionObservation]:
    state = _state()
    hero = state.get_hero(HeroID("hero_razzle"))
    assert hero is not None
    empty_hex = next(
        hex_
        for hex_, tile in state.board.tiles.items()
        if not tile.is_terrain and tile.occupant_id is None
    )
    token = next(token for supply in state.token_pool.values() for token in supply)
    state.place_entity(token.id, empty_hex)
    return [
        _encode(state, Decision("CARD", hero=hero, can_finish_planning=True)),
        _encode(
            state,
            Decision(
                "INPUT",
                request=_request(
                    InputRequestType.SELECT_UNIT_OR_TOKEN,
                    ["hero_arien", str(token.id)],
                ),
            ),
        ),
        _encode(
            state,
            Decision("INPUT", request=_request(InputRequestType.SELECT_HEX, [empty_hex])),
        ),
        _encode(
            state,
            Decision(
                "INPUT",
                request=_request(InputRequestType.SELECT_NUMBER, [0, 2.5], can_skip=True),
            ),
        ),
        _encode(
            state,
            Decision("INPUT", request=_request(InputRequestType.SELECT_OPTION, ["hold"])),
        ),
        _encode(
            state,
            Decision("INPUT", request=_request(InputRequestType.CHOOSE_ACTION, ["advance"])),
        ),
        _encode(
            state,
            Decision("INPUT", request=_request(InputRequestType.CONFIRM_PASSIVE, ["YES"])),
        ),
    ]


@pytest.fixture
def schema() -> TensorFeatureSchema:
    return TensorFeatureSchema.current()


def _config(schema: TensorFeatureSchema, **changes: Any) -> JointModelConfig:
    values = {
        "model_version": 1,
        "schema_digest": schema.digest,
        "token_width": 16,
        "state_width": 24,
        "candidate_width": 16,
        "message_passing_layers": 2,
        "dropout": 0.0,
    }
    values.update(changes)
    return JointModelConfig(**values)


def _model(schema: TensorFeatureSchema, *, seed: int = 7) -> JointPolicyValueModel:
    torch.manual_seed(seed)
    model = JointPolicyValueModel(schema=schema, config=_config(schema))
    model.eval()
    return model


def _batch(schema: TensorFeatureSchema, observations: list[DecisionObservation]) -> DecisionBatch:
    return collate_decisions(observations, schema=schema, training=True)


def test_forward_is_joint_bounded_and_supports_variable_real_game_shapes(
    schema: TensorFeatureSchema,
) -> None:
    two_lane = _state(
        map_path=TWO_LANE_MAP,
        red=["Razzle", "Wasp", "Xargatha"],
        blue=["Arien", "Brogan", "Tali"],
    )
    observations = [
        _unit_observation(small=True),
        _unit_observation(),
        _encode(
            two_lane,
            Decision("INPUT", request=_request(InputRequestType.SELECT_UNIT, ["hero_arien"])),
        ),
    ]
    batch = _batch(schema, observations)

    output = _model(schema)(batch)

    assert output.policy_logits.shape == batch.candidates.mask.shape
    assert output.value.shape == (len(observations),)
    assert torch.isfinite(output.policy_logits[batch.candidates.mask]).all()
    assert torch.isfinite(output.value).all()
    assert ((output.value >= -1.0) & (output.value <= 1.0)).all()
    assert not hasattr(output, "candidate_ids")
    assert not hasattr(output, "candidate_mask")
    assert len(batch.candidate_ids) == len(observations)


def test_train_mode_for_all_candidate_families_has_finite_forward_and_backward(
    schema: TensorFeatureSchema,
) -> None:
    batch = _batch(schema, _decision_families())
    model = _model(schema)
    model.train()

    output = model(batch)
    probabilities = masked_softmax(output.policy_logits, batch.candidates.mask)
    loss = -torch.log(probabilities[batch.candidates.mask].clamp_min(1e-8)).mean()
    loss = loss + output.value.square().mean()
    loss.backward()

    seen = {candidate.kind for row in batch.candidate_ids for candidate in row}
    assert seen == {"FINISH", "SKIP", "CARD", "UNIT", "HEX", "NUMBER", "OPTION", "ACTION", "ENTITY"}
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
    )


def test_candidate_permutation_only_permutes_logits(schema: TensorFeatureSchema) -> None:
    observation = _unit_observation()
    permutation = (1, 0)
    permuted = observation.model_copy(
        update={"candidates": tuple(observation.candidates[index] for index in permutation)}
    )
    batch = _batch(schema, [observation, permuted])

    output = _model(schema)(batch)

    assert torch.allclose(
        output.policy_logits[0], output.policy_logits[1, list(permutation)], atol=1e-6
    )
    assert torch.allclose(output.value[0], output.value[1], atol=1e-6)


def test_token_relationship_and_unordered_collection_storage_permutations_are_invariant(
    schema: TensorFeatureSchema,
) -> None:
    observation = _unit_observation()
    reordered = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={
                    "tokens": tuple(reversed(observation.state.tokens)),
                    "relationships": tuple(reversed(observation.state.relationships)),
                }
            )
        }
    )
    batch = _batch(schema, [observation, reordered])

    output = _model(schema)(batch)

    assert torch.allclose(output.policy_logits[0], output.policy_logits[1], atol=1e-6)
    assert torch.allclose(output.value[0], output.value[1], atol=1e-6)


def test_double_lane_row_reordering_preserves_explicit_topology_semantics(
    schema: TensorFeatureSchema,
) -> None:
    state = _state(
        map_path=TWO_LANE_MAP,
        red=["Razzle", "Wasp", "Xargatha"],
        blue=["Arien", "Brogan", "Tali"],
    )
    observation = _encode(
        state,
        Decision("INPUT", request=_request(InputRequestType.SELECT_UNIT, ["hero_arien"])),
    )
    reordered = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={
                    "tokens": tuple(reversed(observation.state.tokens)),
                    "relationships": tuple(reversed(observation.state.relationships)),
                }
            )
        }
    )
    batch = _batch(schema, [observation, reordered])

    output = _model(schema)(batch)

    assert torch.allclose(output.policy_logits[0], output.policy_logits[1], atol=1e-6)
    assert torch.allclose(output.value[0], output.value[1], atol=1e-6)


def test_relationship_features_observably_feed_the_joint_representation(
    schema: TensorFeatureSchema,
) -> None:
    batch = _batch(schema, [_unit_observation()])
    table = batch.relationships["UNIT_TO_UNIT"]
    relationship_numeric = table.numeric.detach().clone().requires_grad_(True)
    changed_relationships = {
        **batch.relationships,
        "UNIT_TO_UNIT": replace(table, numeric=relationship_numeric),
    }
    differentiable_batch = replace(batch, relationships=changed_relationships)

    output = _model(schema)(differentiable_batch)
    (output.policy_logits[batch.candidates.mask].sum() + output.value.sum()).backward()

    assert relationship_numeric.grad is not None
    assert torch.isfinite(relationship_numeric.grad).all()
    assert torch.count_nonzero(relationship_numeric.grad[table.mask])


def test_graph_targets_are_gathered_and_non_graph_candidates_still_score(
    schema: TensorFeatureSchema,
) -> None:
    graph_batch = _batch(schema, [_unit_observation()])
    graph_output = _model(schema)(graph_batch)
    assert graph_batch.candidates.target_valid.tolist() == [[True, True]]
    assert not torch.isclose(graph_output.policy_logits[0, 0], graph_output.policy_logits[0, 1])

    number = _decision_families()[3]
    non_graph_batch = _batch(schema, [number])
    non_graph_output = _model(schema)(non_graph_batch)
    assert (~non_graph_batch.candidates.target_valid & non_graph_batch.candidates.mask).any()
    assert torch.isfinite(non_graph_output.policy_logits[non_graph_batch.candidates.mask]).all()


def test_shared_encoder_and_both_heads_receive_joint_loss_gradients(
    schema: TensorFeatureSchema,
) -> None:
    batch = _batch(schema, _decision_families()[:3])
    model = _model(schema)
    model.train()
    output = model(batch)
    policy_loss = (
        -masked_softmax(output.policy_logits, batch.candidates.mask)[:, 0]
        .clamp_min(1e-8)
        .log()
        .mean()
    )
    value_loss = (output.value - torch.tensor([0.5, -0.5, 0.25])).square().mean()

    (policy_loss + value_loss).backward()

    groups = model.parameter_groups()
    assert set(groups) == {"shared", "policy", "value"}
    for parameters in groups.values():
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_parameter_groups_partition_all_model_parameters(schema: TensorFeatureSchema) -> None:
    model = _model(schema)

    groups = model.parameter_groups()
    grouped_parameter_ids = [id(parameter) for group in groups.values() for parameter in group]

    assert set(groups) == {"shared", "policy", "value"}
    assert len(grouped_parameter_ids) == len(set(grouped_parameter_ids))
    assert set(grouped_parameter_ids) == {id(parameter) for parameter in model.parameters()}


def test_parameter_groups_reject_an_unclassified_registered_parameter(
    schema: TensorFeatureSchema,
) -> None:
    model = _model(schema)
    model.register_parameter("extension_parameter", torch.nn.Parameter(torch.ones(1)))

    with pytest.raises(RuntimeError, match="exhaustive and disjoint partition"):
        model.parameter_groups()


def test_padding_poison_is_inert_and_padded_probability_is_exactly_zero(
    schema: TensorFeatureSchema,
) -> None:
    batch = _batch(schema, [_unit_observation(), _unit_observation(small=True)])
    model = _model(schema)
    baseline = model(batch)

    poisoned_tokens = {}
    for kind, table in batch.tokens.items():
        numeric = table.numeric.clone()
        categorical = table.categorical.clone()
        references = table.references.clone()
        numeric[~table.mask] = float("nan")
        categorical[~table.mask] = torch.iinfo(torch.int64).max
        references[~table.mask] = torch.iinfo(torch.int64).max
        poisoned_tokens[kind] = replace(
            table, numeric=numeric, categorical=categorical, references=references
        )
    poisoned_relationships = {}
    for kind, table in batch.relationships.items():
        numeric = table.numeric.clone()
        source_indices = table.source_indices.clone()
        target_indices = table.target_indices.clone()
        numeric[~table.mask] = float("nan")
        source_indices[~table.mask] = torch.iinfo(torch.int64).max
        target_indices[~table.mask] = torch.iinfo(torch.int64).max
        poisoned_relationships[kind] = replace(
            table,
            numeric=numeric,
            source_indices=source_indices,
            target_indices=target_indices,
        )
    candidate_numeric = batch.candidates.numeric.clone()
    candidate_kind = batch.candidates.kind_indices.clone()
    candidate_numeric[~batch.candidates.mask] = float("nan")
    candidate_kind[~batch.candidates.mask] = torch.iinfo(torch.int64).max
    poisoned = replace(
        batch,
        tokens=poisoned_tokens,
        relationships=poisoned_relationships,
        candidates=replace(
            batch.candidates, numeric=candidate_numeric, kind_indices=candidate_kind
        ),
    )

    changed = model(poisoned)
    assert torch.allclose(
        baseline.policy_logits[batch.candidates.mask],
        changed.policy_logits[batch.candidates.mask],
        atol=1e-6,
    )
    assert torch.allclose(baseline.value, changed.value, atol=1e-6)
    probabilities = masked_softmax(changed.policy_logits, batch.candidates.mask)
    assert torch.equal(
        probabilities[~batch.candidates.mask],
        torch.zeros_like(probabilities[~batch.candidates.mask]),
    )


def test_padded_numeric_rows_receive_zero_gradient(schema: TensorFeatureSchema) -> None:
    batch = _batch(schema, [_unit_observation(), _unit_observation(small=True)])
    token_numeric = {
        kind: table.numeric.detach().clone().requires_grad_(True)
        for kind, table in batch.tokens.items()
    }
    relationship_numeric = {
        kind: table.numeric.detach().clone().requires_grad_(True)
        for kind, table in batch.relationships.items()
    }
    candidate_numeric = batch.candidates.numeric.detach().clone().requires_grad_(True)
    differentiable = replace(
        batch,
        tokens={
            kind: replace(table, numeric=token_numeric[kind])
            for kind, table in batch.tokens.items()
        },
        relationships={
            kind: replace(table, numeric=relationship_numeric[kind])
            for kind, table in batch.relationships.items()
        },
        candidates=replace(batch.candidates, numeric=candidate_numeric),
    )

    output = _model(schema)(differentiable)
    (output.policy_logits[batch.candidates.mask].sum() + output.value.sum()).backward()

    checked = 0
    tables = [
        *((table.mask, token_numeric[kind]) for kind, table in batch.tokens.items()),
        *((table.mask, relationship_numeric[kind]) for kind, table in batch.relationships.items()),
        (batch.candidates.mask, candidate_numeric),
    ]
    for mask, numeric in tables:
        if numeric.grad is not None and numeric.shape[-1] and (~mask).any():
            assert torch.equal(
                numeric.grad[~mask],
                torch.zeros_like(numeric.grad[~mask]),
            )
            checked += 1
    assert checked > 0


def test_empty_optional_token_and_relationship_kinds_are_supported(
    schema: TensorFeatureSchema,
) -> None:
    number = schema.vectorize(_decision_families()[3], training=True)
    graph_free = number.model_copy(update={"tokens": (), "relationships": ()})
    batch = collate_decisions([graph_free], schema=schema)

    output = _model(schema)(batch)

    assert torch.isfinite(output.policy_logits[batch.candidates.mask]).all()
    assert torch.isfinite(output.value).all()


def test_eval_initialization_is_seed_deterministic(schema: TensorFeatureSchema) -> None:
    batch = _batch(schema, [_unit_observation()])

    first = _model(schema, seed=47)(batch)
    second = _model(schema, seed=47)(batch)

    assert torch.equal(first.policy_logits, second.policy_logits)
    assert torch.equal(first.value, second.value)


def test_config_is_versioned_and_rejects_invalid_dimensions_or_schema(
    schema: TensorFeatureSchema,
) -> None:
    config = _config(schema)
    assert config.model_version == 1
    assert config.schema_digest == schema.digest

    for field in ("token_width", "state_width", "candidate_width"):
        with pytest.raises(ValueError):
            _config(schema, **{field: 0})
    with pytest.raises(ValueError):
        _config(schema, message_passing_layers=0)
    with pytest.raises(ValueError, match="schema"):
        JointPolicyValueModel(
            schema=schema,
            config=_config(schema, schema_digest="0" * 64),
        )
