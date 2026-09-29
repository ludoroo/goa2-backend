"""Candidate-free stable-value tensor schema and batching contracts."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from automata.decision import DecisionDescriptor
from automata.models.contracts import (
    StableValueObservation,
    canonical_json_bytes,
    from_canonical_json,
)
from automata.models.shared_encoder.batching import (
    DecisionBatch,
    GraphBatch,
    collate_decisions,
    collate_stable_values,
)
from automata.models.shared_encoder.schema import (
    STABLE_VALUE_TENSOR_SCHEMA_DIGEST,
    StableValueTensorSchema,
    TensorFeatureSchema,
)
from automata.observation import encode_decision
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.value_boundary import detect_stable_value_boundary
from automata.search.ismcts.engine import legal_keys
from goa2.domain.models import CardState, GamePhase, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.phases import resolve_next_action
from goa2.engine.setup import GameSetup

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))
LEGACY_DIGEST = "608ea298bfbe9b6b751b0a756e2b8ffd87098a5012804f1beaf7d6d1c79ab933"


def _planning_state(*, two_per_team: bool = False):
    return GameSetup.create_game(
        MAP,
        ["Wasp", "Razzle"] if two_per_team else ["Wasp"],
        ["Arien", "Brogan"] if two_per_team else ["Arien"],
        game_type="QUICK",
        seed=73,
    )


def _actor_state():
    state = _planning_state()
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
    assert boundary is not None
    return state, boundary


def _value_observation(
    state: Any, *, viewer: str = "hero_wasp", team: TeamColor = TeamColor.RED
) -> StableValueObservation:
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    return encode_stable_value(
        state,
        boundary,
        viewer_hero_id=viewer,
        perspective_team=team,
    )


def _assert_graph_equal(left: GraphBatch, right: GraphBatch) -> None:
    assert left.token_kinds == right.token_kinds
    assert left.tokens.keys() == right.tokens.keys()
    assert left.relationships.keys() == right.relationships.keys()
    for tables in (left.tokens, left.relationships):
        other = right.tokens if tables is left.tokens else right.relationships
        for kind, table in tables.items():
            for name, value in table.__dict__.items():
                assert torch.equal(value, other[kind].__dict__[name]), (kind, name)


def test_value_schema_is_distinct_pinned_and_leaves_released_decision_schema_unchanged() -> None:
    decision = TensorFeatureSchema.current()
    value = StableValueTensorSchema.current()

    assert decision.digest == LEGACY_DIGEST
    assert value.schema_id == "goa2-stable-value-tensor-v1"
    assert value.schema_version == 1
    assert value.observation_schema_version == 1
    assert value.graph_observation_schema_version == 2
    assert value.digest == STABLE_VALUE_TENSOR_SCHEMA_DIGEST
    assert value.digest == "1be2af48315e64fb02425905e0bb873e490b6b8d2ba1b150dacabb9f0944b4cd"
    assert value.digest != decision.digest
    assert value.tokens == decision.tokens
    assert value.relationships == decision.relationships
    assert canonical_json_bytes(TensorFeatureSchema.current()) == canonical_json_bytes(decision)
    assert not hasattr(value, "candidates")
    assert not hasattr(value, "decision_context")
    encoded = canonical_json_bytes(value)
    assert from_canonical_json(StableValueTensorSchema, encoded) == value
    assert canonical_json_bytes(from_canonical_json(StableValueTensorSchema, encoded)) == encoded

    payload = value.model_dump(mode="json")
    payload["digest"] = "0" * 64
    with pytest.raises(ValidationError, match="digest"):
        StableValueTensorSchema.model_validate(payload)


def test_real_actor_and_planning_observations_collate_without_policy_data() -> None:
    actor_state, _ = _actor_state()
    observations = [
        _value_observation(actor_state, viewer="hero_arien", team=TeamColor.BLUE),
        _value_observation(_planning_state(two_per_team=True)),
    ]

    batch = collate_stable_values(observations, schema=StableValueTensorSchema.current())

    assert isinstance(batch.graph, GraphBatch)
    assert batch.value_context.mask.tolist() == [[True], [True]]
    assert batch.value_context.categorical.tolist() == [[[3]], [[4]]]
    assert batch.viewers == tuple(item.state.viewer for item in observations)
    assert batch.viewers[0].private_hero_id == "hero_arien"
    assert batch.viewers[0].perspective_team == "BLUE"
    assert set(batch.__dict__) == {"graph", "value_context", "viewers"}
    assert not any(
        "candidate" in name or "decision" in name or "logit" in name for name in batch.__dict__
    )


def test_shared_graph_collation_is_identical_for_decision_and_value_paths() -> None:
    state, _ = _actor_state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    decision_observation = encode_decision(
        state,
        DecisionDescriptor("CARD", hero=hero, can_finish_planning=True),
        legal_keys(DecisionDescriptor("CARD", hero=hero, can_finish_planning=True)),
        decision_owner_hero_id="hero_wasp",
        perspective_team="RED",
    )
    value_observation = _value_observation(state)
    assert value_observation.state == decision_observation.state

    decision_batch = collate_decisions(
        [decision_observation], schema=TensorFeatureSchema.current(), training=True
    )
    value_batch = collate_stable_values(
        [value_observation], schema=StableValueTensorSchema.current()
    )

    _assert_graph_equal(decision_batch.graph, value_batch.graph)
    assert list(decision_batch.__dict__) == [field.name for field in fields(DecisionBatch)]
    assert list(decision_batch.__dict__) == [
        "tokens",
        "relationships",
        "decision_context",
        "candidates",
        "candidate_ids",
        "token_kinds",
        "candidate_kinds",
    ]


def test_stable_batching_has_ragged_single_batch_parity_and_empty_families() -> None:
    observations = [
        _value_observation(_planning_state()),
        _value_observation(_planning_state(two_per_team=True)),
    ]
    schema = StableValueTensorSchema.current()
    combined = collate_stable_values(observations, schema=schema)

    for index, observation in enumerate(observations):
        single = collate_stable_values([observation], schema=schema)
        for kind, table in single.graph.tokens.items():
            rows = int(table.mask[0].sum())
            assert torch.equal(
                table.numeric[0, :rows], combined.graph.tokens[kind].numeric[index, :rows]
            )
            assert torch.equal(
                table.references[0, :rows], combined.graph.tokens[kind].references[index, :rows]
            )
        for kind, table in single.graph.relationships.items():
            rows = int(table.mask[0].sum())
            assert torch.equal(
                table.source_indices[0, :rows],
                combined.graph.relationships[kind].source_indices[index, :rows],
            )

    assert combined.graph.tokens["MARKER"].mask.shape[1] == 0
    assert combined.graph.tokens["TOKEN"].mask.shape[1] == 0


def test_value_path_rejects_empty_wrong_contract_schema_and_malformed_prevectorized_data() -> None:
    schema = StableValueTensorSchema.current()
    observation = _value_observation(_planning_state())
    vectorized = schema.vectorize(observation)

    with pytest.raises(ValueError, match="empty"):
        collate_stable_values([], schema=schema)
    with pytest.raises(TypeError, match=r"StableValueObservation|VectorizedStableValue"):
        collate_stable_values([object()], schema=schema)  # type: ignore[list-item]

    state = _planning_state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    descriptor = DecisionDescriptor("CARD", hero=hero, can_finish_planning=True)
    decision = encode_decision(
        state,
        descriptor,
        legal_keys(descriptor),
        decision_owner_hero_id="hero_wasp",
        perspective_team="RED",
    )
    with pytest.raises(TypeError, match=r"StableValueObservation|VectorizedStableValue"):
        collate_stable_values([decision], schema=schema)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="TensorFeatureSchema"):
        collate_stable_values([observation], schema=TensorFeatureSchema.current())  # type: ignore[arg-type]

    wrong_digest = vectorized.model_copy(update={"tensor_schema_digest": "0" * 64})
    with pytest.raises(ValueError, match="digest"):
        collate_stable_values([wrong_digest], schema=schema)
    first = vectorized.tokens[0]
    bad_shape = first.model_copy(update={"numeric": (*first.numeric, 1.0)})
    with pytest.raises(ValueError, match=r"numeric.*width"):
        collate_stable_values(
            [vectorized.model_copy(update={"tokens": (bad_shape, *vectorized.tokens[1:])})],
            schema=schema,
        )
    nonfinite = first.model_copy(update={"numeric": (float("nan"), *first.numeric[1:])})
    with pytest.raises(ValueError, match="finite"):
        collate_stable_values(
            [vectorized.model_copy(update={"tokens": (nonfinite, *vectorized.tokens[1:])})],
            schema=schema,
        )
    unknown = first.model_copy(update={"kind": "PRIVATE_HERO_ID"})
    with pytest.raises(ValueError, match="unknown token kind"):
        collate_stable_values(
            [vectorized.model_copy(update={"tokens": (unknown, *vectorized.tokens[1:])})],
            schema=schema,
        )


@pytest.mark.parametrize("prevectorized", [False, True], ids=["observation", "vectorized"])
def test_value_numeric_inputs_cannot_overflow_the_float32_batch(prevectorized: bool) -> None:
    schema = StableValueTensorSchema.current()
    observation = _value_observation(_planning_state())
    changed_tokens = tuple(
        (
            token.model_copy(update={"features": {**token.features, "round": 10**50}})
            if token.kind == "GLOBAL"
            else token
        )
        for token in observation.state.tokens
    )
    malformed = observation.model_copy(
        update={"state": observation.state.model_copy(update={"tokens": changed_tokens})}
    )
    source = schema.vectorize(malformed) if prevectorized else malformed

    with pytest.raises(ValueError, match=r"finite|float32|range"):
        collate_stable_values([source], schema=schema)


def test_prevectorized_value_cannot_mask_a_required_reference() -> None:
    schema = StableValueTensorSchema.current()
    vectorized = schema.vectorize(_value_observation(_planning_state()))
    hero_schema = next(item for item in schema.tokens if item.kind == "HERO")
    column = next(index for index, field in enumerate(hero_schema.references) if field.required)
    hero = next(item for item in vectorized.tokens if item.kind == "HERO")
    references, validity = list(hero.references), list(hero.reference_valid)
    references[column], validity[column] = -1, False
    malformed = hero.model_copy(
        update={"references": tuple(references), "reference_valid": tuple(validity)}
    )
    source = vectorized.model_copy(
        update={"tokens": tuple(malformed if item is hero else item for item in vectorized.tokens)}
    )

    with pytest.raises(ValueError, match="required reference"):
        collate_stable_values([source], schema=schema)


@pytest.mark.parametrize("prevectorized", [False, True], ids=["observation", "vectorized"])
@pytest.mark.parametrize(
    "change",
    [{"private_hero_id": ""}, {"perspective_team": "GREEN"}, {"schema_version": 1}],
    ids=["empty-viewer", "unsupported-team", "viewer-version"],
)
def test_value_batch_rejects_invalid_viewer_metadata(
    prevectorized: bool, change: dict[str, object]
) -> None:
    schema = StableValueTensorSchema.current()
    observation = _value_observation(_planning_state())
    viewer = observation.state.viewer.model_copy(update=change)
    if prevectorized:
        source = schema.vectorize(observation).model_copy(update={"viewer": viewer})
    else:
        source = observation.model_copy(
            update={"state": observation.state.model_copy(update={"viewer": viewer})}
        )

    with pytest.raises(ValueError, match="viewer"):
        collate_stable_values([source], schema=schema)


@pytest.mark.parametrize("index", [1, 2, 99], ids=["unknown", "missing", "out-of-range"])
def test_prevectorized_boundary_context_requires_a_real_boundary_kind(index: int) -> None:
    schema = StableValueTensorSchema.current()
    vectorized = schema.vectorize(_value_observation(_planning_state()))
    malformed = vectorized.model_copy(
        update={
            "value_context": vectorized.value_context.model_copy(update={"categorical": (index,)})
        }
    )

    with pytest.raises(ValueError, match=r"boundary|categorical"):
        collate_stable_values([malformed], schema=schema)


@pytest.mark.parametrize("valid", [False, True], ids=["wrong-sentinel", "out-of-range"])
def test_prevectorized_token_reference_must_be_masked_or_address_the_graph(valid: bool) -> None:
    schema = StableValueTensorSchema.current()
    vectorized = schema.vectorize(_value_observation(_planning_state()))
    schemas = {item.kind: item for item in schema.tokens}
    index, token, column = next(
        (index, token, column)
        for index, token in enumerate(vectorized.tokens)
        for column, field in enumerate(schemas[token.kind].references)
        if not field.required
    )
    references, validity = list(token.references), list(token.reference_valid)
    references[column], validity[column] = len(vectorized.tokens), valid
    malformed = token.model_copy(
        update={"references": tuple(references), "reference_valid": tuple(validity)}
    )
    records = list(vectorized.tokens)
    records[index] = malformed

    with pytest.raises(ValueError, match="reference"):
        collate_stable_values(
            [vectorized.model_copy(update={"tokens": tuple(records)})], schema=schema
        )


@pytest.mark.parametrize("endpoint", ["source", "target"])
def test_prevectorized_relationship_index_must_identify_its_reference(endpoint: str) -> None:
    schema = StableValueTensorSchema.current()
    vectorized = schema.vectorize(_value_observation(_planning_state()))
    edge = vectorized.relationships[0]
    field = f"{endpoint}_index"
    malformed = edge.model_copy(update={field: (getattr(edge, field) + 1) % len(vectorized.tokens)})
    source = vectorized.model_copy(
        update={"relationships": (malformed, *vectorized.relationships[1:])}
    )

    with pytest.raises(ValueError, match="index/ref mismatch"):
        collate_stable_values([source], schema=schema)


def _replace_local_refs(value: object, aliases: dict[str, str]) -> object:
    if isinstance(value, str):
        return aliases.get(value, value)
    if isinstance(value, list):
        return [_replace_local_refs(item, aliases) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_local_refs(item, aliases) for item in value)
    if isinstance(value, dict):
        return {key: _replace_local_refs(item, aliases) for key, item in value.items()}
    return value


def test_local_ref_aliases_and_ignored_raw_ids_do_not_become_model_features() -> None:
    observation = _value_observation(_planning_state())
    aliases = {
        token.local_ref: f"opaque-ref-{index}"
        for index, token in enumerate(observation.state.tokens)
    }
    raw_id_fields = {"hero_id", "team_id", "entity_id", "card_id", "lane_id", "zone_id"}
    tokens = []
    for index, token in enumerate(observation.state.tokens):
        features = _replace_local_refs(token.features, aliases)
        assert isinstance(features, dict)
        features = {
            key: f"private-raw-id-{index}-{key}" if key in raw_id_fields else value
            for key, value in features.items()
        }
        tokens.append(
            token.model_copy(update={"local_ref": aliases[token.local_ref], "features": features})
        )
    relationships = tuple(
        edge.model_copy(
            update={
                "source_ref": aliases[edge.source_ref],
                "target_ref": aliases[edge.target_ref],
                "features": _replace_local_refs(edge.features, aliases),
            }
        )
        for edge in observation.state.relationships
    )
    aliased = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={"tokens": tuple(tokens), "relationships": relationships}
            )
        }
    )
    schema = StableValueTensorSchema.current()

    baseline = collate_stable_values([observation], schema=schema)
    changed = collate_stable_values([aliased], schema=schema)

    _assert_graph_equal(baseline.graph, changed.graph)
    assert baseline.viewers == changed.viewers


def test_stable_graph_references_resolve_after_token_permutation() -> None:
    observation = _value_observation(_planning_state())
    permuted = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={
                    "tokens": tuple(reversed(observation.state.tokens)),
                    "relationships": tuple(reversed(observation.state.relationships)),
                }
            )
        }
    )
    schema = StableValueTensorSchema.current()
    vectorized = schema.vectorize(permuted)
    batch = collate_stable_values([vectorized], schema=schema)
    kind_indexes = {kind: index for index, kind in enumerate(batch.graph.token_kinds)}

    for record_schema in schema.tokens:
        records = [item for item in vectorized.tokens if item.kind == record_schema.kind]
        table = batch.graph.tokens[record_schema.kind]
        for row, record in enumerate(records):
            for column, (reference, valid) in enumerate(
                zip(record.references, record.reference_valid, strict=True)
            ):
                if not valid:
                    continue
                target = vectorized.tokens[reference]
                target_records = [item for item in vectorized.tokens if item.kind == target.kind]
                target_offset = next(
                    index
                    for index, item in enumerate(target_records)
                    if item.local_ref == target.local_ref
                )
                assert table.references[0, row, column].item() == target_offset
                assert (
                    table.reference_kind_indices[0, row, column].item() == kind_indexes[target.kind]
                )


def test_value_vectorization_does_not_construct_policy_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: object, **kwargs: object) -> None:
        raise AssertionError("policy path must not be called")

    monkeypatch.setattr(TensorFeatureSchema, "vectorize", reject)
    monkeypatch.setattr(TensorFeatureSchema, "_candidate_source", reject)
    monkeypatch.setattr("automata.models.shared_encoder.batching.CandidateTable", reject)
    monkeypatch.setattr("automata.models.shared_encoder.batching.collate_decisions", reject)

    batch = collate_stable_values(
        [_value_observation(_planning_state())],
        schema=StableValueTensorSchema.current(),
    )

    assert batch.graph.tokens["GLOBAL"].mask.tolist() == [[True]]


def test_hidden_opponent_card_changes_do_not_change_value_tensors() -> None:
    state = _planning_state()
    changed = state.model_copy(deep=True)
    opponent = changed.get_hero(HeroID("hero_arien"))
    assert opponent is not None and opponent.hand
    opponent.hand[0].id = "private_replacement_card"
    schema = StableValueTensorSchema.current()

    baseline = collate_stable_values([_value_observation(state)], schema=schema)
    replaced = collate_stable_values([_value_observation(changed)], schema=schema)

    _assert_graph_equal(baseline.graph, replaced.graph)
    assert torch.equal(baseline.value_context.categorical, replaced.value_context.categorical)
