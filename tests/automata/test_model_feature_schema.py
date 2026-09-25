"""Behavioral contract for framework-independent tensor feature vectorization."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import automata.models as nn
import automata.models.shared_encoder.schema as feature_schema_contracts
from automata.decision import DecisionDescriptor as Decision
from automata.observation import encode_decision
from automata.search.ismcts.engine import legal_keys
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import ActiveEffect, DurationType, EffectScope, EffectType, Shape
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))

# This is deliberately an independent audit of the public encoder's fixed fields.
# Adding an encoder field must make this test fail until its tensor policy is declared.
TOKEN_FIELDS = {
    "GLOBAL": {
        "map_id",
        "game_type",
        "phase",
        "round",
        "turn",
        "team_count",
        "hero_count",
        "lane_count",
        "playable_tile_count",
        "tie_breaker_team",
    },
    "TEAM": {
        "team_id",
        "relation",
        "life_counters",
        "hero_count",
        "alive_hero_count",
        "physical_piece_count",
        "minion_count",
        "total_level",
        "mean_level",
        "total_gold",
        "mean_gold",
    },
    "HERO": {
        "hero_id",
        "name",
        "title",
        "team_id",
        "team_ref",
        "relation",
        "level",
        "gold",
        "items",
        "wish_cast_count",
        "rune_slots",
        "is_current_actor",
        "is_decision_owner",
        "is_acting_piece",
        "adapter_features",
    },
    "TILE": {
        "q",
        "r",
        "s",
        "zone_id",
        "is_terrain",
        "has_occupant",
        "spawn_team",
        "spawn_type",
    },
    "ZONE": {"zone_id", "neighbor_count", "spawn_point_count", "is_battle_zone"},
    "LANE": {
        "lane_id",
        "ordered_zone_refs",
        "zone_count",
        "battle_zone_ref",
        "battle_zone_index",
        "signed_battle_zone_advantage",
        "wave_counter",
    },
    "UNIT": {
        "entity_id",
        "unit_type",
        "team_id",
        "relation",
        "owner_ref",
        "is_current_actor",
        "is_decision_owner",
        "is_acting_piece",
        "is_positioned",
        "tile_ref",
        "zone_ref",
        "lane_ref",
        "minion_type",
        "value",
        "is_heavy",
    },
    "CARD": {
        "card_id",
        "owner_ref",
        "area",
        "visibility",
        "count",
        "name",
        "image_id",
        "tier",
        "color",
        "primary_action",
        "primary_action_value",
        "secondary_actions",
        "effect_id",
        "effect_text",
        "initiative",
        "state",
        "is_facedown",
        "is_ranged",
        "range_value",
        "radius_value",
        "item",
        "is_active",
        "spell_rank",
    },
    "EFFECT": {
        "effect_id",
        "effect_type",
        "duration",
        "is_active",
        "scope",
        "stat_type",
        "stat_value",
        "split_axis",
        "split_value",
        "named_color",
    },
    "MARKER": {"marker_type", "target_ref", "source_ref", "value"},
    "TOKEN": {
        "entity_id",
        "name",
        "token_type",
        "owner_ref",
        "is_facedown",
        "is_passable",
        "tile_ref",
    },
    "ENTITY": {"entity_id", "name", "entity_kind", "owner_ref", "is_obstacle", "tile_ref"},
}
RELATIONSHIP_FIELDS = {
    "HEX_ADJACENT": set(),
    "IN_LANE": set(),
    "IN_ZONE": set(),
    "OWNS": set(),
    "POSITIONED_AT": set(),
    "TILE_IN_ZONE": set(),
    "ZONE_ADJACENT": set(),
    "ZONE_IN_LANE": set(),
    "UNIT_TO_UNIT": {
        "delta_q",
        "delta_r",
        "delta_s",
        "hex_distance",
        "path_distance",
        "path_exists",
        "is_adjacent",
        "topology_is_adjacent",
        "is_straight_line",
        "has_line_of_sight",
        "same_zone",
        "same_lane",
        "lane_progress_delta",
        "relation",
        "reachable",
        "threatens",
        "supports",
        "path_distance_valid",
        "has_line_of_sight_valid",
        "reachable_valid",
        "threatens_valid",
        "supports_valid",
    },
}
CANDIDATE_KINDS = {"FINISH", "SKIP", "CARD", "UNIT", "HEX", "NUMBER", "OPTION", "ACTION", "ENTITY"}


def _state():
    return GameSetup.create_game(
        MAP, ["Razzle", "Wasp"], ["Arien", "Brogan"], game_type="QUICK", seed=31
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


def _encode(state: Any, decision: Decision) -> nn.DecisionObservation:
    return encode_decision(
        state,
        decision,
        legal_keys(decision),
        decision_owner_hero_id="hero_razzle",
        perspective_team="RED",
    )


@pytest.fixture(scope="module")
def observation() -> nn.DecisionObservation:
    state = _state()
    return _encode(
        state,
        Decision(
            "INPUT",
            request=_request(
                InputRequestType.SELECT_UNIT,
                ["hero_wasp", "hero_arien"],
                can_skip=True,
            ),
        ),
    )


@pytest.fixture
def schema():
    # The schema and vectorizer are a public, torch-free artifact seam.
    return feature_schema_contracts.TensorFeatureSchema.current()


def _record_schema_fields(record_schema: Any) -> set[str]:
    declarations = (
        tuple(record_schema.numeric)
        + tuple(record_schema.categorical)
        + tuple(record_schema.references)
        + tuple(record_schema.ignored)
    )
    return {declaration.source for declaration in declarations}


def test_current_schema_is_frozen_versioned_and_canonically_artifact_pinnable(schema: Any) -> None:
    assert schema.schema_version >= 1
    assert schema.schema_id
    assert len(schema.digest) == 64
    assert schema == feature_schema_contracts.TensorFeatureSchema.current()

    encoded = nn.canonical_json_bytes(schema)
    restored = nn.from_canonical_json(feature_schema_contracts.TensorFeatureSchema, encoded)
    assert restored == schema
    assert nn.canonical_json_bytes(restored) == encoded
    assert restored.digest == schema.digest

    with pytest.raises((ValidationError, TypeError)):
        schema.schema_id = "mutable"


def test_schema_explicitly_accounts_for_every_current_encoder_field(schema: Any) -> None:
    assert {item.kind for item in schema.tokens} == set(TOKEN_FIELDS)
    assert {item.kind for item in schema.relationships} == set(RELATIONSHIP_FIELDS)
    assert {item.kind for item in schema.candidates} == CANDIDATE_KINDS

    for item in schema.tokens:
        assert _record_schema_fields(item) == TOKEN_FIELDS[item.kind]
    for item in schema.relationships:
        assert _record_schema_fields(item) == RELATIONSHIP_FIELDS[item.kind]
    assert all(ignored.reason.strip() for item in schema.tokens for ignored in item.ignored)

    for item in (*schema.tokens, *schema.relationships, *schema.candidates):
        for feature in item.numeric:
            assert feature.dtype in {"BOOLEAN", "INTEGER", "FLOAT"}
            assert feature.default is not None
            assert feature.normalization in {"NONE", "STANDARD", "MIN_MAX", "SIGNED_LOG"}
        for feature in item.categorical:
            assert feature.policy == "DIRECT"
            assert tuple(feature.vocabulary[:3]) == ("PAD", "UNK", "MISSING")
            assert len(feature.vocabulary) == len(set(feature.vocabulary))


def test_categorical_features_reject_unsupported_vector_policies() -> None:
    with pytest.raises(ValidationError):
        feature_schema_contracts.CategoricalFeature(
            source="tags",
            vocabulary=("PAD", "UNK", "MISSING"),
            policy="MULTI_HOT",
        )


def test_nested_values_have_declared_pooling_summary_or_ignore_policy(schema: Any) -> None:
    nested = {
        ("HERO", "items"),
        ("HERO", "rune_slots"),
        ("HERO", "adapter_features"),
        ("LANE", "ordered_zone_refs"),
        ("CARD", "secondary_actions"),
        ("EFFECT", "scope"),
    }
    by_kind = {item.kind: item for item in schema.tokens}

    for kind, source in nested:
        item = by_kind[kind]
        declarations = [
            declaration
            for declaration in (*item.numeric, *item.categorical, *item.references, *item.ignored)
            if declaration.source == source
        ]
        assert len(declarations) == 1
        declaration = declarations[0]
        assert declaration.policy in {"COUNT", "SUM", "MEAN", "MAX", "IGNORE"}
        assert not hasattr(declaration, "json_stringify") or declaration.json_stringify is False


def test_raw_record_ids_are_not_categorical_model_features(schema: Any) -> None:
    prohibited = {
        ("TEAM", "team_id"),
        ("TILE", "zone_id"),
        ("ZONE", "zone_id"),
        ("LANE", "lane_id"),
        ("UNIT", "entity_id"),
        ("UNIT", "team_id"),
        ("CARD", "card_id"),
        ("TOKEN", "entity_id"),
        ("ENTITY", "entity_id"),
    }
    categorical = {
        (item.kind, declaration.source)
        for item in schema.tokens
        for declaration in item.categorical
    }
    assert prohibited.isdisjoint(categorical)
    artifact = nn.canonical_json_bytes(schema)
    assert b"hero_razzle" not in artifact
    assert b"razzle_card_1" not in artifact


def test_vectorization_is_order_invariant_per_local_record(
    schema: Any, observation: nn.DecisionObservation
) -> None:
    original = schema.vectorize(observation, training=True)
    reordered_tokens = tuple(
        token.model_copy(update={"features": dict(reversed(tuple(token.features.items())))})
        for token in reversed(observation.state.tokens)
    )
    reordered_state = observation.state.model_copy(
        update={
            "tokens": reordered_tokens,
            "relationships": tuple(reversed(observation.state.relationships)),
        }
    )
    reordered_observation = observation.model_copy(update={"state": reordered_state})
    reordered = schema.vectorize(reordered_observation, training=True)

    original_by_ref = {record.local_ref: record for record in original.tokens}
    reordered_by_ref = {record.local_ref: record for record in reordered.tokens}
    assert original_by_ref.keys() == reordered_by_ref.keys()
    for ref in original_by_ref:
        assert original_by_ref[ref].numeric == reordered_by_ref[ref].numeric
        assert original_by_ref[ref].categorical == reordered_by_ref[ref].categorical

    original_edges = {
        (record.kind, record.source_ref, record.target_ref): (record.numeric, record.categorical)
        for record in original.relationships
    }
    reordered_edges = {
        (record.kind, record.source_ref, record.target_ref): (record.numeric, record.categorical)
        for record in reordered.relationships
    }
    assert original_edges == reordered_edges


def test_references_resolve_after_token_permutation_and_optional_missing_is_masked(
    schema: Any, observation: nn.DecisionObservation
) -> None:
    permuted = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={"tokens": tuple(reversed(observation.state.tokens))}
            )
        }
    )
    result = schema.vectorize(permuted, training=True)
    index_by_ref = {record.local_ref: index for index, record in enumerate(result.tokens)}

    unit = next(record for record in result.tokens if record.kind == "UNIT")
    unit_input = next(token for token in permuted.state.tokens if token.local_ref == unit.local_ref)
    unit_schema = next(item for item in schema.tokens if item.kind == "UNIT")
    for declaration, index, valid in zip(
        unit_schema.references, unit.references, unit.reference_valid, strict=True
    ):
        target = unit_input.features[declaration.source]
        if target is None:
            assert declaration.required is False
            assert index == -1 and valid is False
        else:
            assert index == index_by_ref[target] and valid is True


def test_required_missing_reference_and_unexpected_training_field_fail_closed(
    schema: Any, observation: nn.DecisionObservation
) -> None:
    unit_index = next(i for i, token in enumerate(observation.state.tokens) if token.kind == "UNIT")
    unit = observation.state.tokens[unit_index]

    for updates in ({"owner_ref": None}, {**unit.features, "future_encoder_field": 1}):
        features = {**unit.features, **updates}
        tokens = list(observation.state.tokens)
        tokens[unit_index] = unit.model_copy(update={"features": features})
        changed = observation.model_copy(
            update={"state": observation.state.model_copy(update={"tokens": tuple(tokens)})}
        )
        with pytest.raises(ValueError):
            schema.vectorize(changed, training=True)

    future_features = {**unit.features, "future_encoder_field": 1}
    future_tokens = list(observation.state.tokens)
    future_tokens[unit_index] = unit.model_copy(update={"features": future_features})
    inference_input = observation.model_copy(
        update={"state": observation.state.model_copy(update={"tokens": tuple(future_tokens)})}
    )
    baseline = schema.vectorize(observation, training=False)
    inference = schema.vectorize(inference_input, training=False)
    baseline_unit = next(record for record in baseline.tokens if record.local_ref == unit.local_ref)
    inference_unit = next(
        record for record in inference.tokens if record.local_ref == unit.local_ref
    )
    assert inference_unit == baseline_unit


def test_bool_missing_unknown_and_malformed_numbers_have_explicit_behavior(
    schema: Any, observation: nn.DecisionObservation
) -> None:
    global_index = next(
        i for i, token in enumerate(observation.state.tokens) if token.kind == "GLOBAL"
    )
    global_token = observation.state.tokens[global_index]

    def changed(**features: Any) -> nn.DecisionObservation:
        tokens = list(observation.state.tokens)
        tokens[global_index] = global_token.model_copy(
            update={"features": {**global_token.features, **features}}
        )
        return observation.model_copy(
            update={"state": observation.state.model_copy(update={"tokens": tuple(tokens)})}
        )

    with pytest.raises(ValueError):
        schema.vectorize(changed(round=True), training=True)
    with pytest.raises(ValueError):
        schema.vectorize(changed(round=float("nan")), training=True)

    missing = schema.vectorize(changed(phase=None, turn=None), training=True)
    unknown = schema.vectorize(changed(phase="FUTURE_PHASE"), training=True)
    item = next(item for item in schema.tokens if item.kind == "GLOBAL")
    phase_offset = next(
        i for i, feature in enumerate(item.categorical) if feature.source == "phase"
    )
    vocabulary = item.categorical[phase_offset].vocabulary
    missing_global = next(record for record in missing.tokens if record.kind == "GLOBAL")
    unknown_global = next(record for record in unknown.tokens if record.kind == "GLOBAL")
    assert missing_global.categorical[phase_offset] == vocabulary.index("MISSING")
    assert unknown_global.categorical[phase_offset] == vocabulary.index("UNK")
    turn_offset = next(i for i, feature in enumerate(item.numeric) if feature.source == "turn")
    assert missing_global.numeric[turn_offset] == item.numeric[turn_offset].default
    assert missing_global.numeric_valid[turn_offset] is False


def test_effect_duration_from_domain_view_is_vectorized_as_a_closed_categorical(
    schema: Any,
) -> None:
    state = _state()
    state.active_effects.append(
        ActiveEffect(
            id="duration_contract_effect",
            source_id="hero_razzle",
            effect_type=EffectType.AREA_STAT_MODIFIER,
            scope=EffectScope(shape=Shape.RADIUS, range=1, origin_id="hero_razzle"),
            duration=DurationType.THIS_ROUND,
            created_at_turn=1,
            created_at_round=1,
        )
    )
    observation = _encode(
        state,
        Decision(
            "INPUT",
            request=_request(InputRequestType.SELECT_UNIT, ["hero_arien"]),
        ),
    )
    effect = next(token for token in observation.state.tokens if token.kind == "EFFECT")

    assert effect.features["duration"] == "THIS_ROUND"

    effect_schema = next(item for item in schema.tokens if item.kind == "EFFECT")
    duration = next(item for item in effect_schema.categorical if item.source == "duration")
    vectorized = schema.vectorize(observation, training=True)
    effect_vector = next(token for token in vectorized.tokens if token.kind == "EFFECT")

    assert all(item.source != "duration" for item in effect_schema.numeric)
    assert duration.vocabulary == (
        "PAD",
        "UNK",
        "MISSING",
        "THIS_TURN",
        "NEXT_TURN",
        "THIS_ROUND",
        "PASSIVE",
    )
    assert effect_vector.categorical[
        effect_schema.categorical.index(duration)
    ] == duration.vocabulary.index("THIS_ROUND")

    malformed = effect.model_copy(update={"features": {**effect.features, "duration": 3}})
    tokens = tuple(malformed if token is effect else token for token in observation.state.tokens)
    with pytest.raises(ValueError, match="duration must be a string categorical value"):
        schema.vectorize(
            observation.model_copy(
                update={"state": observation.state.model_copy(update={"tokens": tokens})}
            ),
            training=True,
        )


def test_all_current_candidate_kinds_vectorize_and_order_and_ids_stay_python_side(
    schema: Any,
) -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_razzle"))
    assert hero is not None
    token = next(token for supply in state.token_pool.values() for token in supply)
    empty_hex = next(
        hex_
        for hex_, tile in state.board.tiles.items()
        if not tile.is_terrain and tile.occupant_id is None
    )
    state.place_entity(token.id, empty_hex)
    decisions = [
        Decision("CARD", hero=hero, can_finish_planning=True),
        Decision(
            "INPUT",
            request=_request(InputRequestType.SELECT_UNIT_OR_TOKEN, ["hero_wasp", str(token.id)]),
        ),
        Decision(
            "INPUT", request=_request(InputRequestType.SELECT_HEX, [next(iter(state.board.tiles))])
        ),
        Decision(
            "INPUT", request=_request(InputRequestType.SELECT_NUMBER, [0, 2.5], can_skip=True)
        ),
        Decision("INPUT", request=_request(InputRequestType.SELECT_OPTION, ["hold"])),
        Decision("INPUT", request=_request(InputRequestType.CHOOSE_ACTION, ["advance"])),
    ]

    seen: set[str] = set()
    for decision in decisions:
        encoded = _encode(state, decision)
        result = schema.vectorize(encoded, training=True)
        assert result.candidate_ids == tuple(
            candidate.candidate_id for candidate in encoded.candidates
        )
        assert [record.kind for record in result.candidates] == [
            candidate.candidate_id.kind for candidate in encoded.candidates
        ]
        token_index = {record.local_ref: index for index, record in enumerate(result.tokens)}
        for source, record in zip(encoded.candidates, result.candidates, strict=True):
            if source.target_ref is None:
                assert all(valid is False for valid in record.reference_valid)
            else:
                assert token_index[source.target_ref] in record.references
                assert any(record.reference_valid)
        seen.update(record.kind for record in result.candidates)
    assert seen == CANDIDATE_KINDS


def test_empty_candidate_decision_fails_vectorization(
    schema: Any, observation: nn.DecisionObservation
) -> None:
    with pytest.raises(ValueError, match="candidate"):
        schema.vectorize(observation.model_copy(update={"candidates": ()}), training=True)
