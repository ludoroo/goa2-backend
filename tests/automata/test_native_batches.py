"""Native per-head collation contracts for indexed training chunks."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from pydantic import ValidationError

from automata.decision import DecisionDescriptor
from automata.harness.game_runner import DEFAULT_MAP
from automata.models.shared_encoder.batching import DecisionBatch, StableValueBatch
from automata.observation import encode_decision
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import detect_stable_value_boundary
from automata.training.metrics import (
    PolicyMetricInput,
    ValueMetricInput,
    policy_metrics,
    value_metrics,
)
from automata.training.native_batches import (
    NativePolicyTrainingBatch,
    NativeValueTrainingBatch,
    collate_native_policy_records,
    collate_native_value_records,
)
from automata.training.native_dataset import (
    NativeBoundaryProvenance,
    NativeGameIdentity,
    PolicyDatasetRecord,
    ValueDatasetRecord,
    native_game_id,
    native_sample_id,
)
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget
from goa2.domain.models import TeamColor
from goa2.engine.setup import GameSetup


def _state(seed: int):
    register_all_effects()
    return GameSetup.create_game(
        DEFAULT_MAP,
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=seed,
    )


def _identity(seed: int) -> NativeGameIdentity:
    fields: dict[str, Any] = {
        "world_seed": seed,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": "native-batch-tests",
        "source_revision": "test-source",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "search-v1",
        "generator_config_id": "generator-v1",
    }
    return NativeGameIdentity(game_id=native_game_id(**fields), **fields)


def _policy_record(
    seed: int,
    *,
    sample_index: int,
    candidate_count: int,
) -> PolicyDatasetRecord:
    state = _state(seed)
    hero = state.teams[TeamColor.RED].heroes[0]
    hero.hand = hero.hand[:candidate_count]
    legal = tuple(card.id for card in hero.hand)
    observation = encode_decision(
        state,
        DecisionDescriptor("CARD", hero=hero),
        legal,
        decision_owner_hero_id=str(hero.id),
        perspective_team=TeamColor.RED.value,
    )
    visits = tuple(range(candidate_count, 0, -1))
    total_visits = sum(visits)
    prior_total = sum(range(1, candidate_count + 1))
    target = SearchPolicyTarget(
        actions=tuple(
            SearchActionTarget(
                candidate=candidate,
                prior_probability=(index + 1) / prior_total,
                sample_count=visits[index],
                mean_value=0.25 + 0.1 * index,
                value_variance=0.05 * index,
                improved_probability=visits[index] / total_visits,
                selected=index == 0,
            )
            for index, candidate in enumerate(observation.candidates)
        )
    )
    game = _identity(seed)
    return PolicyDatasetRecord(
        game=game,
        sample_id=native_sample_id(
            game_id=game.game_id,
            sample_kind="POLICY",
            sample_index=sample_index,
        ),
        sample_index=sample_index,
        policy_index=sample_index,
        perspective_team="RED",
        observation=observation,
        target=target,
    )


def _value_record(
    seed: int,
    *,
    sample_index: int,
    viewer_team: TeamColor = TeamColor.RED,
) -> ValueDatasetRecord:
    state = _state(seed)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    viewer = state.teams[viewer_team].heroes[0]
    observation = encode_stable_value(
        state,
        boundary,
        viewer_hero_id=str(viewer.id),
        perspective_team=viewer_team,
    )
    self_tokens = tuple(
        token
        for token in observation.state.tokens
        if token.kind == "HERO" and token.features.get("relation") == "SELF"
    )
    assert len(self_tokens) == 1
    game = _identity(seed)
    return ValueDatasetRecord(
        game=game,
        sample_id=native_sample_id(
            game_id=game.game_id,
            sample_kind="VALUE",
            sample_index=sample_index,
        ),
        sample_index=sample_index,
        perspective_team=viewer_team.value,
        boundary=NativeBoundaryProvenance(
            boundary_index=sample_index,
            kind=boundary.kind.value,
            round=boundary.round,
            turn=boundary.turn,
            viewer_ref=self_tokens[0].local_ref,
            actor_ref=None,
        ),
        observation=observation,
        terminal_winner="RED",
        value_target=1 if viewer_team is TeamColor.RED else -1,
    )


def test_policy_collation_is_typed_ragged_ordered_and_weighted_by_full_game() -> None:
    short = _policy_record(11, sample_index=4, candidate_count=2)
    long = _policy_record(12, sample_index=7, candidate_count=4)

    result = collate_native_policy_records(
        [short, long],
        game_policy_counts={short.game.game_id: 8, long.game.game_id: 2},
    )

    assert isinstance(result, NativePolicyTrainingBatch)
    assert isinstance(result.batch, DecisionBatch)
    assert result.sample_ids == (short.sample_id, long.sample_id)
    assert result.sample_indexes == (4, 7)
    assert result.game_ids == (short.game.game_id, long.game.game_id)
    assert result.batch.candidates.mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert torch.allclose(
        result.policy_targets,
        torch.tensor(
            [
                [2 / 3, 1 / 3, 0.0, 0.0],
                [0.4, 0.3, 0.2, 0.1],
            ],
            dtype=torch.float32,
        ),
    )
    assert result.row_weights.dtype == torch.float32
    assert result.row_weights.tolist() == pytest.approx([1 / 8, 1 / 2])
    assert result.row_mask.dtype == torch.bool
    assert result.row_mask.tolist() == [True, True]


def test_policy_metadata_keeps_search_priors_separate_from_visit_targets() -> None:
    record = _policy_record(21, sample_index=0, candidate_count=3)
    result = collate_native_policy_records([record], game_policy_counts={record.game.game_id: 1})
    metadata = result.metric_metadata[0]

    assert metadata.target_probabilities == pytest.approx((0.5, 1 / 3, 1 / 6))
    assert metadata.prior_probabilities == pytest.approx((1 / 6, 1 / 3, 1 / 2))
    assert metadata.q_variances == pytest.approx((0.0, 0.05, 0.1))
    assert metadata.target_probabilities != metadata.prior_probabilities
    metric_input = metadata.to_metric_input(torch.tensor([[9.0, 4.0, 1.0, -99.0]])[0])
    assert isinstance(metric_input, PolicyMetricInput)
    assert metric_input.predicted_logits == (9.0, 4.0, 1.0)
    assert metric_input.input_request_type is None
    assert metric_input.semantic_role == record.observation.semantic_role
    assert policy_metrics([metric_input])["overall"]["count"] == 1


def test_value_collation_is_candidate_free_signed_and_has_typed_metric_metadata() -> None:
    red = _value_record(31, sample_index=2, viewer_team=TeamColor.RED)
    blue = _value_record(31, sample_index=3, viewer_team=TeamColor.BLUE)

    result = collate_native_value_records([red, blue], game_value_counts={red.game.game_id: 4})

    assert isinstance(result, NativeValueTrainingBatch)
    assert isinstance(result.batch, StableValueBatch)
    assert not hasattr(result.batch, "candidates")
    assert not hasattr(result.batch, "candidate_ids")
    assert result.value_targets.tolist() == [1.0, -1.0]
    assert result.row_weights.tolist() == pytest.approx([0.25, 0.25])
    assert result.row_mask.tolist() == [True, True]
    assert result.sample_indexes == (2, 3)
    assert all(
        not any(word in field for word in ("candidate", "policy", "prior", "visit", "variance"))
        for field in result.metric_metadata[0].__dataclass_fields__
    )
    metadata = result.metric_metadata[1]
    assert metadata.hero == "Arien"
    assert metadata.map_id == "forgotten_island"
    assert metadata.composition == "Wasp vs Arien"
    assert metadata.round_bucket == "1"
    assert metadata.boundary_kind == "PLANNING_READY"
    metric_input = metadata.to_metric_input(torch.tensor(-0.75))
    assert isinstance(metric_input, ValueMetricInput)
    assert metric_input.target_value == -1
    assert metric_input.predicted_value == pytest.approx(-0.75)
    assert metric_input.candidate_family is None
    assert value_metrics([metric_input])["overall"]["count"] == 1


def test_full_game_weights_are_independent_across_mixed_game_batch_partitions() -> None:
    game_a = [_policy_record(41, sample_index=index, candidate_count=2) for index in range(4)]
    game_b = [_policy_record(42, sample_index=index, candidate_count=2) for index in range(2)]
    counts = {game_a[0].game.game_id: 4, game_b[0].game.game_id: 2}
    first = collate_native_policy_records(
        [game_a[0], game_b[0], game_a[1]], game_policy_counts=counts
    )
    second = collate_native_policy_records(
        [game_b[1], game_a[2], game_a[3]], game_policy_counts=counts
    )

    weights_by_game: dict[str, list[float]] = {}
    for batch in (first, second):
        for game_id, weight in zip(batch.game_ids, batch.row_weights.tolist(), strict=True):
            weights_by_game.setdefault(game_id, []).append(weight)

    assert weights_by_game[game_a[0].game.game_id] == pytest.approx([0.25] * 4)
    assert weights_by_game[game_b[0].game.game_id] == pytest.approx([0.5] * 2)
    assert {game_id: sum(weights) for game_id, weights in weights_by_game.items()} == pytest.approx(
        {game_a[0].game.game_id: 1.0, game_b[0].game.game_id: 1.0}
    )


@pytest.mark.parametrize("head", ["policy", "value"])
def test_collators_reject_empty_wrong_head_duplicate_and_malformed_records(head: str) -> None:
    policy = _policy_record(51, sample_index=0, candidate_count=2)
    value = _value_record(51, sample_index=0)
    collate: Any
    valid: PolicyDatasetRecord | ValueDatasetRecord
    wrong: PolicyDatasetRecord | ValueDatasetRecord
    if head == "policy":
        collate = collate_native_policy_records
        valid = policy
        wrong = value
        keyword = "game_policy_counts"
    else:
        collate = collate_native_value_records
        valid = value
        wrong = policy
        keyword = "game_value_counts"
    counts = {valid.game.game_id: 2}

    with pytest.raises(ValueError, match="empty"):
        collate([], **{keyword: counts})
    with pytest.raises((TypeError, ValueError, ValidationError), match=r"POLICY|VALUE|kind"):
        collate([wrong], **{keyword: counts})
    with pytest.raises(ValueError, match="unique"):
        collate([valid, valid], **{keyword: counts})
    malformed = valid.model_copy(update={"sample_id": "0" * 64})
    with pytest.raises((ValueError, ValidationError), match="sample_id"):
        collate([malformed], **{keyword: counts})


@pytest.mark.parametrize(
    "counts",
    [
        {},
        {"GAME": 0},
        {"GAME": -1},
        {"GAME": True},
        {"GAME": 1.5},
        {1: 2},
    ],
)
def test_count_mapping_must_supply_strict_positive_full_game_counts(
    counts: dict[Any, Any],
) -> None:
    record = _policy_record(61, sample_index=0, candidate_count=2)
    supplied = counts if counts else {}
    if "GAME" in supplied:
        supplied = {record.game.game_id: supplied["GAME"]}

    with pytest.raises((TypeError, ValueError), match=r"count|mapping|game"):
        collate_native_policy_records([record], game_policy_counts=supplied)


def test_count_mapping_cannot_infer_total_from_the_current_chunk() -> None:
    rows = [
        _value_record(71, sample_index=0),
        _value_record(71, sample_index=1),
    ]

    with pytest.raises(ValueError, match=r"observed|records|count"):
        collate_native_value_records(
            rows,
            game_value_counts={rows[0].game.game_id: 1},
        )
