"""Behavioral contract for torch ragged decision batching."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import torch

from automata.decision import DecisionDescriptor as Decision
from automata.models.contracts import DecisionObservation
from automata.models.shared_encoder.batching import (
    collate_decisions,
    masked_mean,
    masked_softmax,
    safe_gather,
)
from automata.models.shared_encoder.schema import TensorFeatureSchema, VectorizedDecision
from automata.observation import encode_decision
from automata.search.ismcts.engine import legal_keys
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))


def _state(*, two_per_team: bool = True):
    return GameSetup.create_game(
        MAP,
        ["Razzle", "Wasp"] if two_per_team else ["Razzle"],
        ["Arien", "Brogan"] if two_per_team else ["Arien"],
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


def _real_decisions() -> list[DecisionObservation]:
    state = _state()
    hero = state.get_hero(HeroID("hero_razzle"))
    assert hero is not None
    tile = next(hex_ for hex_, value in state.board.tiles.items() if not value.is_terrain)
    return [
        _encode(state, Decision("CARD", hero=hero, can_finish_planning=True)),
        _encode(
            state,
            Decision(
                "INPUT",
                request=_request(InputRequestType.SELECT_UNIT, ["hero_wasp", "hero_arien"]),
            ),
        ),
        _encode(
            state,
            Decision("INPUT", request=_request(InputRequestType.SELECT_HEX, [tile])),
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


def _unit_observation(*, two_per_team: bool) -> DecisionObservation:
    state = _state(two_per_team=two_per_team)
    return _encode(
        state,
        Decision("INPUT", request=_request(InputRequestType.SELECT_UNIT, ["hero_arien"])),
    )


def _assert_feature_table(table: Any, *, batch_size: int, rows: int, schema: Any) -> None:
    assert table.mask.shape == (batch_size, rows)
    assert table.numeric.shape == (batch_size, rows, len(schema.numeric))
    assert table.numeric_valid.shape == table.numeric.shape
    assert table.categorical.shape == (batch_size, rows, len(schema.categorical))
    assert table.references.shape == (batch_size, rows, len(schema.references))
    assert table.reference_valid.shape == table.references.shape
    assert table.numeric.dtype == torch.float32
    assert table.categorical.dtype == torch.int64
    assert table.references.dtype == torch.int64
    assert table.mask.dtype == torch.bool
    assert table.numeric_valid.dtype == torch.bool
    assert table.reference_valid.dtype == torch.bool
    assert not table.numeric_valid[~table.mask].any()
    assert not table.reference_valid[~table.mask].any()


def test_collates_real_decision_families_into_schema_width_tables() -> None:
    schema = TensorFeatureSchema.current()
    observations = [*_real_decisions(), _unit_observation(two_per_team=False)]
    vectorized = [schema.vectorize(item, training=True) for item in observations]

    # Both public input forms are accepted, and an explicit artifact-pinned schema is mandatory.
    from_observations = collate_decisions(observations, schema=schema, training=True)
    batch = collate_decisions(vectorized, schema=schema)

    assert from_observations.candidate_ids == batch.candidate_ids
    assert set(batch.tokens) == {item.kind for item in schema.tokens}
    assert set(batch.relationships) == {item.kind for item in schema.relationships}

    for record_schema in schema.tokens:
        expected_rows = max(
            sum(record.kind == record_schema.kind for record in item.tokens) for item in vectorized
        )
        _assert_feature_table(
            batch.tokens[record_schema.kind],
            batch_size=len(vectorized),
            rows=expected_rows,
            schema=record_schema,
        )

    for record_schema in schema.relationships:
        expected_rows = max(
            sum(record.kind == record_schema.kind for record in item.relationships)
            for item in vectorized
        )
        table = batch.relationships[record_schema.kind]
        _assert_feature_table(
            table, batch_size=len(vectorized), rows=expected_rows, schema=record_schema
        )
        assert table.source_indices.shape == (len(vectorized), expected_rows)
        assert table.target_indices.shape == (len(vectorized), expected_rows)
        assert table.source_indices.dtype == torch.int64
        assert table.target_indices.dtype == torch.int64

    max_candidates = max(len(item.candidates) for item in vectorized)
    assert batch.candidates.mask.shape == (len(vectorized), max_candidates)
    assert batch.candidates.kind_indices.shape == batch.candidates.mask.shape
    assert batch.candidates.kind_indices.dtype == torch.int64
    assert batch.candidates.numeric.shape == (
        len(vectorized),
        max_candidates,
        max(len(item.numeric) for item in schema.candidates),
    )
    assert batch.candidates.numeric_valid.shape == batch.candidates.numeric.shape
    assert batch.candidates.categorical.shape == (
        len(vectorized),
        max_candidates,
        max(len(item.categorical) for item in schema.candidates),
    )
    assert batch.candidates.references.shape == (
        len(vectorized),
        max_candidates,
        max(len(item.references) for item in schema.candidates),
    )
    assert batch.candidates.reference_valid.shape == batch.candidates.references.shape
    assert batch.candidates.numeric.dtype == torch.float32
    assert batch.candidates.categorical.dtype == torch.int64
    assert batch.candidates.references.dtype == torch.int64
    assert batch.candidates.mask.dtype == torch.bool
    assert not batch.candidates.numeric_valid[~batch.candidates.mask].any()
    assert not batch.candidates.reference_valid[~batch.candidates.mask].any()
    assert batch.candidate_ids == tuple(item.candidate_ids for item in vectorized)
    assert batch.candidates.mask.sum(dim=1).tolist() == [
        len(item.candidates) for item in vectorized
    ]

    kinds = {candidate.kind for decision_ids in batch.candidate_ids for candidate in decision_ids}
    assert kinds >= {"CARD", "UNIT", "HEX", "NUMBER", "OPTION", "ACTION", "SKIP", "FINISH"}


def test_empty_families_are_supported_but_empty_candidate_rows_fail_closed() -> None:
    schema = TensorFeatureSchema.current()
    number = schema.vectorize(_real_decisions()[3], training=True)
    graph_free = number.model_copy(update={"tokens": (), "relationships": ()})

    batch = collate_decisions([graph_free], schema=schema)

    assert all(table.mask.shape[1] == 0 for table in batch.tokens.values())
    assert all(table.mask.shape[1] == 0 for table in batch.relationships.values())
    with pytest.raises(ValueError, match="candidate"):
        collate_decisions(
            [graph_free.model_copy(update={"candidates": (), "candidate_ids": ()})],
            schema=schema,
        )


def test_candidate_targets_resolve_after_arbitrary_token_permutation() -> None:
    schema = TensorFeatureSchema.current()
    state = _state(two_per_team=False)
    observation = _encode(
        state,
        Decision("INPUT", request=_request(InputRequestType.SELECT_UNIT, ["hero_arien"])),
    )
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
    decisions = [schema.vectorize(item, training=True) for item in (observation, permuted)]
    batch = collate_decisions(decisions, schema=schema)
    width = 3
    embeddings = {
        kind: torch.zeros((*table.mask.shape, width), dtype=torch.float32)
        for kind, table in batch.tokens.items()
    }
    target_ref = observation.candidates[0].target_ref
    assert target_ref is not None
    for batch_index, decision in enumerate(decisions):
        unit_offset = next(
            index
            for index, token in enumerate(
                record for record in decision.tokens if record.kind == "UNIT"
            )
            if token.local_ref == target_ref
        )
        embeddings["UNIT"][batch_index, unit_offset] = torch.tensor([17.0, 19.0, 23.0])

    gathered, valid = batch.gather_candidate_targets(embeddings)

    assert valid.tolist() == [[True], [True]]
    assert gathered.tolist() == [[[17.0, 19.0, 23.0]], [[17.0, 19.0, 23.0]]]

    edge_table = batch.relationships["UNIT_TO_UNIT"]
    for batch_index, decision in enumerate(decisions):
        edges = [edge for edge in decision.relationships if edge.kind == "UNIT_TO_UNIT"]
        units = [token for token in decision.tokens if token.kind == "UNIT"]
        by_ref = {token.local_ref: index for index, token in enumerate(units)}
        assert edge_table.source_indices[batch_index, : len(edges)].tolist() == [
            by_ref[edge.source_ref] for edge in edges
        ]
        assert edge_table.target_indices[batch_index, : len(edges)].tolist() == [
            by_ref[edge.target_ref] for edge in edges
        ]


def test_public_masking_and_safe_gather_make_padding_values_inert() -> None:
    schema = TensorFeatureSchema.current()
    observations = [
        _unit_observation(two_per_team=True),
        _unit_observation(two_per_team=False),
    ]
    batch = collate_decisions(observations, schema=schema, training=True)
    table = batch.tokens["UNIT"]
    assert (~table.mask).any()
    values = torch.arange(table.mask.numel(), dtype=torch.float32).reshape(table.mask.shape)
    poisoned = values.clone()
    poisoned[~table.mask] = 1.0e30

    assert torch.equal(
        masked_mean(values, table.mask, dim=1), masked_mean(poisoned, table.mask, dim=1)
    )

    source = torch.tensor([[[2.0], [1.0e30]]], requires_grad=True)
    gathered, valid = safe_gather(
        source,
        torch.tensor([[0, -1, 1]], dtype=torch.int64),
        torch.tensor([[True, False, False]]),
    )
    assert valid.tolist() == [[True, False, False]]
    assert gathered.tolist() == [[[2.0], [0.0], [0.0]]]
    gathered.sum().backward()
    assert source.grad is not None
    assert source.grad.tolist() == [[[1.0], [0.0]]]


def test_masked_softmax_is_normalized_exactly_over_legal_candidates() -> None:
    mask = torch.tensor([[True, False, False], [True, True, False]], dtype=torch.bool)
    logits = torch.tensor([[4.0, 99.0, -99.0], [0.0, 1.0, 50.0]], requires_grad=True)

    probabilities = masked_softmax(logits, mask, dim=-1)

    assert probabilities[0].tolist() == [1.0, 0.0, 0.0]
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    probabilities[:, 0].sum().backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))


def test_masked_pooling_gives_padding_zero_gradient() -> None:
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [1.0e20, -1.0e20]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])

    pooled = masked_mean(values, mask, dim=1)
    assert pooled.tolist() == [[2.0, 3.0]]
    pooled.sum().backward()
    assert values.grad is not None
    assert torch.equal(values.grad[:, 2], torch.zeros_like(values.grad[:, 2]))


def test_only_explicit_batching_import_loads_torch() -> None:
    script = textwrap.dedent("""
        import sys

        import automata.models

        assert "torch" not in sys.modules
        assert not any(name.startswith("torch.") for name in sys.modules)

        import automata.models.shared_encoder.batching

        assert "torch" in sys.modules
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("as_observation", [True, False], ids=["observation", "vectorized"])
def test_collation_is_cpu_deterministic(as_observation: bool) -> None:
    schema = TensorFeatureSchema.current()
    inputs: list[DecisionObservation] | list[VectorizedDecision]
    observations = _real_decisions()[1:4]
    inputs = (
        observations
        if as_observation
        else [schema.vectorize(item, training=True) for item in observations]
    )

    first = collate_decisions(inputs, schema=schema, training=True)
    second = collate_decisions(inputs, schema=schema, training=True)

    assert first.candidate_ids == second.candidate_ids
    assert first.candidates.mask.device.type == "cpu"
    assert torch.equal(first.candidates.numeric, second.candidates.numeric)
    assert torch.equal(first.candidates.mask, second.candidates.mask)
