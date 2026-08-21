"""Behavioral contract for the compact expert-search policy dataset."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import pytest

from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from automata.search import (
    POLICY_FEATURE_SCHEMA_ID,
    ISMCTSAgent,
    RootChildStats,
    RootSearchObservation,
    SearchConfig,
    policy_candidate_features,
)
from automata.search.ismcts import Decision
from automata.search.node import action_key
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import TeamColor
from goa2.engine.setup import GameSetup

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]


def _policy_dataset() -> ModuleType:
    """Keep the absent planned module from breaking test collection."""
    try:
        return importlib.import_module("automata.evaluation.policy_dataset")
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned policy dataset module is missing: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _effects_registered() -> None:
    register_all_effects()


def _state(seed: int = 4):
    return GameSetup.create_game(
        DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=seed
    )


def _identity(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "game_id": "policy-game-4",
        "world_seed": 4,
        "red_heroes": RED,
        "blue_heroes": BLUE,
        "source_revision": "abc1234",
        "dirty_tree_hash": "dirty-deadbeef",
        "expert_config": {"iterations": 32, "exploration_c": 1.25, "seed": 9},
        "expert_identity": "ismcts-heuristic-v1",
    }
    values.update(overrides)
    return values


def _input_observation(
    values: list[Any] | None = None,
    *,
    visits: list[int] | None = None,
    chosen_index: int = 0,
) -> tuple[Any, RootSearchObservation]:
    state = _state()
    raw = values or [1, 2, {"q": 0, "r": 0, "s": 0}]
    request = InputRequest(
        id="policy-root",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value(value) for value in raw],
    )
    keys = tuple(action_key(value) for value in raw)
    counts = visits or [3, 1, 0]
    stats = {
        key: RootChildStats(
            visits=count,
            total_value=count * (0.25 + index / 10),
            q=(0.25 + index / 10) if count else 0.0,
        )
        for index, (key, count) in enumerate(zip(keys, counts, strict=True))
    }
    observation = RootSearchObservation(
        decision_owner_hero_id="hero_wasp",
        decision_kind="INPUT",
        request=request,
        legal_keys=keys,
        chosen_key=keys[chosen_index],
        child_stats=MappingProxyType(stats),
    )
    return state, observation


def _record_one(module: ModuleType, path: Path, **identity: Any):
    state, observation = _input_observation()
    recorder = module.PolicyDatasetRecorder(path, **_identity(**identity))
    recorder(state, observation)
    return recorder, state, observation


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_schema_constants_are_stable_and_versioned() -> None:
    policy_dataset = _policy_dataset()
    assert isinstance(policy_dataset.SCHEMA_VERSION, int)
    assert policy_dataset.SCHEMA_VERSION >= 1
    assert policy_dataset.POLICY_FEATURE_SCHEMA_ID == POLICY_FEATURE_SCHEMA_ID


def test_root_observer_buffers_compact_complete_policy_row(
    tmp_path: Path,
) -> None:
    policy_dataset = _policy_dataset()
    path = tmp_path / "policy.jsonl"
    recorder, state, observation = _record_one(policy_dataset, path)
    recorder.record_decision(
        state=state,
        team="RED",
        decision_kind="INPUT",
        player_id="hero_wasp",
        legal_keys=list(observation.legal_keys),
        chosen_key=observation.chosen_key,
    )  # TrajectoryRecorder seam must not duplicate root rows.
    assert not path.exists() or not path.read_text()

    recorder.record_outcome(winner="RED", rounds=6, reason="game_over")
    [row] = _rows(path)
    required = {
        "schema_version",
        "policy_feature_schema_id",
        "game_id",
        "world_seed",
        "decision_index",
        "owner_hero_id",
        "team",
        "decision_kind",
        "request_type",
        "legal_keys",
        "chosen_key",
        "candidates",
        "red_heroes",
        "blue_heroes",
        "source_revision",
        "dirty_tree_hash",
        "expert_config",
        "expert_identity",
    }
    assert required <= row.keys()
    assert row["decision_index"] == 0
    assert row["owner_hero_id"] == "hero_wasp"
    assert row["team"] == "RED"
    assert row["decision_kind"] == "INPUT"
    assert row["request_type"] == "SELECT_OPTION"
    assert row["game_id"] == "policy-game-4"
    assert row["expert_config"] == _identity()["expert_config"]
    assert row["expert_identity"] == "ismcts-heuristic-v1"
    for leaked in ("state", "board", "teams", "execution_stack"):
        assert leaked not in row


def test_candidates_follow_legal_order_and_use_real_sparse_policy_features(
    tmp_path: Path,
) -> None:
    policy_dataset = _policy_dataset()
    path = tmp_path / "features.jsonl"
    recorder, state, observation = _record_one(policy_dataset, path)
    expected = policy_candidate_features(
        state, Decision("INPUT", request=observation.request), observation.legal_keys
    )
    recorder.record_outcome(winner="BLUE", rounds=3, reason="game_over")
    [row] = _rows(path)

    assert [candidate["key"] for candidate in row["candidates"]] == row["legal_keys"]
    assert [candidate["features"] for candidate in row["candidates"]] == [
        expected[key] for key in observation.legal_keys
    ]
    assert [candidate["visits"] for candidate in row["candidates"]] == [3, 1, 0]
    assert [candidate["target_probability"] for candidate in row["candidates"]] == pytest.approx(
        [0.75, 0.25, 0.0]
    )
    assert sum(candidate["target_probability"] for candidate in row["candidates"]) == pytest.approx(1.0)
    assert row["chosen_key"] in row["legal_keys"]
    for candidate in row["candidates"]:
        assert all(math.isfinite(value) for value in candidate["features"].values())
        assert math.isfinite(candidate["total_value"])
        assert math.isfinite(candidate["q"])


def test_zero_visit_root_is_skipped_instead_of_emitting_invalid_target(
    tmp_path: Path,
) -> None:
    policy_dataset = _policy_dataset()
    path = tmp_path / "zero.jsonl"
    state, observation = _input_observation(visits=[0, 0, 0])
    recorder = policy_dataset.PolicyDatasetRecorder(path, **_identity())
    recorder(state, observation)
    recorder.record_outcome(winner="RED", rounds=1, reason="game_over")

    assert not path.exists() or not path.read_text().strip()
    assert recorder.stats.skipped_decisions == 1
    assert recorder.stats.recorded_decisions == 0
    assert recorder.stats.skipped_games == 1


def test_nonterminal_outcome_discards_entire_game_buffer_and_updates_stats(
    tmp_path: Path,
) -> None:
    policy_dataset = _policy_dataset()
    path = tmp_path / "incomplete.jsonl"
    recorder, state, observation = _record_one(policy_dataset, path)
    recorder(state, observation)
    recorder.record_outcome(winner=None, rounds=99, reason="max_steps")

    assert not path.exists() or not path.read_text().strip()
    assert isinstance(recorder.stats, policy_dataset.PolicyDatasetStats)
    assert recorder.stats.recorded_decisions == 0
    assert recorder.stats.skipped_decisions == 2
    assert recorder.stats.recorded_games == 0
    assert recorder.stats.skipped_games == 1


def test_complete_game_flushes_and_fsyncs_dataset_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_dataset = _policy_dataset()
    path = tmp_path / "durable.jsonl"
    recorder, _, _ = _record_one(policy_dataset, path)
    fsync_calls: list[int] = []

    def observe_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        assert _rows(path)[0]["game_id"] == "policy-game-4"

    monkeypatch.setattr(policy_dataset.os, "fsync", observe_fsync)
    recorder.record_outcome(winner="RED", rounds=2, reason="game_over")

    assert len(fsync_calls) == 1


def test_typed_action_key_encoding_roundtrips_without_conflation(
    tmp_path: Path,
) -> None:
    policy_dataset = _policy_dataset()
    path = tmp_path / "typed.jsonl"
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    card_key = hero.hand[0].id
    card_observation = RootSearchObservation(
        decision_owner_hero_id=str(hero.id),
        decision_kind="CARD",
        request=None,
        legal_keys=(None, card_key),
        chosen_key=None,
        child_stats=MappingProxyType(
            {
                None: RootChildStats(1, 0.5, 0.5),
                card_key: RootChildStats(1, 0.5, 0.5),
            }
        ),
    )
    input_state, input_observation = _input_observation(
        values=[1, "option_one", {"q": 1, "r": -1, "s": 0}], visits=[1, 1, 1]
    )
    recorder = policy_dataset.PolicyDatasetRecorder(path, **_identity())
    recorder(state, card_observation)
    recorder(input_state, input_observation)
    recorder.record_outcome(winner="RED", rounds=2, reason="game_over")

    loaded = policy_dataset.load_policy_examples(path)
    encoded = [key for row in loaded for key in row["legal_keys"]]
    assert len({json.dumps(key, sort_keys=True) for key in encoded}) == len(encoded)
    assert {key["type"] for key in encoded} >= {"none", "string", "integer", "tuple"}
    assert next(key for key in encoded if key["type"] == "integer")["value"] == 1
    assert next(
        key
        for key in encoded
        if key["type"] == "string" and key["value"] == "option_one"
    )
    tuple_key = next(key for key in encoded if key["type"] == "tuple")
    assert tuple_key["value"] == ["hex", 1, -1, 0]


def test_output_is_byte_deterministic_and_append_preserves_previous_games(
    tmp_path: Path,
) -> None:
    policy_dataset = _policy_dataset()
    def write(path: Path, *, game_id: str = "same", append: bool = False) -> None:
        recorder, _, _ = _record_one(
            policy_dataset, path, game_id=game_id, append=append
        )
        recorder.record_outcome(winner="RED", rounds=2, reason="game_over")

    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    write(first)
    write(second)
    assert first.read_bytes() == second.read_bytes()

    original = first.read_bytes()
    write(first, game_id="next", append=True)
    assert first.read_bytes().startswith(original)
    assert [row["game_id"] for row in _rows(first)] == ["same", "next"]


def test_loader_accepts_valid_rows_and_rejects_malformed_candidate_contracts(
    tmp_path: Path,
) -> None:
    policy_dataset = _policy_dataset()
    valid_path = tmp_path / "valid.jsonl"
    recorder, _, _ = _record_one(policy_dataset, valid_path)
    recorder.record_outcome(winner="RED", rounds=2, reason="game_over")
    [valid] = policy_dataset.load_policy_examples(valid_path)

    mutations: list[Callable[[dict[str, Any]], None]] = [
        lambda row: row.__setitem__("schema_version", 99999),
        lambda row: row.__setitem__("policy_feature_schema_id", "unknown-v999"),
        lambda row: row["candidates"][0]["features"].__setitem__("bad", float("inf")),
        lambda row: row["candidates"].pop(),
        lambda row: row["candidates"][0].__setitem__("key", {"type": "string", "value": "not-legal"}),
        lambda row: row["candidates"][0].__setitem__("target_probability", 0.99),
        lambda row: row.__setitem__("chosen_key", {"type": "string", "value": "not-legal"}),
    ]
    for index, mutate in enumerate(mutations):
        malformed = json.loads(json.dumps(valid))
        mutate(malformed)
        path = tmp_path / f"malformed-{index}.jsonl"
        path.write_text(json.dumps(malformed, allow_nan=True) + "\n")
        with pytest.raises(ValueError):
            policy_dataset.load_policy_examples(path)


def test_loader_rejects_duplicate_indices_and_inconsistent_game_identity(
    tmp_path: Path,
) -> None:
    policy_dataset = _policy_dataset()
    valid_path = tmp_path / "source.jsonl"
    recorder, state, observation = _record_one(policy_dataset, valid_path)
    recorder(state, observation)
    recorder.record_outcome(winner="RED", rounds=2, reason="game_over")
    rows = _rows(valid_path)

    duplicate = json.loads(json.dumps(rows))
    duplicate[1]["decision_index"] = duplicate[0]["decision_index"]
    inconsistent = json.loads(json.dumps(rows))
    inconsistent[1]["red_heroes"] = ["Wasp", "Dodger"]
    for name, bad_rows in (("duplicate", duplicate), ("identity", inconsistent)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in bad_rows))
        with pytest.raises(ValueError):
            policy_dataset.load_policy_examples(path)


def test_small_real_ismcts_search_calls_recorder_as_root_observer(
    tmp_path: Path,
) -> None:
    policy_dataset = _policy_dataset()
    state = _state(seed=8)
    hero = state.teams[TeamColor.RED].heroes[0]
    path = tmp_path / "real-search.jsonl"
    recorder = policy_dataset.PolicyDatasetRecorder(
        path, **_identity(game_id="real-root", world_seed=8)
    )
    agent = ISMCTSAgent(
        SearchConfig(
            iterations=1,
            cutoff_rounds=0,
            widening_c=1.0,
            widening_alpha=0.5,
            seed=7,
        ),
        root_observer=recorder,
    )

    chosen = agent.choose_card(state, hero)
    assert chosen is not None
    recorder.record_outcome(winner="RED", rounds=1, reason="game_over")
    [row] = policy_dataset.load_policy_examples(path)
    assert row["decision_kind"] == "CARD"
    assert row["chosen_key"] in row["legal_keys"]
    assert len(row["candidates"]) == len(hero.hand)
