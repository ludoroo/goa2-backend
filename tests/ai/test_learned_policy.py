"""Behavioral contract for the portable learned ISMCTS policy prior."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from automata.agents.random_agent import RandomAgent
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from automata.search import POLICY_FEATURE_SCHEMA_ID, ISMCTSAgent, SearchConfig
from automata.search.ismcts import Decision, legal_keys
from automata.search.prior import PolicyResult
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import TeamColor
from goa2.domain.types import HeroID
from goa2.engine.phases import commit_card
from goa2.engine.setup import GameSetup

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]


def _state(*, red: list[str] | None = None, seed: int = 2):
    register_all_effects()
    return GameSetup.create_game(
        DEFAULT_MAP, red or RED, BLUE, game_type="QUICK", seed=seed
    )


def _leaf(value: float) -> dict[str, Any]:
    return {"root": 0, "nodes": [{"value": value}]}


def _split(
    feature: int, threshold: float, left: float, right: float
) -> dict[str, Any]:
    return {
        "root": 0,
        "nodes": [
            {"feature": feature, "threshold": threshold, "left": 1, "right": 2},
            {"value": left},
            {"value": right},
        ],
    }


def _artifact(
    *,
    feature_names: list[str] | None = None,
    trees: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "model_version": "gbm-policy-v1",
        "schema_version": 1,
        "policy_feature_schema_id": POLICY_FEATURE_SCHEMA_ID,
        "red_roster": list(RED),
        "blue_roster": list(BLUE),
        "feature_names": (
            ["candidate.finish"] if feature_names is None else feature_names
        ),
        "base_score": 0.25,
        "learning_rate": 0.5,
        "trees": trees if trees is not None else [_split(0, 0.5, -2.0, 4.0)],
    }


def _policy(source: object):
    module = importlib.import_module("automata.search.learned_policy")
    return module.LearnedPolicy(source)


def _write(tmp_path: Path, artifact: dict[str, Any]) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_import_is_pure_python_and_path_and_mapping_load_identically(
    tmp_path: Path,
) -> None:
    had_numpy = "numpy" in sys.modules
    had_sklearn = "sklearn" in sys.modules
    artifact = _artifact()

    mapping_policy = _policy(artifact)
    path_policy = _policy(_write(tmp_path, artifact))

    if not had_numpy:
        assert "numpy" not in sys.modules
    if not had_sklearn:
        assert "sklearn" not in sys.modules
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    decision = Decision("CARD", hero=hero, can_finish_planning=True)
    legal = legal_keys(decision)
    assert mapping_policy(state, decision, legal) == path_policy(state, decision, legal)


def test_scores_every_legal_key_sparse_in_artifact_order_and_stably_sorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("automata.search.learned_policy")
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    decision = Decision("CARD", hero=hero)
    legal = [hero.hand[2].id, hero.hand[0].id, hero.hand[1].id]
    calls: list[tuple[Decision, list[object]]] = []

    def candidates(state_arg: object, decision_arg: Decision, legal_arg: list[object]):
        assert state_arg is state
        calls.append((decision_arg, list(legal_arg)))
        return {
            legal[0]: {"runtime.extra": 99.0, "known": 2.0},
            legal[1]: {"runtime.extra": -99.0},
            legal[2]: {"known": 0.0},
        }

    monkeypatch.setattr(module, "policy_candidate_features", candidates)
    artifact = _artifact(
        feature_names=["artifact.missing", "known"],
        trees=[_split(0, 0.5, 0.0, 100.0), _split(1, 1.0, -1.0, 2.0)],
    )
    result = module.LearnedPolicy(artifact)(state, decision, legal)

    assert calls == [(decision, legal)]
    assert isinstance(result, PolicyResult)
    assert result.order == [legal[0], legal[1], legal[2]]
    assert result.weights is not None
    assert set(result.weights) == set(legal)
    assert result.weights[legal[0]] == pytest.approx(1.25)
    assert result.weights[legal[1]] == pytest.approx(-0.25)
    assert result.weights[legal[2]] == pytest.approx(-0.25)
    assert all(math.isfinite(weight) for weight in result.weights.values())


def test_hand_calculable_card_tree_can_choose_card_or_finish() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    decision = Decision("CARD", hero=hero, can_finish_planning=True)
    legal = legal_keys(decision)
    shock = next(card.id for card in hero.hand if card.id == "shock")

    card_policy = _policy(
        _artifact(
            feature_names=["card.primary_action_value"],
            trees=[_split(0, 4.0, -3.0, 3.0)],
        )
    )
    finish_policy = _policy(_artifact())

    assert card_policy(state, decision, legal).order[0] == shock
    assert finish_policy(state, decision, legal).order[0] is None


def test_hand_calculable_input_tree_chooses_expected_number() -> None:
    state = _state()
    request = InputRequest(
        id="number",
        request_type=InputRequestType.SELECT_NUMBER,
        player_id="hero_wasp",
        options=[InputOption.from_value(1), InputOption.from_value(3)],
    )
    decision = Decision("INPUT", request=request)
    legal = legal_keys(decision)
    policy = _policy(
        _artifact(feature_names=["number.value"], trees=[_split(0, 2.0, -1.0, 2.0)])
    )

    result = policy(state, decision, legal)
    assert result.order == [3, 1]
    assert result.weights == {1: pytest.approx(-0.25), 3: pytest.approx(1.25)}


def test_learned_policy_integrates_as_explicit_ismcts_prior() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    policy = _policy(
        _artifact(
            feature_names=["card.primary_action_value"],
            trees=[_split(0, 4.0, -3.0, 3.0)],
        )
    )
    decision = Decision("CARD", hero=hero)
    legal = legal_keys(decision)
    expected = policy(state, decision, legal).order[0]
    assert expected is not None
    agent = ISMCTSAgent(
        SearchConfig(
            iterations=1,
            cutoff_rounds=0,
            widening_c=1.0,
            widening_alpha=0.5,
            seed=7,
            use_prior=True,
        ),
        default_policy=RandomAgent(3),
        prior=policy,
    )

    chosen = agent.choose_card(state, hero)
    assert chosen is not None
    assert chosen.id == expected


def test_rosters_are_generic_at_load_and_checked_exactly_at_call_time() -> None:
    declared = _artifact()
    declared["red_roster"] = ["Wasp", "Bain"]
    policy = _policy(declared)
    compatible = _state(red=["Wasp", "Bain"])
    hero = compatible.teams[TeamColor.RED].heroes[0]
    policy(compatible, Decision("CARD", hero=hero), [hero.hand[0].id])

    mismatched = _state()
    wrong_hero = mismatched.teams[TeamColor.RED].heroes[0]
    with pytest.raises(ValueError, match="roster"):
        policy(mismatched, Decision("CARD", hero=wrong_hero), [wrong_hero.hand[0].id])


REQUIRED_FIELDS = (
    "model_version",
    "schema_version",
    "policy_feature_schema_id",
    "red_roster",
    "blue_roster",
    "feature_names",
    "base_score",
    "learning_rate",
    "trees",
)


def _patched(**patch: Any) -> dict[str, Any]:
    artifact = _artifact()
    artifact.update(patch)
    return artifact


def _without(field: str) -> dict[str, Any]:
    artifact = _artifact()
    artifact.pop(field)
    return artifact


@pytest.mark.parametrize(
    "artifact",
    [_without(field) for field in REQUIRED_FIELDS]
    + [
        _patched(model_version="gbm-v1"),
        _patched(schema_version=2),
        _patched(policy_feature_schema_id="unknown-v999"),
        _patched(feature_names=[]),
        _patched(feature_names=["x", "x"]),
        _patched(feature_names=["x", ""]),
        _patched(feature_names=[3]),
        _patched(red_roster="Wasp"),
        _patched(blue_roster=["Arien", 3]),
        _patched(base_score=True),
        _patched(base_score=float("nan")),
        _patched(learning_rate=0.0),
        _patched(learning_rate=-1.0),
        _patched(learning_rate=float("inf")),
        _patched(trees="not-a-list"),
        _patched(trees=[{"root": 0, "nodes": [{"value": float("nan")}]}]),
        _patched(trees=[_split(1, 0.0, 0.0, 1.0)]),
        _patched(trees=[_split(0, float("inf"), 0.0, 1.0)]),
        _patched(
            trees=[
                {
                    "root": 0,
                    "nodes": [
                        {"feature": 0, "threshold": 0.0, "left": 0, "right": 1},
                        {"value": 1.0},
                    ],
                }
            ]
        ),
        _patched(trees=[{"root": 0, "nodes": [{"value": 1.0}, {"value": 2.0}]}]),
    ],
)
def test_strict_artifact_validation_rejects_malformed_models(
    artifact: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _policy(artifact)


def test_digest_is_stable_behavioral_sha256(tmp_path: Path) -> None:
    artifact = _artifact()
    path = _write(tmp_path, artifact)
    digest = _policy(artifact).digest

    assert digest == _policy(path).digest
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)
    for field, changed in (
        ("base_score", 9.0),
        ("learning_rate", 0.25),
        ("feature_names", ["candidate.card"]),
        ("trees", [_leaf(8.0)]),
        ("red_roster", ["Wasp", "Bain"]),
    ):
        assert _policy(_patched(**{field: changed})).digest != digest

    decorated = dict(artifact, provenance={"seed": 999}, metrics={"loss": 0.01})
    assert _policy(decorated).digest == digest


def test_missing_and_invalid_files_match_learned_value_error_policy(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _policy(tmp_path / "missing.json")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError):
        _policy(directory)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        _policy(bad_json)
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        _policy(non_object)


@pytest.mark.parametrize("hidden_team", [TeamColor.RED, TeamColor.BLUE])
def test_hidden_teammate_or_enemy_card_identity_cannot_change_output(
    hidden_team: TeamColor,
) -> None:
    first = _state(seed=11)
    second = _state(seed=11)
    hidden_first = first.teams[hidden_team].heroes[1 if hidden_team == TeamColor.RED else 0]
    hidden_second = second.teams[hidden_team].heroes[1 if hidden_team == TeamColor.RED else 0]
    for hero in (hidden_first, hidden_second):
        for card in hero.hand:
            card.is_facedown = True
    commit_card(first, HeroID(hidden_first.id), hidden_first.hand[0])
    commit_card(second, HeroID(hidden_second.id), hidden_second.hand[1])

    first_actor = first.teams[TeamColor.RED].heroes[0]
    second_actor = second.teams[TeamColor.RED].heroes[0]
    legal = [card.id for card in first_actor.hand]
    assert legal == [card.id for card in second_actor.hand]
    assert _policy(_artifact())(
        first, Decision("CARD", hero=first_actor), legal
    ) == _policy(_artifact())(second, Decision("CARD", hero=second_actor), legal)
