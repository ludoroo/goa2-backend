"""RED contracts for portable gradient-boosted value inference and training."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from automata.evaluation.features import FEATURE_NAMES, RICH_FEATURE_NAMES
from automata.evaluation.value_dataset import SCHEMA_VERSION
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from goa2.domain.models import TeamColor
from goa2.engine.setup import GameSetup

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]
LIFE = FEATURE_NAMES.index("life_diff")


def _artifact() -> dict[str, Any]:
    return {
        "model_version": "gbm-v1",
        "schema_version": 1,
        "red_roster": RED,
        "blue_roster": BLUE,
        "feature_names": list(FEATURE_NAMES),
        "base_raw_score": 0.2,
        "learning_rate": 0.5,
        "trees": [
            {
                "root": 0,
                "nodes": [
                    {"feature": LIFE, "threshold": 0.0, "left": 1, "right": 2},
                    {"value": -2.0},
                    {"value": 1.0},
                ],
            }
        ],
    }


def _state(life_diff: int = 0):
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=2)
    state.teams[TeamColor.RED].life_counters = 5 + life_diff
    state.teams[TeamColor.BLUE].life_counters = 5
    return state


def test_gbm_hand_tree_traversal_and_digest() -> None:
    from automata.evaluation.learned_value import LearnedValue

    artifact = _artifact()
    value = LearnedValue(artifact)
    # raw scores: 0.2 + 0.5*(-2) and 0.2 + 0.5*(1), respectively.
    assert value(_state(-1), TeamColor.RED) == pytest.approx(math.tanh(-0.8 / 2))
    assert value(_state(1), TeamColor.RED) == pytest.approx(math.tanh(0.7 / 2))
    assert LearnedValue(deepcopy(artifact)).digest == value.digest
    changed = deepcopy(artifact)
    changed["trees"][0]["nodes"][2]["value"] = 1.01
    assert LearnedValue(changed).digest != value.digest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.update(base_raw_score=float("nan")),
        lambda a: a.update(learning_rate=0.0),
        lambda a: a.update(feature_names=list(reversed(FEATURE_NAMES))),
        lambda a: a["trees"][0].update(root=9),
        lambda a: a["trees"][0]["nodes"][0].update(feature=len(FEATURE_NAMES)),
        lambda a: a["trees"][0]["nodes"][0].update(threshold=float("inf")),
        lambda a: a["trees"][0]["nodes"][0].update(left=9),
        lambda a: a["trees"][0]["nodes"][0].update(left=0),
        lambda a: a["trees"][0]["nodes"].append({"value": 3.0}),
        lambda a: a["trees"][0]["nodes"][1].update(feature=0),
        lambda a: a["trees"][0]["nodes"][1].update(value=float("nan")),
    ],
    ids=[
        "base",
        "rate",
        "features",
        "root",
        "feature",
        "threshold",
        "child",
        "cycle",
        "unreachable",
        "mixed-node",
        "leaf",
    ],
)
def test_gbm_validation_rejects_malformed_graphs(mutate: Any) -> None:
    from automata.evaluation.learned_value import LearnedValue

    artifact = _artifact()
    LearnedValue(artifact)  # Ensure rejection is not merely “gbm-v1 unsupported”.
    mutate(artifact)
    with pytest.raises(ValueError):
        LearnedValue(artifact)


def _row(game: int, index: int) -> dict[str, Any]:
    winner = "RED" if game % 2 == 0 else "BLUE"
    team = "RED" if index % 2 == 0 else "BLUE"
    won = team == winner
    features = [0.0] * len(FEATURE_NAMES)
    features[LIFE] = 3.0 if won else -3.0
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": f"game-{game}",
        "world_seed": game,
        "team": team,
        "features": features,
        "feature_names": list(FEATURE_NAMES),
        "winner": winner,
        "red_heroes": RED,
        "blue_heroes": BLUE,
        "source_revision": "test-revision",
        "dirty_tree_hash": "clean",
    }


def _write_dataset(path: Path, games: int = 16) -> Path:
    rows = [_row(game, index) for game in range(games) for index in range(4)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _rich_dataset(path: Path, games: int = 16) -> Path:
    rows = [_row(game, index) for game in range(games) for index in range(4)]
    for row in rows:
        signal = row["features"][LIFE]
        row.update(
            feature_schema="rich-v1",
            feature_names=list(RICH_FEATURE_NAMES),
            features=[signal, *([0.0] * (len(RICH_FEATURE_NAMES) - 1))],
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _args(source: Path, out: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--out",
        str(out),
        "--split-seed",
        "17",
        "--val-games",
        "3",
        "--test-games",
        "3",
        "--estimators-grid",
        "2,4",
        "--depth-grid",
        "1,2",
        "--learning-rate-grid",
        "0.1,0.5",
    ]


def _train(source: Path, out: Path) -> dict[str, Any]:
    module_name = "automata.evaluation.train_gbm_value"
    assert importlib.util.find_spec(module_name) is not None, "missing GBM training CLI"
    train_gbm_value = importlib.import_module(module_name)

    assert train_gbm_value.main(_args(source, out)) in (None, 0)
    return json.loads(out.read_text(encoding="utf-8"))


def test_gbm_cli_exports_compatible_model_metrics_and_provenance(tmp_path: Path) -> None:
    from automata.evaluation.learned_value import LearnedValue

    artifact = _train(_write_dataset(tmp_path / "data.jsonl"), tmp_path / "model.json")
    model = LearnedValue(artifact)
    assert artifact["model_version"] == "gbm-v1"
    assert artifact["feature_names"] == list(FEATURE_NAMES)
    assert artifact["red_roster"] == RED and artifact["blue_roster"] == BLUE
    selected = artifact["provenance"]["selected_params"]
    assert selected["n_estimators"] in (2, 4)
    assert selected["max_depth"] in (1, 2)
    assert selected["learning_rate"] in (0.1, 0.5)
    assert artifact["provenance"]["split_counts"] == {
        "train": 10,
        "validation": 3,
        "test": 3,
    }
    assert len(artifact["provenance"]["dataset_digest"]) == 64
    for name in ("test_log_loss", "brier", "accuracy", "ece"):
        assert math.isfinite(artifact["metrics"][name])
    assert artifact["metrics"]["accuracy"] > 0.5
    assert model(_state(1), TeamColor.RED) > model(_state(-1), TeamColor.RED)


def test_gbm_trains_rich_artifact_that_learned_value_scores(tmp_path: Path) -> None:
    from automata.evaluation.learned_value import LearnedValue

    artifact = _train(_rich_dataset(tmp_path / "rich.jsonl"), tmp_path / "rich.json")
    assert artifact["feature_schema"] == "rich-v1"
    assert artifact["feature_names"] == list(RICH_FEATURE_NAMES)
    assert math.isfinite(LearnedValue(artifact)(_state(), TeamColor.RED))


def test_gbm_rejects_mixed_feature_schemas(tmp_path: Path) -> None:
    source = _rich_dataset(tmp_path / "mixed.jsonl")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0].pop("feature_schema")
    rows[0]["feature_names"] = list(FEATURE_NAMES)
    rows[0]["features"] = [0.0] * len(FEATURE_NAMES)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, SystemExit)):
        _train(source, tmp_path / "out.json")


def test_gbm_cli_is_byte_deterministic_and_splits_by_source_game(tmp_path: Path) -> None:
    source = _write_dataset(tmp_path / "data.jsonl")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    one, two = _train(source, first), _train(source, second)
    assert first.read_bytes() == second.read_bytes()
    splits = one["provenance"]["split_games"]
    memberships = [
        {(entry["game_id"], entry["world_seed"]) for entry in splits[name]}
        for name in ("train", "validation", "test")
    ]
    assert not (memberships[0] & memberships[1] or memberships[0] & memberships[2])
    assert not memberships[1] & memberships[2]
    assert len(set().union(*memberships)) == 16
    assert two == one


def test_gbm_training_equalizes_each_source_games_weight(tmp_path: Path) -> None:
    from automata.evaluation.learned_value import LearnedValue

    source = _write_dataset(tmp_path / "base.jsonl")
    base = _train(source, tmp_path / "base.json")
    train_game = base["provenance"]["split_games"]["train"][0]["game_id"]
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    inflated = [
        copy for row in rows for copy in ([row] * (5 if row["game_id"] == train_game else 1))
    ]
    inflated_path = tmp_path / "inflated.jsonl"
    inflated_path.write_text("".join(json.dumps(row) + "\n" for row in inflated), encoding="utf-8")
    boosted = _train(inflated_path, tmp_path / "boosted.json")

    for state in (_state(-1), _state(1)):
        assert LearnedValue(boosted)(state, TeamColor.RED) == pytest.approx(
            LearnedValue(base)(state, TeamColor.RED), abs=1e-10
        )
