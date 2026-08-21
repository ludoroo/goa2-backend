"""RED behavioral contracts for the portable policy GBM trainer."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from automata.evaluation.policy_dataset import (
    POLICY_FEATURE_SCHEMA_ID,
    SCHEMA_VERSION,
    load_policy_examples,
)

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]
EXPERT_CONFIG = {"exploration_c": 1.25, "iterations": 64, "seed": 7}


def _key(value: str) -> dict[str, Any]:
    return {"type": "string", "value": value}


def _row(game: int, decision: int, signal: float) -> dict[str, Any]:
    keys = [_key("action-a"), _key("action-b")]
    a_is_best = signal > 0
    visits = [9, 1] if a_is_best else [1, 9]
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_feature_schema_id": POLICY_FEATURE_SCHEMA_ID,
        "game_id": f"policy-game-{game}",
        "world_seed": 10_000 + game,
        "decision_index": decision,
        "owner_hero_id": "hero_wasp",
        "team": "RED",
        "decision_kind": "INPUT",
        "request_type": "SELECT_OPTION",
        "legal_keys": keys,
        "chosen_key": keys[0 if a_is_best else 1],
        "candidates": [
            {
                "key": keys[0],
                "features": {"action_is_a": 1.0, "state_signal": signal},
                "visits": visits[0],
                "total_value": float(visits[0]) / 2,
                "q": 0.5,
                "target_probability": visits[0] / 10,
            },
            {
                "key": keys[1],
                # Deliberately sparse: omitted features must become zero.
                "features": {"state_signal": signal},
                "visits": visits[1],
                "total_value": float(visits[1]) / 2,
                "q": 0.5,
                "target_probability": visits[1] / 10,
            },
        ],
        "red_heroes": list(RED),
        "blue_heroes": list(BLUE),
        "source_revision": "policy-test-revision",
        "dirty_tree_hash": "clean",
        "expert_config": dict(EXPERT_CONFIG),
        "expert_identity": "ismcts-heuristic-v1",
    }


def _dataset(games: int = 24, decisions: int = 2) -> list[dict[str, Any]]:
    return [
        _row(game, decision, 1.0 if (game + decision) % 2 == 0 else -1.0)
        for game in range(games)
        for decision in range(decisions)
    ]


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _args(source: Path, out: Path, *, seed: int = 17) -> list[str]:
    return [
        "--input",
        str(source),
        "--out",
        str(out),
        "--split-seed",
        str(seed),
        "--val-games",
        "4",
        "--test-games",
        "4",
        "--estimators-grid",
        "8,16",
        "--depth-grid",
        "1,2",
        "--learning-rate-grid",
        "0.1,0.3",
    ]


def _main(argv: list[str]) -> int | None:
    module_name = "automata.evaluation.train_policy"
    assert importlib.util.find_spec(module_name) is not None, "missing policy training CLI"
    return importlib.import_module(module_name).main(argv)


def _train(tmp_path: Path, rows: list[dict[str, Any]], *, seed: int = 17) -> dict[str, Any]:
    source = _write(tmp_path / "policy.jsonl", rows)
    out = tmp_path / "policy-model.json"
    assert _main(_args(source, out, seed=seed)) in (None, 0)
    return json.loads(out.read_text(encoding="utf-8"))


def _members(artifact: dict[str, Any]) -> dict[str, set[tuple[str, int]]]:
    return {
        name: {(item["game_id"], item["world_seed"]) for item in games}
        for name, games in artifact["provenance"]["split_games"].items()
    }


def _portable_score(artifact: dict[str, Any], vector: list[float]) -> float:
    score = float(artifact["base_score"])
    for tree in artifact["trees"]:
        index = tree["root"]
        while "value" not in tree["nodes"][index]:
            node = tree["nodes"][index]
            index = (
                node["left"]
                if vector[node["feature"]] < node["threshold"]
                else node["right"]
            )
        score += artifact["learning_rate"] * tree["nodes"][index]["value"]
    return score


def _vectors(
    rows: list[dict[str, Any]],
    games: set[tuple[str, int]],
    vocabulary: list[str],
) -> tuple[list[list[float]], list[float], list[float]]:
    by_game: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["game_id"], row["world_seed"])
        if key in games:
            by_game[key].append(row)
    x: list[list[float]] = []
    y: list[float] = []
    weights: list[float] = []
    for decisions in by_game.values():
        for row in decisions:
            candidates = row["candidates"]
            weight = 1.0 / len(by_game) / len(decisions) / len(candidates)
            for candidate in candidates:
                features = candidate["features"]
                x.append([float(features.get(name, 0.0)) for name in vocabulary])
                y.append(float(candidate["target_probability"]))
                weights.append(weight)
    return x, y, weights


def test_decision_metrics_give_fractional_top1_credit_for_predicted_ties() -> None:
    from automata.evaluation.train_policy import _decision_metrics

    _, top1, pairwise = _decision_metrics([0.7, 0.2, 0.1], [0.0, 0.0, 0.0])

    assert top1 == pytest.approx(1 / 3)
    assert pairwise == 0.5


def test_decision_metrics_treat_unordered_expert_targets_as_neutral() -> None:
    from automata.evaluation.train_policy import _decision_metrics

    _, top1, pairwise = _decision_metrics([0.5, 0.5], [1.0, 0.0])

    assert top1 == 1.0
    assert pairwise == 0.5


def test_training_module_import_is_lazy_about_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime imports must not make scikit-learn a project dependency."""
    module_name = "automata.evaluation.train_policy"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    real_import = importlib.import_module

    def guarded(name: str, package: str | None = None):
        if name.startswith("sklearn"):
            raise AssertionError("sklearn imported while loading trainer")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded)
    assert importlib.import_module(module_name).__name__ == module_name


@pytest.mark.parametrize(
    "contents",
    ["", "not json\n", json.dumps({"schema_version": SCHEMA_VERSION}) + "\n"],
    ids=["empty", "invalid-json", "incomplete-row"],
)
def test_main_rejects_empty_or_malformed_datasets(tmp_path: Path, contents: str) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        _main(_args(source, tmp_path / "out.json"))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("policy_feature_schema_id", "policy-features-v999"),
        ("red_heroes", ["Wasp", "Bain"]),
        ("expert_identity", "different-expert"),
        ("expert_config", {"iterations": 2}),
    ],
)
def test_main_rejects_mixed_schema_roster_and_expert_provenance(
    tmp_path: Path, field: str, replacement: Any
) -> None:
    rows = _dataset(12)
    rows[-1][field] = replacement
    source = _write(tmp_path / "mixed.jsonl", rows)
    with pytest.raises((ValueError, SystemExit)):
        _main(_args(source, tmp_path / "out.json"))


def test_main_uses_strict_policy_loader(tmp_path: Path) -> None:
    rows = _dataset(12)
    rows[0]["candidates"][0]["target_probability"] = 0.8
    source = _write(tmp_path / "bad-target.jsonl", rows)
    with pytest.raises(ValueError):
        load_policy_examples(source)
    with pytest.raises((ValueError, SystemExit)):
        _main(_args(source, tmp_path / "out.json"))


def test_main_rejects_missing_input_and_insufficient_complete_game_splits(
    tmp_path: Path,
) -> None:
    with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
        _main(_args(tmp_path / "missing.jsonl", tmp_path / "out.json"))
    source = _write(tmp_path / "too-small.jsonl", _dataset(8))
    with pytest.raises((ValueError, SystemExit)):
        _main(_args(source, tmp_path / "out.json"))


def test_split_is_complete_disjoint_deterministic_and_supports_fractions(tmp_path: Path) -> None:
    rows = _dataset()
    source = _write(tmp_path / "policy.jsonl", rows)
    outputs = [tmp_path / f"model-{index}.json" for index in range(3)]
    for out, seed in zip(outputs, (31, 31, 99), strict=True):
        _main(_args(source, out, seed=seed))
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in outputs]
    memberships = [_members(artifact) for artifact in artifacts]
    assert memberships[0] == memberships[1]
    assert memberships[0] != memberships[2]
    train, validation, test = memberships[0].values()
    assert train.isdisjoint(validation | test)
    assert validation.isdisjoint(test)
    assert train | validation | test == {
        (row["game_id"], row["world_seed"]) for row in rows
    }

    fraction_out = tmp_path / "fraction.json"
    fraction_args = _args(source, fraction_out)
    fraction_args[fraction_args.index("--val-games") : fraction_args.index("--val-games") + 2] = [
        "--val-fraction",
        "0.25",
    ]
    fraction_args[
        fraction_args.index("--test-games") : fraction_args.index("--test-games") + 2
    ] = ["--test-fraction", "0.25"]
    _main(fraction_args)
    assert json.loads(fraction_out.read_text())["provenance"]["split_counts"] == {
        "train": 12,
        "validation": 6,
        "test": 6,
    }


def test_artifact_is_deterministic_portable_and_has_complete_provenance(tmp_path: Path) -> None:
    rows = _dataset()
    source = _write(tmp_path / "policy.jsonl", rows)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    _main(_args(source, first))
    _main(_args(source, second))
    assert first.read_bytes() == second.read_bytes()

    artifact = json.loads(first.read_text(encoding="utf-8"))
    assert artifact["model_version"] == "gbm-policy-v1"
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["policy_feature_schema_id"] == POLICY_FEATURE_SCHEMA_ID
    assert artifact["feature_names"] == ["action_is_a", "state_signal"]
    assert artifact["red_roster"] == RED and artifact["blue_roster"] == BLUE
    provenance = artifact["provenance"]
    assert provenance["dataset_digest"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance["source_identities"] == [
        {"source_revision": "policy-test-revision", "dirty_tree_hash": "clean"}
    ]
    assert provenance["expert_identity"] == "ismcts-heuristic-v1"
    assert provenance["expert_config"] == EXPERT_CONFIG
    assert provenance["game_count"] == 24
    assert provenance["row_count"] == 48
    assert provenance["candidate_count"] == 96
    assert provenance["split_counts"] == {"train": 16, "validation": 4, "test": 4}
    assert provenance["selected_params"]["n_estimators"] in (8, 16)
    assert provenance["selected_params"]["max_depth"] in (1, 2)
    assert provenance["selected_params"]["learning_rate"] in (0.1, 0.3)
    for name in ("top1_accuracy", "pairwise_accuracy", "cross_entropy"):
        assert math.isfinite(artifact["metrics"][name])


def test_export_exactly_matches_weighted_sklearn_refit(tmp_path: Path) -> None:
    from sklearn.ensemble import GradientBoostingRegressor

    rows = _dataset()
    artifact = _train(tmp_path, rows)
    memberships = _members(artifact)
    fit_games = memberships["train"] | memberships["validation"]
    x, y, weights = _vectors(rows, fit_games, artifact["feature_names"])
    params = artifact["provenance"]["selected_params"]
    sklearn_model = GradientBoostingRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=artifact["provenance"]["split_seed"],
    ).fit(x, y, sample_weight=weights)
    all_x, _, _ = _vectors(
        rows,
        fit_games | memberships["test"],
        artifact["feature_names"],
    )
    assert [_portable_score(artifact, vector) for vector in all_x] == pytest.approx(
        sklearn_model.predict(all_x), rel=1e-12, abs=1e-12
    )


def test_synthetic_policy_changes_ranking_with_state_and_beats_chance(tmp_path: Path) -> None:
    from automata.search.learned_policy import LearnedPolicy

    artifact = _train(tmp_path, _dataset(32))
    model = LearnedPolicy(artifact)
    assert model is not None
    names = artifact["feature_names"]

    def vector(signal: float, action_is_a: float) -> list[float]:
        values = {"state_signal": signal, "action_is_a": action_is_a}
        return [values.get(name, 0.0) for name in names]

    assert _portable_score(artifact, vector(1.0, 1.0)) > _portable_score(
        artifact, vector(1.0, 0.0)
    )
    assert _portable_score(artifact, vector(-1.0, 0.0)) > _portable_score(
        artifact, vector(-1.0, 1.0)
    )
    assert artifact["metrics"]["top1_accuracy"] > 0.5
    assert artifact["metrics"]["pairwise_accuracy"] > 0.5
