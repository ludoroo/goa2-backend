"""Train and export a portable gradient-boosted policy model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, cast

from .policy_dataset import (
    POLICY_FEATURE_SCHEMA_ID,
    SCHEMA_VERSION,
    load_policy_examples,
)
from .train_value import _membership, _split_count, _split_games

__all__ = ["main"]

_GameKey = tuple[str, int]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--val-games", type=int)
    validation.add_argument("--val-fraction", type=float, default=0.15)
    test = parser.add_mutually_exclusive_group()
    test.add_argument("--test-games", type=int)
    test.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--estimators-grid", default="50,100,200")
    parser.add_argument("--depth-grid", default="1,2,3")
    parser.add_argument("--learning-rate-grid", default="0.03,0.1")
    return parser


def _grid(raw: str, name: str, *, integers: bool = False) -> list[int] | list[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"--{name}-grid must be a comma-separated numeric list") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"--{name}-grid values must be finite and positive")
    if integers:
        if any(not value.is_integer() for value in values):
            raise ValueError(f"--{name}-grid values must be integers")
        return sorted({int(value) for value in values})
    return sorted(set(values))


def _validate_dataset(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str], str, dict[str, Any], tuple[str, str]]:
    if not rows:
        raise ValueError("policy dataset is empty")
    first = rows[0]
    red = list(first["red_heroes"])
    blue = list(first["blue_heroes"])
    expert_identity = first["expert_identity"]
    expert_config = first["expert_config"]
    source = (first["source_revision"], first["dirty_tree_hash"])
    for row in rows[1:]:
        if row["red_heroes"] != red or row["blue_heroes"] != blue:
            raise ValueError("policy dataset contains inconsistent declared rosters")
        if row["expert_identity"] != expert_identity or row["expert_config"] != expert_config:
            raise ValueError("policy dataset contains inconsistent expert provenance")
        if (row["source_revision"], row["dirty_tree_hash"]) != source:
            raise ValueError("policy dataset contains inconsistent source metadata")
    return red, blue, expert_identity, expert_config, source


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            name
            for row in rows
            for candidate in row["candidates"]
            for name in candidate["features"]
        }
    )


def _arrays(
    rows_by_game: dict[_GameKey, list[dict[str, Any]]],
    keys: list[_GameKey],
    names: list[str],
) -> tuple[list[list[float]], list[float], list[float]]:
    features: list[list[float]] = []
    targets: list[float] = []
    weights: list[float] = []
    game_weight = 1.0 / len(keys)
    for key in keys:
        decisions = rows_by_game[key]
        decision_weight = game_weight / len(decisions)
        for row in decisions:
            candidate_weight = decision_weight / len(row["candidates"])
            for candidate in row["candidates"]:
                sparse = candidate["features"]
                features.append([float(sparse.get(name, 0.0)) for name in names])
                targets.append(float(candidate["target_probability"]))
                weights.append(candidate_weight)
    return features, targets, weights


def _fit(
    features: list[list[float]],
    targets: list[float],
    weights: list[float],
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
) -> Any:
    regressor = importlib.import_module("sklearn.ensemble").GradientBoostingRegressor
    model = regressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
    )
    return model.fit(features, targets, sample_weight=weights)


def _softmax(scores: list[float]) -> list[float]:
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _decision_metrics(targets: list[float], scores: list[float]) -> tuple[float, float, float]:
    probabilities = _softmax(scores)
    cross_entropy = -sum(
        target * math.log(max(probability, 1e-300))
        for target, probability in zip(targets, probabilities, strict=True)
    )
    best_target = max(targets)
    best_score = max(scores)
    predicted = {index for index, value in enumerate(scores) if value == best_score}
    expert = {index for index, value in enumerate(targets) if value == best_target}
    top1 = len(predicted & expert) / len(predicted)

    comparisons: list[float] = []
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            target_delta = targets[left] - targets[right]
            if target_delta == 0.0:
                continue
            score_delta = scores[left] - scores[right]
            comparisons.append(
                1.0 if target_delta * score_delta > 0.0 else 0.5 if score_delta == 0.0 else 0.0
            )
    pairwise = sum(comparisons) / len(comparisons) if comparisons else 0.5
    return cross_entropy, top1, pairwise


def _metrics(
    model: Any,
    rows_by_game: dict[_GameKey, list[dict[str, Any]]],
    keys: list[_GameKey],
    names: list[str],
) -> dict[str, float]:
    game_metrics: list[tuple[float, float, float]] = []
    for key in keys:
        decisions = []
        for row in rows_by_game[key]:
            vectors = [
                [float(candidate["features"].get(name, 0.0)) for name in names]
                for candidate in row["candidates"]
            ]
            scores = [float(value) for value in model.predict(vectors)]
            targets = [float(candidate["target_probability"]) for candidate in row["candidates"]]
            decisions.append(_decision_metrics(targets, scores))
        game_metrics.append(
            (
                sum(item[0] for item in decisions) / len(decisions),
                sum(item[1] for item in decisions) / len(decisions),
                sum(item[2] for item in decisions) / len(decisions),
            )
        )
    return {
        "cross_entropy": sum(item[0] for item in game_metrics) / len(game_metrics),
        "top1_accuracy": sum(item[1] for item in game_metrics) / len(game_metrics),
        "pairwise_accuracy": sum(item[2] for item in game_metrics) / len(game_metrics),
    }


def _export_tree(estimator: Any) -> dict[str, Any]:
    tree = estimator.tree_
    nodes: list[dict[str, Any]] = []
    for index in range(tree.node_count):
        left = int(tree.children_left[index])
        if left == -1:
            nodes.append({"value": float(tree.value[index][0][0])})
        else:
            nodes.append(
                {
                    "feature": int(tree.feature[index]),
                    # sklearn sends equality left; portable traversal uses <.
                    "threshold": math.nextafter(float(tree.threshold[index]), math.inf),
                    "left": left,
                    "right": int(tree.children_right[index]),
                }
            )
    return {"root": 0, "nodes": nodes}


def _portable_score(artifact: dict[str, Any], vector: list[float]) -> float:
    score = float(artifact["base_score"])
    for tree in artifact["trees"]:
        index = tree["root"]
        while "value" not in tree["nodes"][index]:
            node = tree["nodes"][index]
            index = node["left"] if vector[node["feature"]] < node["threshold"] else node["right"]
        score += artifact["learning_rate"] * tree["nodes"][index]["value"]
    return score


def _verify_export(model: Any, artifact: dict[str, Any], features: list[list[float]]) -> None:
    expected = [float(value) for value in model.predict(features)]
    actual = [_portable_score(artifact, vector) for vector in features]
    if any(
        not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        for left, right in zip(actual, expected, strict=True)
    ):
        raise RuntimeError("portable policy GBM export does not match sklearn inference")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw_bytes = args.input.read_bytes()
    rows = load_policy_examples(args.input)
    red, blue, expert_identity, expert_config, source = _validate_dataset(rows)
    names = _feature_names(rows)
    if not names:
        raise ValueError("policy dataset has no features")
    estimators = cast(list[int], _grid(args.estimators_grid, "estimators", integers=True))
    depths = cast(list[int], _grid(args.depth_grid, "depth", integers=True))
    rates = cast(list[float], _grid(args.learning_rate_grid, "learning-rate"))

    rows_by_game: dict[_GameKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_game[(row["game_id"], row["world_seed"])].append(row)
    game_count = len(rows_by_game)
    splits = _split_games(
        list(rows_by_game),
        seed=args.split_seed,
        validation_count=_split_count(game_count, args.val_games, args.val_fraction, "val"),
        test_count=_split_count(game_count, args.test_games, args.test_fraction, "test"),
    )
    train_x, train_y, train_w = _arrays(rows_by_game, splits["train"], names)
    candidates: list[tuple[float, int, int, float]] = []
    for count, depth, rate in product(estimators, depths, rates):
        candidate_model = _fit(
            train_x,
            train_y,
            train_w,
            n_estimators=count,
            max_depth=depth,
            learning_rate=rate,
            seed=args.split_seed,
        )
        candidates.append(
            (
                _metrics(candidate_model, rows_by_game, splits["validation"], names)[
                    "cross_entropy"
                ],
                count,
                depth,
                rate,
            )
        )
    # Tuple ordering intentionally favors the smaller model on equal validation loss.
    _, selected_count, selected_depth, selected_rate = min(candidates)

    fit_keys = [*splits["train"], *splits["validation"]]
    fit_x, fit_y, fit_w = _arrays(rows_by_game, fit_keys, names)
    model = _fit(
        fit_x,
        fit_y,
        fit_w,
        n_estimators=selected_count,
        max_depth=selected_depth,
        learning_rate=selected_rate,
        seed=args.split_seed,
    )
    artifact = {
        "model_version": "gbm-policy-v1",
        "schema_version": SCHEMA_VERSION,
        "policy_feature_schema_id": POLICY_FEATURE_SCHEMA_ID,
        "red_roster": red,
        "blue_roster": blue,
        "feature_names": names,
        "base_score": float(model.init_.constant_.flat[0]),
        "learning_rate": float(model.learning_rate),
        "trees": [_export_tree(estimator) for estimator in model.estimators_[:, 0]],
        "provenance": {
            "dataset_digest": hashlib.sha256(raw_bytes).hexdigest(),
            "source_identities": [
                {"source_revision": source[0], "dirty_tree_hash": source[1]}
            ],
            "expert_identity": expert_identity,
            "expert_config": expert_config,
            "split_seed": args.split_seed,
            "game_count": game_count,
            "row_count": len(rows),
            "candidate_count": sum(len(row["candidates"]) for row in rows),
            "selected_params": {
                "n_estimators": selected_count,
                "max_depth": selected_depth,
                "learning_rate": selected_rate,
            },
            "parameter_grid": {
                "n_estimators": estimators,
                "max_depth": depths,
                "learning_rate": rates,
            },
            "split_counts": {name: len(keys) for name, keys in splits.items()},
            "split_games": {name: _membership(keys) for name, keys in splits.items()},
            "training_target": "target_probability",
            "validation_objective": "decision_policy_cross_entropy",
            "sample_weighting": "equal_game_decision_candidate",
        },
        "metrics": _metrics(model, rows_by_game, splits["test"], names),
    }
    all_x, _, _ = _arrays(rows_by_game, [*fit_keys, *splits["test"]], names)
    _verify_export(model, artifact, all_x)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
