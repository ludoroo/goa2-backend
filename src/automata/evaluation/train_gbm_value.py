"""Train and export a portable gradient-boosted value model."""

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

from .features import FEATURE_SCHEMAS
from .train_value import (
    _arrays,
    _log_loss,
    _membership,
    _metrics,
    _split_count,
    _split_games,
    _validate_rows,
)
from .value_dataset import SCHEMA_VERSION, load_examples

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
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"--{name}-grid values must be finite and positive")
    if integers:
        if any(not value.is_integer() for value in values):
            raise ValueError(f"--{name}-grid values must be integers")
        return [int(value) for value in values]
    return values


def _fit(
    features: list[list[float]],
    labels: list[int],
    weights: list[float],
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
) -> Any:
    if len(set(labels)) != 2:
        raise ValueError("training split must contain both outcome labels")
    classifier = importlib.import_module("sklearn.ensemble").GradientBoostingClassifier
    model = classifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
    )
    return model.fit(features, labels, sample_weight=weights)


def _probabilities(model: Any, features: list[list[float]]) -> list[float]:
    return [float(value) for value in model.predict_proba(features)[:, 1]]


def _export_tree(estimator: Any) -> dict[str, Any]:
    tree = estimator.tree_
    nodes: list[dict[str, Any]] = []
    for index in range(tree.node_count):
        left = int(tree.children_left[index])
        if left == -1:
            nodes.append({"value": float(tree.value[index][0][0])})
        else:
            # sklearn routes equality left; the next float preserves that under
            # the portable artifact's strict-less-than traversal.
            threshold = math.nextafter(float(tree.threshold[index]), math.inf)
            nodes.append(
                {
                    "feature": int(tree.feature[index]),
                    "threshold": threshold,
                    "left": left,
                    "right": int(tree.children_right[index]),
                }
            )
    return {"root": 0, "nodes": nodes}


def _base_raw_score(model: Any) -> float:
    probability = float(model.init_.class_prior_[1])
    return math.log(probability / (1.0 - probability))


def _verify_export(model: Any, artifact: dict[str, Any], features: list[list[float]]) -> None:
    expected = [float(value) for value in model.decision_function(features)]
    for row, target in zip(features, expected, strict=True):
        raw = float(artifact["base_raw_score"])
        for tree in artifact["trees"]:
            index = tree["root"]
            while "value" not in tree["nodes"][index]:
                node = tree["nodes"][index]
                index = node["left"] if row[node["feature"]] < node["threshold"] else node["right"]
            raw += artifact["learning_rate"] * tree["nodes"][index]["value"]
        if not math.isclose(raw, target, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError("portable GBM export does not match sklearn inference")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw_bytes = args.input.read_bytes()
    rows = load_examples(args.input)
    red_roster, blue_roster, feature_schema = _validate_rows(rows)
    feature_names = list(FEATURE_SCHEMAS[feature_schema].feature_names)
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
    train_x, train_y, train_w = _arrays(rows_by_game, splits["train"])
    val_x, val_y, val_w = _arrays(rows_by_game, splits["validation"])
    candidates: list[tuple[float, int, int, float]] = []
    for count, depth, rate in product(estimators, depths, rates):
        model = _fit(
            train_x,
            train_y,
            train_w,
            n_estimators=count,
            max_depth=depth,
            learning_rate=rate,
            seed=args.split_seed,
        )
        candidates.append(
            (_log_loss(val_y, _probabilities(model, val_x), val_w), count, depth, rate)
        )
    _, selected_count, selected_depth, selected_rate = min(candidates)

    final_keys = [*splits["train"], *splits["validation"]]
    final_x, final_y, final_w = _arrays(rows_by_game, final_keys)
    model = _fit(
        final_x,
        final_y,
        final_w,
        n_estimators=selected_count,
        max_depth=selected_depth,
        learning_rate=selected_rate,
        seed=args.split_seed,
    )
    test_x, test_y, test_w = _arrays(rows_by_game, splits["test"])
    identities = sorted({(row["source_revision"], row["dirty_tree_hash"]) for row in rows})
    artifact = {
        "model_version": "gbm-v1",
        "schema_version": SCHEMA_VERSION,
        "red_roster": red_roster,
        "blue_roster": blue_roster,
        "feature_names": feature_names,
        "base_raw_score": _base_raw_score(model),
        "learning_rate": float(model.learning_rate),
        "trees": [_export_tree(estimator) for estimator in model.estimators_[:, 0]],
        "provenance": {
            "dataset_digest": hashlib.sha256(raw_bytes).hexdigest(),
            "source_identities": [
                {"source_revision": revision, "dirty_tree_hash": tree_hash}
                for revision, tree_hash in identities
            ],
            "split_seed": args.split_seed,
            "game_count": game_count,
            "row_count": len(rows),
            "selected_params": {
                "n_estimators": selected_count,
                "max_depth": selected_depth,
                "learning_rate": selected_rate,
            },
            "split_counts": {name: len(keys) for name, keys in splits.items()},
            "split_games": {name: _membership(keys) for name, keys in splits.items()},
        },
        "metrics": _metrics(test_y, _probabilities(model, test_x), test_w),
    }
    if feature_schema != "base-v1":
        artifact["feature_schema"] = feature_schema
    _verify_export(model, artifact, [*final_x, *test_x])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
