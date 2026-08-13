"""Train and export a portable logistic value model from ValueDataset JSONL."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .features import FEATURE_SCHEMAS
from .value_dataset import SCHEMA_VERSION, load_examples

__all__ = ["main"]

_MODEL_VERSION = "logistic-v1"
_TEAMS = {"RED", "BLUE"}
_REQUIRED_FIELDS = (
    "schema_version",
    "game_id",
    "world_seed",
    "team",
    "features",
    "feature_names",
    "winner",
    "red_heroes",
    "blue_heroes",
    "source_revision",
    "dirty_tree_hash",
)
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
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0")
    return parser


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) for item in value)


def _validate_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], str]:
    if not rows:
        raise ValueError("value dataset is empty")

    feature_schema: str | None = None
    red_roster: list[str] | None = None
    blue_roster: list[str] | None = None
    by_game_id: dict[str, int] = {}
    by_seed: dict[int, str] = {}
    game_metadata: dict[_GameKey, tuple[tuple[str, str], str, str | None]] = {}
    sample_ids: set[str] = set()
    labels: set[int] = set()

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be a JSON object")
        missing = [field for field in _REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {index} is missing fields: {missing}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"row {index} has unsupported schema_version")
        row_schema = row.get("feature_schema", "base-v1")
        if not isinstance(row_schema, str) or row_schema not in FEATURE_SCHEMAS:
            raise ValueError(f"row {index} has an unknown feature schema")
        if feature_schema is None:
            feature_schema = row_schema
        elif row_schema != feature_schema:
            raise ValueError("dataset contains mixed feature schemas")
        expected_features = list(FEATURE_SCHEMAS[row_schema].feature_names)
        if row["feature_names"] != expected_features:
            raise ValueError(f"row {index} has an incompatible feature schema")
        features = row["features"]
        if (
            not isinstance(features, list)
            or len(features) != len(expected_features)
            or not all(_finite_number(value) for value in features)
        ):
            raise ValueError(f"row {index} must contain {len(expected_features)} finite features")
        if row["team"] not in _TEAMS or row["winner"] not in _TEAMS:
            raise ValueError(f"row {index} must declare valid team and terminal winner")
        if not isinstance(row["game_id"], str) or not row["game_id"]:
            raise ValueError(f"row {index} has an invalid game_id")
        if not isinstance(row["world_seed"], int) or isinstance(row["world_seed"], bool):
            raise ValueError(f"row {index} has an invalid world_seed")
        if not _string_list(row["red_heroes"]) or not _string_list(row["blue_heroes"]):
            raise ValueError(f"row {index} has an invalid declared roster")
        for field in ("source_revision", "dirty_tree_hash"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"row {index} has an invalid {field}")

        sample_id = row.get("sample_id")
        if "sample_id" in row:
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"row {index} has an invalid sample_id")
            if sample_id in sample_ids:
                raise ValueError(f"row {index} has a duplicate sample_id")
            sample_ids.add(sample_id)
            row_kind = "cutoff"
        else:
            row_kind = "natural"

        current_red = list(row["red_heroes"])
        current_blue = list(row["blue_heroes"])
        if red_roster is None:
            red_roster, blue_roster = current_red, current_blue
        elif current_red != red_roster or current_blue != blue_roster:
            raise ValueError("dataset contains inconsistent declared rosters")

        game_id = row["game_id"]
        world_seed = row["world_seed"]
        if game_id in by_game_id and by_game_id[game_id] != world_seed:
            raise ValueError("one game_id maps to multiple world seeds")
        if world_seed in by_seed and by_seed[world_seed] != game_id:
            raise ValueError("one world seed maps to multiple game_ids")
        by_game_id[game_id] = world_seed
        by_seed[world_seed] = game_id

        key = (game_id, world_seed)
        metadata = (
            (row["source_revision"], row["dirty_tree_hash"]),
            row_kind,
            row["winner"] if row_kind == "natural" else None,
        )
        if key in game_metadata and game_metadata[key] != metadata:
            raise ValueError("rows for one game have inconsistent kind, winner, or source identity")
        game_metadata[key] = metadata
        labels.add(int(row["team"] == row["winner"]))

    if labels != {0, 1}:
        raise ValueError("dataset must contain both team-perspective outcome labels")
    assert red_roster is not None and blue_roster is not None
    assert feature_schema is not None
    return red_roster, blue_roster, feature_schema


def _parse_grid(raw: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--c-grid must be a comma-separated list of numbers") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("--c-grid values must be finite and positive")
    return values


def _split_count(total: int, count: int | None, fraction: float, name: str) -> int:
    if count is not None:
        if count < 1:
            raise ValueError(f"--{name}-games must be positive")
        return count
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"--{name}-fraction must be between zero and one")
    return max(1, int(total * fraction))


def _split_games(
    keys: list[_GameKey], *, seed: int, validation_count: int, test_count: int
) -> dict[str, list[_GameKey]]:
    if validation_count + test_count >= len(keys):
        raise ValueError(
            "validation and test splits leave no training games: "
            f"total={len(keys)}, validation={validation_count}, test={test_count}"
        )
    shuffled = sorted(keys)
    random.Random(seed).shuffle(shuffled)
    test = shuffled[:test_count]
    validation = shuffled[test_count : test_count + validation_count]
    train = shuffled[test_count + validation_count :]
    return {"train": train, "validation": validation, "test": test}


def _arrays(
    rows_by_game: dict[_GameKey, list[dict[str, Any]]], keys: list[_GameKey]
) -> tuple[list[list[float]], list[int], list[float]]:
    features: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    for key in keys:
        game_rows = rows_by_game[key]
        row_weight = 1.0 / len(game_rows)
        for row in game_rows:
            features.append([float(value) for value in row["features"]])
            labels.append(int(row["winner"] == row["team"]))
            weights.append(row_weight)
    return features, labels, weights


def _fit_model(
    features: list[list[float]], labels: list[int], weights: list[float], c_value: float
) -> tuple[Any, Any]:
    # Training dependencies remain isolated from import-time runtime dependencies.
    LogisticRegression = importlib.import_module("sklearn.linear_model").LogisticRegression
    StandardScaler = importlib.import_module("sklearn.preprocessing").StandardScaler

    if len(set(labels)) != 2:
        raise ValueError("training split must contain both outcome labels")
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features, sample_weight=weights)
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=1000)
    model.fit(transformed, labels, sample_weight=weights)
    return scaler, model


def _log_loss(labels: list[int], probabilities: list[float], weights: list[float]) -> float:
    epsilon = 1e-15
    losses = [
        -(label * math.log(min(max(probability, epsilon), 1.0 - epsilon)))
        - (1 - label) * math.log(min(max(1.0 - probability, epsilon), 1.0 - epsilon))
        for label, probability in zip(labels, probabilities, strict=True)
    ]
    return sum(loss * weight for loss, weight in zip(losses, weights, strict=True)) / sum(weights)


def _probabilities(scaler: Any, model: Any, features: list[list[float]]) -> list[float]:
    # Labels are encoded as 0/1, so class column 1 is the positive-outcome probability.
    return [float(value) for value in model.predict_proba(scaler.transform(features))[:, 1]]


def _metrics(
    labels: list[int], probabilities: list[float], weights: list[float]
) -> dict[str, float]:
    total_weight = sum(weights)
    brier = (
        sum(
            weight * (probability - label) ** 2
            for label, probability, weight in zip(labels, probabilities, weights, strict=True)
        )
        / total_weight
    )
    accuracy = (
        sum(
            weight * int((probability >= 0.5) == bool(label))
            for label, probability, weight in zip(labels, probabilities, weights, strict=True)
        )
        / total_weight
    )
    bin_totals = [0.0] * 10
    bin_confidence = [0.0] * 10
    bin_accuracy = [0.0] * 10
    for label, probability, weight in zip(labels, probabilities, weights, strict=True):
        index = min(int(probability * 10), 9)
        bin_totals[index] += weight
        bin_confidence[index] += weight * probability
        bin_accuracy[index] += weight * label
    ece = 0.0
    for count, confidence, observed in zip(bin_totals, bin_confidence, bin_accuracy, strict=True):
        if count:
            ece += count / total_weight * abs(confidence / count - observed / count)
    return {
        "test_log_loss": float(_log_loss(labels, probabilities, weights)),
        "brier": float(brier),
        "accuracy": float(accuracy),
        "ece": float(ece),
    }


def _membership(keys: list[_GameKey]) -> list[dict[str, Any]]:
    return [{"game_id": game_id, "world_seed": seed} for game_id, seed in sorted(keys)]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw_bytes = args.input.read_bytes()
    rows = load_examples(args.input)
    red_roster, blue_roster, feature_schema = _validate_rows(rows)
    feature_names = list(FEATURE_SCHEMAS[feature_schema].feature_names)
    c_grid = _parse_grid(args.c_grid)

    rows_by_game: dict[_GameKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_game[(row["game_id"], row["world_seed"])].append(row)
    game_count = len(rows_by_game)
    validation_count = _split_count(game_count, args.val_games, args.val_fraction, "val")
    test_count = _split_count(game_count, args.test_games, args.test_fraction, "test")
    splits = _split_games(
        list(rows_by_game),
        seed=args.split_seed,
        validation_count=validation_count,
        test_count=test_count,
    )

    train_x, train_y, train_w = _arrays(rows_by_game, splits["train"])
    validation_x, validation_y, validation_w = _arrays(rows_by_game, splits["validation"])
    candidates: list[tuple[float, float]] = []
    for c_value in c_grid:
        scaler, model = _fit_model(train_x, train_y, train_w, c_value)
        loss = _log_loss(validation_y, _probabilities(scaler, model, validation_x), validation_w)
        candidates.append((loss, c_value))
    selected_c = min(candidates, key=lambda item: (item[0], item[1]))[1]

    final_keys = [*splits["train"], *splits["validation"]]
    final_x, final_y, final_w = _arrays(rows_by_game, final_keys)
    scaler, model = _fit_model(final_x, final_y, final_w, selected_c)
    test_x, test_y, test_w = _arrays(rows_by_game, splits["test"])
    test_probabilities = _probabilities(scaler, model, test_x)

    identities = sorted({(row["source_revision"], row["dirty_tree_hash"]) for row in rows})
    artifact = {
        "model_version": _MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "red_roster": red_roster,
        "blue_roster": blue_roster,
        "feature_names": feature_names,
        "feature_means": [float(value) for value in scaler.mean_.tolist()],
        "feature_scales": [float(value) for value in scaler.scale_.tolist()],
        "coefficients": [float(value) for value in model.coef_[0].tolist()],
        "intercept": float(model.intercept_[0]),
        "provenance": {
            # Hash the source bytes so provenance identifies the exact JSONL content.
            "dataset_digest": hashlib.sha256(raw_bytes).hexdigest(),
            "source_identities": [
                {"source_revision": revision, "dirty_tree_hash": tree_hash}
                for revision, tree_hash in identities
            ],
            "split_seed": args.split_seed,
            "game_count": game_count,
            "row_count": len(rows),
            "selected_c": float(selected_c),
            "split_counts": {name: len(keys) for name, keys in splits.items()},
            "split_games": {name: _membership(keys) for name, keys in splits.items()},
        },
        "metrics": _metrics(test_y, test_probabilities, test_w),
    }
    if feature_schema != "base-v1":
        artifact["feature_schema"] = feature_schema
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
