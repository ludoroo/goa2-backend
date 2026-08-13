"""Compact JSONL telemetry for nonterminal search cutoffs."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

from .features import FEATURE_NAMES, feature_vector
from .learned_value import LearnedValue
from .value import HeuristicValue

SCHEMA_VERSION: int = 1


def _normalized(value: float, name: str) -> float:
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError(f"{name} must be a finite value in [-1, 1], got {value!r}")
    return value


class CutoffTelemetryRecorder:
    """Append deterministic, immediately durable cutoff comparisons to JSONL."""

    def __init__(self, path: str | Path, case_metadata: dict[str, Any]) -> None:
        self._path = Path(path)
        self._case_metadata = dict(case_metadata)
        self._heuristic = HeuristicValue()

    def __call__(self, state: GameState, team: TeamColor, active_value: float) -> None:
        active = _normalized(active_value, "active_value")
        heuristic = _normalized(self._heuristic(state, team), "heuristic_value")
        row = {
            "schema_version": SCHEMA_VERSION,
            "case_metadata": self._case_metadata,
            "team": team.value,
            "round": state.round,
            "feature_names": list(FEATURE_NAMES),
            "features": feature_vector(state, team),
            "active_value": active,
            "heuristic_value": heuristic,
            "difference": active - heuristic,
        }
        encoded = json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
        with self._path.open("a", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return result


def _validated_row(row: Any, index: int) -> tuple[list[float], float, float, float]:
    prefix = f"telemetry row {index}"
    if not isinstance(row, Mapping):
        raise ValueError(f"{prefix} must be an object")
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{prefix} has incompatible schema_version")
    if row.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError(f"{prefix} has incompatible feature_names")

    raw_features = row.get("features")
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        raise ValueError(f"{prefix} features must be a sequence")
    if len(raw_features) != len(FEATURE_NAMES):
        raise ValueError(
            f"{prefix} has {len(raw_features)} features; expected {len(FEATURE_NAMES)}"
        )
    features = [
        _finite_number(value, f"{prefix} features[{feature_index}]")
        for feature_index, value in enumerate(raw_features)
    ]
    active = _finite_number(row.get("active_value"), f"{prefix} active_value")
    heuristic = _finite_number(row.get("heuristic_value"), f"{prefix} heuristic_value")
    difference = _finite_number(row.get("difference"), f"{prefix} difference")
    return features, active, heuristic, difference


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    sum_sq_x = sum(value * value for value in centered_x)
    sum_sq_y = sum(value * value for value in centered_y)
    if sum_sq_x == 0.0 or sum_sq_y == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / math.sqrt(
        sum_sq_x * sum_sq_y
    )


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def analyze(
    rows: Iterable[Mapping[str, Any]], artifact: str | Path | Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize cutoff comparisons and model-feature distribution drift."""
    model = LearnedValue(artifact)
    validated = [_validated_row(row, index) for index, row in enumerate(rows, start=1)]
    if not validated:
        raise ValueError("telemetry rows must not be empty")

    features = [row[0] for row in validated]
    active = [row[1] for row in validated]
    heuristic = [row[2] for row in validated]
    differences = [row[3] for row in validated]
    count = len(validated)

    abs_z_by_feature: list[list[float]] = [[] for _ in FEATURE_NAMES]
    any_feature_ood = 0
    for feature_row in features:
        row_is_ood = False
        for index, (value, mean, scale) in enumerate(
            zip(feature_row, model.feature_means, model.feature_scales, strict=True)
        ):
            abs_z = abs((value - mean) / scale)
            abs_z_by_feature[index].append(abs_z)
            row_is_ood |= abs_z > 3.0
        any_feature_ood += row_is_ood

    per_feature = {
        name: {
            "mean_abs_z": sum(values) / count,
            "max_abs_z": max(values),
            "p95_abs_z": _nearest_rank(values, 0.95),
        }
        for name, values in zip(FEATURE_NAMES, abs_z_by_feature, strict=True)
    }
    return {
        "row_count": count,
        "pearson_correlation": _pearson(active, heuristic),
        "mean_abs_difference": sum(abs(value) for value in differences) / count,
        "sign_disagreement_rate": sum(
            left * right < 0.0 for left, right in zip(active, heuristic, strict=True)
        )
        / count,
        "active_saturation_rate": sum(abs(value) >= 0.95 for value in active) / count,
        "heuristic_saturation_rate": sum(abs(value) >= 0.95 for value in heuristic) / count,
        "any_feature_ood_rate": any_feature_ood / count,
        "per_feature": per_feature,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze search-cutoff telemetry")
    parser.add_argument("--input", type=Path, required=True, help="cutoff JSONL input")
    parser.add_argument("--model", type=Path, required=True, help="LearnedValue artifact")
    parser.add_argument("--out", type=Path, help="write summary instead of printing it")
    args = parser.parse_args(argv)

    with args.input.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    encoded = json.dumps(analyze(rows, args.model), sort_keys=True)
    if args.out is None:
        print(encoded)
    else:
        args.out.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
