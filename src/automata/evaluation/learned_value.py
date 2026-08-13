"""Portable logistic and gradient-boosted value models — the counterpart to
:class:`HeuristicValue`.

Loads a versioned JSON artifact (from a filesystem path or in-memory mapping)
that carries a logistic-regression model over the fixed feature schema in
:data:`automata.evaluation.features.FEATURE_NAMES`. Inference is pure Python:
standardize the state's feature vector via ``z = (x - mean) / scale``, form
``logit = intercept + sum(coef * z)``, return ``tanh(logit / 2)`` — a finite
scalar in ``[-1, 1]`` matching the :class:`~automata.evaluation.value.ValueFn`
contract.

The artifact is generic over rosters: the trainer / candidate-generator picks
red / blue rosters and stamps them into the artifact. At score time the
runtime rejects states whose rosters diverge from those declared, so an
out-of-distribution score never silently leaks into search.

No numpy / sklearn imports — the module is safe to load in constrained
runtimes (clients, ML-dep-less CI).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

from .features import FEATURE_SCHEMAS, feature_vector

_LOGISTIC_VERSION = "logistic-v1"
_GBM_VERSION = "gbm-v1"
_EXPECTED_SCHEMA_VERSION = 1

_COMMON_FIELDS: tuple[str, ...] = (
    "model_version",
    "schema_version",
    "red_roster",
    "blue_roster",
    "feature_names",
)
_LOGISTIC_FIELDS = (
    *_COMMON_FIELDS,
    "feature_means",
    "feature_scales",
    "coefficients",
    "intercept",
)
_GBM_FIELDS = (*_COMMON_FIELDS, "base_raw_score", "learning_rate", "trees")


class LearnedValue:
    """Portable logistic ValueFn — see module docstring."""

    def __init__(self, source: str | Path | Mapping[str, Any]) -> None:
        artifact = _load_artifact(source)
        required_fields = _validate_artifact(artifact)

        self._model_version = str(artifact["model_version"])
        self._feature_schema = str(artifact.get("feature_schema", "base-v1"))
        self._red_roster: tuple[str, ...] = tuple(artifact["red_roster"])
        self._blue_roster: tuple[str, ...] = tuple(artifact["blue_roster"])
        self._means = tuple(float(x) for x in artifact.get("feature_means", ()))
        self._scales = tuple(float(x) for x in artifact.get("feature_scales", ()))
        self._coefficients = tuple(float(x) for x in artifact.get("coefficients", ()))
        self._intercept = float(artifact.get("intercept", 0.0))
        self._base_raw_score = float(artifact.get("base_raw_score", 0.0))
        self._learning_rate = float(artifact.get("learning_rate", 1.0))
        self._trees: tuple[Mapping[str, Any], ...] = tuple(artifact.get("trees", ()))
        self._digest = _content_digest(artifact, required_fields)

    @property
    def digest(self) -> str:
        """Deterministic 64-char lowercase SHA-256 hex of the artifact content."""
        return self._digest

    @property
    def feature_means(self) -> tuple[float, ...]:
        """Validated training means, exposed for model diagnostics."""
        return self._means

    @property
    def feature_scales(self) -> tuple[float, ...]:
        """Validated training scales, exposed for model diagnostics."""
        return self._scales

    def __call__(self, state: GameState, team: TeamColor) -> float:
        self._check_roster_compat(state)

        feats = feature_vector(state, team, self._feature_schema)
        if self._model_version == _GBM_VERSION:
            raw_score = self._base_raw_score
            for tree in self._trees:
                raw_score += self._learning_rate * _tree_value(tree, feats)
            return math.tanh(raw_score / 2.0)

        logit = self._intercept
        for coef, mean, scale, x in zip(
            self._coefficients, self._means, self._scales, feats, strict=True
        ):
            logit += coef * ((x - mean) / scale)
        return math.tanh(logit / 2.0)

    # -- internals -------------------------------------------------------- #

    def _check_roster_compat(self, state: GameState) -> None:
        actual_red = tuple(h.name for h in state.teams[TeamColor.RED].heroes)
        actual_blue = tuple(h.name for h in state.teams[TeamColor.BLUE].heroes)
        if actual_red != self._red_roster or actual_blue != self._blue_roster:
            raise ValueError(
                "LearnedValue roster mismatch: artifact declared "
                f"red={self._red_roster}, blue={self._blue_roster}; state has "
                f"red={actual_red}, blue={actual_blue}"
            )


# ------------------------------------------------------------------------- #
# Loading
# ------------------------------------------------------------------------- #


def _load_artifact(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load an artifact dict from a Mapping or filesystem path.

    Error policy (consistent with the surrounding package, e.g.
    ``protocol._acquire_writer_lock``): a truly-absent path surfaces as
    ``FileNotFoundError`` so callers can distinguish "no such artifact" from
    "artifact is malformed"; any *other* filesystem or decode failure (a
    directory instead of a file, unreadable target, non-JSON contents,
    non-object JSON) is normalized to ``ValueError`` with the path in the
    message. This mirrors how the tests treat the loader as the single
    validation seam — every artifact-level fault other than pure "missing" is
    a ``ValueError``.
    """
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"LearnedValue artifact not found: {path}")
    if not path.is_file():
        raise ValueError(f"LearnedValue artifact path is not a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"LearnedValue artifact could not be read: {path}") from exc
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LearnedValue artifact is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(
            f"LearnedValue artifact must be a JSON object, got {type(loaded).__name__}: {path}"
        )
    return loaded


# ------------------------------------------------------------------------- #
# Validation
# ------------------------------------------------------------------------- #


def _validate_artifact(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    model_version = artifact.get("model_version")
    if model_version == _LOGISTIC_VERSION:
        required_fields = _LOGISTIC_FIELDS
    elif model_version == _GBM_VERSION:
        required_fields = _GBM_FIELDS
    else:
        raise ValueError(
            "LearnedValue model_version must be one of "
            f"{(_LOGISTIC_VERSION, _GBM_VERSION)!r}, got {model_version!r}"
        )
    missing = [f for f in required_fields if f not in artifact]
    if missing:
        raise ValueError(f"LearnedValue artifact missing required field(s): {missing}")

    if artifact["schema_version"] != _EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"LearnedValue schema_version must be {_EXPECTED_SCHEMA_VERSION!r}, "
            f"got {artifact['schema_version']!r}"
        )

    schema_id = artifact.get("feature_schema", "base-v1")
    if not isinstance(schema_id, str) or schema_id not in FEATURE_SCHEMAS:
        raise ValueError(f"LearnedValue feature_schema is unknown: {schema_id!r}")
    expected_names = FEATURE_SCHEMAS[schema_id].feature_names
    feature_names = artifact["feature_names"]
    if not isinstance(feature_names, list) or tuple(feature_names) != expected_names:
        raise ValueError(
            f"LearnedValue feature_names must equal {list(expected_names)!r} in order, "
            f"got {feature_names!r}"
        )

    if model_version == _LOGISTIC_VERSION:
        n = len(expected_names)
        _validate_finite_vector("feature_means", artifact["feature_means"], n)
        _validate_positive_scale_vector("feature_scales", artifact["feature_scales"], n)
        _validate_finite_vector("coefficients", artifact["coefficients"], n)
        _validate_finite_scalar("intercept", artifact["intercept"])
    else:
        _validate_finite_scalar("base_raw_score", artifact["base_raw_score"])
        _validate_finite_scalar("learning_rate", artifact["learning_rate"])
        if float(artifact["learning_rate"]) <= 0.0:
            raise ValueError("LearnedValue learning_rate must be strictly positive")
        _validate_trees(artifact["trees"], len(expected_names))

    for field in ("red_roster", "blue_roster"):
        roster = artifact[field]
        if not isinstance(roster, list) or not all(isinstance(x, str) for x in roster):
            raise ValueError(f"LearnedValue {field} must be a list of hero-name strings")
    if "feature_schema" in artifact:
        return (*required_fields, "feature_schema")
    return required_fields


def _validate_trees(value: Any, feature_count: int) -> None:
    if not isinstance(value, list):
        raise ValueError("LearnedValue trees must be a list")
    for tree_index, tree in enumerate(value):
        _validate_tree(tree, tree_index, feature_count)


def _validate_tree(tree: Any, tree_index: int, feature_count: int) -> None:
    if not isinstance(tree, dict) or set(tree) != {"root", "nodes"}:
        raise ValueError(f"LearnedValue trees[{tree_index}] is malformed")
    nodes = tree["nodes"]
    root = tree["root"]
    if not isinstance(nodes, list) or not nodes or not _node_index(root, len(nodes)):
        raise ValueError(f"LearnedValue trees[{tree_index}] has an invalid root or nodes")
    visited: set[int] = set()
    active: set[int] = set()

    def visit(index: int) -> None:
        if index in active or index in visited:
            raise ValueError(f"LearnedValue trees[{tree_index}] is not a tree")
        active.add(index)
        node = nodes[index]
        if not isinstance(node, dict):
            raise ValueError(f"LearnedValue trees[{tree_index}] node {index} is malformed")
        if set(node) == {"value"}:
            _validate_finite_scalar(f"trees[{tree_index}].nodes[{index}].value", node["value"])
        elif set(node) == {"feature", "threshold", "left", "right"}:
            feature = node["feature"]
            if not _node_index(feature, feature_count):
                raise ValueError(f"LearnedValue trees[{tree_index}] node feature is invalid")
            _validate_finite_scalar("tree threshold", node["threshold"])
            for child_name in ("left", "right"):
                child = node[child_name]
                if not _node_index(child, len(nodes)):
                    raise ValueError(f"LearnedValue trees[{tree_index}] child is invalid")
                visit(child)
        else:
            raise ValueError(f"LearnedValue trees[{tree_index}] node {index} is malformed")
        active.remove(index)
        visited.add(index)

    visit(root)
    if len(visited) != len(nodes):
        raise ValueError(f"LearnedValue trees[{tree_index}] contains unreachable nodes")


def _node_index(value: Any, length: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < length


def _tree_value(tree: Mapping[str, Any], features: Sequence[float]) -> float:
    nodes = tree["nodes"]
    index = tree["root"]
    while "value" not in nodes[index]:
        node = nodes[index]
        index = node["left"] if features[node["feature"]] < node["threshold"] else node["right"]
    return float(nodes[index]["value"])


def _validate_finite_vector(name: str, value: Any, expected_len: int) -> None:
    if not isinstance(value, list):
        raise ValueError(f"LearnedValue {name} must be a list, got {type(value).__name__}")
    if len(value) != expected_len:
        raise ValueError(f"LearnedValue {name} must have length {expected_len}, got {len(value)}")
    for i, x in enumerate(value):
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(float(x)):
            raise ValueError(f"LearnedValue {name}[{i}] must be a finite number, got {x!r}")


def _validate_positive_scale_vector(name: str, value: Any, expected_len: int) -> None:
    _validate_finite_vector(name, value, expected_len)
    for i, x in enumerate(value):
        if float(x) <= 0.0:
            raise ValueError(f"LearnedValue {name}[{i}] must be strictly positive, got {x!r}")


def _validate_finite_scalar(name: str, value: Any) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"LearnedValue {name} must be a finite number, got {value!r}")


# ------------------------------------------------------------------------- #
# Content digest
# ------------------------------------------------------------------------- #


def _content_digest(artifact: Mapping[str, Any], required_fields: tuple[str, ...]) -> str:
    """SHA-256 hex over canonical JSON of the required content fields.

    Sorted keys + only-required-fields + deterministic separators means the
    digest is independent of load source (path vs. mapping) and stable across
    repeated loads of identical content. Provenance and metrics are deliberately
    excluded: identity follows executable model behavior, not training metadata.
    """
    canonical = {f: artifact[f] for f in required_fields}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
