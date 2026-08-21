"""Portable pure-Python gradient-boosted policy prior."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

from .ismcts import Decision
from .node import Key
from .policy_features import POLICY_FEATURE_SCHEMA_ID, policy_candidate_features
from .prior import PolicyResult

_MODEL_VERSION = "gbm-policy-v1"
_SCHEMA_VERSION = 1
_EXECUTABLE_FIELDS = (
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


class LearnedPolicy:
    """Rank legal actions using a portable gradient-boosted tree artifact.

    ``gbm-policy-v1`` outputs are raw ranking scores, not calibrated probability
    logits. Initial evaluation therefore uses them only for expansion ordering
    with ``puct_c=0``. PUCT use requires calibration or listwise training.
    """

    def __init__(self, source: str | Path | Mapping[str, Any]) -> None:
        artifact = _load_artifact(source)
        _validate_artifact(artifact)
        self._red_roster = tuple(artifact["red_roster"])
        self._blue_roster = tuple(artifact["blue_roster"])
        self._feature_names = tuple(artifact["feature_names"])
        self._base_score = float(artifact["base_score"])
        self._learning_rate = float(artifact["learning_rate"])
        self._trees: tuple[Mapping[str, Any], ...] = tuple(artifact["trees"])
        self._digest = _content_digest(artifact)

    @property
    def digest(self) -> str:
        """Canonical SHA-256 digest of executable artifact fields."""
        return self._digest

    def __call__(
        self, state: GameState, decision: Decision, legal: list[Key]
    ) -> PolicyResult:
        self._check_roster_compat(state)
        candidates = policy_candidate_features(state, decision, legal)
        weights: dict[Key, float] = {}
        for key in legal:
            try:
                sparse = candidates[key]
            except KeyError as exc:
                raise ValueError(f"LearnedPolicy features missing legal key {key!r}") from exc
            features = tuple(float(sparse.get(name, 0.0)) for name in self._feature_names)
            score = self._base_score
            for tree in self._trees:
                score += self._learning_rate * _tree_value(tree, features)
            if not math.isfinite(score):
                raise ValueError(f"LearnedPolicy produced a non-finite score for {key!r}")
            weights[key] = score
        order = sorted(legal, key=weights.__getitem__, reverse=True)
        return PolicyResult(order=order, weights=weights)

    def _check_roster_compat(self, state: GameState) -> None:
        actual_red = tuple(hero.name for hero in state.teams[TeamColor.RED].heroes)
        actual_blue = tuple(hero.name for hero in state.teams[TeamColor.BLUE].heroes)
        if actual_red != self._red_roster or actual_blue != self._blue_roster:
            raise ValueError(
                "LearnedPolicy roster mismatch: artifact declared "
                f"red={self._red_roster}, blue={self._blue_roster}; state has "
                f"red={actual_red}, blue={actual_blue}"
            )


def _load_artifact(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"LearnedPolicy artifact not found: {path}")
    if not path.is_file():
        raise ValueError(f"LearnedPolicy artifact path is not a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"LearnedPolicy artifact could not be read: {path}") from exc
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LearnedPolicy artifact is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(
            "LearnedPolicy artifact must be a JSON object, got "
            f"{type(loaded).__name__}: {path}"
        )
    return loaded


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in _EXECUTABLE_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"LearnedPolicy artifact missing required field(s): {missing}")
    if artifact["model_version"] != _MODEL_VERSION:
        raise ValueError(f"LearnedPolicy model_version must be {_MODEL_VERSION!r}")
    schema = artifact["schema_version"]
    if not isinstance(schema, int) or isinstance(schema, bool) or schema != _SCHEMA_VERSION:
        raise ValueError(f"LearnedPolicy schema_version must be {_SCHEMA_VERSION}")
    if artifact["policy_feature_schema_id"] != POLICY_FEATURE_SCHEMA_ID:
        raise ValueError("LearnedPolicy policy_feature_schema_id is unknown")

    names = artifact["feature_names"]
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("LearnedPolicy feature_names must be unique nonempty strings")

    for field in ("red_roster", "blue_roster"):
        roster = artifact[field]
        if not isinstance(roster, list) or not all(isinstance(x, str) for x in roster):
            raise ValueError(f"LearnedPolicy {field} must be a list of hero-name strings")

    _validate_finite_scalar("base_score", artifact["base_score"])
    _validate_finite_scalar("learning_rate", artifact["learning_rate"])
    if float(artifact["learning_rate"]) <= 0.0:
        raise ValueError("LearnedPolicy learning_rate must be strictly positive")
    _validate_trees(artifact["trees"], len(names))


def _validate_trees(value: Any, feature_count: int) -> None:
    if not isinstance(value, list):
        raise ValueError("LearnedPolicy trees must be a list")
    for tree_index, tree in enumerate(value):
        _validate_tree(tree, tree_index, feature_count)


def _validate_tree(tree: Any, tree_index: int, feature_count: int) -> None:
    if not isinstance(tree, dict) or set(tree) != {"root", "nodes"}:
        raise ValueError(f"LearnedPolicy trees[{tree_index}] is malformed")
    nodes = tree["nodes"]
    root = tree["root"]
    if not isinstance(nodes, list) or not nodes or not _node_index(root, len(nodes)):
        raise ValueError(f"LearnedPolicy trees[{tree_index}] has an invalid root or nodes")
    visited: set[int] = set()
    active: set[int] = set()

    def visit(index: int) -> None:
        if index in active or index in visited:
            raise ValueError(f"LearnedPolicy trees[{tree_index}] is not a tree")
        active.add(index)
        node = nodes[index]
        if not isinstance(node, dict):
            raise ValueError(f"LearnedPolicy trees[{tree_index}] node {index} is malformed")
        if set(node) == {"value"}:
            _validate_finite_scalar(f"trees[{tree_index}].nodes[{index}].value", node["value"])
        elif set(node) == {"feature", "threshold", "left", "right"}:
            if not _node_index(node["feature"], feature_count):
                raise ValueError(f"LearnedPolicy trees[{tree_index}] feature is invalid")
            _validate_finite_scalar("tree threshold", node["threshold"])
            for child_name in ("left", "right"):
                child = node[child_name]
                if not _node_index(child, len(nodes)):
                    raise ValueError(f"LearnedPolicy trees[{tree_index}] child is invalid")
                visit(child)
        else:
            raise ValueError(f"LearnedPolicy trees[{tree_index}] node {index} is malformed")
        active.remove(index)
        visited.add(index)

    visit(root)
    if len(visited) != len(nodes):
        raise ValueError(f"LearnedPolicy trees[{tree_index}] contains unreachable nodes")


def _node_index(value: Any, length: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < length


def _tree_value(tree: Mapping[str, Any], features: Sequence[float]) -> float:
    nodes = tree["nodes"]
    index = tree["root"]
    while "value" not in nodes[index]:
        node = nodes[index]
        index = node["left"] if features[node["feature"]] < node["threshold"] else node["right"]
    return float(nodes[index]["value"])


def _validate_finite_scalar(name: str, value: Any) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"LearnedPolicy {name} must be a finite number, got {value!r}")


def _content_digest(artifact: Mapping[str, Any]) -> str:
    canonical = {field: artifact[field] for field in _EXECUTABLE_FIELDS}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = ["LearnedPolicy"]
