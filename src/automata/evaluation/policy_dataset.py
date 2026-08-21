"""Compact, deterministic expert-search policy datasets."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from automata.search.ismcts import Decision
from automata.search.node import Key
from automata.search.observation import RootSearchObservation
from automata.search.policy_features import (
    POLICY_FEATURE_SCHEMA_ID as SEARCH_POLICY_FEATURE_SCHEMA_ID,
)
from automata.search.policy_features import policy_candidate_features
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

SCHEMA_VERSION: Final[int] = 1
POLICY_FEATURE_SCHEMA_ID: Final[str] = SEARCH_POLICY_FEATURE_SCHEMA_ID


@dataclass
class PolicyDatasetStats:
    """Counts rows and games written or deliberately discarded."""

    recorded_decisions: int = 0
    skipped_decisions: int = 0
    recorded_games: int = 0
    skipped_games: int = 0


class PolicyDatasetRecorder:
    """Extract root policy examples immediately and flush completed games."""

    def __init__(
        self,
        path: str | Path,
        *,
        game_id: str,
        world_seed: int,
        red_heroes: list[str],
        blue_heroes: list[str],
        source_revision: str,
        dirty_tree_hash: str,
        expert_config: dict[str, Any],
        expert_identity: str,
        append: bool = False,
    ) -> None:
        self._path = Path(path)
        self._game_id = game_id
        self._world_seed = int(world_seed)
        self._red_heroes = list(red_heroes)
        self._blue_heroes = list(blue_heroes)
        self._source_revision = source_revision
        self._dirty_tree_hash = dirty_tree_hash
        self._expert_config = _json_copy(expert_config, "expert_config")
        self._expert_identity = expert_identity
        self._append = bool(append)
        self._buffer: list[dict[str, Any]] = []
        self._next_decision_index = 0
        self._stats = PolicyDatasetStats()

    def __call__(
        self, state: GameState, observation: RootSearchObservation
    ) -> None:
        """Synchronously extract one root observation from borrowed state."""
        decision_index = self._next_decision_index
        self._next_decision_index += 1
        legal = observation.legal_keys

        # Keys are part of the public schema; unsupported key types are a
        # programming error rather than a silently unusable training row.
        encoded_legal = [_encode_key(key) for key in legal]
        encoded_chosen = _encode_key(observation.chosen_key)

        hero = state.get_hero(HeroID(observation.decision_owner_hero_id))
        if hero is None or hero.team is None:
            self._stats.skipped_decisions += 1
            return
        if observation.chosen_key not in legal or len(set(legal)) != len(legal):
            self._stats.skipped_decisions += 1
            return

        stats = []
        total_visits = 0
        for key in legal:
            child = observation.child_stats.get(key)
            if child is None or not _valid_child_stats(
                child.visits, child.total_value, child.q
            ):
                self._stats.skipped_decisions += 1
                return
            stats.append(child)
            total_visits += child.visits
        if total_visits <= 0:
            self._stats.skipped_decisions += 1
            return

        decision = Decision(
            observation.decision_kind,
            hero=hero if observation.decision_kind == "CARD" else None,
            request=observation.request,
        )
        try:
            extracted = policy_candidate_features(state, decision, legal)
            candidates = []
            for key, encoded_key, child in zip(
                legal, encoded_legal, stats, strict=True
            ):
                features = {
                    name: float(value)
                    for name, value in sorted(extracted[key].items())
                }
                if not all(math.isfinite(value) for value in features.values()):
                    raise ValueError("non-finite policy feature")
                candidates.append(
                    {
                        "key": encoded_key,
                        "features": features,
                        "visits": child.visits,
                        "total_value": float(child.total_value),
                        "q": float(child.q),
                        "target_probability": child.visits / total_visits,
                    }
                )
        except (KeyError, ValueError):
            self._stats.skipped_decisions += 1
            return

        request_type = (
            observation.request.request_type.value
            if observation.request is not None
            else None
        )
        self._buffer.append(
            {
                "schema_version": SCHEMA_VERSION,
                "policy_feature_schema_id": POLICY_FEATURE_SCHEMA_ID,
                "game_id": self._game_id,
                "world_seed": self._world_seed,
                "decision_index": decision_index,
                "owner_hero_id": observation.decision_owner_hero_id,
                "team": hero.team.value,
                "decision_kind": observation.decision_kind,
                "request_type": request_type,
                "legal_keys": encoded_legal,
                "chosen_key": encoded_chosen,
                "candidates": candidates,
                "red_heroes": list(self._red_heroes),
                "blue_heroes": list(self._blue_heroes),
                "source_revision": self._source_revision,
                "dirty_tree_hash": self._dirty_tree_hash,
                "expert_config": self._expert_config,
                "expert_identity": self._expert_identity,
            }
        )

    def record_decision(
        self,
        *,
        state: GameState,
        team: str,
        decision_kind: str,
        player_id: str,
        legal_keys: list[Any],
        chosen_key: Any,
    ) -> None:
        """Trajectory-recorder seam; root observations are authoritative."""
        del state, team, decision_kind, player_id, legal_keys, chosen_key

    def record_outcome(self, *, winner: str | None, rounds: int, reason: str) -> None:
        """Flush only a complete game; otherwise discard its whole buffer."""
        del winner, rounds
        buffered, self._buffer = self._buffer, []
        if reason != "game_over" or not buffered:
            self._stats.skipped_decisions += len(buffered)
            self._stats.skipped_games += 1
            return
        self._write_rows(buffered)
        self._stats.recorded_decisions += len(buffered)
        self._stats.recorded_games += 1

    @property
    def stats(self) -> PolicyDatasetStats:
        return self._stats

    def close(self) -> None:
        return None

    def __enter__(self) -> PolicyDatasetRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a" if self._append else "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._append = True


def load_policy_examples(path: str | Path) -> list[dict[str, Any]]:
    """Load and strictly validate compact policy JSONL rows."""
    result: list[dict[str, Any]] = []
    identities: dict[str, tuple[Any, ...]] = {}
    indices: set[tuple[str, int]] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                _validate_row(row, identities, indices)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid policy row {line_number}: {exc}") from exc
            result.append(row)
    return result


def _validate_row(
    row: dict[str, Any],
    identities: dict[str, tuple[Any, ...]],
    indices: set[tuple[str, int]],
) -> None:
    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if row["policy_feature_schema_id"] != POLICY_FEATURE_SCHEMA_ID:
        raise ValueError("unsupported policy feature schema")
    game_id = _exact(row["game_id"], str, "game_id")
    index = _exact(row["decision_index"], int, "decision_index")
    _exact(row["world_seed"], int, "world_seed")
    _exact(row["owner_hero_id"], str, "owner_hero_id")
    if row["team"] not in ("RED", "BLUE"):
        raise ValueError("team must be RED or BLUE")
    if row["decision_kind"] not in ("CARD", "INPUT"):
        raise ValueError("decision_kind must be CARD or INPUT")
    request_type = row["request_type"]
    if row["decision_kind"] == "CARD" and request_type is not None:
        raise ValueError("CARD rows cannot have request_type")
    if row["decision_kind"] == "INPUT" and type(request_type) is not str:
        raise ValueError("INPUT rows require request_type")
    for name in ("red_heroes", "blue_heroes"):
        heroes = row[name]
        if not isinstance(heroes, list) or not all(type(hero) is str for hero in heroes):
            raise ValueError(f"{name} must be a string list")
    for name in ("source_revision", "dirty_tree_hash", "expert_identity"):
        _exact(row[name], str, name)
    if not isinstance(row["expert_config"], dict):
        raise ValueError("expert_config must be an object")
    if index < 0 or (game_id, index) in indices:
        raise ValueError("duplicate or negative decision_index")
    indices.add((game_id, index))

    identity = tuple(
        _canonical(row[name])
        for name in (
            "world_seed", "red_heroes", "blue_heroes", "source_revision",
            "dirty_tree_hash", "expert_config", "expert_identity",
        )
    )
    if game_id in identities and identities[game_id] != identity:
        raise ValueError("inconsistent game identity or provenance")
    identities[game_id] = identity

    legal = row["legal_keys"]
    candidates = row["candidates"]
    if not isinstance(legal, list) or not legal or not isinstance(candidates, list):
        raise ValueError("legal_keys and candidates must be non-empty lists")
    decoded = [_decode_key(key) for key in legal]
    if len({_canonical(key) for key in legal}) != len(legal):
        raise ValueError("duplicate legal key")
    for encoded, value in zip(legal, decoded, strict=True):
        if _encode_key(value) != encoded:
            raise ValueError("action key does not round-trip")
    chosen = row["chosen_key"]
    if _encode_key(_decode_key(chosen)) != chosen:
        raise ValueError("chosen action key does not round-trip")
    if _canonical(chosen) not in {_canonical(key) for key in legal}:
        raise ValueError("chosen_key is not legal")
    if len(candidates) != len(legal):
        raise ValueError("candidate/legal length mismatch")

    probabilities = []
    visits = []
    for expected_key, candidate in zip(legal, candidates, strict=True):
        if not isinstance(candidate, dict) or candidate["key"] != expected_key:
            raise ValueError("candidate keys must match legal order")
        _decode_key(candidate["key"])
        features = candidate["features"]
        if not isinstance(features, dict) or list(features) != sorted(features):
            raise ValueError("features must be a sorted object")
        if not all(
            isinstance(name, str) and _finite_number(value)
            for name, value in features.items()
        ):
            raise ValueError("invalid policy feature")
        child_visits = _exact(candidate["visits"], int, "visits")
        total_value, q = candidate["total_value"], candidate["q"]
        if not _valid_child_stats(child_visits, total_value, q):
            raise ValueError("invalid child statistics")
        probability = candidate["target_probability"]
        if not _finite_number(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("invalid target probability")
        visits.append(child_visits)
        probabilities.append(float(probability))
    total_visits = sum(visits)
    if total_visits <= 0 or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError("target distribution is not normalized")
    expected = [value / total_visits for value in visits]
    if any(not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(probabilities, expected, strict=True)):
        raise ValueError("target distribution does not match visits")


def _encode_key(key: Key | None) -> dict[str, Any]:
    if key is None:
        return {"type": "none", "value": None}
    if type(key) is bool:
        return {"type": "bool", "value": key}
    if type(key) is int:
        return {"type": "integer", "value": key}
    if type(key) is float:
        if not math.isfinite(key):
            raise ValueError("non-finite float action key")
        return {"type": "float", "value": key}
    if type(key) is str:
        return {"type": "string", "value": key}
    if type(key) is tuple:
        return {"type": "tuple", "value": [_encode_tuple_item(item) for item in key]}
    raise ValueError(f"unsupported action key type: {type(key).__name__}")


def _encode_tuple_item(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        raise ValueError("non-finite float action key")
    if type(value) is tuple:
        return {"type": "tuple", "value": [_encode_tuple_item(v) for v in value]}
    raise ValueError(f"unsupported tuple action key item: {type(value).__name__}")


def _decode_key(encoded: Any) -> Key | None:
    if not isinstance(encoded, dict) or set(encoded) != {"type", "value"}:
        raise ValueError("typed action key must contain type and value")
    kind, value = encoded["type"], encoded["value"]
    expected: dict[str, type[Any]] = {
        "none": type(None), "bool": bool, "integer": int,
        "float": float, "string": str, "tuple": list,
    }
    if kind not in expected or type(value) is not expected[kind]:
        raise ValueError("typed action key has an invalid value")
    if kind == "float" and not math.isfinite(value):
        raise ValueError("non-finite float action key")
    if kind == "tuple":
        return tuple(_decode_tuple_item(item) for item in value)
    return value


def _decode_tuple_item(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, dict) and value.get("type") == "tuple":
        decoded = _decode_key(value)
        if isinstance(decoded, tuple):
            return decoded
    raise ValueError("invalid tuple action key item")


def _valid_child_stats(visits: Any, total_value: Any, q: Any) -> bool:
    if type(visits) is not int or visits < 0:
        return False
    if not _finite_number(total_value) or not _finite_number(q):
        return False
    expected_q = float(total_value) / visits if visits else 0.0
    return math.isclose(float(q), expected_q, abs_tol=1e-9)


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _exact(value: Any, expected: type[Any], name: str) -> Any:
    if type(value) is not expected:
        raise ValueError(f"{name} has the wrong type")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON data") from exc


__all__ = [
    "POLICY_FEATURE_SCHEMA_ID",
    "SCHEMA_VERSION",
    "PolicyDatasetRecorder",
    "PolicyDatasetStats",
    "load_policy_examples",
]
