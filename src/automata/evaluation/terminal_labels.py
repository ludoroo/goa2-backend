"""Deterministic terminal labels for sampled search cutoffs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from automata.agents.base import Agent
from automata.agents.heuristic_agent import HeuristicAgent
from automata.runtime.clone import clone_state
from automata.runtime.harness import RunResult, continue_game
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

from .features import FEATURE_SCHEMAS, feature_vector
from .value_dataset import SCHEMA_VERSION


@dataclass
class TerminalLabelStats:
    """Counts eligible cutoffs, written labels, and skipped continuations."""

    sampled: int = 0
    recorded_samples: int = 0
    skipped_samples: int = 0


class TerminalLabelObserver:
    """Sample cutoffs at a fixed stride and append terminal value labels."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_game_id: str,
        world_seed: int,
        agent_label: str,
        red_heroes: list[str],
        blue_heroes: list[str],
        source_revision: str,
        dirty_tree_hash: str,
        sample_every: int,
        max_samples: int,
        continuation_max_steps: int,
        continuation_max_rounds: int | None,
        feature_schema: str = "base-v1",
        continuation_fn: Callable[..., RunResult] = continue_game,
    ) -> None:
        if sample_every <= 0:
            raise ValueError("sample_every must be positive")
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        if feature_schema not in FEATURE_SCHEMAS:
            raise ValueError(f"unknown feature schema: {feature_schema!r}")

        self._path = Path(path)
        self._source_game_id = source_game_id
        self._world_seed = int(world_seed)
        self._agent_label = agent_label
        self._red_heroes = list(red_heroes)
        self._blue_heroes = list(blue_heroes)
        self._source_revision = source_revision
        self._dirty_tree_hash = dirty_tree_hash
        self._sample_every = sample_every
        self._max_samples = max_samples
        self._continuation_max_steps = continuation_max_steps
        self._continuation_max_rounds = continuation_max_rounds
        self._feature_schema = feature_schema
        self._continuation_fn = continuation_fn
        self._ordinal = 0
        self._stats = TerminalLabelStats()

        config = {
            "source_game_id": source_game_id,
            "world_seed": self._world_seed,
            "agent_label": agent_label,
            "red_heroes": self._red_heroes,
            "blue_heroes": self._blue_heroes,
            "source_revision": source_revision,
            "dirty_tree_hash": dirty_tree_hash,
            "sample_every": sample_every,
            "max_samples": max_samples,
            "continuation_max_steps": continuation_max_steps,
            "continuation_max_rounds": continuation_max_rounds,
        }
        if feature_schema != "base-v1":
            config["feature_schema"] = feature_schema
        self._config_id = _identity(config)
        self._existing_ids = self._read_existing_ids()

    @property
    def stats(self) -> TerminalLabelStats:
        return self._stats

    def __call__(self, state: GameState, team: TeamColor, active_value: float) -> None:
        ordinal = self._ordinal
        self._ordinal += 1
        if ordinal % self._sample_every or self._stats.sampled >= self._max_samples:
            return

        self._stats.sampled += 1
        sample_id = _identity(
            {"config_id": self._config_id, "cutoff_ordinal": ordinal, "team": team.value}
        )
        if sample_id in self._existing_ids:
            self._stats.skipped_samples += 1
            return

        schema = FEATURE_SCHEMAS[self._feature_schema]
        features = [float(value) for value in feature_vector(state, team, self._feature_schema)]
        continuation_state = clone_state(state)
        result = self._continuation_fn(
            continuation_state,
            self._agents(state, ordinal),
            max_steps=self._continuation_max_steps,
            max_rounds=self._continuation_max_rounds,
        )
        if result.reason != "game_over" or result.winner not in {"RED", "BLUE"}:
            self._stats.skipped_samples += 1
            return

        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "game_id": self._source_game_id,
            "world_seed": self._world_seed,
            "team": team.value,
            "features": features,
            "feature_names": list(schema.feature_names),
            "winner": result.winner,
            "red_heroes": list(self._red_heroes),
            "blue_heroes": list(self._blue_heroes),
            "source_revision": self._source_revision,
            "dirty_tree_hash": self._dirty_tree_hash,
            "agent_label": self._agent_label,
            "sample_id": sample_id,
            "config_id": self._config_id,
            "cutoff_ordinal": ordinal,
            "cutoff_round": state.round,
            "active_value": float(active_value),
            "continuation": {
                "max_steps": self._continuation_max_steps,
                "max_rounds": self._continuation_max_rounds,
                "rounds": result.rounds,
                "steps": result.steps,
                "reason": result.reason,
            },
        }
        if self._feature_schema != "base-v1":
            row["feature_schema"] = self._feature_schema
        self._append(row)
        self._existing_ids.add(sample_id)
        self._stats.recorded_samples += 1

    def _agents(self, state: GameState, ordinal: int) -> Mapping[str, Agent]:
        agents: dict[str, Agent] = {}
        for color in (TeamColor.RED, TeamColor.BLUE):
            seed = int(
                _identity(
                    {"config_id": self._config_id, "cutoff_ordinal": ordinal, "team": color.value}
                )[:16],
                16,
            )
            shared = HeuristicAgent(seed=seed)
            for hero in state.teams[color].heroes:
                agents[hero.id] = shared
        return agents

    def _read_existing_ids(self) -> set[str]:
        if not self._path.exists():
            return set()
        ids: set[str] = set()
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sample_id = json.loads(line).get("sample_id")
                    if isinstance(sample_id, str):
                        ids.add(sample_id)
        return ids

    def _append(self, row: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _identity(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["TerminalLabelObserver", "TerminalLabelStats"]
