"""Compatible, complete-game replay storage and deterministic sampling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from automata.models.shared_encoder.artifacts import ModelArtifactManifest
from automata.training.contracts.experiment import SeedRegistry
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT

from .dataset import SCHEMA_VERSION, JointDataset, JointDatasetRow

ReplaySource = Literal["LATEST", "RECENT", "HISTORICAL_HARD"]
_SOURCES: tuple[ReplaySource, ...] = ("LATEST", "RECENT", "HISTORICAL_HARD")
_SHARES = (0.5, 0.3, 0.2)
_DEFAULT_REPLAY_SEED_PURPOSES = ("bootstrap", "training")


class ReplayCompatibilityError(ValueError):
    """A generation is incompatible with its parent or this replay buffer."""


@dataclass(frozen=True, slots=True)
class SampledReplayGame:
    """One complete sampled game and the replay stratum that supplied it."""

    game_id: str
    generation_id: str
    champion_parent_digest: str
    source: ReplaySource
    rows: tuple[JointDatasetRow, ...]


@dataclass(frozen=True, slots=True)
class ReplaySample:
    """Rows and aligned weights from a deterministic complete-game sample."""

    games: tuple[SampledReplayGame, ...]
    rows: tuple[JointDatasetRow, ...]
    decision_weights: tuple[float, ...]

    @property
    def game_ids(self) -> tuple[str, ...]:
        return tuple(game.game_id for game in self.games)


@dataclass(frozen=True, slots=True)
class _StoredGame:
    game_id: str
    generation_id: str
    champion_parent_digest: str
    rows: tuple[JointDatasetRow, ...]
    historical_champion: bool
    insertion_index: int


def _artifact_compatibility(manifest: ModelArtifactManifest) -> tuple[object, ...]:
    """Return the executable data contract, excluding learned weights."""
    return (
        manifest.schema_version,
        manifest.observation_schema_version,
        manifest.map_schema_version,
        manifest.runtime_compatibility_version,
        tuple(sorted(manifest.hero_adapter_versions.items())),
        manifest.tensor_schema_id,
        manifest.tensor_schema_version,
        manifest.tensor_schema_digest,
    )


def _quotas(game_count: int) -> tuple[int, int, int]:
    raw = tuple(game_count * share for share in _SHARES)
    quotas = [math.floor(value) for value in raw]
    missing = game_count - sum(quotas)
    remainders = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - quotas[index]), index),
    )
    for index in remainders[:missing]:
        quotas[index] += 1
    return quotas[0], quotas[1], quotas[2]


def _redistributed_quotas(
    game_count: int, capacities: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Approximate 50/30/20, deterministically filling unavailable strata."""
    if sum(capacities) < game_count:
        raise ValueError(f"replay buffer has {sum(capacities)} games, requires {game_count}")
    allocations = [
        min(quota, capacity)
        for quota, capacity in zip(_quotas(game_count), capacities, strict=True)
    ]
    remaining = game_count - sum(allocations)
    while remaining:
        available = [
            index for index, capacity in enumerate(capacities) if allocations[index] < capacity
        ]
        weight_total = sum(_SHARES[index] for index in available)
        raw = {index: remaining * _SHARES[index] / weight_total for index in available}
        additions = {
            index: min(math.floor(raw[index]), capacities[index] - allocations[index])
            for index in available
        }
        added = sum(additions.values())
        if added:
            for index, addition in additions.items():
                allocations[index] += addition
            remaining -= added
            continue
        index = min(available, key=lambda item: (-raw[item], item))
        allocations[index] += 1
        remaining -= 1
    return allocations[0], allocations[1], allocations[2]


class ReplayBuffer:
    """Store and sample games owned by explicit non-arena seed purposes."""

    def __init__(
        self,
        *,
        max_games: int | None = None,
        seed_registry: SeedRegistry | None = None,
        replay_seed_purposes: tuple[str, ...] = _DEFAULT_REPLAY_SEED_PURPOSES,
    ) -> None:
        if max_games is not None and max_games <= 0:
            raise ValueError("max_games must be positive")
        self._max_games = max_games
        self._seed_registry = seed_registry or PHASE0_EXPERIMENT.seed_registry
        if not replay_seed_purposes:
            raise ValueError("replay_seed_purposes must not be empty")
        for purpose in replay_seed_purposes:
            self._seed_registry.range_for(purpose)
        self._replay_seed_purposes = replay_seed_purposes
        self._games: dict[str, _StoredGame] = {}
        self._generation_order: list[str] = []
        self._hard_game_ids: set[str] = set()
        self._compatibility: tuple[object, ...] | None = None
        self._next_insertion_index = 0

    @property
    def game_ids(self) -> tuple[str, ...]:
        """Return retained game IDs in insertion order."""
        return tuple(self._games)

    @property
    def hard_game_ids(self) -> tuple[str, ...]:
        """Return retained permanent hard cases in insertion order."""
        return tuple(game_id for game_id in self._games if game_id in self._hard_game_ids)

    def add_generation(
        self,
        dataset: JointDataset,
        *,
        champion_parent: ModelArtifactManifest,
        historical_champion: bool = False,
        hard_game_ids: tuple[str, ...] = (),
    ) -> None:
        """Add one generation after validating lineage and data compatibility."""
        if not dataset.game_ids:
            raise ValueError("replay generation must contain at least one game")
        generation_ids = {row.generation_id for row in dataset.rows}
        if len(generation_ids) != 1:
            raise ReplayCompatibilityError("dataset must contain exactly one generation")
        generation_id = next(iter(generation_ids))
        if generation_id in self._generation_order:
            raise ValueError(f"replay generation already exists: {generation_id!r}")
        duplicates = set(dataset.game_ids) & set(self._games)
        if duplicates:
            raise ValueError(f"replay game IDs already exist: {sorted(duplicates)!r}")
        unknown_hard = set(hard_game_ids) - set(dataset.game_ids)
        if unknown_hard:
            raise ValueError(
                f"hard-case game IDs are absent from generation: {sorted(unknown_hard)!r}"
            )

        self._validate_dataset_seeds(dataset.rows)
        self._validate_compatibility(dataset, champion_parent)
        for game_id in dataset.game_ids:
            rows = dataset.rows_by_game[game_id]
            self._games[game_id] = _StoredGame(
                game_id=game_id,
                generation_id=generation_id,
                champion_parent_digest=champion_parent.model_digest,
                rows=rows,
                historical_champion=historical_champion,
                insertion_index=self._next_insertion_index,
            )
            self._next_insertion_index += 1
        self._generation_order.append(generation_id)
        self._hard_game_ids.update(hard_game_ids)
        self._evict_if_needed()

    def sample(self, *, game_count: int, seed: int) -> ReplaySample:
        """Sample the declared 50/30/20 strata using a local seeded RNG."""
        if game_count <= 0:
            raise ValueError("game_count must be positive")
        self._validate_replay_seed(seed)
        self._validate_dataset_seeds(
            tuple(row for game in self._games.values() for row in game.rows)
        )
        buckets = self._buckets()
        quotas = _redistributed_quotas(
            game_count,
            (
                len(buckets["LATEST"]),
                len(buckets["RECENT"]),
                len(buckets["HISTORICAL_HARD"]),
            ),
        )

        rng = random.Random(seed)
        sampled: list[SampledReplayGame] = []
        for source, quota in zip(_SOURCES, quotas, strict=True):
            population = sorted(buckets[source], key=lambda game: game.game_id)
            for game in rng.sample(population, quota):
                sampled.append(
                    SampledReplayGame(
                        game_id=game.game_id,
                        generation_id=game.generation_id,
                        champion_parent_digest=game.champion_parent_digest,
                        source=source,
                        rows=game.rows,
                    )
                )

        rows: list[JointDatasetRow] = []
        weights: list[float] = []
        game_weight = 1.0 / game_count
        for sampled_game in sampled:
            rows.extend(sampled_game.rows)
            weights.extend([game_weight / len(sampled_game.rows)] * len(sampled_game.rows))
        return ReplaySample(tuple(sampled), tuple(rows), tuple(weights))

    def _validate_dataset_seeds(self, rows: tuple[JointDatasetRow, ...]) -> None:
        for row in rows:
            self._validate_replay_seed(row.world_seed, label="world seed")

    def _validate_replay_seed(self, seed: int, *, label: str = "sampling seed") -> None:
        if any(
            seed in self._seed_registry.range_for(purpose) for purpose in self._replay_seed_purposes
        ):
            return
        raise ValueError(
            f"{label} {seed} is not eligible for replay; "
            f"allowed purposes are {self._replay_seed_purposes!r}"
        )

    def _validate_compatibility(
        self, dataset: JointDataset, champion_parent: ModelArtifactManifest
    ) -> None:
        if dataset.metadata.schema_version != SCHEMA_VERSION or any(
            row.schema_version != SCHEMA_VERSION for row in dataset.rows
        ):
            raise ReplayCompatibilityError("incompatible joint dataset schema version")
        if any(row.source_model_digest != champion_parent.model_digest for row in dataset.rows):
            raise ReplayCompatibilityError("dataset champion parent digest does not match artifact")
        if any(
            row.observation.state.schema_version != champion_parent.observation_schema_version
            for row in dataset.rows
        ):
            raise ReplayCompatibilityError("dataset observation schema does not match artifact")
        if any(
            row.map_id not in champion_parent.supported_maps
            or row.game_type not in champion_parent.supported_game_types
            or not set((*row.red_composition, *row.blue_composition)).issubset(
                champion_parent.supported_heroes
            )
            for row in dataset.rows
        ):
            raise ReplayCompatibilityError("dataset scope is unsupported by champion artifact")
        compatibility = _artifact_compatibility(champion_parent)
        if self._compatibility is not None and compatibility != self._compatibility:
            raise ReplayCompatibilityError("champion artifact is incompatible with replay buffer")
        self._compatibility = compatibility

    def _buckets(self) -> dict[ReplaySource, list[_StoredGame]]:
        buckets: dict[ReplaySource, list[_StoredGame]] = {source: [] for source in _SOURCES}
        latest = self._generation_order[-1] if self._generation_order else None
        for game in self._games.values():
            if game.game_id in self._hard_game_ids or game.historical_champion:
                source: ReplaySource = "HISTORICAL_HARD"
            elif game.generation_id == latest:
                source = "LATEST"
            else:
                source = "RECENT"
            buckets[source].append(game)
        return buckets

    def _evict_if_needed(self) -> None:
        if self._max_games is None:
            return
        excess = len(self._games) - self._max_games
        evictable = sorted(
            (game for game in self._games.values() if game.game_id not in self._hard_game_ids),
            key=lambda game: game.insertion_index,
        )
        for game in evictable[: max(0, excess)]:
            del self._games[game.game_id]


__all__ = [
    "ReplayBuffer",
    "ReplayCompatibilityError",
    "ReplaySample",
    "ReplaySource",
    "SampledReplayGame",
]
