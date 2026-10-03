"""Immutable, seed-only split assignments for native training data.

The split recipe hashes canonical JSON with exactly ``recipe``, ``namespace``,
``salt``, and ``world_seed``. A bootstrap or training seed is validation when::

    int(SHA256(recipe_input), 16) * fraction_denominator
        < fraction_numerator * 2**256

where the numerator and denominator come from ``validation_fraction.as_integer_ratio()``.
Dedicated validation ranges always map to validation. Evaluation, arena, screen,
and promotion ranges are declarations for other workers and cannot be enrolled here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from automata.models.contracts import canonical_json_bytes

NativeSeedPurpose = Literal[
    "bootstrap",
    "training",
    "validation",
    "evaluation",
    "arena",
    "screen",
    "promotion",
]
NativeSplitName = Literal["train", "validation"]

_RECIPE: Literal["native-seed-split-v1"] = "native-seed-split-v1"
_SPLITTABLE_PURPOSES = frozenset({"bootstrap", "training"})
_EXCLUDED_PURPOSES = frozenset({"evaluation", "arena", "screen", "promotion"})
_HASH_SPACE_SIZE = 1 << 256


class _RecipeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recipe: Literal["native-seed-split-v1"]
    namespace: str
    salt: str
    world_seed: StrictInt


class NativeSeedRange(BaseModel):
    """One explicit half-open range reserved for a native seed purpose."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    purpose: NativeSeedPurpose
    start: StrictInt = Field(ge=0)
    stop: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _nonempty(self) -> NativeSeedRange:
        if self.stop <= self.start:
            raise ValueError("native seed range stop must be greater than start")
        return self


class NativeSplitConfig(BaseModel):
    """Complete immutable declaration of the native seed split recipe."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    recipe: Literal["native-seed-split-v1"] = _RECIPE
    namespace: str = Field(min_length=1)
    salt: str = Field(min_length=1)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    seed_ranges: tuple[NativeSeedRange, ...] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _valid_config(self) -> NativeSplitConfig:
        if not math.isfinite(self.validation_fraction):
            raise ValueError("validation fraction must be finite")
        ordered = sorted(
            self.seed_ranges, key=lambda seed_range: (seed_range.start, seed_range.stop)
        )
        for previous, current in pairwise(ordered):
            if current.start < previous.stop:
                raise ValueError("native seed ranges must not overlap")
        return self

    def _revalidated(self) -> NativeSplitConfig:
        return NativeSplitConfig.model_validate(self.model_dump(mode="python"), strict=True)

    def canonical_bytes(self) -> bytes:
        """Return validated canonical JSON including the recipe identity."""
        return canonical_json_bytes(self._revalidated())

    @property
    def digest(self) -> str:
        """Return the SHA256 digest of validated canonical config bytes."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class NativeSeedAssignment(BaseModel):
    """The sole split assignment for one enrolled world seed."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    world_seed: StrictInt = Field(ge=0)
    split: NativeSplitName


class NativeSeedSplitLedger(BaseModel):
    """Canonical append-only membership ledger, sorted by world seed."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    config: NativeSplitConfig
    assignments: tuple[NativeSeedAssignment, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _valid_assignments(self) -> NativeSeedSplitLedger:
        seeds = tuple(assignment.world_seed for assignment in self.assignments)
        if seeds != tuple(sorted(set(seeds))):
            raise ValueError("native seed assignments must have unique sorted world seeds")
        for assignment in self.assignments:
            expected = _assigned_split(self.config, assignment.world_seed)
            if assignment.split != expected:
                raise ValueError(
                    f"assignment for world seed {assignment.world_seed} does not match split recipe"
                )
        return self

    def _revalidated(self) -> NativeSeedSplitLedger:
        return NativeSeedSplitLedger.model_validate(self.model_dump(mode="python"), strict=True)

    def canonical_bytes(self) -> bytes:
        """Return validated canonical JSON for durable embedding in a manifest."""
        return canonical_json_bytes(self._revalidated())

    @property
    def digest(self) -> str:
        """Return the SHA256 digest of validated canonical ledger bytes."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def split_for_seed(self, world_seed: int) -> NativeSplitName:
        """Return an existing assignment; this method never enrolls an unknown seed."""
        seed = _strict_world_seed(world_seed)
        ledger = self._revalidated()
        for assignment in ledger.assignments:
            if assignment.world_seed == seed:
                return assignment.split
        raise ValueError(f"world seed {seed} is not enrolled in the native split ledger")


def _strict_world_seed(world_seed: int) -> int:
    if isinstance(world_seed, bool) or not isinstance(world_seed, int):
        raise TypeError("world_seed must be a non-negative integer")
    if world_seed < 0:
        raise ValueError("world_seed must be a non-negative integer")
    return world_seed


def _purpose_for_seed(config: NativeSplitConfig, world_seed: int) -> NativeSeedPurpose:
    for seed_range in config.seed_ranges:
        if seed_range.start <= world_seed < seed_range.stop:
            return seed_range.purpose
    raise ValueError(f"world seed {world_seed} is not declared by the native split config")


def native_seed_purpose(config: NativeSplitConfig, world_seed: int) -> NativeSeedPurpose:
    """Return the declared purpose after strictly revalidating both inputs."""
    if not isinstance(config, NativeSplitConfig):
        raise TypeError("config must be a NativeSplitConfig")
    validated = NativeSplitConfig.model_validate(config.model_dump(mode="python"), strict=True)
    return _purpose_for_seed(validated, _strict_world_seed(world_seed))


def _recipe_digest(config: NativeSplitConfig, world_seed: int) -> bytes:
    payload = _RecipeInput(
        recipe=_RECIPE,
        namespace=config.namespace,
        salt=config.salt,
        world_seed=world_seed,
    )
    return hashlib.sha256(canonical_json_bytes(payload)).digest()


def _assigned_split(config: NativeSplitConfig, world_seed: int) -> NativeSplitName:
    purpose = _purpose_for_seed(config, world_seed)
    if purpose == "validation":
        return "validation"
    if purpose in _EXCLUDED_PURPOSES:
        raise ValueError(
            f"world seed {world_seed} purpose {purpose!r} is not eligible for native replay"
        )
    if purpose not in _SPLITTABLE_PURPOSES:  # pragma: no cover - exhaustive type guard
        raise ValueError(f"unsupported native seed purpose {purpose!r}")

    numerator, denominator = config.validation_fraction.as_integer_ratio()
    score = int.from_bytes(_recipe_digest(config, world_seed), "big")
    if score * denominator < numerator * _HASH_SPACE_SIZE:
        return "validation"
    return "train"


def create_native_split_ledger(config: NativeSplitConfig) -> NativeSeedSplitLedger:
    """Create an empty ledger after strictly revalidating the supplied config."""
    validated = NativeSplitConfig.model_validate(config.model_dump(mode="python"), strict=True)
    return NativeSeedSplitLedger(config=validated, assignments=())


def extend_native_split_ledger(
    ledger: NativeSeedSplitLedger,
    world_seeds: Iterable[int],
) -> NativeSeedSplitLedger:
    """Purely extend a ledger, atomically rejecting any invalid or excluded seed."""
    validated = NativeSeedSplitLedger.model_validate(ledger.model_dump(mode="python"), strict=True)
    requested = tuple(_strict_world_seed(seed) for seed in world_seeds)

    assignments = {item.world_seed: item for item in validated.assignments}
    additions: dict[int, NativeSeedAssignment] = {}
    for seed in requested:
        if seed in assignments or seed in additions:
            continue
        additions[seed] = NativeSeedAssignment(
            world_seed=seed,
            split=_assigned_split(validated.config, seed),
        )

    combined = {**assignments, **additions}
    return NativeSeedSplitLedger(
        config=validated.config,
        assignments=tuple(combined[seed] for seed in sorted(combined)),
    )


__all__ = [
    "NativeSeedAssignment",
    "NativeSeedPurpose",
    "NativeSeedRange",
    "NativeSeedSplitLedger",
    "NativeSplitConfig",
    "NativeSplitName",
    "create_native_split_ledger",
    "extend_native_split_ledger",
    "native_seed_purpose",
]
