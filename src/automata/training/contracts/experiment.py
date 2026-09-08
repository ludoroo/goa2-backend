"""Generic experiment and seed-ownership contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class SeedRange:
    """An immutable half-open seed interval."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.start >= self.stop:
            raise ValueError("seed range must be non-negative and non-empty")

    def __contains__(self, seed: object) -> bool:
        return isinstance(seed, int) and self.start <= seed < self.stop


@dataclass(frozen=True, slots=True)
class SeedRegistry:
    _ranges: Mapping[str, SeedRange]

    def __post_init__(self) -> None:
        copied = dict(self._ranges)
        if not copied or any(not purpose for purpose in copied):
            raise ValueError("seed purposes must be non-empty")
        values = tuple(copied.values())
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                if left.start < right.stop and right.start < left.stop:
                    raise ValueError("seed ranges must be disjoint")
        object.__setattr__(self, "_ranges", MappingProxyType(copied))

    def purposes(self) -> tuple[str, ...]:
        return tuple(self._ranges)

    def range_for(self, purpose: str) -> SeedRange:
        try:
            return self._ranges[purpose]
        except KeyError as exc:
            raise ValueError(f"unknown seed purpose {purpose!r}") from exc

    def require_seed(self, seed: int, *, purpose: str) -> int:
        owned = self.range_for(purpose)
        if seed in owned:
            return seed
        actual = next((name for name, seeds in self._ranges.items() if seed in seeds), None)
        owner = f"; seed belongs to {actual!r}" if actual else "; seed is unregistered"
        raise ValueError(f"seed {seed} is not owned by purpose {purpose!r}{owner}")


@dataclass(frozen=True, slots=True)
class ExperimentScope:
    map_id: str
    game_type: str
    red_heroes: tuple[str, ...]
    blue_heroes: tuple[str, ...]
    seed_registry: SeedRegistry


__all__ = ["ExperimentScope", "SeedRange", "SeedRegistry"]
