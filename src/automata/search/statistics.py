"""Generic online scalar statistics used by search implementations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunningStatistics:
    count: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def add(self, value: float) -> RunningStatistics:
        count = self.count + 1
        delta = value - self.mean
        mean = self.mean + delta / count
        return RunningStatistics(count, mean, self._m2 + delta * (value - mean))

    @property
    def variance(self) -> float:
        return self._m2 / self.count if self.count else 0.0


__all__ = ["RunningStatistics"]
