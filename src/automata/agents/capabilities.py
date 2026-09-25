"""Structural runtime capabilities for expensive agents."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .contracts import Agent


@dataclass(frozen=True)
class BoundedComputeCapability:
    decision_timeout_seconds: float
    fallback_factory: Callable[[int], Agent]

    def __post_init__(self) -> None:
        if not math.isfinite(self.decision_timeout_seconds) or self.decision_timeout_seconds <= 0:
            raise ValueError("decision_timeout_seconds must be finite and positive")

    def create_fallback(self, *, seed: int) -> Agent:
        return self.fallback_factory(seed)


def bounded_compute_capability(agent: object) -> BoundedComputeCapability | None:
    capability = getattr(agent, "bounded_compute", None)
    return capability if isinstance(capability, BoundedComputeCapability) else None


def heuristic_fallback(seed: int) -> Agent:
    from .heuristic_agent import HeuristicAgent

    return HeuristicAgent(seed=seed)


__all__ = [
    "BoundedComputeCapability",
    "bounded_compute_capability",
    "heuristic_fallback",
]
