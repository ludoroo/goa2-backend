"""Neutral descriptor for an engine decision consumed by search and encoders."""

from __future__ import annotations

from dataclasses import dataclass

from goa2.domain.input import InputRequest
from goa2.domain.models.unit import Hero


@dataclass
class DecisionDescriptor:
    kind: str
    hero: Hero | None = None
    request: InputRequest | None = None
    winner: str | None = None
    can_finish_planning: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.kind == "OVER"


__all__ = ["DecisionDescriptor"]
