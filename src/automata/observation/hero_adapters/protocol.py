"""Contract for information-safe hero observation extensions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from automata.models.contracts import LearnedObservation, PublicSnapshot


@runtime_checkable
class HeroObservationAdapter(Protocol):
    hero_definition_name: str
    adapter_id: str
    schema_version: int

    def augment(
        self, snapshot: PublicSnapshot, observation: LearnedObservation
    ) -> Mapping[str, JsonValue]: ...


__all__ = ["HeroObservationAdapter"]
