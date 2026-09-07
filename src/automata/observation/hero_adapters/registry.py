"""Information-safe, schema-stable hero observation adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping

from pydantic import JsonValue

from automata.models.contracts import LearnedObservation, ObservationToken, PublicSnapshot

from .protocol import HeroObservationAdapter


class _GenericHeroObservationAdapter:
    hero_definition_name = "*"
    adapter_id = "generic"
    schema_version = 1

    def augment(
        self, snapshot: PublicSnapshot, observation: LearnedObservation
    ) -> Mapping[str, JsonValue]:
        return {}


class HeroObservationAdapterRegistry:
    """Resolve and safely apply public hero graph augmentations."""

    def __init__(self) -> None:
        self._generic: HeroObservationAdapter = _GenericHeroObservationAdapter()
        self._by_hero: dict[str, HeroObservationAdapter] = {}
        self._adapter_ids = {self._generic.adapter_id}

    def register(self, adapter: HeroObservationAdapter) -> None:
        hero_name = getattr(adapter, "hero_definition_name", None)
        adapter_id = getattr(adapter, "adapter_id", None)
        version = getattr(adapter, "schema_version", None)
        if (
            not isinstance(hero_name, str)
            or not hero_name.strip()
            or hero_name != hero_name.strip()
            or hero_name == "*"
        ):
            raise ValueError("invalid hero adapter hero_definition_name metadata")
        if (
            not isinstance(adapter_id, str)
            or not adapter_id.strip()
            or adapter_id != adapter_id.strip()
        ):
            raise ValueError("invalid hero adapter identity metadata")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("invalid hero adapter schema_version metadata")
        if not callable(getattr(adapter, "augment", None)):
            raise ValueError("invalid hero adapter augment contract")
        if hero_name in self._by_hero:
            raise ValueError(f"duplicate hero adapter registration for {hero_name!r}")
        if adapter_id in self._adapter_ids:
            raise ValueError(f"duplicate adapter identity {adapter_id!r}")
        self._by_hero[hero_name] = adapter
        self._adapter_ids.add(adapter_id)

    @property
    def generic_version(self) -> int:
        """Schema version used when a hero has no registered adapter."""
        return self._generic.schema_version

    @property
    def registered_versions(self) -> dict[str, int]:
        """Current adapter schema versions keyed by hero definition name."""
        return {name: adapter.schema_version for name, adapter in self._by_hero.items()}

    def resolve(self, hero_definition_name: str) -> HeroObservationAdapter:
        return self._by_hero.get(hero_definition_name, self._generic)

    def apply(
        self,
        hero_definition_name: str,
        snapshot: PublicSnapshot,
        observation: LearnedObservation,
    ) -> LearnedObservation:
        adapter = self.resolve(hero_definition_name)
        raw = adapter.augment(snapshot, observation)
        if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
            raise ValueError("adapter augmentation must be a JSON object")
        augmentation = dict(raw)
        try:
            # Round-tripping also rejects Python-only containers and objects.
            encoded = json.dumps(augmentation, allow_nan=False, separators=(",", ":"))
            normalized = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError("adapter augmentation must contain finite JSON values") from exc
        if normalized != augmentation:
            raise ValueError("adapter augmentation must contain exact JSON values")

        tokens: list[ObservationToken] = []
        matched = False
        for token in observation.tokens:
            if token.kind == "HERO" and token.features.get("name") == hero_definition_name:
                features = dict(token.features)
                features["adapter_features"] = normalized
                token = token.model_copy(update={"features": features}, deep=True)
                matched = True
            else:
                token = token.model_copy(deep=True)
            tokens.append(token)
        if augmentation and not matched:
            raise ValueError(f"hero {hero_definition_name!r} has no matching HERO token")
        return observation.model_copy(update={"tokens": tuple(tokens)}, deep=True)


__all__ = ["HeroObservationAdapterRegistry"]
