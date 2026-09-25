"""Behavioral contract for information-safe hero observation adapters."""

from __future__ import annotations

import math
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from automata.models import LearnedObservation, PublicSnapshot, Viewer, canonical_json_bytes
from automata.observation import encode_snapshot, project_snapshot
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID
from goa2.engine.setup import GameSetup

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))


def _state() -> GameState:
    return GameSetup.create_game(
        MAP,
        ["Razzle", "Wasp"],
        ["Arien", "Brogan"],
        game_type="QUICK",
        seed=37,
    )


def _public_inputs(state: GameState | None = None) -> tuple[PublicSnapshot, LearnedObservation]:
    snapshot = project_snapshot(
        state or _state(),
        Viewer(schema_version=2, private_hero_id="hero_razzle", perspective_team="RED"),
    )
    return snapshot, encode_snapshot(snapshot)


def _registry():
    module = import_module("automata.observation")
    return module.HeroObservationAdapterRegistry()


def _dump(value: Any) -> dict[str, Any]:
    dumped = value.model_dump(mode="json")
    assert isinstance(dumped, dict)
    return dumped


def test_unknown_and_initial_benchmark_heroes_resolve_to_deterministic_generic_adapter() -> None:
    registry = _registry()

    resolved = [
        registry.resolve(hero) for hero in ("Unknown", "Wasp", "Xargatha", "Arien", "Brogan")
    ]

    assert {adapter.adapter_id for adapter in resolved} == {"generic"}
    assert {adapter.schema_version for adapter in resolved} == {1}
    assert [
        registry.resolve(hero).adapter_id
        for hero in ("Unknown", "Wasp", "Xargatha", "Arien", "Brogan")
    ] == [adapter.adapter_id for adapter in resolved]
    assert registry.generic_version == resolved[0].schema_version
    assert registry.registered_versions == {}


def test_generic_adapter_application_is_deterministic() -> None:
    registry = _registry()
    snapshot, observation = _public_inputs()

    first = registry.apply("Unknown", snapshot, observation)
    second = registry.apply("Unknown", snapshot, observation)

    assert isinstance(first, LearnedObservation)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_registered_adapter_receives_only_frozen_public_contracts() -> None:
    received: list[tuple[object, object]] = []

    class PublicOnlyAdapter:
        hero_definition_name = "Wasp"
        adapter_id = "wasp-public-contract"
        schema_version = 1

        def augment(
            self, snapshot: PublicSnapshot, observation: LearnedObservation
        ) -> dict[str, object]:
            received.append((snapshot, observation))
            with pytest.raises((AttributeError, TypeError, ValueError)):
                snapshot.map_id = "mutated"
            return {"public_capability": True}

    registry = _registry()
    registry.register(PublicOnlyAdapter())
    assert registry.registered_versions == {"Wasp": PublicOnlyAdapter.schema_version}
    snapshot, observation = _public_inputs()

    applied = registry.apply("Wasp", snapshot, observation)

    assert isinstance(applied, LearnedObservation)
    assert received == [(snapshot, observation)]
    assert all(not isinstance(value, GameState) for call in received for value in call)


class _Adapter:
    schema_version = 1

    def __init__(self, hero_definition_name: str, adapter_id: str, result: object = None) -> None:
        self.hero_definition_name = hero_definition_name
        self.adapter_id = adapter_id
        self.result = {} if result is None else result

    def augment(self, snapshot: PublicSnapshot, observation: LearnedObservation) -> object:
        return self.result


def test_duplicate_hero_registration_fails_clearly() -> None:
    registry = _registry()
    registry.register(_Adapter("Wasp", "wasp-one"))

    with pytest.raises(ValueError, match=r"duplicate|Wasp|hero"):
        registry.register(_Adapter("Wasp", "wasp-two"))


def test_duplicate_adapter_identity_fails_clearly() -> None:
    registry = _registry()
    registry.register(_Adapter("Wasp", "shared-adapter"))

    with pytest.raises(ValueError, match=r"duplicate|shared-adapter|identity"):
        registry.register(_Adapter("Arien", "shared-adapter"))


@pytest.mark.parametrize("bad", [math.nan, math.inf, {"not-json"}])
def test_non_finite_or_non_json_adapter_augmentation_is_rejected(bad: object) -> None:
    registry = _registry()
    registry.register(_Adapter("Wasp", "invalid-output", {"bad": bad}))
    snapshot, observation = _public_inputs()

    with pytest.raises(ValueError, match=r"finite|JSON|serializ|augmentation"):
        registry.apply("Wasp", snapshot, observation)


def test_razzle_uses_generic_adapter_without_collapsing_its_physical_pieces() -> None:
    state = _state()
    state.acting_piece_id = BoardEntityID("hero_razzle_piece_1")
    snapshot, observation = _public_inputs(state)
    registry = _registry()

    adapted = registry.apply("Razzle", snapshot, observation)
    tokens = _dump(adapted)["tokens"]
    razzle_units = [
        token
        for token in tokens
        if token["kind"] == "UNIT"
        and str(token["features"]["entity_id"]).startswith("hero_razzle_piece_")
    ]

    assert registry.resolve("Razzle").adapter_id == "generic"
    assert {token["features"]["entity_id"] for token in razzle_units} == {
        f"hero_razzle_piece_{index}" for index in range(1, 5)
    }
    assert [token["features"]["is_acting_piece"] for token in razzle_units].count(True) == 1
    assert all(token["features"]["team_id"] == TeamColor.RED.value for token in razzle_units)
