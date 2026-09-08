"""Behavioral contract for Phase 1's variable relationship graph."""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any, cast

from automata.models import INITIAL_RELATIONSHIP_NAMES, PublicSnapshot, Viewer
from automata.observation import project_snapshot
from goa2.domain.board import Board, Zone
from goa2.domain.hex import Hex
from goa2.domain.models import (
    ActiveEffect,
    DurationType,
    EffectScope,
    EffectType,
    Hero,
    HeroPiece,
    Shape,
    Team,
    TeamColor,
)
from goa2.domain.state import GameState
from goa2.domain.tile import Tile
from goa2.domain.types import BoardEntityID, HeroID


def _encode(snapshot: PublicSnapshot):
    return import_module("automata.observation").encode_snapshot(snapshot)


def _dump(value: Any) -> dict[str, Any]:
    dumped = value.model_dump(mode="json")
    assert isinstance(dumped, dict)
    return dumped


def _state() -> GameState:
    left = Hex(q=-1, r=0, s=1)
    center = Hex(q=0, r=0, s=0)
    right = Hex(q=1, r=0, s=-1)
    bent = Hex(q=0, r=1, s=-1)
    board = Board(
        map_id="relationship_contract",
        zones={
            "red": Zone(id="red", hexes={cast(Any, left)}, neighbors=["mid"]),
            "mid": Zone(
                id="mid",
                hexes={cast(Any, center), cast(Any, bent)},
                neighbors=["red", "blue"],
            ),
            "blue": Zone(id="blue", hexes={cast(Any, right)}, neighbors=["mid"]),
        },
        lanes={"lane_1": ["red", "mid", "blue"]},
    )
    for hex_, zone in ((left, "red"), (center, "mid"), (right, "blue"), (bent, "mid")):
        board.tiles[hex_] = Tile(hex=hex_, zone_id=zone)

    self_hero = Hero(id=BoardEntityID("hero_self"), name="Self", team=TeamColor.RED, deck=[])
    ally = Hero(id=BoardEntityID("hero_ally"), name="Ally", team=TeamColor.RED, deck=[])
    enemy = Hero(id=BoardEntityID("hero_enemy"), name="Enemy", team=TeamColor.BLUE, deck=[])
    offboard = Hero(
        id=BoardEntityID("hero_offboard"), name="Offboard", team=TeamColor.BLUE, deck=[]
    )
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[self_hero, ally], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[enemy, offboard], minions=[]),
        },
        battle_zones={"lane_1": "mid"},
        wave_counters={"lane_1": 3},
        current_actor_id=HeroID("hero_self"),
    )
    state.place_entity(BoardEntityID("hero_self"), left)
    state.place_entity(BoardEntityID("hero_ally"), center)
    state.place_entity(BoardEntityID("hero_enemy"), right)
    unplaced_piece = HeroPiece(
        id=BoardEntityID("hero_self_piece_supply"),
        name="Supply piece",
        team=TeamColor.RED,
        owner_hero_id="hero_self",
    )
    state.misc_entities[unplaced_piece.id] = unplaced_piece
    return state


def _observation(state: GameState | None = None):
    state = state or _state()
    snapshot = project_snapshot(
        state,
        Viewer(schema_version=2, private_hero_id="hero_self", perspective_team="RED"),
    )
    return _encode(snapshot)


def _tokens(observation: Any, kind: str | None = None) -> list[dict[str, Any]]:
    tokens = _dump(observation)["tokens"]
    if kind is None:
        return tokens
    return [token for token in tokens if token["kind"] == kind]


def _edges(observation: Any, kind: str | None = None) -> list[dict[str, Any]]:
    edges = _dump(observation)["relationships"]
    if kind is None:
        return edges
    return [edge for edge in edges if edge["kind"] == kind]


def _unit_refs(observation: Any) -> dict[str, str]:
    return {
        token["features"]["entity_id"]: token["local_ref"] for token in _tokens(observation, "UNIT")
    }


def _edge(observation: Any, source_id: str, target_id: str) -> dict[str, Any]:
    refs = _unit_refs(observation)
    return next(
        edge
        for edge in _edges(observation, "UNIT_TO_UNIT")
        if edge["source_ref"] == refs[source_id] and edge["target_ref"] == refs[target_id]
    )


def test_positioned_units_have_a_complete_directed_fixed_field_graph() -> None:
    observation = _observation()
    positioned = {"hero_self", "hero_ally", "hero_enemy"}
    unit_edges = _edges(observation, "UNIT_TO_UNIT")

    assert len(unit_edges) == len(positioned) ** 2
    assert len({frozenset(edge["features"]) for edge in unit_edges}) == 1
    assert all(set(edge["features"]) == set(INITIAL_RELATIONSHIP_NAMES) for edge in unit_edges)
    assert all(
        set(edge) == {"schema_version", "source_ref", "target_ref", "kind", "features"}
        for edge in unit_edges
    )


def test_directed_unit_relationships_encode_exact_geometry_topology_and_lane_context() -> None:
    observation = _observation()
    ally = _edge(observation, "hero_self", "hero_ally")["features"]
    enemy = _edge(observation, "hero_self", "hero_enemy")["features"]
    itself = _edge(observation, "hero_self", "hero_self")["features"]

    assert (ally["delta_q"], ally["delta_r"], ally["delta_s"]) == (1, 0, -1)
    assert ally["hex_distance"] == 1
    assert ally["is_adjacent"] is True
    assert ally["topology_is_adjacent"] is True
    assert ally["path_exists"] is False
    assert ally["path_distance_valid"] is False
    assert math.isfinite(ally["path_distance"])
    assert ally["is_straight_line"] is True
    assert ally["same_zone"] is False
    assert ally["same_lane"] is True
    assert ally["lane_progress_delta"] == 0.5
    assert ally["relation"] == "ALLY"

    assert (enemy["delta_q"], enemy["delta_r"], enemy["delta_s"]) == (2, 0, -2)
    assert enemy["hex_distance"] == 2
    assert enemy["is_adjacent"] is False
    assert enemy["relation"] == "ENEMY"
    assert itself["relation"] == "SELF"
    assert itself["same_zone"] is True
    assert itself["lane_progress_delta"] == 0.0


def test_structural_edges_use_observation_local_references() -> None:
    observation = _observation()
    refs = {token["local_ref"] for token in _tokens(observation)}
    structural_kinds = {
        "OWNS",
        "POSITIONED_AT",
        "TILE_IN_ZONE",
        "ZONE_IN_LANE",
        "HEX_ADJACENT",
        "ZONE_ADJACENT",
    }
    structural = [edge for edge in _edges(observation) if edge["kind"] in structural_kinds]

    assert {edge["kind"] for edge in structural} == structural_kinds
    assert all(edge["source_ref"] in refs and edge["target_ref"] in refs for edge in structural)
    assert all("source_id" not in edge and "target_id" not in edge for edge in structural)
    positioned_refs = {edge["source_ref"] for edge in structural if edge["kind"] == "POSITIONED_AT"}
    assert positioned_refs == {
        _unit_refs(observation)[entity] for entity in ("hero_self", "hero_ally", "hero_enemy")
    }


def test_topology_split_has_finite_explicitly_invalid_distance() -> None:
    state = _state()
    state.active_effects.append(
        ActiveEffect(
            id="split",
            source_id="hero_self",
            effect_type=EffectType.TOPOLOGY_SPLIT,
            split_axis="q",
            split_value=0,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_ROUND,
            created_at_turn=1,
            created_at_round=1,
            is_active=True,
        )
    )

    relationship = _edge(_observation(state), "hero_self", "hero_enemy")["features"]

    assert relationship["hex_distance"] == 2
    assert relationship["path_exists"] is False
    assert relationship["path_distance_valid"] is False
    assert math.isfinite(relationship["path_distance"])
    assert relationship["topology_is_adjacent"] is False
    assert relationship["is_straight_line"] is False


def test_topology_isolation_preserves_directional_engine_behavior() -> None:
    state = _state()
    state.active_effects.append(
        ActiveEffect(
            id="isolation",
            source_id="hero_ally",
            effect_type=EffectType.TOPOLOGY_ISOLATION,
            split_axis="q",
            split_value=0,
            isolated_hex=Hex(q=0, r=0, s=0),
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_ROUND,
            created_at_turn=1,
            created_at_round=1,
            is_active=True,
        )
    )
    observation = _observation(state)

    toward_caster = _edge(observation, "hero_self", "hero_ally")["features"]
    from_caster = _edge(observation, "hero_ally", "hero_self")["features"]

    assert toward_caster["is_adjacent"] is True
    assert toward_caster["topology_is_adjacent"] is False
    assert toward_caster["path_exists"] is False
    assert toward_caster["path_distance_valid"] is False
    assert from_caster["path_exists"] is False
    assert from_caster["path_distance_valid"] is False


def test_unplaced_entities_have_no_fabricated_spatial_relationships() -> None:
    observation = _observation()
    units = {token["features"]["entity_id"]: token for token in _tokens(observation, "UNIT")}
    offboard_refs = {
        token["local_ref"]
        for entity_id, token in units.items()
        if entity_id in {"hero_offboard", "hero_self_piece_supply"}
    }
    spatial_kinds = {"UNIT_TO_UNIT", "POSITIONED_AT", "IN_ZONE", "IN_LANE"}

    assert "hero_self_piece_supply" in units
    assert all(
        not (
            edge["kind"] in spatial_kinds
            and (edge["source_ref"] in offboard_refs or edge["target_ref"] in offboard_refs)
        )
        for edge in _edges(observation)
    )


def test_request_specific_relationships_are_unavailable_not_false() -> None:
    features = _edge(_observation(), "hero_self", "hero_enemy")["features"]

    for name in ("path_distance", "reachable", "has_line_of_sight", "threatens", "supports"):
        assert features[f"{name}_valid"] is False
    assert math.isfinite(features["path_distance"])


def test_unit_permutation_only_permutes_equivalent_tokens_and_edges() -> None:
    state = _state()
    original = _dump(_observation(state))
    permuted_state = state.model_copy(deep=True)
    for team in permuted_state.teams.values():
        team.heroes.reverse()
        team.minions.reverse()
    permuted = _dump(_observation(permuted_state))

    def normalized(records: list[dict[str, Any]]) -> set[str]:
        import json

        return {json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records}

    assert normalized(original["tokens"]) == normalized(permuted["tokens"])
    assert normalized(original["relationships"]) == normalized(permuted["relationships"])


def test_removing_a_unit_changes_graph_size_not_record_feature_width() -> None:
    state = _state()
    complete = _observation(state)
    removed = state.model_copy(deep=True)
    removed.teams[TeamColor.BLUE].heroes = [
        hero for hero in removed.teams[TeamColor.BLUE].heroes if str(hero.id) != "hero_enemy"
    ]
    removed.remove_entity(BoardEntityID("hero_enemy"))
    reduced = _observation(removed)

    assert len(_tokens(complete, "UNIT")) == len(_tokens(reduced, "UNIT")) + 1
    assert len(_edges(complete, "UNIT_TO_UNIT")) == 9
    assert len(_edges(reduced, "UNIT_TO_UNIT")) == 4
    assert {frozenset(edge["features"]) for edge in _edges(complete, "UNIT_TO_UNIT")} == {
        frozenset(edge["features"]) for edge in _edges(reduced, "UNIT_TO_UNIT")
    }


def test_all_relationship_numbers_are_finite() -> None:
    state = _state()
    state.active_effects.append(
        ActiveEffect(
            id="split",
            source_id="hero_self",
            effect_type=EffectType.TOPOLOGY_SPLIT,
            split_axis="q",
            split_value=0,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_ROUND,
            created_at_turn=1,
            created_at_round=1,
            is_active=True,
        )
    )

    for edge in _edges(_observation(state)):
        for value in edge["features"].values():
            if isinstance(value, float):
                assert math.isfinite(value)
