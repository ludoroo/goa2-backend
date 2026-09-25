"""Behavioral contract for Phase 1's complete token observation encoder."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest

from automata.models import PublicSnapshot, Viewer, canonical_json_bytes
from automata.observation import project_snapshot
from automata.runtime.clone import clone_state
from goa2.domain.models import (
    ActiveEffect,
    DurationType,
    EffectScope,
    EffectType,
    MarkerType,
    Shape,
    TeamColor,
    Turret,
)
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID, HeroID
from goa2.engine.map_logic import endgame_totals
from goa2.engine.setup import GameSetup

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))
TWO_LANE_MAP = str(Path("src/goa2/data/maps/across_the_river.json"))


def _viewer(hero_id: str = "hero_razzle", team: str = "RED") -> Viewer:
    return Viewer(schema_version=2, private_hero_id=hero_id, perspective_team=team)


def _state() -> GameState:
    return GameSetup.create_game(
        MAP,
        ["Razzle", "Wasp"],
        ["Arien", "Brogan"],
        game_type="QUICK",
        seed=19,
    )


def _snapshot(state: GameState | None = None, viewer: Viewer | None = None) -> PublicSnapshot:
    return project_snapshot(state or _state(), viewer or _viewer())


def _encode(snapshot: PublicSnapshot):
    # Import at the public package boundary so the rest of this specification
    # still collects while the Phase 1 API is being introduced.
    return import_module("automata.observation").encode_snapshot(snapshot)


def _dump(value: Any) -> dict[str, Any]:
    dumped = value.model_dump(mode="json")
    assert isinstance(dumped, dict)
    return dumped


def _tokens(encoded: Any, kind: str | None = None) -> list[dict[str, Any]]:
    records = _dump(encoded)["tokens"]
    assert isinstance(records, list)
    if kind is None:
        return records
    return [record for record in records if record["kind"] == kind]


def _feature(token: dict[str, Any], name: str) -> Any:
    features = token["features"]
    assert isinstance(features, dict)
    return features[name]


def _observable_fixture() -> GameState:
    state = _state()
    state.current_actor_id = HeroID("hero_razzle")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.acting_piece_id = BoardEntityID("hero_razzle_piece_1")

    state.active_effects.append(
        ActiveEffect(
            id="public_contract_effect",
            source_id="hero_wasp",
            effect_type=EffectType.AREA_STAT_MODIFIER,
            scope=EffectScope(shape=Shape.RADIUS, range=1, origin_id="hero_wasp"),
            duration=DurationType.THIS_ROUND,
            created_at_turn=1,
            created_at_round=1,
            is_active=True,
        )
    )
    state.place_marker(MarkerType.VENOM, "hero_arien", -1, "hero_razzle")

    empty = next(
        hex_
        for hex_, tile in state.board.tiles.items()
        if not tile.is_terrain and tile.occupant_id is None
    )
    turret = Turret(
        id=BoardEntityID("contract_turret"), name="Contract turret", owner_id="hero_wasp"
    )
    state.misc_entities[BoardEntityID(str(turret.id))] = turret
    state.place_entity(BoardEntityID(str(turret.id)), empty)
    token = next(token for supply in state.token_pool.values() for token in supply)
    token.owner_id = HeroID("hero_wasp")
    token_hex = next(
        hex_
        for hex_, tile in state.board.tiles.items()
        if not tile.is_terrain and tile.occupant_id is None
    )
    state.place_entity(BoardEntityID(str(token.id)), token_hex)
    return state


def _reorder_observable_collections(snapshot: PublicSnapshot) -> PublicSnapshot:
    public = cast(dict[str, Any], deepcopy(snapshot.public_state))
    teams = cast(dict[str, Any], public["teams"])
    public["teams"] = dict(reversed(list(teams.items())))
    for team in public["teams"].values():
        team["heroes"] = list(reversed(team["heroes"]))
        team["minions"] = list(reversed(team["minions"]))
    board = cast(dict[str, Any], public["board"])
    for name in ("tiles", "zones", "entity_locations"):
        board[name] = dict(reversed(list(board[name].items())))
    for name in ("effects", "tokens", "board_entities", "unresolved_cards"):
        public[name] = list(reversed(public[name]))
    for name in ("markers", "hero_pieces", "battle_zones", "wave_counters"):
        public[name] = dict(reversed(list(public[name].items())))
    return snapshot.model_copy(update={"public_state": public})


def test_quick_setup_mode_survives_serialization_clone_and_projection() -> None:
    state = _state()
    serialized = GameState.model_validate_json(state.model_dump_json())

    snapshots = (_snapshot(state), _snapshot(serialized), _snapshot(clone_state(state)))

    assert {snapshot.game_type for snapshot in snapshots} == {"QUICK"}
    assert len({canonical_json_bytes(_encode(snapshot)) for snapshot in snapshots}) == 1


def test_encoding_is_canonical_and_invariant_to_semantically_irrelevant_order() -> None:
    snapshot = _snapshot(_observable_fixture())

    first = _encode(snapshot)
    second = _encode(snapshot)
    reordered = _encode(_reorder_observable_collections(snapshot))

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first) == canonical_json_bytes(reordered)


def test_complete_observation_has_immutable_fixed_field_token_records() -> None:
    encoded = _encode(_snapshot(_observable_fixture()))
    records = _tokens(encoded)

    assert Counter(record["kind"] for record in records).keys() >= {
        "GLOBAL",
        "TEAM",
        "HERO",
        "UNIT",
        "CARD",
        "TILE",
        "ZONE",
        "LANE",
        "EFFECT",
        "MARKER",
        "TOKEN",
        "ENTITY",
    }
    assert all(
        set(record) == {"schema_version", "local_ref", "kind", "features"} for record in records
    )
    for kind in {record["kind"] for record in records}:
        assert (
            len({frozenset(record["features"]) for record in records if record["kind"] == kind})
            == 1
        )

    first_token = cast(Any, encoded.tokens[0])
    with pytest.raises((AttributeError, TypeError, ValueError)):
        encoded.tokens = ()
    with pytest.raises((AttributeError, TypeError, ValueError)):
        first_token.local_ref = "changed"


def test_global_and_team_tokens_expose_observable_state_without_lossy_progress() -> None:
    encoded = _encode(_snapshot(_observable_fixture()))
    global_features = _tokens(encoded, "GLOBAL")[0]["features"]
    teams = {_feature(token, "team_id"): token for token in _tokens(encoded, "TEAM")}

    assert (
        global_features.items()
        >= {
            "game_type": "QUICK",
            "phase": "PLANNING",
            "round": 1,
            "turn": 1,
            "team_count": 2,
            "hero_count": 4,
            "lane_count": 1,
        }.items()
    )
    assert global_features["playable_tile_count"] > 0
    assert _feature(teams["RED"], "relation") == "OWN"
    assert _feature(teams["BLUE"], "relation") == "ENEMY"
    for token in teams.values():
        assert set(token["features"]) >= {
            "life_counters",
            "hero_count",
            "alive_hero_count",
            "physical_piece_count",
            "minion_count",
            "total_level",
            "mean_level",
            "total_gold",
            "mean_gold",
        }


def test_hero_and_unit_perspective_is_separate_from_action_roles() -> None:
    state = _observable_fixture()
    encoded = _encode(_snapshot(state))
    heroes = {_feature(token, "hero_id"): token for token in _tokens(encoded, "HERO")}
    units = {_feature(token, "entity_id"): token for token in _tokens(encoded, "UNIT")}

    assert _feature(heroes["hero_razzle"], "relation") == "SELF"
    assert _feature(heroes["hero_wasp"], "relation") == "ALLY"
    assert _feature(heroes["hero_arien"], "relation") == "ENEMY"
    assert _feature(heroes["hero_razzle"], "is_current_actor") is True
    assert _feature(heroes["hero_wasp"], "is_decision_owner") is True
    assert _feature(units["hero_razzle_piece_1"], "is_acting_piece") is True

    no_context = state.model_copy(deep=True)
    no_context.current_actor_id = None
    no_context.resolution_owner_id = None
    no_context.acting_piece_id = None
    context_free = _encode(_snapshot(no_context))
    for token in [*_tokens(context_free, "HERO"), *_tokens(context_free, "UNIT")]:
        assert _feature(token, "is_current_actor") is False
        assert _feature(token, "is_decision_owner") is False
        assert _feature(token, "is_acting_piece") is False


def test_team_perspective_orients_values_without_granting_a_private_identity() -> None:
    viewer = Viewer(schema_version=2, private_hero_id=None, perspective_team="RED")
    encoded = _encode(_snapshot(_observable_fixture(), viewer))
    heroes = {_feature(token, "hero_id"): token for token in _tokens(encoded, "HERO")}

    assert _feature(heroes["hero_razzle"], "relation") == "ALLY"
    assert _feature(heroes["hero_arien"], "relation") == "ENEMY"


def test_razzle_player_and_each_piece_are_distinct_without_fabricated_hero_unit() -> None:
    encoded = _encode(_snapshot())
    heroes = {_feature(token, "hero_id"): token for token in _tokens(encoded, "HERO")}
    units = {_feature(token, "entity_id"): token for token in _tokens(encoded, "UNIT")}
    pieces = {entity_id for entity_id in units if entity_id.startswith("hero_razzle_piece_")}

    assert "hero_razzle" in heroes
    assert "hero_razzle" not in units
    assert pieces == {f"hero_razzle_piece_{number}" for number in range(1, 5)}
    assert all(
        _feature(units[piece], "owner_ref") == heroes["hero_razzle"]["local_ref"]
        for piece in pieces
    )


def test_each_lane_preserves_ordered_position_and_viewer_relative_advantage() -> None:
    distributed_state = GameSetup.create_game(
        TWO_LANE_MAP,
        ["Razzle", "Wasp", "Xargatha"],
        ["Arien", "Brogan", "Tali"],
        game_type="QUICK",
        seed=23,
    )
    even_state = distributed_state.model_copy(deep=True)
    lane_ids = sorted(distributed_state.board.lanes)
    assert len(lane_ids) == 2

    first_lane, second_lane = lane_ids
    distributed_state.battle_zones[first_lane] = distributed_state.board.lanes[first_lane][1]
    distributed_state.battle_zones[second_lane] = distributed_state.board.lanes[second_lane][3]
    even_state.battle_zones[first_lane] = even_state.board.lanes[first_lane][2]
    even_state.battle_zones[second_lane] = even_state.board.lanes[second_lane][2]

    assert (
        endgame_totals(distributed_state)
        == endgame_totals(even_state)
        == {
            TeamColor.RED: 2,
            TeamColor.BLUE: 4,
        }
    )

    red = _encode(_snapshot(distributed_state))
    even = _encode(_snapshot(even_state))
    blue = _encode(_snapshot(distributed_state, _viewer("hero_arien", "BLUE")))
    red_lanes = {_feature(token, "lane_id"): token for token in _tokens(red, "LANE")}
    blue_lanes = {_feature(token, "lane_id"): token for token in _tokens(blue, "LANE")}

    assert set(red_lanes) == set(distributed_state.board.lanes)
    assert len(red_lanes) == len(distributed_state.board.lanes)
    for lane_id, zones in distributed_state.board.lanes.items():
        assert len(_feature(red_lanes[lane_id], "ordered_zone_refs")) == len(zones)
        assert _feature(red_lanes[lane_id], "battle_zone_index") == zones.index(
            distributed_state.battle_zones[lane_id]
        )
        assert _feature(red_lanes[lane_id], "signed_battle_zone_advantage") == -_feature(
            blue_lanes[lane_id], "signed_battle_zone_advantage"
        )
    assert len({_feature(token, "battle_zone_index") for token in red_lanes.values()}) == 2
    distributed_advantage = sorted(
        _feature(token, "signed_battle_zone_advantage") for token in red_lanes.values()
    )
    even_advantage = sorted(
        _feature(token, "signed_battle_zone_advantage") for token in _tokens(even, "LANE")
    )
    assert sum(distributed_advantage) == sum(even_advantage)
    assert distributed_advantage != even_advantage


def test_hidden_substitution_is_equal_but_identified_cards_remain_distinguishable() -> None:
    state = _state()
    enemy = state.get_hero(HeroID("hero_arien"))
    owner = state.get_hero(HeroID("hero_razzle"))
    assert enemy is not None and owner is not None
    hidden_card = enemy.hand.pop()
    hidden_card.is_facedown = True
    enemy.current_turn_card = hidden_card
    public_card = hidden_card.model_copy(deep=True)
    public_card.id = "public_card_alpha"
    public_card.is_facedown = False
    enemy.played_cards.append(public_card)
    baseline = _snapshot(state)

    hidden_change = state.model_copy(deep=True)
    changed_enemy = hidden_change.get_hero(HeroID("hero_arien"))
    assert changed_enemy is not None and changed_enemy.current_turn_card is not None
    changed_enemy.current_turn_card.id = "different_hidden_card"
    owner_change = state.model_copy(deep=True)
    changed_owner = owner_change.get_hero(HeroID("hero_razzle"))
    assert changed_owner is not None
    changed_owner.hand[0].id = "different_visible_card"
    public_change = state.model_copy(deep=True)
    changed_public_owner = public_change.get_hero(HeroID("hero_arien"))
    assert changed_public_owner is not None
    changed_public_card = changed_public_owner.played_cards[0]
    assert changed_public_card is not None
    changed_public_card.id = "public_card_beta"

    assert canonical_json_bytes(_encode(_snapshot(hidden_change))) == canonical_json_bytes(
        _encode(baseline)
    )
    assert canonical_json_bytes(_encode(_snapshot(owner_change))) != canonical_json_bytes(
        _encode(baseline)
    )
    assert canonical_json_bytes(_encode(_snapshot(public_change))) != canonical_json_bytes(
        _encode(baseline)
    )
    anonymous = [
        token
        for token in _tokens(_encode(baseline), "CARD")
        if _feature(token, "visibility") == "ANONYMOUS_BACK"
    ]
    assert anonymous
    assert all(_feature(token, "card_id") is None for token in anonymous)
    assert all("source_id" not in token["features"] for token in anonymous)
    assert all("different_hidden_card" not in token["local_ref"] for token in anonymous)
    assert {token["features"]["visibility"] for token in _tokens(_encode(baseline), "CARD")} >= {
        "IDENTIFIED",
        "ANONYMOUS_BACK",
        "COUNT_ONLY",
    }


def test_primary_tokens_exclude_mean_hero_progress() -> None:
    payload = _dump(_encode(_snapshot()))

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert "own_hero_progress_mean" not in keys(payload)
    assert "enemy_hero_progress_mean" not in keys(payload)
    assert "mean_hero_progress" not in keys(payload)


def test_adding_a_unit_changes_counts_without_changing_public_feature_fields() -> None:
    state = _state()
    before = _encode(_snapshot(state))
    without_state = state.model_copy(deep=True)
    blue = without_state.teams[TeamColor.BLUE]
    added = blue.minions.pop()
    # Removing and restoring through public state collections gives two legal
    # observable states with different physical-unit populations.
    without_state.remove_entity(added.id)
    without = _encode(_snapshot(without_state))

    assert len(_tokens(before, "UNIT")) == len(_tokens(without, "UNIT")) + 1
    assert {frozenset(token["features"]) for token in _tokens(before, "UNIT")} == {
        frozenset(token["features"]) for token in _tokens(without, "UNIT")
    }
