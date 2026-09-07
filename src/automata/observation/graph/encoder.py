"""Deterministic graph encoding of an information-safe public snapshot."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from automata.models.contracts import (
    LearnedObservation,
    ObservationRelationship,
    ObservationToken,
    PublicSnapshot,
)

_CARD_FIELDS = (
    "name",
    "image_id",
    "tier",
    "color",
    "primary_action",
    "primary_action_value",
    "secondary_actions",
    "effect_id",
    "effect_text",
    "initiative",
    "state",
    "is_facedown",
    "is_ranged",
    "range_value",
    "radius_value",
    "item",
    "is_active",
    "spell_rank",
)
_UNAVAILABLE_DISTANCE = -1.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: object) -> Iterable[tuple[str, Any]]:
    return sorted(_mapping(value).items(), key=lambda item: str(item[0]))


def _relation(team: object, snapshot: PublicSnapshot, entity_id: str | None = None) -> str:
    perspective = snapshot.viewer.perspective_team
    if (
        perspective is not None
        and team == perspective
        and entity_id is not None
        and entity_id == snapshot.viewer.private_hero_id
    ):
        return "SELF"
    if perspective is not None and team == perspective:
        return "ALLY"
    if perspective is None:
        return "PUBLIC"
    return "ENEMY"


def _hex_key(hex_: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(hex_.get("q", 0)), int(hex_.get("r", 0)), int(hex_.get("s", 0)))


def _tile_ref(hex_: Mapping[str, Any]) -> str:
    q, r, s = _hex_key(hex_)
    return f"tile:{q}:{r}:{s}"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class _GraphBuilder:
    def __init__(self, snapshot: PublicSnapshot) -> None:
        self.snapshot = snapshot
        self.public = _mapping(snapshot.public_state)
        self.tokens: list[ObservationToken] = []
        self.relationships: list[ObservationRelationship] = []
        self.refs: set[str] = set()
        self.team_refs: dict[str, str] = {}
        self.hero_refs: dict[str, str] = {}
        self.unit_refs: dict[str, str] = {}
        self.tile_refs: dict[tuple[int, int, int], str] = {}
        self.zone_refs: dict[str, str] = {}
        self.lane_refs: dict[str, str] = {}
        self.unit_data: dict[str, dict[str, Any]] = {}

    def context(self, name: str) -> Any:
        return _mapping(self.public.get("automata_context")).get(name)

    def token(self, local_ref: str, kind: str, features: dict[str, Any]) -> str:
        self.tokens.append(
            ObservationToken(schema_version=1, local_ref=local_ref, kind=kind, features=features)
        )
        self.refs.add(local_ref)
        return local_ref

    def edge(
        self, source_ref: str, target_ref: str, kind: str, features: dict[str, Any] | None = None
    ) -> None:
        if source_ref not in self.refs or target_ref not in self.refs:
            return
        self.relationships.append(
            ObservationRelationship(
                schema_version=1,
                source_ref=source_ref,
                target_ref=target_ref,
                kind=kind,
                features=features or {},
            )
        )

    def build(self) -> LearnedObservation:
        self._global_and_teams()
        self._heroes()
        self._board()
        self._units()
        self._cards()
        self._public_objects()
        self._structural_edges()
        self._unit_relationships()
        tokens = tuple(sorted(self.tokens, key=lambda token: (token.kind, token.local_ref)))
        relationships = tuple(
            sorted(
                self.relationships,
                key=lambda edge: (
                    edge.kind,
                    edge.source_ref,
                    edge.target_ref,
                    _canonical(edge.features),
                ),
            )
        )
        return LearnedObservation(
            schema_version=2,
            viewer=self.snapshot.viewer,
            decision_kind="SNAPSHOT",
            candidate_ids=(),
            features={},
            tokens=tokens,
            relationships=relationships,
        )

    def _global_and_teams(self) -> None:
        teams = _mapping(self.public.get("teams"))
        topology = _mapping(self.public.get("automata_topology"))
        lanes = _mapping(topology.get("lanes"))
        board = _mapping(self.public.get("board"))
        tiles = _mapping(board.get("tiles"))
        heroes = [hero for _, team in _items(teams) for hero in team.get("heroes", [])]
        self.token(
            "global:0",
            "GLOBAL",
            {
                "map_id": self.snapshot.map_id,
                "game_type": self.snapshot.game_type,
                "phase": self.public.get("phase"),
                "round": self.public.get("round"),
                "turn": self.public.get("turn"),
                "team_count": len(teams),
                "hero_count": len(heroes),
                "lane_count": len(lanes),
                "playable_tile_count": sum(
                    not bool(_mapping(tile).get("is_terrain")) for tile in tiles.values()
                ),
                "tie_breaker_team": self.public.get("tie_breaker_team"),
            },
        )
        pieces = _mapping(self.public.get("hero_pieces"))
        locations = _mapping(board.get("entity_locations"))
        for team_id, raw_team in _items(teams):
            team = _mapping(raw_team)
            team_heroes = list(team.get("heroes", []))
            minions = list(team.get("minions", []))
            levels = [int(_mapping(hero).get("level", 0)) for hero in team_heroes]
            gold = [int(_mapping(hero).get("gold", 0)) for hero in team_heroes]
            hero_ids = {str(_mapping(hero).get("id")) for hero in team_heroes}
            piece_count = sum(
                1 for piece in pieces.values() if _mapping(piece).get("owner_hero_id") in hero_ids
            )
            physical_count = piece_count + sum(
                1
                for hero_id in hero_ids
                if not any(
                    _mapping(piece).get("owner_hero_id") == hero_id for piece in pieces.values()
                )
            )
            alive = sum(
                hero_id in locations
                or any(
                    _mapping(piece).get("owner_hero_id") == hero_id
                    and _mapping(piece).get("position") is not None
                    for piece in pieces.values()
                )
                for hero_id in hero_ids
            )
            ref = self.token(
                f"team:{team_id}",
                "TEAM",
                {
                    "team_id": team_id,
                    "relation": (
                        "PUBLIC"
                        if self.snapshot.viewer.perspective_team is None
                        else (
                            "OWN" if team_id == self.snapshot.viewer.perspective_team else "ENEMY"
                        )
                    ),
                    "life_counters": team.get("life_counters"),
                    "hero_count": len(team_heroes),
                    "alive_hero_count": alive,
                    "physical_piece_count": physical_count,
                    "minion_count": len(minions),
                    "total_level": sum(levels),
                    "mean_level": sum(levels) / len(levels) if levels else 0.0,
                    "total_gold": sum(gold),
                    "mean_gold": sum(gold) / len(gold) if gold else 0.0,
                },
            )
            self.team_refs[team_id] = ref

    def _heroes(self) -> None:
        teams = _mapping(self.public.get("teams"))
        for team_id, raw_team in _items(teams):
            heroes = sorted(
                raw_team.get("heroes", []), key=lambda hero: str(_mapping(hero).get("id"))
            )
            for raw_hero in heroes:
                hero = _mapping(raw_hero)
                hero_id = str(hero.get("id"))
                ref = self.token(
                    f"hero:{hero_id}",
                    "HERO",
                    {
                        "hero_id": hero_id,
                        "name": hero.get("name"),
                        "title": hero.get("title"),
                        "team_id": team_id,
                        "team_ref": self.team_refs[team_id],
                        "relation": _relation(team_id, self.snapshot, hero_id),
                        "level": hero.get("level"),
                        "gold": hero.get("gold"),
                        "items": hero.get("items", {}),
                        "wish_cast_count": hero.get("wish_cast_count", 0),
                        "rune_slots": hero.get("rune_slots", {}),
                        "is_current_actor": hero_id == self.public.get("current_actor_id"),
                        "is_decision_owner": hero_id == self.context("resolution_owner_id"),
                        "is_acting_piece": hero_id == self.context("acting_piece_id"),
                        "adapter_features": {},
                    },
                )
                self.hero_refs[hero_id] = ref

    def _board(self) -> None:
        board = _mapping(self.public.get("board"))
        for _, raw_tile in _items(board.get("tiles")):
            tile = _mapping(raw_tile)
            hex_ = _mapping(tile.get("hex"))
            key = _hex_key(hex_)
            ref = self.token(
                _tile_ref(hex_),
                "TILE",
                {
                    "q": key[0],
                    "r": key[1],
                    "s": key[2],
                    "zone_id": tile.get("zone_id"),
                    "is_terrain": bool(tile.get("is_terrain")),
                    "has_occupant": tile.get("occupant_id") is not None,
                    "spawn_team": _mapping(tile.get("spawn_point")).get("team"),
                    "spawn_type": _mapping(tile.get("spawn_point")).get("type"),
                },
            )
            self.tile_refs[key] = ref
        for zone_id, raw_zone in _items(board.get("zones")):
            zone = _mapping(raw_zone)
            ref = self.token(
                f"zone:{zone_id}",
                "ZONE",
                {
                    "zone_id": zone_id,
                    "neighbor_count": len(zone.get("neighbors", [])),
                    "spawn_point_count": len(zone.get("spawn_points", [])),
                    "is_battle_zone": zone_id in _mapping(self.public.get("battle_zones")).values(),
                },
            )
            self.zone_refs[zone_id] = ref
        topology = _mapping(self.public.get("automata_topology"))
        battle_zones = _mapping(self.public.get("battle_zones"))
        waves = _mapping(self.public.get("wave_counters"))
        for lane_id, raw_zones in _items(topology.get("lanes")):
            zones = [str(zone) for zone in raw_zones]
            battle_zone = battle_zones.get(lane_id)
            index = zones.index(battle_zone) if battle_zone in zones else -1
            # Half-zone units preserve exact per-lane displacement without the
            # rounding drift introduced by normalizing each lane separately.
            advantage = index - (len(zones) - 1) / 2 if index >= 0 else 0.0
            if self.snapshot.viewer.perspective_team == "BLUE":
                advantage = -advantage
            ref = self.token(
                f"lane:{lane_id}",
                "LANE",
                {
                    "lane_id": lane_id,
                    "ordered_zone_refs": [self.zone_refs[zone] for zone in zones],
                    "zone_count": len(zones),
                    "battle_zone_ref": self.zone_refs.get(str(battle_zone)),
                    "battle_zone_index": index,
                    "signed_battle_zone_advantage": advantage,
                    "wave_counter": waves.get(lane_id),
                },
            )
            self.lane_refs[lane_id] = ref

    def _unit_features(
        self,
        *,
        entity_id: str,
        unit_type: str,
        team_id: str | None,
        owner_ref: str,
        position: Mapping[str, Any] | None,
        minion: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        tile_ref = _tile_ref(position) if position else None
        tile = next(
            (
                _mapping(raw)
                for raw in _mapping(_mapping(self.public.get("board")).get("tiles")).values()
                if position and _hex_key(_mapping(_mapping(raw).get("hex"))) == _hex_key(position)
            ),
            {},
        )
        zone_id = tile.get("zone_id")
        lane_id = next(
            (
                lane
                for lane, zones in _items(
                    _mapping(self.public.get("automata_topology")).get("lanes")
                )
                if zone_id in zones
            ),
            None,
        )
        role_id = entity_id
        if unit_type == "HERO_PIECE":
            role_id = next(
                (hero_id for hero_id, ref in self.hero_refs.items() if ref == owner_ref), entity_id
            )
        return {
            "entity_id": entity_id,
            "unit_type": unit_type,
            "team_id": team_id,
            "relation": _relation(team_id, self.snapshot, role_id),
            "owner_ref": owner_ref,
            "is_current_actor": role_id == self.public.get("current_actor_id"),
            "is_decision_owner": role_id == self.context("resolution_owner_id"),
            "is_acting_piece": entity_id == self.context("acting_piece_id"),
            "is_positioned": position is not None,
            "tile_ref": tile_ref,
            "zone_ref": self.zone_refs.get(str(zone_id)) if zone_id is not None else None,
            "lane_ref": self.lane_refs.get(str(lane_id)) if lane_id is not None else None,
            "minion_type": minion.get("type") if minion else None,
            "value": minion.get("value") if minion else None,
            "is_heavy": minion.get("is_heavy") if minion else None,
        }

    def _units(self) -> None:
        board = _mapping(self.public.get("board"))
        locations = _mapping(board.get("entity_locations"))
        pieces = _mapping(self.public.get("hero_pieces"))
        piece_owners = {_mapping(piece).get("owner_hero_id") for piece in pieces.values()}
        teams = _mapping(self.public.get("teams"))
        for team_id, team in _items(teams):
            for raw_hero in sorted(
                team.get("heroes", []), key=lambda hero: str(_mapping(hero).get("id"))
            ):
                hero_id = str(_mapping(raw_hero).get("id"))
                if hero_id in piece_owners and hero_id not in locations:
                    continue
                position = _mapping(locations[hero_id]) if hero_id in locations else None
                features = self._unit_features(
                    entity_id=hero_id,
                    unit_type="HERO",
                    team_id=team_id,
                    owner_ref=self.hero_refs[hero_id],
                    position=position,
                )
                self.unit_refs[hero_id] = self.token(f"unit:{hero_id}", "UNIT", features)
                self.unit_data[hero_id] = {"position": position, "features": features}
            for raw_minion in sorted(
                team.get("minions", []), key=lambda unit: str(_mapping(unit).get("id"))
            ):
                minion = _mapping(raw_minion)
                entity_id = str(minion.get("id"))
                position = _mapping(locations[entity_id]) if entity_id in locations else None
                features = self._unit_features(
                    entity_id=entity_id,
                    unit_type="MINION",
                    team_id=team_id,
                    owner_ref=self.team_refs[team_id],
                    position=position,
                    minion=minion,
                )
                self.unit_refs[entity_id] = self.token(f"unit:{entity_id}", "UNIT", features)
                self.unit_data[entity_id] = {"position": position, "features": features}
        for entity_id, raw_piece in _items(pieces):
            piece = _mapping(raw_piece)
            owner_id = str(piece.get("owner_hero_id"))
            position = (
                _mapping(piece.get("position")) if piece.get("position") is not None else None
            )
            features = self._unit_features(
                entity_id=entity_id,
                unit_type="HERO_PIECE",
                team_id=piece.get("team"),
                owner_ref=self.hero_refs[owner_id],
                position=position,
            )
            self.unit_refs[entity_id] = self.token(f"unit:{entity_id}", "UNIT", features)
            self.unit_data[entity_id] = {"position": position, "features": features}

    def _cards(self) -> None:
        card_records: list[tuple[str, str, str, Mapping[str, Any] | None, int]] = []
        for _, team in _items(self.public.get("teams")):
            for raw_hero in sorted(
                team.get("heroes", []), key=lambda hero: str(_mapping(hero).get("id"))
            ):
                hero = _mapping(raw_hero)
                owner_id = str(hero.get("id"))
                for area in (
                    "hand",
                    "deck",
                    "spellbook",
                    "cast_spells",
                    "played_cards",
                    "current_turn_card",
                    "extra_turn_card",
                    "discard_pile",
                    "ultimate_card",
                ):
                    raw = hero.get(area)
                    if isinstance(raw, Mapping) and set(raw) == {"count"}:
                        card_records.append((owner_id, area, "COUNT_ONLY", None, int(raw["count"])))
                    elif isinstance(raw, list):
                        for card in raw:
                            if isinstance(card, Mapping):
                                visibility = "IDENTIFIED" if "id" in card else "ANONYMOUS_BACK"
                                card_records.append((owner_id, area, visibility, card, 1))
                    elif isinstance(raw, Mapping):
                        visibility = "IDENTIFIED" if "id" in raw else "ANONYMOUS_BACK"
                        card_records.append((owner_id, area, visibility, raw, 1))
        card_records.sort(
            key=lambda item: (item[0], item[1], item[2], _canonical(item[3]), item[4])
        )
        anonymous_ordinals: dict[tuple[str, str, str], int] = {}
        for owner_id, area, visibility, card, count in card_records:
            card_id = str(card.get("id")) if card and card.get("id") is not None else None
            if visibility == "IDENTIFIED":
                local_ref = f"card:{owner_id}:{area}:{card_id}"
            else:
                ordinal_key = (owner_id, area, visibility)
                ordinal = anonymous_ordinals.get(ordinal_key, 0)
                anonymous_ordinals[ordinal_key] = ordinal + 1
                local_ref = f"card:{visibility.lower()}:{owner_id}:{area}:{ordinal}"
            details = {field: card.get(field) if card else None for field in _CARD_FIELDS}
            features = {
                "card_id": card_id,
                "owner_ref": self.hero_refs[owner_id],
                "area": area,
                "visibility": visibility,
                "count": count,
                **details,
            }
            self.token(local_ref, "CARD", features)

    def _public_objects(self) -> None:
        for raw_effect in sorted(
            self.public.get("effects", []), key=lambda item: str(_mapping(item).get("id"))
        ):
            effect = _mapping(raw_effect)
            effect_id = str(effect.get("id"))
            self.token(
                f"effect:{effect_id}",
                "EFFECT",
                {
                    "effect_id": effect_id,
                    "effect_type": effect.get("type"),
                    "duration": effect.get("duration"),
                    "is_active": effect.get("is_active"),
                    "scope": effect.get("scope"),
                    "stat_type": effect.get("stat_type"),
                    "stat_value": effect.get("stat_value"),
                    "split_axis": effect.get("split_axis"),
                    "split_value": effect.get("split_value"),
                    "named_color": effect.get("named_color"),
                },
            )
        for marker_type, raw_marker in _items(self.public.get("markers")):
            marker = _mapping(raw_marker)
            self.token(
                f"marker:{marker_type}",
                "MARKER",
                {
                    "marker_type": marker_type,
                    "target_ref": self.unit_refs.get(str(marker.get("target_id")))
                    or self.hero_refs.get(str(marker.get("target_id"))),
                    "source_ref": self.hero_refs.get(str(marker.get("source_id"))),
                    "value": marker.get("value"),
                },
            )
        for raw_token in sorted(
            self.public.get("tokens", []), key=lambda item: str(_mapping(item).get("id"))
        ):
            token = _mapping(raw_token)
            entity_id = str(token.get("id"))
            self.token(
                f"token:{entity_id}",
                "TOKEN",
                {
                    "entity_id": entity_id,
                    "name": token.get("name"),
                    "token_type": token.get("token_type"),
                    "owner_ref": self.hero_refs.get(str(token.get("owner_id"))),
                    "is_facedown": token.get("is_facedown"),
                    "is_passable": token.get("is_passable"),
                    "tile_ref": _tile_ref(_mapping(token.get("hex"))),
                },
            )
        for raw_entity in sorted(
            self.public.get("board_entities", []), key=lambda item: str(_mapping(item).get("id"))
        ):
            entity = _mapping(raw_entity)
            entity_id = str(entity.get("id"))
            hex_ = _mapping(entity.get("hex")) if entity.get("hex") is not None else None
            self.token(
                f"entity:{entity_id}",
                "ENTITY",
                {
                    "entity_id": entity_id,
                    "name": entity.get("name"),
                    "entity_kind": entity.get("entity_kind"),
                    "owner_ref": self.hero_refs.get(str(entity.get("owner_id"))),
                    "is_obstacle": entity.get("is_obstacle"),
                    "tile_ref": _tile_ref(hex_) if hex_ else None,
                },
            )

    def _structural_edges(self) -> None:
        for team_id, team_ref in self.team_refs.items():
            for hero_id, hero_ref in self.hero_refs.items():
                hero = next(
                    (
                        _mapping(item)
                        for team in _mapping(self.public.get("teams")).values()
                        for item in team.get("heroes", [])
                        if str(_mapping(item).get("id")) == hero_id
                    ),
                    {},
                )
                if hero.get("team") == team_id:
                    self.edge(team_ref, hero_ref, "OWNS")
        for entity_id, unit_ref in self.unit_refs.items():
            owner_ref = self.unit_data[entity_id]["features"]["owner_ref"]
            self.edge(owner_ref, unit_ref, "OWNS")
            position = self.unit_data[entity_id]["position"]
            if position:
                tile_ref = _tile_ref(position)
                self.edge(unit_ref, tile_ref, "POSITIONED_AT")
                zone_ref = self.unit_data[entity_id]["features"]["zone_ref"]
                lane_ref = self.unit_data[entity_id]["features"]["lane_ref"]
                if zone_ref:
                    self.edge(unit_ref, zone_ref, "IN_ZONE")
                if lane_ref:
                    self.edge(unit_ref, lane_ref, "IN_LANE")
        for token in self.tokens:
            if token.kind == "CARD" and isinstance(token.features.get("owner_ref"), str):
                self.edge(str(token.features["owner_ref"]), token.local_ref, "OWNS")
        board = _mapping(self.public.get("board"))
        for raw_tile in _mapping(board.get("tiles")).values():
            tile = _mapping(raw_tile)
            zone_ref = self.zone_refs.get(str(tile.get("zone_id")))
            if zone_ref:
                self.edge(_tile_ref(_mapping(tile.get("hex"))), zone_ref, "TILE_IN_ZONE")
        lanes = _mapping(_mapping(self.public.get("automata_topology")).get("lanes"))
        for lane_id, zones in _items(lanes):
            for zone_id in zones:
                self.edge(self.zone_refs[str(zone_id)], self.lane_refs[lane_id], "ZONE_IN_LANE")
        tile_keys = sorted(self.tile_refs)
        for source in tile_keys:
            for target in tile_keys:
                if source != target and max(abs(source[i] - target[i]) for i in range(3)) == 1:
                    self.edge(self.tile_refs[source], self.tile_refs[target], "HEX_ADJACENT")
        for zone_id, raw_zone in _items(board.get("zones")):
            for neighbor in sorted(raw_zone.get("neighbors", [])):
                if str(neighbor) in self.zone_refs:
                    self.edge(
                        self.zone_refs[zone_id], self.zone_refs[str(neighbor)], "ZONE_ADJACENT"
                    )

    def _unit_relationships(self) -> None:
        topology_relationships = _mapping(
            _mapping(self.public.get("automata_topology")).get("unit_relationships")
        )
        positioned = sorted(
            entity_id for entity_id, data in self.unit_data.items() if data["position"] is not None
        )
        lanes = _mapping(_mapping(self.public.get("automata_topology")).get("lanes"))
        for source_id in positioned:
            for target_id in positioned:
                source = _mapping(self.unit_data[source_id]["position"])
                target = _mapping(self.unit_data[target_id]["position"])
                source_key = _hex_key(source)
                target_key = _hex_key(target)
                delta = tuple(target_key[index] - source_key[index] for index in range(3))
                distance = max(abs(value) for value in delta)
                topology_facts = _mapping(
                    _mapping(topology_relationships.get(source_id)).get(target_id)
                )
                geometric_adjacent = distance == 1
                topology_adjacent = bool(topology_facts.get("is_adjacent", False))
                source_zone_ref = self.unit_data[source_id]["features"]["zone_ref"]
                target_zone_ref = self.unit_data[target_id]["features"]["zone_ref"]
                source_lane_ref = self.unit_data[source_id]["features"]["lane_ref"]
                target_lane_ref = self.unit_data[target_id]["features"]["lane_ref"]
                same_lane = source_lane_ref is not None and source_lane_ref == target_lane_ref
                progress_delta = 0.0
                if same_lane:
                    lane_id = str(source_lane_ref).removeprefix("lane:")
                    zone_ids = [str(zone) for zone in lanes.get(lane_id, [])]
                    source_zone = str(source_zone_ref).removeprefix("zone:")
                    target_zone = str(target_zone_ref).removeprefix("zone:")
                    if source_zone in zone_ids and target_zone in zone_ids:
                        progress_delta = (
                            zone_ids.index(target_zone) - zone_ids.index(source_zone)
                        ) / max(len(zone_ids) - 1, 1)
                values: dict[str, Any] = {
                    "delta_q": delta[0],
                    "delta_r": delta[1],
                    "delta_s": delta[2],
                    "hex_distance": distance,
                    "path_distance": _UNAVAILABLE_DISTANCE,
                    "path_exists": False,
                    "is_adjacent": geometric_adjacent,
                    "topology_is_adjacent": topology_adjacent,
                    "is_straight_line": bool(topology_facts.get("is_straight_line", False)),
                    "has_line_of_sight": False,
                    "same_zone": source_zone_ref is not None and source_zone_ref == target_zone_ref,
                    "same_lane": same_lane,
                    "lane_progress_delta": progress_delta,
                    "relation": self.unit_data[target_id]["features"]["relation"],
                    "reachable": False,
                    "threatens": False,
                    "supports": False,
                    "path_distance_valid": False,
                    "has_line_of_sight_valid": False,
                    "reachable_valid": False,
                    "threatens_valid": False,
                    "supports_valid": False,
                }
                self.edge(
                    self.unit_refs[source_id],
                    self.unit_refs[target_id],
                    "UNIT_TO_UNIT",
                    values,
                )


def encode_snapshot(snapshot: PublicSnapshot) -> LearnedObservation:
    """Encode only the information available in ``snapshot`` into graph v1."""
    return _GraphBuilder(snapshot).build()


__all__ = ["encode_snapshot"]
