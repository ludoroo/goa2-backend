"""Project mutable engine state through the authoritative visibility boundary."""

from __future__ import annotations

from typing import Any

from automata.models.contracts import PublicSnapshot, Viewer
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.domain.views import build_view
from goa2.engine.topology import get_topology_service

_RUNTIME_OR_NONDETERMINISTIC_FIELDS = frozenset(
    {
        "cheats_enabled",
        "clock",
        "time_control",
    }
)


def _viewing_hero_id(state: GameState, viewer: Viewer) -> HeroID | None:
    if viewer.private_hero_id is None:
        return None

    hero_id = HeroID(viewer.private_hero_id)
    hero = state.get_hero(hero_id)
    if hero is None:
        raise ValueError(f"viewer hero does not exist: {viewer.private_hero_id!r}")
    if viewer.perspective_team is not None and (
        hero.team is None or hero.team.value != viewer.perspective_team
    ):
        raise ValueError(
            f"viewer hero {viewer.private_hero_id!r} does not belong to team "
            f"{viewer.perspective_team!r}"
        )
    return hero_id


def _canonicalize_source_order(view: dict[str, Any]) -> None:
    """Stabilize list sections built from unordered state mappings."""
    board_entities = view.get("board_entities")
    if isinstance(board_entities, list):
        board_entities.sort(key=lambda entity: str(entity.get("id", "")))


def _public_topology(state: GameState) -> dict[str, Any]:
    """Project neutral topology facts without granting any unit immunity.

    Topology effects and every positioned entity are public. Clearing tile
    occupants on a private copy prevents the engine helper from applying
    unit-specific immunity while retaining moving isolation-source positions.
    """
    neutral = state.model_copy(deep=True)
    for tile in neutral.board.tiles.values():
        tile.occupant_id = None
    topology = get_topology_service()
    positioned = sorted(
        (str(entity_id), hex_) for entity_id, hex_ in state.entity_locations.items()
    )
    relationships: dict[str, dict[str, dict[str, bool]]] = {}
    for source_id, source_hex in positioned:
        relationships[source_id] = {
            target_id: {
                "is_adjacent": topology.are_adjacent(source_hex, target_hex, neutral, unit_ids=()),
                "is_straight_line": topology.is_straight_line(
                    source_hex, target_hex, neutral, unit_ids=()
                ),
            }
            for target_id, target_hex in positioned
        }
    return {
        "lanes": {
            lane_id: list(zone_ids) for lane_id, zone_ids in sorted(state.board.lanes.items())
        },
        "unit_relationships": relationships,
    }


def project_snapshot(
    state: GameState,
    viewer: Viewer,
    *,
    current_owner_id: str | None = None,
) -> PublicSnapshot:
    """Return the deterministic information available at ``viewer``'s scope."""
    if viewer.perspective_team is not None and viewer.perspective_team not in {
        team.value for team in state.teams
    }:
        raise ValueError(f"viewer team does not exist: {viewer.perspective_team!r}")
    hero_id = _viewing_hero_id(state, viewer)
    # Perspective never grants secrets; only private_hero_id reaches build_view.
    # Opponent hand size is intentionally omitted with the hidden hand itself.
    view = build_view(state, for_hero_id=hero_id, now_ms=0)
    for field in _RUNTIME_OR_NONDETERMINISTIC_FIELDS:
        view.pop(field, None)
    _canonicalize_source_order(view)
    view["automata_topology"] = _public_topology(state)
    view["automata_context"] = {
        "resolution_owner_id": current_owner_id or state.resolution_owner_id,
        "acting_piece_id": state.acting_piece_id,
    }

    return PublicSnapshot(
        schema_version=2,
        viewer=viewer,
        map_id=state.board.map_id,
        game_type=state.game_type.value,
        public_state=view,
    )


__all__ = ["project_snapshot"]
