from collections import deque
from collections.abc import Callable

from goa2.domain.board import DEFAULT_LANE_ID
from goa2.domain.hex import Hex
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.engine.topology import get_connected_neighbors


def check_lane_push_trigger(state: GameState, active_zone_id: str) -> TeamColor | None:
    """
    Checks if a Lane Push should occur in the active zone.
    Condition: Minion Count for a Team in BattleZone == 0.
    Returns the LOSING team (the one with 0 minions), or None.
    """
    if not active_zone_id:
        return None

    zone = state.board.zones.get(active_zone_id)
    if not zone:
        return None

    red_minions = 0
    blue_minions = 0

    for team_color, team in state.teams.items():
        count = 0
        for minion in team.minions:
            loc = state.unit_locations.get(minion.id)
            if loc and loc in zone.hexes:
                count += 1

        if team_color == TeamColor.RED:
            red_minions = count
        elif team_color == TeamColor.BLUE:
            blue_minions = count

    # If Red has 0 and Blue > 0 -> Red Loses Zone (Blue Pushes)
    # If Blue has 0 and Red > 0 -> Blue Loses Zone (Red Pushes)

    if red_minions == 0 and blue_minions > 0:
        return TeamColor.RED  # Red lost control
    elif blue_minions == 0 and red_minions > 0:
        return TeamColor.BLUE  # Blue lost control

    return None


def get_push_target_zone_id(
    state: GameState, losing_team: TeamColor, lane_id: str = DEFAULT_LANE_ID
) -> tuple[str | None, bool]:
    """
    Calculates the next zone ID for a push on the given lane.
    Returns (next_zone_id, is_game_over).
    """
    current_id = state.battle_zone_for_lane(lane_id)
    if not current_id:
        return None, False

    lane = state.board.lanes.get(lane_id, [])
    if not lane or current_id not in lane:
        return None, False

    idx = lane.index(current_id)

    # Lane is ordered RedBase -> BlueBase
    # Red Loses -> Index - 1 (Towards Red Base)
    # Blue Loses -> Index + 1 (Towards Blue Base)

    if losing_team == TeamColor.RED:
        new_idx = idx - 1
        if new_idx <= 0:
            return None, True  # Reached Red Base — game over
    else:  # BLUE
        new_idx = idx + 1
        if new_idx >= len(lane) - 1:
            return None, True  # Reached Blue Base — game over

    return lane[new_idx], False


def zones_between(state: GameState, team: TeamColor, lane_id: str, zone_id: str) -> int:
    """
    Number of zones strictly between a team's Throne and the given zone on
    a lane. Lanes are ordered RedBase -> BlueBase; for a zone at index i in
    a lane of length n: RED distance = i - 1, BLUE distance = n - 2 - i.
    Returns 0 for zones not on the lane.
    """
    lane = state.board.lanes.get(lane_id, [])
    if zone_id not in lane:
        return 0
    idx = lane.index(zone_id)
    if team == TeamColor.RED:
        return max(0, idx - 1)
    return max(0, len(lane) - 2 - idx)


def endgame_totals(
    state: GameState, bz_overrides: dict[str, str] | None = None
) -> dict[TeamColor, int]:
    """
    Total zones between each team's Throne and every lane's Battle Zone
    (the double-lane last-wave comparison). `bz_overrides` supplies
    hypothetical post-push positions as {lane_id: zone_id}.
    """
    overrides = bz_overrides or {}
    totals = {TeamColor.RED: 0, TeamColor.BLUE: 0}
    for lane_id, current_zone_id in state.battle_zones.items():
        zone_id = overrides.get(lane_id, current_zone_id)
        for team in (TeamColor.RED, TeamColor.BLUE):
            totals[team] += zones_between(state, team, lane_id, zone_id)
    return totals


def count_enemies(state: GameState, zone_id: str, team: TeamColor) -> int:
    """
    Counts HOSTILE units (Minions + Heroes) in a zone.
    """
    zone = state.board.zones.get(zone_id)
    if not zone:
        return 0

    count = 0
    for team_obj in state.teams.values():
        if team_obj.color != team:  # Hostile Team
            for minion in team_obj.minions:
                loc = state.get_position(str(minion.id))
                if loc and loc in zone.hexes:
                    count += 1

    for t_color, t_obj in state.teams.items():
        if t_color != team:
            for hero in t_obj.heroes:
                for loc in state.get_positions(str(hero.id)):
                    if loc in zone.hexes:
                        count += 1

    return count


def find_nearest_empty_hexes(
    state: GameState,
    start_hex: Hex,
    zone_id: str,
    respect_obstacles: bool = False,
    actor_id: str | None = None,
    *,
    destination_allowed: Callable[[Hex], bool] | None = None,
) -> list[Hex]:
    """
    Finds the nearest empty hex(es) to start_hex within the specified zone.
    Used for displacement/collision resolution.
    Returns equally-distant legal destinations; rejected spaces do not stop the search.
    """
    zone = state.board.zones.get(zone_id)
    if not zone:
        return []

    queue = deque([(start_hex, 0)])
    visited = {start_hex}

    candidates = []
    found_distance = None

    while queue:
        current, dist = queue.popleft()

        # Optimization: If we found candidates at distance X,
        # stop processing anything at distance X+1
        if found_distance is not None and dist > found_distance:
            break

        # Check Validity (Only if not start hex)
        if dist > 0 and current in zone.hexes:
            tile = state.board.get_tile(current)
            # Check for Obstacle/Occupancy
            # Note: Token is an obstacle. Unit is an occupant. Terrain is an obstacle.
            # Valid = Not Obstacle (terrain OR occupant).
            if (
                tile
                and not tile.is_obstacle
                and (destination_allowed is None or destination_allowed(current))
            ):
                candidates.append(current)
                found_distance = dist

        # Expand (only if we haven't found a closer layer yet)
        if found_distance is None:
            if respect_obstacles:
                from goa2.engine.topology import get_topology_service

                neighbors = get_topology_service().get_traversable_neighbors(
                    current, state, actor_id=actor_id, movement_origin=start_hex
                )
            else:
                neighbors = get_connected_neighbors(
                    current, state, unit_ids=[actor_id] if actor_id else None
                )

            for neighbor in neighbors:
                # SAFETY: Only expand to hexes that exist on the board
                if neighbor not in visited and state.board.is_on_map(neighbor):
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return candidates
