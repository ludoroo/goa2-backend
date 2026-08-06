from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from goa2.domain.hex import Hex
from goa2.domain.models import FilterType
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID, UnitID

# -----------------------------------------------------------------------------
# Base Filter
# -----------------------------------------------------------------------------
from goa2.engine.filters_base import BATCH_FREED_HEXES_KEY, FilterCondition
from goa2.engine.topology import get_topology_service


def _hexes_from_context_value(raw: Any) -> list[Hex]:
    """Normalize a context value into a list of Hex (accepts Hex or dict items)."""
    if not raw:
        return []
    items = raw if isinstance(raw, list) else [raw]
    return [Hex(**h) if isinstance(h, dict) else h for h in items]


class LineBehindTargetFilter(FilterCondition):
    """
    Selects hexes (or units on hexes) that are in a straight line directly BEHIND a target.
    Direction is defined by Origin -> Target.
    """

    type: FilterType = FilterType.LINE_BEHIND_TARGET
    target_key: str
    length: int = 1
    origin_id: str | None = None

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        # Resolve Target Location
        target_id = context.get(self.target_key)
        if not target_id:
            return False

        target_hex = state.entity_locations.get(BoardEntityID(target_id))
        if not target_hex:
            return False
        if isinstance(target_id, Hex):
            target_hex = target_id
        # Resolve Origin Location
        origin_uid = self.origin_id or state.current_actor_id
        if not origin_uid:
            return False

        origin_hex = state.get_position(str(origin_uid))
        if not origin_hex:
            return False

        # Resolve Candidate Location
        cand_hex = None
        if isinstance(candidate, Hex):
            cand_hex = candidate
        elif isinstance(candidate, str):
            cand_hex = state.entity_locations.get(BoardEntityID(candidate))

        if not cand_hex:
            return False

        # Logic:
        # 1. Origin and Target must be in straight line to establish direction.
        direction_idx = origin_hex.direction_to(target_hex)
        if direction_idx is None:
            return False

        # 2. Target and Candidate must be in same direction from Target
        # Note: Candidate must be strictly BEHIND target, not AT target.
        if cand_hex == target_hex:
            return False

        cand_dir = target_hex.direction_to(cand_hex)
        if cand_dir != direction_idx:
            return False

        # 3. Distance check (topology-aware)
        topology = get_topology_service()
        dist = topology.distance(target_hex, cand_hex, state)
        return dist <= self.length


class NotInStraightLineFilter(FilterCondition):
    """
    Excludes targets in a straight line from the actor.
    Uses topology-aware is_straight_line() (respects reality splits).

    Per card text: "Units adjacent to you are in a straight line from you."
    Adjacent hexes are always in a straight line in cube coordinates.

    Used by: Charged Boomerang, Telekinesis, Mass Telekinesis, Thunder Boomerang
    """

    type: FilterType = FilterType.NOT_IN_STRAIGHT_LINE
    origin_id: str | None = None  # Literal ID (defaults to current actor)
    origin_key: str | None = None  # Key in context to find ID

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        # Resolve origin
        origin_uid = None
        if self.origin_id:
            origin_uid = self.origin_id
        elif self.origin_key:
            origin_uid = context.get(self.origin_key)

        if not origin_uid:
            origin_uid = state.current_actor_id

        if not origin_uid:
            return False

        origin_hex = state.get_position(str(origin_uid))
        if not origin_hex:
            return False

        # Resolve candidate hex
        target_hex = None
        if isinstance(candidate, Hex):
            target_hex = candidate
        elif isinstance(candidate, str):
            target_hex = state.entity_locations.get(BoardEntityID(candidate))

        if not target_hex:
            return False

        # Use topology-aware is_straight_line (respects reality splits)
        # Returns True if NOT in straight line (i.e., valid target)
        return not get_topology_service().is_straight_line(origin_hex, target_hex, state)


class InStraightLineFilter(FilterCondition):
    """
    Includes targets in a straight line from the actor.
    Uses topology-aware is_straight_line() (respects reality splits).

    Per card text: "Units adjacent to you are in a straight line from you."
    Adjacent hexes are always in a straight line in cube coordinates.

    Used by: Cards that require targets to be aligned with the actor
    """

    type: FilterType = FilterType.IN_STRAIGHT_LINE
    origin_id: str | None = None  # Literal ID (defaults to current actor)
    origin_key: str | None = None  # Key in context to find ID

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        # Resolve origin
        origin_uid = None
        if self.origin_id:
            origin_uid = self.origin_id
        elif self.origin_key:
            origin_uid = context.get(self.origin_key)

        if not origin_uid:
            origin_uid = state.current_actor_id

        if not origin_uid:
            return False

        origin_hex = state.get_position(str(origin_uid))
        if not origin_hex:
            return False

        # Resolve candidate hex
        target_hex = None
        if isinstance(candidate, Hex):
            target_hex = candidate
        elif isinstance(candidate, str):
            target_hex = state.entity_locations.get(BoardEntityID(candidate))

        if not target_hex:
            return False

        # Use topology-aware is_straight_line (respects reality splits)
        # Returns True if IN straight line (i.e., valid target)
        return get_topology_service().is_straight_line(origin_hex, target_hex, state)


class StraightLinePathFilter(FilterCondition):
    """
    Validates that the straight-line path between origin and candidate is
    traversable — every intermediate hex must exist on the board and be clear.

    Unlike MovementPathFilter (BFS-based), this checks only the direct
    straight-line path, blocking if any intermediate hex is occupied or missing.
    """

    type: FilterType = FilterType.STRAIGHT_LINE_PATH
    origin_id: str | None = None
    origin_key: str | None = None
    pass_through_obstacles: bool = False

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, Hex):
            return False

        # Resolve origin
        origin_uid = None
        if self.origin_id:
            origin_uid = self.origin_id
        elif self.origin_key:
            origin_uid = context.get(self.origin_key)

        if not origin_uid:
            origin_uid = state.current_actor_id

        if not origin_uid:
            return False

        origin_hex = state.get_position(str(origin_uid))
        if not origin_hex:
            return False

        # Not in straight line → reject
        if not origin_hex.is_straight_line(candidate):
            return False

        # Get intermediate hexes (line_to returns origin-exclusive, destination-inclusive)
        try:
            path = origin_hex.line_to(candidate)
        except ValueError:
            return False

        actor_id = str(origin_uid) if origin_uid else None

        # Check all intermediate hexes (everything except the final destination)
        for hex_pos in path[:-1]:
            if hex_pos not in state.board.tiles:
                return False

            if self.pass_through_obstacles:
                continue

            if state.validator.is_obstacle_for_actor(
                state, hex_pos, actor_id, context
            ) and not state.validator.is_passable_token(state, hex_pos):
                return False

        return True


class HasStraightLineDestinationFilter(FilterCondition):
    """
    Passes candidate units that could actually be moved ``distance`` spaces in
    some straight line — i.e. at least one of the six rays ends on a free,
    reachable hex.

    Without this, a hero with every line blocked would still be offered as a
    target and then dead-end at the destination select. It deliberately reuses
    the same filters the destination select applies, so "offered" and
    "selectable" can never disagree.
    """

    type: FilterType = FilterType.HAS_STRAIGHT_LINE_DESTINATION
    distance: int = 2
    pass_through_obstacles: bool = False

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, str):
            return False

        origin = state.get_position(candidate)
        if origin is None:
            return False

        from goa2.engine.filters_hex import ObstacleFilter, RangeFilter

        in_range = RangeFilter(
            min_range=self.distance, max_range=self.distance, origin_id=candidate
        )
        path_clear = StraightLinePathFilter(
            origin_id=candidate, pass_through_obstacles=self.pass_through_obstacles
        )
        landing_free = ObstacleFilter(is_obstacle=False)

        for direction in range(6):
            step = origin.neighbor(direction) - origin
            destination = origin + step.scale(self.distance)
            if destination not in state.board.tiles:
                continue
            if not in_range.apply(destination, state, context):
                continue
            if not path_clear.apply(destination, state, context):
                continue
            if landing_free.apply(destination, state, context):
                return True

        return False


class FarthestEmptyAdjacentFilter(FilterCondition):
    """Passes only the empty hex(es) adjacent to an anchor that are farthest from
    the origin (topology distance).

    For Stone Grip: "place ... adjacent to an enemy hero ... as far away from you
    as possible." Recomputed per call, so after each placement (which occupies a
    hex) the next-farthest empties become eligible. Ties at the max distance all
    pass (the actor picks among them).

    Batch-search hints: because this filter derives emptiness itself, the batch
    completability search (which cannot mutate the board while exploring) feeds
    it hypotheses through context —

    - hexes at ``occupied_hex_keys`` (earlier batch slots) count as occupied;
    - hexes under ``BATCH_FREED_HEXES_KEY`` (removals assumed by the search)
      count as empty.

    Both adjust the candidate set AND the max-distance computation, keeping the
    search consistent with what the live board will look like at placement time.
    """

    type: FilterType = FilterType.FARTHEST_EMPTY_ADJACENT
    origin_id: str | None = None
    origin_key: str | None = None
    anchor_key: str = "anchor_unit"
    occupied_hex_keys: list[str] = Field(default_factory=list)

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, Hex):
            return False

        origin_uid = self.origin_id
        if not origin_uid and self.origin_key:
            origin_uid = context.get(self.origin_key)
        if not origin_uid:
            origin_uid = state.current_actor_id
        if not origin_uid:
            return False
        origin_hex = state.get_position(str(origin_uid))

        anchor_id = context.get(self.anchor_key)
        anchor_hex = state.get_position(str(anchor_id)) if anchor_id else None
        if not origin_hex or not anchor_hex:
            return False

        freed = _hexes_from_context_value(context.get(BATCH_FREED_HEXES_KEY))
        occupied: list[Hex] = []
        for key in self.occupied_hex_keys:
            occupied.extend(_hexes_from_context_value(context.get(key)))

        topology = get_topology_service()
        empties = [
            h
            for h in topology.get_connected_ring(anchor_hex, 1, state)
            if state.board.is_on_map(h)
            and h not in occupied
            and (not state.board.get_tile(h).is_obstacle or h in freed)
        ]
        if candidate not in empties:
            return False
        max_dist = max(topology.distance(origin_hex, h, state) for h in empties)
        return topology.distance(origin_hex, candidate, state) == max_dist


class SameDirectionFromOriginFilter(FilterCondition):
    """Candidate hex is in the same direction from the origin as a reference hex.

    Used for "move in the direction of the push": record the push target's
    ORIGINAL hex, then offer the actor destinations along that exact ray (the
    direction is one of the 6 axes). The origin hex itself never matches.
    """

    type: FilterType = FilterType.SAME_DIRECTION_FROM_ORIGIN
    origin_id: str | None = None
    origin_key: str | None = None
    origin_hex_key: str | None = None  # context key holding a Hex (or dict) origin
    reference_key: str = "direction_reference_hex"

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        # Candidate may be a hex (destination selection) or a unit-id string
        # (e.g. "a hero in the direction of the move").
        if isinstance(candidate, Hex):
            cand_hex: Hex | None = candidate
        elif isinstance(candidate, str):
            cand_hex = state.entity_locations.get(BoardEntityID(candidate))
        else:
            return False
        if cand_hex is None:
            return False

        # Origin may be a recorded hex (e.g. a pre-move position that is no longer
        # any unit's location) or resolved from a unit id.
        origin_hex = None
        if self.origin_hex_key:
            raw_origin = context.get(self.origin_hex_key)
            if raw_origin is None:
                return False
            origin_hex = Hex(**raw_origin) if isinstance(raw_origin, dict) else raw_origin
        else:
            origin_uid = self.origin_id
            if not origin_uid and self.origin_key:
                origin_uid = context.get(self.origin_key)
            if not origin_uid:
                origin_uid = state.current_actor_id
            if not origin_uid:
                return False
            origin_hex = state.get_position(str(origin_uid))
        if not origin_hex:
            return False

        ref = context.get(self.reference_key)
        if ref is None:
            return False
        ref_hex = Hex(**ref) if isinstance(ref, dict) else ref

        # Candidate must be in the same reality as the origin (a reality split
        # blocks "the direction of the move" just like it blocks distance/line).
        if not get_topology_service().are_connected(origin_hex, cand_hex, state):
            return False

        ref_dir = origin_hex.direction_to(ref_hex)
        cand_dir = origin_hex.direction_to(cand_hex)
        return ref_dir is not None and cand_dir == ref_dir


class MaxEmptySpacesInLineFilter(FilterCondition):
    """Caps the number of EMPTY hexes a straight-line move passes through.

    For "Move any number of spaces in a straight line, ignoring obstacles,
    without moving through more than N empty spaces." Only on-map empty interior
    hexes (no terrain, no occupant) count toward the budget; obstacles passed
    through do not. Start and destination hexes never count (line_to is
    origin-exclusive, and we drop the destination). Pair with
    StraightLinePathFilter(pass_through_obstacles=True), which already rejects
    off-map interiors.
    """

    type: FilterType = FilterType.MAX_EMPTY_SPACES_IN_LINE
    origin_id: str | None = None
    origin_key: str | None = None
    max_empty: int = 1

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, Hex):
            return False

        origin_uid = self.origin_id
        if not origin_uid and self.origin_key:
            origin_uid = context.get(self.origin_key)
        if not origin_uid:
            origin_uid = state.current_actor_id
        if not origin_uid:
            return False

        origin_hex = state.get_position(str(origin_uid))
        if not origin_hex or not origin_hex.is_straight_line(candidate):
            return False

        try:
            path = origin_hex.line_to(candidate)
        except ValueError:
            return False

        empties = 0
        for hex_pos in path[:-1]:  # interior hexes only (destination dropped)
            if not state.board.is_on_map(hex_pos):
                continue
            tile = state.board.get_tile(hex_pos)
            if not tile.is_obstacle:  # on-map empty floor consumes the budget
                empties += 1
        return empties <= self.max_empty


class SpaceBehindEmptyFilter(FilterCondition):
    """
    For unit targeting: validates that the hex directly behind the candidate
    (from the origin's perspective) exists on the board and is not an obstacle.

    Used by Blink Strike to ensure the hero can land behind the selected enemy.
    """

    type: FilterType = FilterType.SPACE_BEHIND_EMPTY
    origin_id: str | None = None
    origin_key: str | None = None

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        # Candidate is a unit ID string
        if not isinstance(candidate, str):
            return False

        # Resolve origin (hero position)
        origin_uid = None
        if self.origin_id:
            origin_uid = self.origin_id
        elif self.origin_key:
            origin_uid = context.get(self.origin_key)
        if not origin_uid:
            origin_uid = state.current_actor_id
        if not origin_uid:
            return False

        origin_hex = state.get_position(str(origin_uid))
        if not origin_hex:
            return False

        # Get candidate unit's hex
        candidate_hex = state.entity_locations.get(BoardEntityID(str(candidate)))
        if not candidate_hex:
            return False

        # Compute behind hex: candidate + (candidate - origin)
        diff = candidate_hex - origin_hex
        behind = candidate_hex + diff

        # Must be on the board
        if behind not in state.board.tiles:
            return False

        # Must not be an obstacle for the actor
        actor_id = str(origin_uid)
        is_obs = state.validator.is_obstacle_for_actor(state, behind, actor_id, context)
        return not is_obs


class RelativeDistanceFilter(FilterCondition):
    """
    Compares the distance(origin, candidate) against the distance(origin, reference)
    using a configurable operator.

    General-purpose replacement for PreserveDistanceFilter (operator="==").
    Also supports "farther away" (operator=">"), "closer" (operator="<"), etc.
    """

    type: FilterType = FilterType.RELATIVE_DISTANCE
    reference_key: str
    origin_id: str | None = None
    operator: Literal[">", ">=", "==", "<=", "<"] = ">"
    origin_key: str | None = None

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        origin_uid = self.origin_id
        if not origin_uid and self.origin_key:
            origin_uid = context.get(self.origin_key)
        if not origin_uid:
            origin_uid = state.current_actor_id
        if not origin_uid:
            return False
        origin_hex = state.get_position(str(origin_uid))
        if not origin_hex:
            return False

        ref_uid = context.get(self.reference_key)
        if not ref_uid:
            return False
        ref_hex = state.get_position(str(ref_uid))
        if not ref_hex:
            return False

        cand_hex = None
        if isinstance(candidate, Hex):
            cand_hex = candidate
        elif isinstance(candidate, str):
            cand_hex = state.entity_locations.get(BoardEntityID(candidate))
        if not cand_hex:
            return False

        topology = get_topology_service()
        current_dist = topology.distance(origin_hex, ref_hex, state)
        new_dist = topology.distance(origin_hex, cand_hex, state)

        ops = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
            "<=": lambda a, b: a <= b,
            "<": lambda a, b: a < b,
        }
        return ops[self.operator](new_dist, current_dist)


class ClearLineOfSightFilter(FilterCondition):
    """
    Validates that the straight-line path between origin and candidate has no
    blocking hexes in between.  Only intermediate hexes are checked — the
    destination itself is never a blocker.  Candidates not in a straight line
    from the origin are rejected outright.

    Configurable blockers:
    - blocked_by_units: occupied hexes block the line
    - blocked_by_terrain: terrain hexes block the line (uses validator for
      PETRIFY-awareness)

    Works with both Hex and unit-ID candidates (resolves unit → hex).
    """

    type: FilterType = FilterType.CLEAR_LINE_OF_SIGHT
    blocked_by_units: bool = True
    blocked_by_terrain: bool = True
    blocked_by_obstacles: bool = False
    origin_id: str | None = None
    origin_key: str | None = None

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        # Resolve candidate hex
        cand_hex: Hex | None = None
        if isinstance(candidate, Hex):
            cand_hex = candidate
        elif isinstance(candidate, str):
            cand_hex = state.entity_locations.get(BoardEntityID(candidate))
        if not cand_hex:
            return False

        # Resolve origin
        origin_uid = self.origin_id
        if not origin_uid and self.origin_key:
            origin_uid = context.get(self.origin_key)
        if not origin_uid:
            origin_uid = state.current_actor_id
        if not origin_uid:
            return False

        origin_hex = state.get_position(str(origin_uid))
        if not origin_hex:
            return False

        if not origin_hex.is_straight_line(cand_hex):
            return False

        try:
            path = origin_hex.line_to(cand_hex)
        except ValueError:
            return False

        # Check only intermediate hexes (exclude destination)
        for hex_pos in path[:-1]:
            if hex_pos not in state.board.tiles:
                return False

            tile = state.board.tiles[hex_pos]

            if self.blocked_by_terrain:
                is_terrain = (
                    state.validator.is_terrain_hex(state, hex_pos)
                    if state.validator
                    else tile.is_terrain
                )
                if is_terrain:
                    return False

            if (
                self.blocked_by_units
                and tile.occupant_id is not None
                and state.get_unit(UnitID(str(tile.occupant_id))) is not None
            ):
                return False

            if self.blocked_by_obstacles and state.validator:
                actor_uid = str(origin_uid) if origin_uid else None
                if state.validator.is_obstacle_for_actor(state, hex_pos, actor_uid):
                    return False

        return True


class CoMoverValidHexFilter(FilterCondition):
    """
    Validates a candidate hex as a destination for an anchor unit when a
    partner unit must mirror the move co-directionally (same offset vector).

    Resolves anchor and partner from context. For a candidate hex C, computes
    offset = C - anchor_current and partner_dest = partner_current + offset.
    Validates:

      * Anchor's landing (C) is on-board, not blocked terrain/obstacle, and
        not occupied by any unit other than the partner (which is leaving).
      * Partner's landing (partner_dest) satisfies the same, with anchor
        excluded.
      * If ignore_path_obstacles=False, both units' straight-line paths from
        their starts to their landings are clear, treating each unit's own
        starting hex as empty on the other's path (they "move together").

    The straight-line and distance constraints on the candidate hex itself
    are intentionally NOT checked here — compose with `InStraightLineFilter`
    and `RangeFilter(min_range, max_range)` at the call site.
    """

    type: FilterType = FilterType.CO_MOVER_VALID_HEX
    anchor_key: str
    partner_key: str
    ignore_path_obstacles: bool = False

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, Hex):
            return False

        anchor_id = context.get(self.anchor_key)
        partner_id = context.get(self.partner_key)
        if not anchor_id or not partner_id:
            return False

        anchor_hex = state.entity_locations.get(BoardEntityID(str(anchor_id)))
        partner_hex = state.entity_locations.get(BoardEntityID(str(partner_id)))
        if not anchor_hex or not partner_hex or anchor_hex == partner_hex:
            return False

        if not anchor_hex.is_straight_line(candidate) or candidate == anchor_hex:
            return False

        offset = candidate - anchor_hex
        partner_dest = partner_hex + offset

        actor_id = str(state.current_actor_id) if state.current_actor_id else None
        moving_ids = {str(anchor_id), str(partner_id)}

        if not self._landing_ok(state, candidate, moving_ids, actor_id, context):
            return False
        if not self._landing_ok(state, partner_dest, moving_ids, actor_id, context):
            return False

        if not self.ignore_path_obstacles:
            if not self._path_clear(
                state, anchor_hex, candidate, str(partner_id), actor_id, context
            ):
                return False
            if not self._path_clear(
                state, partner_hex, partner_dest, str(anchor_id), actor_id, context
            ):
                return False

        return True

    @staticmethod
    def _landing_ok(
        state: GameState,
        hex_pos: Hex,
        moving_ids: set[str],
        actor_id: str | None,
        context: dict,
    ) -> bool:
        if hex_pos not in state.board.tiles:
            return False
        tile = state.board.tiles[hex_pos]
        # Reject if a non-moving unit occupies the landing.
        if tile.occupant_id and str(tile.occupant_id) not in moving_ids:
            return False
        # Terrain/wall checks still apply even on a hex that one of the
        # moving units is vacating.
        if state.validator and state.validator.is_terrain_hex(state, hex_pos):
            return False
        # If empty (or only the leaving co-mover), use the validator for any
        # remaining obstacle classes (e.g. impassable tokens).
        if not tile.occupant_id:
            return not state.validator.is_obstacle_for_actor(state, hex_pos, actor_id, context)
        return True

    @staticmethod
    def _path_clear(
        state: GameState,
        start: Hex,
        end: Hex,
        excluded_unit_id: str,
        actor_id: str | None,
        context: dict,
    ) -> bool:
        try:
            path = start.line_to(end)
        except ValueError:
            return False
        # path is [step1, ..., end]; intermediates only — landing is checked
        # separately by _landing_ok.
        for hex_pos in path[:-1]:
            if hex_pos not in state.board.tiles:
                return False
            tile = state.board.tiles[hex_pos]
            if tile.occupant_id and str(tile.occupant_id) == excluded_unit_id:
                continue
            if state.validator.is_obstacle_for_actor(
                state, hex_pos, actor_id, context
            ) and not state.validator.is_passable_token(state, hex_pos):
                return False
        return True


class BetweenHexesFilter(FilterCondition):
    """
    Unit filter: passes if the candidate unit sits on the straight-line path
    between two hexes stored in context (exclusive of both endpoints).

    Used by Misa's BLUE cards to find enemies crossed during a straight-line
    move through: select the destination, then find any enemy who was between
    the origin and destination.
    """

    type: FilterType = FilterType.BETWEEN_HEXES
    from_hex_key: str
    to_hex_key: str

    def _resolve_hex(self, context: dict, key: str) -> Hex | None:
        raw = context.get(key)
        if isinstance(raw, Hex):
            return raw
        if isinstance(raw, dict):
            try:
                return Hex(**raw)
            except Exception:
                return None
        return None

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        from_hex = self._resolve_hex(context, self.from_hex_key)
        to_hex = self._resolve_hex(context, self.to_hex_key)
        if from_hex is None or to_hex is None:
            return False
        if from_hex == to_hex:
            return False
        if not from_hex.is_straight_line(to_hex):
            return False

        # Resolve candidate's current hex
        cand_hex: Hex | None = None
        if isinstance(candidate, Hex):
            cand_hex = candidate
        elif isinstance(candidate, str):
            cand_hex = state.entity_locations.get(BoardEntityID(candidate))
        if cand_hex is None:
            return False

        # line_to returns [next, next, ..., to_hex]; strip the endpoint so we
        # only keep strictly intermediate hexes.
        try:
            path = from_hex.line_to(to_hex)
        except ValueError:
            return False
        intermediate = path[:-1]
        return cand_hex in intermediate
