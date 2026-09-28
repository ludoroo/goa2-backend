from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from goa2.domain.hex import Hex
from goa2.domain.models import FilterType, Minion, Unit, is_hero_unit
from goa2.domain.models.enums import (
    ActionType,
    MinionType,
    TokenType,
)
from goa2.domain.models.marker import MarkerType
from goa2.domain.models.token import Token
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID

# -----------------------------------------------------------------------------
# Base Filter
# -----------------------------------------------------------------------------
from goa2.engine.filters_base import FilterCondition
from goa2.engine.topology import get_topology_service


class TeamFilter(FilterCondition):
    type: FilterType = FilterType.TEAM
    relation: Literal["FRIENDLY", "ENEMY", "SELF"]
    # RELATIVE to the actor executing the step

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        actor_id = state.current_actor_id
        if not actor_id:
            return False

        actor = state.get_entity(BoardEntityID(actor_id))
        target = state.get_entity(BoardEntityID(candidate)) if isinstance(candidate, str) else None

        if not actor or not target:
            # Only warn if strict logic required. For now, fail silently (filter mismatch)
            return False

        # Explicitly check if attributes are not None for Mypy
        actor_team = getattr(actor, "team", None)
        target_team = getattr(target, "team", None)

        # Illusion tokens count as minions of the equivalence team while the
        # effect's source performs actions (NebKher Illusionary Force/Army).
        if target_team is None and isinstance(candidate, str):
            from goa2.engine.rules import illusion_minion_team, is_equivalent_illusion

            if is_equivalent_illusion(state, candidate):
                target_team = illusion_minion_team(state)

        if actor_team is None or target_team is None:
            return False

        if self.relation == "SELF":
            return actor.id == target.id

        is_same_team = actor_team == target_team

        if self.relation == "FRIENDLY":
            if not is_same_team or actor.id == target.id:
                return False
            # The acting piece of a multi-piece hero is "you", not "another
            # hero"; her other pieces remain valid friendly heroes.
            is_acting_piece = bool(
                state.acting_piece_id
                and str(target.id) == str(state.acting_piece_id)
                and getattr(target, "owner_hero_id", None) == str(actor.id)
            )
            return not is_acting_piece
        elif self.relation == "ENEMY":
            return not is_same_team

        return False


class UnitTypeFilter(FilterCondition):
    type: FilterType = FilterType.UNIT_TYPE
    unit_type: Literal["HERO", "MINION", "TOKEN"]

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        entity = state.get_entity(BoardEntityID(candidate)) if isinstance(candidate, str) else None
        if not entity:
            return False

        if self.unit_type == "HERO":
            return is_hero_unit(entity)
        elif self.unit_type == "MINION":
            if isinstance(entity, Minion):
                return True
            # Illusion tokens count as minions while equivalence is active
            # for the current actor (NebKher Illusionary Force/Army).
            from goa2.engine.rules import is_equivalent_illusion

            return is_equivalent_illusion(state, str(candidate))
        elif self.unit_type == "TOKEN":
            return isinstance(entity, Token)
        return False


class ContextIdsFilter(FilterCondition):
    """Restrict a unit selection to IDs captured earlier in the action."""

    type: FilterType = FilterType.CONTEXT_IDS
    ids_key: str

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        ids = context.get(self.ids_key, [])
        return isinstance(ids, list) and str(candidate) in {str(item) for item in ids}


class HeroPieceFilter(FilterCondition):
    """Candidate is a HeroPiece owned by the current actor."""

    type: FilterType = FilterType.HERO_PIECE
    exclude_acting: bool = True

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        from goa2.domain.models.unit import HeroPiece

        if not isinstance(candidate, str):
            return False
        entity = state.misc_entities.get(BoardEntityID(candidate))
        if not isinstance(entity, HeroPiece):
            return False
        actor_owner = (
            state.get_hero(BoardEntityID(str(state.current_actor_id)))
            if state.current_actor_id
            else None
        )
        if actor_owner is None or entity.owner_hero_id != str(actor_owner.id):
            return False
        return not (self.exclude_acting and str(state.acting_piece_id) == candidate)


class TokenTypeFilter(FilterCondition):
    type: FilterType = FilterType.TOKEN_TYPE
    token_type: TokenType | None = None  # None matches any token

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        entity = state.get_entity(BoardEntityID(candidate)) if isinstance(candidate, str) else None
        if not isinstance(entity, Token):
            return False
        return self.token_type is None or entity.token_type == self.token_type


class MinionTypesFilter(FilterCondition):
    type: FilterType = FilterType.MINION_TYPES
    minion_types: list[MinionType]

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        entity = state.get_entity(BoardEntityID(candidate)) if isinstance(candidate, str) else None
        if not entity:
            return False
        if not isinstance(entity, Minion):
            # Illusion tokens count as MELEE minions while equivalence is
            # active for the current actor (NebKher Illusionary Force/Army).
            from goa2.engine.rules import is_equivalent_illusion

            return MinionType.MELEE in self.minion_types and is_equivalent_illusion(
                state, str(candidate)
            )

        return entity.type in self.minion_types


class AdjacencyFilter(FilterCondition):
    """
    Requires the target to be adjacent to a unit matching specific tags.
    E.g. "Adjacent to a Friendly Hero"

    This is a presence check, not a targeting check: immune units count. Under
    the second-printing rules an immune unit still satisfies conditions like
    "if you are adjacent to an enemy" — immunity only blocks being targeted or
    affected, which ``ImmunityFilter`` enforces on UNIT selections.
    """

    type: FilterType = FilterType.ADJACENCY
    target_tags: list[
        Literal["FRIENDLY", "ENEMY", "HERO", "MINION"]
    ]  # Tags are checked in AND fashion (must match all)

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        cand_hex = None
        if isinstance(candidate, Hex):
            cand_hex = candidate
        elif isinstance(candidate, str):
            cand_hex = state.entity_locations.get(BoardEntityID(candidate))

        if not cand_hex:
            return False

        # Use topology-aware neighbors (respects reality splits)
        topology = get_topology_service()
        neighbors = topology.get_connected_neighbors(cand_hex, state)

        for n in neighbors:
            tile = state.board.get_tile(n)
            if not tile or not tile.occupant_id:
                continue

            # Use Unified Lookup
            occupant = state.get_entity(tile.occupant_id)
            if not occupant:
                continue

            actor_id = state.current_actor_id
            actor = state.get_entity(BoardEntityID(str(actor_id))) if actor_id else None

            if not actor:
                continue

            matches = True
            for tag in self.target_tags:
                if tag == "FRIENDLY":
                    # Mypy safety checks
                    occ_team = getattr(occupant, "team", None)
                    act_team = getattr(actor, "team", None)
                    if (
                        occ_team is None
                        or act_team is None
                        or occ_team != act_team
                        or occupant.id == actor.id
                    ):
                        matches = False
                elif tag == "ENEMY":
                    occ_team = getattr(occupant, "team", None)
                    act_team = getattr(actor, "team", None)
                    if occ_team is None or act_team is None or occ_team == act_team:
                        matches = False
                elif tag == "HERO":
                    if not is_hero_unit(occupant):
                        matches = False
                elif tag == "MINION":
                    if not isinstance(occupant, Minion):
                        matches = False

            if matches:
                return True

        return False


class ImmunityFilter(FilterCondition):
    """
    Filters out candidates that are Immune.

    Checks two sources of immunity:
    1. Standard minion immunity (Heavy minions with friendly support)
    2. ATTACK_IMMUNITY effects (e.g., Expert Duelist - immune to attacks except from specific attacker)
    """

    type: FilterType = FilterType.IMMUNITY

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        from goa2.domain.models.effect import EffectType
        from goa2.engine import rules  # Import inside to be safe
        from goa2.engine.stats import _is_effect_active

        target = state.get_entity(BoardEntityID(candidate)) if isinstance(candidate, str) else None
        if not target:
            return False

        if isinstance(target, Token) and target.is_immune_to_enemy_actions:
            actor_id = state.current_actor_id
            owner_id = target.owner_id
            actor = state.get_entity(BoardEntityID(str(actor_id))) if actor_id else None
            owner = state.get_hero(owner_id) if owner_id else None
            actor_team = getattr(actor, "team", None)
            owner_team = getattr(owner, "team", None)
            if actor_team is not None and owner_team is not None and actor_team != owner_team:
                return False

        # Check 1: Standard minion immunity (Heavy with support)
        if isinstance(target, Unit) and rules.is_immune(target, state):
            return False  # Immune = fails filter

        # Check 2: ATTACK_IMMUNITY effects
        # Only applies when current action is ATTACK
        current_action = context.get("current_action_type")
        if current_action == ActionType.ATTACK:
            current_actor_id = str(state.current_actor_id) if state.current_actor_id else None

            # Look for ATTACK_IMMUNITY effects where target is the protected unit
            for effect in state.active_effects:
                if effect.effect_type != EffectType.ATTACK_IMMUNITY:
                    continue
                # Resolving a card sets is_active even for NEXT_TURN effects.
                # The effect's duration must also include the current turn.
                if not effect.is_active or not _is_effect_active(effect, state):
                    continue

                # The effect protects its source_id (the hero who played the defense card)
                if effect.source_id != candidate:
                    continue

                # Some immunities protect only against basic (Gold/Silver)
                # attacks. AttackSequenceStep writes this source-card
                # classification for every attack, including nested performed
                # card actions, so a missing/false flag is non-basic here.
                if effect.basic_attacks_only and not context.get("attack_is_basic", False):
                    continue
                if effect.non_basic_attacks_only and context.get("attack_is_basic", False):
                    continue

                # Check if current attacker is in the exception list
                if current_actor_id and current_actor_id in effect.except_attacker_ids:
                    continue  # This attacker is allowed to target

                # Target is immune to this attack
                return False

        return True  # Passes filter (not immune)


class UnitOnSpawnPointFilter(FilterCondition):
    """
    Filters unit candidates to those occupying a hex with a spawn point.
    """

    type: FilterType = FilterType.UNIT_ON_SPAWN_POINT

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if isinstance(candidate, str):
            loc = state.entity_locations.get(BoardEntityID(candidate))
            if not loc:
                return False
            tile = state.board.get_tile(loc)
            if not tile:
                return False
            return tile.spawn_point is not None
        return False


class AdjacencyToContextFilter(FilterCondition):
    """
    Selects units adjacent to the entity ID stored in a context variable.
    """

    type: FilterType = FilterType.ADJACENCY_TO_CONTEXT
    target_key: str

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        target_id = context.get(self.target_key)

        if not target_id:
            return False

        target_hex = state.entity_locations.get(BoardEntityID(target_id))

        if not target_hex:
            return False

        cand_hex = None

        if isinstance(candidate, Hex):
            cand_hex = candidate
        elif isinstance(candidate, str):
            cand_hex = state.entity_locations.get(BoardEntityID(candidate))

        if not cand_hex:
            return False

        # "Check via tile": Ensure both are valid board positions
        if not state.board.is_on_map(target_hex) or not state.board.is_on_map(cand_hex):
            return False
        # Use topology-aware adjacency (respects reality splits)
        topology = get_topology_service()
        return topology.are_adjacent(cand_hex, target_hex, state)


class ExcludeIdentityFilter(FilterCondition):
    """
    Excludes specific unit IDs or hexes from selection.
    Can exclude self and/or values found in context keys.
    Works for both unit ID (str) and Hex candidates.
    """

    type: FilterType = FilterType.EXCLUDE_IDENTITY
    exclude_self: bool = True
    exclude_keys: list[str] = Field(default_factory=list)

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if self.exclude_self and isinstance(candidate, str):
            if candidate == state.current_actor_id:
                return False
            # The acting piece of a multi-piece hero is "you" too; her other
            # pieces stay selectable as distinct hero units.
            if (
                state.acting_piece_id
                and candidate == str(state.acting_piece_id)
                and state.hero_owner_id(candidate) == str(state.current_actor_id)
            ):
                return False
        for key in self.exclude_keys:
            val = context.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                if candidate in val:
                    return False
            elif val == candidate:
                return False
        return True


class HasTurnStartPositionFilter(FilterCondition):
    """
    Passes units that were on the board at the turn boundary, whether or not
    they have moved since. Units that spawned or respawned after the boundary
    have no start-of-turn space to be sent back to.
    """

    type: FilterType = FilterType.HAS_TURN_START_POSITION

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, str):
            return False

        return BoardEntityID(candidate) in state.last_turn_positions


class RemainedInPlaceFilter(FilterCondition):
    """
    Passes units standing on the exact hex they occupied at the turn boundary.

    "Remained in the same space since the last turn" is a literal comparison
    against ``state.last_turn_positions``: leaving and returning within the
    turn still counts as remaining, and being displaced by someone else counts
    as having moved. A unit with no snapshot entry (it spawned or respawned
    after the boundary) never passes.
    """

    type: FilterType = FilterType.REMAINED_IN_PLACE

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, str):
            return False

        snapshot_hex = state.last_turn_positions.get(BoardEntityID(candidate))
        if snapshot_hex is None:
            return False

        return state.get_position(candidate) == snapshot_hex


class CanBePlacedByActorFilter(FilterCondition):
    """
    Filters out units that cannot be placed by the current actor.
    Delegates to ValidationService for actual logic.
    """

    type: FilterType = FilterType.CAN_BE_PLACED_BY_ACTOR

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, str):
            return False

        actor_id = state.current_actor_id
        if not actor_id:
            return True  # No actor context, allow selection

        result = state.validator.can_be_placed(
            state=state, unit_id=candidate, actor_id=actor_id, context=context
        )

        return result.allowed


class CanBeMovedByActorFilter(FilterCondition):
    """
    Filters out units the current actor cannot force into a walked move.

    The placement variant above asks about teleports/placements. Protection
    effects distinguish the two (Wasp blocks PLACE and SWAP but not MOVE), so
    a step that ends in MoveUnitStep must screen its targets with this filter
    or it will offer heroes it cannot move, and refuse heroes it could.
    """

    type: FilterType = FilterType.CAN_BE_MOVED_BY_ACTOR

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, str):
            return False

        actor_id = state.current_actor_id
        if not actor_id:
            return True  # No actor context, allow selection

        result = state.validator.can_be_moved(
            state=state, unit_id=candidate, actor_id=actor_id, context=context
        )

        return result.allowed


class HasMarkerFilter(FilterCondition):
    """
    Filters unit candidates to those that currently hold a specific marker.
    Non-hero candidates are rejected.
    """

    type: FilterType = FilterType.HAS_MARKER
    marker_type: MarkerType

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        if not isinstance(candidate, str):
            return False
        marker = state.markers.get(self.marker_type)
        if marker is None or not marker.is_placed:
            return False
        return marker.target_id == candidate


class AdjacentToTokenFilter(FilterCondition):
    """
    Passes only if the candidate is adjacent to a token of `token_type`
    which is itself within `max_range` of the actor.
    """

    type: FilterType = FilterType.ADJACENT_TO_TOKEN_IN_RANGE
    token_type: TokenType = TokenType.ILLUSION
    max_range: int

    def apply(self, candidate: Any, state: GameState, context: dict) -> bool:
        cand_hex = None
        if isinstance(candidate, Hex):
            cand_hex = candidate
        elif isinstance(candidate, str):
            cand_hex = state.entity_locations.get(BoardEntityID(candidate))

        if not cand_hex:
            return False

        actor_id = state.current_actor_id
        if not actor_id:
            return False
        actor_hex = state.entity_locations.get(BoardEntityID(str(actor_id)))
        if not actor_hex:
            return False

        topology = get_topology_service()
        # Find all neighbor hexes of the candidate (connected to respect splits)
        neighbors = topology.get_connected_neighbors(cand_hex, state)

        for n in neighbors:
            tile = state.board.get_tile(n)
            if not tile or not tile.occupant_id:
                continue

            # Check if occupant is a token of the correct type
            occupant = state.get_entity(tile.occupant_id)
            if (
                not occupant
                or not isinstance(occupant, Token)
                or occupant.token_type != self.token_type
            ):
                continue

            # Check if this token is within range of the actor (respecting topology)
            dist = topology.distance(actor_hex, n, state)
            if dist <= self.max_range:
                return True

        return False
