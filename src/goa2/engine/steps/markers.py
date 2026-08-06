"""Token and marker placement/removal steps."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field, field_serializer, field_validator

from goa2.domain.events import GameEvent, GameEventType, _hex_dict
from goa2.domain.hex import Hex
from goa2.domain.input import InputOption, InputRequestType, create_input_request
from goa2.domain.models import RuneType, StepType, TargetType, Token, TokenType, Turret
from goa2.domain.models.marker import MarkerType
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID, HeroID, UnitID
from goa2.engine import rules
from goa2.engine.filters_base import (
    FilterCondition,
    dump_filter_grid,
    revalidate_filter_grid,
    revalidate_filters,
)
from goa2.engine.filters_units import TokenTypeFilter, UnitTypeFilter
from goa2.engine.steps.base import GameStep, StepResult

logger = logging.getLogger(__name__)

TURRET_ID = BoardEntityID("trinkets_turret")


def _remove_token_from_board(state: GameState, token_id: str) -> tuple[Hex | None, int]:
    from_hex = state.entity_locations.get(BoardEntityID(token_id))
    if not from_hex:
        return None, 0

    state.remove_entity(BoardEntityID(token_id))

    # The token's own effects end here — this is the only thing that ends a
    # token-bound effect (EffectManager._is_token_bound).
    initial_count = len(state.active_effects)
    state.active_effects = [
        e
        for e in state.active_effects
        if e.token_id != token_id and e.source_id != token_id and e.scope.origin_id != token_id
    ]
    removed_effects = initial_count - len(state.active_effects)

    if removed_effects > 0:
        logger.debug(f"   [TOKEN] Removed {removed_effects} linked effect(s) from token {token_id}")

    return from_hex, removed_effects


TOKEN_TYPE_OVERRIDE_KEY = "token_type_override"
SKIP_MARKERS_KEY = "skip_markers"


def effective_token_type(context: dict[str, Any], default: TokenType) -> TokenType:
    """Token-type override for copied actions (NebKher's Mind Grip: "if you
    would place any tokens this way, place Illusion tokens instead").
    PerformCardActionStep sets the override flag around the copied steps; every
    token-placing step resolves its type through this helper so the
    substitution reaches nested templates and runtime-built steps."""
    override = context.get(TOKEN_TYPE_OVERRIDE_KEY)
    return TokenType(str(override)) if override else default


class RemoveTokenStep(GameStep):
    type: StepType = StepType.REMOVE_TOKEN
    token_id: str | None = None
    token_key: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        target_id = self.token_id
        if self.token_key:
            target_id = context.get(self.token_key)
        if not target_id:
            return StepResult(is_finished=True)

        token = state.misc_entities.get(BoardEntityID(target_id))
        if not isinstance(token, Token):
            return StepResult(is_finished=True)

        from_hex, removed_effects = _remove_token_from_board(state, target_id)
        if not from_hex:
            return StepResult(is_finished=True)

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.TOKEN_REMOVED,
                    actor_id=str(state.current_actor_id) if state.current_actor_id else None,
                    target_id=target_id,
                    from_hex=_hex_dict(from_hex),
                    metadata={"effects_removed": removed_effects},
                )
            ],
        )


class PlaceTokenStep(GameStep):
    type: StepType = StepType.PLACE_TOKEN
    token_type: TokenType
    hex_key: str = "target_hex"
    owner_id_key: str | None = None
    output_key: str | None = None
    existing_token_key: str | None = None
    overflow_selection_key: str = "overflow_token_to_remove"
    is_immune_to_enemy_actions: bool = False

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.selection import SelectStep

        token_type = effective_token_type(context, self.token_type)
        dest_val = context.get(self.hex_key)
        if not dest_val or dest_val == "SKIP":
            return StepResult(is_finished=True)

        dest_hex = Hex(**dest_val) if isinstance(dest_val, dict) else dest_val
        tile = state.board.get_tile(dest_hex)
        if tile and tile.is_occupied:
            logger.debug(f"   [TOKEN] Cannot place {token_type.value} at {dest_hex}: occupied.")
            return StepResult(is_finished=True)

        owner_id = context.get(self.owner_id_key) if self.owner_id_key else None
        if owner_id is None:
            owner_id = state.current_actor_id

        # Some effects re-place a particular token that is already on the
        # board. Relocate that entity directly so effects whose source/origin
        # is the token remain attached. The ordinary supply-overflow path
        # removes a token first, which intentionally removes linked effects.
        existing_token_id = (
            context.get(self.existing_token_key) if self.existing_token_key else None
        )
        if existing_token_id:
            existing = state.misc_entities.get(BoardEntityID(str(existing_token_id)))
            from_hex = state.entity_locations.get(BoardEntityID(str(existing_token_id)))
            if (
                not isinstance(existing, Token)
                or existing.token_type != token_type
                or from_hex is None
            ):
                return StepResult(is_finished=True)
            if from_hex == dest_hex:
                if self.output_key:
                    context[self.output_key] = str(existing.id)
                return StepResult(is_finished=True)

            state.place_entity(BoardEntityID(str(existing.id)), dest_hex)
            if self.output_key:
                context[self.output_key] = str(existing.id)
            return StepResult(
                is_finished=True,
                events=[
                    GameEvent(
                        event_type=GameEventType.TOKEN_MOVED,
                        actor_id=str(owner_id) if owner_id else None,
                        target_id=str(existing.id),
                        from_hex=_hex_dict(from_hex),
                        to_hex=_hex_dict(dest_hex),
                        metadata={"token_type": token_type.value, "replaced": True},
                    )
                ],
            )

        pool = state.token_pool.get(token_type, [])
        available = next(
            (t for t in pool if BoardEntityID(str(t.id)) not in state.entity_locations),
            None,
        )

        events: list[GameEvent] = []

        if available is None:
            placed = [t for t in pool if BoardEntityID(str(t.id)) in state.entity_locations]
            if not placed:
                return StepResult(is_finished=True)
            return StepResult(
                is_finished=True,
                new_steps=[
                    SelectStep(
                        target_type=TargetType.UNIT_OR_TOKEN,
                        prompt=f"Select a {token_type.value} token to remove from the board.",
                        output_key=self.overflow_selection_key,
                        skip_immunity_filter=True,
                        skip_self_filter=True,
                        is_mandatory=True,
                        filters=[
                            UnitTypeFilter(unit_type="TOKEN"),
                            TokenTypeFilter(token_type=token_type),
                        ],
                        override_player_id_key=self.owner_id_key,
                    ),
                    RemoveTokenStep(token_key=self.overflow_selection_key),
                    PlaceTokenStep(
                        token_type=token_type,
                        hex_key=self.hex_key,
                        owner_id_key=self.owner_id_key,
                        output_key=self.output_key,
                        existing_token_key=self.existing_token_key,
                        overflow_selection_key=self.overflow_selection_key,
                        is_immune_to_enemy_actions=self.is_immune_to_enemy_actions,
                    ),
                ],
            )

        if owner_id is not None:
            available.owner_id = HeroID(str(owner_id))
        # Properties of the token this step was authored for don't carry over
        # to a substituted one (Mind Grip): an Illusion standing in for a Totem
        # is a plain Illusion.
        substituted = token_type is not self.token_type
        available.is_immune_to_enemy_actions = (
            False if substituted else self.is_immune_to_enemy_actions
        )

        state.place_entity(BoardEntityID(str(available.id)), dest_hex)
        if self.output_key:
            context[self.output_key] = str(available.id)

        events.append(
            GameEvent(
                event_type=GameEventType.TOKEN_PLACED,
                actor_id=str(owner_id) if owner_id else None,
                target_id=str(available.id),
                to_hex=_hex_dict(dest_hex),
                metadata={"token_type": token_type.value},
            )
        )
        return StepResult(is_finished=True, events=events)


def _token_shortfall_removal_steps(
    state: GameState,
    *,
    token_type: TokenType,
    needed: int,
    key_prefix: str,
    owner_id_key: str | None = None,
) -> list[GameStep]:
    """Prompt removals needed to free enough supply before a multi-token placement."""
    from goa2.engine.steps.selection import SelectStep

    pool = state.token_pool.get(token_type, [])
    on_board_ids = [
        str(t.id) for t in pool if state.entity_locations.get(BoardEntityID(str(t.id))) is not None
    ]
    free = len(pool) - len(on_board_ids)
    removal_count = min(max(0, needed - free), len(on_board_ids))

    steps: list[GameStep] = []
    for i in range(removal_count):
        rm_key = f"{key_prefix}_rm_{i}"
        steps += [
            SelectStep(
                target_type=TargetType.UNIT_OR_TOKEN,
                prompt=f"Select a {token_type.value} token to remove from the board.",
                output_key=rm_key,
                skip_immunity_filter=True,
                skip_self_filter=True,
                is_mandatory=True,
                filters=[
                    UnitTypeFilter(unit_type="TOKEN"),
                    TokenTypeFilter(token_type=token_type),
                ],
                override_player_id_key=owner_id_key,
            ),
            RemoveTokenStep(token_key=rm_key),
        ]
    return steps


class PlaceTokenBatchStep(GameStep):
    """Places ``count`` tokens of one type as a single batch with upfront
    supply reconciliation.

    1. ``count`` greater than the TOTAL supply → fail (abort when mandatory).
    2. All-or-nothing feasibility precheck: there must exist ``count`` hexes
       passing each slot's filters, counting empty hexes plus hexes the
       forced removals of step 4 could free — at most ``count - free_supply``
       of those. Infeasible → skip silently — with an opt-in prompt
       configured it is simply never shown ("option unavailable") — or abort
       when mandatory.
    3. Optional opt-in prompt ("You may place …"); declining skips the batch.
    4. If the free supply is short, the owner removes on-board tokens of the
       type until ``count`` are free. Removal happens BEFORE any placement,
       so only pre-existing tokens are ever removed — never batch members.
       Each removal prompt offers only tokens from which the remaining
       removals and placements can still complete
       (``TokenRemovalCompletableFilter``), so no removal choice can
       dead-end the batch.
    5. One hex-selection + PlaceTokenStep pair per token; each selection
       offers only choices from which the remaining slots can still complete.
    6. ``placed_flag_key`` is set only when the batch goes through, for
       downstream "if you do" riders (gate with ``active_if_key``).

    Assumptions — revisit the design if either stops holding:

    - **Removal monotonicity.** Slot filters must never REQUIRE the presence
      of on-board tokens of this type (e.g. "adjacent to an existing X
      token"); removing a token may only ever ADD legal placement hexes. The
      precheck and removal steering treat token hexes as empty-with-budget,
      which is unsound for such filters. If one is ever introduced, replace
      the budgeted search with subset simulation: enumerate the C(M, R)
      removal subsets, apply each to a deep-copied state through the real
      removal helper (so linked active_effects are stripped too), and run
      the assignment search per subset. Removals commute, so subsets — not
      orderings — suffice.
    - **``count`` <= 4.** The completability search is
      O(candidates ** count) per evaluation. For larger batches, memoize on
      the chosen-hex set, or — when all slots share identical filters with
      no cross-slot constraints — replace the search with a simple count of
      legal hexes.
    """

    type: StepType = StepType.PLACE_TOKEN_BATCH
    token_type: TokenType
    count: int
    slot_filters: list[list[FilterCondition]] = Field(default_factory=list)
    opt_in_prompt: str | None = None
    owner_id: str | None = None
    placed_flag_key: str | None = None
    key_prefix: str = "tkb"

    @field_serializer("slot_filters", mode="plain")
    def _serialize_slot_filters(self, value: list[list[Any]]) -> list[list[Any]]:
        return dump_filter_grid(value)

    @field_validator("slot_filters", mode="before")
    @classmethod
    def _deserialize_slot_filters(cls, value: list[list[Any]]) -> list[list[FilterCondition]]:
        return revalidate_filter_grid(value)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.filters_hex import (
            HexBatchCompletableFilter,
            TokenRemovalCompletableFilter,
        )
        from goa2.engine.steps.selection import SelectStep
        from goa2.engine.steps.utility import CheckContextConditionStep, SetContextFlagStep

        if self.should_skip(context):
            return StepResult(is_finished=True)

        token_type = effective_token_type(context, self.token_type)
        pool = state.token_pool.get(token_type, [])
        if self.count > len(pool):
            logger.debug(
                f"   [TOKEN BATCH] {self.count} {token_type.value} exceeds supply "
                f"({len(pool)})."
            )
            return StepResult(is_finished=True, abort_action=self.is_mandatory)

        slot_keys = [f"{self.key_prefix}_hex_{i}" for i in range(self.count)]
        normalized_slot_filters = self._normalized_slot_filters()
        if normalized_slot_filters is None:
            logger.debug(
                f"   [TOKEN BATCH] Invalid slot filter count for {self.count} "
                f"{token_type.value} tokens."
            )
            return StepResult(is_finished=True, abort_action=self.is_mandatory)

        on_board_hexes = [
            pos
            for t in pool
            if (pos := state.entity_locations.get(BoardEntityID(str(t.id)))) is not None
        ]
        free = len(pool) - len(on_board_hexes)
        removal_count = max(0, self.count - free)

        if not self._is_feasible(
            state, context, slot_keys, normalized_slot_filters, on_board_hexes, removal_count
        ):
            logger.debug(
                f"   [TOKEN BATCH] No placement of {self.count} {token_type.value} "
                "tokens fits the slot filters."
            )
            return StepResult(is_finished=True, abort_action=self.is_mandatory)

        owner_key = f"{self.key_prefix}_owner"
        context[owner_key] = str(self.owner_id or state.current_actor_id or "")

        gate_key: str | None = None
        new_steps: list[GameStep] = []
        if self.opt_in_prompt:
            choice_key = f"{self.key_prefix}_optin"
            gate_key = f"{self.key_prefix}_go"
            new_steps += [
                SelectStep(
                    target_type=TargetType.NUMBER,
                    prompt=self.opt_in_prompt,
                    output_key=choice_key,
                    number_options=[1, 0],
                    number_labels={1: "Place tokens", 0: "Skip"},
                    override_player_id_key=owner_key,
                    is_mandatory=True,
                ),
                CheckContextConditionStep(
                    input_key=choice_key, operator="==", threshold=1, output_key=gate_key
                ),
            ]

        for j in range(removal_count):
            rm_key = f"{self.key_prefix}_rm_{j}"
            new_steps += [
                SelectStep(
                    target_type=TargetType.UNIT_OR_TOKEN,
                    prompt=f"Select a {token_type.value} token to remove from the board.",
                    output_key=rm_key,
                    skip_immunity_filter=True,
                    skip_self_filter=True,
                    is_mandatory=True,
                    filters=[
                        UnitTypeFilter(unit_type="TOKEN"),
                        TokenTypeFilter(token_type=token_type),
                        TokenRemovalCompletableFilter(
                            token_type=token_type,
                            slot_keys=slot_keys,
                            slot_filters=normalized_slot_filters,
                            remaining_removals=removal_count - j,
                        ),
                    ],
                    override_player_id_key=owner_key,
                    active_if_key=gate_key,
                ),
                RemoveTokenStep(token_key=rm_key, active_if_key=gate_key),
            ]

        for i in range(self.count):
            hex_key = slot_keys[i]
            filters: list[FilterCondition] = [
                *normalized_slot_filters[i],
                HexBatchCompletableFilter(
                    slot_index=i,
                    slot_keys=slot_keys,
                    slot_filters=normalized_slot_filters,
                ),
            ]
            new_steps += [
                SelectStep(
                    target_type=TargetType.HEX,
                    prompt=f"Place {token_type.value} token {i + 1}/{self.count}",
                    output_key=hex_key,
                    filters=filters,
                    is_mandatory=True,
                    override_player_id_key=owner_key,
                    active_if_key=gate_key,
                ),
                PlaceTokenStep(
                    token_type=token_type,
                    hex_key=hex_key,
                    owner_id_key=owner_key,
                    output_key=f"{self.key_prefix}_token_{i}",
                    active_if_key=gate_key,
                ),
            ]

        if self.placed_flag_key:
            new_steps.append(
                SetContextFlagStep(key=self.placed_flag_key, value=True, active_if_key=gate_key)
            )

        return StepResult(is_finished=True, new_steps=new_steps)

    def _normalized_slot_filters(self) -> list[list[FilterCondition]] | None:
        from goa2.engine.filters_hex import ObstacleFilter

        if self.count < 0:
            return None
        raw_slot_filters = self.slot_filters or [[] for _ in range(self.count)]
        if len(raw_slot_filters) != self.count:
            return None
        # Revalidate: after a save/load the entries may be degraded base
        # FilterCondition instances (see filters_base.revalidate_filter).
        return [
            [*revalidate_filters(filters), ObstacleFilter(is_obstacle=False)]
            for filters in raw_slot_filters
        ]

    def _is_feasible(
        self,
        state: GameState,
        context: dict[str, Any],
        slot_keys: list[str],
        slot_filters: list[list[FilterCondition]],
        removable_hexes: list[Hex],
        removal_budget: int,
    ) -> bool:
        """Does at least one full slot assignment exist, counting hexes the
        forced removals could free (at most ``removal_budget`` of them)?"""
        from goa2.engine.filters_hex import HexBatchCompletableFilter

        if self.count == 0:
            return True
        return HexBatchCompletableFilter(
            slot_index=0,
            slot_keys=slot_keys,
            slot_filters=slot_filters,
            removable_hexes=removable_hexes,
            removal_budget=removal_budget,
        ).feasible(state, context)


class PlaceTokensInLineStep(GameStep):
    """Places up to ``count`` tokens in the first empty hexes along a straight
    line from the origin toward a reference hex.

    Scans indefinitely past obstacles (which are skipped, not landed on),
    stopping only at the board edge. Used by Fissure: "Place a Rock token in
    each of the first three empty spaces in the straight line from you in the
    direction of the attack."

    When free supply is short, removals are prompted before any placement and
    this step is re-run afterward. Re-running is intentional: a removed token
    may itself have occupied one of the line hexes, so the "first empty spaces"
    must be computed against the post-removal board.
    """

    type: StepType = StepType.PLACE_TOKENS_IN_LINE
    token_type: TokenType
    origin_id: str | None = None
    origin_key: str | None = None
    direction_ref_key: str = "line_direction_ref"
    count: int = 3
    owner_id_key: str | None = None
    output_key: str | None = None  # stores the list of target hex dicts that get a token

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        token_type = effective_token_type(context, self.token_type)
        origin_uid = self.origin_id
        if not origin_uid and self.origin_key:
            origin_uid = context.get(self.origin_key)
        if not origin_uid:
            origin_uid = state.current_actor_id
        origin_hex = (
            state.entity_locations.get(BoardEntityID(str(origin_uid))) if origin_uid else None
        )

        ref = context.get(self.direction_ref_key)
        if origin_hex is None or ref is None:
            return StepResult(is_finished=True)
        ref_hex = Hex(**ref) if isinstance(ref, dict) else ref
        if origin_hex == ref_hex or not origin_hex.is_straight_line(ref_hex):
            return StepResult(is_finished=True)

        dist = origin_hex.distance(ref_hex)
        dq = (ref_hex.q - origin_hex.q) // dist
        dr = (ref_hex.r - origin_hex.r) // dist
        ds = (ref_hex.s - origin_hex.s) // dist

        target_hexes: list[Hex] = []
        current = origin_hex
        for _ in range(50):  # safety bound; the board edge normally stops the scan
            current = Hex(q=current.q + dq, r=current.r + dr, s=current.s + ds)
            if not state.board.is_on_map(current):
                break  # board edge
            if not state.board.get_tile(current).is_obstacle:
                target_hexes.append(current)
                if len(target_hexes) >= self.count:
                    break

        removal_steps = _token_shortfall_removal_steps(
            state,
            token_type=token_type,
            needed=len(target_hexes),
            key_prefix="_line_token",
            owner_id_key=self.owner_id_key,
        )
        if removal_steps:
            return StepResult(
                is_finished=True,
                new_steps=[*removal_steps, self.model_copy(deep=True)],
            )

        new_steps: list[GameStep] = []
        for i, h in enumerate(target_hexes):
            key = f"_line_rock_hex_{i}"
            context[key] = h.model_dump()
            new_steps.append(
                PlaceTokenStep(token_type=token_type, hex_key=key, owner_id_key=self.owner_id_key)
            )
        if self.output_key is not None:
            context[self.output_key] = [h.model_dump() for h in target_hexes]
        return StepResult(is_finished=True, new_steps=new_steps)


class PlaceTokenTrailStep(GameStep):
    """Places a token in every empty hex along the straight line from
    ``origin_hex_key`` toward ``dest_key``, INCLUDING the origin ("moved out of")
    and EXCLUDING the destination ("moved to").

    Used by Ignatia's Path cards: after moving up to N in a straight line, drop a
    Magma token in each empty space moved through or out of. If origin == dest
    (moved zero) or the two are not in a straight line, nothing is placed.

    When free supply is short, removals are prompted before any placement, then
    the trail placement resumes. This keeps newly placed trail tokens from being
    offered as overflow removals for later tokens in the same trail.
    """

    type: StepType = StepType.PLACE_TOKEN_TRAIL
    token_type: TokenType
    origin_hex_key: str
    dest_key: str
    owner_id_key: str | None = None
    excluded_hexes_key: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        token_type = effective_token_type(context, self.token_type)
        raw_origin = context.get(self.origin_hex_key)
        raw_dest = context.get(self.dest_key)
        if not raw_origin or not raw_dest:
            return StepResult(is_finished=True)

        origin = Hex(**raw_origin) if isinstance(raw_origin, dict) else raw_origin
        dest = Hex(**raw_dest) if isinstance(raw_dest, dict) else raw_dest
        if origin == dest or not origin.is_straight_line(dest):
            return StepResult(is_finished=True)

        dist = origin.distance(dest)
        dq = (dest.q - origin.q) // dist
        dr = (dest.r - origin.r) // dist
        ds = (dest.s - origin.s) // dist

        target_hexes: list[Hex] = []
        excluded_key = self.excluded_hexes_key
        excluded_values = context.get(excluded_key, []) if excluded_key is not None else []
        excluded_hexes = {
            Hex(**value) if isinstance(value, dict) else value for value in excluded_values
        }
        current = origin
        for _ in range(dist):  # origin, origin+1, ..., dest-1 (destination excluded)
            if current not in excluded_hexes:
                target_hexes.append(current)
            current = Hex(q=current.q + dq, r=current.r + dr, s=current.s + ds)

        removal_steps = _token_shortfall_removal_steps(
            state,
            token_type=token_type,
            needed=len(target_hexes),
            key_prefix="_trail_token",
            owner_id_key=self.owner_id_key,
        )
        if removal_steps:
            return StepResult(
                is_finished=True,
                new_steps=[*removal_steps, self.model_copy(deep=True)],
            )

        new_steps: list[GameStep] = []
        for i, current in enumerate(target_hexes):
            key = f"_trail_hex_{i}"
            context[key] = current.model_dump()
            new_steps.append(
                PlaceTokenStep(token_type=token_type, hex_key=key, owner_id_key=self.owner_id_key)
            )
        return StepResult(is_finished=True, new_steps=new_steps)


class OfferRockUltimateStep(GameStep):
    """Mrak's ultimate "Rock and a Hard Place" (active at level >= 8).

    After a placement batch, each enemy hero adjacent to a rock placed in that
    batch discards a card (if able). Optional and once per turn — the actor is
    prompted, and a per-turn flag stops further offers once used. Offered per
    placement group so Ground Shaker can save its single use for the repeat.

    Batch hexes come from ``rock_hex_keys`` (fixed context keys, each a hex) and
    ``rock_hexes_key`` (a key holding a list of hexes).
    """

    type: StepType = StepType.OFFER_ROCK_ULTIMATE
    rock_hex_keys: list[str] = Field(default_factory=list)
    rock_hexes_key: str | None = None
    used_flag_key: str = "rock_ultimate_used"

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        actor_id = state.current_actor_id
        actor = state.get_hero(actor_id) if actor_id else None
        # Gate: ultimate unlocked (level >= 8), has an ultimate card, not yet used.
        if not actor or actor.level < 8 or not actor.ultimate_card:
            return StepResult(is_finished=True)
        if context.get(self.used_flag_key):
            return StepResult(is_finished=True)

        # Gather the batch's rock hexes.
        hexes: list[Hex] = []
        for key in self.rock_hex_keys:
            v = context.get(key)
            if v:
                hexes.append(Hex(**v) if isinstance(v, dict) else v)
        if self.rock_hexes_key:
            for v in context.get(self.rock_hexes_key) or []:
                hexes.append(Hex(**v) if isinstance(v, dict) else v)
        if not hexes:
            return StepResult(is_finished=True)

        # Enemy heroes adjacent to any batch hex (each affected once).
        from goa2.engine.topology import get_topology_service

        topology = get_topology_service()
        affected: list[str] = []
        for team_color, team in state.teams.items():
            if team_color == actor.team:
                continue
            for enemy_hero in team.heroes:
                for enemy_id in state.get_piece_ids(str(enemy_hero.id)):
                    hloc = state.get_position(enemy_id)
                    if hloc is None:
                        continue
                    if any(topology.distance(hloc, rh, state) == 1 for rh in hexes):
                        unit = state.get_unit(UnitID(enemy_id))
                        if unit is None or rules.is_immune_to_actor(
                            unit, state, actor_id=str(actor.id)
                        ):
                            continue
                        affected.append(enemy_id)

        if not affected:
            return StepResult(is_finished=True)

        context["_rock_ult_affected"] = affected

        from goa2.engine.steps.cards import ForceDiscardStep
        from goa2.engine.steps.selection import SelectStep
        from goa2.engine.steps.utility import (
            CheckContextConditionStep,
            ForEachStep,
            SetContextFlagStep,
        )

        return StepResult(
            is_finished=True,
            new_steps=[
                SelectStep(
                    target_type=TargetType.NUMBER,
                    prompt="Rock and a Hard Place: each enemy hero adjacent to a "
                    "placed Rock discards a card. Apply?",
                    output_key="rock_ult_choice",
                    number_options=[1, 0],
                    number_labels={1: "Apply", 0: "Skip"},
                ),
                CheckContextConditionStep(
                    input_key="rock_ult_choice",
                    operator="==",
                    threshold=1,
                    output_key="rock_ult_yes",
                ),
                SetContextFlagStep(
                    key=self.used_flag_key, value=True, active_if_key="rock_ult_yes"
                ),
                ForEachStep(
                    list_key="_rock_ult_affected",
                    item_key="_rock_ult_victim",
                    steps_template=[ForceDiscardStep(victim_key="_rock_ult_victim")],
                    active_if_key="rock_ult_yes",
                ),
            ],
        )


class PlaceTurretStep(GameStep):
    type: StepType = StepType.PLACE_TURRET
    hex_key: str = "target_hex"
    owner_id: str | None = None
    owner_id_key: str | None = None
    output_key: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        dest_val = context.get(self.hex_key)
        if not dest_val:
            return StepResult(is_finished=True, abort_action=self.is_mandatory)

        dest_hex = Hex(**dest_val) if isinstance(dest_val, dict) else dest_val
        target_tile = state.board.get_tile(dest_hex)
        old_hex = state.entity_locations.get(TURRET_ID)

        if target_tile and target_tile.is_occupied and target_tile.occupant_id != TURRET_ID:
            logger.debug("   [TURRET] Cannot place Turret at %s: occupied.", dest_hex)
            return StepResult(is_finished=True, abort_action=self.is_mandatory)

        owner_id: str | None = self.owner_id
        if self.owner_id_key:
            owner_id = context.get(self.owner_id_key, owner_id)
        if owner_id is None and state.current_actor_id is not None:
            owner_id = str(state.current_actor_id)

        turret = state.misc_entities.get(TURRET_ID)
        if not isinstance(turret, Turret):
            turret = Turret(id=TURRET_ID, name="Turret", owner_id=owner_id)
            state.register_entity(turret, "board_entity")
        else:
            turret.owner_id = owner_id

        state.place_entity(TURRET_ID, dest_hex)

        if self.output_key:
            context[self.output_key] = str(TURRET_ID)

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.BOARD_ENTITY_PLACED,
                    actor_id=str(owner_id) if owner_id else None,
                    target_id=str(TURRET_ID),
                    from_hex=_hex_dict(old_hex),
                    to_hex=_hex_dict(dest_hex),
                    metadata={"entity_kind": turret.entity_kind, "is_obstacle": turret.is_obstacle},
                )
            ],
        )


class RemoveTurretStep(GameStep):
    type: StepType = StepType.REMOVE_TURRET
    output_key: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        turret = state.misc_entities.get(TURRET_ID)
        if not isinstance(turret, Turret):
            return StepResult(is_finished=True, abort_action=self.is_mandatory)

        from_hex = state.entity_locations.get(TURRET_ID)
        if not from_hex:
            return StepResult(is_finished=True, abort_action=self.is_mandatory)

        state.remove_entity(TURRET_ID)
        if self.output_key:
            context[self.output_key] = str(TURRET_ID)

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.BOARD_ENTITY_REMOVED,
                    actor_id=str(state.current_actor_id) if state.current_actor_id else None,
                    target_id=str(TURRET_ID),
                    from_hex=_hex_dict(from_hex),
                    metadata={"entity_kind": "turret"},
                )
            ],
        )


class MoveTokenStep(GameStep):
    type: StepType = StepType.MOVE_TOKEN
    token_key: str
    destination_key: str = "target_hex"
    range_val: int = 1
    pass_through_obstacles: bool = False
    force_straight_line: bool = False

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        token_id = context.get(self.token_key)
        dest_val = context.get(self.destination_key)
        if not token_id or not dest_val:
            return StepResult(is_finished=True)

        token = state.misc_entities.get(BoardEntityID(str(token_id)))
        if not isinstance(token, Token):
            return StepResult(is_finished=True)

        dest_hex = Hex(**dest_val) if isinstance(dest_val, dict) else dest_val
        from_hex = state.entity_locations.get(BoardEntityID(str(token_id)))
        if not from_hex:
            return StepResult(is_finished=True)

        tile = state.board.get_tile(dest_hex)
        if tile and tile.is_occupied and str(tile.occupant_id) != str(token_id):
            logger.debug(f"   [TOKEN] Cannot move {token_id} to {dest_hex}: occupied.")
            return StepResult(is_finished=True)

        if from_hex != dest_hex:
            is_valid = rules.validate_movement_path(
                board=state.board,
                start=from_hex,
                end=dest_hex,
                max_steps=self.range_val,
                state=state,
                actor_id=str(state.current_actor_id) if state.current_actor_id else None,
                pass_through_obstacles=self.pass_through_obstacles,
            )
            if not is_valid:
                logger.debug(f"   [TOKEN] Invalid token move {token_id}: blocked or out of range.")
                return StepResult(is_finished=True)

            if self.force_straight_line:
                from goa2.engine.filters_geometry import StraightLinePathFilter

                if not StraightLinePathFilter(
                    origin_id=str(token_id),
                    pass_through_obstacles=self.pass_through_obstacles,
                ).apply(dest_hex, state, context):
                    logger.debug(f"   [TOKEN] Invalid token move {token_id}: not a clear line.")
                    return StepResult(is_finished=True)
        else:
            return StepResult(is_finished=True)

        state.place_entity(BoardEntityID(str(token_id)), dest_hex)
        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.TOKEN_MOVED,
                    actor_id=str(state.current_actor_id) if state.current_actor_id else None,
                    target_id=str(token_id),
                    from_hex=_hex_dict(from_hex),
                    to_hex=_hex_dict(dest_hex),
                    metadata={"range": self.range_val},
                )
            ],
        )


class PlaceMarkerStep(GameStep):
    """
    Places a marker on a target hero.

    Markers are singletons - placing on a new target automatically removes
    from the previous target. The marker's effects are applied via
    get_computed_stat() which reads markers directly.

    Usage:
        PlaceMarkerStep(
            marker_type=MarkerType.VENOM,
            target_key="victim_id",
            value=-1,
        )
    """

    type: StepType = StepType.PLACE_MARKER
    marker_type: MarkerType
    target_id: str | None = None  # Direct target ID
    target_key: str | None = None  # Context key for target ID
    value: int = 0  # Effect magnitude (e.g., -1 or -2 for Venom)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.effects import CheckPassiveAbilitiesStep

        if self.should_skip(context):
            return StepResult(is_finished=True)

        # Copied actions may skip marker giving entirely (NebKher's Mind
        # Grip: "skip giving markers") — the rest of the effect continues.
        if context.get(SKIP_MARKERS_KEY):
            logger.debug(f"   [MARKER] Skipping {self.marker_type.value} (skip_markers active).")
            return StepResult(is_finished=True)

        # Resolve target
        target = self.target_id
        if not target and self.target_key:
            target = context.get(self.target_key)

        if not target:
            logger.debug(f"   [SKIP] No target for PlaceMarkerStep ({self.marker_type.value})")
            return StepResult(is_finished=True)

        # Get source (current actor)
        source_id = str(state.current_actor_id) if state.current_actor_id else "system"

        # Place the marker
        state.place_marker(
            marker_type=self.marker_type,
            target_id=target,
            value=self.value,
            source_id=source_id,
        )

        logger.debug(
            f"   [MARKER] Placed {self.marker_type.value} on {target} "
            f"(value={self.value}, source={source_id})"
        )

        # Fire AFTER_PLACE_MARKER passive trigger
        from goa2.domain.models.enums import PassiveTrigger

        context["marker_target_id"] = target
        post_steps: list[GameStep] = [
            CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_PLACE_MARKER.value)
        ]

        return StepResult(
            is_finished=True,
            new_steps=post_steps,
            events=[
                GameEvent(
                    event_type=GameEventType.MARKER_PLACED,
                    actor_id=source_id,
                    target_id=target,
                    metadata={
                        "marker_type": self.marker_type.value,
                        "value": self.value,
                    },
                )
            ],
        )


class RemoveMarkerStep(GameStep):
    """
    Removes a marker, returning it to supply.

    Usage:
        RemoveMarkerStep(marker_type=MarkerType.VENOM)
    """

    type: StepType = StepType.REMOVE_MARKER
    marker_type: MarkerType

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        marker = state.remove_marker(self.marker_type)

        if marker:
            logger.debug(f"   [MARKER] Removed {self.marker_type.value} from play")
            return StepResult(
                is_finished=True,
                events=[
                    GameEvent(
                        event_type=GameEventType.MARKER_REMOVED,
                        metadata={"marker_type": self.marker_type.value},
                    )
                ],
            )
        else:
            logger.debug(f"   [MARKER] {self.marker_type.value} not in play")

        return StepResult(is_finished=True)


class PlaceRunesStep(GameStep):
    """Snorri's Inscribe the Runes: place one rune below each of the 4 turn
    slots via up to 3 sequential SELECT_OPTION prompts (the 4th and last
    rune is auto-placed, since there is only one left to choose from).

    Overwrites ``hero.rune_slots`` entirely once all 4 runes are placed
    (interp 2: every play of Inscribe re-arranges all 4 runes freely) and
    emits ``GameEvent(RUNES_PLACED)`` with the final arrangement.

    Placement progress lives in ``placed`` (a step field), not local
    variables, so this step survives a save/load round-trip while paused
    mid-prompt.
    """

    type: StepType = StepType.PLACE_RUNES
    hero_id: str
    placed: dict[int, RuneType] = Field(default_factory=dict)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        if self.pending_input:
            selection = self.pending_input.get("selection")
            self.pending_input = None
            try:
                chosen = RuneType(str(selection))
            except ValueError:
                chosen = None
            if chosen is not None and chosen not in self.placed.values():
                self.placed[len(self.placed) + 1] = chosen

        remaining = [rune for rune in RuneType if rune not in self.placed.values()]
        if len(remaining) == 1:
            self.placed[len(self.placed) + 1] = remaining[0]
            remaining = []

        if not remaining:
            hero = state.get_hero(HeroID(self.hero_id))
            if not hero:
                return StepResult(is_finished=True)
            hero.rune_slots = dict(self.placed)
            return StepResult(
                is_finished=True,
                events=[
                    GameEvent(
                        event_type=GameEventType.RUNES_PLACED,
                        actor_id=self.hero_id,
                        metadata={
                            "rune_slots": {
                                str(slot): rune.value for slot, rune in hero.rune_slots.items()
                            }
                        },
                    )
                ],
            )

        from goa2.engine.steps.selection import normalize_prompt_player_id

        slot = len(self.placed) + 1
        player_id = normalize_prompt_player_id(state, HeroID(self.hero_id))
        return StepResult(
            is_finished=False,
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_OPTION,
                player_id=player_id,
                prompt=f"Place a rune below turn slot {slot}",
                options=[InputOption(id=rune.value, text=rune.value.title()) for rune in remaining],
            ),
        )
