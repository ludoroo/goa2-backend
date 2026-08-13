"""Selection and input choice steps."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import Field

from goa2.domain.events import GameEvent, GameEventType
from goa2.domain.input import (
    DONE,
    SKIP,
    InputOption,
    InputRequestType,
    create_input_request,
    parse_hex_selection,
)
from goa2.domain.models import (
    ActionType,
    CardColor,
    CardContainerType,
    CardState,
    CardTier,
    RuneType,
    StepType,
    TargetType,
    TeamColor,
)
from goa2.domain.state import GameState
from goa2.domain.types import HeroID, UnitID
from goa2.engine.filters_base import FilterCondition
from goa2.engine.steps.base import GameStep, StepResult

logger = logging.getLogger(__name__)


def normalize_prompt_player_id(state: GameState, raw_id: Any) -> str:
    """Resolve an entity ID to the player who answers input prompts.

    Hero-piece IDs (hero_razzle_piece_N) normalize to the owning hero;
    anything else (hero IDs, team:XXX strings) passes through unchanged.
    """
    hero = state.get_hero(HeroID(str(raw_id)))
    return str(hero.id) if hero else str(raw_id)


class SelectStep(GameStep):
    """
    Unified selection step using the Filter System.
    Replaces SelectTargetStep and SelectHexStep.

    Supports target types: "UNIT", "HEX", "CARD", "NUMBER"
    For NUMBER type, use number_options to specify valid choices.

    Note: For UNIT selections, ImmunityFilter is automatically applied unless
    skip_immunity_filter=True is set. ExcludeIdentityFilter (self-exclusion) is
    also auto-applied unless skip_self_filter=True is set.
    """

    type: StepType = StepType.SELECT
    target_type: TargetType  # "UNIT", "HEX", "CARD", "NUMBER"
    prompt: str
    output_key: str = "selection"
    filters: list[FilterCondition] = Field(default_factory=list)
    auto_select_if_one: bool = False
    context_hero_id: str | None = None  # Non-empty literal wins over context_hero_id_key
    context_hero_id_key: str | None = None  # Key in context to find hero (for CARD/HAND selection)
    card_container: CardContainerType = (
        CardContainerType.HAND
    )  # "HAND", "PLAYED", "DISCARD", "DECK"
    card_containers: list[CardContainerType] | None = (
        None  # When set, merges candidates from multiple containers
    )
    restrict_played_to_shields: bool = (
        False  # For the PLAYED container, only include active discard-shield cards (Mrak)
    )
    # Facedown cards in the DISCARD/PLAYED containers have lost their identity
    # (rulebook), so they are not offered as selection candidates. Set True for
    # effects that only move a card between zones (e.g. Takahide's retrieve).
    include_facedown: bool = False
    number_options: list[int] = Field(default_factory=list)  # For NUMBER target type
    number_labels: dict[int, str] = Field(default_factory=dict)  # Display text per number option
    skip_immunity_filter: bool = False  # Set True to disable automatic ImmunityFilter
    skip_self_filter: bool = False  # Set True to allow selecting self (e.g. "yourself")
    override_player_id: str | None = None  # Non-empty literal wins over override_player_id_key
    override_player_id_key: str | None = None  # Key in context to find player ID who provides input
    # Card property filters (applied before candidate extraction for CARD selections)
    card_action_types: list[ActionType] | None = (
        None  # Only include cards with primary_action in this list
    )
    card_colors: list[CardColor] | None = None  # Only include cards with these colors
    card_color_key: str | None = None  # Context key containing a CardColor/string to match
    card_tier_key: str | None = None  # Context key containing a CardTier/string to match
    card_is_basic: bool | None = None  # Only include basic (True) or non-basic (False)
    card_is_active: bool | None = None  # Only include active (True) or inactive (False) cards
    card_has_item: bool | None = None  # Only include cards with (or without) an item stat
    allowed_card_ids: list[str] | None = None  # Whitelist: only include cards with these IDs
    # Context keys holding card IDs to EXCLUDE (e.g. "pick two different
    # cards": the second select excludes the first pick's key).
    exclude_card_id_keys: list[str] | None = None
    # Only include cards whose state is in this list (e.g. RESOLVED only).
    card_states: list[CardState] | None = None
    exclude_card_states: list[CardState] | None = None
    selected_card_color_key: str | None = None
    selected_card_tier_key: str | None = None

    @staticmethod
    def _store_selected_card_metadata(
        card_id: str,
        source_list: list[Any],
        context: dict[str, Any],
        color_key: str | None,
        tier_key: str | None,
    ) -> None:
        card = next((candidate for candidate in source_list if candidate.id == card_id), None)
        if card is None:
            return
        if color_key:
            context[color_key] = card.color.value if card.color else None
        if tier_key:
            context[tier_key] = card.tier.value

    def _get_effective_filters(self) -> list[FilterCondition]:
        """
        Returns the effective filter list, auto-adding filters for UNIT selections:
        - ExcludeIdentityFilter (self-exclusion) unless skip_self_filter is True
        - ImmunityFilter unless skip_immunity_filter is True
        """
        from goa2.engine.filters_units import ExcludeIdentityFilter, ImmunityFilter

        effective = list(self.filters)

        if self.target_type in (TargetType.UNIT, TargetType.UNIT_OR_TOKEN):
            # Auto-add ExcludeIdentityFilter for self-exclusion
            if not self.skip_self_filter:
                has_self_exclusion = any(
                    isinstance(f, ExcludeIdentityFilter) and f.exclude_self for f in effective
                )
                if not has_self_exclusion:
                    effective.append(ExcludeIdentityFilter(exclude_self=True))

            # Auto-add ImmunityFilter
            if not self.skip_immunity_filter:
                has_immunity = any(isinstance(f, ImmunityFilter) for f in effective)
                if not has_immunity:
                    effective.append(ImmunityFilter())

        return effective

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            logger.debug(
                f"   [SKIP] Conditional Step '{self.prompt}' skipped (Key '{self.active_if_key}' missing)."
            )
            return StepResult(is_finished=True)

        actor_id = state.current_actor_id
        prompt_player_id = normalize_prompt_player_id(state, actor_id) if actor_id else None
        # Non-empty literal wins; empty literal falls back to key lookup.
        if self.override_player_id:
            actor_id = HeroID(str(self.override_player_id))
            prompt_player_id = normalize_prompt_player_id(state, self.override_player_id)
        elif self.override_player_id_key:
            found = context.get(self.override_player_id_key)
            if found:
                actor_id = HeroID(str(found))
                prompt_player_id = normalize_prompt_player_id(state, found)

        candidates: list[Any] = []
        card_candidates: list[Any] = []
        if self.target_type == TargetType.UNIT:
            # Filter entity_locations for things that are actually Units
            all_entities = list(state.entity_locations.keys())
            candidates = [eid for eid in all_entities if state.get_unit(UnitID(str(eid)))]
            # Illusion tokens count as friendly melee minions while the
            # equivalence effect's source acts (NebKher Illusionary Force /
            # Army) — they must be OFFERED as unit candidates, not merely
            # pass filters.
            from goa2.engine.rules import is_equivalent_illusion

            candidates.extend(
                eid
                for eid in all_entities
                if eid not in candidates and is_equivalent_illusion(state, str(eid))
            )
        elif self.target_type == TargetType.UNIT_OR_TOKEN:
            # Use helper method that filters for Units and Tokens only
            # (excludes future entity types like Structures, Hazards, etc.)
            candidates = state.get_units_and_tokens()
        elif self.target_type == TargetType.HEX:
            # Optimization: If there is a RangeFilter, use it to narrow search area
            # For now, simplistic iteration over all tiles
            candidates = list(state.board.tiles.keys())
        elif self.target_type == TargetType.NUMBER:
            candidates = list(self.number_options)
        elif self.target_type == TargetType.CARD:
            target_id = actor_id
            # Non-empty literal wins; empty literal falls back to key lookup.
            if self.context_hero_id:
                target_id = HeroID(str(self.context_hero_id))
            elif self.context_hero_id_key:
                found_id = context.get(self.context_hero_id_key)
                if found_id:
                    target_id = HeroID(str(found_id))

            hero = state.get_hero(HeroID(str(target_id)))
            if hero:
                source_list = []
                containers = self.card_containers or [self.card_container]
                for container in containers:
                    if container == CardContainerType.HAND:
                        source_list.extend(hero.hand)
                    elif container == CardContainerType.PLAYED:
                        if self.restrict_played_to_shields:
                            from goa2.engine.effects import get_active_shield_cards

                            source_list.extend(get_active_shield_cards(state, hero))
                        else:
                            source_list.extend(c for c in hero.played_cards if c is not None)
                    elif container == CardContainerType.DISCARD:
                        source_list.extend(hero.discard_pile)
                    elif container == CardContainerType.DECK:
                        source_list.extend(hero.deck)

                # A facedown card in the discard/resolved area has no identity to
                # select on (rulebook) — drop it unless the effect opted back in.
                if not self.include_facedown and any(
                    container in (CardContainerType.DISCARD, CardContainerType.PLAYED)
                    for container in containers
                ):
                    source_list = [
                        c
                        for c in source_list
                        if not (
                            c.is_facedown and c.state in (CardState.DISCARD, CardState.RESOLVED)
                        )
                    ]

                # Apply card property filters before extracting IDs
                selected_colors = list(self.card_colors or [])
                if self.card_color_key:
                    color_val = context.get(self.card_color_key)
                    if color_val:
                        selected_colors.append(CardColor(str(color_val)))

                # Rulebook FACEDOWN: a facedown resolved/discarded card has lost
                # its type, color and actions, so it can never match an identity
                # filter — even when include_facedown re-added it above.
                wants_card_identity = (
                    self.card_action_types is not None
                    or bool(selected_colors)
                    or bool(self.card_tier_key and context.get(self.card_tier_key))
                    or self.card_is_basic is not None
                )
                if wants_card_identity:
                    source_list = [
                        c
                        for c in source_list
                        if not (
                            c.is_facedown and c.state in (CardState.DISCARD, CardState.RESOLVED)
                        )
                    ]

                if self.card_action_types is not None:
                    source_list = [
                        c for c in source_list if c.primary_action in self.card_action_types
                    ]
                if selected_colors:
                    source_list = [c for c in source_list if c.color in selected_colors]
                if self.card_tier_key:
                    tier_val = context.get(self.card_tier_key)
                    if tier_val:
                        source_list = [c for c in source_list if c.tier == CardTier(str(tier_val))]
                if self.card_is_basic is not None:
                    source_list = [c for c in source_list if c.is_basic == self.card_is_basic]
                if self.card_is_active is not None:
                    source_list = [c for c in source_list if c.is_active == self.card_is_active]
                if self.card_has_item is not None:
                    source_list = [
                        c for c in source_list if (c.item is not None) == self.card_has_item
                    ]
                if self.allowed_card_ids is not None:
                    source_list = [c for c in source_list if c.id in self.allowed_card_ids]
                if self.exclude_card_id_keys:
                    excluded = {
                        str(context.get(k)) for k in self.exclude_card_id_keys if context.get(k)
                    }
                    source_list = [c for c in source_list if c.id not in excluded]
                if self.card_states is not None:
                    source_list = [c for c in source_list if c.state in self.card_states]
                if self.exclude_card_states is not None:
                    source_list = [
                        c for c in source_list if c.state not in self.exclude_card_states
                    ]

                card_candidates = source_list
                candidates = [c.id for c in source_list]

        valid_candidates = []
        effective_filters = self._get_effective_filters()
        for c in candidates:
            # Intrinsic Validation for UNITS: Check can_be_targeted (LOS, etc.)
            # Only actual units are validated — tokens skip it in both modes
            # (UNIT candidates can include equivalence Illusion tokens, which
            # skip unit-targeting validation like tokens do).
            if (
                self.target_type in (TargetType.UNIT, TargetType.UNIT_OR_TOKEN)
                and actor_id
                and state.get_unit(UnitID(str(c)))
            ):
                val_res = state.validator.can_be_targeted(state, str(actor_id), str(c), context)
                if not val_res.allowed:
                    continue

            is_valid = True
            for f in effective_filters:
                if not f.apply(c, state, context):
                    is_valid = False
                    break
            if is_valid:
                valid_candidates.append(c)

        if not valid_candidates:
            if self.is_mandatory:
                logger.debug(
                    f"   [ABORT] Mandatory selection '{self.prompt}' failed. No candidates."
                )
                return StepResult(is_finished=True, abort_action=True)
            else:
                logger.debug(
                    f"   [SKIP] Optional selection '{self.prompt}' skipped. No candidates."
                )
                return StepResult(is_finished=True)

        if self.auto_select_if_one and len(valid_candidates) == 1 and self.is_mandatory:
            choice = valid_candidates[0]
            context[self.output_key] = choice
            self.pending_input = None
            self.pending_request_id = None
            if self.target_type == TargetType.CARD:
                self._store_selected_card_metadata(
                    str(choice),
                    card_candidates,
                    context,
                    self.selected_card_color_key,
                    self.selected_card_tier_key,
                )
            logger.debug(f"   [AUTO] Only one valid option: {choice}. Selected automatically.")
            return StepResult(is_finished=True)

        if self.pending_input:
            selection = self.pending_input.get("selection")

            if selection == SKIP and not self.is_mandatory:
                logger.debug("   [SKIP] Player chose to skip optional selection.")
                self.pending_input = None
                self.pending_request_id = None
                return StepResult(is_finished=True)

            # Type Conversion for Hex/Number. A malformed client value (a hex
            # dict missing q/r/s, a non-numeric "number") must not raise: the
            # step has already been popped, so raising here would drop it from
            # the stack and corrupt the game. Treat any coercion failure as an
            # invalid choice and re-request input.
            coercion_failed = False
            if self.target_type == TargetType.HEX:
                parsed_hex = parse_hex_selection(selection)
                if parsed_hex is None:
                    coercion_failed = True
                else:
                    selection = parsed_hex
            elif self.target_type == TargetType.NUMBER and selection is not None:
                try:
                    selection = int(selection)
                except (TypeError, ValueError):
                    coercion_failed = True

            if not coercion_failed and selection in valid_candidates:
                context[self.output_key] = selection
                self.pending_input = None
                self.pending_request_id = None
                if self.target_type == TargetType.CARD:
                    self._store_selected_card_metadata(
                        str(selection),
                        card_candidates,
                        context,
                        self.selected_card_color_key,
                        self.selected_card_tier_key,
                    )
                logger.debug(f"   [INPUT] Player {actor_id} selected {selection}")
                return StepResult(is_finished=True)
            else:
                # Invalid choice, re-request
                pass

        # Map target_type to InputRequestType
        type_map = {
            TargetType.UNIT: InputRequestType.SELECT_UNIT,
            TargetType.UNIT_OR_TOKEN: InputRequestType.SELECT_UNIT_OR_TOKEN,
            TargetType.HEX: InputRequestType.SELECT_HEX,
            TargetType.CARD: InputRequestType.SELECT_CARD,
            TargetType.NUMBER: InputRequestType.SELECT_NUMBER,
        }
        request_type = type_map.get(self.target_type, InputRequestType.SELECT_UNIT)

        # Apply labels to number options if provided
        options_for_request = valid_candidates
        if self.target_type == TargetType.NUMBER and self.number_labels:
            options_for_request = [
                InputOption(
                    id=str(n),
                    text=self.number_labels.get(n, str(n)),
                    metadata={"raw": n},
                )
                for n in valid_candidates
            ]

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=request_type,
                player_id=prompt_player_id or str(actor_id),
                prompt=self.prompt,
                options=options_for_request,
                can_skip=not self.is_mandatory,
            ),
        )


class MultiSelectStep(GameStep):
    """
    Allows selecting up to N targets sequentially.
    Stores results as a list in context.

    The step prompts for selection repeatedly until:
    - Player selects "DONE" (if min_selections met)
    - max_selections is reached
    - No more valid candidates

    Uses the same filtering system as SelectStep.
    """

    type: StepType = StepType.MULTI_SELECT
    target_type: TargetType  # "UNIT", "HEX", etc.
    prompt: str
    output_key: str  # Context key for result list
    max_selections: int
    min_selections: int = 0  # 0 = fully optional
    filters: list[FilterCondition] = Field(default_factory=list)
    skip_immunity_filter: bool = False
    skip_self_filter: bool = False  # Set True to allow selecting self

    # Internal state (preserved when pushed back to stack)
    selections: list[str] = Field(default_factory=list)

    def _get_effective_filters(self) -> list[FilterCondition]:
        """Returns filters, auto-adding ExcludeIdentityFilter and ImmunityFilter for UNIT selections."""
        from goa2.engine.filters_units import ExcludeIdentityFilter, ImmunityFilter

        effective = list(self.filters)
        if self.target_type in (TargetType.UNIT, TargetType.UNIT_OR_TOKEN):
            if not self.skip_self_filter:
                has_self_exclusion = any(
                    isinstance(f, ExcludeIdentityFilter) and f.exclude_self for f in effective
                )
                if not has_self_exclusion:
                    effective.append(ExcludeIdentityFilter(exclude_self=True))

            if not self.skip_immunity_filter:
                has_immunity = any(isinstance(f, ImmunityFilter) for f in effective)
                if not has_immunity:
                    effective.append(ImmunityFilter())
        return effective

    def _get_candidates(self, state: GameState, context: dict[str, Any]) -> list[str]:
        """Get valid candidates, excluding already-selected items."""
        actor_id = state.current_actor_id

        # Build initial candidate list based on target type
        candidates: list[Any] = []
        if self.target_type == TargetType.UNIT:
            all_entities = list(state.entity_locations.keys())
            candidates = [eid for eid in all_entities if state.get_unit(UnitID(str(eid)))]
        elif self.target_type == TargetType.UNIT_OR_TOKEN:
            candidates = state.get_units_and_tokens()
        elif self.target_type == TargetType.HEX:
            candidates = list(state.board.tiles.keys())

        # Apply filters
        valid = []
        effective_filters = self._get_effective_filters()
        for c in candidates:
            # Skip already selected
            if str(c) in self.selections:
                continue

            # Targeting validation for units
            if self.target_type == TargetType.UNIT and actor_id:
                val_res = state.validator.can_be_targeted(state, str(actor_id), str(c), context)
                if not val_res.allowed:
                    continue
            elif self.target_type == TargetType.UNIT_OR_TOKEN and actor_id:
                if state.get_unit(UnitID(str(c))):
                    val_res = state.validator.can_be_targeted(state, str(actor_id), str(c), context)
                    if not val_res.allowed:
                        continue

            # Apply custom filters
            is_valid = True
            for f in effective_filters:
                if not f.apply(c, state, context):
                    is_valid = False
                    break
            if is_valid:
                valid.append(str(c))

        return valid

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            context[self.output_key] = []
            return StepResult(is_finished=True)

        actor_id = state.current_actor_id
        prompt_player_id = normalize_prompt_player_id(state, actor_id) if actor_id else None
        if not actor_id:
            context[self.output_key] = self.selections
            return StepResult(is_finished=True)

        # Handle input from previous prompt
        if self.pending_input:
            selection = self.pending_input.get("selection")

            if selection in (DONE, SKIP):
                # Only honor an early finish once the minimum is satisfied.
                # A client must not under-select a mandatory minimum via DONE.
                if len(self.selections) >= self.min_selections:
                    logger.debug(
                        f"   [MULTI-SELECT] Player chose DONE with {len(self.selections)} selections."
                    )
                    context[self.output_key] = list(self.selections)
                    return StepResult(is_finished=True)
                logger.debug(
                    f"   [MULTI-SELECT] DONE ignored: {len(self.selections)} < min "
                    f"{self.min_selections}. Re-requesting."
                )
                self.pending_input = None
            else:
                # Validate the submitted id is in the currently offered candidate
                # set before accepting it (mirrors SelectStep). An invalid id is
                # dropped and the step re-requests input.
                valid_now = self._get_candidates(state, context)
                if str(selection) in valid_now:
                    self.selections.append(str(selection))
                    context[self.output_key] = list(self.selections)
                    logger.debug(
                        f"   [MULTI-SELECT] Added {selection}. "
                        f"Total: {len(self.selections)}/{self.max_selections}"
                    )

                    # Hit max? Done
                    if len(self.selections) >= self.max_selections:
                        logger.debug("   [MULTI-SELECT] Max reached. Finishing.")
                        self.pending_input = None
                        return StepResult(is_finished=True)
                else:
                    logger.debug(
                        f"   [MULTI-SELECT] Rejected invalid selection {selection!r}. "
                        "Re-requesting."
                    )
                self.pending_input = None

        # Get remaining valid candidates
        candidates = self._get_candidates(state, context)

        # No more candidates? Finish
        if not candidates:
            logger.debug("   [MULTI-SELECT] No more candidates. Finishing.")
            context[self.output_key] = list(self.selections)
            # If mandatory and below min, abort
            if self.is_mandatory and len(self.selections) < self.min_selections:
                logger.debug(
                    f"   [ABORT] MultiSelectStep: Only {len(self.selections)} selected, need {self.min_selections}."
                )
                return StepResult(is_finished=True, abort_action=True)
            return StepResult(is_finished=True)

        # Can player skip/finish early?
        allow_done = len(self.selections) >= self.min_selections

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_UNIT,
                player_id=prompt_player_id or str(actor_id),
                prompt=f"{self.prompt} ({len(self.selections)}/{self.max_selections})",
                options=candidates,
                can_skip=allow_done,
            ),
        )


class ChooseMinionRemovalStep(GameStep):
    """
    Self-looping step: the losing team chooses which minion to remove.
    Heavy minions can only be chosen once all non-heavy minions are gone.

    The prompt is skipped only when the casualty is forced: either exactly one
    minion is currently eligible, or every currently eligible minion must go.
    Eligibility is recalculated after each removal, so a batch that clears the
    ordinary minions still asks about the heavy ones that follow.
    """

    type: StepType = StepType.CHOOSE_MINION_REMOVAL
    losing_team: str  # "RED" or "BLUE"
    remaining_to_remove: int
    zone_id: str

    def _get_loser_minions(self, state: GameState) -> list:
        """Get losing team's minions in the active zone."""
        zone = state.board.zones.get(self.zone_id)
        if not zone:
            return []
        team_color = TeamColor(self.losing_team)
        minions = []
        for unit_id, loc in state.unit_locations.items():
            if loc in zone.hexes:
                unit = state.get_unit(UnitID(unit_id))
                if (
                    unit
                    and hasattr(unit, "type")
                    and hasattr(unit, "is_heavy")
                    and unit.team == team_color
                ):
                    minions.append(unit)
        return minions

    def _get_valid_choices(self, minions: list) -> list:
        """Return selectable minions: non-heavy first, heavy only when no non-heavy remain."""
        non_heavy = [m for m in minions if not m.is_heavy]
        if non_heavy:
            return non_heavy
        return minions

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.combat import RemoveUnitStep

        if self.remaining_to_remove <= 0:
            return StepResult(is_finished=True)

        minions = self._get_loser_minions(state)
        if not minions:
            return StepResult(is_finished=True)

        valid = self._get_valid_choices(minions)

        # Forced casualties: only one eligible minion, or every eligible minion
        # must be removed. Either way there is nothing to decide.
        if len(valid) == 1 or self.remaining_to_remove >= len(valid):
            forced = valid[: self.remaining_to_remove]
            removal_steps: list[GameStep] = [RemoveUnitStep(unit_id=str(m.id)) for m in forced]
            left = self.remaining_to_remove - len(forced)
            if left > 0:
                # Eligibility changes once these are gone (heavy minions become
                # selectable), so re-enter rather than removing blindly.
                removal_steps.append(
                    ChooseMinionRemovalStep(
                        losing_team=self.losing_team,
                        remaining_to_remove=left,
                        zone_id=self.zone_id,
                    )
                )
            return StepResult(is_finished=True, new_steps=removal_steps)

        # Player choice needed
        if self.pending_input:
            chosen_id = self.pending_input.get("selection")
            self.pending_input = None
            # Validate the choice was actually offered: a client must not be able
            # to remove a heavy minion while non-heavy minions remain.
            if chosen_id and str(chosen_id) in {str(m.id) for m in valid}:
                logger.debug(f"   [BATTLE] {self.losing_team} chose to remove {chosen_id}.")
                new_steps: list[GameStep] = [
                    RemoveUnitStep(unit_id=str(chosen_id)),
                    ChooseMinionRemovalStep(
                        losing_team=self.losing_team,
                        remaining_to_remove=self.remaining_to_remove - 1,
                        zone_id=self.zone_id,
                    ),
                ]
                return StepResult(is_finished=True, new_steps=new_steps)
            logger.debug(
                f"   [BATTLE] Rejected invalid minion removal {chosen_id!r}; re-requesting."
            )

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_UNIT,
                player_id=f"team:{self.losing_team}",
                prompt=f"Team {self.losing_team}, choose a minion to remove ({self.remaining_to_remove} remaining).",
                options=[str(m.id) for m in valid],
            ),
        )


class AskConfirmationStep(GameStep):
    """
    Prompts the player for a Yes/No confirmation.
    Useful for optional repeats or effects.
    """

    type: StepType = StepType.ASK_CONFIRMATION
    prompt: str
    output_key: str = "confirmation"
    player_id: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        actor_id = self.player_id or state.current_actor_id
        if not actor_id:
            return StepResult(is_finished=True)
        actor_id = normalize_prompt_player_id(state, actor_id)

        if self.pending_input:
            selection = self.pending_input.get("selection")
            # Logic: "YES" = True, "NO" = False
            context[self.output_key] = selection == "YES"
            logger.debug(f"   [INPUT] {actor_id} chose {selection} for '{self.prompt}'")
            return StepResult(is_finished=True)

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_OPTION,
                player_id=str(actor_id),
                prompt=self.prompt,
                options=[
                    InputOption(id="YES", text="Yes"),
                    InputOption(id="NO", text="No"),
                ],
            ),
        )


class ResolveTieBreakerStep(GameStep):
    """
    Recursive handler for tied initiative players.
    1. Determines next winner (via Coin Flip or Team Choice).
    2. Pushes Winner's logic to stack.
    3. Pushes remaining players back via another TieBreakerStep.
    """

    type: StepType = StepType.RESOLVE_TIE_BREAKER
    tied_hero_ids: list[HeroID]

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.cards import ResolveCardStep
        from goa2.engine.steps.combat import RespawnHeroStep
        from goa2.engine.steps.phases import FinalizeHeroTurnStep
        from goa2.engine.steps.reactions import ConfirmResolutionStep

        if not self.tied_hero_ids:
            return StepResult(is_finished=True)

        teams_represented: dict[TeamColor, list[str]] = {}
        for h_id in self.tied_hero_ids:
            hero = state.get_hero(HeroID(h_id))
            if hero and hero.team:
                teams_represented.setdefault(hero.team, []).append(h_id)

        winner_id = None
        needs_input = False
        target_team = None
        candidates = []
        # The coin flips only AFTER a cross-team tie winner's turn resolves
        # (before the next initiative check), not when they are picked. This
        # lets Ignatia read the correct (pre-flip) face on her turn.
        flip_after = False

        # LOGIC:
        # A. If multiple teams -> Use Tie Breaker Coin to pick the FAVORED Team.
        if len(teams_represented) > 1:
            favored_team = state.tie_breaker_team
            if favored_team in teams_represented:
                candidates = teams_represented[favored_team]
                target_team = favored_team
            else:
                target_team = next(iter(teams_represented.keys()))
                candidates = teams_represented[target_team]

            if len(candidates) > 1:
                needs_input = True
            else:
                winner_id = candidates[0]
                flip_after = True
                logger.debug(
                    f"   [TIE] Coin wins for {favored_team.name}. {winner_id} acts. "
                    "Coin flips after their turn."
                )

        # B. If only one team -> they must choose who goes next
        else:
            target_team = next(iter(teams_represented.keys()))
            candidates = teams_represented[target_team]
            if len(candidates) > 1:
                needs_input = True
            else:
                winner_id = candidates[0]

        if needs_input:
            if self.pending_input:
                submitted = self.pending_input.get("selection")
                self.pending_input = None
                # Validate the chosen hero is one of the tied candidates. A client
                # must not be able to install an arbitrary (or wrong-team) hero.
                if str(submitted) in {str(c) for c in candidates}:
                    winner_id = submitted
                    logger.debug(
                        f"   [TIE] Team {target_team.name} chose {winner_id} to act first."
                    )
                    if len(teams_represented) > 1:
                        flip_after = True
                else:
                    logger.debug(
                        f"   [TIE] Rejected invalid tie-break selection {submitted!r}; "
                        "re-requesting."
                    )
            if winner_id is None:
                return StepResult(
                    requires_input=True,
                    input_request=create_input_request(
                        request_type=InputRequestType.CHOOSE_ACTOR,
                        player_id=f"team:{target_team.value}",
                        prompt=f"Team {target_team.name}, choose who acts first between {candidates}.",
                        options=candidates,
                        team=target_team,
                    ),
                )

        # We have a winner!
        if not winner_id:
            raise ValueError("No winner identified in tie breaker.")

        if not state.get_hero(HeroID(winner_id)):
            raise ValueError(f"Tie breaker winner {winner_id!r} is not a known hero.")

        # CRITICAL: Remove winner from unresolved pool so they don't act again immediately
        if winner_id in state.unresolved_hero_ids:
            state.unresolved_hero_ids.remove(HeroID(winner_id))

        state.current_actor_id = HeroID(winner_id)
        state.resolution_owner_id = HeroID(winner_id)

        new_steps: list[GameStep] = []
        if winner_id not in state.entity_locations:
            new_steps.append(RespawnHeroStep(hero_id=winner_id))
        new_steps.append(ResolveCardStep(hero_id=winner_id))
        new_steps.append(ConfirmResolutionStep(hero_id=winner_id))
        if flip_after:
            from goa2.engine.steps.utility import FlipTieBreakerCoinStep

            new_steps.append(FlipTieBreakerCoinStep())
        new_steps.append(FinalizeHeroTurnStep(hero_id=winner_id))

        return StepResult(is_finished=True, new_steps=new_steps)


def _record_guess_attempt(state: GameState, attempt: int, **fields: Any) -> None:
    """Upsert one attempt into the public guess state on GameState.

    This lives on GameState rather than in execution_context because the
    reveal, the discard and FinalizeHeroTurnStep all drain in a single
    process_stack pass: FinalizeHeroTurnStep clears execution_context, so a
    context-derived view would already be empty when the post-mutation view is
    built and broadcast. Clients would then never see the flipped card.
    """
    guesser_id = str(state.current_actor_id) if state.current_actor_id else None
    guess = state.card_guess
    # Attempt 1 always starts a fresh guess. The stored state is only cleared
    # when some *other* hero finishes a turn, so a guesser who acts last in one
    # round and first in the next would otherwise inherit the previous guess's
    # attempt 2 and render it alongside the new one.
    if guess is None or guess.get("guesser_id") != guesser_id or attempt == 1:
        guess = {"guesser_id": guesser_id, "attempts": []}
    attempts = [a for a in guess["attempts"] if a["attempt"] != attempt]
    existing: dict[str, Any] = next((a for a in guess["attempts"] if a["attempt"] == attempt), {})
    attempts.append({**existing, "attempt": attempt, **fields})
    attempts.sort(key=lambda a: a["attempt"])
    state.card_guess = {"guesser_id": guesser_id, "attempts": attempts}


class GuessCardColorStep(GameStep):
    """Prompts the actor to guess a card color.

    Offers the 5 standard card colors: BLUE, GOLD, GREEN, RED, SILVER. If
    ``victim_key`` is set, the options are restricted to colors that could be
    in that hero's hand (i.e. colors actually present in their hand), per the
    "You can only guess colors that could be in that player's hand" rule.
    The actor picks one via SELECT_OPTION.
    """

    VALID_COLORS: ClassVar[list[str]] = ["BLUE", "GOLD", "GREEN", "RED", "SILVER"]

    type: StepType = StepType.GUESS_CARD_COLOR
    # See _record_guess_attempt for why this is state, not execution_context.
    output_key: str  # where to store the guessed color string
    victim_key: str = ""  # context key → hero ID whose hand restricts the options
    card_key: str = ""  # context key → the facedown card selected for this guess
    attempt: int = 1
    selection_announced: bool = False

    def _valid_colors(self, state: GameState, context: dict[str, Any]) -> list[str]:
        """Colors that could be in the victim's hand, from the guesser's view.

        The guess universe is the victim's in-play cards: hand + played cards
        + current/extra turn card + discard. Faceup cards in that universe are
        publicly accounted for; a color is guessable while at least one copy
        remains unaccounted — i.e. it sits in the hand or facedown somewhere
        (Takahide's Bushido, hidden commits). The deck zone is outside the
        universe and never contributes.
        """
        if self.victim_key:
            victim_id = context.get(self.victim_key)
            if victim_id:
                victim = state.get_hero(HeroID(str(victim_id)))
                if victim:
                    possible = {c.color.value for c in victim.hand if c.color}
                    outside_hand = [
                        *victim.played_cards,
                        *victim.discard_pile,
                        victim.current_turn_card,
                        victim.extra_turn_card,
                    ]
                    possible |= {
                        c.color.value
                        for c in outside_hand
                        if c is not None and c.is_facedown and c.color
                    }
                    restricted = [c for c in self.VALID_COLORS if c in possible]
                    if restricted:
                        return restricted
        return list(self.VALID_COLORS)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        if self.pending_input:
            selection = self.pending_input.get("selection")
            context[self.output_key] = selection
            return StepResult(is_finished=True)

        options = [
            InputOption(id=color, text=color) for color in self._valid_colors(state, context)
        ]
        player_id = (
            normalize_prompt_player_id(state, state.current_actor_id)
            if state.current_actor_id
            else ""
        )

        events: list[GameEvent] = []
        victim_id = context.get(self.victim_key) if self.victim_key else None
        selected_card_id = context.get(self.card_key) if self.card_key else None
        if not self.selection_announced and victim_id and selected_card_id:
            # Publish only the physical/public fact that a facedown card was
            # placed for the guess. Its identity remains private until the
            # RevealAndResolveGuessStep flips it.
            _record_guess_attempt(
                state,
                self.attempt,
                victim_id=str(victim_id),
                card_id=str(selected_card_id),
                guessed_color=None,
                actual_color=None,
                correct=None,
            )
            events.append(
                GameEvent(
                    event_type=GameEventType.CARD_SELECTED_FOR_GUESS,
                    actor_id=str(state.current_actor_id) if state.current_actor_id else None,
                    target_id=str(victim_id),
                    metadata={"attempt": self.attempt},
                )
            )
            self.selection_announced = True

        return StepResult(
            is_finished=False,
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_OPTION,
                player_id=player_id,
                prompt="Guess the card's color",
                options=options,
            ),
            events=events,
        )


class ChooseCardColorStep(GameStep):
    """Prompts a player to name one of the five standard card colors."""

    VALID_COLORS: ClassVar[list[str]] = ["BLUE", "GOLD", "GREEN", "RED", "SILVER"]

    type: StepType = StepType.CHOOSE_CARD_COLOR
    output_key: str
    player_id_key: str | None = None
    prompt: str = "Name a card color"

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        player_id = str(state.current_actor_id) if state.current_actor_id else ""
        if self.player_id_key:
            selected_player = context.get(self.player_id_key)
            if selected_player:
                selected_player = normalize_prompt_player_id(state, selected_player)
            if selected_player:
                player_id = str(selected_player)

        if not player_id:
            return StepResult(is_finished=True)

        if self.pending_input:
            selection = self.pending_input.get("selection")
            if selection in self.VALID_COLORS:
                context[self.output_key] = selection
                logger.debug(f"   [INPUT] Player {player_id} named color {selection}")
                return StepResult(is_finished=True)
            self.pending_input = None

        return StepResult(
            is_finished=False,
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_OPTION,
                player_id=player_id,
                prompt=self.prompt,
                options=[InputOption(id=color, text=color) for color in self.VALID_COLORS],
            ),
        )


class ChooseRuneStep(GameStep):
    """Prompt the current actor to choose one rune from a supplied set.

    ``matching_options`` is optional metadata for effects such as Snorri's
    Oath of Perseverance: it records whether the selected rune satisfies the
    action-specific condition without coupling this general selector to combat.
    """

    type: StepType = StepType.CHOOSE_RUNE
    output_key: str
    options: list[RuneType]
    prompt: str
    matching_options: list[RuneType] = Field(default_factory=list)
    matches_output_key: str | None = None
    value_map: dict[str, str] = Field(default_factory=dict)
    value_output_key: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        option_values = {rune.value for rune in self.options}
        if self.pending_input:
            selection = str(self.pending_input.get("selection"))
            self.pending_input = None
            if selection in option_values:
                context[self.output_key] = selection
                if self.matches_output_key:
                    matching_values = {rune.value for rune in self.matching_options}
                    context[self.matches_output_key] = (
                        True if selection in matching_values else None
                    )
                if self.value_output_key and selection in self.value_map:
                    context[self.value_output_key] = self.value_map[selection]
                return StepResult(is_finished=True)

        if len(self.options) == 1:
            selected = self.options[0]
            context[self.output_key] = selected.value
            if self.matches_output_key:
                context[self.matches_output_key] = (
                    True if selected in self.matching_options else None
                )
            if self.value_output_key and selected.value in self.value_map:
                context[self.value_output_key] = self.value_map[selected.value]
            return StepResult(is_finished=True)

        player_id = (
            normalize_prompt_player_id(state, state.current_actor_id)
            if state.current_actor_id
            else ""
        )
        return StepResult(
            is_finished=False,
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_OPTION,
                player_id=player_id,
                prompt=self.prompt,
                options=[
                    InputOption(id=rune.value, text=rune.value.title()) for rune in self.options
                ],
            ),
        )


class RevealAndResolveGuessStep(GameStep):
    """Reveals the chosen card and compares its color to the guessed color.

    Sets correct_output_key to True if correct (None otherwise),
    and wrong_output_key to True if wrong (None otherwise).
    This dual-flag approach works with active_if_key branching.
    """

    type: StepType = StepType.REVEAL_AND_RESOLVE_GUESS
    card_key: str  # context key → chosen card ID
    guess_key: str  # context key → guessed color string
    victim_key: str  # context key → victim hero ID
    correct_output_key: str
    wrong_output_key: str
    attempt: int = 1

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        card_id = context.get(self.card_key)
        guessed_color = context.get(self.guess_key)
        victim_id = context.get(self.victim_key)

        if not card_id or not guessed_color or not victim_id:
            return StepResult(is_finished=True)

        victim = state.get_hero(HeroID(str(victim_id)))
        if not victim:
            return StepResult(is_finished=True)

        # Find the card in victim's hand
        target_card = next((c for c in victim.hand if c.id == card_id), None)
        if not target_card:
            return StepResult(is_finished=True)

        actual_color = target_card.color.value if target_card.color else None
        is_correct = guessed_color == actual_color

        if is_correct:
            context[self.correct_output_key] = True
            context[self.wrong_output_key] = None
            logger.debug(f"   [GUESS] Correct! Card is {actual_color}, guessed {guessed_color}")
        else:
            context[self.correct_output_key] = None
            context[self.wrong_output_key] = True
            logger.debug(f"   [GUESS] Wrong! Card is {actual_color}, guessed {guessed_color}")

        # The reveal leaks hidden hand info to the actor; rolling back past it
        # would allow re-guessing with that knowledge. This is a segment
        # boundary (same shape as a mine trigger or foreign input), not a
        # hard freeze.
        context["rollback_reanchor_pending"] = True

        # The flip is public, so this event names the card for every recipient.
        # The card face itself stays in the view: a wrong guess leaves the card
        # in an otherwise-private hand, and the view is what tracks that.
        _record_guess_attempt(
            state,
            self.attempt,
            victim_id=str(victim_id),
            card_id=str(card_id),
            guessed_color=guessed_color,
            actual_color=actual_color,
            correct=is_correct,
        )
        state.record_public_revealed_card(victim.id, str(target_card.id))

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.GUESSED_CARD_REVEALED,
                    actor_id=str(state.current_actor_id) if state.current_actor_id else None,
                    target_id=str(victim_id),
                    metadata={
                        "attempt": self.attempt,
                        "card_id": card_id,
                        "card_name": target_card.name,
                        "card_color": actual_color,
                        "guessed_color": guessed_color,
                        "guess_correct": is_correct,
                    },
                )
            ],
        )
