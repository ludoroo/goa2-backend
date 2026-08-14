"""Active effects and passive ability steps."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from goa2.domain.events import GameEvent, GameEventType
from goa2.domain.hex import Hex
from goa2.domain.input import InputOption, InputRequestType, create_input_request
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardState,
    MinionType,
    StepType,
    TeamColor,
)
from goa2.domain.models.effect import ActiveEffect, DurationType, EffectScope, EffectType, Shape
from goa2.domain.models.enums import DisplacementType, StatType
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID, HeroID
from goa2.engine.effect_manager import EffectManager
from goa2.engine.steps.base import GameStep, StepResult, isolate_steps
from goa2.engine.topology import get_topology_service

logger = logging.getLogger(__name__)


class CreateEffectStep(GameStep):
    """Creates a spatial ActiveEffect in the game state."""

    type: StepType = StepType.CREATE_EFFECT

    effect_type: EffectType
    scope: EffectScope
    duration: DurationType = DurationType.THIS_TURN

    restrictions: list[ActionType] = Field(default_factory=list)
    displacement_blocks: list[DisplacementType] = Field(default_factory=list)
    except_card_colors: list[CardColor] = Field(default_factory=list)
    except_attacker_ids: list[str] = Field(default_factory=list)  # Direct list of attacker IDs
    except_attacker_key: str | None = None  # Context key to read attacker ID from
    basic_attacks_only: bool = False
    non_basic_attacks_only: bool = False
    stat_type: StatType | None = None
    stat_value: int = 0
    apply_stat_value_only_if_result_at_least: int | None = None
    max_value: int | None = None
    limit_actions_only: bool = False

    blocks_enemy_actors: bool = True
    blocks_friendly_actors: bool = False
    blocks_self: bool = False
    is_active: bool = False  # Override default dormant state if True
    granted_passive_effect_id: str | None = None

    # Card linkage (for card-based effects). Steps built through the CardEffect
    # API get source_card_id stamped at build time by effects.bind_effect_cards,
    # which is the only place that reliably knows the card being performed. The
    # context fallback below survives for steps constructed outside that API.
    source_card_id: str | None = None  # Explicit card ID
    use_context_card: bool = True  # If True, infer the card performing the current action

    # Origin action type - tracks whether effect came from skill or attack
    origin_action_type: ActionType | None = None

    # Dynamic origin: read scope.origin_id from context at resolve time
    origin_id_key: str | None = None

    # Token-bound effect: skip card binding so lifecycle is tied to token, not card
    is_token_effect: bool = False

    # Static Barrier parameters (Wasp)
    barrier_radius: int = 0  # The radius boundary for the barrier
    barrier_origin_id: str | None = None  # Entity ID for radius calculation

    # Allowed discard colors for MINION_PROTECTION effects (Brogan)
    allowed_discard_colors: list[CardColor] = Field(default_factory=list)
    protected_minion_types: list[MinionType] = Field(default_factory=list)
    sacrifice_origin_token: bool = False

    # Steps to execute when this effect expires (for DELAYED_TRIGGER effects)
    # Patched to List[AnyStep] in step_types.py.
    finishing_steps: list[GameStep] = Field(default_factory=list)

    # MOVEMENT_AURA_ZONE payload (Silverarrow - Trailblazer)
    grants_pass_through_obstacles: bool = False

    # PRE_ACTION_DISCARD payload (Trinkets - Disruptor family)
    discard_or_defeat: bool = False

    # Topology payload (NebKher - Crack in Reality / Shift Reality).
    # split_axis/split_value define the line; isolated_hex is the Tier-3
    # fallback anchor (the live source position takes precedence — S6).
    split_axis: str | None = None
    split_value: int = 0
    isolated_hex: Hex | None = None

    # Publicly named card color (NebKher - Imbue Doubt family): context key
    # holding the color the player announced. Stored on the effect so clients
    # can display it for everyone while the effect is pending.
    named_color_key: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.markers import TOKEN_TYPE_OVERRIDE_KEY

        if self.should_skip(context):
            return StepResult(is_finished=True)

        # A copied action that substitutes token types (NebKher's Mind Grip)
        # never places the token this effect is anchored to, so the ongoing
        # effect has no referent: an Illusion is not a Smoke bomb / Ice /
        # Totem / Barrier. Drop it; the substituted token is still placed.
        if self.is_token_effect and context.get(TOKEN_TYPE_OVERRIDE_KEY):
            return StepResult(is_finished=True)

        # Resolve source card ID (skip for token-bound effects)
        card_id = None
        if not self.is_token_effect:
            card_id = self.source_card_id
            if card_id is None and self.use_context_card:
                card_id = context.get("current_card_id")

        # Resolve origin action type: use explicit value or fall back to context
        action_type = self.origin_action_type
        if action_type is None:
            action_type = context.get("current_action_type")

        # Resolve except_attacker_ids: combine direct list with context key
        resolved_except_attackers = list(self.except_attacker_ids)
        if self.except_attacker_key:
            attacker_from_context = context.get(self.except_attacker_key)
            if attacker_from_context and attacker_from_context not in resolved_except_attackers:
                resolved_except_attackers.append(attacker_from_context)

        # Resolve dynamic origin_id from context if specified
        resolved_scope = self.scope
        if self.origin_id_key:
            origin_id_from_ctx = context.get(self.origin_id_key)
            if origin_id_from_ctx:
                resolved_scope = self.scope.model_copy(
                    update={"origin_id": str(origin_id_from_ctx)}
                )

        def resolve_steps(steps: list[GameStep]) -> list[GameStep]:
            resolved = []
            for step in steps:
                if (
                    hasattr(step, "color_key")
                    and step.color_key
                    and getattr(step, "color", None) is None
                ):
                    color_val = context.get(step.color_key)
                    if color_val:
                        step = step.model_copy(update={"color": CardColor(str(color_val))})
                if hasattr(step, "steps_template") and step.steps_template:
                    template = resolve_steps(step.steps_template)
                    step = step.model_copy(update={"steps_template": template})
                if hasattr(step, "new_steps") and step.new_steps:
                    new_s = resolve_steps(step.new_steps)
                    step = step.model_copy(update={"new_steps": new_s})
                resolved.append(step)
            return resolved

        # `resolve_steps` rebuilds steps with `model_copy`, which skips validators —
        # so the steps it returns are still the ones stored on this step, and the
        # Effect would share run state (pending_input, counters) with its template.
        # `Effect.finishing_steps` is a plain `list[Any]` and isolates nothing, so
        # the copy has to happen here, at the step -> Effect boundary.
        resolved_finishing = isolate_steps(resolve_steps(self.finishing_steps))

        named_color: CardColor | None = None
        if self.named_color_key:
            color_val = context.get(self.named_color_key)
            if color_val:
                named_color = CardColor(str(color_val))

        # A token-bound effect belongs to the token it is anchored to, so record
        # that token: it is the effect's whole lifecycle.
        token_id = (
            str(resolved_scope.origin_id)
            if self.is_token_effect and resolved_scope.origin_id
            else None
        )

        EffectManager.create_effect(
            state=state,
            source_id=(str(state.current_actor_id) if state.current_actor_id else "system"),
            source_card_id=card_id,
            token_id=token_id,
            effect_type=self.effect_type,
            scope=resolved_scope,
            duration=self.duration,
            restrictions=self.restrictions,
            displacement_blocks=self.displacement_blocks,
            except_card_colors=self.except_card_colors,
            except_attacker_ids=resolved_except_attackers,
            basic_attacks_only=self.basic_attacks_only,
            non_basic_attacks_only=self.non_basic_attacks_only,
            stat_type=self.stat_type,
            stat_value=self.stat_value,
            apply_stat_value_only_if_result_at_least=(
                self.apply_stat_value_only_if_result_at_least
            ),
            max_value=self.max_value,
            limit_actions_only=self.limit_actions_only,
            blocks_enemy_actors=self.blocks_enemy_actors,
            blocks_friendly_actors=self.blocks_friendly_actors,
            blocks_self=self.blocks_self,
            is_active=self.is_active,
            granted_passive_effect_id=self.granted_passive_effect_id,
            origin_action_type=action_type,
            barrier_radius=self.barrier_radius,
            barrier_origin_id=self.barrier_origin_id,
            finishing_steps=resolved_finishing,
            allowed_discard_colors=self.allowed_discard_colors,
            protected_minion_types=self.protected_minion_types,
            sacrifice_origin_token=self.sacrifice_origin_token,
            grants_pass_through_obstacles=self.grants_pass_through_obstacles,
            discard_or_defeat=self.discard_or_defeat,
            split_axis=self.split_axis,
            split_value=self.split_value,
            isolated_hex=self.isolated_hex,
            named_color=named_color,
        )

        source = str(state.current_actor_id) if state.current_actor_id else "system"
        logger.debug(f"   [EFFECT] Created {self.effect_type.value} from {source}")

        metadata: dict[str, Any] = {
            "effect_type": self.effect_type.value,
            "duration": self.duration.value,
        }
        # Advertise the split geometry so clients can draw the line.
        if self.split_axis is not None:
            metadata["split_axis"] = self.split_axis
            metadata["split_value"] = self.split_value
        # Advertise the publicly named color (Imbue Doubt family).
        if named_color is not None:
            metadata["named_color"] = named_color.value

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.EFFECT_CREATED,
                    actor_id=source,
                    metadata=metadata,
                )
            ],
        )


class ScheduleJourneyReturnStep(GameStep):
    """Hanu's Journey line (Unexpected Journey / There and Back Again / Safe
    Travels). After Hanu swaps with an enemy hero, this step:

    1. Makes that displaced hero immune (heavy-style, ``IMMUNITY_ENEMY_ACTIONS``)
       to Hanu's team for the rest of the turn.
    2. Schedules an end-of-turn forced swap-back (``DELAYED_TRIGGER`` whose
       ``finishing_steps`` swap the two heroes back). ``SwapUnitsStep`` ignores
       range and immunity, matching "regardless of radius and immunity".
    3. For Safe Travels (``move_after``), appends an optional 1-space move after
       the swap-back.

    Both hero ids are baked into the finishing steps as literals so the
    end-of-turn payload does not depend on the (by-then cleared) turn context.

    Hanu owns both effects (``source_id``); the displaced hero is the immunity's
    ``subject_id``. That split is what lets his defeat end the protection he
    granted — the swap-back fizzles then too, so the displaced hero keeps the
    position.
    """

    type: StepType = StepType.SCHEDULE_JOURNEY_RETURN
    enemy_key: str
    move_after: bool = False
    # Stamped by effects.bind_effect_cards: both halves are the Journey card's
    # active effect, so they end if the card leaves play.
    source_card_id: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        enemy_id = context.get(self.enemy_key)
        actor_id = str(state.current_actor_id) if state.current_actor_id else None
        if not enemy_id or not actor_id:
            return StepResult(is_finished=True)
        enemy_id = str(enemy_id)

        from goa2.engine.steps.movement import MoveUnitStep, SwapUnitsStep

        # (1) Displaced hero becomes immune to EVERYONE this turn (heavy-style):
        # blocks_friendly_actors so even its own allies cannot affect it.
        # Hanu owns it (source_id) so his defeat ends it; the displaced hero is
        # its subject, which is who it protects.
        EffectManager.create_effect(
            state=state,
            source_id=actor_id,
            subject_id=enemy_id,
            source_card_id=self.source_card_id,
            effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_TURN,
            is_active=True,
            blocks_friendly_actors=True,
        )

        # (2) End-of-turn forced swap-back (regardless of range and immunity).
        finishing: list[GameStep] = [SwapUnitsStep(unit_a_id=actor_id, unit_b_id=enemy_id)]
        # (3) Safe Travels: optional 1-space move after the swap-back.
        if self.move_after:
            from goa2.domain.models import TargetType
            from goa2.engine.filters_hex import MovementPathFilter, ObstacleFilter
            from goa2.engine.steps.selection import SelectStep

            finishing.extend(
                [
                    SelectStep(
                        target_type=TargetType.HEX,
                        prompt="You may move 1 space",
                        output_key="journey_move_dest",
                        is_mandatory=False,
                        filters=[
                            MovementPathFilter(range_val=1, unit_id=actor_id),
                            ObstacleFilter(is_obstacle=False),
                        ],
                    ),
                    MoveUnitStep(
                        unit_id=actor_id,
                        destination_key="journey_move_dest",
                        range_val=1,
                        is_movement_action=False,
                        active_if_key="journey_move_dest",
                    ),
                ]
            )

        EffectManager.create_effect(
            state=state,
            source_id=actor_id,
            source_card_id=self.source_card_id,
            effect_type=EffectType.DELAYED_TRIGGER,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_TURN,
            is_active=True,
            finishing_steps=finishing,
        )

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.EFFECT_CREATED,
                    actor_id=actor_id,
                    target_id=enemy_id,
                    metadata={"effect_type": EffectType.IMMUNITY_ENEMY_ACTIONS.value},
                )
            ],
        )


class ScheduleActionControlStep(GameStep):
    """Hanu's ultimate (The Ultimate Trick): after Hurry Up! targets a hero,
    record a CONTROL_NEXT_ACTION effect so that when that hero acts to resolve
    the targeted card, every input addressed to them is answered by Hanu's
    player instead (player_id remap in the handler). Only the decision-maker
    changes — the actor, and therefore all legality, remains the controlled
    hero. No-ops unless the acting hero's ultimate is unlocked (level >= 8
    with an ultimate card). Control fizzles if the card leaves UNRESOLVED any
    other way (controlled_card_id guard) or the round ends (THIS_ROUND)."""

    type: StepType = StepType.SCHEDULE_ACTION_CONTROL
    hero_key: str

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        if state.current_actor_id is None:
            return StepResult(is_finished=True)
        actor_id = str(state.current_actor_id)
        actor = state.get_hero(HeroID(actor_id))
        if not actor or actor.level < 8 or not actor.ultimate_card:
            return StepResult(is_finished=True)

        target_id = context.get(self.hero_key)
        if not target_id:
            return StepResult(is_finished=True)
        target = state.get_hero(HeroID(str(target_id)))
        if not target or target.current_turn_card is None:
            return StepResult(is_finished=True)
        card = target.current_turn_card
        if card.state != CardState.UNRESOLVED:
            return StepResult(is_finished=True)

        EffectManager.create_effect(
            state=state,
            source_id=actor_id,
            effect_type=EffectType.CONTROL_NEXT_ACTION,
            scope=EffectScope(shape=Shape.POINT, origin_id=str(target_id)),
            duration=DurationType.THIS_ROUND,
            is_active=True,
            controlled_card_id=card.id,
        )

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.EFFECT_CREATED,
                    actor_id=actor_id,
                    target_id=str(target_id),
                    metadata={
                        "effect": "action_control",
                        "controller_id": actor_id,
                        "card_id": card.id,
                    },
                )
            ],
        )


class FinishedExpiringEffectStep(GameStep):
    """
    Placeholder step to indicate that an expiring effect has finished resolving.
    Primarily used to correctly mark the end of abort action cascade.
    """

    type: StepType = StepType.FINISHED_EXPIRING_EFFECT

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        logger.debug("   [EFFECT] Finished resolving expiring effect.")
        return StepResult(is_finished=True)


class CancelEffectsStep(GameStep):
    """
    Cancels (removes) active effects matching specified criteria.

    Usage:
        CancelEffectsStep(
            effect_types=[EffectType.TARGET_PREVENTION],
            origin_action_types=[ActionType.SKILL],
            source_team=TeamColor.RED,
            scope=EffectScope(shape=Shape.RADIUS, range=3, origin_id="turret_1"),
        )
    """

    type: StepType = StepType.CANCEL_EFFECTS

    effect_types: list[EffectType] = Field(default_factory=list)
    origin_action_types: list[ActionType] = Field(default_factory=list)
    source_team: TeamColor | None = None
    source_ids: list[str] = Field(default_factory=list)
    scope: EffectScope | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        def effect_matches(effect: ActiveEffect) -> bool:
            if self.effect_types and effect.effect_type not in self.effect_types:
                return False

            if (
                self.origin_action_types
                and effect.origin_action_type not in self.origin_action_types
            ):
                return False

            if self.source_ids and effect.source_id not in self.source_ids:
                return False

            if self.source_team:
                source_entity = state.get_entity(BoardEntityID(effect.source_id))
                if source_entity is None:
                    return False
                source_team_color = getattr(source_entity, "team", None)
                if source_team_color != self.source_team:
                    return False

            if self.scope:
                effect_origin = self._get_effect_origin(effect, state)
                scope_origin = self._get_scope_origin(state)
                if not effect_origin or not scope_origin:
                    return False
                if not self._hex_in_scope(effect_origin, scope_origin, state):
                    return False

            return True

        initial_count = len(state.active_effects)
        state.active_effects = [e for e in state.active_effects if not effect_matches(e)]
        cancelled_count = initial_count - len(state.active_effects)

        if cancelled_count > 0:
            logger.debug(f"   [EFFECT] Cancelled {cancelled_count} active effect(s)")

        return StepResult(is_finished=True)

    def _get_effect_origin(self, effect: ActiveEffect, state: GameState) -> Hex | None:
        if effect.scope.origin_hex:
            return effect.scope.origin_hex
        origin_id = effect.scope.origin_id or effect.source_id
        return state.get_position(origin_id)

    def _get_scope_origin(self, state: GameState) -> Hex | None:
        if not self.scope:
            return None
        if self.scope.origin_hex:
            return self.scope.origin_hex
        if self.scope.origin_id:
            return state.get_position(self.scope.origin_id)
        return None

    def _hex_in_scope(self, hex: Hex, origin: Hex, state: GameState) -> bool:
        if not origin:
            return False
        if self.scope is None:
            return False

        # Use TopologyService for consolidated, topology-aware scope checking
        topology = get_topology_service()
        return topology.hex_in_scope(
            origin,
            hex,
            self.scope.shape,
            self.scope.range,
            state,
            self.scope.direction,
        )

    def _get_zone_for_hex(self, hex: Hex, state: GameState) -> str | None:
        for zone_id, zone in state.board.zones.items():
            if hex in zone.hexes:
                return zone_id
        return None


class CheckPassiveAbilitiesStep(GameStep):
    """
    Checks for passive abilities that trigger at the specified point.
    For each eligible passive, spawns an OfferPassiveStep.

    Scans three sources:
    1. Regular cards in played_cards (RESOLVED + face-up)
    2. Ultimate card (if hero.level >= 8)
    3. Active effects that explicitly grant their creator a passive
    """

    type: StepType = StepType.CHECK_PASSIVE_ABILITIES
    trigger: str  # PassiveTrigger value as string for serialization
    hero_id: str | None = None  # Override which hero is scanned (default: current actor)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.domain.models.enums import PassiveTrigger
        from goa2.engine.effects import CardEffectRegistry

        scan_id = self.hero_id or state.current_actor_id
        hero = state.get_hero(HeroID(str(scan_id))) if scan_id else None
        if not hero:
            return StepResult(is_finished=True)

        trigger_enum = PassiveTrigger(self.trigger)
        offer_steps: list[GameStep] = []

        seen_card_ids: set[str] = set()

        def check_card_for_passive(
            card: Card,
            *,
            provider_effect: ActiveEffect | None = None,
        ) -> None:
            """Helper to check a card for matching passive ability."""
            effect_id = (
                provider_effect.granted_passive_effect_id
                if provider_effect is not None
                else card.current_effect_id
            )
            if not effect_id:
                return

            effect = CardEffectRegistry.get(effect_id)
            if not effect:
                return

            for config in effect.get_passive_configs():
                if config.trigger != trigger_enum:
                    continue

                # Check usage limit
                uses_this_turn = (
                    provider_effect.passive_uses_this_turn
                    if provider_effect is not None
                    else card.passive_uses_this_turn
                )
                if config.uses_per_turn > 0 and uses_this_turn >= config.uses_per_turn:
                    logger.debug(
                        f"   [PASSIVE] {card.name} already used "
                        f"{uses_this_turn}/{config.uses_per_turn} times this turn"
                    )
                    continue

                # Runtime predicate (e.g. Battle Fury filters on discard_source)
                if not effect.should_offer_passive(state, hero, card, trigger_enum, context):
                    continue

                # Spawn offer step for this passive
                offer_steps.append(
                    OfferPassiveStep(
                        card_id=card.id,
                        trigger=self.trigger,
                        is_optional=config.is_optional,
                        prompt=config.prompt or f"Use {card.name} passive ability?",
                        hero_id=str(hero.id) if self.hero_id else None,
                        active_effect_id=(
                            provider_effect.id if provider_effect is not None else None
                        ),
                    )
                )

        # 1. Check regular cards: must be RESOLVED and face-up
        for card in hero.played_cards:
            if card and card.state == CardState.RESOLVED and not card.is_facedown:
                seen_card_ids.add(str(card.id))
                check_card_for_passive(card)

        # 2. Check ultimate card: active if level >= 8
        if hero.level >= 8 and hero.ultimate_card:
            seen_card_ids.add(str(hero.ultimate_card.id))
            check_card_for_passive(hero.ultimate_card)

        # 3. An active effect can explicitly grant its passive to the hero who
        # created it. This is distinct from card ownership: Mind Grip performs
        # an enemy card, but any ongoing effect and its "you" clauses belong to
        # NebKher. Cards already scanned above win, preventing duplicate offers
        # for an ordinarily played card such as Cordelia's own Jinx.
        for provider_effect in state.active_effects:
            card_id = provider_effect.source_card_id
            if (
                str(provider_effect.source_id) != str(hero.id)
                or not provider_effect.is_active
                or not provider_effect.granted_passive_effect_id
                or not card_id
                or card_id in seen_card_ids
            ):
                continue
            card = state.get_card_by_id(card_id)
            if card is None:
                continue
            seen_card_ids.add(card_id)
            check_card_for_passive(card, provider_effect=provider_effect)

        if offer_steps:
            logger.debug(
                f"   [PASSIVE] Found {len(offer_steps)} passive(s) for trigger {trigger_enum.value}"
            )

        return StepResult(is_finished=True, new_steps=offer_steps)


class OfferPassiveStep(GameStep):
    """
    Offers player a choice to use an optional passive ability.
    If mandatory or accepted, spawns the passive steps.
    """

    type: StepType = StepType.OFFER_PASSIVE
    card_id: str
    trigger: str  # PassiveTrigger value as string
    is_optional: bool = True
    prompt: str = ""
    hero_id: str | None = None  # Override: whose passive this is (default: current actor)
    active_effect_id: str | None = None  # Effect-backed provider for copied ongoing text

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.domain.models.enums import PassiveTrigger
        from goa2.engine.effects import CardEffectRegistry

        owner_id = self.hero_id or state.current_actor_id
        hero = state.get_hero(HeroID(str(owner_id))) if owner_id else None
        if not hero:
            return StepResult(is_finished=True)

        provider_effect = None
        if self.active_effect_id is not None:
            provider_effect = next(
                (
                    active
                    for active in state.active_effects
                    if active.id == self.active_effect_id
                    and active.is_active
                    and str(active.source_id) == str(hero.id)
                    and active.source_card_id == self.card_id
                    and active.granted_passive_effect_id
                ),
                None,
            )
            card = state.get_card_by_id(self.card_id) if provider_effect else None
            effect_id = provider_effect.granted_passive_effect_id if provider_effect else None
        else:
            # Find the card (could be in played_cards or ultimate_card)
            card = next((c for c in hero.played_cards if c and c.id == self.card_id), None)
            if not card and hero.ultimate_card and hero.ultimate_card.id == self.card_id:
                card = hero.ultimate_card
            effect_id = card.current_effect_id if card else None

        if not card:
            logger.debug(f"   [PASSIVE] Card {self.card_id} not found")
            return StepResult(is_finished=True)

        if effect_id is None:
            return StepResult(is_finished=True)

        effect = CardEffectRegistry.get(effect_id)
        if not effect:
            return StepResult(is_finished=True)

        trigger_enum = PassiveTrigger(self.trigger)

        def execute_passive() -> StepResult:
            """Helper to spawn the passive steps and mark used."""
            passive_steps = effect.get_passive_steps(state, hero, card, trigger_enum, context)
            if passive_steps:
                logger.debug(f"   [PASSIVE] Activating {card.name}: {len(passive_steps)} step(s)")
                # Add MarkPassiveUsedStep after the passive steps
                return StepResult(
                    is_finished=True,
                    new_steps=[
                        *passive_steps,
                        MarkPassiveUsedStep(
                            card_id=self.card_id,
                            active_effect_id=self.active_effect_id,
                        ),
                    ],
                )
            return StepResult(is_finished=True)

        # If not optional, auto-execute
        if not self.is_optional:
            return execute_passive()

        # Handle player input for optional passives
        if self.pending_input:
            choice = self.pending_input.get("selection")
            if choice == "YES":
                return execute_passive()
            else:  # "NO" or "SKIP"
                logger.debug(f"   [PASSIVE] Player declined {card.name} passive")
                return StepResult(is_finished=True)

        # Request input from player
        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.CONFIRM_PASSIVE,
                player_id=str(hero.id),
                prompt=self.prompt,
                options=[
                    InputOption(id="YES", text="Yes"),
                    InputOption(id="NO", text="No"),
                ],
                card_id=self.card_id,
                card_name=card.name,
            ),
        )


class LaughStep(GameStep):
    """NebKher's diabolical laugh: a YES/NO confirm routed to the acting hero.

    On YES: stores True under ``output_key``, emits HERO_LAUGHED, and fires
    AFTER_LAUGH passives immediately — before any steps queued after this one
    (the ultimate reacts to the laugh itself, not the rest of the action).
    On NO: nothing happens (downstream steps gate on ``output_key``).
    """

    type: StepType = StepType.LAUGH
    output_key: str = "laughed"
    prompt: str = "Laugh diabolically?"

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        actor_id = str(state.current_actor_id) if state.current_actor_id else None
        if not actor_id:
            return StepResult(is_finished=True)

        if self.pending_input:
            choice = self.pending_input.get("selection")
            if choice != "YES":
                logger.debug("   [LAUGH] Player declined to laugh.")
                return StepResult(is_finished=True)

            context[self.output_key] = True
            from goa2.domain.models.enums import PassiveTrigger

            return StepResult(
                is_finished=True,
                events=[
                    GameEvent(
                        event_type=GameEventType.HERO_LAUGHED,
                        actor_id=actor_id,
                    )
                ],
                new_steps=[CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_LAUGH.value)],
            )

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.CONFIRM_PASSIVE,
                player_id=actor_id,
                prompt=self.prompt,
                options=[
                    InputOption(id="YES", text="Yes"),
                    InputOption(id="NO", text="No"),
                ],
            ),
        )


class MarkPassiveUsedStep(GameStep):
    """Marks a passive ability as used for this turn."""

    type: StepType = StepType.MARK_PASSIVE_USED
    card_id: str
    active_effect_id: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.active_effect_id is not None:
            provider_effect = next(
                (
                    effect
                    for effect in state.active_effects
                    if effect.id == self.active_effect_id and effect.source_card_id == self.card_id
                ),
                None,
            )
            if provider_effect:
                provider_effect.passive_uses_this_turn += 1
            return StepResult(is_finished=True)

        hero = state.get_hero(HeroID(str(state.current_actor_id)))
        if not hero:
            return StepResult(is_finished=True)

        # Find the card (could be in played_cards or ultimate_card)
        card = next((c for c in hero.played_cards if c and c.id == self.card_id), None)
        if not card and hero.ultimate_card and hero.ultimate_card.id == self.card_id:
            card = hero.ultimate_card

        if card:
            card.passive_uses_this_turn += 1
            logger.debug(
                f"   [PASSIVE] {card.name} used ({card.passive_uses_this_turn} time(s) this turn)"
            )

        return StepResult(is_finished=True)
