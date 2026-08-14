"""Attack sequence, combat resolution, defeat, respawn, and lane push steps."""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import Field, ValidationError

from goa2.domain.board import DEFAULT_LANE_ID
from goa2.domain.events import GameEvent, GameEventType, _hex_dict
from goa2.domain.hex import Hex
from goa2.domain.input import (
    InputOption,
    InputRequestType,
    create_input_request,
    parse_hex_selection,
)
from goa2.domain.models import (
    CardColor,
    GamePhase,
    Hero,
    StatType,
    StepType,
    TargetType,
    TeamColor,
    Token,
    is_hero_unit,
)
from goa2.domain.models.effect import EffectType
from goa2.domain.models.marker import MarkerType
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID, HeroID, UnitID
from goa2.engine.effect_manager import EffectManager
from goa2.engine.filters_base import FilterCondition
from goa2.engine.filters_hex import RangeFilter
from goa2.engine.steps.base import GameStep, StepResult

logger = logging.getLogger(__name__)


class SnapshotAdjacentHeroesStep(GameStep):
    """Capture eligible adjacent heroes when an action selects its target.

    The stored IDs remain valid even if the attack subsequently defeats or
    displaces that target, allowing later riders to use targeting-time
    adjacency rather than recalculating a changed board state.
    """

    type: StepType = StepType.SNAPSHOT_ADJACENT_HEROES
    target_key: str = "target_id"
    output_key: str
    relation: str = "ENEMY"

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        target_id = context.get(self.target_key)
        target_hex = state.get_position(str(target_id)) if target_id else None
        actor = (
            state.get_entity(BoardEntityID(str(state.current_actor_id)))
            if state.current_actor_id
            else None
        )
        actor_team = getattr(actor, "team", None)
        if target_hex is None or actor_team is None:
            context[self.output_key] = []
            return StepResult(is_finished=True)

        from goa2.engine.filters_units import ImmunityFilter
        from goa2.engine.topology import get_topology_service

        topology = get_topology_service()
        hero_ids: list[str] = []
        for candidate_id, candidate_hex in state.entity_locations.items():
            candidate = state.get_entity(candidate_id)
            candidate_team = getattr(candidate, "team", None)
            if not candidate or not is_hero_unit(candidate) or candidate_team is None:
                continue
            if self.relation == "ENEMY" and candidate_team == actor_team:
                continue
            if self.relation == "FRIENDLY" and candidate_team != actor_team:
                continue
            if not topology.are_adjacent(target_hex, candidate_hex, state):
                continue
            if not ImmunityFilter().apply(str(candidate_id), state, context):
                continue
            hero_ids.append(str(candidate_id))

        context[self.output_key] = hero_ids
        return StepResult(is_finished=True)


class AttackSequenceStep(GameStep):
    """
    Composite Step.
    Expands into: Select Target -> Reaction Window -> Defense Effect -> Resolve Combat -> On Block Effect.

    Stores in context for defense effect resolution:
    - attack_is_ranged: True if is_ranged=True
    - attacker_id: The ID of the attacking unit
    - attack_is_basic: True if the attack's source card is GOLD/SILVER

    target_id_key reads a target selected by an earlier step.
    target_output_key stores a target selected by this attack.
    If target_filters is provided, adds those filters to the target selection.
    """

    type: StepType = StepType.ATTACK_SEQUENCE
    damage: int
    range_val: int = 1
    is_ranged: bool = False
    target_id_key: str | None = None  # Optional: Use existing context key instead of selecting
    target_output_key: str | None = None
    target_filters: list[FilterCondition] = Field(
        default_factory=list
    )  # Additional filters for target selection
    damage_bonus_key: str | None = None  # Add int from context to damage
    range_bonus_key: str | None = None  # Add int from context to range_val
    damage_stat_type: StatType | None = None  # Recompute damage after actor binding
    range_stat_type: StatType | None = None  # Recompute range after actor binding
    # Emmitt's Temporal line: the defender blocks with their card+items
    # Initiative instead of Defense. Written to context on every resolve (not
    # only when True) so it can never leak into a later attack this turn.
    defense_uses_initiative: bool = False

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.cards import ResolvePreActionDiscardStep
        from goa2.engine.steps.effects import CheckPassiveAbilitiesStep
        from goa2.engine.steps.movement import ResolvePreActionMovementStep
        from goa2.engine.steps.phases import RestoreActionTypeStep
        from goa2.engine.steps.reactions import (
            ReactionWindowStep,
            ResolveDefenseTextStep,
            ResolveOnBlockEffectStep,
        )
        from goa2.engine.steps.selection import SelectStep
        from goa2.engine.steps.utility import SetActorStep

        if self.should_skip(context):
            return StepResult(is_finished=True)

        if self.target_id_key and self.target_id_key in context:
            key = self.target_id_key
            select_target = False
        elif self.target_output_key:
            key = self.target_output_key
            select_target = True
        elif self.target_id_key:
            return StepResult(is_finished=True, abort_action=self.is_mandatory)
        else:
            key = "victim_id"
            select_target = True

        base_actor_id = str(state.current_actor_id) if state.current_actor_id else None
        board_actor_id = state.resolve_board_actor(base_actor_id) if base_actor_id else None

        effective_damage = self.damage
        effective_range = self.range_val
        if board_actor_id:
            from goa2.engine.stats import get_computed_stat

            if self.damage_stat_type is not None:
                effective_damage = get_computed_stat(
                    state, UnitID(board_actor_id), self.damage_stat_type, self.damage
                )
            if self.range_stat_type is not None:
                effective_range = get_computed_stat(
                    state, UnitID(board_actor_id), self.range_stat_type, self.range_val
                )

        if self.damage_bonus_key:
            effective_damage += int(context.get(self.damage_bonus_key, 0))
        if self.range_bonus_key:
            effective_range += int(context.get(self.range_bonus_key, 0))

        logger.debug(
            f"   [MACRO] Expanding Attack Sequence (Dmg: {effective_damage}, Rng: {effective_range})"
        )

        from goa2.engine.filters_units import TeamFilter

        # Store attack context for defense effect resolution
        context["attack_is_ranged"] = self.is_ranged
        context["attacker_id"] = str(state.current_actor_id) if state.current_actor_id else None
        context["attack_damage"] = effective_damage
        context["defense_uses_initiative"] = self.defense_uses_initiative
        # Snorri's Oath line (and any future basic-attack-gated defense) needs
        # to know if the attack's source card is GOLD/SILVER (basic) or a
        # colored tier (non-basic). Source card = the attacker's
        # current_turn_card, unless a re-perform (Bullet Time, Reload, ...)
        # already pinned the actual acted-upon card via reperforming_card_id
        # — that card can differ from current_turn_card. Written on every
        # resolve, like attack_is_ranged above, so it can't leak between
        # attacks.
        context["attack_is_basic"] = self._resolve_attack_is_basic(state, base_actor_id, context)
        logger.debug(
            f"   [ATTACK SEQ] Set attack_is_ranged={context['attack_is_ranged']}, is_ranged={self.is_ranged}, range_val={effective_range}"
        )

        new_steps: list[GameStep] = []

        if select_target:
            # Base filters + any custom target_filters
            all_filters: list[FilterCondition] = [
                RangeFilter(max_range=effective_range),
                TeamFilter(relation="ENEMY"),
            ]
            all_filters.extend(self.target_filters)

            new_steps.append(
                SelectStep(
                    target_type=TargetType.UNIT,
                    prompt="Select Attack Target",
                    output_key=key,
                    filters=all_filters,
                    is_mandatory=self.is_mandatory,
                )
            )

        from goa2.domain.models.enums import PassiveTrigger as _PT

        new_steps.extend(
            [
                ReactionWindowStep(target_player_key=key),
                SetActorStep(actor_key="defender_id", save_key="_pre_pam_actor"),
                CheckPassiveAbilitiesStep(trigger=_PT.BEFORE_ACTION.value),
                # "Before you perform a primary action" triggers. Blocking with a
                # primary-DEFENSE card IS a primary action; a secondary defense or
                # a PASS is not (gated on is_primary_defense). These fire after the
                # defense card is discarded and before the defense action text.
                # Pre-action movement (Misa).
                ResolvePreActionMovementStep(
                    hero_key="defender_id", active_if_key="is_primary_defense"
                ),
                # Pre-action discard / disruptor (Trinkets). abort_on_defeat is
                # False: a Grid defeat of the defender must not abort the ATTACKER's
                # turn (the defeated defender is handled by the on-board guards in
                # ResolveDefenseTextStep / ResolveCombatStep below).
                ResolvePreActionDiscardStep(
                    hero_key="defender_id",
                    active_if_key="is_primary_defense",
                    abort_on_defeat=False,
                ),
                SetActorStep(actor_key="_pre_pam_actor", save_key="_discard_pam"),
                ResolveDefenseTextStep(),
                ResolveCombatStep(damage=effective_damage, target_key=key),
                ResolveOnBlockEffectStep(),
                RestoreActionTypeStep(),
            ]
        )

        return StepResult(is_finished=True, new_steps=new_steps)

    @staticmethod
    def _resolve_attack_is_basic(
        state: GameState, base_actor_id: str | None, context: dict[str, Any]
    ) -> bool:
        """Is the attack's source card GOLD/SILVER (basic)?

        Source card = the attacker's current_turn_card, unless the context
        carries a performing-card override (reperforming_card_id) pinning a
        different card the attack machinery is actually acting on — see
        PerformPrimaryActionStep/PerformCardActionStep, which set that key
        while re-performing an already-resolved card (Bullet Time, Reload).
        """
        if not base_actor_id:
            return False
        hero = state.get_hero(HeroID(base_actor_id))
        if not hero:
            return False

        source_card = state.get_performing_card(base_actor_id)

        if source_card is None:
            return False
        return source_card.current_color in (CardColor.GOLD, CardColor.SILVER)


class ResolveCombatStep(GameStep):
    """
    Compares Attack vs Defense and applies results.

    Checks context flags from defense effects:
    - defense_invalid: Defense fails entirely (e.g., stop_projectiles vs melee)
    - auto_block: Block succeeds regardless of values (e.g., stop_projectiles vs ranged)
    - ignore_minion_defense: Skip minion modifier calculation (e.g., aspiring_duelist)

    Sets context flag for on_block effects:
    - block_succeeded: True if the attack was blocked
    """

    type: StepType = StepType.RESOLVE_COMBAT
    damage: int  # Base attack value from the card
    target_key: str = "victim_id"

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        target_id = context.get(self.target_key)
        if not target_id:
            logger.debug("[COMBAT] No target selected. Combat cancelled.")
            context["block_succeeded"] = False
            return StepResult(is_finished=True)

        # Target already left the board (e.g. a Trinkets disruptor defeated the
        # defender during the reaction window). There is no combat to resolve and
        # defeating again would double-credit the kill.
        if BoardEntityID(str(target_id)) not in state.entity_locations:
            logger.debug(f"[COMBAT] Target {target_id} is off-board. Combat cancelled.")
            context["block_succeeded"] = False
            return StepResult(is_finished=True)

        # Store under a standard key so passives can reference the combat target
        # regardless of what custom key the effect used (e.g. "blink_victim")
        context["last_combat_target"] = target_id

        defense_card_val = context.get("defense_value")
        attack_val = self.damage
        actor_id = state.current_actor_id

        def _combat_event(outcome: str, defense: int | None = None, modifier: int = 0) -> GameEvent:
            return GameEvent(
                event_type=GameEventType.COMBAT_RESOLVED,
                actor_id=str(actor_id) if actor_id else None,
                target_id=target_id,
                metadata={
                    "attack_value": attack_val,
                    "defense_value": defense,
                    "modifier_value": modifier,
                    "outcome": outcome,
                },
            )

        logger.debug(
            f"   [COMBAT] Checking context flags: auto_block={context.get('auto_block')}, defense_invalid={context.get('defense_invalid')}"
        )

        # Check for defense_invalid (e.g., stop_projectiles vs melee attack)
        if context.get("defense_invalid"):
            logger.debug(
                "   [COMBAT] Defense is invalid (conditions not met) - treating as no defense."
            )
            context["block_succeeded"] = False
            return StepResult(
                is_finished=True,
                new_steps=[DefeatUnitStep(victim_id=target_id, killer_id=actor_id)],
                events=[_combat_event("DEFEATED")],
            )

        # Check for auto_block (e.g., stop_projectiles vs ranged attack)
        if context.get("auto_block"):
            logger.debug(f"   [COMBAT] Auto-block triggered! {target_id} is safe.")
            context["block_succeeded"] = True
            return StepResult(
                is_finished=True,
                events=[_combat_event("BLOCKED")],
            )

        # Calculate Passive Modifiers (unless ignored by defense effect)
        from goa2.engine.stats import calculate_minion_defense_modifier

        mod_val: int
        if context.get("ignore_minion_defense"):
            mod_val = 0
            logger.debug("   [COMBAT] Ignoring minion defense modifiers (effect active).")
        else:
            cached_mod = context.get("minion_defense_modifier")
            if cached_mod is None:
                mod_val = calculate_minion_defense_modifier(state, target_id)
            else:
                mod_val = cached_mod
        if defense_card_val is None:
            # No defense played (minion or passed) - unit is defeated
            logger.debug(f"   [RESULT] No defense! {target_id} is DEFEATED!")
            context["block_succeeded"] = False
            return StepResult(
                is_finished=True,
                new_steps=[DefeatUnitStep(victim_id=target_id, killer_id=actor_id)],
                events=[_combat_event("DEFEATED")],
            )
        # Card-granted bonuses ("+2 Defense if...") are Defense-stat bonuses, so
        # they contribute nothing when the block is made with a different stat.
        # This keeps them consistent with Defense auras, which already drop out
        # because get_computed_stat looks up modifiers by the stat in use.
        defense_bonus = (
            0 if context.get("defense_uses_initiative") else int(context.get("defense_bonus", 0))
        )
        total_defense = defense_card_val + defense_bonus + mod_val

        logger.debug(
            f"   [COMBAT] Attack ({attack_val}) vs Defense "
            f"({defense_card_val} Card + {defense_bonus} Bonus + {mod_val} Mod = {total_defense})"
        )

        if total_defense >= attack_val:
            logger.debug(f"   [RESULT] Attack BLOCKED! {target_id} is safe.")
            context["block_succeeded"] = True
            return StepResult(
                is_finished=True,
                events=[_combat_event("BLOCKED", defense_card_val, mod_val)],
            )
        else:
            logger.debug(f"   [RESULT] Attack HITS! {target_id} is DEFEATED!")
            context["block_succeeded"] = False
            return StepResult(
                is_finished=True,
                new_steps=[DefeatUnitStep(victim_id=target_id, killer_id=actor_id)],
                events=[_combat_event("DEFEATED", defense_card_val, mod_val)],
            )


class RemoveUnitStep(GameStep):
    """
    Purely removes a unit from the board.
    Does NOT grant rewards. Used by 'Remove' effects and as a sub-step of Defeat.
    """

    type: StepType = StepType.REMOVE_UNIT
    unit_id: str | None = None
    unit_key: str | None = None  # Read unit ID from context

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        target_id = self.unit_id
        if self.unit_key:
            target_id = context.get(self.unit_key)
        if not target_id:
            return StepResult(is_finished=True)

        logger.debug(f"   [LOGIC] Removing {target_id} from board.")
        from_hex = state.entity_locations.get(BoardEntityID(target_id))
        state.remove_unit(UnitID(target_id))
        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.UNIT_REMOVED,
                    target_id=target_id,
                    from_hex=_hex_dict(from_hex),
                )
            ],
        )


class DefeatUnitStep(GameStep):
    """
    Processes the defeat of a unit (Combat/Skill Kill):
    1. Awards Gold (Killer + Assists).
    2. Updates Life Counters (if Hero).
    3. Returns markers from/to the defeated unit.
    4. Spawns RemoveUnitStep.
    """

    type: StepType = StepType.DEFEAT_UNIT
    victim_id: str | None = None  # Direct ID
    victim_key: str | None = None  # Context key for victim ID
    killer_id: str | None = None
    assist_multiplier: int = 1  # Multiplier for assist coins (e.g. 3 for Glorious Triumph)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        # Resolve victim_id from key if needed
        actual_victim_id = self.victim_id
        if not actual_victim_id and self.victim_key:
            actual_victim_id = context.get(self.victim_key)

        if not actual_victim_id:
            return StepResult(is_finished=True)  # Nothing to defeat

        logger.debug(f"   [DEATH] Processing Defeat of {actual_victim_id}...")

        victim = state.get_unit(UnitID(actual_victim_id))
        if not victim:
            raise ValueError(f"Cannot defeat unknown unit: {actual_victim_id}")

        from goa2.domain.models.unit import HeroPiece

        if (
            isinstance(victim, HeroPiece)
            and BoardEntityID(str(actual_victim_id)) not in state.entity_locations
        ):
            logger.debug(
                "   [DEATH] Skipping stale defeat of off-board piece %s.", actual_victim_id
            )
            return StepResult(is_finished=True)
        if (
            isinstance(victim, Hero)
            and victim.is_multi_piece
            and not state.has_board_presence(str(victim.id))
        ):
            logger.debug("   [DEATH] Skipping stale defeat of off-board hero %s.", actual_victim_id)
            return StepResult(is_finished=True)

        # Multi-piece heroes: defeating ANY piece defeats the hero.
        # Translate the victim to the owning Hero for all player-level
        # consequences, and remove every on-board piece at the end.
        removal_ids = [actual_victim_id]
        if isinstance(victim, HeroPiece):
            owner = state.get_hero(HeroID(actual_victim_id))
            if owner is None:
                raise ValueError(f"HeroPiece {actual_victim_id} has no owner hero")
            removal_ids = state.get_piece_ids(str(owner.id))
            victim = owner
            actual_victim_id = str(owner.id)
            logger.debug(
                f"   [DEATH] Piece defeat → hero defeat of {actual_victim_id}; "
                f"removing pieces {removal_ids}"
            )

        killer = state.get_unit(UnitID(self.killer_id)) if self.killer_id else None

        if hasattr(victim, "value") and self._has_minion_protection(
            state, actual_victim_id, victim
        ):
            # The minion is under at least one protection. Defer the outcome
            # (kill coins, UNIT_DEFEATED, removal) to CheckMinionProtectionStep,
            # which awards by which protection actually fires.
            return StepResult(
                is_finished=True,
                new_steps=[
                    CheckMinionProtectionStep(minion_id=actual_victim_id, killer_id=self.killer_id)
                ],
            )

        events: list[GameEvent] = [
            GameEvent(
                event_type=GameEventType.UNIT_DEFEATED,
                actor_id=self.killer_id,
                target_id=actual_victim_id,
            )
        ]

        # Check for bounty marker BEFORE markers are returned. The marker may
        # sit on a hero piece; it pays out for the owning hero's defeat.
        bounty = state.markers.get(MarkerType.BOUNTY)
        has_bounty = (
            bounty is not None
            and bounty.is_placed
            and bounty.target_id is not None
            and state.marker_target_belongs_to_hero(bounty.target_id, actual_victim_id)
        )

        # Return markers from the defeated hero
        markers_from = state.return_markers_from_hero(actual_victim_id)
        if markers_from:
            logger.debug(
                f"   [DEATH] Returned {len(markers_from)} marker(s) from defeated {actual_victim_id}"
            )

        # Markers this hero PLACED are deliberately left alone: they belong to
        # the heroes carrying them and stay until end of round (or until that
        # hero is defeated). Bain's Bounty still costs its extra life counter
        # after Bain is gone.

        # Cancel all active effects created by the defeated unit
        EffectManager.expire_by_source(state, actual_victim_id)

        # If the defeated hero has an unresolved card, resolve it without action
        if hasattr(victim, "current_turn_card") and victim.current_turn_card:
            hero = cast(Hero, victim)
            hero.resolve_current_card()

        # Remove from unresolved pool so they don't get another turn this round
        if HeroID(actual_victim_id) in state.unresolved_hero_ids:
            state.unresolved_hero_ids.remove(HeroID(actual_victim_id))

        # Track hero defeats for round-scoped effects (e.g., War Drummer)
        if hasattr(victim, "level"):
            hero_id = HeroID(actual_victim_id)
            if hero_id not in state.heroes_defeated_this_round:
                state.heroes_defeated_this_round.append(hero_id)

        if hasattr(victim, "level"):  # Is Hero
            level = getattr(victim, "level", 1)

            # Level: (Kill Reward, Assist Reward, Death Penalty)
            rewards_table = {
                1: (1, 1, 1),
                2: (2, 1, 1),
                3: (3, 1, 1),
                4: (4, 2, 2),
                5: (5, 2, 2),
                6: (6, 2, 2),
                7: (7, 3, 3),
                8: (8, 3, 3),
            }
            kill_gold, assist_gold, penalty_counters = rewards_table.get(level, (level, 1, 1))

            # Bounty marker: defeated hero spends 1 additional life counter
            if has_bounty:
                penalty_counters += 1
                logger.debug(
                    f"   [BOUNTY] {actual_victim_id} has Bounty marker — +1 Life Counter penalty."
                )

            if killer and hasattr(killer, "gold"):
                killer.gold += kill_gold
                logger.debug(f"   [ECONOMY] Killer {killer.id} gains {kill_gold} Gold.")
                events.append(
                    GameEvent(
                        event_type=GameEventType.GOLD_GAINED,
                        actor_id=killer.id,
                        metadata={"amount": kill_gold, "reason": "kill"},
                    )
                )

            if killer and hasattr(killer, "team"):
                killer_team_color = getattr(killer, "team", None)
                if killer_team_color and killer_team_color in state.teams:
                    killer_team = state.teams[killer_team_color]
                    if killer_team:
                        for ally in killer_team.heroes:
                            if ally.id != killer.id:
                                actual_assist = assist_gold * self.assist_multiplier
                                ally.gold += actual_assist
                                logger.debug(
                                    f"   [ECONOMY] Assist: {ally.id} gains {actual_assist} Gold."
                                )
                                events.append(
                                    GameEvent(
                                        event_type=GameEventType.GOLD_GAINED,
                                        actor_id=ally.id,
                                        metadata={
                                            "amount": actual_assist,
                                            "reason": "assist",
                                        },
                                    )
                                )

            if hasattr(victim, "team"):
                victim_team_color = getattr(victim, "team", None)
                if victim_team_color and victim_team_color in state.teams:
                    victim_team = state.teams[victim_team_color]
                    if victim_team:
                        victim_team.life_counters = max(
                            0, victim_team.life_counters - penalty_counters
                        )
                        logger.debug(
                            f"   [SCORE] Team {victim_team_color.name} loses {penalty_counters} Life Counter(s). Remaining: {victim_team.life_counters}"
                        )
                        events.append(
                            GameEvent(
                                event_type=GameEventType.LIFE_COUNTER_CHANGED,
                                metadata={
                                    "team": victim_team_color.name,
                                    "change": -penalty_counters,
                                    "remaining": victim_team.life_counters,
                                },
                            )
                        )

                        if victim_team.life_counters == 0:
                            logger.debug(
                                f"   [GAME OVER] Team {victim_team_color.name} has 0 Life Counters! ANNIHILATION."
                            )
                            winning_team = (
                                TeamColor.BLUE
                                if victim_team_color == TeamColor.RED
                                else TeamColor.RED
                            )
                            return StepResult(
                                is_finished=True,
                                new_steps=[
                                    *[RemoveUnitStep(unit_id=rid) for rid in removal_ids],
                                    TriggerGameOverStep(
                                        winner=winning_team, condition="ANNIHILATION"
                                    ),
                                ],
                                events=events,
                            )

        elif hasattr(victim, "value"):  # Is Minion (unprotected — protected ones
            # are routed to CheckMinionProtectionStep above before reaching here)
            # Record the genuine defeat so post-attack passives (e.g. Reign of
            # Winter) can distinguish a real defeat from a totem save.
            context["last_defeated_minion_id"] = actual_victim_id
            reward = victim.value
            logger.debug(f"   [DEATH] Minion Defeated! Killer gains {reward} Gold.")
            if killer and hasattr(killer, "gold"):
                killer.gold += reward
                events.append(
                    GameEvent(
                        event_type=GameEventType.GOLD_GAINED,
                        actor_id=killer.id,
                        metadata={"amount": reward, "reason": "minion_kill"},
                    )
                )

            # Swift's Mark for Death / Hunting Season: pay coin bounties for an
            # enemy minion defeated in radius (minion still on the board here).
            events.extend(EffectManager.award_minion_defeat_bounties(state, actual_victim_id))

        return StepResult(
            is_finished=True,
            new_steps=[RemoveUnitStep(unit_id=rid) for rid in removal_ids],
            events=events,
        )

    def _has_minion_protection(self, state: GameState, minion_id: str, minion: Any) -> bool:
        """Any active MINION_PROTECTION (totem sacrifice or card-discard) covering
        this minion. Such a minion defers its defeat outcome to
        CheckMinionProtectionStep."""
        from goa2.engine.stats import _is_effect_active, is_unit_in_effect_scope

        for effect in state.active_effects:
            if effect.effect_type != EffectType.MINION_PROTECTION:
                continue
            if not _is_effect_active(effect, state):
                continue
            if not is_unit_in_effect_scope(effect, minion_id, state):
                continue
            if (
                effect.protected_minion_types
                and getattr(minion, "type", None) not in effect.protected_minion_types
            ):
                continue
            return True

        return False


class CheckMinionProtectionStep(GameStep):
    """
    Checks if any MINION_PROTECTION effect can save a defeated minion.
    Asks the protecting hero if they want to discard a qualifying card.
    If yes: discard card, emit MINION_PROTECTED event, minion stays.
    If no/skip: try next effect, or push RemoveUnitStep.
    """

    type: StepType = StepType.CHECK_MINION_PROTECTION
    minion_id: str
    killer_id: str | None = None
    tried_effect_ids: list[str] = Field(default_factory=list)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.stats import _is_effect_active, is_unit_in_effect_scope

        minion = state.get_unit(UnitID(self.minion_id))
        if not minion:
            return StepResult(is_finished=True)

        # Collect untried active MINION_PROTECTION effects covering this minion.
        # The mandatory totem sacrifice replaces the prospective defeat FIRST;
        # optional (card-discard) protections like Brogan's are only considered
        # when no totem applies to this minion.
        candidates = [
            e
            for e in state.active_effects
            if e.effect_type == EffectType.MINION_PROTECTION
            and e.id not in self.tried_effect_ids
            and _is_effect_active(e, state)
            and is_unit_in_effect_scope(e, self.minion_id, state)
            and (
                not e.protected_minion_types
                or getattr(minion, "type", None) in e.protected_minion_types
            )
        ]
        candidates.sort(key=lambda e: not e.sacrifice_origin_token)
        protection = candidates[0] if candidates else None

        if not protection:
            # No protection fired — a genuine defeat. Award the kill coins,
            # emit UNIT_DEFEATED, and remove the minion.
            events = self._defeat_events(state, minion, context)
            # Swift's Mark for Death / Hunting Season bounty (minion still on
            # the board here; removal happens in the spawned RemoveUnitStep).
            events.extend(EffectManager.award_minion_defeat_bounties(state, self.minion_id))
            return StepResult(
                is_finished=True,
                new_steps=[RemoveUnitStep(unit_id=self.minion_id)],
                events=events,
            )

        # Check if protector has qualifying cards in hand
        protector = state.get_hero(HeroID(protection.source_id))
        if not protector:
            return self._try_next(protection.id)

        if protection.sacrifice_origin_token:
            token_id = protection.scope.origin_id
            if not token_id:
                return self._try_next(protection.id)
            token = state.misc_entities.get(BoardEntityID(token_id))
            if not isinstance(token, Token):
                return self._try_next(protection.id)

            from goa2.engine.steps.markers import _remove_token_from_board

            from_hex, removed_effects = _remove_token_from_board(state, token_id)
            if not from_hex:
                return self._try_next(protection.id)

            logger.debug(
                f"   [PROTECT] {protector.id}'s Totem protects {self.minion_id} and is removed."
            )
            return StepResult(
                is_finished=True,
                events=[
                    GameEvent(
                        event_type=GameEventType.TOKEN_REMOVED,
                        actor_id=protector.id,
                        target_id=token_id,
                        from_hex=_hex_dict(from_hex),
                        metadata={"effects_removed": removed_effects},
                    ),
                    GameEvent(
                        event_type=GameEventType.MINION_PROTECTED,
                        actor_id=protector.id,
                        target_id=self.minion_id,
                        metadata={
                            "sacrificed_token_id": token_id,
                            "effect_id": protection.id,
                        },
                    ),
                ],
            )

        qualifying_cards = [
            c for c in protector.hand if c.color in protection.allowed_discard_colors
        ]

        if not qualifying_cards:
            logger.debug(f"   [PROTECT] {protector.id} has no qualifying cards for protection.")
            return self._try_next(protection.id)

        # Ask protector if they want to discard
        if self.pending_input is None:
            options = [
                InputOption(
                    id=c.id,
                    text=c.name,
                    metadata={"card_id": c.id, "color": c.color.value if c.color else None},
                )
                for c in qualifying_cards
            ]
            return StepResult(
                is_finished=False,
                requires_input=True,
                input_request=create_input_request(
                    request_type=InputRequestType.SELECT_CARD,
                    prompt=f"Discard a card to protect {self.minion_id}?",
                    options=options,
                    player_id=protector.id,
                    can_skip=True,
                ),
            )

        # Process input
        selection = self.pending_input
        if isinstance(selection, dict):
            selection = selection.get("selected_card_id") or selection.get("selection")

        if selection == "SKIP" or selection is None:
            logger.debug(f"   [PROTECT] {protector.id} declined to protect {self.minion_id}.")
            return self._try_next(protection.id)

        # Find the selected card and discard it through the single discard entry
        # point. The minion-protection cost IS a discard, so routing it through
        # DiscardCardStep keeps discard handling (effect expiry, AFTER_CARD_DISCARD
        # trigger) consistent with every other discard.
        card_to_discard = next((c for c in qualifying_cards if c.id == selection), None)
        if not card_to_discard:
            return self._try_next(protection.id)

        logger.debug(
            f"   [PROTECT] {protector.id} discards {card_to_discard.name} to protect {self.minion_id}!"
        )

        from goa2.engine.steps.cards import DiscardCardStep

        # Card-discard protection (e.g. Brogan): the minion is still defeated for
        # scoring purposes — the killer keeps the coins and UNIT_DEFEATED fires —
        # it simply isn't removed from the board.
        return StepResult(
            is_finished=True,
            new_steps=[DiscardCardStep(card_id=card_to_discard.id, hero_id=protector.id)],
            events=[
                *self._defeat_events(state, minion, context),
                GameEvent(
                    event_type=GameEventType.MINION_PROTECTED,
                    actor_id=protector.id,
                    target_id=self.minion_id,
                    metadata={
                        "discarded_card_id": card_to_discard.id,
                        "effect_id": protection.id,
                    },
                ),
            ],
        )

    def _defeat_events(
        self, state: GameState, minion: Any, context: dict[str, Any]
    ) -> list[GameEvent]:
        """UNIT_DEFEATED plus the killer's kill coins, for a genuine defeat or a
        card-discard save (totem sacrifices deliberately do NOT call this).

        Records the defeat in context so post-attack passives (Reign of Winter)
        fire for a defeated-but-not-removed minion as well as a real kill."""
        context["last_defeated_minion_id"] = self.minion_id
        events: list[GameEvent] = [
            GameEvent(
                event_type=GameEventType.UNIT_DEFEATED,
                actor_id=self.killer_id,
                target_id=self.minion_id,
            )
        ]
        killer = state.get_unit(UnitID(self.killer_id)) if self.killer_id else None
        if killer and hasattr(killer, "gold"):
            reward = minion.value
            killer.gold += reward
            events.append(
                GameEvent(
                    event_type=GameEventType.GOLD_GAINED,
                    actor_id=killer.id,
                    metadata={"amount": reward, "reason": "minion_kill"},
                )
            )
        return events

    def _try_next(self, effect_id: str) -> StepResult:
        """Try the next protection effect."""
        return StepResult(
            is_finished=True,
            new_steps=[
                CheckMinionProtectionStep(
                    minion_id=self.minion_id,
                    killer_id=self.killer_id,
                    tried_effect_ids=[*self.tried_effect_ids, effect_id],
                )
            ],
        )


def _token_on_hex(state: GameState, hex_pos: Hex) -> Token | None:
    """The Token occupying `hex_pos`, if the hex is blocked only by a token.

    Returns None for empty hexes, terrain, and hexes held by a unit.
    """
    tile = state.board.get_tile(hex_pos)
    if not tile or tile.is_terrain or not tile.occupant_id:
        return None
    occupant = state.get_entity(BoardEntityID(str(tile.occupant_id)))
    return occupant if isinstance(occupant, Token) else None


class RespawnHeroStep(GameStep):
    """
    Handles the Hero Respawn choice.
    If Hero is defeated, requests player input: Respawn or Pass.
    """

    type: StepType = StepType.RESPAWN_HERO
    hero_id: str

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        hero = state.get_hero(HeroID(self.hero_id))
        if not hero:
            return StepResult(is_finished=True)

        # Only respawn if not on board (any piece counts for multi-piece heroes)
        if state.has_board_presence(self.hero_id):
            return StepResult(is_finished=True)

        # PASS is handled before computing spawn hexes: declining to respawn is
        # always valid, even when no spawn hex is available.
        if self.pending_input and self.pending_input.get("selection") == "PASS":
            logger.debug(f"   [RESPAWN] {self.hero_id} chose NOT to respawn.")
            context["skipped_respawn"] = True
            return StepResult(is_finished=True)

        # Find hero spawn points for this team that aren't obstacles. Computed up
        # front so a submitted hex can be validated against the offered set
        # before placing the hero (a player must not respawn just anywhere).
        valid_hexes = []
        team_spawn_hexes = []
        for sp in state.board.spawn_points:
            if sp.is_hero_spawn and sp.team == hero.team:
                team_spawn_hexes.append(sp.location)
                blocked = state.validator.is_obstacle_for_actor(state, sp.location, self.hero_id)
                # A token never blocks a marked respawn space: the hero
                # respawns onto it and clears the token.
                if blocked and _token_on_hex(state, sp.location) is not None:
                    blocked = False
                if not blocked:
                    valid_hexes.append(sp.location)

        # Fallback: BFS from spawn points to find nearest non-obstacle hex
        if not valid_hexes and team_spawn_hexes:
            from goa2.engine.map_logic import find_nearest_empty_hexes

            for spawn_hex in team_spawn_hexes:
                zone_id = state.board.get_zone_for_hex(spawn_hex)
                if zone_id:
                    candidates = find_nearest_empty_hexes(state, spawn_hex, zone_id)
                    if candidates:
                        valid_hexes.extend(candidates)
                        break

        if not valid_hexes:
            logger.debug(f"   [RESPAWN] No empty spawn points for {self.hero_id}!")
            return StepResult(is_finished=True)

        def _hex_request() -> StepResult:
            return StepResult(
                requires_input=True,
                input_request=create_input_request(
                    request_type=InputRequestType.CHOOSE_RESPAWN_HEX,
                    player_id=self.hero_id,
                    prompt=f"Select spawn location for {self.hero_id}",
                    options=valid_hexes,
                    valid_hexes=valid_hexes,  # Pass raw Hex objects
                ),
            )

        if self.pending_input:
            selection = self.pending_input.get("selection")
            selected_hex_dict = selection if isinstance(selection, dict) else None
            if selected_hex_dict:
                try:
                    selected_hex = Hex(**selected_hex_dict)
                except (TypeError, ValueError, ValidationError):
                    selected_hex = None
                # Only accept a hex the player was actually offered; anything
                # else (malformed, or an arbitrary board hex) re-opens the hex
                # prompt rather than placing the hero somewhere illegal.
                if selected_hex is None or selected_hex not in valid_hexes:
                    logger.debug(f"   [RESPAWN] Rejected respawn hex {selection!r}; re-requesting.")
                    return _hex_request()
                logger.debug(f"   [RESPAWN] {self.hero_id} respawning at {selected_hex}")
                events: list[GameEvent] = []
                cleared = _token_on_hex(state, selected_hex)
                if cleared is not None:
                    from goa2.engine.steps.markers import _remove_token_from_board

                    from_hex, _ = _remove_token_from_board(state, str(cleared.id))
                    logger.debug(f"   [RESPAWN] Cleared token {cleared.id} from {selected_hex}")
                    events.append(
                        GameEvent(
                            event_type=GameEventType.TOKEN_REMOVED,
                            actor_id=self.hero_id,
                            target_id=str(cleared.id),
                            from_hex=from_hex.model_dump() if from_hex else None,
                            metadata={"reason": "respawn"},
                        )
                    )
                if hero.is_multi_piece:
                    from goa2.engine.hero_pieces import pieces_in_supply

                    supply = pieces_in_supply(state, hero)
                    state.place_entity(BoardEntityID(supply[0]), selected_hex)
                else:
                    state.move_unit(UnitID(self.hero_id), selected_hex)
                return StepResult(is_finished=True, events=events)

        # If user already chose RESPAWN but hasn't picked hex yet, show hexes
        if self.pending_input and self.pending_input.get("selection") == "RESPAWN":
            return _hex_request()

        # Otherwise, show YES/NO prompt first
        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.CHOOSE_RESPAWN,
                player_id=self.hero_id,
                prompt=f"Hero {self.hero_id} is defeated. Respawn at an empty spawn point?",
                options=["RESPAWN", "PASS"],
                valid_hexes=valid_hexes,  # Pass raw Hex objects
            ),
        )


class RespawnMinionStep(GameStep):
    """
    Respawns a minion of a certain type/team in its lane's Battle Zone.
    """

    type: StepType = StepType.RESPAWN_MINION
    team: TeamColor
    minion_type: Any  # MinionType enum
    lane_id: str = DEFAULT_LANE_ID

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        zone_id = state.battle_zone_for_lane(self.lane_id)
        if not zone_id:
            return StepResult(is_finished=True)

        zone = state.board.zones.get(zone_id)
        if not zone:
            return StepResult(is_finished=True)

        target_minion = None
        # Check if minion exists in team roster but not on board (limbo).
        # Minions respawn only in the lane they are bound to.
        team_obj = state.teams.get(self.team) if self.team else None
        if not team_obj:
            return StepResult(is_finished=True)

        for m in team_obj.minions:
            if (
                m.type == self.minion_type
                and m.lane_id == self.lane_id
                and m.id not in state.entity_locations
            ):
                target_minion = m
                break

        if not target_minion:
            # Safely handle team name if team exists, otherwise default
            team_name = self.team.name if self.team else "Unknown"
            logger.debug(f"   [RESPAWN] No available {team_name} {self.minion_type} to respawn.")
            return StepResult(is_finished=True)

        if self.pending_input:
            selected_hex_dict = self.pending_input.get("selection")
            if isinstance(selected_hex_dict, dict):
                selected_hex = Hex(**selected_hex_dict)
                tile = state.board.get_tile(selected_hex)
                if tile and tile.is_occupied:
                    logger.debug(
                        f"   [ERROR] Cannot respawn {self.minion_type} at {selected_hex}. Occupied."
                    )
                    return StepResult(is_finished=True)

                state.move_unit(UnitID(target_minion.id), selected_hex)
                logger.debug(f"   [RESPAWN] Respawned {target_minion.id} at {selected_hex}")
                return StepResult(is_finished=True)

        valid_spaces = [h for h in zone.hexes if not state.board.get_tile(h).is_occupied]
        if not valid_spaces:
            return StepResult(is_finished=True)

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_HEX,
                player_id=(str(state.current_actor_id) if state.current_actor_id else "system"),
                prompt=f"Select space to respawn {self.minion_type}.",
                options=valid_spaces,
            ),
        )


_LANE_RESPAWN_HEX_KEY = "lane_respawn_hex"


class RespawnMinionAtHexStep(GameStep):
    """
    Respawns a minion at a hex chosen from filtered candidates.

    Two modes:
    - Legacy (unit_key set): the minion was chosen upstream; this step only
      picks the hex.
    - Lane-bound (lane_bound=True): hex-first. The player picks a spawn
      hex (only hexes whose lane has limbo supply are offered), then the
      minion comes from THAT lane's limbo supply (auto-picked when only
      one type is available). Hexes outside any lane (e.g. Tide of
      Darkness spawn points) fall back to the full limbo supply.

    Emits UNIT_PLACED event.
    """

    type: StepType = StepType.RESPAWN_MINION_AT_HEX
    team: TeamColor
    unit_key: str | None = None  # Context key containing minion ID (legacy mode)
    lane_bound: bool = False
    hex_filters: list[FilterCondition] = Field(default_factory=list)

    def _hex_passes_filters(self, state: GameState, context: dict[str, Any], h: Hex) -> bool:
        """A submitted respawn hex is legal only if it is on the board, empty,
        and satisfies every placement filter — the same constraints used to
        build the offered hex list. This stops a client from respawning a minion
        on an unoffered hex (out of range, off the spawn points, etc.)."""
        tile = state.board.get_tile(h)
        if tile is None or tile.is_occupied:
            return False
        return all(f.apply(h, state, context) for f in self.hex_filters)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)
        if self.lane_bound:
            return self._resolve_lane_bound(state, context)
        return self._resolve_legacy(state, context)

    # ------------------------------------------------------------------
    # Lane-bound mode
    # ------------------------------------------------------------------
    def _zone_type_headroom(
        self, state: GameState, team_obj: Any, zone_id: str | None
    ) -> dict[Any, int] | None:
        """Per minion-type respawn headroom in `zone_id` for this team:
        `#spawn points (type, team) - #miniatures (type, team) present`.

        Rulebook: a minion may be respawned only where that Battle Zone has
        MORE spawn points of its (type, color), empty or not, than miniatures
        of it present. Returns None for hexes with no zone (e.g. Tide of
        Darkness spaces), which are uncapped.
        """
        zone = state.board.zones.get(zone_id) if zone_id else None
        if not zone:
            return None
        spawn_count: dict[Any, int] = {}
        for sp in zone.spawn_points:
            if sp.is_minion_spawn and sp.team == self.team and sp.minion_type is not None:
                spawn_count[sp.minion_type] = spawn_count.get(sp.minion_type, 0) + 1
        present: dict[Any, int] = {}
        for m in team_obj.minions:
            loc = state.get_position(str(m.id))
            if loc is not None and loc in zone.hexes:
                present[m.type] = present.get(m.type, 0) + 1
        return {
            t: spawn_count.get(t, 0) - present.get(t, 0) for t in set(spawn_count) | set(present)
        }

    def _limbo_minions_for_hex(self, state: GameState, team_obj: Any, h: Hex) -> list[Any]:
        """Limbo minions legal at hex h: bound to the hex's lane, one per
        type, and only for types whose Battle Zone still has spawn-point
        headroom (see `_zone_type_headroom`). Hexes outside any lane fall
        back to all limbo minions."""
        tile = state.board.get_tile(h)
        lane_id = state.lane_of_zone(tile.zone_id) if tile and tile.zone_id else None
        headroom = self._zone_type_headroom(state, team_obj, tile.zone_id if tile else None)
        seen: set[Any] = set()
        result = []
        for m in team_obj.minions:
            if state.has_board_presence(str(m.id)):
                continue
            if lane_id is not None and m.lane_id != lane_id:
                continue
            if m.type in seen:
                continue
            if headroom is not None and headroom.get(m.type, 0) <= 0:
                continue  # zone already at spawn-point capacity for this type
            seen.add(m.type)
            result.append(m)
        return result

    def _place_minion(self, state: GameState, minion: Any, selected_hex: Hex) -> StepResult:
        tile = state.board.get_tile(selected_hex)
        if tile and tile.is_occupied:
            logger.debug(f"   [ERROR] Cannot respawn {minion.id} at {selected_hex}. Occupied.")
            return StepResult(is_finished=True)
        state.move_unit(UnitID(minion.id), selected_hex)
        logger.debug(f"   [RESPAWN] Respawned {minion.id} at {selected_hex}")
        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.UNIT_PLACED,
                    actor_id=str(minion.id),
                    from_hex=None,
                    to_hex=_hex_dict(selected_hex),
                )
            ],
        )

    def _resolve_lane_bound(self, state: GameState, context: dict[str, Any]) -> StepResult:
        team_obj = state.teams.get(self.team)
        if not team_obj:
            return StepResult(is_finished=True)

        stored_hex = context.get(_LANE_RESPAWN_HEX_KEY)

        if self.pending_input:
            selection = self.pending_input.get("selection")

            # Round 1 answer: the hex.
            if stored_hex is None and isinstance(selection, dict):
                try:
                    hex_obj = Hex(**selection)
                except (TypeError, ValueError, ValidationError):
                    hex_obj = None
                # Only accept a hex that satisfies the placement filters (the
                # same set offered in Round 0); anything else re-requests rather
                # than placing the minion on an unoffered/illegal hex.
                if hex_obj is None or not self._hex_passes_filters(state, context, hex_obj):
                    logger.debug("   [RESPAWN] Rejected respawn hex %r; re-requesting.", selection)
                    self.pending_input = None
                else:
                    candidates = self._limbo_minions_for_hex(state, team_obj, hex_obj)
                    if not candidates:
                        logger.debug("   [RESPAWN] No limbo supply for chosen hex's lane.")
                        return StepResult(is_finished=True)
                    if len(candidates) == 1:
                        return self._place_minion(state, candidates[0], hex_obj)
                    context[_LANE_RESPAWN_HEX_KEY] = selection
                    return StepResult(
                        requires_input=True,
                        input_request=create_input_request(
                            request_type=InputRequestType.SELECT_OPTION,
                            player_id=(
                                str(state.current_actor_id) if state.current_actor_id else "system"
                            ),
                            prompt="Choose a minion to respawn.",
                            options=[
                                {"id": str(m.id), "text": f"{m.type.value} Minion"}
                                for m in candidates
                            ],
                        ),
                    )

            # Round 2 answer: the minion.
            if stored_hex is not None and isinstance(selection, str):
                hex_obj = Hex(**stored_hex)
                context.pop(_LANE_RESPAWN_HEX_KEY, None)
                minion = next(
                    (
                        m
                        for m in self._limbo_minions_for_hex(state, team_obj, hex_obj)
                        if str(m.id) == selection
                    ),
                    None,
                )
                if not minion:
                    logger.debug(f"   [RESPAWN] Minion {selection} not available.")
                    return StepResult(is_finished=True)
                return self._place_minion(state, minion, hex_obj)

        # Round 0: offer spawn hexes whose lane has available supply.
        valid_hexes = []
        for h, tile in state.board.tiles.items():
            if tile.is_occupied:
                continue
            if not all(f.apply(h, state, context) for f in self.hex_filters):
                continue
            if not self._limbo_minions_for_hex(state, team_obj, h):
                continue
            valid_hexes.append(h)

        if not valid_hexes:
            logger.debug("   [RESPAWN] No valid lane-bound respawn hexes.")
            return StepResult(is_finished=True)

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_HEX,
                player_id=(str(state.current_actor_id) if state.current_actor_id else "system"),
                prompt="Select space to respawn a minion.",
                options=valid_hexes,
            ),
        )

    # ------------------------------------------------------------------
    # Legacy mode (unit_key)
    # ------------------------------------------------------------------
    def _resolve_legacy(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if not self.unit_key:
            logger.debug("   [RESPAWN] No unit_key configured.")
            return StepResult(is_finished=True)

        # Read minion ID from context
        minion_id = context.get(self.unit_key)
        if not minion_id:
            logger.debug(f"   [RESPAWN] No minion ID in context['{self.unit_key}'].")
            return StepResult(is_finished=True)

        # Verify minion exists and is in limbo
        team_obj = state.teams.get(self.team)
        if not team_obj:
            return StepResult(is_finished=True)

        target_minion = None
        for m in team_obj.minions:
            if m.id == minion_id and m.id not in state.entity_locations:
                target_minion = m
                break

        if not target_minion:
            logger.debug(f"   [RESPAWN] Minion {minion_id} not found in limbo.")
            return StepResult(is_finished=True)

        # Handle input
        if self.pending_input:
            selected_hex_dict = self.pending_input.get("selection")
            if isinstance(selected_hex_dict, dict):
                try:
                    selected_hex = Hex(**selected_hex_dict)
                except (TypeError, ValueError, ValidationError):
                    selected_hex = None
                # Only accept a hex that satisfies the placement filters (occupancy
                # is included), so a client can't respawn on an unoffered hex.
                if selected_hex is not None and self._hex_passes_filters(
                    state, context, selected_hex
                ):
                    state.move_unit(UnitID(target_minion.id), selected_hex)
                    logger.debug(f"   [RESPAWN] Respawned {target_minion.id} at {selected_hex}")
                    return StepResult(
                        is_finished=True,
                        events=[
                            GameEvent(
                                event_type=GameEventType.UNIT_PLACED,
                                actor_id=str(target_minion.id),
                                from_hex=None,
                                to_hex=_hex_dict(selected_hex),
                            )
                        ],
                    )
                logger.debug(
                    "   [RESPAWN] Rejected respawn hex %r; re-requesting.", selected_hex_dict
                )

        # Collect valid hexes using filters
        valid_hexes = []
        for h, tile in state.board.tiles.items():
            if tile.is_occupied:
                continue
            is_valid = True
            for f in self.hex_filters:
                if not f.apply(h, state, context):
                    is_valid = False
                    break
            if is_valid:
                valid_hexes.append(h)

        if not valid_hexes:
            logger.debug("   [RESPAWN] No valid hexes for respawn.")
            return StepResult(is_finished=True)

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_HEX,
                player_id=(str(state.current_actor_id) if state.current_actor_id else "system"),
                prompt=f"Select space to respawn {target_minion.id}.",
                options=valid_hexes,
            ),
        )


def _wipe_minions_in_zone(state: GameState, zone_id: str) -> None:
    """Remove every minion physically inside the zone (push wipe)."""
    zone = state.board.zones.get(zone_id)
    if not zone:
        return
    to_remove = []
    for uid, loc in state.unit_locations.items():
        if loc in zone.hexes:
            unit = state.get_unit(UnitID(uid))
            if hasattr(unit, "type") and hasattr(unit, "value"):  # Duck typing Minion
                to_remove.append(uid)
    for uid in to_remove:
        state.remove_unit(uid)
        logger.debug(f"   [PUSH] Wiped {uid} from zone {zone_id}.")


def _respawn_minions_at_spawn_points(
    state: GameState, lane_id: str, zone_id: str
) -> list[tuple[str, Hex]]:
    """
    Spawn limbo minions bound to `lane_id` at the zone's minion spawn
    points (both teams). Returns blocked spawns as pending displacements.
    """
    from goa2.engine.steps.markers import _remove_token_from_board

    zone = state.board.zones.get(zone_id)
    pending_displacements: list[tuple[str, Hex]] = []
    if not zone:
        return pending_displacements

    def _next_candidate(sp: Any, reserved: set[str]) -> Any:
        team = state.teams.get(sp.team)
        if not team:
            return None
        return next(
            (
                m
                for m in team.minions
                if m.type == sp.minion_type
                and m.lane_id == lane_id
                and m.id not in state.unit_locations
                and str(m.id) not in reserved
            ),
            None,
        )

    # Two passes over the spawn points so a minion is assigned to at most one
    # spawn point. Pass 1 places minions at directly-spawnable points (empty,
    # or occupied only by a removable token), which takes them out of limbo.
    # Pass 2 then queues displacements for genuinely blocked points using the
    # minions that remain — never re-selecting one already placed or queued.
    # Doing displacement selection in a single pass would re-pick a
    # queued-but-still-in-limbo minion, double-booking it and stranding another.
    blocked_points: list[Any] = []
    for sp in zone.spawn_points:
        if not sp.is_minion_spawn:
            continue
        tile = state.board.get_tile(sp.location)
        occupant_id = str(tile.occupant_id) if tile and tile.occupant_id else None
        occupant = state.misc_entities.get(BoardEntityID(occupant_id)) if occupant_id else None

        if tile and not tile.is_occupied:
            candidate = _next_candidate(sp, set())
            if candidate:
                state.move_unit(candidate.id, sp.location)
                logger.debug(f"   [PUSH] Spawning {candidate.id} at {sp.location}")
        elif isinstance(occupant, Token) and occupant_id:
            candidate = _next_candidate(sp, set())
            if candidate:
                _remove_token_from_board(state, occupant_id)
                state.move_unit(candidate.id, sp.location)
                logger.debug(
                    f"   [PUSH] Removed token {occupant_id} and spawned "
                    f"{candidate.id} at {sp.location}"
                )
        else:
            blocked_points.append(sp)

    reserved: set[str] = set()
    for sp in blocked_points:
        candidate = _next_candidate(sp, reserved)
        if not candidate:
            continue
        reserved.add(str(candidate.id))
        logger.debug(f"   [PUSH] Spawn blocked at {sp.location} (Displacement Queued)")
        pending_displacements.append((str(candidate.id), sp.location))
    return pending_displacements


class CheckLanePushStep(GameStep):
    """
    Checks if a Battle Zone meets the condition for a Lane Push (0 minions
    for one team) and decides what happens.

    Single-lane games: spawns a classic LanePushStep (which owns the wave
    counter and endgame rules).

    Multi-lane games: acts as the endgame coordinator. Pushes triggered by
    the same check are simultaneous per the double-lane rules. Outcomes are
    pre-computed BEFORE any mutation, then:
      - throne pushes favoring both teams  -> tie remedy
      - throne push favoring one team      -> LANE_PUSH victory
      - any wave counter at 0 after flips  -> zone-count comparison at
        post-push positions (LAST_PUSH victory, or tie remedy on equal)
      - otherwise                          -> mechanics-only LanePushSteps

    Tie remedy: zones do not move, counters stay flipped, and each
    triggered lane gets a full wipe + spawn-point respawn in its unmoved
    Battle Zone (rulebook: "spawn all minions in the Zones they occupied
    before the push and continue playing").

    With lane_id=None (default) every lane is checked, so existing call
    sites remain correct on multi-lane maps.
    """

    type: StepType = StepType.CHECK_LANE_PUSH
    lane_id: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.map_logic import check_lane_push_trigger

        lanes_to_check = (
            [self.lane_id] if self.lane_id is not None else list(state.battle_zones.keys())
        )

        triggered: list[tuple[str, TeamColor]] = []
        for lane_id in lanes_to_check:
            zone_id = state.battle_zone_for_lane(lane_id)
            if not zone_id:
                continue
            losing_team = check_lane_push_trigger(state, zone_id)
            if losing_team:
                logger.debug(
                    f"   [CHECK] Lane Push Condition Met for {losing_team.name} on {lane_id}"
                )
                triggered.append((lane_id, losing_team))

        if not triggered:
            return StepResult(is_finished=True)

        if len(state.board.lanes) <= 1:
            return StepResult(
                is_finished=True,
                new_steps=[
                    LanePushStep(lane_id=lane_id, losing_team=losing_team)
                    for lane_id, losing_team in triggered
                ],
            )

        return self._resolve_multi_lane(state, triggered)

    def _resolve_multi_lane(
        self, state: GameState, triggered: list[tuple[str, TeamColor]]
    ) -> StepResult:
        from goa2.engine.map_logic import endgame_totals, get_push_target_zone_id

        # Pre-compute every push's outcome before mutating anything.
        outcomes: list[tuple[str, TeamColor, str | None, bool]] = []
        for lane_id, losing_team in triggered:
            target, reaches_throne = get_push_target_zone_id(state, losing_team, lane_id)
            outcomes.append((lane_id, losing_team, target, reaches_throne))

        # Flip counters for all triggered lanes (they stay flipped even on a tie).
        for lane_id, _, _, _ in outcomes:
            state.wave_counters[lane_id] = max(0, state.wave_counters.get(lane_id, 0) - 1)

        # Throne precedence.
        throne_winners = {
            TeamColor.BLUE if losing_team == TeamColor.RED else TeamColor.RED
            for _, losing_team, _, reaches_throne in outcomes
            if reaches_throne
        }
        if len(throne_winners) == 2:
            logger.debug("   [ENDGAME] Simultaneous throne pushes — tie remedy.")
            return self._tie_remedy(state, outcomes)
        if len(throne_winners) == 1:
            winner = throne_winners.pop()
            logger.debug(f"   [GAME OVER] Lane Push Victory for {winner.name}!")
            return StepResult(
                is_finished=True,
                new_steps=[TriggerGameOverStep(winner=winner, condition="LANE_PUSH")],
            )

        # Last-wave comparison: once any lane's counters are exhausted, every
        # push re-runs the comparison at post-push positions.
        if any(count <= 0 for count in state.wave_counters.values()):
            overrides = {
                lane_id: target for lane_id, _, target, _ in outcomes if target is not None
            }
            totals = endgame_totals(state, overrides)
            red, blue = totals[TeamColor.RED], totals[TeamColor.BLUE]
            logger.debug(f"   [ENDGAME] Last-wave comparison: RED {red} vs BLUE {blue}")
            if red != blue:
                winner = TeamColor.RED if red > blue else TeamColor.BLUE
                return StepResult(
                    is_finished=True,
                    new_steps=[TriggerGameOverStep(winner=winner, condition="LAST_PUSH")],
                )
            logger.debug("   [ENDGAME] Equal totals — tie remedy, play continues.")
            return self._tie_remedy(state, outcomes)

        # No endgame: mechanics-only pushes with pre-computed targets.
        push_new_steps: list[GameStep] = []
        for lane_id, losing_team, target, _ in outcomes:
            if target is None:
                logger.debug(f"   [ERROR] No push target for {lane_id}; skipping.")
                continue
            push_new_steps.append(
                LanePushStep(
                    lane_id=lane_id,
                    losing_team=losing_team,
                    target_zone_id=target,
                    skip_wave_counter=True,
                )
            )
        return StepResult(is_finished=True, new_steps=push_new_steps)

    def _tie_remedy(
        self, state: GameState, outcomes: list[tuple[str, TeamColor, str | None, bool]]
    ) -> StepResult:
        from goa2.engine.steps.movement import ResolveDisplacementStep

        displacements: list[tuple[str, Hex]] = []
        for lane_id, _, _, _ in outcomes:
            zone_id = state.battle_zone_for_lane(lane_id)
            if not zone_id:
                continue
            _wipe_minions_in_zone(state, zone_id)
            displacements.extend(_respawn_minions_at_spawn_points(state, lane_id, zone_id))
        if displacements:
            return StepResult(
                is_finished=True,
                new_steps=[ResolveDisplacementStep(displacements=displacements)],
            )
        return StepResult(is_finished=True)


class LanePushStep(GameStep):
    """
    Executes a Lane Push:
    1. Removes Wave Counter.
    2. Moves Battle Zone.
    3. Wipes Minions in old zone.
    4. Respawns Minions in new zone.
    5. Checks Victory Conditions (Throne or Last Push) — single-lane only;
       on multi-lane games CheckLanePushStep decides the endgame BEFORE
       spawning this step and passes target_zone_id/skip_wave_counter.
    """

    type: StepType = StepType.LANE_PUSH
    lane_id: str = DEFAULT_LANE_ID
    losing_team: TeamColor
    # Set by CheckLanePushStep on multi-lane games (mechanics-only mode).
    target_zone_id: str | None = None
    skip_wave_counter: bool = False

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.map_logic import get_push_target_zone_id
        from goa2.engine.steps.movement import ResolveDisplacementStep

        logger.debug(
            f"   [PUSH] Lane Push Triggered on {self.lane_id}! "
            f"Losing Team: {self.losing_team.name}"
        )

        if not self.skip_wave_counter:
            remaining_waves = state.wave_counters.get(self.lane_id, 0) - 1
            state.wave_counters[self.lane_id] = remaining_waves
            logger.debug(
                f"   [PUSH] Wave Counter removed ({self.lane_id}). Remaining: {remaining_waves}"
            )

            if remaining_waves <= 0:
                # Single-lane last-push rule (pusher wins). Multi-lane games
                # never reach this branch: the coordinator resolves the
                # comparison and spawns this step with skip_wave_counter=True.
                logger.debug("   [GAME OVER] Last Push Victory!")
                winning_team = (
                    TeamColor.BLUE if self.losing_team == TeamColor.RED else TeamColor.RED
                )
                return StepResult(
                    is_finished=True,
                    new_steps=[TriggerGameOverStep(winner=winning_team, condition="LAST_PUSH")],
                )

        if self.target_zone_id is not None:
            next_zone_id: str | None = self.target_zone_id
        else:
            next_zone_id, is_game_over = get_push_target_zone_id(
                state, self.losing_team, self.lane_id
            )

            if is_game_over:
                logger.debug(
                    f"   [GAME OVER] Lane Push Victory! {self.losing_team.name} Throne reached."
                )
                winning_team = (
                    TeamColor.BLUE if self.losing_team == TeamColor.RED else TeamColor.RED
                )
                return StepResult(
                    is_finished=True,
                    new_steps=[TriggerGameOverStep(winner=winning_team, condition="LANE_PUSH")],
                )

        if not next_zone_id:
            logger.debug("   [ERROR] Could not determine next zone for push.")
            return StepResult(is_finished=True)

        current_zone_id = state.battle_zone_for_lane(self.lane_id)
        if not current_zone_id:
            logger.debug("   [ERROR] No active zone for push.")
            return StepResult(is_finished=True)

        _wipe_minions_in_zone(state, current_zone_id)

        logger.debug(f"   [PUSH] Battle Zone moved: {current_zone_id} -> {next_zone_id}")
        state.battle_zones[self.lane_id] = next_zone_id

        pending_displacements = _respawn_minions_at_spawn_points(state, self.lane_id, next_zone_id)

        if pending_displacements:
            return StepResult(
                is_finished=True,
                new_steps=[ResolveDisplacementStep(displacements=pending_displacements)],
            )

        return StepResult(is_finished=True)


class MinionBattleStep(GameStep):
    """
    Compare minion counts in each Battle Zone and queue removals for the losing team.
    Separated from EndPhaseStep so finishing steps can execute first,
    ensuring battle counts reflect post-finishing-step state.

    With lane_id=None (default) battles resolve for every lane (per the
    double-lane rules: "separately, but simultaneously").
    """

    type: StepType = StepType.MINION_BATTLE
    lane_id: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        lanes = [self.lane_id] if self.lane_id is not None else list(state.battle_zones.keys())
        battle_steps: list[GameStep] = []
        for lane_id in lanes:
            battle_steps.extend(self._resolve_minion_battle(state, lane_id))
        return StepResult(is_finished=True, new_steps=battle_steps)

    def _resolve_minion_battle(self, state: GameState, lane_id: str) -> list[GameStep]:
        from goa2.engine.steps.cards import _one_man_army_bonus
        from goa2.engine.steps.selection import ChooseMinionRemovalStep

        zone_id = state.battle_zone_for_lane(lane_id)
        if not zone_id:
            return []

        zone = state.board.zones.get(zone_id)
        if not zone:
            return []

        red_count = 0
        blue_count = 0

        for unit_id, loc in state.unit_locations.items():
            if loc in zone.hexes:
                unit = state.get_unit(UnitID(unit_id))
                if unit and hasattr(unit, "type") and hasattr(unit, "is_heavy"):
                    if unit.team == TeamColor.RED:
                        red_count += 1
                    elif unit.team == TeamColor.BLUE:
                        blue_count += 1

        # One Man Army: heroes with active ultimate count as +1 minion
        bonus = _one_man_army_bonus(state, zone)
        red_count += bonus[TeamColor.RED]
        blue_count += bonus[TeamColor.BLUE]

        # Minion battle exclusion (Wuk - Claim/Assert Dominance): enemy minions
        # adjacent to the origin hero do not count, capped by the summed
        # max_value of that hero's exclusion effects (minions deduped via a set).
        exclusion = self._battle_exclusion_reductions(state, zone)
        red_count -= exclusion[TeamColor.RED]
        blue_count -= exclusion[TeamColor.BLUE]

        diff = abs(red_count - blue_count)

        if diff == 0:
            logger.debug("   [BATTLE] Minion count tied. No removals.")
            return []

        loser_team = TeamColor.RED if red_count < blue_count else TeamColor.BLUE
        logger.debug(f"   [BATTLE] {loser_team.name} loses {diff} minion(s).")

        return [
            ChooseMinionRemovalStep(
                losing_team=loser_team.value,
                remaining_to_remove=diff,
                zone_id=zone_id,
            )
        ]

    def _battle_exclusion_reductions(self, state: GameState, zone) -> dict[TeamColor, int]:
        """How many minions to subtract from each team's battle count due to
        active MINION_BATTLE_EXCLUSION effects (Wuk's Dominance cards).

        Per origin hero: reduction = min(sum of that hero's exclusion caps,
        distinct enemy minions adjacent to the hero inside the zone), counting
        adjacent minions regardless of immunity. Minions are deduped per enemy
        team so overlapping origins never exclude the same minion twice.
        """
        from collections import defaultdict

        from goa2.domain.models import Minion
        from goa2.engine.topology import get_topology_service

        reductions = {TeamColor.RED: 0, TeamColor.BLUE: 0}
        effects = [
            e
            for e in state.active_effects
            if e.effect_type == EffectType.MINION_BATTLE_EXCLUSION and e.is_active
        ]
        if not effects:
            return reductions

        caps_by_origin: dict[str, int] = defaultdict(int)
        for e in effects:
            caps_by_origin[e.source_id] += e.max_value or 0

        topology = get_topology_service()
        excluded: dict[TeamColor, set[str]] = {TeamColor.RED: set(), TeamColor.BLUE: set()}

        for origin_id, cap in caps_by_origin.items():
            if cap <= 0:
                continue
            origin = state.get_entity(BoardEntityID(origin_id))
            origin_team = getattr(origin, "team", None)
            origin_loc = state.entity_locations.get(BoardEntityID(origin_id))
            if origin is None or origin_team is None or origin_loc is None:
                continue
            enemy_team = TeamColor.BLUE if origin_team == TeamColor.RED else TeamColor.RED

            adjacent: list[str] = []
            for n in topology.get_connected_neighbors(origin_loc, state):
                if n not in zone.hexes:
                    continue
                tile = state.board.get_tile(n)
                if not tile or not tile.occupant_id:
                    continue
                occ = state.get_entity(tile.occupant_id)
                if (
                    isinstance(occ, Minion)
                    and occ.team == enemy_team
                    and str(occ.id) not in excluded[enemy_team]
                ):
                    adjacent.append(str(occ.id))

            take = min(cap, len(adjacent))
            for mid in adjacent[:take]:
                excluded[enemy_team].add(mid)
            reductions[enemy_team] += take

        return reductions


class ReturnMinionToZoneStep(GameStep):
    """
    Returns minions that ended up outside the active zone.

    Per manual: "If any minion miniature ends up outside the Battle Zone
    after you perform an action, move it by the shortest path of empty
    spaces to an empty space in the same Battle Zone."

    If multiple shortest paths exist, the minion's team chooses.
    Processes minions in tie-breaker coin order.
    """

    type: StepType = StepType.RETURN_MINION_TO_ZONE

    def _get_minions_outside_zone(self, state: GameState) -> list[tuple[str, TeamColor]]:
        """Find all minions outside the Battle Zone of their own lane."""
        outside = []
        for team in state.teams.values():
            for minion in team.minions:
                zone_id = state.battle_zone_for_lane(minion.lane_id)
                zone = state.board.zones.get(zone_id) if zone_id else None
                if not zone:
                    continue
                loc = state.unit_locations.get(minion.id)
                if loc and loc not in zone.hexes:
                    if not minion.team:
                        logger.debug(
                            f"   [WARNING] Minion {minion.id} has no team, skipping zone return"
                        )
                    else:
                        outside.append((str(minion.id), minion.team))
        return outside

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.movement import PlaceUnitStep

        outside_minions = self._get_minions_outside_zone(state)

        if not outside_minions:
            return StepResult(is_finished=True)

        # Sort by tie-breaker: favored team's minions first
        favored = state.tie_breaker_team
        outside_minions.sort(key=lambda x: (0 if x[1] == favored else 1, str(x[0])))

        # Process first minion
        minion_id, team = outside_minions[0]
        remaining = outside_minions[1:]

        loc = state.entity_locations.get(BoardEntityID(minion_id))
        if not loc:
            # Minion somehow has no location, skip
            if remaining:
                return StepResult(
                    is_finished=True,
                    new_steps=[ReturnMinionToZoneStep()],
                )
            return StepResult(is_finished=True)

        # Return the minion to the Battle Zone of its own lane
        minion = state.get_unit(UnitID(minion_id))
        home_zone_id = state.battle_zone_for_lane(getattr(minion, "lane_id", DEFAULT_LANE_ID))
        if not home_zone_id:
            # No active zone for this lane, skip
            if remaining:
                return StepResult(
                    is_finished=True,
                    new_steps=[ReturnMinionToZoneStep()],
                )
            return StepResult(is_finished=True)

        from goa2.engine.map_logic import find_nearest_empty_hexes

        candidates = find_nearest_empty_hexes(
            state, loc, home_zone_id, respect_obstacles=True, actor_id=minion_id
        )

        if not candidates:
            # Fallback: "If no path exists, it is Placed instead via shortest distance to an empty space within the BattleZone."
            candidates = find_nearest_empty_hexes(state, loc, home_zone_id, respect_obstacles=False)

        if not candidates:
            logger.debug(f"   [ZONE] No empty space in zone for {minion_id}!")
            # Skip this minion, process remaining
            if remaining:
                return StepResult(
                    is_finished=True,
                    new_steps=[ReturnMinionToZoneStep()],
                )
            return StepResult(is_finished=True)

        # If pending input, process it
        if self.pending_input:
            selection = self.pending_input.get("selection")
            target_hex = parse_hex_selection(selection)
            if target_hex in candidates:
                logger.debug(f"   [ZONE] Returning {minion_id} to zone at {target_hex}")
                new_steps: list[GameStep] = [
                    PlaceUnitStep(unit_id=minion_id, target_hex_arg=target_hex),
                ]
                if remaining:
                    new_steps.append(ReturnMinionToZoneStep())
                return StepResult(is_finished=True, new_steps=new_steps)
            logger.debug("   [ZONE] Rejected invalid hex %r; re-requesting.", selection)
            self.pending_input = None

        # Auto-move if only one candidate
        if len(candidates) == 1:
            target = candidates[0]
            logger.debug(f"   [ZONE] Auto-returning {minion_id} to zone at {target}")
            new_steps = [
                PlaceUnitStep(unit_id=minion_id, target_hex_arg=target),
            ]
            if remaining:
                new_steps.append(ReturnMinionToZoneStep())
            return StepResult(is_finished=True, new_steps=new_steps)

        # Multiple candidates - need team input
        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_HEX,
                player_id=f"team:{team.value}",
                prompt=f"Team {team.value}, choose destination for {minion_id} to return to Battle Zone.",
                options=candidates,
            ),
        )


class SpendAdditionalLifeCounterStep(GameStep):
    """
    Decrements the victim's team life counters by 1 (additional penalty).
    Used by Ursafar's Tear: "if you defeated a hero, that hero spends 1
    additional Life counter."
    """

    type: StepType = StepType.SPEND_ADDITIONAL_LIFE_COUNTER
    victim_key: str = "victim_id"
    amount: int = 1

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        victim_id = context.get(self.victim_key)
        if not victim_id:
            return StepResult(is_finished=True)

        victim = state.get_hero(HeroID(str(victim_id)))
        if not victim:
            return StepResult(is_finished=True)

        victim_team_color = getattr(victim, "team", None)
        if not victim_team_color or victim_team_color not in state.teams:
            return StepResult(is_finished=True)

        victim_team = state.teams[victim_team_color]
        victim_team.life_counters = max(0, victim_team.life_counters - self.amount)
        logger.debug(
            f"   [SCORE] Team {victim_team_color.name} loses {self.amount} additional Life Counter(s). "
            f"Remaining: {victim_team.life_counters}"
        )

        events = [
            GameEvent(
                event_type=GameEventType.LIFE_COUNTER_CHANGED,
                target_id=str(victim_id),
                metadata={
                    "team": victim_team_color.name,
                    "change": -self.amount,
                    "remaining": victim_team.life_counters,
                    "reason": "additional_life_counter",
                },
            )
        ]

        if victim_team.life_counters == 0:
            from goa2.domain.models import TeamColor

            winning_team = TeamColor.BLUE if victim_team_color == TeamColor.RED else TeamColor.RED
            events.append(
                GameEvent(
                    event_type=GameEventType.GAME_OVER,
                    metadata={
                        "reason": "annihilation",
                        "winning_team": winning_team.name,
                    },
                )
            )
            return StepResult(
                is_finished=True,
                new_steps=[TriggerGameOverStep(winner=winning_team, condition="ANNIHILATION")],
                events=events,
            )

        return StepResult(is_finished=True, events=events)


class RestoreSpentLifeCounterStep(GameStep):
    """Restore Life counters to a hero's team, capped at its starting supply."""

    type: StepType = StepType.RESTORE_SPENT_LIFE_COUNTER
    hero_id: str | None = None
    hero_key: str | None = None
    amount: int = 1

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        hero_id = self.hero_id
        if self.hero_key:
            hero_id = context.get(self.hero_key) or hero_id
        if hero_id is None:
            hero_id = str(state.current_actor_id) if state.current_actor_id else None
        if hero_id is None:
            return StepResult(is_finished=True)

        hero = state.get_hero(HeroID(str(hero_id)))
        if hero is None or hero.team is None:
            return StepResult(is_finished=True)
        team = state.teams.get(hero.team)
        if team is None or team.starting_life_counters is None:
            return StepResult(is_finished=True)

        restored = min(self.amount, team.starting_life_counters - team.life_counters)
        if restored <= 0:
            return StepResult(is_finished=True)

        team.life_counters += restored
        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.LIFE_COUNTER_CHANGED,
                    actor_id=str(hero.id),
                    metadata={
                        "team": hero.team.value,
                        "change": restored,
                        "remaining": team.life_counters,
                        "reason": "life_counter_restored",
                    },
                )
            ],
        )


class TriggerGameOverStep(GameStep):
    """
    Executes an immediate Game Over sequence.
    1. Sets winner and condition.
    2. Changes Phase to GAME_OVER.
    3. PURGES execution and input stacks to stop all gameplay.
    """

    type: StepType = StepType.TRIGGER_GAME_OVER
    winner: TeamColor | None = None
    individual_winner_id: HeroID | None = None
    condition: str

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if (self.winner is None) == (self.individual_winner_id is None):
            raise ValueError("TriggerGameOverStep requires exactly one team or individual winner")

        if self.individual_winner_id is not None:
            winner = str(self.individual_winner_id)
            winner_type = "HERO"
        else:
            assert self.winner is not None
            winner = self.winner.value
            winner_type = "TEAM"

        logger.debug(f"   [GAME OVER] Victory for {winner}! Reason: {self.condition}")

        state.winner = self.winner
        state.individual_winner_id = self.individual_winner_id
        state.victory_condition = self.condition
        state.phase = GamePhase.GAME_OVER

        # Hard Stop: Clear everything pending
        state.execution_stack.clear()
        state.input_stack.clear()

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.GAME_OVER,
                    metadata={
                        "winner": winner,
                        "winner_type": winner_type,
                        "condition": self.condition,
                    },
                )
            ],
        )
