from __future__ import annotations

from typing import TYPE_CHECKING

from goa2.domain.models import (
    AffectsFilter,
    DurationType,
    EffectType,
    Shape,
)
from goa2.domain.models.effect import EffectScope
from goa2.domain.models.enums import CardContainerType, TargetType
from goa2.engine.effects import CardEffect, register_effect
from goa2.engine.filters_cards import CardsInContainerFilter
from goa2.engine.filters_composite import (
    AndFilter,
    OrFilter,
)
from goa2.engine.filters_hex import (
    AdjacentSpawnPointFilter,
    BattleZoneFilter,
    MovementPathFilter,
    ObstacleFilter,
    RangeFilter,
    SpawnPointFilter,
)
from goa2.engine.filters_units import (
    ExcludeIdentityFilter,
    TeamFilter,
    UnitOnSpawnPointFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CountStep,
    CreateEffectStep,
    DefeatUnitStep,
    ForceDefenseCardMovementStep,
    ForceDiscardOrDefeatStep,
    ForceDiscardStep,
    ForEachStep,
    GameStep,
    MayRepeatOnceStep,
    MoveUnitStep,
    MultiSelectStep,
    PlaceUnitStep,
    RetrieveCardStep,
    SelectStep,
    SwapUnitsStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


# =============================================================================
# HIGH-LEVEL DESIGN: Whisper Card Effects
# =============================================================================
#
# Whisper is an assassin-style hero themed around spawn points, swapping,
# teleportation, and punishing enemies who have discarded cards. Her card
# families scale from Tier I through III with increasing values/options.
#
# =============================================================================
# CROSS-HERO PRECEDENTS (patterns from existing hero effects)
# =============================================================================
#
# The following mechanics already exist across other heroes and serve as
# implementation blueprints for Whisper's cards:
#
# TELEPORT/PLACEMENT:
#   - Arien (liquid_leap, magical_current, stranger_tide): SelectStep(HEX)
#     with SpawnPointFilter(has_spawn_point=False) + OccupiedFilter(False)
#     + AdjacentSpawnPointFilter → PlaceUnitStep. Arien AVOIDS spawn points;
#     Whisper TARGETS them — flip SpawnPointFilter(has_spawn_point=True).
#   - Tigerclaw (blink_strike): ComputeHexStep to find hex behind target,
#     then PlaceUnitStep. Good pattern for computed destinations.
#   - Dodger (blend_into_shadows): Conditional teleport + ATTACK_IMMUNITY
#     grant — combines placement with immunity, similar to Death Seeker.
#
# SWAP:
#   - Arien (arcane_whirlpool, ebb_and_flow): SwapUnitsStep(unit_a_id=hero,
#     unit_b_key=ctx). Ebb and Flow uses CheckAdjacencyStep BEFORE swap to
#     gate MayRepeatOnceStep — pattern for conditional post-swap actions.
#   - Sabina (back_to_back): Swap with friendly minion.
#     Pattern: SelectStep(UNIT) → SwapUnitsStep. Direct match for Cruel Twist.
#
# FORCED MOVEMENT OF OTHER UNITS:
#   - Xargatha (sirens_call): MoveUnitStep(unit_key=target, range_val=3,
#     is_movement_action=False). Forced movement toward self.
#   - Xargatha (charm/control/dominate): Move enemy minion N spaces.
#     SelectStep(HEX) destination from target origin → MoveUnitStep.
#   - Dodger (burning_skull, blazing_skull): Move adjacent minions 1 space.
#     Direct match for Seeds of Fear family's "move unit N spaces."
#
# OPPONENT'S CHOICE:
#   - Bain (were_not_done_yet): SelectStep(NUMBER, override_player_id_key)
#     routes input to opponent. Used for "choose one" opponent decisions.
#   - Pattern: SelectStep(NUMBER) with override_player_id_key=victim_key
#     → CheckContextConditionStep per option → active_if_key branches.
#     Direct match for Lesser Evil / Greater Good.
#
# COUNT + CONDITIONAL LOGIC:
#   - Dodger (_count_empty_spawn_points helper): Counts empty spawn points
#     in radius within battle zone. Accounts for Tide of Darkness. Can be
#     reused directly for Crimson Trail family.
#   - Xargatha (threatening_slash, deadly_swipe): CountStep(UNIT) with
#     filters → conditional attack bonus. Pattern for Blood Fury condition.
#   - Dodger (backstab): CountStep on friendly units adjacent to target.
#
# CONDITIONAL REPEAT:
#   - Arien (violent_torrent): MayRepeatOnceStep with ExcludeIdentityFilter
#     to prevent targeting same unit. Direct match for Blood Fury.
#   - Wasp (center_of_mass): MayRepeatNTimesStep(max_repeats=2).
#     Pattern for Blood Frenzy (max_repeats=5).
#   - Xargatha (rapid_thrusts): Re-counts condition each repeat.
#     Needed for Blood Frenzy's "if enemy hero has card in discard" re-check.
#
# IMMUNITY:
#   - Arien (expert_duelist, master_duelist): CreateEffectStep(
#     EffectType.ATTACK_IMMUNITY, DurationType.THIS_TURN, EffectScope(POINT)).
#     Uses except_attacker_key for exemptions. Direct pattern for Death Seeker.
#   - Dodger (blend_into_shadows): ATTACK_IMMUNITY with DurationType.NEXT_TURN.
#   - Note: Death Seeker says "immune to enemy actions" (not just attacks).
#     May need ATTACK_IMMUNITY + TARGET_PREVENTION combo, or just
#     ATTACK_IMMUNITY if game rules treat "actions" as attacks only.
#
# CARD RETRIEVAL:
#   - Xargatha (devoted_followers, fresh_converts): CountStep → condition
#     → SelectStep(CARD, DISCARD) → RetrieveCardStep. Exact pattern for
#     Death Seeker's retrieval path.
#   - Brogan (bulwark): Optional retrieve from discard.
#   - Bain (drinking_buddies, another_one): Retrieve for self and others.
#
# STRAIGHT-LINE FORCED MOVEMENT:
#   - Brogan (mad_dash, bullrush, furious_charge): InStraightLineFilter +
#     StraightLinePathFilter → MoveUnitStep(range_val=99). Forces movement
#     along a line with full distance. Pattern for Swift Justice's forced
#     defender movement — but needs adaptation to read distance from the
#     defense card's movement stat.
#
# PASSIVE ABILITIES:
#   - Arien (living_tsunami): PassiveConfig(BEFORE_ATTACK, uses=1, optional).
#   - Sabina (cloak_and_daggers): PassiveConfig(AFTER_BASIC_ACTION, uses=1).
#     AFTER_BASIC_ACTION trigger exists — relevant for Grim Reaper which
#     modifies basic action selection. Could use this trigger to grant a
#     second basic action after the first completes.
#   - Brogan (big_sodding_gun): PassiveConfig(AFTER_PUSH, unlimited, auto).
#   - Bain (a_complicated_profession): PassiveConfig(AFTER_PLACE_MARKER).
#
# MULTI-SELECT + DEFEAT:
#   - Wasp (kinetic_repulse): MultiSelectStep(max=2) + ForEachStep +
#     conditional ForceDiscardStep. Pattern for Greater Good's "defeat
#     up to 3 adjacent minions" — MultiSelectStep(max=3) + ForEachStep
#     + DefeatUnitStep per selected minion.
#
# =============================================================================
# CARD FAMILIES & IMPLEMENTATION NOTES
# =============================================================================
#
# FAMILY 1: SHADOW TELEPORT (Green Cards)
#   Shadow Step (I)  / Shadow Walk (II)  / Creeping Shadow (III)
#   ---------------------------------------------------------------
#   Mechanic: Place yourself into an empty minion spawn point in range
#             (in the battle zone). Creeping Shadow adds "or adjacent to
#             such a spawn point."
#   Difficulty: EASY
#   Precedent: Arien's liquid_leap (inverted SpawnPointFilter logic)
#   Steps: SelectStep(HEX) with SpawnPointFilter(has_spawn_point=True) +
#          OccupiedFilter(False) + BattleZoneFilter + RangeFilter
#          → PlaceUnitStep(unit_id=hero.id, destination_key=...)
#   Notes:
#   - Shadow Step (I) and Shadow Walk (II) are identical in logic; range
#     comes from stats. Share implementation via single base class.
#   - Creeping Shadow (III): OR condition — hex IS a spawn point, OR hex
#     is adjacent to one. Use AdjacentSpawnPointFilter(must_not_have=False,
#     is_empty=True) as an alternative filter set. Since filters are AND'd,
#     may need build-time logic to produce two candidate sets and merge,
#     or a new composite OR filter.
#   - All three require BattleZoneFilter for "in the battle zone."
#   - Need SpawnPointTeamFilter or check spawn.type == MINION to ensure
#     only minion spawn points qualify (not hero spawn points).
#   - "Empty" = unoccupied (OccupiedFilter) + not obstacle (ObstacleFilter).
#
#
# FAMILY 2: SEEDS OF FEAR (Blue Cards — Move enemy on spawn point)
#   Seeds of Fear (I) / Sprouts of Panic (II) / Blooming Nightmare (III)
#   ---------------------------------------------------------------
#   Mechanic: Target an enemy unit in radius occupying a spawn point.
#             Move that unit 1/2/3 spaces.
#   Difficulty: MEDIUM
#   Precedent: Xargatha's charm/control (forced unit movement), Dodger's
#              burning_skull (move adjacent minion)
#   Steps:
#     1. SelectStep(UNIT) with RangeFilter(radius) + TeamFilter(ENEMY) +
#        [new: UnitOnSpawnPointFilter] → output "fear_target"
#     2. SelectStep(HEX) with MovementPathFilter(origin_key="fear_target",
#        range_val=N) → output "fear_dest"
#     3. MoveUnitStep(unit_key="fear_target", destination_key="fear_dest",
#        is_movement_action=False)
#   Notes:
#   - NEW FILTER needed: UnitOnSpawnPointFilter — resolves unit's hex from
#     entity_locations, checks tile.spawn_point is not None.
#     → Add FilterType.UNIT_ON_SPAWN_POINT to enums.py
#     → Implement in filters.py (simple: get hex, check tile)
#     → Register in step_types.py AnyFilter union
#   - Forced movement destination: actor selects hex for the moved unit.
#     Use MovementPathFilter with origin_key to validate path from target's
#     position. See Xargatha's charm pattern.
#   - Move distance scales: 1 (T1), 2 (T2), 3 (T3). Read from card stats
#     or hardcode per tier. Sharing via base class with move distance param.
#
#
# FAMILY 3: CRIMSON TRAIL (Red Cards — Pre-attack spawn-point movement)
#   Crimson Trail (I) / Sanguine Path (II) / Blood Pilgrimage (III)
#   ---------------------------------------------------------------
#   Mechanic: Before the attack, move up to N spaces based on empty
#             minion spawn points in radius in the battle zone. Then
#             target an adjacent unit.
#   Difficulty: MEDIUM-HARD
#   Precedent: Dodger's _count_empty_spawn_points helper (reusable),
#              Dodger's backstab CountStep pattern
#   Steps:
#     1. CountStep(HEX) with SpawnPointFilter(has_spawn_point=True) +
#        OccupiedFilter(False) + BattleZoneFilter + RangeFilter(radius)
#        → "spawn_count"
#        OR: reuse/adapt Dodger's _count_empty_spawn_points() at build time
#     2. Cap: CheckContextConditionStep or build-time min(count, max)
#     3. If count > 0: MoveSequenceStep(range=min(count,max), is_mandatory=False)
#     4. AttackSequenceStep(damage=stats.primary_value, range_val=1)
#   Notes:
#   - Dodger's _count_empty_spawn_points() operates at build time (reads
#     state directly). For Whisper, this works since the count is evaluated
#     when the card resolves, before movement.
#   - Crimson Trail (I): "move up to 1 space IF there is an empty spawn
#     point" → count >= 1 ? move 1 : skip. Binary.
#   - Sanguine Path (II): "1 space PER empty spawn point, max 2" →
#     min(count, 2).
#   - Blood Pilgrimage (III): same, max 3.
#   - Build-time approach: compute count in build_steps(), cap it, pass
#     as range_val to MoveSequenceStep. Simpler than runtime CountStep.
#   - Share via base class with max_move parameter.
#
#
# FAMILY 4: BLOOD FURY (Red Cards — Conditional repeat attack)
#   Blood Fury (II) / Blood Frenzy (III)
#   ---------------------------------------------------------------
#   Mechanic: Target adjacent unit. After attack, if enemy hero in radius
#             has a card in discard, may repeat (once for T2, up to 5x for T3)
#             on different targets.
#   Difficulty: HARD
#   Precedent: Arien's violent_torrent (MayRepeatOnceStep + exclude),
#              Xargatha's rapid_thrusts (re-check condition per repeat),
#              Wasp's center_of_mass (MayRepeatNTimesStep)
#   Steps:
#     1. AttackSequenceStep(damage=stats.primary_value, range_val=1)
#     2. CountStep(UNIT) with TeamFilter(ENEMY) + UnitTypeFilter(HERO) +
#        RangeFilter(stats.radius) + CardsInContainerFilter(DISCARD, min=1)
#        → "discard_hero_count"
#     3. CheckContextConditionStep("discard_hero_count" >= 1) → "can_repeat"
#     4. MayRepeatOnceStep / MayRepeatNTimesStep(max=5) with
#        active_if_key="can_repeat", steps_template includes re-check +
#        ExcludeIdentityFilter(exclude_keys=["victim_id"])
#   Notes:
#   - Blood Fury (II): Single repeat. MayRepeatOnceStep with
#     ExcludeIdentityFilter(exclude_keys=["victim_id"]). The condition
#     check (step 2-3) runs once after initial attack.
#   - Blood Frenzy (III): Up to 5 repeats. Each repeat's steps_template
#     must include the condition re-check (CountStep + CheckCondition)
#     because discards change mid-combat. MayRepeatNTimesStep(max=5)
#     with the condition check inside the template.
#   - Target exclusion across repeats: MayRepeatNTimesStep re-uses the
#     same steps_template, but victim_id in context gets overwritten.
#     Need to track all previous victims. Options:
#     a) Use a list accumulator in context (would need new step or logic)
#     b) Accept that only the most recent victim is excluded (simpler
#        but not fully correct for 5 repeats)
#     c) Build all 5 repeat blocks at build-time with unique output keys
#        and chained ExcludeIdentityFilter(exclude_keys=[all previous])
#     Option (c) is safest — build 5 explicit attack+condition blocks.
#
#
# FAMILY 5: CRUEL TWIST (Green Cards — Swap + Move)
#   Cruel Twist (II) / Sealed Fate (III)
#   ---------------------------------------------------------------
#   Mechanic: Swap with adjacent unit, OR swap with enemy hero in range
#             who has a card in discard. Then move up to 2/3 spaces.
#   Difficulty: MEDIUM
#   Precedent: Arien's ebb_and_flow (SwapUnitsStep), Bain's hand_crossbow
#              (two-mode NUMBER select + CheckContextConditionStep branches)
#   Steps:
#     1. SelectStep(NUMBER) — choose mode: adjacent swap vs ranged swap
#     2. CheckContextConditionStep branches for each mode
#     3a. Path A: SelectStep(UNIT, RangeFilter(1)) → SwapUnitsStep
#     3b. Path B: SelectStep(UNIT, RangeFilter(stats.range) + TeamFilter(ENEMY)
#         + UnitTypeFilter(HERO) + CardsInContainerFilter(DISCARD, min=1))
#         → SwapUnitsStep
#     4. MoveSequenceStep(range=N) — standard movement action
#   Notes:
#   - Follows Bain's HandCrossbowEffect two-mode pattern exactly.
#   - Path A: "adjacent unit" = any unit (friend or foe), range 1.
#   - Path B: "enemy hero in range with card in discard" = specific filters.
#   - Movement after swap is a normal movement action (MoveSequenceStep).
#   - Share via base class, move distance differs (2 vs 3).
#
#
# FAMILY 6: LESSER EVIL (Blue Cards — Opponent's choice)
#   Lesser Evil (II) / Greater Good (III)
#   ---------------------------------------------------------------
#   Mechanic: Enemy hero in range chooses one of two options.
#   Difficulty: HARD
#   Precedent: Bain's guess mechanic (override_player_id_key for opponent
#              input), Wasp's kinetic_repulse (MultiSelectStep + ForEachStep
#              + DefeatUnitStep for multi-defeat)
#   Steps:
#     1. SelectStep(UNIT) with RangeFilter(stats.range) + TeamFilter(ENEMY)
#        + UnitTypeFilter(HERO) → "le_victim"
#     2. SelectStep(NUMBER, override_player_id_key="le_victim") — victim
#        picks option 1 or 2
#     3. CheckContextConditionStep branches
#     4a/4b. Execute chosen path
#   Notes:
#   - override_player_id_key routes the input prompt to the opponent player.
#     Bain already uses this for the guess mechanic — proven pattern.
#   - Lesser Evil:
#     * Option 1: ForceDiscardStep(victim_key="le_victim") — "if able" =
#       just ForceDiscardStep (no penalty if no cards).
#     * Option 2: Optional SelectStep(UNIT) for adjacent minion →
#       DefeatUnitStep. Actor selects which minion (not victim).
#     * "(Any option can be chosen, even if no effect)" — both always valid.
#   - Greater Good:
#     * Option 1: ForceDiscardOrDefeatStep(victim_key="gg_victim") —
#       "discards a card, or is defeated."
#     * Option 2: MultiSelectStep(UNIT, max=3) with RangeFilter(1) +
#       TeamFilter(ENEMY) + UnitTypeFilter(MINION) → ForEachStep +
#       DefeatUnitStep. Actor selects up to 3 adjacent minions to defeat.
#     * Note: "you defeat" = actor's action on actor's adjacent minions.
#
#
# STANDALONE: DEATH SEEKER (Silver — Conditional immunity or retrieval)
#   ---------------------------------------------------------------
#   Mechanic: If enemy hero in radius has card in discard, choose:
#             immune to enemy actions this turn, OR retrieve a discarded card.
#   Difficulty: MEDIUM-HARD
#   Precedent: Arien's expert_duelist (ATTACK_IMMUNITY), Xargatha's
#              devoted_followers (conditional retrieval), Dodger's
#              blend_into_shadows (teleport + immunity combo)
#   Steps:
#     1. CountStep(UNIT) with TeamFilter(ENEMY) + UnitTypeFilter(HERO) +
#        RangeFilter(stats.radius) + CardsInContainerFilter(DISCARD, min=1)
#        → "ds_hero_count"
#     2. CheckContextConditionStep("ds_hero_count" >= 1) → "ds_condition"
#     3. SelectStep(NUMBER, active_if_key="ds_condition") — choose path
#     4a. Path A: CreateEffectStep(EffectType.ATTACK_IMMUNITY,
#         DurationType.THIS_TURN, EffectScope(shape=Shape.POINT,
#         origin_id=hero.id), is_active=True)
#     4b. Path B: SelectStep(CARD, DISCARD) → RetrieveCardStep
#   Notes:
#   - "Immune to enemy actions" — ATTACK_IMMUNITY covers attacks.
#     For full "actions" immunity (skills too), may additionally need
#     TARGET_PREVENTION(restrictions=[ActionType.SKILL]) or verify that
#     the game rules mean "attacks" specifically. If "actions" truly
#     means everything, combine ATTACK_IMMUNITY + TARGET_PREVENTION.
#   - Retrieval path follows Xargatha's devoted_followers exactly:
#     SelectStep(CARD, DISCARD) → RetrieveCardStep(active_if_key).
#   - Condition check is mandatory — if no enemy hero has discards, the
#     entire effect is skipped (no options presented).
#
#
# STANDALONE: SWIFT JUSTICE (Gold — Choose attack mode)
#   ---------------------------------------------------------------
#   Mechanic: Choose one:
#     * Target hero in range with empty discard. After attack: that hero
#       performs a movement action on card they defended with, moving
#       full distance in a straight line.
#     * Target adjacent unit.
#   Difficulty: VERY HARD
#   Precedent: Brogan's bullrush/furious_charge (InStraightLineFilter +
#              StraightLinePathFilter forced movement), Bain's hand_crossbow
#              (two-mode attack selection)
#   Steps:
#     1. SelectStep(NUMBER) — choose mode
#     2a. Path A: AttackSequenceStep with target_filters=[UnitTypeFilter(HERO),
#         CardsInContainerFilter(DISCARD, max=0)]
#         → After attack resolves: read defender's defense card from context,
#         extract movement secondary action value, force defender to move
#         that many spaces in a straight line
#     2b. Path B: AttackSequenceStep(range=1)
#   Notes:
#   - Path A after-attack effect is the hard part:
#     * AttackSequenceStep stores "defender_card_id" in context (need to
#       verify this exists or add it).
#     * New step: ReadCardStatStep — reads a card's secondary action value
#       (MOVEMENT) and stores it in context.
#     * Brogan's straight-line pattern: InStraightLineFilter +
#       StraightLinePathFilter validate direction, then MoveUnitStep
#       (range_val=movement_value). But here the defender must move FULL
#       distance (not "up to"), and direction needs to be selected.
#     * May need ForceStraightLineMoveStep or adapt MoveUnitStep with a
#       "must move full distance" flag.
#   - The "empty discard" filter ensures target has all cards and WILL
#     defend, guaranteeing a defense card exists to read movement from.
#   - "Performs a movement action on the card they defended with" = the
#     movement value is from the DEFENSE card, not the attack card.
#   - Alternative simpler approach: after attack, store defense card in
#     context during ResolveCombatStep, then add steps to extract movement
#     and force the move.
#
#
# ULTIMATE: GRIM REAPER (Purple — Passive)
#   ---------------------------------------------------------------
#   Mechanic: When performing basic actions, you may choose one, or both.
#   Difficulty: VERY HARD (engine-level change)
#   Precedent: Sabina's cloak_and_daggers (AFTER_BASIC_ACTION trigger —
#              fires after a basic action completes, grants a second action)
#   Notes:
#   - "Basic actions" = secondary actions on a card (Defense, Movement).
#     Normally a hero picks ONE. Grim Reaper lets Whisper pick BOTH.
#   - Sabina's AFTER_BASIC_ACTION trigger is the closest precedent. Her
#     passive repeats the basic action on a different target. Grim Reaper
#     could use the same trigger to grant a DIFFERENT basic action type.
#   - Implementation approach (using existing trigger):
#     a) PassiveConfig(trigger=AFTER_BASIC_ACTION, uses=1, optional=True)
#     b) get_passive_steps(): if hero just did Movement, offer Defense
#        action; if hero just did Defense, offer Movement action.
#     c) The passive's steps would push the second basic action onto
#        the execution stack.
#   - This may be simpler than initially estimated IF the AFTER_BASIC_ACTION
#     trigger provides enough context (which action was taken, which card).
#     Need to verify what context the trigger passes.
#   - Alternative: modify FinalizeHeroTurnStep to check for a "dual basic
#     action" flag and push both actions. More invasive but cleaner.
#   - Upgraded difficulty: HARD (not VERY HARD) if AFTER_BASIC_ACTION
#     trigger approach works. Needs investigation of the trigger's context.
#
#
# =============================================================================
# DIFFICULTY SUMMARY
# =============================================================================
#
#   EASY (2 cards):
#     - Shadow Step (I)          — teleport to spawn point
#     - Shadow Walk (II)         — same, more range
#
#   MEDIUM (6 cards):
#     - Seeds of Fear (I)        — move enemy on spawn point (new filter)
#     - Sprouts of Panic (II)    — same, more distance
#     - Blooming Nightmare (III) — same, more distance
#     - Cruel Twist (II)         — swap + move (two-mode choice)
#     - Sealed Fate (III)        — same, more move distance
#     - Creeping Shadow (III)    — teleport to spawn point OR adjacent
#
#   MEDIUM-HARD (4 cards):
#     - Crimson Trail (I)        — count spawn points, conditional pre-move
#     - Sanguine Path (II)       — same, higher cap
#     - Blood Pilgrimage (III)   — same, highest cap
#     - Death Seeker             — conditional immunity or retrieval
#
#   HARD (4 cards):
#     - Blood Fury (II)          — attack + conditional repeat once
#     - Lesser Evil (II)         — opponent's choice mechanic
#     - Greater Good (III)       — opponent's choice with multi-defeat
#     - Blood Frenzy (III)       — attack + repeat up to 5x (target tracking)
#
#   VERY HARD (1 card):
#     - Swift Justice             — forced movement on defense card stats
#
#   HARD* (1 card — pending trigger investigation):
#     - Grim Reaper (Ultimate)   — dual basic action (may be MEDIUM-HARD
#                                  if AFTER_BASIC_ACTION trigger works)
#
#
# =============================================================================
# IMPLEMENTATION ORDER (recommended)
# =============================================================================
#
#   Phase A — Foundation (EASY, establish patterns):
#     1. Shadow Step / Shadow Walk       — teleport to spawn point
#     2. Creeping Shadow                 — extend with OR filter
#
#   Phase B — Core Mechanics (MEDIUM, new filter + forced movement):
#     3. UnitOnSpawnPointFilter          — prerequisite for Fear family
#     4. Seeds of Fear family            — forced enemy movement
#     5. Cruel Twist / Sealed Fate       — swap + move
#
#   Phase C — Conditional Logic (MEDIUM-HARD):
#     6. Crimson Trail family            — count spawn points + pre-move
#     7. Death Seeker                    — condition check + branch
#
#   Phase D — Complex Interactions (HARD):
#     8. Lesser Evil / Greater Good      — opponent choice
#     9. Blood Fury                      — conditional repeat once
#    10. Blood Frenzy                    — conditional repeat 5x
#
#   Phase E — Engine Extensions (VERY HARD):
#    11. Swift Justice                   — defense card stat reading
#    12. Grim Reaper                     — dual basic action passive
#
#
# =============================================================================
# NEW COMPONENTS NEEDED
# =============================================================================
#
#   NEW FILTER (required):
#     - UnitOnSpawnPointFilter: Filters unit candidates to those whose
#       current hex has a spawn point. Needed for Seeds of Fear family.
#       → Add FilterType.UNIT_ON_SPAWN_POINT to enums.py
#       → Implement in filters.py (resolve unit hex → check tile.spawn_point)
#       → Register in step_types.py AnyFilter union
#       Estimated effort: small (follows existing filter patterns)
#
#   POSSIBLE NEW STEPS (for Phase E):
#     - ReadDefenseCardStatStep: For Swift Justice — reads the defender's
#       defense card's secondary action value (MOVEMENT) from context and
#       stores it for MoveUnitStep. Alternatively, extend ResolveCombatStep
#       to always store defense card info in context.
#     - ForceStraightLineMoveStep: Forces a unit to move exact distance
#       in a straight line. Could adapt MoveUnitStep with must_move_full=True
#       + InStraightLineFilter. Follow Brogan's bullrush pattern.
#
#   POSSIBLE TRIGGER INVESTIGATION (for Phase E):
#     - Check AFTER_BASIC_ACTION trigger context: does it include which
#       action type was performed and which card? If yes, Grim Reaper
#       can use it directly. If not, need to extend the trigger context.
#
# =============================================================================


# =============================================================================
# SHADOW STEP / SHADOW WALK (Green Tier I / II)
# "Place yourself into an empty minion spawn point in range in the battle zone."
# =============================================================================


@register_effect("shadow_step")
@register_effect("shadow_walk")
class _ShadowTeleportEffect(CardEffect):
    """
    Card text: "Place yourself into an empty minion spawn point in range
    in the battle zone."
    Used by: Shadow Step (T1), Shadow Walk (T2)
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select an empty spawn point to teleport to",
                output_key="shadow_dest",
                filters=[
                    RangeFilter(max_range=stats.range),
                    ObstacleFilter(is_obstacle=False),
                    SpawnPointFilter(has_spawn_point=True, minion_only=True),
                    BattleZoneFilter(),
                ],
                is_mandatory=True,
            ),
            PlaceUnitStep(unit_id=hero.id, destination_key="shadow_dest"),
        ]


# =============================================================================
# CREEPING SHADOW (Green Tier III)
# "Place yourself into an empty minion spawn point in range in the battle zone,
#  or into a space in range adjacent to such a spawn point."
# =============================================================================


@register_effect("creeping_shadow")
class CreepingShadowEffect(CardEffect):
    """
    Card text: "Place yourself into an empty minion spawn point in range
    in the battle zone, or into a space in range adjacent to such a spawn point."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select destination to teleport to",
                output_key="shadow_dest",
                filters=[
                    RangeFilter(max_range=stats.range),
                    ObstacleFilter(is_obstacle=False),
                    OrFilter(
                        filters=[
                            AndFilter(
                                filters=[
                                    SpawnPointFilter(has_spawn_point=True, minion_only=True),
                                    BattleZoneFilter(),
                                ]
                            ),
                            AdjacentSpawnPointFilter(
                                is_empty=True,
                                must_not_have=False,
                                battle_zone_only=True,
                                minion_only=True,
                            ),
                        ]
                    ),
                ],
                is_mandatory=True,
            ),
            PlaceUnitStep(unit_id=hero.id, destination_key="shadow_dest"),
        ]


# =============================================================================
# SEEDS OF FEAR / SPROUTS OF PANIC / BLOOMING NIGHTMARE (Blue I/II/III)
# "Target an enemy unit in radius occupying a spawn point.
#  Move that unit N spaces."
# =============================================================================


@register_effect("seeds_of_fear")
class SeedsOfFearEffect(CardEffect):
    """
    Card text: "Target an enemy unit in radius occupying a spawn point.
    Move that unit 1 space."
    """

    move_distance: int = 1

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy unit on a spawn point",
                output_key="fear_target",
                filters=[
                    RangeFilter(max_range=stats.radius),
                    TeamFilter(relation="ENEMY"),
                    UnitOnSpawnPointFilter(),
                ],
                is_mandatory=True,
            ),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select destination for the target",
                output_key="fear_dest",
                filters=[
                    MovementPathFilter(range_val=self.move_distance, unit_key="fear_target"),
                ],
                is_mandatory=True,
            ),
            MoveUnitStep(
                unit_key="fear_target",
                destination_key="fear_dest",
                range_val=self.move_distance,
                is_movement_action=False,
            ),
        ]


@register_effect("sprouts_of_panic")
class SproutsOfPanicEffect(SeedsOfFearEffect):
    """
    Card text: "Target an enemy unit in radius occupying a spawn point.
    Move that unit up to 2 spaces."
    """

    move_distance: int = 2


@register_effect("blooming_nightmare")
class BloomingNightmareEffect(SeedsOfFearEffect):
    """
    Card text: "Target an enemy unit in radius occupying a spawn point.
    Move that unit up to 3 spaces."
    """

    move_distance: int = 3


# =============================================================================
# CRIMSON TRAIL / SANGUINE PATH / BLOOD PILGRIMAGE (Red I/II/III)
# "Before the attack: Move up to N spaces based on empty spawn points
#  in radius in the battle zone. Target adjacent unit."
# =============================================================================


@register_effect("crimson_trail")
class CrimsonTrailEffect(CardEffect):
    """
    Card text: "Before the attack: You may move up to 1 space if there is
    an empty minion spawn point in radius in the battle zone. Target a unit
    adjacent to you."
    """

    max_move: int = 1

    def _compute_move_distance(self, state: GameState, hero: Hero, radius: int) -> int:
        from goa2.scripts.dodger_effects import _count_empty_spawn_points

        count = _count_empty_spawn_points(state, hero.id, radius, minion_only=True)
        return min(count, self.max_move)

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        move_dist = self._compute_move_distance(state, hero, stats.radius or 0)
        steps: list[GameStep] = []
        if move_dist > 0:
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.HEX,
                        prompt="Select destination to move to",
                        output_key="crimson_dest",
                        filters=[
                            MovementPathFilter(range_val=move_dist, unit_id=hero.id),
                        ],
                        is_mandatory=False,
                    ),
                    MoveUnitStep(
                        unit_id=hero.id,
                        destination_key="crimson_dest",
                        range_val=move_dist,
                        is_movement_action=False,
                    ),
                ]
            )
        steps.append(
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
            )
        )
        return steps


@register_effect("sanguine_path")
class SanguinePathEffect(CrimsonTrailEffect):
    """
    Card text: "Before the attack: You may move up to 1 space for every
    empty minion spawn point in radius in the battle zone, up to a maximum
    of 2 spaces. Target a unit adjacent to you."
    """

    max_move: int = 2


@register_effect("blood_pilgrimage")
class BloodPilgrimageEffect(CrimsonTrailEffect):
    """
    Card text: "Before the attack: You may move up to 1 space for every
    empty minion spawn point in radius in the battle zone, up to a maximum
    of 3 spaces. Target a unit adjacent to you."
    """

    max_move: int = 3


# =============================================================================
# CRUEL TWIST / SEALED FATE (Green II/III)
# "Swap with adjacent unit, or enemy hero in range with card in discard.
#  Move up to N spaces."
# =============================================================================


@register_effect("cruel_twist")
class CruelTwistEffect(CardEffect):
    """
    Card text: "Swap with a unit adjacent to you, or with an enemy hero
    in range with a card in the discard. Move up to 2 spaces."
    """

    move_distance: int = 2

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a unit to swap with",
                output_key="swap_target",
                filters=[
                    OrFilter(
                        filters=[
                            RangeFilter(max_range=1),
                            AndFilter(
                                filters=[
                                    TeamFilter(relation="ENEMY"),
                                    UnitTypeFilter(unit_type="HERO"),
                                    RangeFilter(max_range=stats.range),
                                    CardsInContainerFilter(
                                        container=CardContainerType.DISCARD,
                                        min_cards=1,
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                is_mandatory=True,
            ),
            SwapUnitsStep(unit_a_id=hero.id, unit_b_key="swap_target"),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select destination to move to",
                output_key="swap_move_dest",
                filters=[
                    MovementPathFilter(range_val=self.move_distance, unit_id=hero.id),
                ],
                is_mandatory=False,
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="swap_move_dest",
                range_val=self.move_distance,
                is_movement_action=False,
            ),
        ]


@register_effect("sealed_fate")
class SealedFateEffect(CruelTwistEffect):
    """
    Card text: "Swap with a unit adjacent to you, or with an enemy hero
    in range with a card in the discard. Move up to 3 spaces."
    """

    move_distance: int = 3


# =============================================================================
# BLOOD FURY (Red Tier II)
# "Target adjacent. After attack: if enemy hero in radius has card in discard,
#  may repeat once on different target."
# =============================================================================


@register_effect("blood_fury")
class BloodFuryEffect(CardEffect):
    """
    Card text: "Target a unit adjacent to you. After the attack: If an enemy
    hero in radius has a card in the discard, may repeat once on a different target."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
                target_output_key="bf_victim_1",
            ),
            # Condition check
            CountStep(
                target_type=TargetType.UNIT,
                filters=[
                    RangeFilter(max_range=stats.radius),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="HERO"),
                    CardsInContainerFilter(container=CardContainerType.DISCARD, min_cards=1),
                ],
                output_key="bf_discard_count",
            ),
            CheckContextConditionStep(
                input_key="bf_discard_count",
                operator=">=",
                threshold=1,
                output_key="bf_can_repeat",
            ),
            MayRepeatOnceStep(
                prompt="Repeat attack on a different target?",
                active_if_key="bf_can_repeat",
                steps_template=[
                    AttackSequenceStep(
                        damage=stats.primary_value,
                        range_val=1,
                        target_output_key="bf_victim_2",
                        target_filters=[
                            ExcludeIdentityFilter(exclude_keys=["bf_victim_1"]),
                        ],
                    ),
                ],
            ),
        ]


# =============================================================================
# BLOOD FRENZY (Red Tier III)
# "Target adjacent. After attack: if enemy hero in radius has card in discard,
#  repeat up to 5 times on different targets."
# =============================================================================


@register_effect("blood_frenzy")
class BloodFrenzyEffect(CardEffect):
    """
    Card text: "Target a unit adjacent to you. After the attack: If an enemy
    hero in radius has a card in the discard, repeat up to five times on
    different targets."

    Builds 5 explicit attack blocks with chained ExcludeIdentityFilter to
    track all previous victims.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
                target_output_key="bfr_victim_1",
            ),
            # Condition check (once, after initial attack)
            CountStep(
                target_type=TargetType.UNIT,
                filters=[
                    RangeFilter(max_range=stats.radius),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="HERO"),
                    CardsInContainerFilter(container=CardContainerType.DISCARD, min_cards=1),
                ],
                output_key="bfr_discard_count",
            ),
            CheckContextConditionStep(
                input_key="bfr_discard_count",
                operator=">=",
                threshold=1,
                output_key="bfr_can_repeat",
            ),
        ]

        # Build 5 explicit repeat blocks with chained exclusions
        for i in range(2, 7):  # victims 2 through 6
            exclude_keys = [f"bfr_victim_{j}" for j in range(1, i)]
            repeat_steps: list[GameStep] = [
                AttackSequenceStep(
                    damage=stats.primary_value,
                    range_val=1,
                    target_output_key=f"bfr_victim_{i}",
                    target_filters=[
                        ExcludeIdentityFilter(exclude_keys=exclude_keys),
                    ],
                ),
            ]
            steps.append(
                MayRepeatOnceStep(
                    prompt=f"Repeat attack? ({i - 1}/5)",
                    active_if_key="bfr_can_repeat",
                    steps_template=repeat_steps,
                ),
            )

        return steps


# =============================================================================
# LESSER EVIL (Blue Tier II)
# "Enemy hero in range chooses: discard a card if able, OR you may defeat
#  an adjacent minion."
# =============================================================================


@register_effect("lesser_evil")
class LesserEvilEffect(CardEffect):
    """
    Card text: "An enemy hero in range chooses one —
    • That hero discards a card, if able.
    • You may defeat a minion adjacent to you.
    (Any option can be chosen, even if it would have no effect.)"
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select enemy hero target
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero in range",
                output_key="le_victim",
                filters=[
                    RangeFilter(max_range=stats.range),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="HERO"),
                ],
                is_mandatory=True,
            ),
            # 2. Victim chooses
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one",
                output_key="le_choice",
                number_options=[1, 2],
                number_labels={
                    1: "Discard a card (if able)",
                    2: "Opponent may defeat a minion adjacent to them",
                },
                override_player_id_key="le_victim",
                is_mandatory=True,
            ),
            # 3a. Discard path
            CheckContextConditionStep(
                input_key="le_choice",
                operator="==",
                threshold=1,
                output_key="le_chose_discard",
            ),
            ForceDiscardStep(
                victim_key="le_victim",
                active_if_key="le_chose_discard",
            ),
            # 3b. Defeat minion path — actor selects adjacent minion
            CheckContextConditionStep(
                input_key="le_choice",
                operator="==",
                threshold=2,
                output_key="le_chose_defeat",
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an adjacent minion to defeat",
                output_key="le_minion_target",
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="MINION"),
                ],
                is_mandatory=False,
                active_if_key="le_chose_defeat",
            ),
            DefeatUnitStep(
                victim_key="le_minion_target",
                killer_id=str(hero.id),
                active_if_key="le_minion_target",
            ),
        ]


# =============================================================================
# GREATER GOOD (Blue Tier III)
# "Enemy hero in range chooses: discard a card or be defeated, OR you defeat
#  up to 3 adjacent minions."
# =============================================================================


@register_effect("greater_good")
class GreaterGoodEffect(CardEffect):
    """
    Card text: "An enemy hero in range chooses one —
    • That hero discards a card, or is defeated.
    • You defeat up to 3 minions adjacent to you."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select enemy hero target
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero in range",
                output_key="gg_victim",
                filters=[
                    RangeFilter(max_range=stats.range),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="HERO"),
                ],
                is_mandatory=True,
            ),
            # 2. Victim chooses
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one",
                output_key="gg_choice",
                number_options=[1, 2],
                number_labels={
                    1: "Discard a card, or be defeated",
                    2: "Opponent defeats up to 3 minions adjacent to them",
                },
                override_player_id_key="gg_victim",
                is_mandatory=True,
            ),
            # 3a. Discard or defeat path
            CheckContextConditionStep(
                input_key="gg_choice",
                operator="==",
                threshold=1,
                output_key="gg_chose_discard",
            ),
            ForceDiscardOrDefeatStep(
                victim_key="gg_victim",
                active_if_key="gg_chose_discard",
            ),
            # 3b. Defeat up to 3 adjacent minions path
            CheckContextConditionStep(
                input_key="gg_choice",
                operator="==",
                threshold=2,
                output_key="gg_chose_defeat",
            ),
            MultiSelectStep(
                target_type=TargetType.UNIT,
                prompt="Select up to 3 adjacent minions to defeat",
                output_key="gg_minion_targets",
                max_selections=3,
                min_selections=0,
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="MINION"),
                ],
                active_if_key="gg_chose_defeat",
            ),
            ForEachStep(
                list_key="gg_minion_targets",
                item_key="gg_current_minion",
                active_if_key="gg_chose_defeat",
                steps_template=[
                    DefeatUnitStep(victim_key="gg_current_minion", killer_id=str(hero.id)),
                ],
            ),
        ]


# =============================================================================
# DEATH SEEKER (Silver — Untiered)
# "If enemy hero in radius has card in discard, choose: immune to enemy
#  actions this turn, OR retrieve a discarded card."
# =============================================================================


@register_effect("death_seeker")
class DeathSeekerEffect(CardEffect):
    """
    Card text: "If an enemy hero in radius has a card in the discard, choose one —
    • This turn: You are immune to enemy actions.
    • You may retrieve a discarded card."

    With Grim Reaper ultimate active: after choosing one, optionally do the other.
    """

    def _has_ultimate(self, hero: Hero) -> bool:
        return hero.ultimate_card is not None and hero.level >= 8

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        has_ult = self._has_ultimate(hero)

        steps: list[GameStep] = [
            # 1. Check condition: enemy hero in radius with card in discard
            CountStep(
                target_type=TargetType.UNIT,
                filters=[
                    RangeFilter(max_range=stats.radius),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="HERO"),
                    CardsInContainerFilter(container=CardContainerType.DISCARD, min_cards=1),
                ],
                output_key="ds_hero_count",
            ),
            CheckContextConditionStep(
                input_key="ds_hero_count",
                operator=">=",
                threshold=1,
                output_key="ds_condition",
            ),
            # 2. Choose one
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one",
                output_key="ds_choice",
                number_options=[1, 2],
                number_labels={
                    1: "Immunity to enemy actions this turn",
                    2: "Retrieve a discarded card",
                },
                active_if_key="ds_condition",
                is_mandatory=True,
            ),
            # 3a. Immunity path
            CheckContextConditionStep(
                input_key="ds_choice",
                operator="==",
                threshold=1,
                output_key="ds_chose_immunity",
            ),
            CreateEffectStep(
                effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
                scope=EffectScope(
                    shape=Shape.POINT,
                    origin_id=hero.id,
                    affects=AffectsFilter.SELF,
                ),
                duration=DurationType.THIS_TURN,
                is_active=True,
                active_if_key="ds_chose_immunity",
            ),
            # 3b. Retrieval path
            CheckContextConditionStep(
                input_key="ds_choice",
                operator="==",
                threshold=2,
                output_key="ds_chose_retrieval",
            ),
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                prompt="Select a discarded card to retrieve",
                output_key="ds_retrieved_card",
                is_mandatory=False,
                active_if_key="ds_chose_retrieval",
            ),
            RetrieveCardStep(
                card_key="ds_retrieved_card",
                active_if_key="ds_retrieved_card",
            ),
        ]

        # 4. With ultimate: offer the unchosen option
        if has_ult:
            # If chose immunity (1), also offer retrieval
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.CARD,
                        card_container=CardContainerType.DISCARD,
                        prompt="Grim Reaper: Also retrieve a discarded card?",
                        output_key="ds_ult_retrieved_card",
                        is_mandatory=False,
                        active_if_key="ds_chose_immunity",
                    ),
                    RetrieveCardStep(
                        card_key="ds_ult_retrieved_card",
                        active_if_key="ds_ult_retrieved_card",
                    ),
                ]
            )
            # If chose retrieval (2), offer immunity as optional
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.NUMBER,
                        prompt="Grim Reaper: Also become immune to enemy actions?",
                        output_key="ds_ult_immunity_choice",
                        number_options=[1, 2],
                        number_labels={1: "Yes", 2: "No"},
                        active_if_key="ds_chose_retrieval",
                        is_mandatory=True,
                    ),
                    CheckContextConditionStep(
                        input_key="ds_ult_immunity_choice",
                        operator="==",
                        threshold=1,
                        output_key="ds_ult_wants_immunity",
                    ),
                    CreateEffectStep(
                        effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
                        scope=EffectScope(
                            shape=Shape.POINT,
                            origin_id=hero.id,
                            affects=AffectsFilter.SELF,
                        ),
                        duration=DurationType.THIS_TURN,
                        is_active=True,
                        active_if_key="ds_ult_wants_immunity",
                    ),
                ]
            )

        return steps


# =============================================================================
# SWIFT JUSTICE (Gold — Untiered)
# "Choose: target hero in range with empty discard (after attack: forced
#  straight-line movement on defense card), OR target adjacent unit."
# =============================================================================


@register_effect("swift_justice")
class SwiftJusticeEffect(CardEffect):
    """
    Card text: "Choose one —
    • Target a hero in range with an empty discard. After the attack: If able,
      that hero performs a movement action on the card they defended with,
      moving full distance in a straight line.
    • Target a unit adjacent to you."

    With Grim Reaper ultimate active: after choosing one, optionally do the other.
    """

    def _has_ultimate(self, hero: Hero) -> bool:
        return hero.ultimate_card is not None and hero.level >= 8

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        has_ult = self._has_ultimate(hero)

        steps: list[GameStep] = [
            # 1. Choose mode
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one",
                output_key="sj_choice",
                number_options=[1, 2],
                number_labels={
                    1: "Target hero in range with empty discard (forced movement)",
                    2: "Target adjacent unit",
                },
                is_mandatory=True,
            ),
            # 2a. Path A: ranged attack on hero with empty discard + forced movement
            CheckContextConditionStep(
                input_key="sj_choice",
                operator="==",
                threshold=1,
                output_key="sj_chose_ranged",
            ),
            AttackSequenceStep(
                damage=stats.primary_value,
                is_ranged=True,
                range_val=stats.range,
                target_output_key="sj_victim_a",
                target_filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    CardsInContainerFilter(container=CardContainerType.DISCARD, max_cards=0),
                ],
                active_if_key="sj_chose_ranged",
            ),
            ForceDefenseCardMovementStep(
                defender_key="sj_victim_a",
                active_if_key="sj_chose_ranged",
            ),
            # 2b. Path B: attack adjacent unit
            CheckContextConditionStep(
                input_key="sj_choice",
                operator="==",
                threshold=2,
                output_key="sj_chose_adjacent",
            ),
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
                target_output_key="sj_victim_b",
                active_if_key="sj_chose_adjacent",
                is_ranged=True,
            ),
        ]

        # 3. With ultimate: offer the unchosen path
        if has_ult:
            # If chose ranged (1), also offer adjacent attack
            steps.append(
                AttackSequenceStep(
                    damage=stats.primary_value,
                    range_val=1,
                    target_output_key="sj_ult_victim_b",
                    active_if_key="sj_chose_ranged",
                    is_ranged=True,
                    is_mandatory=False,
                ),
            )
            # If chose adjacent (2), also offer ranged attack + forced movement
            steps.extend(
                [
                    AttackSequenceStep(
                        damage=stats.primary_value,
                        range_val=stats.range,
                        is_ranged=True,
                        target_output_key="sj_ult_victim_a",
                        target_filters=[
                            UnitTypeFilter(unit_type="HERO"),
                            CardsInContainerFilter(
                                container=CardContainerType.DISCARD, max_cards=0
                            ),
                        ],
                        is_mandatory=False,
                        active_if_key="sj_chose_adjacent",
                    ),
                    ForceDefenseCardMovementStep(
                        defender_key="sj_ult_victim_a",
                        active_if_key="sj_ult_victim_a",
                    ),
                ]
            )

        return steps


# =============================================================================
# GRIM REAPER (Purple — Ultimate Passive)
# "When performing basic actions, you may choose one, or both."
# =============================================================================


@register_effect("grim_reaper")
class GrimReaperEffect(CardEffect):
    """
    Passive: When performing basic actions, you may choose one, or both.

    This passive's effect is handled at build time by Death Seeker and
    Swift Justice, which check hero.ultimate_card.state == CardState.PASSIVE
    and offer the unchosen option when the ultimate is active.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return []
