# Effect Author Reference

Quick-reference for writing card effects in the GoA2 engine. Covers every step, filter, and pattern you need.

**How effects work:** You subclass `CardEffect`, override `build_steps()`, and return a list of `GameStep` objects. The engine executes them in order via the LIFO stack. Stats are pre-computed and passed as `CardStats(primary_value, range, radius)`.

```python
from goa2.engine.effects import CardEffect, register_effect

@register_effect("my_card")
class MyCardEffect(CardEffect):
    def build_steps(self, state, hero, card, stats):
        return [
            AttackSequenceStep(damage=stats.primary_value, range_val=stats.range),
        ]
```

**Other override methods:**
- `build_defense_steps(state, defender, card, stats, context)` — When used as primary DEFENSE in reaction. Return `None` to fall back to `build_steps()`. **`state.current_actor_id` is the attacker inside this method.** The engine swaps the actor to the defender only for the steps you return, so read `defender` for "you"; never `current_actor_id`. Filters inside your returned steps are fine — they run after the swap.
- `build_on_block_steps(state, defender, card, stats, context)` — After successful block ("if you do" effects).
- `get_passive_config()` — Return `PassiveConfig(trigger, uses_per_turn, is_optional, prompt)` for passive abilities.
- `get_passive_steps(state, hero, card, trigger, context)` — Steps when passive triggers.

---

## Table of Contents

- [A. Step Catalog](#a-step-catalog)
  - [Combat](#combat)
  - [Selection](#selection)
  - [Movement](#movement)
  - [Placement](#placement)
  - [Push](#push)
  - [Discard & Defeat](#discard--defeat)
  - [Effects & Markers](#effects--markers)
  - [Logic & Conditions](#logic--conditions)
  - [Control Flow](#control-flow)
  - [Card Management](#card-management)
- [B. Filter Catalog](#b-filter-catalog)
  - [Spatial Filters](#spatial-filters)
  - [Unit Filters](#unit-filters)
  - [Hex Filters](#hex-filters)
  - [Identity Filters](#identity-filters)
  - [Validation Filters](#validation-filters)
- [C. Pattern Library](#c-pattern-library)
- [D. Multi-Piece Heroes (Razzle)](#d-multi-piece-heroes-razzle)

---

## A. Step Catalog

All steps inherit from `GameStep` which provides these base fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `is_mandatory` | `bool` | `True` | If `True` and step fails, `abort_action=True` (skips to `FinalizeHeroTurnStep`). If `False`, step is skipped on failure. |
| `active_if_key` | `str \| None` | `None` | If set, step only runs when this key exists (non-None) in context. |

---

### Combat

#### `AttackSequenceStep`

**Category:** COMBAT
**Description:** Composite step that expands into: Select Target → Reaction Window → Defense Effect → Resolve Combat → On Block Effect.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `damage` | `int` | *required* | Base attack damage |
| `range_val` | `int` | `1` | Attack range (1 = adjacent) |
| `is_ranged` | `bool` | `False` | **Explicit ranged flag.** True = ranged attack, False = melee attack. Effects must set this explicitly for attacks that should be treated as ranged for defense card purposes. |
| `target_id_key` | `str \| None` | `None` | **Input.** Read a target chosen by an earlier step. Skips selection. If the key is missing, there is no attack — see "Choosing the target channel" below. |
| `target_output_key` | `str \| None` | `None` | **Output.** This step picks its own target (restrictions go in `target_filters`) and stores it under this key so later steps can read who was hit. |
| `target_filters` | `List[FilterCondition]` | `[]` | Additional filters for target selection (added to default `RangeFilter` + `TeamFilter(ENEMY)`). **Only read when this step does the picking** — ignored when a target arrives via `target_id_key`. |
| `damage_bonus_key` | `str \| None` | `None` | Context key containing `int` to add to damage |
| `range_bonus_key` | `str \| None` | `None` | Context key containing `int` to add to range |

**Context Written:**
- `attack_is_ranged` → `bool` (True if `is_ranged=True`, False otherwise)
- `attacker_id` → `str` (current actor ID)
- `attack_damage` → `int` (effective damage value)
- `attack_is_basic` → `bool` (source card is GOLD/SILVER)
- `victim_id` → `str` (selected target, via inner `SelectStep`; under `target_output_key` instead when set)
- `defense_value`, `defense_card_id`, `defender_id`, `is_primary_defense` → set by `ReactionWindowStep`
- `block_succeeded` → `bool` (set by `ResolveCombatStep`)

None of these are written when the attack fizzles (see below) — a swing that never happened leaves no trace for later steps in the same turn to misread.

**Context Read:**
- Value of `damage_bonus_key` if set
- Value of `range_bonus_key` if set
- Value of `target_id_key` if set

**Mandatory/Optional:** Uses `is_mandatory` from base. The inner `SelectStep` inherits mandatory behavior — if no valid targets exist and step is mandatory, action aborts.

**Choosing the target channel:**

The two keys are separate channels, and which you set declares what a missing target means:

| You set | Target resolution | If nothing is there |
|---|---|---|
| neither | picks its own, stores under `victim_id` | mandatory select aborts as usual |
| `target_id_key` only | consumes the preselection | **no attack**; aborts the action if `is_mandatory`, fizzles quietly otherwise |
| `target_output_key` only | picks its own, stores under that key | mandatory select aborts as usual |
| both | prefers the preselection, else picks into the output key | — |

**Rule of thumb:** if the restriction lives on a preceding `SelectStep`, use `target_id_key`. If it lives in `target_filters`, use `target_output_key`.

```python
# INPUT — a preceding select carries the restriction.
SelectStep(target_type=TargetType.UNIT, output_key="thunder_target", is_mandatory=True,
           filters=[RangeFilter(max_range=r), TeamFilter(relation="ENEMY"),
                    NotInStraightLineFilter()]),
AttackSequenceStep(damage=..., target_id_key="thunder_target"),

# OUTPUT — no producer; this step is the picker, and later steps read the victim.
AttackSequenceStep(damage=..., target_output_key="ff_victim",
                   target_filters=[InStraightLineFilter()]),
CheckContextConditionStep(input_key="block_succeeded", ...),   # reads the aftermath
```

Do not reach for `target_id_key` as an output channel. The step used to infer the difference, and a missing key silently fell back to a picker carrying only range + enemy team — dropping whatever restriction the producing select held, so the player could hit any enemy in range (issue #25). Splitting the channels makes the worst misclassification a card that no-ops, which is visible and testable.

---

#### `CountAdjacentEnemiesStep`

**Category:** COMBAT
**Description:** Counts enemy units adjacent to the current actor and stores a computed bonus.

Formula: `bonus = max(0, count - subtract) * multiplier`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_key` | `str` | `"adjacent_enemy_bonus"` | Context key for the bonus value |
| `multiplier` | `int` | `1` | Bonus per adjacent enemy |
| `subtract` | `int` | `0` | Subtracted from count before multiplying (use `1` for "other enemies" — excludes the target) |

**Context Written:** `{output_key}` → `int`

**Context Read:** None (uses `state.current_actor_id`)

---

### Selection

#### `SelectStep`

**Category:** SELECTION
**Description:** Unified selection step for units, hexes, cards, or numbers. Applies composable filters to determine valid candidates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_type` | `TargetType` | *required* | `UNIT`, `HEX`, `CARD`, `NUMBER`, `UNIT_OR_TOKEN` |
| `prompt` | `str` | *required* | Text shown to player |
| `output_key` | `str` | `"selection"` | Context key for the selected value |
| `filters` | `List[FilterCondition]` | `[]` | Composable filters (see Filter Catalog) |
| `auto_select_if_one` | `bool` | `False` | Auto-select if only one valid candidate (mandatory only) |
| `context_hero_id_key` | `str \| None` | `None` | For `CARD` selections — context key for which hero's cards to show |
| `card_container` | `CardContainerType` | `HAND` | `HAND`, `PLAYED`, `DISCARD`, `DECK` |
| `number_options` | `List[int]` | `[]` | Valid choices for `NUMBER` target type |
| `skip_immunity_filter` | `bool` | `False` | Disable automatic `ImmunityFilter` for UNIT selections |
| `override_player_id_key` | `str \| None` | `None` | Context key for which player provides input (e.g., victim chooses their own card) |

**Context Written:** `{output_key}` → selected value (unit ID `str`, `Hex`, card ID `str`, or `int`)

**Context Read:** Values referenced by `context_hero_id_key`, `override_player_id_key`

**Notes:**
- For `UNIT` selections, `ImmunityFilter` is **automatically added** unless `skip_immunity_filter=True`.
- If no valid candidates and `is_mandatory=True` → `abort_action=True`.
- If no valid candidates and `is_mandatory=False` → step skipped silently.
- Player can submit `"SKIP"` for optional selections.

---

#### `MultiSelectStep`

**Category:** SELECTION
**Description:** Select up to N targets sequentially. Results stored as a list.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_type` | `TargetType` | *required* | `UNIT`, `HEX`, `UNIT_OR_TOKEN` |
| `prompt` | `str` | *required* | Text shown to player |
| `output_key` | `str` | *required* | Context key for result list |
| `max_selections` | `int` | *required* | Maximum targets to select |
| `min_selections` | `int` | `0` | Minimum required (0 = fully optional) |
| `filters` | `List[FilterCondition]` | `[]` | Same as `SelectStep` |
| `skip_immunity_filter` | `bool` | `False` | Disable automatic `ImmunityFilter` |

**Context Written:** `{output_key}` → `List[str]` (list of selected IDs)

**Context Read:** None

**Notes:** Player can submit `"DONE"` when `min_selections` is met. Already-selected items are excluded from subsequent prompts.

---

### Movement

#### `MoveSequenceStep`

**Category:** MOVEMENT
**Description:** Composite step for movement actions. Expands into: Select Destination Hex → Move Unit. **Only use for Movement Actions** (primary or secondary). For non-action movement, use `MoveUnitStep` directly.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unit_id` | `str \| None` | `None` | Unit to move (defaults to current actor) |
| `range_val` | `int` | `1` | Movement range |
| `destination_key` | `str` | `"target_hex"` | Context key for destination |

**Context Written:** `{destination_key}` → `Hex`

**Context Read:** `{destination_key}` if already set (skips selection)

**Notes:** Respects `MOVEMENT_ZONE` effects that cap movement range. Sets `is_movement_action=True` on inner `MoveUnitStep`.

---

#### `MoveUnitStep`

**Category:** MOVEMENT
**Description:** Moves a unit with pathfinding validation. Used for movement within an action.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unit_id` | `str \| None` | `None` | Unit to move |
| `unit_key` | `str \| None` | `None` | Context key for unit ID |
| `destination_key` | `str` | `"target_hex"` | Context key for destination hex |
| `target_hex_arg` | `Hex \| None` | `None` | Explicit destination (overrides context) |
| `range_val` | `int` | `1` | Maximum movement distance |
| `is_movement_action` | `bool` | `False` | True for primary/secondary movement actions |

**Context Written:** None

**Context Read:** `{unit_key}`, `{destination_key}`

**Events:** `UNIT_MOVED`

---

### Placement

#### `PlaceUnitStep`

**Category:** PLACEMENT
**Description:** Places a unit at a hex directly. No pathfinding — used for teleports, swaps, and forced placements.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unit_id` | `str \| None` | `None` | Unit to place |
| `unit_key` | `str \| None` | `None` | Context key for unit ID |
| `destination_key` | `str` | `"target_hex"` | Context key for destination |
| `target_hex_arg` | `Hex \| None` | `None` | Explicit destination (overrides context) |

**Context Written:** None

**Context Read:** `{unit_key}`, `{destination_key}`

**Events:** `UNIT_PLACED`

**Notes:** Validates occupancy and effects. Blocked by `PLACEMENT_PREVENTION` effects.

---

#### `SwapUnitsStep`

**Category:** PLACEMENT
**Description:** Swaps the positions of two units.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unit_a_id` | `str \| None` | `None` | First unit (direct ID) |
| `unit_b_id` | `str \| None` | `None` | Second unit (direct ID) |
| `unit_a_key` | `str \| None` | `None` | Context key for first unit |
| `unit_b_key` | `str \| None` | `None` | Context key for second unit |

**Context Written:** None

**Context Read:** `{unit_a_key}`, `{unit_b_key}`

**Events:** `UNITS_SWAPPED`

**Notes:** Validates via `can_be_swapped()`. Blocked by `PLACEMENT_PREVENTION` effects.

---

### Push

#### `PushUnitStep`

**Category:** PUSH
**Description:** Pushes a unit away from a source location in a straight line. Stops at obstacles or board edge.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_id` | `str \| None` | `None` | Direct target ID |
| `target_key` | `str \| None` | `None` | Context key for target |
| `source_hex` | `Hex \| None` | `None` | Push origin (defaults to current actor's location) |
| `distance` | `int` | `1` | Push distance (default/fallback) |
| `distance_key` | `str \| None` | `None` | Context key for distance (overrides `distance`) |
| `collision_output_key` | `str \| None` | `None` | If set, stores `True` in context when push is stopped by obstacle |

**Context Written:** `{collision_output_key}` → `bool` (if set)

**Context Read:** `{target_key}`, `{distance_key}`

**Events:** `UNIT_PUSHED` (with `distance` and `collision` metadata)

**Notes:**
- Source and target must be in a straight line (uses `direction_to()`).
- Validates via `can_be_pushed()`. Blocked by push prevention effects.
- Push resolves one hex at a time — stops at first obstacle, board edge, or topology split.

---

### Discard & Defeat

#### `ForceDiscardStep`

**Category:** DISCARD
**Description:** Forces a victim to discard a card from hand. If victim has no cards, nothing happens (safe).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `victim_key` | `str` | *required* | Context key for victim's hero ID |

**Context Written:** `card_to_discard` → `str` (via inner `SelectStep`)

**Context Read:** `{victim_key}`

**Notes:** The victim (not the actor) chooses which card to discard. Uses `override_player_id_key` internally.

---

#### `ForceDiscardOrDefeatStep`

**Category:** DISCARD
**Description:** Forces victim to discard a card. If victim has no cards, they are **defeated** instead.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `victim_key` | `str` | *required* | Context key for victim's hero ID |

**Context Written:** Same as `ForceDiscardStep`

**Context Read:** `{victim_key}`

---

#### `DefeatUnitStep`

**Category:** DEFEAT
**Description:** Processes full defeat sequence: awards gold, updates life counters, returns markers, removes unit.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `victim_id` | `str \| None` | `None` | Direct victim ID |
| `victim_key` | `str \| None` | `None` | Context key for victim ID |
| `killer_id` | `str \| None` | `None` | Killer's ID (for gold rewards) |

**Context Written:** None

**Context Read:** `{victim_key}`

**Events:** `UNIT_DEFEATED`, `GOLD_GAINED`, `LIFE_COUNTER_CHANGED`

---

#### `DiscardCardStep`

**Category:** DISCARD
**Description:** Discards a specific card from a hero's hand. Lower-level step used internally by `ForceDiscardStep`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `card_id` | `str \| None` | `None` | Direct card ID |
| `card_key` | `str \| None` | `None` | Context key for card ID |
| `hero_id` | `str \| None` | `None` | Direct hero ID |
| `hero_key` | `str \| None` | `None` | Context key for hero ID |

**Context Read:** `{card_key}`, `{hero_key}`

---

### Effects & Markers

#### `CreateEffectStep`

**Category:** EFFECT
**Description:** Creates a spatial `ActiveEffect` in the game state.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effect_type` | `EffectType` | *required* | `TARGET_PREVENTION`, `MOVEMENT_ZONE`, `PLACEMENT_PREVENTION`, `ATTACK_IMMUNITY`, `STATIC_BARRIER`, etc. |
| `scope` | `EffectScope` | *required* | Where the effect applies (see below) |
| `duration` | `DurationType` | `THIS_TURN` | `THIS_TURN` or `THIS_ROUND` |
| `restrictions` | `List[ActionType]` | `[]` | Actions blocked (for `TARGET_PREVENTION`) |
| `displacement_blocks` | `List[DisplacementType]` | `[]` | Displacement types blocked: `PLACE`, `SWAP`, `PUSH`, `MOVE` |
| `except_card_colors` | `List[CardColor]` | `[]` | Card colors exempt from restrictions |
| `except_attacker_ids` | `List[str]` | `[]` | Attacker IDs exempt from restrictions |
| `except_attacker_key` | `str \| None` | `None` | Context key for exempt attacker ID |
| `stat_type` | `StatType \| None` | `None` | For stat modifier effects |
| `stat_value` | `int` | `0` | Modifier value |
| `max_value` | `int \| None` | `None` | For `MOVEMENT_ZONE` — max movement allowed |
| `limit_actions_only` | `bool` | `False` | Only limit action-based movement (not forced) |
| `blocks_enemy_actors` | `bool` | `True` | Effect blocks enemy actors |
| `blocks_friendly_actors` | `bool` | `False` | Effect blocks friendly actors |
| `blocks_self` | `bool` | `False` | Effect blocks self |
| `is_active` | `bool` | `False` | If `True`, effect is immediately active (use for defense effects) |
| `source_card_id` | `str \| None` | `None` | Explicit card ID. Leave unset: the card you are writing is stamped in automatically (see below) |
| `use_context_card` | `bool` | `True` | Legacy fallback — reads `current_card_id` from context when no card was stamped |
| `origin_action_type` | `ActionType \| None` | `None` | Whether effect came from SKILL or ATTACK |
| `barrier_radius` | `int` | `0` | For `STATIC_BARRIER` — radius boundary |
| `barrier_origin_id` | `str \| None` | `None` | For `STATIC_BARRIER` — entity for radius calc |

**EffectScope fields:**
- `shape`: `Shape.POINT`, `Shape.ADJACENT`, `Shape.RADIUS`, `Shape.GLOBAL`
- `range`: Range for `RADIUS` shape
- `origin_id`: Entity ID for scope center
- `affects`: `AffectsFilter.ENEMY_HEROES`, `ENEMY_UNITS`, `SELF`, etc.

**Context Read:** `current_card_id`, `current_action_type`, `{except_attacker_key}`

**Events:** `EFFECT_CREATED`

**Card binding and one instance per card**

Effects belong to the card whose text created them. `engine/effects.py`'s
`bind_effect_cards` stamps `source_card_id` onto every `CreateEffectStep` your
`build_steps` returns — and onto any other step declaring a `source_card_id`
field, such as Hanu's `ScheduleJourneyReturnStep`, which creates its effects
directly (including nested `steps_template` / `finishing_steps`) — so leave the
field unset — that is what makes a re-performance (Bullet Time,
Reload, Mind Grip) attribute the effect to the card being performed rather than
the card that granted the re-performance.

Two consequences to write against:

- **Repeats do not stack.** Per the rules, only one instance of an active effect
  per card can be active. `EffectManager.create_effect` reuses an existing row
  matching `(source_id, source_card_id, effect_type, scope)` and returns it
  untouched — no refreshed charges, no restarted duration. A card that needs
  several distinct payloads (Brogan's Bulwark, Dodger's Enfeeblement) still emits
  one `CreateEffectStep` per payload; those differ in type or scope, so they
  coexist. The source is in the key so a card performed by someone else (Gydion's
  spells, Mind Grip) gives the copier their own instance.
- **Token effects opt out.** `is_token_effect=True` skips card binding entirely
  and records the anchor token in `ActiveEffect.token_id`. That token *is* the
  effect's whole lifecycle: it survives its placer's defeat, has no card to
  leave play, is skipped by every duration sweep, and ends only when the token
  leaves the board (`_remove_token_from_board`). Several identical tokens each
  get their own row.

Engine code must never call `build_steps()` directly — use `get_steps()` or
`get_steps_with_stats()`, which apply the binding. A guardrail test enforces
this.

---

#### `PlaceMarkerStep`

**Category:** MARKER
**Description:** Places a marker on a target hero. Markers are singletons — placing on a new target removes from previous.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marker_type` | `MarkerType` | *required* | e.g., `MarkerType.VENOM` |
| `target_id` | `str \| None` | `None` | Direct target ID |
| `target_key` | `str \| None` | `None` | Context key for target |
| `value` | `int` | `0` | Effect magnitude (e.g., `-1` for Venom debuff) |

**Context Read:** `{target_key}`

**Events:** `MARKER_PLACED`

---

#### `RemoveMarkerStep`

**Category:** MARKER
**Description:** Removes a marker, returning it to supply.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marker_type` | `MarkerType` | *required* | Marker to remove |

**Events:** `MARKER_REMOVED`

---

### Logic & Conditions

#### `CheckAdjacencyStep`

**Category:** LOGIC
**Description:** Checks if two units are adjacent and sets a boolean context flag.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unit_a_id` | `str \| None` | `None` | Direct ID |
| `unit_b_id` | `str \| None` | `None` | Direct ID |
| `unit_a_key` | `str \| None` | `None` | Context key |
| `unit_b_key` | `str \| None` | `None` | Context key |
| `output_key` | `str` | `"is_adjacent"` | Context key for result |

**Context Written:** `{output_key}` → `bool`

---

#### `CheckUnitTypeStep`

**Category:** LOGIC
**Description:** Checks if a unit is a HERO or MINION. Used for conditional "if target was a hero" effects.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unit_id` | `str \| None` | `None` | Direct unit ID |
| `unit_key` | `str \| None` | `None` | Context key |
| `expected_type` | `str` | `"HERO"` | `"HERO"` or `"MINION"` |
| `output_key` | `str` | `"is_expected_type"` | Context key for result |

**Context Written:** `{output_key}` → `bool`

---

#### `CombineBooleanContextStep`

**Category:** LOGIC
**Description:** Combines two boolean context values using AND or OR.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key_a` | `str` | *required* | First boolean context key |
| `key_b` | `str` | *required* | Second boolean context key |
| `output_key` | `str` | *required* | Where to store result |
| `operation` | `str` | `"AND"` | `"AND"` or `"OR"` |

**Context Written:** `{output_key}` → `bool`

**Context Read:** `{key_a}`, `{key_b}`

---

#### `SetContextFlagStep`

**Category:** LOGIC
**Description:** Sets a flag/value in execution context. Primarily used by defense effects.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | `str` | *required* | Context key to set |
| `value` | `Any` | `True` | Value to set |

**Context Written:** `{key}` → `{value}`

**Common flags:**
- `auto_block` → `True` — Block succeeds regardless of defense value (e.g., Stop Projectiles vs ranged)
- `defense_invalid` → `True` — Defense fails entirely (e.g., Stop Projectiles vs melee)
- `ignore_minion_defense` → `True` — Skip minion defense modifier in combat resolution

---

### Control Flow

#### `MayRepeatOnceStep`

**Category:** CONTROL
**Description:** Asks player if they want to repeat a sequence of steps once.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps_template` | `List[GameStep]` | `[]` | Steps to repeat (deep-copied each time) |
| `prompt` | `str` | `"Repeat action?"` | Prompt text |

**Context Read:** Validated via `can_repeat_action()` — respects repeat prevention effects.

**Notes:** Subclass of `MayRepeatNTimesStep` with `max_repeats=1`. Supports `active_if_key` for conditional repeats (e.g., "may repeat if target was adjacent").

---

#### `MayRepeatNTimesStep`

**Category:** CONTROL
**Description:** Asks player if they want to repeat a sequence up to N times.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps_template` | `List[GameStep]` | `[]` | Steps to repeat |
| `max_repeats` | `int` | `1` | Maximum repetitions |
| `prompt` | `str` | `"Repeat action?"` | Prompt text |

---

#### `ForEachStep`

**Category:** CONTROL
**Description:** Executes template steps for each item in a context list.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `list_key` | `str` | *required* | Context key containing the list |
| `item_key` | `str` | *required* | Context key to store current item |
| `steps_template` | `List[GameStep]` | `[]` | Steps to execute per item (deep-copied) |

**Context Written:** `{item_key}` → current item from list

**Context Read:** `{list_key}`

**Notes:** Stays on stack (using `is_finished=False`) until all items processed.

---

### Card Management

#### `SwapCardStep`

**Category:** CARD
**Description:** Swaps a hero's current turn card with another card.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_card_id` | `str \| None` | `None` | Direct card ID |
| `target_card_key` | `str \| None` | `None` | Context key for card ID |
| `context_hero_id_key` | `str \| None` | `None` | Context key for hero ID (defaults to current actor) |

**Context Read:** `{target_card_key}`, `{context_hero_id_key}`

---

## B. Filter Catalog

Filters are composed in `SelectStep.filters` (and `MultiSelectStep.filters`). All filters implement `apply(candidate, state, context) → bool`. A candidate passes if **all** filters return `True`.

### Spatial Filters

#### `RangeFilter`

**Description:** Checks distance from an origin. Uses topology-aware distance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_range` | `int` | *required* | Maximum distance |
| `min_range` | `int` | `0` | Minimum distance |
| `origin_id` | `str \| None` | `None` | Literal origin ID |
| `origin_key` | `str \| None` | `None` | Context key for origin ID |

Falls back to current actor if no origin specified.

---

#### `AdjacencyToContextFilter`

**Description:** Selects candidates adjacent to an entity stored in context.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_key` | `str` | *required* | Context key for the reference entity |

---

#### `AdjacencyFilter`

**Description:** Requires the candidate to be adjacent to a unit matching specific tags.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_tags` | `List[str]` | *required* | e.g., `["FRIENDLY", "HERO"]` — all tags must match |

---

#### `LineBehindTargetFilter`

**Description:** Selects hexes/units in a straight line directly behind a target (relative to actor → target direction).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_key` | `str` | *required* | Context key for the target unit |
| `length` | `int` | `1` | How many hexes behind to include |
| `origin_id` | `str \| None` | `None` | Override origin (defaults to current actor) |

---

#### `NotInStraightLineFilter`

**Description:** Excludes targets in a straight line from the actor. Adjacent units are always in a straight line.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `origin_id` | `str \| None` | `None` | Override origin |
| `origin_key` | `str \| None` | `None` | Context key for origin |

---

#### `MovementPathFilter`

**Description:** Filters hexes to only those reachable via valid movement path (BFS pathfinding).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `range_val` | `int` | *required* | Maximum path length |
| `unit_id` | `str \| None` | `None` | Unit doing the moving |
| `unit_key` | `str \| None` | `None` | Context key for unit ID |

---

#### `RelativeDistanceFilter`

**Description:** Compares distance(origin, candidate) against distance(origin, reference unit) using a configurable operator. General-purpose filter for "farther away", "closer", "same distance", etc.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_key` | `str` | *required* | Context key for the reference unit |
| `origin_id` | `str \| None` | `None` | Override origin (defaults to current actor) |
| `origin_key` | `str \| None` | `None` | Context key for origin unit ID |
| `operator` | `str` | `">"` | Comparison: `">"`, `">="`, `"=="`, `"<="`, `"<"` |

---

### Unit Filters

#### `TeamFilter`

**Description:** Filters by team relationship to the current actor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relation` | `str` | *required* | `"FRIENDLY"`, `"ENEMY"`, or `"SELF"` |

**Note:** `"FRIENDLY"` excludes self. Use `"SELF"` to match only the actor.

---

#### `UnitTypeFilter`

**Description:** Filters by unit type.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unit_type` | `str` | *required* | `"HERO"` or `"MINION"` |

---

#### `ImmunityFilter`

**Description:** Filters out immune units (heavy minions with support, or units with `ATTACK_IMMUNITY` effects). **Automatically added** to `SelectStep` for `UNIT` selections unless `skip_immunity_filter=True`.

No parameters.

---

### Hex Filters

#### `ObstacleFilter`

**Description:** Filters hexes by occupancy status.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_obstacle` | `bool` | `False` | `False` = must be empty, `True` = must be occupied |
| `exclude_id` | `str \| None` | `None` | Ignore this entity when checking occupancy |

---

#### `TerrainFilter`

**Description:** Filters hexes by terrain status.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_terrain` | `bool` | `True` | `True` = must be terrain, `False` = must not be terrain |

---

#### `SpawnPointFilter`

**Description:** Filters hexes by spawn point presence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `has_spawn_point` | `bool` | `False` | Whether hex must have a spawn point |

---

#### `AdjacentSpawnPointFilter`

**Description:** Filters hexes based on proximity to spawn points.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_empty` | `bool` | `True` | Only count empty spawn points |
| `must_not_have` | `bool` | `True` | `True` = must NOT be adjacent to spawn, `False` = must be adjacent |

---

#### `FastTravelDestinationFilter`

**Description:** Filters hexes to valid Fast Travel destinations (safe zone, empty).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unit_id` | `str \| None` | `None` | Unit doing the travel |

---

### Identity Filters

#### `ExcludeIdentityFilter`

**Description:** Excludes specific unit IDs from selection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exclude_self` | `bool` | `True` | Exclude the current actor |
| `exclude_keys` | `List[str]` | `[]` | Context keys whose values to exclude (supports list values) |

---

### Validation Filters

#### `HasEmptyNeighborFilter`

**Description:** Ensures candidate has at least one valid empty neighboring hex. Use before move/nudge effects.

No parameters.

---

#### `ForcedMovementByEnemyFilter`

**Description:** Checks if candidate can be displaced by the current actor. Delegates to `ValidationService.can_be_placed()`.

No parameters.

---

#### `CanBePlacedByActorFilter`

**Description:** Filters out units that cannot be placed by the current actor. Delegates to `ValidationService.can_be_placed()`.

No parameters.

---

## C. Pattern Library

Common card text patterns with copy-paste step sequences. Each pattern references real hero implementations.

---

### 1. "Target a unit adjacent to you"

Basic melee attack.

```python
AttackSequenceStep(damage=stats.primary_value, range_val=1)
```

**Example:** Xargatha — Cleave (`xargatha_effects.py`)

---

### 2. "Target a unit in range"

Standard ranged attack using the card's range stat.

```python
AttackSequenceStep(damage=stats.primary_value, range_val=stats.range)
```

**Example:** Arien — Rogue Wave (`arien_effects.py`)

---

### 3. "Not in a straight line"

Attack that excludes targets in a straight line from the actor.

```python
AttackSequenceStep(
    damage=stats.primary_value,
    range_val=stats.range,
    target_filters=[NotInStraightLineFilter()],
)
```

**Example:** Wasp — Charged Boomerang (`wasp_effects.py`)

---

### 4. "+N Attack for each adjacent enemy"

Dynamic damage bonus based on adjacent enemies. Use `subtract=1` to exclude the target ("other enemies").

```python
CountAdjacentEnemiesStep(output_key="adj_atk_bonus", multiplier=1, subtract=1),
AttackSequenceStep(
    damage=stats.primary_value,
    range_val=1,
    damage_bonus_key="adj_atk_bonus",
),
```

**Example:** Xargatha — Threatening Slash (+1), Deadly Swipe (+2), Lethal Spin (+3) (`xargatha_effects.py`)

---

### 5. "+N Range for each adjacent enemy"

Dynamic range bonus based on adjacent enemies. Use `subtract=0` (count all adjacent enemies).

```python
CountAdjacentEnemiesStep(output_key="adj_rng_bonus", multiplier=1, subtract=0),
AttackSequenceStep(
    damage=stats.primary_value,
    range_val=stats.range or 1,
    range_bonus_key="adj_rng_bonus",
),
```

**Example:** Xargatha — Long Thrust (`xargatha_effects.py`)

---

### 6. "May repeat once"

Optional repeat of the same action.

```python
AttackSequenceStep(damage=stats.primary_value, range_val=1),
MayRepeatOnceStep(
    steps_template=[
        AttackSequenceStep(damage=stats.primary_value, range_val=1),
    ],
),
```

**Example:** Xargatha — Cleave (`xargatha_effects.py`)

---

### 7. "May repeat on a different target"

Repeat with exclusion filter to prevent targeting the same unit.

```python
AttackSequenceStep(damage=stats.primary_value, range_val=1),
MayRepeatOnceStep(
    steps_template=[
        AttackSequenceStep(
            damage=stats.primary_value,
            range_val=1,
            target_filters=[
                UnitTypeFilter(unit_type="HERO"),  # Optional: restrict to heroes
                ExcludeIdentityFilter(exclude_keys=["victim_id"]),
            ],
        ),
    ],
),
```

**Example:** Xargatha — Cleave (`xargatha_effects.py`), Rapid Thrusts (`xargatha_effects.py`)

---

### 8. "Swap with an enemy minion"

Select minion, then swap positions.

```python
SelectStep(
    target_type=TargetType.UNIT,
    filters=[
        UnitTypeFilter(unit_type="MINION"),
        TeamFilter(relation="ENEMY"),
        RangeFilter(max_range=stats.range),
    ],
    prompt="Select an enemy minion to swap with.",
    output_key="swap_target_id",
),
SwapUnitsStep(unit_a_id=hero.id, unit_b_key="swap_target_id"),
```

**Example:** Arien — Arcane Whirlpool (`arien_effects.py`)

---

### 9. "Place yourself in range"

Teleport to a hex in range. Add `SpawnPointFilter` / `AdjacentSpawnPointFilter` as needed.

```python
SelectStep(
    target_type=TargetType.HEX,
    prompt="Select destination",
    output_key="target_hex",
    filters=[
        RangeFilter(max_range=stats.range),
        ObstacleFilter(is_obstacle=False),
        SpawnPointFilter(has_spawn_point=False),  # Optional constraint
    ],
    is_mandatory=True,
),
PlaceUnitStep(unit_id=hero.id, destination_key="target_hex"),
```

**Example:** Arien — Liquid Leap (`arien_effects.py`)

---

### 10. "Push up to N spaces"

Select push distance, then push. Distance selection via `SelectStep(NUMBER)`.

```python
# After selecting target (e.g., via AttackSequenceStep or SelectStep)...
SelectStep(
    target_type=TargetType.NUMBER,
    prompt="Choose push distance (0-2)",
    output_key="push_distance",
    number_options=[0, 1, 2],
    active_if_key="push_target_id",
),
PushUnitStep(
    target_key="push_target_id",
    distance_key="push_distance",
    active_if_key="push_target_id",
    is_mandatory=False,
),
```

**Example:** Arien — Rogue Wave (push 0-2), Tidal Blast (push 0-3) (`arien_effects.py`)

---

### 11. "Discard a card, if able"

Safe discard — does nothing if victim has no cards.

```python
ForceDiscardStep(victim_key="shock_victim")
```

**Example:** Wasp — Shock, Electrocute (`wasp_effects.py`)

---

### 12. "Discard a card, or is defeated"

Forced discard — if no cards, defeat instead.

```python
ForceDiscardOrDefeatStep(victim_key="backstab_victim_id")
```

**Example:** Arien — Dangerous Current (`arien_effects.py`)

---

### 13. "This turn: cannot..."

Create a `TARGET_PREVENTION` or `MOVEMENT_ZONE` effect with appropriate scope and restrictions.

```python
# "This turn: Enemy heroes in radius cannot perform skill actions, except on gold cards."
CreateEffectStep(
    effect_type=EffectType.TARGET_PREVENTION,
    scope=EffectScope(
        shape=Shape.RADIUS,
        range=stats.radius or 3,
        origin_id=hero.id,
        affects=AffectsFilter.ENEMY_HEROES,
    ),
    duration=DurationType.THIS_TURN,
    restrictions=[ActionType.SKILL],
    except_card_colors=[CardColor.GOLD],
),
```

```python
# "This turn: Enemy heroes adjacent to you cannot move more than 1 space."
CreateEffectStep(
    effect_type=EffectType.MOVEMENT_ZONE,
    scope=EffectScope(
        shape=Shape.ADJACENT,
        origin_id=hero.id,
        affects=AffectsFilter.ENEMY_HEROES,
    ),
    duration=DurationType.THIS_TURN,
    max_value=1,
    limit_actions_only=True,
    restrictions=[ActionType.FAST_TRAVEL],  # Also block fast travel
),
```

**Examples:** Arien — Spell Break, Slippery Ground, Deluge (`arien_effects.py`)

---

### 14. "Block a ranged attack"

Defense effect using `build_defense_steps`. Sets `auto_block` for ranged, `defense_invalid` for melee.

```python
def build_defense_steps(self, state, defender, card, stats, context):
    if context.get("attack_is_ranged"):
        return [SetContextFlagStep(key="auto_block", value=True)]
    else:
        return [SetContextFlagStep(key="defense_invalid", value=True)]
```

**Example:** Wasp — Stop Projectiles (`wasp_effects.py`)

---

### 15. "If you do" (on block)

On-block effects via `build_on_block_steps`. Only called when `block_succeeded=True`.

```python
def build_on_block_steps(self, state, defender, card, stats, context):
    return [
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select enemy hero in range to discard",
            output_key="reflect_victim",
            is_mandatory=False,
            filters=[
                UnitTypeFilter(unit_type="HERO"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=stats.range or 3),
            ],
        ),
        ForceDiscardStep(victim_key="reflect_victim"),
    ]
```

**Example:** Wasp — Reflect Projectiles, Deflect Projectiles (`wasp_effects.py`)

---

### 16. "Before/after action passive"

Passive abilities via `get_passive_config()` and `get_passive_steps()`.

```python
def get_passive_config(self):
    from goa2.engine.effects import PassiveConfig
    from goa2.domain.models.enums import PassiveTrigger

    return PassiveConfig(
        trigger=PassiveTrigger.BEFORE_ATTACK,
        uses_per_turn=1,
        is_optional=True,
        prompt="Move 1 space before attacking?",
    )

def get_passive_steps(self, state, hero, card, trigger, context):
    return [
        MoveSequenceStep(unit_id=hero.id, range_val=1, is_mandatory=False),
    ]
```

**Available triggers:** `BEFORE_ATTACK`, `BEFORE_MOVEMENT`, `BEFORE_SKILL`, `AFTER_BASIC_SKILL`

**Example:** Arien — Living Tsunami (`arien_effects.py`), Wasp — High Voltage (`wasp_effects.py`)

---

### 17. "Push up to N enemy units"

Multi-select targets, then process each with `ForEachStep`.

```python
MultiSelectStep(
    target_type=TargetType.UNIT,
    prompt="Select up to 2 adjacent enemies to push",
    output_key="push_targets",
    max_selections=2,
    min_selections=0,
    is_mandatory=False,
    filters=[RangeFilter(max_range=1), TeamFilter(relation="ENEMY")],
),
ForEachStep(
    list_key="push_targets",
    item_key="current_push_target",
    steps_template=[
        PushUnitStep(
            target_key="current_push_target",
            distance=3,
            collision_output_key="push_collision",
        ),
        # Optional: Check collision + unit type for conditional effects
        CheckUnitTypeStep(unit_key="current_push_target", expected_type="HERO", output_key="is_hero"),
        CombineBooleanContextStep(key_a="push_collision", key_b="is_hero", output_key="should_discard"),
        ForceDiscardStep(victim_key="current_push_target", active_if_key="should_discard"),
    ],
),
```

**Example:** Wasp — Kinetic Repulse, Kinetic Blast (`wasp_effects.py`)

---

### 18. "Move preserving distance" (Orbit)

Move a unit without changing its distance from the actor.

```python
SelectStep(
    target_type=TargetType.UNIT_OR_TOKEN,
    prompt="Select unit to orbit",
    output_key="orbit_target",
    is_mandatory=True,
    filters=[RangeFilter(max_range=stats.radius or 2)],
),
SelectStep(
    target_type=TargetType.HEX,
    prompt="Select orbit destination",
    output_key="orbit_dest",
    is_mandatory=True,
    filters=[
        AdjacencyToContextFilter(target_key="orbit_target"),
        RelativeDistanceFilter(reference_key="orbit_target", operator="=="),
        ObstacleFilter(is_obstacle=False),
    ],
),
PlaceUnitStep(unit_key="orbit_target", destination_key="orbit_dest"),
```

**Example:** Wasp — Lift Up, Control Gravity, Center of Mass (`wasp_effects.py`)

---

### 19. "If target was a hero, then..."

Conditional effects based on unit type. Use `CheckUnitTypeStep` + `active_if_key`.

```python
# After attack on "thunder_target_1"...
CheckUnitTypeStep(
    unit_key="thunder_target_1",
    expected_type="HERO",
    output_key="can_repeat_thunder",
),
MayRepeatOnceStep(
    active_if_key="can_repeat_thunder",
    steps_template=[
        AttackSequenceStep(
            damage=stats.primary_value,
            range_val=stats.range,
            target_filters=[ExcludeIdentityFilter(exclude_keys=["thunder_target_1"])],
        ),
    ],
),
```

**Example:** Wasp — Thunder Boomerang (`wasp_effects.py`)

---

## D. Multi-Piece Heroes (Razzle)

A multi-piece hero exists on the board only as `HeroPiece` entities
(`hero_razzle_piece_1..4`); the owner ID (`hero_razzle`) has **no board
position**. Effects written with the safe primitives below work for every
hero automatically — a conventions test
(`tests/engine/test_multipiece_conventions.py`) fails the build if a script
under `src/goa2/scripts/` reads raw location dicts or type-checks `Hero`
directly.

### Rules of thumb

1. **Never read `state.entity_locations` / `state.unit_locations` directly.**
   Use:
   - `state.get_position(id)` — one position; resolves the acting piece for a
     bound multi-piece hero, `None` when unbound.
   - `state.get_positions(id)` — all board positions (each piece).
   - `state.has_board_presence(id)` — replaces `id in entity_locations`.
2. **Enumerate hero board presence with `state.get_piece_ids(hero_id)`** —
   returns `[hero_id]` for a normal on-board hero, all piece IDs for a
   multi-piece hero, `[]` if off-board. Required whenever you loop
   `team.heroes` and care about positions/targeting (see Death Trap in
   `dodger_effects.py` or the marker AoE in `steps/markers.py`).
3. **Type checks: `is_hero_unit(entity)`** (from `goa2.domain.models`), never
   `isinstance(entity, Hero)` — pieces must count as hero units.
4. **Identity ("is this you?"): compare `state.hero_owner_id(a) ==
   state.hero_owner_id(b)`.** Raw ID equality misses piece-vs-owner pairs.
   `AffectsFilter.SELF` / `FRIENDLY_HEROES` and `blocks_self` already do this.
5. **Targets and victims carry piece IDs.** Anything you store for later board
   use (persistent effect origins, delayed returns/swaps) must keep the piece
   ID. `EffectManager.create_effect` already binds implicit origins to the
   acting piece. Player-level lookups (`state.get_hero(piece_id)`) resolve to
   the owner automatically, and input prompts normalize `player_id` to the
   owner.
6. **Stats:** `compute_card_stats`/`get_computed_stat` handle everything —
   acting-piece redirect for attack/defense/movement, owner-level initiative
   aggregation (each positional effect/marker counted once across pieces).
   Don't hand-roll positional stat reads.

### What you get for free

`SelectStep`/filters offer pieces as independent enemy hero units; the acting
piece is excluded as "self" while her other pieces stay selectable; defense
windows route to the owner with `defender_id` keeping the piece; markers are a
shared hero inventory (a marker on any piece affects every piece's
attack/defense and counts once for initiative); piece defeat
cascades to hero defeat.
