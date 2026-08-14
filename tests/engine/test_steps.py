import pytest

from goa2.domain.board import Board
from goa2.domain.hex import Hex
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardTier,
    Hero,
    Minion,
    MinionType,
    Team,
    TeamColor,
)
from goa2.domain.models.enums import TargetType
from goa2.domain.state import GameState
from goa2.domain.tile import Tile
from goa2.domain.types import CardID, HeroID
from goa2.engine.filters import RangeFilter, TeamFilter
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckUnitTypeStep,
    ForEachStep,
    LogMessageStep,
    MoveUnitStep,
    MultiSelectStep,
    PushUnitStep,
    ReactionWindowStep,
    ResolveCombatStep,
    SelectStep,
    SetContextFlagStep,
)

# --- Fixtures ---


@pytest.fixture
def empty_state():
    board = Board()
    hero_red = Hero(id="hero_red", name="Red Hero", team=TeamColor.RED, deck=[])
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[hero_red], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
        current_actor_id="hero_red",
    )
    actor_hex = Hex(q=2, r=0, s=-2)
    board.tiles[actor_hex] = board.get_tile(actor_hex)
    state.place_entity("hero_red", actor_hex)
    return state


@pytest.fixture
def populated_state():
    h1 = Hero(id="h1", name="Hero1", team=TeamColor.RED, deck=[])
    m1 = Minion(id="m1", name="Minion1", type=MinionType.MELEE, team=TeamColor.RED)

    # Create Board with necessary tiles
    start_hex = Hex(q=0, r=0, s=0)
    target_hex = Hex(q=1, r=0, s=-1)

    board = Board()
    board.tiles[start_hex] = Tile(hex=start_hex)
    board.tiles[target_hex] = Tile(hex=target_hex)

    # Use Unified Placement
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[h1], minions=[m1]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
        entity_locations={},
    )
    state.place_entity("h1", start_hex)
    return state


@pytest.fixture
def combat_state(empty_state):
    # Setup Red Attacker
    attacker = Hero(name="Red Hero", id=HeroID("hero_red"), team=TeamColor.RED, deck=[])
    empty_state.teams[TeamColor.RED].heroes.append(attacker)

    # Setup Blue Defender with Defense Cards
    def_card = Card(
        id=CardID("def_card_1"),
        name="Shield",
        tier=CardTier.I,
        color=CardColor.BLUE,
        initiative=1,
        primary_action=ActionType.DEFENSE,
        primary_action_value=5,
        effect_id="e1",
        effect_text="e1",
        is_facedown=False,
    )
    # Ensure it's in hand
    defender = Hero(name="Blue Hero", id=HeroID("hero_blue"), team=TeamColor.BLUE, deck=[def_card])
    defender.hand.append(def_card)

    empty_state.teams[TeamColor.BLUE].heroes.append(defender)

    return empty_state


# --- Tests ---


def test_select_target_flow(empty_state):
    # SelectStep requires at least 1 candidate to not auto-finish with "No candidates"
    # So we must spoof a unit location
    empty_state.place_entity("target_1", Hex(q=0, r=0, s=0))
    # Note: SelectStep now filters for actual units. We need to add target_1 to a team or mock get_unit.
    # We'll use a real unit ID "hero_red" which exists in empty_state default (no, default has empty heroes)

    # Add a mock hero so filtering passes
    hero = Hero(id="target_1", name="Target", team=TeamColor.BLUE, deck=[])
    empty_state.teams[TeamColor.BLUE].heroes.append(hero)

    step = SelectStep(target_type="UNIT", prompt="Choose", output_key="target_id")
    push_steps(empty_state, [step])

    # Pass 1: Request
    req = process_stack(empty_state).input_request
    assert req is not None
    assert req["type"] == "SELECT_UNIT"

    # Pass 2: Provide Input
    empty_state.execution_stack[-1].pending_input = {"selection": "target_1"}
    req = process_stack(empty_state).input_request

    assert req is None  # Done
    assert empty_state.execution_context["target_id"] == "target_1"


def test_reaction_window_validation(combat_state):
    # Target Blue Hero
    combat_state.execution_context["target_id"] = "hero_blue"

    step = ReactionWindowStep(target_player_key="target_id")
    push_steps(combat_state, [step])

    # Pass 1: Request Input
    req = process_stack(combat_state).input_request
    assert req["type"] == "SELECT_CARD_OR_PASS"
    assert any(o["id"] == "def_card_1" for o in req["options"])
    assert any(o["id"] == "PASS" for o in req["options"])

    # Pass 2: Select Invalid Card
    combat_state.execution_stack[-1].pending_input = {"selection": "def_card_1"}
    _ = process_stack(combat_state).input_request

    assert combat_state.execution_context["defense_value"] == 5


def test_combat_resolution_block(combat_state):
    # Defense 5 vs Attack 3 -> Blocked
    combat_state.execution_context["defense_value"] = 5
    combat_state.execution_context["victim_id"] = "hero_blue"

    step = ResolveCombatStep(damage=3, target_key="victim_id")
    push_steps(combat_state, [step])

    _ = process_stack(combat_state).input_request


def test_combat_resolution_hit(combat_state):
    # Defense 0 vs Attack 3 -> Hit
    combat_state.execution_context["defense_value"] = 0
    combat_state.execution_context["victim_id"] = "hero_blue"

    step = ResolveCombatStep(damage=3, target_key="victim_id")
    push_steps(combat_state, [step])

    _ = process_stack(combat_state).input_request


def test_attack_sequence_expansion(combat_state):
    # Add an enemy so SelectStep pauses for input
    # combat_state has Red Hero (current actor). Add Blue Hero in Range 1.
    target_hex = Hex(q=1, r=0, s=-1)
    combat_state.place_entity("hero_blue", target_hex)
    combat_state.place_entity("hero_red", Hex(q=0, r=0, s=0))

    # Check that the Macro step expands into 3 steps
    step = AttackSequenceStep(damage=3, range_val=1)
    push_steps(combat_state, [step])

    # Run once to expand
    _ = process_stack(combat_state).input_request

    # Now stack should have SelectStep at top waiting for input
    current = combat_state.execution_stack[-1]
    assert isinstance(current, SelectStep)


def test_attack_sequence_missing_preselected_target_aborts(combat_state):
    result = AttackSequenceStep(
        damage=3,
        range_val=1,
        target_id_key="restricted_target",
    ).resolve(combat_state, combat_state.execution_context)

    assert result.abort_action is True
    assert result.new_steps == []


def test_optional_attack_sequence_missing_preselected_target_fizzles(combat_state):
    result = AttackSequenceStep(
        damage=3,
        range_val=1,
        target_id_key="restricted_target",
        is_mandatory=False,
    ).resolve(combat_state, combat_state.execution_context)

    assert result.abort_action is False
    assert result.new_steps == []


def test_attack_sequence_uses_preselected_target_without_picker(combat_state):
    combat_state.execution_context["restricted_target"] = "hero_blue"

    result = AttackSequenceStep(
        damage=3,
        range_val=1,
        target_id_key="restricted_target",
    ).resolve(combat_state, combat_state.execution_context)

    assert not any(isinstance(step, SelectStep) for step in result.new_steps)
    assert isinstance(result.new_steps[0], ReactionWindowStep)
    assert result.new_steps[0].target_player_key == "restricted_target"


def test_attack_sequence_can_explicitly_pick_and_store_own_target(combat_state):
    result = AttackSequenceStep(
        damage=3,
        range_val=1,
        target_output_key="picked_target",
    ).resolve(combat_state, combat_state.execution_context)

    assert isinstance(result.new_steps[0], SelectStep)
    assert result.new_steps[0].output_key == "picked_target"


def test_attack_sequence_uses_output_fallback_when_preselection_is_missing(combat_state):
    result = AttackSequenceStep(
        damage=3,
        target_id_key="preferred_target",
        target_output_key="fallback_target",
    ).resolve(combat_state, combat_state.execution_context)

    assert isinstance(result.new_steps[0], SelectStep)
    assert result.new_steps[0].output_key == "fallback_target"


def test_attack_sequence_prefers_preselected_target_over_output(combat_state):
    combat_state.execution_context["preferred_target"] = "hero_blue"

    result = AttackSequenceStep(
        damage=3,
        target_id_key="preferred_target",
        target_output_key="fallback_target",
    ).resolve(combat_state, combat_state.execution_context)

    assert not any(isinstance(step, SelectStep) for step in result.new_steps)
    assert isinstance(result.new_steps[0], ReactionWindowStep)
    assert result.new_steps[0].target_player_key == "preferred_target"


def test_attack_sequence_target_keys_round_trip(combat_state):
    combat_state.execution_stack = [
        AttackSequenceStep(
            damage=3,
            target_id_key="preferred_target",
            target_output_key="fallback_target",
        )
    ]

    restored = GameState.model_validate_json(combat_state.model_dump_json())

    assert isinstance(restored.execution_stack[0], AttackSequenceStep)
    assert restored.execution_stack[0].target_id_key == "preferred_target"
    assert restored.execution_stack[0].target_output_key == "fallback_target"


def test_attack_sequence_is_ranged_false(combat_state):
    # Test explicit is_ranged=False sets context correctly
    combat_state.place_entity("hero_blue", Hex(q=1, r=0, s=-1))
    combat_state.place_entity("hero_red", Hex(q=0, r=0, s=0))

    step = AttackSequenceStep(damage=3, range_val=1, is_ranged=False)
    step.resolve(combat_state, combat_state.execution_context)

    assert combat_state.execution_context["attack_is_ranged"] is False


def test_attack_sequence_is_ranged_true(combat_state):
    # Test explicit is_ranged=True sets context correctly
    combat_state.place_entity("hero_blue", Hex(q=3, r=0, s=-3))
    combat_state.place_entity("hero_red", Hex(q=0, r=0, s=0))

    step = AttackSequenceStep(damage=3, range_val=3, is_ranged=True)
    step.resolve(combat_state, combat_state.execution_context)

    assert combat_state.execution_context["attack_is_ranged"] is True


def test_attack_sequence_default_is_ranged(combat_state):
    # Test default is_ranged=False (backward compatibility)
    combat_state.place_entity("hero_blue", Hex(q=3, r=0, s=-3))
    combat_state.place_entity("hero_red", Hex(q=0, r=0, s=0))

    # Explicit is_ranged not set, should default to False
    step = AttackSequenceStep(damage=3, range_val=3)
    step.resolve(combat_state, combat_state.execution_context)

    assert combat_state.execution_context["attack_is_ranged"] is False


# --- Pathfinding Tests ---


def test_move_unit_pathfinding(populated_state):
    # Valid Move: 1 step
    populated_state.execution_context["target_hex"] = Hex(q=1, r=0, s=-1)
    step = MoveUnitStep(unit_id="h1", range_val=1)

    res = step.resolve(populated_state, populated_state.execution_context)
    assert res.is_finished
    assert populated_state.entity_locations["h1"] == Hex(q=1, r=0, s=-1)


def test_move_unit_invalid_path(populated_state):
    # Invalid Move: 5 steps away, range 1
    populated_state.execution_context["target_hex"] = Hex(q=5, r=0, s=-5)
    step = MoveUnitStep(unit_id="h1", range_val=1)

    start_loc = populated_state.entity_locations["h1"]
    res = step.resolve(populated_state, populated_state.execution_context)

    assert res.is_finished
    # Should NOT have moved
    assert populated_state.entity_locations["h1"] == start_loc


# --- Error Handling Tests ---


def test_move_unit_error_handling(empty_state):
    # No actor
    step = MoveUnitStep(unit_id=None, destination_key="dest")
    empty_state.current_actor_id = None
    res = step.resolve(empty_state, {"dest": Hex(q=0, r=0, s=0)})
    assert res.is_finished

    # No destination
    step2 = MoveUnitStep(unit_id="h1", destination_key="missing_key")
    res2 = step2.resolve(empty_state, {})
    assert res2.is_finished


def test_log_message(empty_state):
    step = LogMessageStep(message="Hello {name}")
    res = step.resolve(empty_state, {"name": "World"})
    assert res.is_finished


# --- Reaction Window Minion Tests ---


def test_reaction_window_minion_skip(combat_state):
    # Setup: Add a Minion to Blue Team
    m1 = Minion(id="minion_blue", name="Blue Minion", type=MinionType.MELEE, team=TeamColor.BLUE)
    combat_state.teams[TeamColor.BLUE].minions.append(m1)

    # Target the Minion
    combat_state.execution_context["target_id"] = "minion_blue"

    step = ReactionWindowStep(target_player_key="target_id")
    push_steps(combat_state, [step])

    # Run stack
    req = process_stack(combat_state).input_request

    # Assertions
    assert req is None  # Should not request input
    assert combat_state.execution_context.get("defense_value") is None
    assert not combat_state.execution_stack  # Stack should be empty


def test_reaction_window_hero_prompt(combat_state):
    # Target Blue Hero (who has cards, from fixture)
    combat_state.execution_context["target_id"] = "hero_blue"

    step = ReactionWindowStep(target_player_key="target_id")
    push_steps(combat_state, [step])

    # Run stack
    req = process_stack(combat_state).input_request

    # Assertions
    assert req is not None
    assert req["type"] == "SELECT_CARD_OR_PASS"
    assert req["player_id"] == "hero_blue"


# =============================================================================
# Multi-Unit State Fixture (for MultiSelect, ForEach, CheckUnitType, Push tests)
# =============================================================================


@pytest.fixture
def multi_unit_state():
    """
    State with multiple units for testing iteration and selection steps.

    Layout (5x5 grid around origin):
        Wasp at (0,0,0) - RED hero, current actor
        Enemy1 at (1,0,-1) - BLUE hero, adjacent
        Enemy2 at (-1,0,1) - BLUE hero, adjacent
        Enemy3 at (2,0,-2) - BLUE hero, range 2
        Minion1 at (0,1,-1) - BLUE minion, adjacent

    Obstacle (terrain) at (3,0,-3) for push testing.
    """
    board = Board()

    # Create hex grid
    for q in range(-3, 4):
        for r in range(-3, 4):
            s = -q - r
            if abs(s) <= 3:
                h = Hex(q=q, r=r, s=s)
                tile = Tile(hex=h)
                board.tiles[h] = tile

    # Add obstacle for push testing
    obstacle_hex = Hex(q=3, r=0, s=-3)
    if obstacle_hex in board.tiles:
        board.tiles[obstacle_hex].is_terrain = True

    # Create units
    wasp = Hero(id="wasp", name="Wasp", team=TeamColor.RED, deck=[], level=1)
    enemy1 = Hero(id="enemy1", name="Enemy1", team=TeamColor.BLUE, deck=[], level=1)
    enemy2 = Hero(id="enemy2", name="Enemy2", team=TeamColor.BLUE, deck=[], level=1)
    enemy3 = Hero(id="enemy3", name="Enemy3", team=TeamColor.BLUE, deck=[], level=1)
    minion1 = Minion(id="minion1", name="Blue Melee", type=MinionType.MELEE, team=TeamColor.BLUE)

    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[wasp], minions=[]),
            TeamColor.BLUE: Team(
                color=TeamColor.BLUE,
                heroes=[enemy1, enemy2, enemy3],
                minions=[minion1],
            ),
        },
    )

    # Place entities
    state.place_entity("wasp", Hex(q=0, r=0, s=0))
    state.place_entity("enemy1", Hex(q=1, r=0, s=-1))
    state.place_entity("enemy2", Hex(q=-1, r=0, s=1))
    state.place_entity("enemy3", Hex(q=2, r=0, s=-2))
    state.place_entity("minion1", Hex(q=0, r=1, s=-1))

    state.current_actor_id = "wasp"
    return state


# =============================================================================
# MultiSelectStep Tests
# =============================================================================


class TestMultiSelectStep:
    """Tests for MultiSelectStep - select up to N targets."""

    def test_returns_input_request(self, multi_unit_state):
        """MultiSelectStep requests input when candidates exist."""
        step = MultiSelectStep(
            target_type=TargetType.UNIT,
            prompt="Select up to 2 enemies",
            output_key="selected_targets",
            max_selections=2,
            filters=[RangeFilter(max_range=1), TeamFilter(relation="ENEMY")],
        )

        result = step.resolve(multi_unit_state, {})

        assert result.requires_input
        assert result.input_request["type"] == "SELECT_UNIT"
        assert "candidates" in result.input_request
        assert len(result.input_request["candidates"]) >= 2

    def test_accumulates_selections(self, multi_unit_state):
        """Selections accumulate in internal state."""
        step = MultiSelectStep(
            target_type=TargetType.UNIT,
            prompt="Select up to 2 enemies",
            output_key="selected_targets",
            max_selections=2,
            filters=[RangeFilter(max_range=1), TeamFilter(relation="ENEMY")],
        )
        context = {}

        step.pending_input = {"selection": "enemy1"}
        result = step.resolve(multi_unit_state, context)

        assert step.selections == ["enemy1"]
        assert context["selected_targets"] == ["enemy1"]
        assert result.requires_input or not result.is_finished

    def test_finishes_at_max(self, multi_unit_state):
        """Step finishes when max_selections is reached."""
        step = MultiSelectStep(
            target_type=TargetType.UNIT,
            prompt="Select up to 2 enemies",
            output_key="selected_targets",
            max_selections=2,
            filters=[RangeFilter(max_range=1), TeamFilter(relation="ENEMY")],
        )
        context = {}

        step.pending_input = {"selection": "enemy1"}
        step.resolve(multi_unit_state, context)
        step.pending_input = None

        step.pending_input = {"selection": "enemy2"}
        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["selected_targets"] == ["enemy1", "enemy2"]

    def test_done_before_max(self, multi_unit_state):
        """Player can choose DONE before reaching max."""
        step = MultiSelectStep(
            target_type=TargetType.UNIT,
            prompt="Select up to 2 enemies",
            output_key="selected_targets",
            max_selections=2,
            min_selections=0,
            filters=[RangeFilter(max_range=1), TeamFilter(relation="ENEMY")],
        )
        context = {}

        step.pending_input = {"selection": "enemy1"}
        step.resolve(multi_unit_state, context)
        step.pending_input = None

        step.pending_input = {"selection": "DONE"}
        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["selected_targets"] == ["enemy1"]

    def test_excludes_already_selected(self, multi_unit_state):
        """Already-selected units are excluded from candidates."""
        step = MultiSelectStep(
            target_type=TargetType.UNIT,
            prompt="Select up to 2 enemies",
            output_key="selected_targets",
            max_selections=2,
            filters=[RangeFilter(max_range=1), TeamFilter(relation="ENEMY")],
        )
        context = {}

        step.pending_input = {"selection": "enemy1"}
        step.resolve(multi_unit_state, context)
        step.pending_input = None

        candidates = step._get_candidates(multi_unit_state, context)
        assert "enemy1" not in candidates

    def test_empty_list_if_no_candidates(self, multi_unit_state):
        """Empty list is stored if no candidates match filters."""
        step = MultiSelectStep(
            target_type=TargetType.UNIT,
            prompt="Select enemies at range 10",
            output_key="selected_targets",
            max_selections=2,
            filters=[RangeFilter(max_range=10, min_range=9)],
        )
        context = {}

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["selected_targets"] == []

    def test_rejects_selection_outside_candidate_set(self, multi_unit_state):
        """A client-submitted id that is filtered out must be rejected, not accepted."""
        step = MultiSelectStep(
            target_type=TargetType.UNIT,
            prompt="Select up to 2 enemies",
            output_key="selected_targets",
            max_selections=2,
            filters=[RangeFilter(max_range=1), TeamFilter(relation="ENEMY")],
        )
        context = {}

        # enemy3 is at range 2 -> excluded by RangeFilter(max_range=1).
        step.pending_input = {"selection": "enemy3"}
        result = step.resolve(multi_unit_state, context)

        # The invalid id is not accepted; the step re-requests input.
        assert "enemy3" not in step.selections
        assert context.get("selected_targets", []) == []
        assert result.requires_input

    def test_rejects_done_below_min_selections(self, multi_unit_state):
        """DONE/SKIP cannot finish the step while below min_selections."""
        step = MultiSelectStep(
            target_type=TargetType.UNIT,
            prompt="Select 2 enemies",
            output_key="selected_targets",
            max_selections=2,
            min_selections=2,
            filters=[RangeFilter(max_range=1), TeamFilter(relation="ENEMY")],
        )
        context = {}

        step.pending_input = {"selection": "DONE"}
        result = step.resolve(multi_unit_state, context)

        # Below min: DONE is ignored and the step keeps prompting for input.
        assert result.requires_input
        assert step.selections == []


# =============================================================================
# ForEachStep Tests
# =============================================================================


class TestForEachStep:
    """Tests for ForEachStep - iterate over context list."""

    def test_iterates_over_list(self, multi_unit_state):
        """ForEachStep iterates over all items."""
        context = {"targets": ["enemy1", "enemy2"]}

        step = ForEachStep(
            list_key="targets",
            item_key="current_target",
            steps_template=[
                SetContextFlagStep(key="processed", value=True),
            ],
        )

        result1 = step.resolve(multi_unit_state, context)
        assert context["current_target"] == "enemy1"
        assert not result1.is_finished
        assert len(result1.new_steps) == 1

        result2 = step.resolve(multi_unit_state, context)
        assert context["current_target"] == "enemy2"
        assert result2.is_finished
        assert len(result2.new_steps) == 1

    def test_empty_list(self, multi_unit_state):
        """ForEachStep finishes immediately with empty list."""
        context = {"targets": []}

        step = ForEachStep(
            list_key="targets",
            item_key="current_target",
            steps_template=[
                SetContextFlagStep(key="should_not_run", value=True),
            ],
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert result.new_steps is None or len(result.new_steps) == 0
        assert "should_not_run" not in context

    def test_missing_list_key(self, multi_unit_state):
        """ForEachStep handles missing list key gracefully."""
        context = {}

        step = ForEachStep(
            list_key="targets",
            item_key="current_target",
            steps_template=[
                SetContextFlagStep(key="should_not_run", value=True),
            ],
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert "should_not_run" not in context

    def test_deep_copies_template(self, multi_unit_state):
        """Template steps are deep copied for each iteration."""
        context = {"targets": ["enemy1", "enemy2"]}

        template_step = SetContextFlagStep(key="test", value=True)
        step = ForEachStep(
            list_key="targets",
            item_key="current_target",
            steps_template=[template_step],
        )

        result1 = step.resolve(multi_unit_state, context)
        result2 = step.resolve(multi_unit_state, context)

        assert result1.new_steps[0] is not result2.new_steps[0]
        assert result1.new_steps[0] is not template_step


# =============================================================================
# CheckUnitTypeStep Tests
# =============================================================================


class TestCheckUnitTypeStep:
    """Tests for CheckUnitTypeStep - check if unit is HERO/MINION."""

    def test_identifies_hero(self, multi_unit_state):
        """CheckUnitTypeStep correctly identifies heroes."""
        context = {"target": "enemy1"}

        step = CheckUnitTypeStep(
            unit_key="target",
            expected_type="HERO",
            output_key="is_hero",
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["is_hero"] is True

    def test_identifies_minion(self, multi_unit_state):
        """CheckUnitTypeStep correctly identifies minions."""
        context = {"target": "minion1"}

        step = CheckUnitTypeStep(
            unit_key="target",
            expected_type="MINION",
            output_key="is_minion",
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["is_minion"] is True

    def test_hero_is_not_minion(self, multi_unit_state):
        """Heroes are not identified as minions."""
        context = {"target": "enemy1"}

        step = CheckUnitTypeStep(
            unit_key="target",
            expected_type="MINION",
            output_key="is_minion",
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["is_minion"] is False

    def test_missing_unit_returns_false(self, multi_unit_state):
        """Missing unit returns False."""
        context = {}

        step = CheckUnitTypeStep(
            unit_key="target",
            expected_type="HERO",
            output_key="is_hero",
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["is_hero"] is False

    def test_direct_unit_id(self, multi_unit_state):
        """CheckUnitTypeStep works with direct unit_id parameter."""
        context = {}

        step = CheckUnitTypeStep(
            unit_id="enemy1",
            expected_type="HERO",
            output_key="is_hero",
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["is_hero"] is True


# =============================================================================
# PushUnitStep Collision Tests
# =============================================================================


class TestPushUnitStepCollision:
    """Tests for PushUnitStep collision detection via collision_output_key."""

    def test_no_collision(self, multi_unit_state):
        """Push without collision sets flag to False."""
        multi_unit_state.remove_entity("enemy3")
        context = {}

        step = PushUnitStep(
            target_id="enemy1",
            distance=1,
            collision_output_key="was_blocked",
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["was_blocked"] is False
        assert multi_unit_state.entity_locations["enemy1"] == Hex(q=2, r=0, s=-2)

    def test_collision_with_obstacle(self, multi_unit_state):
        """Push stopped by obstacle sets flag to True."""
        multi_unit_state.remove_entity("enemy3")
        multi_unit_state.move_unit("enemy1", Hex(q=2, r=0, s=-2))
        context = {}

        step = PushUnitStep(
            target_id="enemy1",
            distance=2,
            collision_output_key="was_blocked",
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["was_blocked"] is True

    def test_collision_with_board_edge(self, multi_unit_state):
        """Push stopped by board edge sets flag to True."""
        context = {}

        step = PushUnitStep(
            target_id="enemy2",
            distance=5,
            collision_output_key="was_blocked",
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert context["was_blocked"] is True

    def test_no_collision_key_doesnt_set_context(self, multi_unit_state):
        """Without collision_output_key, no context is set."""
        multi_unit_state.remove_entity("enemy3")
        context = {}

        step = PushUnitStep(
            target_id="enemy1",
            distance=1,
        )

        result = step.resolve(multi_unit_state, context)

        assert result.is_finished
        assert "was_blocked" not in context
