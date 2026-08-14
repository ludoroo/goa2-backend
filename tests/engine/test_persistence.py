"""Tests for Phase 6: State Persistence — serialization round-trips."""

import os
import tempfile

import pytest

from goa2.domain.board import Board
from goa2.domain.hex import Hex
from goa2.domain.models import CardState, GamePhase, Team, TeamColor
from goa2.domain.models.enums import TargetType
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.engine.filters import (
    RangeFilter,
    TeamFilter,
)
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.persistence import delete_game_save, load_all_games, load_game, save_game
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup
from goa2.engine.steps import (
    AskConfirmationStep,
    ForceDiscardStep,
    ForEachStep,
    LogMessageStep,
    MayRepeatNTimesStep,
    MoveUnitStep,
    SelectStep,
    TriggerMineStep,
)

MAP_PATH = "src/goa2/data/maps/forgotten_island.json"


@pytest.fixture
def save_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def full_state():
    """A fully initialized game state via GameSetup."""
    return GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"])


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------


def test_round_trip_fresh_game(full_state, save_dir):
    """Save and load a fresh game — all fields should match."""
    path = save_game(
        game_id="test123",
        state=full_state,
        player_tokens={"tok_a": "hero_arien", "tok_b": "hero_wasp"},
        spectator_token="spec_tok",
        hero_to_token={"hero_arien": "tok_a", "hero_wasp": "tok_b"},
        created_at=1000.0,
        save_dir=save_dir,
    )

    assert path.exists()
    data = load_game(str(path))

    assert data["game_id"] == "test123"
    assert data["player_tokens"] == {"tok_a": "hero_arien", "tok_b": "hero_wasp"}
    assert data["spectator_token"] == "spec_tok"
    assert data["hero_to_token"] == {"hero_arien": "tok_a", "hero_wasp": "tok_b"}
    assert data["created_at"] == 1000.0

    restored_state = data["session"].state
    assert restored_state.phase == full_state.phase
    assert restored_state.round == full_state.round
    assert len(restored_state.teams) == len(full_state.teams)


def test_round_trip_preserves_entity_locations(full_state, save_dir):
    """Entity locations survive round-trip."""
    original_locs = dict(full_state.entity_locations)
    assert len(original_locs) > 0  # Game has placed entities

    save_game(
        game_id="locs",
        state=full_state,
        player_tokens={},
        spectator_token="s",
        hero_to_token={},
        created_at=0,
        save_dir=save_dir,
    )
    data = load_game(os.path.join(save_dir, "locs.json"))
    restored = data["session"].state

    assert len(restored.entity_locations) == len(original_locs)
    for eid, _hex_val in original_locs.items():
        assert str(eid) in [str(k) for k in restored.entity_locations]


def test_round_trip_relinks_active_cards_to_master_deck(full_state):
    """Lifecycle containers must keep sharing the deck's canonical Card objects."""
    hero = full_state.get_hero("hero_arien")
    active = hero.hand[0].model_copy(deep=True)
    active.state = CardState.UNRESOLVED
    active.played_this_round = True
    hero.hand = [card for card in hero.hand if card.id != active.id]
    hero.current_turn_card = active
    full_state.pending_inputs[hero.id] = active

    restored = GameState.model_validate(full_state.model_dump(mode="json"))
    restored_hero = restored.get_hero("hero_arien")
    deck_card = next(card for card in restored_hero.deck if card.id == active.id)

    assert restored_hero.current_turn_card is deck_card
    assert restored.pending_inputs[restored_hero.id] is deck_card
    assert deck_card.state == CardState.UNRESOLVED
    assert deck_card.played_this_round is True


def test_round_trip_relinks_every_hand_card_to_master_deck(full_state):
    restored = GameState.model_validate(full_state.model_dump(mode="json"))

    for team in restored.teams.values():
        for hero in team.heroes:
            deck_by_id = {card.id: card for card in hero.deck}
            assert all(deck_by_id[card.id] is card for card in hero.hand)


# ---------------------------------------------------------------------------
# Steps on stack
# ---------------------------------------------------------------------------


def test_round_trip_with_steps_on_stack():
    """Steps on the execution stack survive round-trip with correct types."""
    board = Board()
    hero = Hero(id="hero_a", name="HeroA", team=TeamColor.RED, deck=[])
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[hero], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
        current_actor_id="hero_a",
    )
    h = Hex(q=0, r=0, s=0)
    board.tiles[h] = board.get_tile(h)
    state.place_entity("hero_a", h)

    push_steps(
        state,
        [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Pick target",
                filters=[RangeFilter(max_range=2), TeamFilter(relation="ENEMY")],
            ),
            MoveUnitStep(unit_id="hero_a"),
            LogMessageStep(message="done"),
        ],
    )

    # Round-trip via model serialization (no process_stack)
    data = state.model_dump(mode="json")
    restored = GameState.model_validate(data)

    assert len(restored.execution_stack) == 3
    assert type(restored.execution_stack[0]).__name__ == "LogMessageStep"
    assert type(restored.execution_stack[1]).__name__ == "MoveUnitStep"

    select = restored.execution_stack[2]
    assert type(select).__name__ == "SelectStep"
    assert len(select.filters) == 2
    assert type(select.filters[0]).__name__ == "RangeFilter"
    assert type(select.filters[1]).__name__ == "TeamFilter"


def _bare_state_with_hero(hero_id: str = "hero_a") -> GameState:
    hero = Hero(id=hero_id, name="Hero", team=TeamColor.RED, deck=[])
    return GameState(
        board=Board(),
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[hero], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
    )


def _round_trip_step(step, *, drop_fields: tuple[str, ...] = ()) -> object:
    """Serialize a state carrying ``step``, optionally strip fields from the
    step payload to model a pre-migration serialized shape, and re-validate.
    """
    state = _bare_state_with_hero()
    state.execution_stack = [step]
    data = state.model_dump(mode="json")
    if drop_fields:
        payload = data["execution_stack"][0]
        for field in drop_fields:
            payload.pop(field, None)
    return GameState.model_validate(data).execution_stack[0]


@pytest.mark.parametrize(
    "step_factory, drop_fields, expected",
    [
        # Older TriggerMineStep payloads had no literal fields at all.
        (
            lambda: TriggerMineStep(),
            ("mine_ids", "victim_id"),
            {
                "mine_ids": None,
                "victim_id": None,
                "mine_ids_key": "triggered_mine_ids",
                "victim_key": "mine_victim_id",
            },
        ),
        # Older ForceDiscardStep serialized only ``victim_key``.
        (
            lambda: ForceDiscardStep(victim_key="v"),
            ("victim_id",),
            {"victim_id": None, "victim_key": "v"},
        ),
        # Older SelectStep serialized only the ``*_key`` fields.
        (
            lambda: SelectStep(
                target_type=TargetType.CARD,
                prompt="pick",
                context_hero_id_key="v",
                override_player_id_key="v",
            ),
            ("context_hero_id", "override_player_id"),
            {
                "context_hero_id": None,
                "override_player_id": None,
                "context_hero_id_key": "v",
                "override_player_id_key": "v",
            },
        ),
    ],
    ids=["trigger_mine", "force_discard", "select"],
)
def test_legacy_step_payload_without_new_fields_round_trips(step_factory, drop_fields, expected):
    """Genuinely-old payloads (new fields absent) validate with default None literals."""
    step = _round_trip_step(step_factory(), drop_fields=drop_fields)
    for name, value in expected.items():
        assert getattr(step, name) == value


@pytest.mark.parametrize(
    "step_factory, expected",
    [
        (
            lambda: TriggerMineStep(mine_ids=["m1", "m2"], victim_id="hero_a"),
            {"mine_ids": ["m1", "m2"], "victim_id": "hero_a"},
        ),
        (
            lambda: ForceDiscardStep(victim_id="hero_a"),
            {"victim_id": "hero_a", "victim_key": None},
        ),
        (
            lambda: SelectStep(
                target_type=TargetType.CARD,
                prompt="pick",
                context_hero_id="hero_a",
                override_player_id="hero_a",
            ),
            {
                "context_hero_id": "hero_a",
                "override_player_id": "hero_a",
                "context_hero_id_key": None,
                "override_player_id_key": None,
            },
        ),
    ],
    ids=["trigger_mine", "force_discard", "select"],
)
def test_new_literal_step_payload_round_trips(step_factory, expected):
    step = _round_trip_step(step_factory())
    for name, value in expected.items():
        assert getattr(step, name) == value


def test_pending_input_request_id_survives_round_trip():
    state = GameState(
        board=Board(),
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
    )
    push_steps(state, [AskConfirmationStep(player_id="hero_a", prompt="Continue?")])
    first = process_stack(state).input_request
    assert first is not None

    restored = GameState.model_validate(state.model_dump(mode="json"))
    second = process_stack(restored).input_request

    assert second is not None
    assert second.id == first.id


# ---------------------------------------------------------------------------
# Nested step templates
# ---------------------------------------------------------------------------


def test_round_trip_foreach_step():
    """ForEachStep with steps_template round-trips via model serialization."""
    board = Board()
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
    )

    foreach = ForEachStep(
        list_key="targets",
        item_key="current",
        steps_template=[
            MoveUnitStep(unit_id="hero_a"),
            LogMessageStep(message="moved"),
        ],
    )
    state.execution_stack.append(foreach)

    data = state.model_dump(mode="json")
    restored = GameState.model_validate(data)

    step = restored.execution_stack[0]
    assert type(step).__name__ == "ForEachStep"
    assert len(step.steps_template) == 2
    assert type(step.steps_template[0]).__name__ == "MoveUnitStep"
    assert type(step.steps_template[1]).__name__ == "LogMessageStep"


def test_round_trip_may_repeat_step():
    """MayRepeatNTimesStep with nested steps_template round-trips."""
    board = Board()
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
    )

    repeat = MayRepeatNTimesStep(
        max_repeats=3,
        prompt="Again?",
        steps_template=[LogMessageStep(message="repeated")],
    )
    state.execution_stack.append(repeat)

    data = state.model_dump(mode="json")
    restored = GameState.model_validate(data)

    step = restored.execution_stack[0]
    assert type(step).__name__ == "MayRepeatNTimesStep"
    assert step.max_repeats == 3
    assert len(step.steps_template) == 1


# ---------------------------------------------------------------------------
# All step types serialize
# ---------------------------------------------------------------------------


def test_every_step_type_has_unique_discriminator():
    """Every concrete step class has a unique StepType (no GENERIC collisions)."""
    import inspect

    import goa2.engine.steps as steps_mod

    seen = {}
    for name, cls in inspect.getmembers(steps_mod, inspect.isclass):
        if not issubclass(cls, steps_mod.GameStep) or cls is steps_mod.GameStep:
            continue
        # MayRepeatOnceStep shares with MayRepeatNTimesStep intentionally
        if name == "MayRepeatOnceStep":
            continue
        step_type = cls.model_fields["type"].default
        if step_type in seen:
            pytest.fail(f"{name} and {seen[step_type]} share StepType {step_type}")
        seen[step_type] = name


def test_step_registry_covers_concrete_step_classes():
    """Step serialization registry is derived from concrete subclasses."""
    import inspect
    from typing import get_args

    from goa2.domain.models.enums import StepType
    from goa2.engine import steps as steps_mod
    from goa2.engine.step_types import _registered_union

    any_step = _registered_union(
        steps_mod.GameStep,
        field_name="type",
        ignored_tags={StepType.GENERIC.value},
        ignored_classes={steps_mod.MayRepeatOnceStep},
        aliases={StepType.MAY_REPEAT_ONCE.value: steps_mod.MayRepeatNTimesStep},
    )
    union_members = get_args(any_step)
    registered_classes = {get_args(member)[0] for member in union_members}

    concrete_classes = {
        cls
        for _, cls in inspect.getmembers(steps_mod, inspect.isclass)
        if issubclass(cls, steps_mod.GameStep)
        and cls not in {steps_mod.GameStep, steps_mod.MayRepeatOnceStep}
    }

    assert registered_classes == concrete_classes


# ---------------------------------------------------------------------------
# Filter union
# ---------------------------------------------------------------------------


def _minimal_filter_instance(filter_class):
    """Construct every registered filter without maintaining a class allow-list."""
    from enum import Enum
    from typing import Literal, get_args, get_origin

    kwargs = {}
    for field_name, field_info in filter_class.model_fields.items():
        if field_name == "type" or not field_info.is_required():
            continue
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if annotation is str:
            value = "test"
        elif annotation is int:
            value = 1
        elif origin is list:
            value = []
        elif origin is Literal:
            value = get_args(annotation)[0]
        elif isinstance(annotation, type) and issubclass(annotation, Enum):
            value = next(iter(annotation))
        else:
            pytest.fail(
                f"No minimal value factory for {filter_class.__name__}.{field_name}: "
                f"{annotation!r}"
            )
        kwargs[field_name] = value
    return filter_class(**kwargs)


def test_all_filter_types_round_trip(save_dir):
    """Each filter subclass serializes and deserializes correctly."""
    import inspect

    from goa2.engine import filters as f_mod

    board = Board()
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
    )

    # Collect all concrete filter subclasses
    filter_classes = []
    for _name, cls in inspect.getmembers(f_mod, inspect.isclass):
        if issubclass(cls, f_mod.FilterCondition) and cls is not f_mod.FilterCondition:
            filter_classes.append(cls)

    assert len(filter_classes) >= 19  # Sanity check

    # Instantiate every class; adding a filter with a new required field must
    # update the generic value factory instead of silently skipping coverage.
    for fc in filter_classes:
        instance = _minimal_filter_instance(fc)

        step = SelectStep(
            target_type=TargetType.UNIT,
            prompt="test",
            filters=[instance],
        )
        state.execution_stack = [step]

        data = state.model_dump(mode="json")
        restored = GameState.model_validate(data)
        restored_filter = restored.execution_stack[0].filters[0]
        assert (
            type(restored_filter).__name__ == type(instance).__name__
        ), f"Filter {type(instance).__name__} did not round-trip correctly"
        assert restored_filter.model_dump(mode="json") == instance.model_dump(mode="json")


def test_every_filter_type_round_trips_at_arbitrary_composite_depth():
    """Every registered filter survives below all recursive composite containers."""
    import inspect

    from goa2.engine import filters as f_mod

    instances = [
        _minimal_filter_instance(cls)
        for _, cls in inspect.getmembers(f_mod, inspect.isclass)
        if issubclass(cls, f_mod.FilterCondition) and cls is not f_mod.FilterCondition
    ]
    nested = f_mod.OrFilter(
        filters=[
            f_mod.AndFilter(filters=[f_mod.CountMatchFilter(sub_filters=instances, min_count=2)])
        ]
    )
    state = GameState(
        board=Board(),
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
        execution_stack=[SelectStep(target_type=TargetType.UNIT, prompt="test", filters=[nested])],
    )

    restored = GameState.model_validate(state.model_dump(mode="json"))
    restored_or = restored.execution_stack[0].filters[0]
    restored_and = restored_or.filters[0]
    restored_count = restored_and.filters[0]
    restored_instances = restored_count.sub_filters

    assert len(restored_instances) == len(instances)
    for actual, expected in zip(restored_instances, instances, strict=True):
        assert type(actual) is type(expected)
        assert actual.model_dump(mode="json") == expected.model_dump(mode="json")


def test_filter_registry_covers_concrete_filter_classes():
    """Filter serialization registry is derived from concrete subclasses."""
    import inspect
    from typing import get_args

    from goa2.engine import filters as f_mod
    from goa2.engine.step_types import _registered_union

    any_filter = _registered_union(f_mod.FilterCondition, field_name="type")
    union_members = get_args(any_filter)
    registered_classes = {get_args(member)[0] for member in union_members}

    concrete_classes = {
        cls
        for _, cls in inspect.getmembers(f_mod, inspect.isclass)
        if issubclass(cls, f_mod.FilterCondition) and cls is not f_mod.FilterCondition
    }

    assert registered_classes == concrete_classes


# ---------------------------------------------------------------------------
# Re-derivation of last_result
# ---------------------------------------------------------------------------


def test_last_result_re_derived_on_load(save_dir):
    """When loading a game waiting for input, last_result is re-derived."""
    board = Board()
    hero_a = Hero(id="hero_a", name="HeroA", team=TeamColor.RED, deck=[])
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[hero_a], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
        current_actor_id="hero_a",
        phase=GamePhase.RESOLUTION,
    )

    h0 = Hex(q=0, r=0, s=0)
    board.tiles[h0] = board.get_tile(h0)
    state.place_entity("hero_a", h0)

    # Push a NUMBER SelectStep — always finds candidates (no board filtering)
    push_steps(
        state,
        [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Pick a number",
                output_key="chosen_number",
                number_options=[1, 2, 3],
            ),
        ],
    )

    # Process stack — pauses at SelectStep waiting for input
    result = process_stack(state)
    assert result.input_request is not None

    # Save while waiting for input
    save_game(
        game_id="input_pending",
        state=state,
        player_tokens={"tok": "hero_a"},
        spectator_token="s",
        hero_to_token={"hero_a": "tok"},
        created_at=0,
        save_dir=save_dir,
    )

    data = load_game(os.path.join(save_dir, "input_pending.json"))
    assert data["last_result"] is not None
    assert data["last_result"].input_request is not None


# ---------------------------------------------------------------------------
# load_all_games / delete
# ---------------------------------------------------------------------------


def test_load_all_games(full_state, save_dir):
    """load_all_games loads all JSON files in directory."""
    for i in range(3):
        save_game(
            game_id=f"game_{i}",
            state=full_state,
            player_tokens={},
            spectator_token="s",
            hero_to_token={},
            created_at=float(i),
            save_dir=save_dir,
        )

    games = load_all_games(save_dir)
    assert len(games) == 3
    ids = {g["game_id"] for g in games}
    assert ids == {"game_0", "game_1", "game_2"}


def test_load_all_games_skips_corrupt(full_state, save_dir):
    """Corrupt files are skipped without crashing."""
    save_game(
        game_id="good",
        state=full_state,
        player_tokens={},
        spectator_token="s",
        hero_to_token={},
        created_at=0,
        save_dir=save_dir,
    )
    # Write a corrupt file
    with open(os.path.join(save_dir, "bad.json"), "w") as f:
        f.write("{corrupt")

    games = load_all_games(save_dir)
    assert len(games) == 1
    assert games[0]["game_id"] == "good"


def test_load_all_games_empty_dir(save_dir):
    """Empty directory returns empty list."""
    assert load_all_games(save_dir) == []


def test_load_all_games_missing_dir():
    """Non-existent directory returns empty list."""
    assert load_all_games("/nonexistent/path") == []


def test_delete_game_save(full_state, save_dir):
    """delete_game_save removes the file."""
    save_game(
        game_id="del_me",
        state=full_state,
        player_tokens={},
        spectator_token="s",
        hero_to_token={},
        created_at=0,
        save_dir=save_dir,
    )
    assert os.path.exists(os.path.join(save_dir, "del_me.json"))

    delete_game_save("del_me", save_dir)
    assert not os.path.exists(os.path.join(save_dir, "del_me.json"))


def test_delete_nonexistent_save(save_dir):
    """Deleting a nonexistent save doesn't raise."""
    delete_game_save("nope", save_dir)  # Should not raise


# ---------------------------------------------------------------------------
# Mid-resolution round-trip (full game state)
# ---------------------------------------------------------------------------


def test_mid_resolution_round_trip(save_dir):
    """A game mid-resolution (with steps, context, etc.) round-trips correctly."""
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"])

    # Transition to resolution by committing cards for all heroes
    session = GameSession(state)
    for team in state.teams.values():
        for hero in team.heroes:
            if hero.hand:
                session.commit_card(hero.id, hero.hand[0])

    # Now in resolution — there should be steps on the stack
    save_game(
        game_id="midres",
        state=state,
        player_tokens={"t1": "hero_arien", "t2": "hero_wasp"},
        spectator_token="spec",
        hero_to_token={"hero_arien": "t1", "hero_wasp": "t2"},
        created_at=42.0,
        save_dir=save_dir,
    )

    data = load_game(os.path.join(save_dir, "midres.json"))
    restored = data["session"].state
    assert restored.phase == state.phase
    assert restored.round == state.round
    assert restored.turn == state.turn
