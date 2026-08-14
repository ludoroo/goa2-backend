from __future__ import annotations

import pytest

from goa2.domain.events import GameEventType
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequestType
from goa2.domain.models import CardState, Token, TokenType
from goa2.engine.effects import CardEffectRegistry
from goa2.engine.setup import GameSetup

from ..builders import EffectScenarioBuilder, hero_card
from ..runner import run_card


def _option_set(run) -> set:
    assert run.latest_request is not None
    options = set()
    for option in run.latest_request.options:
        if hasattr(option, "metadata") and option.metadata and "raw" in option.metadata:
            options.add(option.metadata.get("raw"))
        elif hasattr(option, "id"):
            options.add(option.id)
        else:
            options.add(option)
    return options


def _add_pyro_pool(state) -> None:
    state.token_pool[TokenType.PYRO] = []
    token = Token(
        id="pyro_1",
        name="Pyro",
        token_type=TokenType.PYRO,
        persists_end_of_round=True,
    )
    state.register_entity(token)
    state.token_pool[TokenType.PYRO].append(token)


def _place_passable_mine(state, at: Hex, *, owner_id: str, mine_id: str = "mine_1") -> None:
    """Put a passable mine on the board, owned by `owner_id` (a hero in state)."""
    mine = Token(
        id=mine_id,
        name="Mine",
        token_type=TokenType.MINE_DUD,
        owner_id=owner_id,
        is_passable=True,
    )
    state.register_entity(mine, "token")
    state.token_pool.setdefault(TokenType.MINE_DUD, []).append(mine)
    state.place_entity(mine_id, at)


def _activate_dragon_knight(state, hero_id: str = "hero_widget") -> None:
    """Configure widget so the dragon_knight ultimate passive is active."""
    from goa2.data.heroes.registry import HeroRegistry

    widget = state.get_hero(hero_id)
    assert widget is not None
    widget.level = 8
    template = HeroRegistry.get("Widget")
    assert template is not None and template.ultimate_card is not None
    widget.ultimate_card = template.ultimate_card


@pytest.mark.effect_contract
def test_widget_easy_effects_are_registered() -> None:
    for effect_id in [
        "dragon_bond",
        "take_off",
        "all_aboard",
        "safe_landing",
        "diversionary_strike",
        "fight_as_one",
        "diversionary_attack",
        "diversionary_assault",
        "airborne_attack",
        "airborne_assault",
        "nibble",
        "gnaw",
        "fiery_breath",
        "flaming_breath",
        "scorching_breath",
    ]:
        assert CardEffectRegistry.get(effect_id) is not None


@pytest.mark.effect_contract
def test_setup_creates_persistent_pyro_token() -> None:
    state = EffectScenarioBuilder().line_board().red_hero("hero_widget", at=(0, 0, 0)).build()
    GameSetup._initialize_token_pool(state)

    assert len(state.token_pool[TokenType.PYRO]) == 1
    assert state.token_pool[TokenType.PYRO][0].persists_end_of_round


@pytest.mark.effect_flow
def test_dragon_bond_places_pyro_in_radius() -> None:
    pyro_hex = Hex(q=1, r=0, s=-1)
    far_hex = Hex(q=3, r=0, s=-3)
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (3, 0, -3)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "dragon_bond"),
        )
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).expect_input(InputRequestType.SELECT_HEX)
    assert pyro_hex in _option_set(run)
    assert far_hex not in _option_set(run)

    run.choose(pyro_hex).finish()

    assert state.entity_locations["pyro_1"] == pyro_hex
    assert state.token_pool[TokenType.PYRO][0].persists_end_of_round
    assert any(e.event_type == GameEventType.TOKEN_PLACED for e in run.events)


@pytest.mark.effect_flow
def test_dragon_bond_move_branch_aborts_if_pyro_is_not_in_play() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "dragon_bond"),
        )
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(2).finish()

    assert "pyro_1" not in state.entity_locations
    assert state.entity_locations["hero_widget"] == Hex(q=0, r=0, s=0)


@pytest.mark.effect_flow
def test_dragon_bond_moves_pyro_then_widget_when_chosen() -> None:
    widget_start = Hex(q=0, r=0, s=0)
    pyro_start = Hex(q=2, r=0, s=-2)
    pyro_dest = Hex(q=3, r=0, s=-3)
    widget_dest = Hex(q=1, r=0, s=-1)
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (3, 0, -3)])
        .red_hero(
            "hero_widget",
            at=widget_start,
            current_card=hero_card("Widget", "dragon_bond"),
        )
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(2).expect_input(InputRequestType.SELECT_NUMBER)
    assert state.execution_context["dragon_bond_pyro"] == "pyro_1"

    run.choose(2).expect_input(InputRequestType.SELECT_HEX)
    assert pyro_dest in _option_set(run)

    run.choose(pyro_dest).expect_input(InputRequestType.SELECT_HEX)
    assert widget_dest in _option_set(run)

    run.choose(widget_dest).finish()

    assert state.entity_locations["pyro_1"] == pyro_dest
    assert state.entity_locations["hero_widget"] == widget_dest
    assert sum(e.event_type == GameEventType.TOKEN_MOVED for e in run.events) == 1
    assert sum(e.event_type == GameEventType.UNIT_MOVED for e in run.events) == 1


@pytest.mark.effect_flow
def test_take_off_swaps_pyro_with_widget() -> None:
    widget_start = Hex(q=0, r=0, s=0)
    pyro_start = Hex(q=2, r=0, s=-2)
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2)])
        .red_hero(
            "hero_widget",
            at=widget_start,
            current_card=hero_card("Widget", "take_off"),
        )
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    assert state.execution_context["pyro_swap_id"] == "pyro_1"

    assert _option_set(run) == {"hero_widget"}

    run.choose("hero_widget").finish()

    assert state.entity_locations["hero_widget"] == pyro_start
    assert state.entity_locations["pyro_1"] == widget_start
    assert any(e.event_type == GameEventType.UNITS_SWAPPED for e in run.events)


@pytest.mark.effect_flow
def test_safe_landing_may_move_pyro_then_swap_with_friendly_hero() -> None:
    pyro_start = Hex(q=2, r=0, s=-2)
    pyro_move_dest = Hex(q=2, r=1, s=-3)
    ally_start = Hex(q=1, r=0, s=-1)
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (2, 1, -3)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "safe_landing"),
        )
        .red_hero("friendly_widget_ally", at=ally_start)
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN)
    run.choose("pyro_1").expect_input(InputRequestType.SELECT_HEX)
    assert pyro_move_dest in _option_set(run)

    run.choose(pyro_move_dest).expect_input(InputRequestType.SELECT_UNIT)
    assert state.execution_context["pyro_swap_id"] == "pyro_1"
    assert _option_set(run) == {"hero_widget", "friendly_widget_ally"}

    run.choose("friendly_widget_ally").finish()

    assert state.entity_locations["pyro_1"] == ally_start
    assert state.entity_locations["friendly_widget_ally"] == pyro_move_dest
    assert any(e.event_type == GameEventType.TOKEN_MOVED for e in run.events)
    assert any(e.event_type == GameEventType.UNITS_SWAPPED for e in run.events)


@pytest.mark.effect_flow
def test_diversionary_strike_attacks_then_moves_pyro_up_to_two() -> None:
    pyro_start = Hex(q=1, r=1, s=-2)
    pyro_dest = Hex(q=3, r=1, s=-4)
    state = (
        EffectScenarioBuilder()
        .with_hexes(
            [
                (0, 0, 0),
                (1, 0, -1),
                (1, 1, -2),
                (2, 1, -3),
                (3, 1, -4),
            ]
        )
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "diversionary_strike"),
        )
        .blue_minion("blue_minion", at=(1, 0, -1))
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN)
    assert _option_set(run) == {"pyro_1"}

    run.choose("pyro_1").expect_input(InputRequestType.SELECT_HEX)
    assert pyro_dest in _option_set(run)

    run.choose(pyro_dest).finish()

    assert state.entity_locations["pyro_1"] == pyro_dest
    combat_events = [e for e in run.events if e.event_type == GameEventType.COMBAT_RESOLVED]
    assert combat_events
    assert combat_events[-1].metadata["attack_value"] == 5
    assert any(e.event_type == GameEventType.TOKEN_MOVED for e in run.events)


@pytest.mark.effect_flow
def test_fight_as_one_replays_resolved_skill_against_different_unit() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (0, 1, -1), (1, 0, -1), (1, 1, -2)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "fight_as_one"),
        )
        .blue_hero("blue_initial_target", at=(1, 0, -1))
        .blue_hero("blue_replay_target", at=(1, 1, -2))
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", Hex(q=0, r=1, s=-1))

    widget = state.get_hero("hero_widget")
    assert widget is not None
    played_skill = hero_card("Widget", "fiery_breath")
    played_skill.state = CardState.RESOLVED
    widget.played_cards.append(played_skill)

    initial_target = state.get_hero("blue_initial_target")
    assert initial_target is not None
    initial_target.hand.append(hero_card("Widget", "dragon_bond"))

    replay_target = state.get_hero("blue_replay_target")
    assert replay_target is not None
    replay_target.hand.append(hero_card("Widget", "all_aboard"))

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_initial_target").expect_input(InputRequestType.SELECT_CARD_OR_PASS)
    run.choose("dragon_bond").expect_input(InputRequestType.SELECT_CARD)
    assert _option_set(run) == {"fiery_breath"}

    run.choose("fiery_breath").expect_input(InputRequestType.SELECT_UNIT)
    assert state.execution_context["pyro_breath_id"] == "pyro_1"
    assert _option_set(run) == {"blue_replay_target"}

    run.choose("blue_replay_target").expect_input(InputRequestType.SELECT_CARD)
    run.choose("all_aboard").finish()

    assert state.execution_context["fight_as_one_initial_target"] == "blue_initial_target"
    assert state.entity_locations["blue_initial_target"] == Hex(q=1, r=0, s=-1)
    assert len(replay_target.hand) == 0
    assert replay_target.discard_pile[0].id == "all_aboard"


@pytest.mark.effect_flow
def test_airborne_assault_can_swap_before_and_after_attack() -> None:
    widget_start = Hex(q=0, r=0, s=0)
    pyro_start = Hex(q=2, r=0, s=-2)
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (3, 0, -3)])
        .red_hero(
            "hero_widget",
            at=widget_start,
            current_card=hero_card("Widget", "airborne_assault"),
        )
        .red_hero("friendly_widget_ally", at=(3, 0, -3))
        .blue_minion("blue_minion", at=(1, 0, -1))
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN)
    run.choose("pyro_1").expect_input(InputRequestType.SELECT_UNIT)
    assert _option_set(run) == {"hero_widget"}

    run.choose("hero_widget").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN)
    run.choose("pyro_1").expect_input(InputRequestType.SELECT_UNIT)
    assert _option_set(run) == {"hero_widget"}

    run.choose("hero_widget").finish()

    assert state.entity_locations["hero_widget"] == widget_start
    assert state.entity_locations["pyro_1"] == pyro_start
    assert sum(e.event_type == GameEventType.UNITS_SWAPPED for e in run.events) == 2
    combat_events = [e for e in run.events if e.event_type == GameEventType.COMBAT_RESOLVED]
    assert combat_events[-1].metadata["attack_value"] == 4


@pytest.mark.effect_flow
def test_nibble_removes_enemy_minion_adjacent_to_pyro_then_removes_pyro() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (3, 0, -3), (4, 0, -4)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "nibble"),
        )
        .blue_minion("blue_minion", at=(3, 0, -3))
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", Hex(q=4, r=0, s=-4))

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    assert state.execution_context["pyro_skill_id"] == "pyro_1"
    assert _option_set(run) == {"blue_minion"}

    run.choose("blue_minion").finish()

    assert "blue_minion" not in state.entity_locations
    assert "pyro_1" not in state.entity_locations
    assert any(e.event_type == GameEventType.UNIT_REMOVED for e in run.events)
    assert any(e.event_type == GameEventType.TOKEN_REMOVED for e in run.events)


_DRAG_LINE = [
    (-3, 0, 3),
    (-2, 0, 2),
    (-1, 0, 1),
    (0, 0, 0),
    (1, 0, -1),
    (2, 0, -2),
    (3, 0, -3),
    (4, 0, -4),
    (5, 0, -5),
]
_WIDGET_PERCH = (0, 1, -1)  # Off-line hex so Widget never blocks the action.


@pytest.mark.effect_flow
def test_drag_off_drags_pyro_and_enemy_in_same_direction() -> None:
    """Pyro at (0,0,0), enemy adjacent at (1,0,-1). Pyro picks (3,0,-3) (distance 3,
    direction toward the enemy). Both shift +3 along that axis: Pyro to (3,0,-3),
    enemy to (4,0,-4). Pyro's path passes through the enemy's starting hex, which
    is fine because the enemy is moving out of it."""
    pyro_start = Hex(q=0, r=0, s=0)
    enemy_start = Hex(q=1, r=0, s=-1)
    pyro_dest = Hex(q=3, r=0, s=-3)
    enemy_dest = Hex(q=4, r=0, s=-4)
    state = (
        EffectScenarioBuilder()
        .with_hexes([*_DRAG_LINE, _WIDGET_PERCH])
        .red_hero("hero_widget", at=_WIDGET_PERCH, current_card=hero_card("Widget", "drag_off"))
        .blue_minion("blue_minion", at=enemy_start)
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    assert _option_set(run) == {"blue_minion"}

    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)
    options = _option_set(run)
    # Distance must be 2 or 3 — distance 1 (enemy_start) excluded.
    assert pyro_dest in options
    assert Hex(q=2, r=0, s=-2) in options
    assert enemy_start not in options

    run.choose(pyro_dest).finish()

    assert state.entity_locations["pyro_1"] == pyro_dest
    assert state.entity_locations["blue_minion"] == enemy_dest
    assert any(e.event_type == GameEventType.TOKEN_MOVED for e in run.events)
    assert any(e.event_type == GameEventType.UNIT_MOVED for e in run.events)


def _hex_disk(radius: int) -> list[tuple[int, int, int]]:
    return [
        (q, r, -q - r)
        for q in range(-radius, radius + 1)
        for r in range(-radius, radius + 1)
        if abs(q + r) <= radius
    ]


# Drag Off geometry, direction +q, distance 3. The enemy starts adjacent to
# Pyro but OFF the drag axis, so the two movers travel parallel lines and each
# one's crossed hexes are disjoint — a mine can sit in front of either alone.
_DRAG_PYRO_START = Hex(q=0, r=0, s=0)
_DRAG_ENEMY_START = Hex(q=0, r=1, s=-1)
_DRAG_PYRO_DEST = Hex(q=3, r=0, s=-3)
_DRAG_ENEMY_DEST = Hex(q=3, r=1, s=-4)
_DRAG_PYRO_PATH = Hex(q=2, r=0, s=-2)  # crossed by Pyro only
_DRAG_ENEMY_PATH = Hex(q=2, r=1, s=-3)  # crossed by the enemy only
_DRAG_WIDGET_PERCH = (0, -1, 1)  # off both lines


def _drag_off_mine_state(mine_owner: str, mine_hexes=(_DRAG_PYRO_PATH,)):
    """Pyro at (0,0,0) → (3,0,-3); adjacent enemy at (0,1,-1) → (3,1,-4).

    A passable mine is placed on each hex in `mine_hexes`, owned by
    `mine_owner`. Neither mover is a hero (a token and a minion), so mines
    never detonate here whoever owns them — ownership only exercises the
    traversal rule.
    """
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero(
            "hero_widget", at=_DRAG_WIDGET_PERCH, current_card=hero_card("Widget", "drag_off")
        )
        .blue_minion("blue_minion", at=_DRAG_ENEMY_START)
        .blue_hero("blue_mine_owner", at=Hex(q=-3, r=0, s=3))
        .red_hero("red_mine_owner", at=Hex(q=-4, r=0, s=4))
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", _DRAG_PYRO_START)
    for idx, mine_hex in enumerate(mine_hexes, start=1):
        _place_passable_mine(state, mine_hex, owner_id=mine_owner, mine_id=f"mine_{idx}")
    return state


@pytest.mark.effect_flow
def test_drag_off_off_axis_pair_moves_with_no_mine() -> None:
    """Control for the mine cases: the same off-axis drag works on a clear board.

    Pins the geometry itself, so a failure in the mine tests below can only be
    about the mine.
    """
    state = _drag_off_mine_state("blue_mine_owner", mine_hexes=())

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)

    assert _DRAG_PYRO_DEST in _option_set(run)
    run.choose(_DRAG_PYRO_DEST).finish()

    assert state.entity_locations["pyro_1"] == _DRAG_PYRO_DEST
    assert state.entity_locations["blue_minion"] == _DRAG_ENEMY_DEST


@pytest.mark.effect_flow
@pytest.mark.parametrize(
    "mine_hexes",
    [(_DRAG_PYRO_PATH,), (_DRAG_ENEMY_PATH,), (_DRAG_PYRO_PATH, _DRAG_ENEMY_PATH)],
    ids=["pyro-path", "enemy-path", "both-paths"],
)
@pytest.mark.parametrize(
    "mine_owner", ["blue_mine_owner", "red_mine_owner"], ids=["enemy", "friendly"]
)
def test_drag_off_crosses_passable_mines(mine_hexes, mine_owner: str) -> None:
    """The drag may cross passable mines in front of either mover, either owner."""
    state = _drag_off_mine_state(mine_owner, mine_hexes)
    mine_ids = [f"mine_{i}" for i in range(1, len(mine_hexes) + 1)]

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)

    assert _DRAG_PYRO_DEST in _option_set(run)
    run.choose(_DRAG_PYRO_DEST).finish()

    assert state.entity_locations["pyro_1"] == _DRAG_PYRO_DEST
    assert state.entity_locations["blue_minion"] == _DRAG_ENEMY_DEST
    # A token and a minion crossed them — neither is a hero, so all survive.
    for mine_id, mine_hex in zip(mine_ids, mine_hexes, strict=True):
        assert state.entity_locations[mine_id] == mine_hex
    assert not [e for e in run.events if e.event_type == GameEventType.MINE_TRIGGERED]


@pytest.mark.effect_flow
@pytest.mark.parametrize("mine_owner", ["blue_mine_owner", "red_mine_owner"])
def test_drag_off_cannot_land_pyro_on_passable_mine(mine_owner: str) -> None:
    """The mine hex is traversable but never offered as a landing hex."""
    state = _drag_off_mine_state(mine_owner, mine_hexes=(_DRAG_PYRO_DEST,))

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)

    assert _DRAG_PYRO_DEST not in _option_set(run)


@pytest.mark.effect_flow
@pytest.mark.parametrize("mine_owner", ["blue_mine_owner", "red_mine_owner"])
def test_drag_off_rejects_direction_when_enemy_would_land_on_mine(mine_owner: str) -> None:
    """A mine on the ENEMY's landing hex kills the direction, though Pyro's is clear."""
    state = _drag_off_mine_state(mine_owner, mine_hexes=(_DRAG_ENEMY_DEST,))

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)

    assert _DRAG_PYRO_DEST not in _option_set(run)


@pytest.mark.effect_flow
def test_drag_off_works_in_opposite_direction_through_pyro() -> None:
    """Direction away from the enemy: enemy ends up passing through Pyro's
    starting hex. Enemy at (1,0,-1), Pyro at (0,0,0), Pyro picks (-2,0,2).
    Enemy lands at (-1,0,1) — its path crosses Pyro's old hex (0,0,0), which
    is treated as empty because Pyro is leaving."""
    pyro_start = Hex(q=0, r=0, s=0)
    enemy_start = Hex(q=1, r=0, s=-1)
    pyro_dest = Hex(q=-2, r=0, s=2)
    enemy_dest = Hex(q=-1, r=0, s=1)
    state = (
        EffectScenarioBuilder()
        .with_hexes([*_DRAG_LINE, _WIDGET_PERCH])
        .red_hero("hero_widget", at=_WIDGET_PERCH, current_card=hero_card("Widget", "drag_off"))
        .blue_minion("blue_minion", at=enemy_start)
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)
    assert pyro_dest in _option_set(run)

    run.choose(pyro_dest).finish()

    assert state.entity_locations["pyro_1"] == pyro_dest
    assert state.entity_locations["blue_minion"] == enemy_dest


@pytest.mark.effect_flow
def test_drag_off_rejects_destination_when_enemy_path_blocked() -> None:
    """Blocker on enemy's intermediate path. drag_off respects path obstacles,
    so destinations forcing the enemy through the blocker must be filtered.
    Opposite-direction destinations remain available, proving the rejection
    is path-specific rather than total."""
    pyro_start = Hex(q=0, r=0, s=0)
    enemy_start = Hex(q=1, r=0, s=-1)
    # Blocker at (3,0,-3) is the *intermediate* hex on the enemy's slide if
    # Pyro picks (3,0,-3) → enemy lands at (4,0,-4) crossing (2,0,-2)+(3,0,-3).
    # It also makes Pyro's landing (3,0,-3) itself invalid.
    blocker_hex = Hex(q=3, r=0, s=-3)
    state = (
        EffectScenarioBuilder()
        .with_hexes([*_DRAG_LINE, _WIDGET_PERCH])
        .red_hero("hero_widget", at=_WIDGET_PERCH, current_card=hero_card("Widget", "drag_off"))
        .blue_minion("blue_minion", at=enemy_start)
        .blue_minion("blocker", at=blocker_hex)
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)

    options = _option_set(run)
    # Pyro→(2,0,-2) would land enemy on the blocker. Reject.
    assert Hex(q=2, r=0, s=-2) not in options
    # Pyro→(3,0,-3) is the blocker hex itself. Reject.
    assert blocker_hex not in options
    # Opposite direction is unaffected by the blocker — must remain selectable.
    assert Hex(q=-2, r=0, s=2) in options
    assert Hex(q=-3, r=0, s=3) in options


@pytest.mark.effect_flow
def test_carry_away_ignores_blocked_path_for_partner() -> None:
    """Same blocker-on-path setup as drag_off. carry_away ignores path
    obstacles, so the destination that drag_off would reject must now be
    selectable, *and* selecting it must successfully drag both units past
    the blocker."""
    pyro_start = Hex(q=0, r=0, s=0)
    enemy_start = Hex(q=1, r=0, s=-1)
    # Blocker at (2,0,-2) sits on Pyro's path when Pyro picks (3,0,-3) and
    # on the enemy's path when the enemy slides to (4,0,-4).
    blocker_hex = Hex(q=2, r=0, s=-2)
    pyro_dest = Hex(q=3, r=0, s=-3)
    enemy_dest = Hex(q=4, r=0, s=-4)
    state = (
        EffectScenarioBuilder()
        .with_hexes([*_DRAG_LINE, _WIDGET_PERCH])
        .red_hero("hero_widget", at=_WIDGET_PERCH, current_card=hero_card("Widget", "carry_away"))
        .blue_minion("blue_minion", at=enemy_start)
        .blue_minion("blocker", at=blocker_hex)
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)

    options = _option_set(run)
    assert pyro_dest in options
    # Even ignoring path obstacles, landings must still be free — Pyro
    # cannot land on the blocker.
    assert blocker_hex not in options

    run.choose(pyro_dest).finish()
    assert state.entity_locations["pyro_1"] == pyro_dest
    assert state.entity_locations["blue_minion"] == enemy_dest


@pytest.mark.effect_flow
def test_carry_away_rejects_destination_when_enemy_landing_occupied() -> None:
    """Even with ignore_obstacles, a unit on the enemy's *landing* hex still
    blocks the destination."""
    pyro_start = Hex(q=0, r=0, s=0)
    enemy_start = Hex(q=1, r=0, s=-1)
    enemy_landing = Hex(q=4, r=0, s=-4)
    state = (
        EffectScenarioBuilder()
        .with_hexes([*_DRAG_LINE, _WIDGET_PERCH])
        .red_hero("hero_widget", at=_WIDGET_PERCH, current_card=hero_card("Widget", "carry_away"))
        .blue_minion("blue_minion", at=enemy_start)
        .blue_minion("blocker", at=enemy_landing)  # Where enemy *would* land.
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", pyro_start)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)

    # Pyro→(3,0,-3) would push enemy to (4,0,-4) (occupied by blocker).
    options = _option_set(run)
    assert Hex(q=3, r=0, s=-3) not in options
    # Sanity: opposite direction is still selectable.
    assert Hex(q=-2, r=0, s=2) in options


@pytest.mark.effect_flow
def test_fiery_breath_forces_straight_line_enemy_hero_to_discard() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (2, 1, -3)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "fiery_breath"),
        )
        .blue_hero("blue_target", at=(2, 0, -2))
        .blue_hero("blue_offline", at=(2, 1, -3))
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", Hex(q=1, r=0, s=-1))
    target = state.get_hero("blue_target")
    assert target is not None
    target.hand.append(hero_card("Widget", "all_aboard"))

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    assert state.execution_context["pyro_breath_id"] == "pyro_1"
    assert _option_set(run) == {"blue_target"}

    run.choose("blue_target").expect_input(InputRequestType.SELECT_CARD)
    run.choose("all_aboard").finish()

    assert len(target.hand) == 0
    assert len(target.discard_pile) == 1
    assert target.discard_pile[0].id == "all_aboard"


@pytest.mark.effect_contract
def test_widget_dragon_knight_is_registered() -> None:
    assert CardEffectRegistry.get("dragon_knight") is not None


@pytest.mark.effect_flow
def test_dragon_knight_offers_perform_after_movement_secondary() -> None:
    """Choosing the MOVEMENT secondary action triggers AFTER_MOVEMENT,
    which offers Dragon Knight when there's a faceup skill card to perform."""
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (3, 0, -3)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "take_off"),
        )
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    _activate_dragon_knight(state)

    widget = state.get_hero("hero_widget")
    assert widget is not None
    played_skill = hero_card("Widget", "fiery_breath")
    played_skill.state = CardState.RESOLVED
    widget.played_cards.append(played_skill)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("MOVEMENT").expect_input(InputRequestType.SELECT_HEX)
    run.choose(Hex(q=1, r=0, s=-1)).expect_input(InputRequestType.CONFIRM_PASSIVE)

    assert run.latest_request is not None
    assert "Dragon Knight" in run.latest_request.prompt

    run.choose("NO").finish()

    assert state.entity_locations["hero_widget"] == Hex(q=1, r=0, s=-1)


@pytest.mark.effect_flow
def test_dragon_knight_skipped_when_no_faceup_skill_cards() -> None:
    """should_offer_passive must short-circuit when there are no faceup
    skill cards — the YES/NO prompt should never appear."""
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "take_off"),
        )
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    _activate_dragon_knight(state)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("MOVEMENT").expect_input(InputRequestType.SELECT_HEX)
    run.choose(Hex(q=1, r=0, s=-1)).finish()

    assert state.entity_locations["hero_widget"] == Hex(q=1, r=0, s=-1)


@pytest.mark.effect_flow
def test_dragon_knight_performs_selected_skill_card_primary_action() -> None:
    """Accepting the offer should let the player pick a faceup skill card,
    and then run that card's primary action (here, fiery_breath forces a
    discard against an enemy hero in a straight line from Pyro)."""
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (3, 0, -3)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "take_off"),
        )
        .blue_hero("blue_target", at=(3, 0, -3))
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", Hex(q=2, r=0, s=-2))
    _activate_dragon_knight(state)

    widget = state.get_hero("hero_widget")
    assert widget is not None
    played_skill = hero_card("Widget", "fiery_breath")
    played_skill.state = CardState.RESOLVED
    widget.played_cards.append(played_skill)

    target = state.get_hero("blue_target")
    assert target is not None
    target.hand.append(hero_card("Widget", "all_aboard"))

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("MOVEMENT").expect_input(InputRequestType.SELECT_HEX)
    run.choose(Hex(q=1, r=0, s=-1)).expect_input(InputRequestType.CONFIRM_PASSIVE)
    run.choose("YES").expect_input(InputRequestType.SELECT_CARD)
    assert _option_set(run) == {"fiery_breath"}

    run.choose("fiery_breath").expect_input(InputRequestType.SELECT_UNIT)
    assert _option_set(run) == {"blue_target"}

    run.choose("blue_target").expect_input(InputRequestType.SELECT_CARD)
    run.choose("all_aboard").finish()

    assert len(target.hand) == 0
    assert target.discard_pile[0].id == "all_aboard"


# =============================================================================
# Arien's Spell Break vs. Widget's skill re-performs
#
# Spell Break: "This turn: Enemy heroes in radius cannot perform skill actions,
# except on gold cards." Widget's gold (Fight As One) and ultimate (Dragon
# Knight) both perform the primary action *on a skill card*. The action being
# performed is a Skill action on that skill card — not "on the gold card" — so
# Spell Break prevents it.
# =============================================================================


def _apply_spell_break(state, arien_id: str = "hero_arien") -> None:
    """Resolve Arien's Spell Break so its prevention effect is live."""
    from goa2.scripts.arien_effects import SpellBreakEffect

    arien = state.get_hero(arien_id)
    assert arien is not None
    # The card sits resolved in Arien's played area: effects are dormant until
    # their card resolves, so an unattached card would create a dormant effect.
    card = hero_card("Arien", "spell_break")
    card.state = CardState.RESOLVED
    arien.played_cards.append(card)
    previous_actor = state.current_actor_id
    state.current_actor_id = arien_id
    for step in SpellBreakEffect().get_steps(state, arien, card):
        step.resolve(state, {})
    state.current_actor_id = previous_actor


@pytest.mark.effect_flow
def test_fight_as_one_cannot_perform_skill_inside_spell_break() -> None:
    """The gold card's attack still happens, but the skill re-perform is a Skill
    action on a non-gold card, so Spell Break blocks it."""
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (0, 1, -1), (1, 0, -1), (1, 1, -2), (2, -1, -1)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "fight_as_one"),
        )
        .blue_hero("blue_initial_target", at=(1, 0, -1))
        .blue_hero("blue_replay_target", at=(1, 1, -2))
        .blue_hero("hero_arien", at=(2, -1, -1))
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    state.place_entity("pyro_1", Hex(q=0, r=1, s=-1))

    widget = state.get_hero("hero_widget")
    assert widget is not None
    played_skill = hero_card("Widget", "fiery_breath")
    played_skill.state = CardState.RESOLVED
    widget.played_cards.append(played_skill)

    replay_target = state.get_hero("blue_replay_target")
    assert replay_target is not None
    replay_target.hand.append(hero_card("Widget", "all_aboard"))

    _apply_spell_break(state)

    run = run_card(state, "hero_widget")

    # The gold card's own ATTACK is unaffected.
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_initial_target").expect_input(InputRequestType.SELECT_CARD_OR_PASS)
    run.choose("PASS").finish()

    # ...but the skill re-perform never happens: no card is offered, and the
    # would-be victim of Fiery Breath keeps their hand.
    assert "fight_as_one_skill_card" not in state.execution_context
    assert len(replay_target.hand) == 1
    assert replay_target.discard_pile == []


@pytest.mark.effect_flow
def test_dragon_knight_cannot_perform_skill_inside_spell_break() -> None:
    """Dragon Knight performs the primary action on a faceup *skill* card —
    also a Skill action on a non-gold card, so Spell Break blocks it."""
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (3, 0, -3)])
        .red_hero(
            "hero_widget",
            at=(0, 0, 0),
            current_card=hero_card("Widget", "take_off"),
        )
        .blue_hero("hero_arien", at=(2, 0, -2))
        .with_actor("hero_widget")
        .build()
    )
    _add_pyro_pool(state)
    _activate_dragon_knight(state)

    widget = state.get_hero("hero_widget")
    assert widget is not None
    played_skill = hero_card("Widget", "fiery_breath")
    played_skill.state = CardState.RESOLVED
    widget.played_cards.append(played_skill)

    _apply_spell_break(state)

    run = run_card(state, "hero_widget")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("MOVEMENT").expect_input(InputRequestType.SELECT_HEX)

    # Movement still works; Dragon Knight must not be offered afterwards.
    run.choose(Hex(q=1, r=0, s=-1)).finish()

    assert state.entity_locations["hero_widget"] == Hex(q=1, r=0, s=-1)
    assert "dragon_knight_skill_card" not in state.execution_context
