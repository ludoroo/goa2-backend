"""Behavioral contract for the information-safe public snapshot projector."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from automata.models import Viewer, canonical_json_bytes
from automata.observation import project_snapshot
from goa2.domain.board import Board, Zone
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models import (
    ActionType,
    ActiveEffect,
    AffectsFilter,
    Card,
    CardColor,
    CardState,
    CardTier,
    DurationType,
    EffectScope,
    EffectType,
    GamePhase,
    Hero,
    HeroPiece,
    MarkerType,
    Minion,
    MinionType,
    Shape,
    SpellCard,
    Team,
    TeamColor,
    Token,
    TokenType,
    Turret,
)
from goa2.domain.state import GameState
from goa2.domain.tile import Tile
from goa2.domain.types import BoardEntityID, HeroID


def _card(card_id: str, *, facedown: bool = False) -> Card:
    return Card(
        id=card_id,
        name=card_id.replace("_", " ").title(),
        tier=CardTier.UNTIERED,
        color=CardColor.SILVER,
        initiative=card_id.count("a") + 1,
        primary_action=ActionType.MOVEMENT,
        primary_action_value=2,
        secondary_actions={},
        effect_id=f"effect_{card_id}",
        effect_text=f"Public text for {card_id}",
        is_facedown=facedown,
    )


def _spell(spell_id: str) -> SpellCard:
    spell = SpellCard.define(
        id=spell_id,
        name=spell_id.replace("_", " ").title(),
        spell_rank=1,
        tier=CardTier.I,
        color=CardColor.BLUE,
        primary_action=ActionType.SKILL,
        effect_text=f"Private text for {spell_id}",
    )
    spell.state = CardState.SPELLBOOK
    spell.is_facedown = True
    return spell


def _hero(hero_id: str, team: TeamColor) -> Hero:
    return Hero(
        id=BoardEntityID(hero_id),
        name=hero_id.replace("hero_", "").title(),
        team=team,
        deck=[],
    )


def _get_hero(state: GameState, hero_id: str) -> Hero:
    hero = state.get_hero(HeroID(hero_id))
    assert hero is not None
    return hero


def _snapshot_state() -> GameState:
    board = Board(map_id="two_lane_contract")
    lanes: dict[str, list[str]] = {}
    for lane_number, r in ((1, 0), (2, 3)):
        zone_ids = [f"l{lane_number}_{name}" for name in ("red", "mid", "blue")]
        lanes[f"lane_{lane_number}"] = zone_ids
        for q, zone_id in enumerate(zone_ids):
            hex_ = Hex(q=q, r=r, s=-q - r)
            zone_hexes: set[Hex] = set()
            zone_hexes.add(hex_)
            board.zones[zone_id] = Zone(id=zone_id, hexes=zone_hexes)
            board.tiles[hex_] = Tile(hex=hex_, zone_id=zone_id, is_terrain=q == 2)
    board.lanes = lanes

    owner = _hero("hero_razzle", TeamColor.RED)
    owner.piece_supply = 4
    owner.hand = [_card("owner_hand_alpha")]
    owner.deck = [_card("owner_upgrade_alpha", facedown=True)]

    ally = _hero("hero_ally", TeamColor.RED)
    ally.hand = [_card("ally_hand_alpha")]
    ally.deck = [_card("ally_upgrade_alpha", facedown=True)]

    enemy = _hero("hero_enemy", TeamColor.BLUE)
    enemy.hand = [_card("enemy_hand_alpha")]
    enemy.current_turn_card = _card("enemy_commit_alpha", facedown=True)
    enemy.spells = [_spell("enemy_spell_alpha")]

    public_enemy = _hero("hero_public_enemy", TeamColor.BLUE)
    public_enemy.played_cards = [_card("public_faceup_card")]

    red_minion = Minion(
        id=BoardEntityID("red_minion_lane_2"),
        name="Red minion",
        team=TeamColor.RED,
        type=MinionType.MELEE,
        lane_id="lane_2",
    )
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[owner, ally], minions=[red_minion]),
            TeamColor.BLUE: Team(
                color=TeamColor.BLUE,
                heroes=[enemy, public_enemy],
                minions=[],
            ),
        },
        phase=GamePhase.PLANNING,
        battle_zones={"lane_1": "l1_mid", "lane_2": "l2_red"},
        wave_counters={"lane_1": 3, "lane_2": 2},
        current_actor_id=HeroID("hero_razzle"),
    )

    for index, at in enumerate((Hex(q=0, r=0, s=0), Hex(q=1, r=0, s=-1)), start=1):
        piece = HeroPiece(
            id=BoardEntityID(f"hero_razzle_piece_{index}"),
            name="Razzle",
            team=TeamColor.RED,
            owner_hero_id="hero_razzle",
        )
        state.misc_entities[piece.id] = piece
        state.entity_locations[piece.id] = at
    state.entity_locations[ally.id] = Hex(q=2, r=0, s=-2)
    state.entity_locations[enemy.id] = Hex(q=1, r=3, s=-4)
    state.entity_locations[red_minion.id] = Hex(q=0, r=3, s=-3)

    mine = Token(
        id=BoardEntityID("hidden_mine"),
        name="Blast mine",
        token_type=TokenType.MINE_BLAST,
        owner_id=HeroID("hero_enemy"),
        is_facedown=True,
    )
    state.token_pool = {TokenType.MINE_BLAST: [mine]}
    state.misc_entities[mine.id] = mine
    state.entity_locations[mine.id] = Hex(q=2, r=3, s=-5)

    turret = Turret(
        id=BoardEntityID("public_turret"),
        name="Turret",
        owner_id="hero_ally",
    )
    state.misc_entities[turret.id] = turret
    state.entity_locations[turret.id] = Hex(q=2, r=0, s=-2)
    state.active_effects = [
        ActiveEffect(
            id="public_aura",
            source_id="hero_ally",
            source_card_id="public_faceup_card",
            effect_type=EffectType.AREA_STAT_MODIFIER,
            scope=EffectScope(
                shape=Shape.RADIUS,
                range=1,
                origin_id="hero_ally",
                affects=AffectsFilter.FRIENDLY_UNITS,
            ),
            duration=DurationType.THIS_ROUND,
            created_at_turn=1,
            created_at_round=1,
            is_active=True,
        )
    ]
    state.place_marker(MarkerType.VENOM, "hero_enemy", -1, "hero_razzle")
    return state


def _viewer(
    *,
    private_hero_id: str | None,
    perspective_team: str | None,
) -> Viewer:
    return Viewer(
        schema_version=2,
        private_hero_id=private_hero_id,
        perspective_team=perspective_team,
    )


def _bytes(state: GameState, viewer: Viewer) -> bytes:
    return canonical_json_bytes(project_snapshot(state, viewer))


def _hero_view(snapshot, hero_id: str) -> dict[str, object]:
    teams = snapshot.public_state["teams"]
    assert isinstance(teams, dict)
    for team in teams.values():
        assert isinstance(team, dict)
        heroes = team["heroes"]
        assert isinstance(heroes, list)
        for hero in heroes:
            assert isinstance(hero, dict)
            if hero["id"] == hero_id:
                return hero
    raise AssertionError(f"missing public hero view for {hero_id}")


def _card_ids(cards: object) -> set[str]:
    assert isinstance(cards, list)
    return {
        card_id
        for card in cards
        if isinstance(card, dict) and isinstance((card_id := card.get("id")), str)
    }


def _assert_count_only(cards: object, expected: int) -> None:
    assert cards == {"count": expected}


def test_hero_scope_identifies_only_the_viewing_heros_private_cards() -> None:
    state = _snapshot_state()
    viewer = _viewer(private_hero_id="hero_razzle", perspective_team="RED")

    snapshot = project_snapshot(state, viewer)

    owner = _hero_view(snapshot, "hero_razzle")
    ally = _hero_view(snapshot, "hero_ally")
    enemy = _hero_view(snapshot, "hero_enemy")
    public_enemy = _hero_view(snapshot, "hero_public_enemy")

    assert _card_ids(owner["hand"]) == {"owner_hand_alpha"}
    assert ally["hand"] == []
    _assert_count_only(ally["deck"], 1)
    assert enemy["hand"] == []
    _assert_count_only(enemy["spellbook"], 1)
    assert isinstance(enemy["current_turn_card"], dict)
    assert "id" not in enemy["current_turn_card"]
    assert _card_ids(public_enemy["played_cards"]) == {"public_faceup_card"}


def test_team_and_public_scopes_never_identify_hero_private_cards() -> None:
    state = _snapshot_state()
    team_snapshot = project_snapshot(
        state,
        _viewer(private_hero_id=None, perspective_team="RED"),
    )
    public_snapshot = project_snapshot(
        state,
        _viewer(private_hero_id=None, perspective_team=None),
    )

    for snapshot in (team_snapshot, public_snapshot):
        owner = _hero_view(snapshot, "hero_razzle")
        enemy = _hero_view(snapshot, "hero_enemy")
        public_enemy = _hero_view(snapshot, "hero_public_enemy")

        assert owner["hand"] == []
        _assert_count_only(owner["deck"], 1)
        assert enemy["hand"] == []
        _assert_count_only(enemy["spellbook"], 1)
        assert isinstance(enemy["current_turn_card"], dict)
        assert "id" not in enemy["current_turn_card"]
        assert _card_ids(public_enemy["played_cards"]) == {"public_faceup_card"}


@pytest.mark.parametrize(
    ("private_hero_id", "perspective_team"),
    [(None, None), (None, "RED"), ("hero_razzle", None), ("hero_razzle", "RED")],
)
def test_viewer_accepts_independent_visibility_and_perspective(
    private_hero_id: str | None, perspective_team: str | None
) -> None:
    snapshot = project_snapshot(
        _snapshot_state(),
        _viewer(private_hero_id=private_hero_id, perspective_team=perspective_team),
    )

    assert snapshot.viewer.private_hero_id == private_hero_id
    assert snapshot.viewer.perspective_team == perspective_team


def test_private_hero_and_explicit_perspective_must_be_team_consistent() -> None:
    with pytest.raises(ValueError, match="does not belong"):
        project_snapshot(
            _snapshot_state(),
            _viewer(private_hero_id="hero_razzle", perspective_team="BLUE"),
        )


def test_project_snapshot_rejects_team_viewer_absent_from_state() -> None:
    state = _snapshot_state()
    viewer = _viewer(private_hero_id=None, perspective_team="GREEN")

    with pytest.raises(ValueError):
        project_snapshot(state, viewer)


HiddenMutation = Callable[[GameState], None]


def _replace_ally_hand(state: GameState) -> None:
    _get_hero(state, "hero_ally").hand[0] = _card("ally_hand_beta")


def _replace_enemy_commit(state: GameState) -> None:
    _get_hero(state, "hero_enemy").current_turn_card = _card("enemy_commit_beta", facedown=True)


def _replace_hidden_upgrade(state: GameState) -> None:
    _get_hero(state, "hero_ally").deck[0] = _card("ally_upgrade_beta", facedown=True)


def _replace_prepared_spell(state: GameState) -> None:
    _get_hero(state, "hero_enemy").spells[0] = _spell("enemy_spell_beta")


def _replace_facedown_token_subtype(state: GameState) -> None:
    token = state.token_pool[TokenType.MINE_BLAST].pop()
    token.token_type = TokenType.MINE_DUD
    token.name = "Dud mine"
    state.token_pool = {TokenType.MINE_DUD: [token]}


@pytest.mark.parametrize(
    "replace_hidden_identity",
    [
        _replace_ally_hand,
        _replace_enemy_commit,
        _replace_hidden_upgrade,
        _replace_prepared_spell,
        _replace_facedown_token_subtype,
    ],
    ids=["hand", "commitment", "upgrade-loadout", "prepared-spell", "token-subtype"],
)
def test_unavailable_hidden_identity_substitution_is_observationally_equivalent(
    replace_hidden_identity: HiddenMutation,
) -> None:
    viewer = _viewer(private_hero_id="hero_razzle", perspective_team="RED")
    original = _snapshot_state()
    substituted = original.model_copy(deep=True)

    replace_hidden_identity(substituted)

    assert _bytes(substituted, viewer) == _bytes(original, viewer)


def test_owner_private_identity_and_public_reveal_change_snapshot_bytes() -> None:
    viewer = _viewer(private_hero_id="hero_razzle", perspective_team="RED")
    state = _snapshot_state()
    baseline = _bytes(state, viewer)

    owner_changed = state.model_copy(deep=True)
    _get_hero(owner_changed, "hero_razzle").hand[0] = _card("owner_hand_beta")
    revealed = state.model_copy(deep=True)
    committed = _get_hero(revealed, "hero_enemy").current_turn_card
    assert committed is not None
    committed.is_facedown = False

    assert _bytes(owner_changed, viewer) != baseline
    assert _bytes(revealed, viewer) != baseline


def test_runtime_only_mutation_does_not_change_snapshot_bytes() -> None:
    state = _snapshot_state()
    viewer = _viewer(private_hero_id="hero_razzle", perspective_team="RED")
    baseline = _bytes(state, viewer)

    state.execution_stack.append({"internal_step": "do not project"})
    state.execution_context["selected_secret"] = "enemy_hand_alpha"
    state.input_stack.append(
        InputRequest(
            id="runtime-request-id",
            request_type=InputRequestType.SELECT_CARD,
            player_id="hero_enemy",
            prompt="Internal pending request",
        )
    )
    state.rng_seed = 987654
    state.next_entity_id += 100

    assert _bytes(state, viewer) == baseline


def test_snapshot_retains_broad_public_gameplay_sections() -> None:
    snapshot = project_snapshot(
        _snapshot_state(),
        _viewer(private_hero_id="hero_razzle", perspective_team="RED"),
    )

    public_state = snapshot.public_state
    assert set(public_state) >= {
        "teams",
        "board",
        "battle_zones",
        "wave_counters",
        "effects",
        "markers",
        "tokens",
        "board_entities",
        "hero_pieces",
    }
    assert set(public_state["teams"]) == {"RED", "BLUE"}
    assert public_state["board"]
    assert public_state["battle_zones"] == {"lane_1": "l1_mid", "lane_2": "l2_red"}
    assert public_state["wave_counters"] == {"lane_1": 3, "lane_2": 2}
    assert public_state["effects"]
    assert public_state["markers"]
    assert public_state["tokens"]
    assert public_state["board_entities"]
    assert public_state["hero_pieces"]


def test_snapshot_serialization_is_deterministic_and_snapshot_is_frozen() -> None:
    state = _snapshot_state()
    viewer = _viewer(private_hero_id="hero_razzle", perspective_team="RED")
    reordered = state.model_copy(deep=True)
    reordered.teams = dict(reversed(tuple(reordered.teams.items())))
    reordered.misc_entities = dict(reversed(tuple(reordered.misc_entities.items())))
    reordered.entity_locations = dict(reversed(tuple(reordered.entity_locations.items())))

    snapshot = project_snapshot(state, viewer)
    reordered_snapshot = project_snapshot(reordered, viewer)

    assert canonical_json_bytes(snapshot) == canonical_json_bytes(reordered_snapshot)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        snapshot.map_id = "another_map"
