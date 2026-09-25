"""
Emmitt card-effect tests (non-ultimate cards).

TDD paths: docs/superpowers/plans/2026-07-05-emmitt-tdd-paths.md.
Ultimate tests live in tests/engine/test_emmitt_ultimate.py.
"""

from __future__ import annotations

import pytest

import goa2.scripts.dodger_effects  # registers shield_of_decay for the defense tests
import goa2.scripts.emmitt_effects  # noqa: F401
from goa2.domain.events import GameEventType
from goa2.domain.hex import Hex
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardState,
    CardTier,
    Minion,
    MinionType,
    StatType,
    TeamColor,
    Token,
    TokenType,
)
from goa2.domain.models.effect import (
    ActiveEffect,
    AffectsFilter,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.types import BoardEntityID
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.rules import is_immune
from goa2.engine.steps import AttackSequenceStep

from ..builders import EffectScenarioBuilder, hero_card, skill_card
from ..runner import run_card

# =============================================================================
# P7 primitive: radius-scoped IMMUNITY_ENEMY_ACTIONS aura in is_immune()
# =============================================================================


def _aura_effect(origin_id: str, radius: int = 4) -> ActiveEffect:
    return ActiveEffect(
        id="aura_test",
        source_id=origin_id,
        effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
        scope=EffectScope(
            shape=Shape.RADIUS,
            range=radius,
            origin_id=origin_id,
            affects=AffectsFilter.FRIENDLY_HEROES,
        ),
        duration=DurationType.THIS_TURN,
        is_active=True,
        created_at_turn=1,
        created_at_round=1,
    )


def _aura_state():
    """Emmitt + ally (RED) vs enemy (BLUE, acting) on a line board."""
    state = (
        EffectScenarioBuilder()
        .line_board(8)
        .red_hero("hero_emmitt", at=(0, 0, 0))
        .red_hero("hero_ally", at=(1, 0, -1))
        .blue_hero("hero_enemy", at=(3, 0, -3))
        .with_actor("hero_enemy")
        .build()
    )
    state.active_effects.append(_aura_effect("hero_emmitt"))
    return state


@pytest.mark.effect_contract
class TestAuraImmunityPrimitive:
    def test_friendly_hero_in_radius_is_immune(self):
        state = _aura_state()
        ally = state.get_hero("hero_ally")
        assert is_immune(ally, state) is True

    def test_origin_hero_is_not_immune(self):
        """FRIENDLY_HEROES excludes the aura's owner (Emmitt himself)."""
        state = _aura_state()
        emmitt = state.get_hero("hero_emmitt")
        assert is_immune(emmitt, state) is False

    def test_hero_outside_radius_not_immune_until_entering(self):
        """Aura semantics: evaluated at check time from the origin's current
        position — entering the radius gains immunity, leaving loses it."""
        state = _aura_state()
        ally = state.get_hero("hero_ally")
        state.place_entity("hero_ally", Hex(q=6, r=0, s=-6))  # outside radius 4
        assert is_immune(ally, state) is False
        state.place_entity("hero_ally", Hex(q=2, r=0, s=-2))  # back inside
        assert is_immune(ally, state) is True

    def test_friendly_actor_not_blocked(self):
        """Immunity applies to ENEMY actions only."""
        state = _aura_state()
        state.current_actor_id = "hero_emmitt"
        ally = state.get_hero("hero_ally")
        assert is_immune(ally, state) is False

    def test_friendly_minion_not_covered(self):
        state = (
            EffectScenarioBuilder()
            .line_board(8)
            .red_hero("hero_emmitt", at=(0, 0, 0))
            .red_minion("minion_red", at=(1, 0, -1))
            .blue_hero("hero_enemy", at=(3, 0, -3))
            .with_actor("hero_enemy")
            .build()
        )
        state.active_effects.append(_aura_effect("hero_emmitt"))
        minion = next(m for m in state.teams[TeamColor.RED].minions if m.id == "minion_red")
        assert is_immune(minion, state) is False

    def test_inactive_aura_does_not_protect(self):
        state = _aura_state()
        state.active_effects[-1].is_active = False
        ally = state.get_hero("hero_ally")
        assert is_immune(ally, state) is False


# =============================================================================
# P4 primitive: turn-scoped discard log
# =============================================================================


@pytest.mark.effect_contract
class TestTurnDiscardLog:
    def _state_with_hand(self):
        from ..builders import skill_card

        state = (
            EffectScenarioBuilder()
            .line_board(6)
            .red_hero("hero_a", at=(0, 0, 0))
            .blue_hero("hero_b", at=(3, 0, -3))
            .with_actor("hero_a")
            .build()
        )
        hero = state.get_hero("hero_a")
        card = skill_card("logged_card")
        card.state = "HAND"
        hero.hand.append(card)
        return state, hero, card

    def test_discard_records_in_turn_log(self):
        from goa2.engine.handler import process_stack, push_steps
        from goa2.engine.steps import DiscardCardStep

        state, _hero, card = self._state_with_hand()
        push_steps(state, [DiscardCardStep(card_id=card.id, hero_id="hero_a")])
        process_stack(state)
        assert state.turn_discard_log.get("hero_a") == [card.id]

    def test_end_turn_clears_log(self):
        from goa2.engine.phases import end_turn

        state, _hero, card = self._state_with_hand()
        state.turn_discard_log = {"hero_a": [card.id]}
        state.unresolved_hero_ids = []
        end_turn(state)
        assert state.turn_discard_log == {}


# =============================================================================
# P3 primitive: turn-boundary position snapshot (state.last_turn_positions)
#
# "The space where that unit was at the start of this turn" (Back to the Future
# A) and "remained in the same space since the last turn" (Time Walk / Fast
# Forward) name the same instant — the turn boundary — because a turn is the
# whole plan-and-act cycle and nothing moves during Planning. One snapshot,
# recorded wherever the phase becomes PLANNING, serves all three cards.
# =============================================================================


@pytest.mark.effect_contract
class TestPositionSnapshotPrimitive:
    def _state(self):
        state = (
            EffectScenarioBuilder()
            .line_board(6)
            .red_hero("hero_a", at=(0, 0, 0))
            .blue_hero("hero_b", at=(3, 0, -3))
            .with_actor("hero_a")
            .build()
        )
        state.unresolved_hero_ids = []
        return state

    def test_end_turn_records_positions_before_planning(self):
        from goa2.engine.phases import end_turn

        state = self._state()
        end_turn(state)
        assert state.last_turn_positions["hero_a"] == Hex(q=0, r=0, s=0)
        assert state.last_turn_positions["hero_b"] == Hex(q=3, r=0, s=-3)

    def test_snapshot_is_a_copy_not_a_live_alias(self):
        """Later movement must not retroactively rewrite the snapshot."""
        from goa2.engine.phases import end_turn

        state = self._state()
        end_turn(state)
        state.place_entity("hero_a", Hex(q=2, r=0, s=-2))
        assert state.last_turn_positions["hero_a"] == Hex(q=0, r=0, s=0)

    def test_advance_turn_step_records_positions(self):
        """Deferred advancement (finishing steps ran) snapshots too."""
        from goa2.engine.handler import process_stack, push_steps
        from goa2.engine.steps import AdvanceTurnStep

        state = self._state()
        state.last_turn_positions = {}
        push_steps(state, [AdvanceTurnStep()])
        process_stack(state)
        assert state.last_turn_positions["hero_a"] == Hex(q=0, r=0, s=0)

    def test_round_reset_records_positions(self):
        """Turn 1 of a new round snapshots the end of the previous round."""
        from goa2.engine.handler import process_stack, push_steps
        from goa2.engine.steps import RoundResetStep

        state = self._state()
        state.last_turn_positions = {}
        push_steps(state, [RoundResetStep()])
        process_stack(state)
        assert state.last_turn_positions["hero_a"] == Hex(q=0, r=0, s=0)

    def test_unit_placed_after_snapshot_has_no_entry(self):
        """S2: spawned/respawned units have no snapshot entry, so 'remained' is
        false and Back to the Future A has no defined start-of-turn space."""
        from goa2.domain.models import Minion, MinionType
        from goa2.engine.phases import end_turn

        state = self._state()
        end_turn(state)
        minion = Minion(id="minion_new", name="New", team=TeamColor.RED, type=MinionType.MELEE)
        state.teams[TeamColor.RED].minions.append(minion)
        state.place_entity("minion_new", Hex(q=1, r=0, s=-1))
        assert "minion_new" not in state.last_turn_positions

    def test_snapshot_populated_at_game_creation(self):
        """Initial setup counts as the first snapshot."""
        from goa2.engine.setup import GameSetup

        state = GameSetup.create_game(
            "src/goa2/data/maps/forgotten_island.json", ["Arien"], ["Wasp"]
        )
        assert state.last_turn_positions
        assert state.last_turn_positions["hero_arien"] == state.get_position("hero_arien")

    def test_snapshot_round_trips_through_persistence(self):
        import json

        from goa2.domain.state import GameState

        state = self._state()
        state.last_turn_positions = {"hero_a": Hex(q=1, r=-1, s=0)}
        restored = GameState.model_validate(json.loads(state.model_dump_json()))
        assert restored.last_turn_positions["hero_a"] == Hex(q=1, r=-1, s=0)


# =============================================================================
# TIME CAPSULE (§8): "You, and friendly heroes in radius, may retrieve all
# cards discarded this turn."
# =============================================================================


def _capsule_state(ally_at=(2, 0, -2)):
    """Emmitt (with time_capsule, radius 4) + ally; both have one card
    discarded this turn and one discarded earlier."""
    from ..builders import skill_card

    state = (
        EffectScenarioBuilder()
        .line_board(10)
        .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "time_capsule"))
        .red_hero("hero_ally", at=ally_at)
        .blue_hero("hero_enemy", at=(5, 0, -5))
        .with_actor("hero_emmitt")
        .build()
    )
    for hid, prefix in (("hero_emmitt", "em"), ("hero_ally", "al")):
        hero = state.get_hero(hid)
        fresh = skill_card(f"{prefix}_fresh")
        old = skill_card(f"{prefix}_old")
        for c in (fresh, old):
            c.state = "DISCARD"
            hero.discard_pile.append(c)
        state.turn_discard_log.setdefault(hid, []).append(fresh.id)
    return state


@pytest.mark.effect_flow
class TestTimeCapsule:
    def test_both_heroes_retrieve_their_own_this_turn_discards(self):
        state = _capsule_state()
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        # Emmitt decides first (choice routed to him), then the ally
        run.expect_input("SELECT_NUMBER")
        assert run.latest_request.player_id == "hero_emmitt"
        run.choose(1)
        run.expect_input("SELECT_NUMBER")
        assert run.latest_request.player_id == "hero_ally"
        run.choose(1)
        run.finish()

        emmitt = state.get_hero("hero_emmitt")
        ally = state.get_hero("hero_ally")
        assert "em_fresh" in [c.id for c in emmitt.hand]
        assert "al_fresh" in [c.id for c in ally.hand]
        # Cards discarded on earlier turns stay put
        assert "em_old" in [c.id for c in emmitt.discard_pile]
        assert "al_old" in [c.id for c in ally.discard_pile]

    def test_decline_is_all_or_nothing(self):
        state = _capsule_state()
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(0)  # Emmitt declines
        run.expect_input("SELECT_NUMBER").choose(1)  # ally accepts
        run.finish()
        emmitt = state.get_hero("hero_emmitt")
        assert "em_fresh" in [c.id for c in emmitt.discard_pile]
        assert "al_fresh" in [c.id for c in state.get_hero("hero_ally").hand]

    def test_ally_outside_radius_not_offered(self):
        state = _capsule_state(ally_at=(6, 0, -6))  # beyond radius 4
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER")
        assert run.latest_request.player_id == "hero_emmitt"
        run.choose(1)
        run.finish()  # no second prompt
        assert "al_fresh" in [c.id for c in state.get_hero("hero_ally").discard_pile]

    def test_hero_without_this_turn_discards_not_prompted(self):
        state = _capsule_state()
        state.turn_discard_log.pop("hero_ally")
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(1)
        run.finish()  # ally never prompted
        assert "al_fresh" in [c.id for c in state.get_hero("hero_ally").discard_pile]

    def test_enemy_never_offered(self):
        state = _capsule_state()
        enemy = state.get_hero("hero_enemy")
        from ..builders import skill_card

        card = skill_card("en_fresh")
        card.state = "DISCARD"
        enemy.discard_pile.append(card)
        state.turn_discard_log["hero_enemy"] = [card.id]
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(1)
        run.expect_input("SELECT_NUMBER").choose(1)
        run.finish()
        assert "en_fresh" in [c.id for c in enemy.discard_pile]


# =============================================================================
# FUTURE PROOF (§9): Choose one — Time Capsule retrieval / aura immunity
# =============================================================================


@pytest.mark.effect_flow
class TestFutureProof:
    def _state(self):
        state = (
            EffectScenarioBuilder()
            .line_board(10)
            .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "future_proof"))
            .red_hero("hero_ally", at=(2, 0, -2))
            .blue_hero("hero_enemy", at=(5, 0, -5))
            .with_actor("hero_emmitt")
            .build()
        )
        return state

    def test_immunity_branch_creates_aura(self):
        state = self._state()
        enemy_card = skill_card("enemy_card")
        enemy_card.state = CardState.UNRESOLVED
        state.get_hero("hero_enemy").current_turn_card = enemy_card
        run = run_card(state, "hero_emmitt", finalize_turn=True)
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER")  # choose-one branch
        run.choose(2)  # immunity
        # Keep the turn open on the enemy's valid unresolved card so the
        # THIS_TURN aura can be asserted before it expires.
        run.expect_input("CHOOSE_ACTION")

        auras = [
            e
            for e in state.active_effects
            if e.effect_type == EffectType.IMMUNITY_ENEMY_ACTIONS and e.scope.shape == Shape.RADIUS
        ]
        assert len(auras) == 1
        assert auras[0].scope.affects == AffectsFilter.FRIENDLY_HEROES
        assert auras[0].scope.range == 4
        assert auras[0].is_active

        state.current_actor_id = "hero_enemy"
        assert is_immune(state.get_hero("hero_ally"), state) is True
        assert is_immune(state.get_hero("hero_emmitt"), state) is False

    def test_immunity_expires_at_end_of_turn(self):
        from goa2.engine.phases import end_turn

        state = self._state()
        run = run_card(state, "hero_emmitt", finalize_turn=True)
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(2)
        run.finish()
        state.unresolved_hero_ids = []
        end_turn(state)
        state.current_actor_id = "hero_enemy"
        assert is_immune(state.get_hero("hero_ally"), state) is False

    def test_retrieval_branch_behaves_like_time_capsule(self):
        from ..builders import skill_card

        state = self._state()
        emmitt = state.get_hero("hero_emmitt")
        card = skill_card("em_fresh")
        card.state = "DISCARD"
        emmitt.discard_pile.append(card)
        state.turn_discard_log["hero_emmitt"] = [card.id]

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(1)  # retrieval branch
        run.expect_input("SELECT_NUMBER").choose(1)  # Emmitt accepts
        run.finish()
        assert "em_fresh" in [c.id for c in emmitt.hand]


# =============================================================================
# REVERSE TIME (§12): attack; next turn lower initiative acts first
# =============================================================================


def _order_state(init_a: int, init_b: int, *, items_a: int = 0):
    """Two opposing heroes with committed cards, ready for resolve_next_action."""
    from goa2.domain.models import StatType

    from ..builders import skill_card

    state = (
        EffectScenarioBuilder()
        .line_board(6)
        .red_hero("hero_a", at=(0, 0, 0), current_card=skill_card("card_a", initiative=init_a))
        .blue_hero("hero_b", at=(3, 0, -3), current_card=skill_card("card_b", initiative=init_b))
        .build()
    )
    if items_a:
        state.get_hero("hero_a").items[StatType.INITIATIVE] = items_a
    state.unresolved_hero_ids = ["hero_a", "hero_b"]
    state.current_actor_id = None
    return state


def _reversal(created_turn: int, created_round: int = 1) -> ActiveEffect:
    return ActiveEffect(
        id="rev_test",
        source_id="hero_emmitt_src",
        effect_type=EffectType.REVERSED_INITIATIVE,
        scope=EffectScope(shape=Shape.GLOBAL),
        duration=DurationType.NEXT_TURN,
        is_active=True,
        created_at_turn=created_turn,
        created_at_round=created_round,
    )


@pytest.mark.effect_contract
class TestReversedInitiativePrimitive:
    def test_reversed_order_next_turn(self):
        from goa2.engine.phases import resolve_next_action

        state = _order_state(3, 9)
        state.turn = 2
        state.active_effects.append(_reversal(created_turn=1))
        resolve_next_action(state)
        assert str(state.current_actor_id) == "hero_a"  # lowest first

    def test_normal_order_without_effect(self):
        from goa2.engine.phases import resolve_next_action

        state = _order_state(3, 9)
        state.turn = 2
        resolve_next_action(state)
        assert str(state.current_actor_id) == "hero_b"

    def test_reversal_dormant_on_creation_turn(self):
        from goa2.engine.phases import resolve_next_action

        state = _order_state(3, 9)
        state.turn = 1
        state.active_effects.append(_reversal(created_turn=1))
        resolve_next_action(state)
        assert str(state.current_actor_id) == "hero_b"

    def test_reversal_over_after_next_turn(self):
        from goa2.engine.phases import resolve_next_action

        state = _order_state(3, 9)
        state.turn = 3
        state.active_effects.append(_reversal(created_turn=1))
        resolve_next_action(state)
        assert str(state.current_actor_id) == "hero_b"

    def test_reversal_fizzles_across_round_boundary(self):
        from goa2.engine.phases import resolve_next_action

        state = _order_state(3, 9)
        state.round = 2
        state.turn = 1
        state.active_effects.append(_reversal(created_turn=4, created_round=1))
        resolve_next_action(state)
        assert str(state.current_actor_id) == "hero_b"

    def test_reversed_order_uses_computed_initiative(self):
        from goa2.engine.phases import resolve_next_action

        # hero_a: card 5 + item 5 = 10; hero_b: 6 → reversed picks b (6 < 10)
        state = _order_state(5, 6, items_a=5)
        state.turn = 2
        state.active_effects.append(_reversal(created_turn=1))
        resolve_next_action(state)
        assert str(state.current_actor_id) == "hero_b"

    def test_ties_still_go_to_tie_breaker(self):
        from goa2.engine.phases import resolve_next_action
        from goa2.engine.steps import ResolveTieBreakerStep

        state = _order_state(4, 4)
        state.turn = 2
        state.active_effects.append(_reversal(created_turn=1))
        resolve_next_action(state)
        assert isinstance(state.execution_stack[-1], ResolveTieBreakerStep)


@pytest.mark.effect_flow
class TestReverseTime:
    def _state(self, with_target: bool = True):
        builder = (
            EffectScenarioBuilder()
            .line_board(8)
            .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "reverse_time"))
            .blue_hero("hero_far", at=(5, 0, -5))
            .with_actor("hero_emmitt")
        )
        if with_target:
            builder = builder.blue_minion("minion_target", at=(1, 0, -1))
        state = builder.build()
        next_turn_card = skill_card("far_next_turn")
        next_turn_card.state = CardState.HAND
        state.get_hero("hero_far").hand.append(next_turn_card)
        return state

    def test_attack_then_creates_next_turn_reversal(self):
        state = self._state()
        run = run_card(state, "hero_emmitt", finalize_turn=True)
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("minion_target")
        run.finish()

        reversals = [
            e for e in state.active_effects if e.effect_type == EffectType.REVERSED_INITIATIVE
        ]
        assert len(reversals) == 1
        assert reversals[0].duration == DurationType.NEXT_TURN
        assert reversals[0].source_id == "hero_emmitt"

    def test_no_effect_when_attack_aborts(self):
        state = self._state(with_target=False)
        run = run_card(state, "hero_emmitt", finalize_turn=True)
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.finish()  # no adjacent unit → mandatory targeting fails → abort
        assert not [
            e for e in state.active_effects if e.effect_type == EffectType.REVERSED_INITIATIVE
        ]

    def test_defeat_of_emmitt_removes_reversal(self):
        from goa2.engine.handler import process_stack, push_steps
        from goa2.engine.steps import DefeatUnitStep

        state = self._state()
        run = run_card(state, "hero_emmitt", finalize_turn=True)
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("minion_target")
        run.finish()
        assert any(e.effect_type == EffectType.REVERSED_INITIATIVE for e in state.active_effects)

        state.current_actor_id = "hero_far"
        push_steps(state, [DefeatUnitStep(victim_id="hero_emmitt", killer_id="hero_far")])
        process_stack(state)
        assert not [
            e for e in state.active_effects if e.effect_type == EffectType.REVERSED_INITIATIVE
        ]


# =============================================================================
# P1 primitive: HasResolvedCardFilter — "has already resolved a card this turn"
# =============================================================================


def _resolved_card(card_id: str, initiative: int = 5):
    from goa2.domain.models.enums import CardState

    from ..builders import skill_card

    card = skill_card(card_id, initiative=initiative)
    card.state = CardState.RESOLVED
    return card


def _unresolved_card(card_id: str, initiative: int = 5):
    from goa2.domain.models.enums import CardState

    from ..builders import skill_card

    card = skill_card(card_id, initiative=initiative)
    card.state = CardState.UNRESOLVED
    return card


def _hand_card(card_id: str):
    from goa2.domain.models.enums import CardState

    from ..builders import skill_card

    card = skill_card(card_id)
    card.state = CardState.HAND
    return card


@pytest.mark.effect_contract
class TestHasResolvedCardFilterPrimitive:
    def _state(self):
        return (
            EffectScenarioBuilder()
            .line_board(8)
            .red_hero("hero_emmitt", at=(0, 0, 0))
            .blue_hero("hero_enemy", at=(3, 0, -3))
            .with_actor("hero_emmitt")
            .build()
        )

    def _passes(self, state, candidate="hero_enemy"):
        from goa2.engine.filters import HasResolvedCardFilter

        return HasResolvedCardFilter().apply(candidate, state, {})

    def test_enemy_finalized_this_turn_passes(self):
        """Turn 1 (actor resolved_turn_count=0): enemy's slot 0 is filled →
        they resolved a card this turn."""
        state = self._state()
        enemy = state.get_hero("hero_enemy")
        enemy.played_cards = [_resolved_card("prev_resolved")]
        enemy.resolved_turn_count = 1
        assert self._passes(state) is True

    def test_enemy_with_only_unresolved_card_fails(self):
        state = self._state()
        state.get_hero("hero_enemy").current_turn_card = _unresolved_card("pending")
        assert self._passes(state) is False

    def test_resolved_on_earlier_turn_only_fails(self):
        """Turn 2 (actor resolved_turn_count=1): enemy resolved on turn 1 but
        has an unresolved card THIS turn → 'this turn' condition fails."""
        state = self._state()
        state.get_hero("hero_emmitt").resolved_turn_count = 1
        state.turn = 2
        enemy = state.get_hero("hero_enemy")
        enemy.played_cards = [_resolved_card("turn1_card")]
        enemy.resolved_turn_count = 1
        enemy.current_turn_card = _unresolved_card("pending")
        assert self._passes(state) is False

    def test_resolved_but_not_yet_finalized_passes(self):
        """current_turn_card RESOLVED (mid-finalization / action-control
        window) counts as resolved this turn."""
        state = self._state()
        state.get_hero("hero_enemy").current_turn_card = _resolved_card("just_done")
        assert self._passes(state) is True

    def test_non_hero_candidate_fails(self):
        state = self._state()
        assert self._passes(state, candidate="minion_1") is False


# =============================================================================
# TIME SNARE / TIME TRAP (§2): "An enemy hero in range who has already resolved
# a card this turn discards a card, if able."
# TIME BOMB (§3): same targeting, but "discards a card, or is defeated."
# =============================================================================


def _discard_state(
    card_id: str = "time_snare",
    *,
    enemy_at=(2, 0, -2),
    enemy_status: str = "resolved",
    hand_count: int = 1,
):
    state = (
        EffectScenarioBuilder()
        .line_board(8)
        .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", card_id))
        .blue_hero("hero_enemy", at=enemy_at)
        .with_actor("hero_emmitt")
        .build()
    )
    enemy = state.get_hero("hero_enemy")
    if enemy_status == "resolved":
        enemy.played_cards = [_resolved_card("enemy_resolved", initiative=9)]
        enemy.resolved_turn_count = 1
    elif enemy_status == "unresolved":
        enemy.current_turn_card = _unresolved_card("enemy_pending", initiative=9)
    elif enemy_status != "passed":
        raise ValueError(f"Unknown enemy status: {enemy_status}")
    enemy.hand = [_hand_card(f"enemy_hand_{i}") for i in range(hand_count)]
    return state


@pytest.mark.effect_flow
class TestTimeSnareTrap:
    def test_time_snare_forces_victim_to_discard_if_able(self):
        state = _discard_state("time_snare", hand_count=2)
        enemy = state.get_hero("hero_enemy")
        discarded = enemy.hand[1]

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_CARD")
        assert run.latest_request.player_id == "hero_enemy"
        run.choose(discarded.id)
        run.finish()

        assert discarded in enemy.discard_pile
        assert discarded not in enemy.hand
        assert state.turn_discard_log["hero_enemy"] == [discarded.id]
        assert state.get_position("hero_enemy") == Hex(q=2, r=0, s=-2)

    def test_time_snare_two_valid_targets_emmitt_chooses_which(self):
        state = (
            EffectScenarioBuilder()
            .line_board(8)
            .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "time_snare"))
            .blue_hero("hero_enemy_a", at=(1, 0, -1))
            .blue_hero("hero_enemy_b", at=(2, 0, -2))
            .with_actor("hero_emmitt")
            .build()
        )
        for suffix in ("a", "b"):
            enemy = state.get_hero(f"hero_enemy_{suffix}")
            enemy.played_cards = [_resolved_card(f"enemy_{suffix}_resolved")]
            enemy.resolved_turn_count = 1
            enemy.hand = [_hand_card(f"enemy_{suffix}_hand")]

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT")
        assert {o.id for o in run.latest_request.options} == {"hero_enemy_a", "hero_enemy_b"}
        run.choose("hero_enemy_b")
        run.expect_input("SELECT_CARD")
        assert run.latest_request.player_id == "hero_enemy_b"
        run.choose("enemy_b_hand")
        run.finish()

        assert [c.id for c in state.get_hero("hero_enemy_a").hand] == ["enemy_a_hand"]
        assert [c.id for c in state.get_hero("hero_enemy_b").discard_pile] == ["enemy_b_hand"]

    def test_time_trap_reaches_range_three(self):
        state = _discard_state("time_trap", enemy_at=(3, 0, -3))
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_CARD").choose("enemy_hand_0")
        run.finish()

        assert [c.id for c in state.get_hero("hero_enemy").discard_pile] == ["enemy_hand_0"]

    def test_time_snare_out_of_range_aborts(self):
        state = _discard_state("time_snare", enemy_at=(3, 0, -3))
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()

        enemy = state.get_hero("hero_enemy")
        assert [c.id for c in enemy.hand] == ["enemy_hand_0"]

    def test_unresolved_or_passed_enemy_not_targetable(self):
        for status in ("unresolved", "passed"):
            state = _discard_state("time_snare", enemy_status=status)
            run = run_card(state, "hero_emmitt")
            run.expect_input("CHOOSE_ACTION").choose("SKILL")
            run.finish()
            assert [c.id for c in state.get_hero("hero_enemy").hand] == ["enemy_hand_0"]

    def test_empty_hand_has_no_penalty_for_if_able_cards(self):
        state = _discard_state("time_snare", hand_count=0)
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=2, r=0, s=-2)
        assert not any(e.event_type == GameEventType.UNIT_DEFEATED for e in run.events)

    def test_immune_enemy_excluded(self):
        state = _discard_state("time_snare")
        state.active_effects.append(
            ActiveEffect(
                id="snare_immune",
                source_id="hero_enemy",
                effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
                scope=EffectScope(shape=Shape.POINT, origin_id="hero_enemy"),
                duration=DurationType.THIS_ROUND,
                is_active=True,
                created_at_turn=1,
                created_at_round=1,
            )
        )

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()

        assert [c.id for c in state.get_hero("hero_enemy").hand] == ["enemy_hand_0"]


@pytest.mark.effect_flow
class TestTimeBomb:
    def test_victim_with_cards_discards_instead_of_defeat(self):
        state = _discard_state("time_bomb", enemy_at=(3, 0, -3), hand_count=1)
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_CARD")
        assert run.latest_request.player_id == "hero_enemy"
        run.choose("enemy_hand_0")
        run.finish()

        enemy = state.get_hero("hero_enemy")
        assert [c.id for c in enemy.discard_pile] == ["enemy_hand_0"]
        assert state.get_position("hero_enemy") == Hex(q=3, r=0, s=-3)
        assert not any(e.event_type == GameEventType.UNIT_DEFEATED for e in run.events)

    def test_victim_with_empty_hand_is_defeated(self):
        state = _discard_state("time_bomb", enemy_at=(3, 0, -3), hand_count=0)
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.finish()

        defeated = [
            e
            for e in run.events
            if e.event_type == GameEventType.UNIT_DEFEATED and e.target_id == "hero_enemy"
        ]
        assert defeated
        assert defeated[-1].actor_id == "hero_emmitt"
        assert state.get_position("hero_enemy") is None
        assert state.get_hero("hero_emmitt").gold == 1

    def test_same_targeting_gate_as_time_snare(self):
        state = _discard_state("time_bomb", enemy_status="unresolved")
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=2, r=0, s=-2)
        assert [c.id for c in state.get_hero("hero_enemy").hand] == ["enemy_hand_0"]


# =============================================================================
# TIME LOOP (§4): "Swap with an enemy hero in range who has already resolved
# a card this turn." (range 4)
# =============================================================================


def _swap_block_effect(origin_id: str) -> ActiveEffect:
    """Bulwark-style area effect: enemy actors cannot SWAP covered heroes."""
    from goa2.domain.models.enums import DisplacementType

    return ActiveEffect(
        id="swap_block_test",
        source_id=origin_id,
        effect_type=EffectType.PLACEMENT_PREVENTION,
        scope=EffectScope(
            shape=Shape.RADIUS,
            range=1,
            origin_id=origin_id,
            affects=AffectsFilter.FRIENDLY_HEROES,
        ),
        duration=DurationType.THIS_ROUND,
        is_active=True,
        created_at_turn=1,
        created_at_round=1,
        displacement_blocks=[DisplacementType.SWAP],
        blocks_enemy_actors=True,
    )


def _loop_state(card_id: str = "time_loop", *, enemy_resolved: bool = True, enemy_at=(3, 0, -3)):
    state = (
        EffectScenarioBuilder()
        .line_board(8)
        .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", card_id))
        .blue_hero("hero_enemy", at=enemy_at)
        .with_actor("hero_emmitt")
        .build()
    )
    enemy = state.get_hero("hero_enemy")
    if enemy_resolved:
        enemy.played_cards = [_resolved_card("prev_resolved", initiative=9)]
        enemy.resolved_turn_count = 1
    else:
        enemy.current_turn_card = _unresolved_card("pending")
    return state


@pytest.mark.effect_flow
class TestTimeLoop:
    def test_swap_with_resolved_enemy(self):
        from goa2.domain.events import GameEventType

        state = _loop_state()
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.finish()

        assert state.get_position("hero_emmitt") == Hex(q=3, r=0, s=-3)
        assert state.get_position("hero_enemy") == Hex(q=0, r=0, s=0)
        assert any(e.event_type == GameEventType.UNITS_SWAPPED for e in run.events)

    def test_no_resolved_enemy_aborts(self):
        state = _loop_state(enemy_resolved=False)
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()  # mandatory select has no candidates → abort
        assert state.get_position("hero_emmitt") == Hex(q=0, r=0, s=0)
        assert state.get_position("hero_enemy") == Hex(q=3, r=0, s=-3)

    def test_enemy_out_of_range_aborts(self):
        state = _loop_state(enemy_at=(5, 0, -5))  # range 4
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()
        assert state.get_position("hero_emmitt") == Hex(q=0, r=0, s=0)

    def test_swap_prevention_denies_and_aborts(self):
        """Displacement validation: a swap-blocking area effect on the target
        denies the swap; mandatory → abort, positions unchanged."""
        state = _loop_state()
        # Guard next to the enemy projects the swap-blocking aura over them.
        state = (
            EffectScenarioBuilder()
            .line_board(8)
            .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "time_loop"))
            .blue_hero("hero_enemy", at=(3, 0, -3))
            .blue_hero("hero_guard", at=(4, 0, -4))
            .with_actor("hero_emmitt")
            .build()
        )
        enemy = state.get_hero("hero_enemy")
        enemy.played_cards = [_resolved_card("prev_resolved")]
        enemy.resolved_turn_count = 1
        state.active_effects.append(_swap_block_effect("hero_guard"))

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.finish()
        assert state.get_position("hero_emmitt") == Hex(q=0, r=0, s=0)
        assert state.get_position("hero_enemy") == Hex(q=3, r=0, s=-3)

    def test_immune_enemy_excluded(self):
        state = _loop_state()
        state.active_effects.append(
            ActiveEffect(
                id="imm_test",
                source_id="hero_enemy",
                effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
                scope=EffectScope(shape=Shape.POINT, origin_id="hero_enemy"),
                duration=DurationType.THIS_ROUND,
                is_active=True,
                created_at_turn=1,
                created_at_round=1,
            )
        )
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()  # only candidate immune → abort
        assert state.get_position("hero_emmitt") == Hex(q=0, r=0, s=0)


# =============================================================================
# TIME WARP (§5): Choose one — (A) Time Loop position swap / (B) enemy swaps
# their unresolved card with one of their resolved cards, of THEIR choice.
# =============================================================================


def _warp_state(*, enemy_has_resolved: bool = True, enemy_at=(3, 0, -3)):
    """Enemy with an unresolved card this turn (bullet B shape); optionally
    one resolved card from an earlier turn to swap in."""
    state = (
        EffectScenarioBuilder()
        .line_board(8)
        .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "time_warp"))
        .blue_hero("hero_enemy", at=enemy_at)
        .with_actor("hero_emmitt")
        .build()
    )
    enemy = state.get_hero("hero_enemy")
    enemy.current_turn_card = _unresolved_card("enemy_pending", initiative=7)
    if enemy_has_resolved:
        enemy.played_cards = [_resolved_card("enemy_prev", initiative=2)]
        enemy.resolved_turn_count = 1
    return state


@pytest.mark.effect_flow
class TestTimeWarp:
    def test_bullet_a_swaps_positions_like_time_loop(self):
        state = _loop_state("time_warp")
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(1)
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.finish()
        assert state.get_position("hero_emmitt") == Hex(q=3, r=0, s=-3)
        assert state.get_position("hero_enemy") == Hex(q=0, r=0, s=0)

    def test_bullet_b_enemy_picks_resolved_card_and_swaps(self):
        from goa2.domain.models.enums import CardState

        state = _warp_state()
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(2)
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_CARD")
        assert run.latest_request.player_id == "hero_enemy"  # THEIR choice
        run.choose("enemy_prev")
        run.finish()

        enemy = state.get_hero("hero_enemy")
        # Old resolved card is now the (unresolved) current turn card…
        assert enemy.current_turn_card is not None
        assert enemy.current_turn_card.id == "enemy_prev"
        assert enemy.current_turn_card.state == CardState.UNRESOLVED
        # …and the swapped-out card sits RESOLVED in the played slot (U3:
        # it never resolves its action).
        assert [c.id for c in enemy.played_cards if c is not None] == ["enemy_pending"]
        assert enemy.played_cards[0].state == CardState.RESOLVED

    def test_bullet_b_resolution_order_follows_swapped_in_card(self):
        """H3/H4: after the swap the enemy still acts this turn, ordered by
        the swapped-in card's initiative (7→2 drops them behind an init-5
        hero)."""
        state = (
            EffectScenarioBuilder()
            .line_board(8)
            .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "time_warp"))
            .blue_hero("hero_enemy", at=(3, 0, -3))
            .blue_hero("hero_other", at=(5, 0, -5))
            .with_actor("hero_emmitt")
            .build()
        )
        enemy = state.get_hero("hero_enemy")
        enemy.current_turn_card = _unresolved_card("enemy_pending", initiative=7)
        enemy.played_cards = [_resolved_card("enemy_prev", initiative=2)]
        enemy.resolved_turn_count = 1
        state.get_hero("hero_other").current_turn_card = _unresolved_card(
            "other_card", initiative=5
        )

        run = run_card(state, "hero_emmitt", finalize_turn=True)
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(2)
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_CARD").choose("enemy_prev")

        # Finalizing Emmitt's turn hands over to the next actor: without the
        # swap hero_enemy (init 7) would act before hero_other (init 5); with
        # the swapped-in init-2 card, hero_other goes first.
        run.expect_input("CHOOSE_ACTION")
        assert run.latest_request.player_id == "hero_other"
        assert str(state.current_actor_id) == "hero_other"

    def test_bullet_b_enemy_without_resolved_card_not_targetable(self):
        state = _warp_state(enemy_has_resolved=False)
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(2)
        run.finish()  # no valid target → mandatory → abort
        enemy = state.get_hero("hero_enemy")
        assert enemy.current_turn_card is not None
        assert enemy.current_turn_card.id == "enemy_pending"

    def test_bullet_b_no_enemy_in_range_aborts(self):
        state = _warp_state(enemy_at=(6, 0, -6))  # beyond range 4
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(2)
        run.finish()
        enemy = state.get_hero("hero_enemy")
        assert enemy.current_turn_card is not None
        assert enemy.current_turn_card.id == "enemy_pending"


# =============================================================================
# P8 primitive: TokenType.GLITCH, supply 3, non-persistent
# =============================================================================


@pytest.mark.effect_contract
def test_glitch_token_supply_is_three():
    from goa2.domain.models import TokenType
    from goa2.domain.models.token import TOKEN_SUPPLY
    from goa2.engine.setup import GameSetup

    assert TOKEN_SUPPLY[TokenType.GLITCH] == 3

    state = (
        EffectScenarioBuilder()
        .line_board(4)
        .red_hero("hero_a", at=(0, 0, 0))
        .blue_hero("hero_b", at=(2, 0, -2))
        .build()
    )
    GameSetup._initialize_token_pool(state)
    glitch = state.token_pool[TokenType.GLITCH]
    assert len(glitch) == 3
    assert all(not t.persists_end_of_round for t in glitch)
    assert all(not t.is_passable for t in glitch)  # tokens are obstacles (U6)


# =============================================================================
# P10 primitive: PlaceTokenBatchStep — upfront supply reconciliation, batch
# spacing, all-or-nothing feasibility, optional opt-in prompt
# =============================================================================


def _add_glitch_pool(state, on_board=()) -> list[str]:
    from goa2.domain.models import TokenType
    from goa2.domain.models.token import Token

    state.token_pool[TokenType.GLITCH] = []
    ids: list[str] = []
    for i in range(3):
        token = Token(id=f"glitch_{i + 1}", name="Glitch", token_type=TokenType.GLITCH)
        state.register_entity(token)
        state.token_pool[TokenType.GLITCH].append(token)
        ids.append(f"glitch_{i + 1}")
    for i, at in enumerate(on_board):
        state.place_entity(f"glitch_{i + 1}", Hex(q=at[0], r=at[1], s=at[2]))
    return ids


def _glitch_on_board(state) -> dict[str, Hex]:
    from goa2.domain.models import TokenType

    out = {}
    for token in state.token_pool.get(TokenType.GLITCH, []):
        pos = state.get_position(str(token.id))
        if pos is not None:
            out[str(token.id)] = pos
    return out


def _batch_state(board_len: int = 12, on_board=(), actor: str = "hero_emmitt"):
    state = (
        EffectScenarioBuilder()
        .line_board(board_len)
        .red_hero("hero_emmitt", at=(0, 0, 0))
        .blue_hero("hero_enemy", at=(board_len - 1, 0, -(board_len - 1)))
        .with_actor(actor)
        .build()
    )
    _add_glitch_pool(state, on_board=on_board)
    return state


def _push_batch(state, **kwargs):
    from goa2.domain.models import TokenType
    from goa2.engine.filters import RangeFilter
    from goa2.engine.handler import process_stack, push_steps
    from goa2.engine.steps import PlaceTokenBatchStep

    kwargs.setdefault("token_type", TokenType.GLITCH)
    kwargs.setdefault("is_mandatory", False)
    kwargs.setdefault("placed_flag_key", "glitch_placed")
    if "slot_filters" not in kwargs:
        count = kwargs.get("count", 0)
        key_prefix = kwargs.get("key_prefix", "tkb")
        kwargs["slot_filters"] = [
            [
                RangeFilter(min_range=3, max_range=None, origin_hex_key=f"{key_prefix}_hex_{j}")
                for j in range(i)
            ]
            for i in range(count)
        ]
    push_steps(state, [PlaceTokenBatchStep(**kwargs)])
    return process_stack(state)


def _answer(state, value):
    from goa2.engine.handler import process_stack

    state.execution_stack[-1].pending_input = {"selection": value}
    return process_stack(state)


def _hex_dict(q: int) -> dict:
    return {"q": q, "r": 0, "s": -q}


def _option_qs(request) -> set[int]:
    """Extract q coordinates from a SELECT_HEX request's options."""
    qs = set()
    for option in request.options:
        meta = getattr(option, "metadata", None) or {}
        raw = meta.get("hex") or meta.get("raw")
        if raw is None:
            continue
        qs.add(raw["q"] if isinstance(raw, dict) else raw.q)
    return qs


def _option_hex_tuples(request) -> set[tuple[int, int, int]]:
    out = set()
    for option in request.options:
        meta = getattr(option, "metadata", None) or {}
        raw = meta.get("hex") or meta.get("raw")
        if raw is None:
            continue
        if isinstance(raw, dict):
            out.add((raw["q"], raw["r"], raw["s"]))
        else:
            out.add((raw.q, raw.r, raw.s))
    return out


@pytest.mark.effect_contract
class TestPlaceTokenBatchStep:
    def test_count_exceeding_total_supply_fails(self):
        state = _batch_state()
        result = _push_batch(state, count=4)
        assert result.input_request is None
        assert _glitch_on_board(state) == {}
        assert state.execution_context.get("glitch_placed") is None

    def test_places_all_from_free_supply(self):
        state = _batch_state()
        result = _push_batch(state, count=3)
        for q in (1, 4, 7):
            assert result.input_request is not None
            assert result.input_request.request_type.value == "SELECT_HEX"
            result = _answer(state, _hex_dict(q))
        assert result.input_request is None
        placed = _glitch_on_board(state)
        assert len(placed) == 3
        assert {h.q for h in placed.values()} == {1, 4, 7}
        assert state.execution_context.get("glitch_placed") is True

    def test_spacing_enforced_between_batch_picks(self):
        state = _batch_state()
        result = _push_batch(state, count=2)
        result = _answer(state, _hex_dict(4))  # first token at q=4
        assert result.input_request is not None
        qs = _option_qs(result.input_request)
        # distance ≥ 3 from q=4: q ∈ {1, 7, 8, 9, 10} (0 and 11 occupied)
        assert qs == {1, 7, 8, 9, 10}

    def test_removal_of_preexisting_prompted_before_placement(self):
        state = _batch_state(on_board=[(9, 0, -9)])
        result = _push_batch(state, count=3)
        # Free supply is 2 < 3 → first prompt removes a pre-existing token.
        assert result.input_request is not None
        assert result.input_request.request_type.value != "SELECT_HEX"
        option_ids = {o.id for o in result.input_request.options}
        assert option_ids == {"glitch_1"}  # only the pre-existing token
        result = _answer(state, "glitch_1")
        for q in (1, 4, 7):
            assert result.input_request is not None
            assert result.input_request.request_type.value == "SELECT_HEX"
            result = _answer(state, _hex_dict(q))
        placed = _glitch_on_board(state)
        assert len(placed) == 3
        assert {h.q for h in placed.values()} == {1, 4, 7}  # q=9 freed
        assert state.execution_context.get("glitch_placed") is True

    def test_infeasible_spacing_skips_without_prompts(self):
        state = _batch_state(board_len=4)  # q 0..3, 0 and 3 occupied
        result = _push_batch(state, count=2)  # q1/q2 are distance 1 apart
        assert result.input_request is None
        assert _glitch_on_board(state) == {}
        assert state.execution_context.get("glitch_placed") is None

    def test_opt_in_decline_places_nothing(self):
        state = _batch_state()
        result = _push_batch(state, count=2, opt_in_prompt="Place Glitch tokens?")
        assert result.input_request is not None
        assert result.input_request.request_type.value == "SELECT_NUMBER"
        result = _answer(state, 0)
        assert result.input_request is None
        assert _glitch_on_board(state) == {}
        assert state.execution_context.get("glitch_placed") is None

    def test_opt_in_accept_flows_to_placement(self):
        state = _batch_state()
        result = _push_batch(state, count=2, opt_in_prompt="Place Glitch tokens?")
        result = _answer(state, 1)
        for q in (1, 4):
            assert result.input_request is not None
            assert result.input_request.request_type.value == "SELECT_HEX"
            result = _answer(state, _hex_dict(q))
        assert len(_glitch_on_board(state)) == 2
        assert state.execution_context.get("glitch_placed") is True

    def test_opt_in_not_offered_when_infeasible(self):
        state = _batch_state(board_len=4)
        result = _push_batch(state, count=2, opt_in_prompt="Place Glitch tokens?")
        assert result.input_request is None
        assert _glitch_on_board(state) == {}

    def test_prompts_routed_to_owner_not_actor(self):
        """Defense mode: the enemy is the current actor but Emmitt owns the
        placement — every prompt goes to Emmitt."""
        state = _batch_state(on_board=[(9, 0, -9)], actor="hero_enemy")
        result = _push_batch(state, count=3, owner_id="hero_emmitt")
        assert result.input_request is not None
        assert result.input_request.player_id == "hero_emmitt"  # removal
        result = _answer(state, "glitch_1")
        assert result.input_request is not None
        assert result.input_request.player_id == "hero_emmitt"  # hex select


@pytest.mark.effect_contract
class TestPlaceTokenBatchRemovalFeasibility:
    """Rules fidelity for supply-short batches: the feasibility precheck counts
    hexes freed by the forced removals (within the removal budget), and each
    removal prompt only offers tokens from which the batch can still complete.
    """

    def test_batch_feasible_only_via_removed_token_hex_proceeds(self):
        # Board q0..8, heroes at 0/8, glitch_1 at q4. The empty hexes
        # {1,2,3,5,6,7} admit no triple with pairwise spacing >= 3; freeing
        # q4 admits (1,4,7), and one removal is forced (free supply 2 < 3).
        state = _batch_state(board_len=9, on_board=[(4, 0, -4)])
        result = _push_batch(state, count=3)
        assert result.input_request is not None
        assert result.input_request.request_type.value != "SELECT_HEX"  # removal first
        assert {o.id for o in result.input_request.options} == {"glitch_1"}
        result = _answer(state, "glitch_1")
        for q in (1, 4, 7):
            assert result.input_request is not None
            assert result.input_request.request_type.value == "SELECT_HEX"
            result = _answer(state, _hex_dict(q))
        assert result.input_request is None
        placed = _glitch_on_board(state)
        assert {h.q for h in placed.values()} == {1, 4, 7}
        assert state.execution_context.get("glitch_placed") is True

    def test_removal_prompt_excludes_dead_end_tokens(self):
        # Board q0..5, heroes at 0/5, glitch_1 at q1, glitch_2 at q2; empty
        # {3,4}. One removal is forced. Only freeing q1 enables a spaced pair
        # (1,4); freeing q2 leaves {2,3,4} with max distance 2 — so glitch_2
        # must not be offered.
        state = _batch_state(board_len=6, on_board=[(1, 0, -1), (2, 0, -2)])
        result = _push_batch(state, count=2)
        assert result.input_request is not None
        assert {o.id for o in result.input_request.options} == {"glitch_1"}
        result = _answer(state, "glitch_1")
        result = _answer(state, _hex_dict(1))
        result = _answer(state, _hex_dict(4))
        assert result.input_request is None
        # glitch_2 stays at q2; the batch pair lands on q1 (freed) and q4.
        assert {h.q for h in _glitch_on_board(state).values()} == {1, 2, 4}

    def test_freed_hexes_beyond_removal_budget_do_not_count(self):
        # glitch_1 at q1, glitch_2 at q4; empty {2,3}. The only spaced pair
        # (1,4) needs BOTH token hexes freed, but only one removal is forced —
        # the batch must be skipped and both tokens stay on the board.
        state = _batch_state(board_len=6, on_board=[(1, 0, -1), (4, 0, -4)])
        result = _push_batch(state, count=2)
        assert result.input_request is None
        assert {h.q for h in _glitch_on_board(state).values()} == {1, 4}
        assert state.execution_context.get("glitch_placed") is None

    def test_sequential_removals_track_remaining_budget(self):
        # All three tokens on board (q1, q2, q3); empty {4,5}; count 2 forces
        # two removals. Every first removal keeps a completion reachable; the
        # second prompt re-evaluates against the post-removal board.
        state = _batch_state(board_len=7, on_board=[(1, 0, -1), (2, 0, -2), (3, 0, -3)])
        result = _push_batch(state, count=2)
        assert result.input_request is not None
        assert {o.id for o in result.input_request.options} == {
            "glitch_1",
            "glitch_2",
            "glitch_3",
        }
        result = _answer(state, "glitch_3")  # frees q3
        assert result.input_request is not None
        assert {o.id for o in result.input_request.options} == {"glitch_1", "glitch_2"}
        result = _answer(state, "glitch_1")  # frees q1
        result = _answer(state, _hex_dict(1))
        result = _answer(state, _hex_dict(4))
        assert result.input_request is None
        placed = _glitch_on_board(state)
        assert {h.q for h in placed.values()} == {1, 2, 4}  # glitch_2 untouched at q2
        assert state.execution_context.get("glitch_placed") is True

    def test_removal_steering_survives_save_load(self):
        # Persistence restores steps whose nested slot filters deserialize as
        # base FilterCondition; the steering filter must still evaluate.
        from goa2.domain.state import GameState
        from goa2.engine.handler import process_stack

        state = _batch_state(board_len=6, on_board=[(1, 0, -1), (2, 0, -2)])
        result = _push_batch(state, count=2)
        assert result.input_request is not None

        restored = GameState.model_validate(state.model_dump(mode="json"))
        result = process_stack(restored)  # re-resolve the pending removal select
        assert result.input_request is not None
        assert {o.id for o in result.input_request.options} == {"glitch_1"}
        result = _answer(restored, "glitch_1")
        result = _answer(restored, _hex_dict(1))
        result = _answer(restored, _hex_dict(4))
        assert result.input_request is None
        assert {h.q for h in _glitch_on_board(restored).values()} == {1, 2, 4}

    def test_opt_in_shown_when_feasible_only_via_removal(self):
        # The opt-in prompt must appear (the batch IS feasible thanks to the
        # forced removal), and declining must leave the board untouched.
        state = _batch_state(board_len=9, on_board=[(4, 0, -4)])
        result = _push_batch(state, count=3, opt_in_prompt="Place Glitch tokens?")
        assert result.input_request is not None
        assert result.input_request.request_type.value == "SELECT_NUMBER"
        result = _answer(state, 0)  # decline
        assert result.input_request is None
        assert {h.q for h in _glitch_on_board(state).values()} == {4}  # nothing removed
        assert state.execution_context.get("glitch_placed") is None


# =============================================================================
# FLASHBACK / DÉJÀ VU (§10): attack adjacent; after the attack you may place
# 3/2 Glitch tokens in radius (spacing ≥ 3); if you do, up to 1 enemy hero in
# radius swaps with a Glitch token of their choice. End of turn: remove all.
# =============================================================================


def _disc_hexes(radius: int) -> list[tuple[int, int, int]]:
    return [
        (q, r, -q - r)
        for q in range(-radius, radius + 1)
        for r in range(max(-radius, -radius - q), min(radius, radius - q) + 1)
    ]


_TOKEN_SPOTS = [(3, 0, -3), (0, 3, -3), (-3, 0, 3)]  # pairwise distance ≥ 3


def _glitch_card_state(
    card_id: str = "flashback",
    *,
    board_radius: int = 4,
    with_minion: bool = True,
    with_far_hero: bool = False,
):
    builder = (
        EffectScenarioBuilder()
        .with_hexes(_disc_hexes(board_radius))
        .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", card_id))
        .blue_hero("hero_victim", at=(0, -2, 2))
        .with_actor("hero_emmitt")
    )
    if with_minion:
        builder = builder.blue_minion("minion_target", at=(1, 0, -1))
    if with_far_hero:
        builder = builder.blue_hero("hero_far", at=(4, 0, -4))
    state = builder.build()
    _add_glitch_pool(state)
    return state


@pytest.mark.effect_flow
class TestFlashback:
    def _run_through_placement(self, state, count: int = 3):
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("minion_target")
        run.expect_input("SELECT_NUMBER")  # opt-in
        assert run.latest_request.player_id == "hero_emmitt"
        run.choose(1)
        for q, r, s in _TOKEN_SPOTS[:count]:
            run.expect_input("SELECT_HEX").choose({"q": q, "r": r, "s": s})
        return run

    def test_full_flow_place_three_then_enemy_swaps(self):
        state = _glitch_card_state(with_far_hero=True)
        run = self._run_through_placement(state)

        # Rider: Emmitt may pick 1 enemy hero in radius (far hero excluded, U4)
        run.expect_input("SELECT_UNIT")
        victim_options = {o.id for o in run.latest_request.options}
        assert "hero_victim" in victim_options
        assert "hero_far" not in victim_options
        run.choose("hero_victim")

        # That hero's player picks the Glitch token (U3-style routing)
        run.expect_input("SELECT_UNIT_OR_TOKEN")
        assert run.latest_request.player_id == "hero_victim"
        run.choose("glitch_1")  # placed at (3, 0, -3)
        run.finish()

        assert state.get_position("hero_victim") == Hex(q=3, r=0, s=-3)
        assert state.get_position("glitch_1") == Hex(q=0, r=-2, s=2)
        assert len(_glitch_on_board(state)) == 3

    def test_rider_is_optional(self):
        """H4: 'up to 1' — Emmitt may skip the swap after placing."""
        state = _glitch_card_state()
        run = self._run_through_placement(state)
        run.expect_input("SELECT_UNIT").skip()
        run.finish()
        assert state.get_position("hero_victim") == Hex(q=0, r=-2, s=2)
        assert len(_glitch_on_board(state)) == 3

    def test_decline_placement_no_tokens_no_rider(self):
        """U2: opting out places nothing and never offers the swap."""
        state = _glitch_card_state()
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("minion_target")
        run.expect_input("SELECT_NUMBER").choose(0)
        run.finish()
        assert _glitch_on_board(state) == {}

    def test_attack_abort_skips_everything(self):
        """U1: no adjacent unit → attack aborts → no placement offer."""
        state = _glitch_card_state(with_minion=False)  # victim is 2 away
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.finish()
        assert _glitch_on_board(state) == {}

    def test_infeasible_board_never_offers_placement(self):
        """U3/U8: board can't fit all 3 with spacing → option unavailable."""
        state = _glitch_card_state(board_radius=1)
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("minion_target")
        run.finish()  # no opt-in prompt, no rider
        assert _glitch_on_board(state) == {}

    def test_end_of_turn_removes_all_glitch_tokens(self):
        """H5: THIS_TURN cleanup through the real end_turn."""
        from goa2.engine.handler import process_stack
        from goa2.engine.phases import end_turn

        state = _glitch_card_state()
        run = self._run_through_placement(state)
        run.expect_input("SELECT_UNIT").skip()
        run.finish()
        assert len(_glitch_on_board(state)) == 3

        state.unresolved_hero_ids = []
        end_turn(state)
        process_stack(state)
        assert _glitch_on_board(state) == {}


@pytest.mark.effect_flow
class TestDejaVu:
    def test_places_two_tokens_then_rider(self):
        state = _glitch_card_state("deja_vu")
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("minion_target")
        run.expect_input("SELECT_NUMBER").choose(1)
        for q, r, s in _TOKEN_SPOTS[:2]:
            run.expect_input("SELECT_HEX").choose({"q": q, "r": r, "s": s})
        run.expect_input("SELECT_UNIT").skip()  # rider offered after 2 tokens
        run.finish()
        assert len(_glitch_on_board(state)) == 2


# =============================================================================
# UNSTABLE TIMELINE (§11): place 2 Glitch tokens in radius (3 as a defense);
# an enemy hero in play (Emmitt picks which) chooses one; EMMITT swaps with
# that token. End of turn: remove all Glitch tokens.
# =============================================================================


def _attack_card(card_id: str = "enemy_attack", value: int = 5):
    from goa2.domain.models import ActionType, Card, CardColor, CardTier

    return Card(
        id=card_id,
        name="Enemy Attack",
        tier=CardTier.I,
        color=CardColor.RED,
        initiative=5,
        primary_action=ActionType.ATTACK,
        primary_action_value=value,
        secondary_actions={},
        is_ranged=False,
        range_value=1,
        effect_id="",
        effect_text="",
        is_facedown=False,
    )


def _timeline_skill_state(*, board_radius: int = 5, with_enemy: bool = True):
    builder = (
        EffectScenarioBuilder()
        .with_hexes(_disc_hexes(board_radius))
        .red_hero(
            "hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "unstable_timeline")
        )
        .with_actor("hero_emmitt")
    )
    if with_enemy:
        # Distance 5 from Emmitt — outside the card's radius 4, proving the
        # chooser has no range limit ("an enemy hero in play").
        builder = builder.blue_hero("hero_enemy", at=(0, -5, 5))
    state = builder.build()
    _add_glitch_pool(state)
    return state


@pytest.mark.effect_flow
class TestUnstableTimelineSkill:
    @pytest.mark.parametrize("round_trip", [False, True])
    def test_spacing_across_reality_split_is_physical(self, round_trip):
        from goa2.domain.state import GameState

        state = _timeline_skill_state(with_enemy=False)
        state.active_effects.append(
            ActiveEffect(
                id="split",
                source_id="hero_nebkher",
                effect_type=EffectType.TOPOLOGY_SPLIT,
                created_at_turn=1,
                created_at_round=1,
                is_active=True,
                scope=EffectScope(shape=Shape.GLOBAL),
                duration=DurationType.THIS_TURN,
                split_axis="q",
                split_value=0,
            )
        )
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_HEX").choose({"q": -1, "r": 0, "s": 1})
        run.expect_input("SELECT_HEX")
        if round_trip:
            state = GameState.model_validate_json(state.model_dump_json())
            run.state = state
            run.expect_input("SELECT_HEX")
        options = _option_hex_tuples(run.latest_request)
        assert (1, 0, -1) not in options  # distance 2, even across the split
        assert (2, 0, -2) in options  # distance 3 on the opposite side is legal
        run.choose({"q": 1, "r": 0, "s": -1}).expect_input("SELECT_HEX")
        assert len(_glitch_on_board(state)) == 1
        run.choose({"q": 2, "r": 0, "s": -2}).finish()
        assert len(_glitch_on_board(state)) == 2

    def test_split_does_not_make_an_infeasible_batch_possible(self):
        state = (
            EffectScenarioBuilder()
            .with_hexes([(-1, 0, 1), (0, 0, 0), (1, 0, -1)])
            .red_hero(
                "hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "unstable_timeline")
            )
            .with_actor("hero_emmitt")
            .build()
        )
        _add_glitch_pool(state)
        state.active_effects.append(
            ActiveEffect(
                id="split",
                source_id="hero_nebkher",
                effect_type=EffectType.TOPOLOGY_SPLIT,
                created_at_turn=1,
                created_at_round=1,
                is_active=True,
                scope=EffectScope(shape=Shape.GLOBAL),
                duration=DurationType.THIS_TURN,
                split_axis="q",
                split_value=0,
            )
        )
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL").finish()
        assert _glitch_on_board(state) == {}

    def test_place_two_enemy_chooses_emmitt_swaps(self):
        state = _timeline_skill_state()
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        for q, r, s in _TOKEN_SPOTS[:2]:
            run.expect_input("SELECT_HEX").choose({"q": q, "r": r, "s": s})
        # Emmitt picks the chooser — even beyond any range (distance 5)
        run.expect_input("SELECT_UNIT")
        assert run.latest_request.player_id == "hero_emmitt"
        run.choose("hero_enemy")
        # The chosen enemy picks the token
        run.expect_input("SELECT_UNIT_OR_TOKEN")
        assert run.latest_request.player_id == "hero_enemy"
        run.choose("glitch_2")  # placed at (0, 3, -3)
        run.finish()

        assert state.get_position("hero_emmitt") == Hex(q=0, r=3, s=-3)
        assert state.get_position("glitch_2") == Hex(q=0, r=0, s=0)
        assert len(_glitch_on_board(state)) == 2

    def test_infeasible_placement_stops_text(self):
        """U1: tokens can't all be placed → whole text stops, no swap."""
        state = _timeline_skill_state(board_radius=1)
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()  # no placement, no chooser
        assert _glitch_on_board(state) == {}
        assert state.get_position("hero_emmitt") == Hex(q=0, r=0, s=0)

    def test_no_enemy_hero_in_play_stops_before_swap(self):
        """U2: tokens are placed but stay; no chooser, no swap."""
        state = _timeline_skill_state()
        state.remove_entity("hero_enemy")  # defeated / not in play
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        for q, r, s in _TOKEN_SPOTS[:2]:
            run.expect_input("SELECT_HEX").choose({"q": q, "r": r, "s": s})
        run.finish()  # no chooser prompt
        assert len(_glitch_on_board(state)) == 2  # tokens remain (U2)
        assert state.get_position("hero_emmitt") == Hex(q=0, r=0, s=0)

    def test_end_of_turn_removes_tokens(self):
        from goa2.engine.handler import process_stack
        from goa2.engine.phases import end_turn

        state = _timeline_skill_state()
        state.remove_entity("hero_enemy")
        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        for q, r, s in _TOKEN_SPOTS[:2]:
            run.expect_input("SELECT_HEX").choose({"q": q, "r": r, "s": s})
        run.finish()

        state.unresolved_hero_ids = []
        end_turn(state)
        process_stack(state)
        assert _glitch_on_board(state) == {}


@pytest.mark.effect_flow
class TestUnstableTimelineDefense:
    def _state(self, *, board_radius: int = 4):
        from goa2.domain.models.enums import CardState

        state = (
            EffectScenarioBuilder()
            .with_hexes(_disc_hexes(board_radius))
            .blue_hero("hero_enemy", at=(1, 0, -1), current_card=_attack_card())
            .red_hero("hero_emmitt", at=(0, 0, 0))
            .with_actor("hero_enemy")
            .build()
        )
        emmitt = state.get_hero("hero_emmitt")
        defense = hero_card("Emmitt", "unstable_timeline")
        defense.state = CardState.HAND
        emmitt.hand = [defense]
        _add_glitch_pool(state)
        return state

    def _corner_paint_state(self):
        from goa2.domain.models.enums import CardState

        # The three non-X token hexes are mutually spacing-valid, so the batch
        # precheck passes. X itself is individually legal but too close to all
        # remaining legal hexes, so choosing it first makes token 2 impossible.
        state = (
            EffectScenarioBuilder()
            .with_hexes(
                [
                    (0, 0, 0),
                    (1, 0, -1),
                    (-4, 2, 2),  # X
                    (-4, 0, 4),
                    (-4, 4, 0),
                    (-2, 1, 1),
                ]
            )
            .blue_hero("hero_enemy", at=(1, 0, -1), current_card=_attack_card())
            .red_hero("hero_emmitt", at=(0, 0, 0))
            .with_actor("hero_enemy")
            .build()
        )
        emmitt = state.get_hero("hero_emmitt")
        defense = hero_card("Emmitt", "unstable_timeline")
        defense.state = CardState.HAND
        emmitt.hand = [defense]
        _add_glitch_pool(state)
        return state

    def test_defense_places_three_swaps_then_blocks(self):
        """H2: defense text resolves before the combat total; 3 tokens; the
        attack then resolves vs defense 6 with Emmitt at his new location."""
        state = self._state()
        run = run_card(state, "hero_enemy")
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("hero_emmitt")
        run.expect_input("SELECT_CARD_OR_PASS")
        assert run.latest_request.player_id == "hero_emmitt"
        run.choose("unstable_timeline")

        for q, r, s in _TOKEN_SPOTS[:3]:  # defense mode: 3 tokens
            run.expect_input("SELECT_HEX")
            assert run.latest_request.player_id == "hero_emmitt"
            run.choose({"q": q, "r": r, "s": s})

        run.expect_input("SELECT_UNIT")  # Emmitt picks the chooser
        assert run.latest_request.player_id == "hero_emmitt"
        run.choose("hero_enemy")
        run.expect_input("SELECT_UNIT_OR_TOKEN")  # attacker picks the token
        assert run.latest_request.player_id == "hero_enemy"
        run.choose("glitch_1")  # at (3, 0, -3)
        run.finish()

        # Swap happened before combat; attack 5 vs defense 6 → blocked.
        assert state.get_position("hero_emmitt") == Hex(q=3, r=0, s=-3)
        assert state.get_position("glitch_1") == Hex(q=0, r=0, s=0)
        assert len(_glitch_on_board(state)) == 3
        cleanup = next(
            effect
            for effect in state.active_effects
            if effect.effect_type == EffectType.DELAYED_TRIGGER
        )
        assert cleanup.source_card_id == "unstable_timeline"
        attacker = state.get_hero("hero_enemy")
        emmitt = state.get_hero("hero_emmitt")
        assert attacker is not None and attacker.current_turn_card is not None
        assert emmitt is not None
        assert attacker.current_turn_card.is_active is False
        assert emmitt.discard_pile[0].is_active is True

    def test_defense_infeasible_still_blocks_with_value_six(self):
        """U1 (defense): text stops but the defense value 6 still counts."""
        state = self._state(board_radius=1)
        run = run_card(state, "hero_enemy")
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("hero_emmitt")
        run.expect_input("SELECT_CARD_OR_PASS").choose("unstable_timeline")
        run.finish()  # no placement prompts; combat resolves

        assert _glitch_on_board(state) == {}
        # Attack 5 < defense 6 → blocked; Emmitt survives in place.
        assert state.get_position("hero_emmitt") == Hex(q=0, r=0, s=0)

    def test_defense_placement_lookahead_excludes_dead_end(self):
        """A locally legal first token hex is hidden when it cannot complete
        the remaining batch."""
        state = self._corner_paint_state()
        run = run_card(state, "hero_enemy")
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("hero_emmitt")
        run.expect_input("SELECT_CARD_OR_PASS").choose("unstable_timeline")

        run.expect_input("SELECT_HEX")
        options = _option_hex_tuples(run.latest_request)
        assert (-4, 2, 2) not in options
        assert options == {(-4, 0, 4), (-4, 4, 0), (-2, 1, 1)}

        run.choose({"q": -4, "r": 0, "s": 4})
        run.expect_input("SELECT_HEX").choose({"q": -4, "r": 4, "s": 0})
        run.expect_input("SELECT_HEX").choose({"q": -2, "r": 1, "s": 1})
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_UNIT_OR_TOKEN").choose("glitch_1")
        run.finish()

        combat_events = [
            event for event in run.events if event.event_type == GameEventType.COMBAT_RESOLVED
        ]
        assert combat_events
        assert combat_events[-1].metadata["outcome"] == "BLOCKED"
        assert state.execution_context["block_succeeded"] is True
        assert state.current_actor_id == "hero_enemy"
        assert state.get_position("hero_emmitt") == Hex(q=-4, r=0, s=4)

    def test_defense_tokens_removed_at_end_of_enemy_turn(self):
        """H3: tokens placed during the enemy's turn are removed at the end
        of THAT turn."""
        from goa2.engine.handler import process_stack
        from goa2.engine.phases import end_turn

        state = self._state()
        run = run_card(state, "hero_enemy")
        run.expect_input("CHOOSE_ACTION").choose("ATTACK")
        run.expect_input("SELECT_UNIT").choose("hero_emmitt")
        run.expect_input("SELECT_CARD_OR_PASS").choose("unstable_timeline")
        for q, r, s in _TOKEN_SPOTS[:3]:
            run.expect_input("SELECT_HEX").choose({"q": q, "r": r, "s": s})
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_UNIT_OR_TOKEN").choose("glitch_1")
        run.finish()
        assert len(_glitch_on_board(state)) == 3

        state.unresolved_hero_ids = []
        end_turn(state)
        process_stack(state)
        assert _glitch_on_board(state) == {}


# =============================================================================
# TIME WALK (§6, range 3) / FAST FORWARD (§6, range 4)
# "Move an enemy hero in range, who remained in the same space since the last
#  turn, 2 spaces in a straight line."
# =============================================================================


def _snapshot_now(state):
    """Freeze current positions as the turn-boundary snapshot, so every unit
    currently on the board counts as 'remained in the same space'."""
    from goa2.engine.phases import record_position_snapshot

    record_position_snapshot(state)


def _self_immunity(hero_id: str) -> ActiveEffect:
    return ActiveEffect(
        id=f"immunity_{hero_id}",
        source_id=hero_id,
        effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
        scope=EffectScope(shape=Shape.POINT, origin_id=hero_id, affects=AffectsFilter.SELF),
        duration=DurationType.THIS_TURN,
        is_active=True,
        created_at_turn=1,
        created_at_round=1,
    )


def _placement_prevention(hero_id: str, blocks) -> ActiveEffect:
    return ActiveEffect(
        id=f"prevent_{hero_id}",
        source_id=hero_id,
        effect_type=EffectType.PLACEMENT_PREVENTION,
        scope=EffectScope(shape=Shape.POINT, origin_id=hero_id, affects=AffectsFilter.SELF),
        duration=DurationType.THIS_TURN,
        displacement_blocks=blocks,
        blocks_enemy_actors=True,
        blocks_friendly_actors=False,
        blocks_self=False,
        is_active=True,
        created_at_turn=1,
        created_at_round=1,
    )


def _walk_state(card_id: str = "time_walk", *, enemy_at=(1, 0, -1), board_radius: int = 4):
    """Emmitt at origin, one enemy hero that has remained in place."""
    state = (
        EffectScenarioBuilder()
        .with_hexes(_disc_hexes(board_radius))
        .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", card_id))
        .blue_hero("hero_enemy", at=enemy_at)
        .with_actor("hero_emmitt")
        .build()
    )
    _snapshot_now(state)
    return state


@pytest.mark.effect_flow
class TestTimeWalkFastForward:
    def test_moves_remained_enemy_exactly_two_in_a_straight_line(self):
        """H1: Emmitt picks the destination; the enemy lands 2 away."""
        state = _walk_state()

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_HEX").choose({"q": 3, "r": 0, "s": -3})
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=3, r=0, s=-3)

    def test_destination_options_are_exactly_two_away(self):
        """H1: distance-1 and distance-3 hexes are never offered."""
        state = _walk_state()

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_HEX")
        offered = _option_hex_tuples(run.latest_request)

        assert (3, 0, -3) in offered  # exactly 2 along the +q axis
        assert (2, 0, -2) not in offered  # 1 away
        assert (4, 0, -4) not in offered  # 3 away
        assert all(max(abs(q - 1), abs(r), abs(s + 1)) == 2 for q, r, s in offered)

    def test_enemy_that_moved_this_turn_is_not_a_valid_target(self):
        """U1: only the hero still standing on its snapshot hex is offered."""
        state = (
            EffectScenarioBuilder()
            .with_hexes(_disc_hexes(4))
            .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "time_walk"))
            .blue_hero("hero_mover", at=(1, 0, -1))
            .blue_hero("hero_stayer", at=(0, 1, -1))
            .with_actor("hero_emmitt")
            .build()
        )
        _snapshot_now(state)
        # The mover began the turn one hex over and has since walked away.
        state.last_turn_positions["hero_mover"] = Hex(q=2, r=0, s=-2)

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT")

        assert {o.id for o in run.latest_request.options} == {"hero_stayer"}

    def test_sole_target_that_moved_this_turn_aborts_the_action(self):
        """U1: with no remaining valid target the mandatory select aborts."""
        state = _walk_state()
        state.last_turn_positions["hero_enemy"] = Hex(q=2, r=0, s=-2)

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=1, r=0, s=-1)

    def test_enemy_with_no_legal_two_space_line_is_not_offered(self):
        """U3: a hero with no exactly-2 straight-line destination is an invalid
        target, not a target that aborts the action at the hex step."""
        state = (
            EffectScenarioBuilder()
            .with_hexes(
                [
                    (0, 0, 0),  # Emmitt
                    # Arm A: the trapped enemy. Its only on-board 2-space lines
                    # are (3,-3,0) -- blocked below -- and (-1,1,0), off-board.
                    (1, -1, 0),
                    (2, -2, 0),
                    (3, -3, 0),
                    # Arm B: a free enemy with a clear 2-space line to (0,3,-3).
                    (0, 1, -1),
                    (0, 2, -2),
                    (0, 3, -3),
                ]
            )
            .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "time_walk"))
            .blue_hero("hero_trapped", at=(1, -1, 0))
            .blue_hero("hero_free", at=(0, 1, -1))
            .blue_minion("blocker", at=(3, -3, 0))
            .with_actor("hero_emmitt")
            .build()
        )
        _snapshot_now(state)

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT")

        assert {o.id for o in run.latest_request.options} == {"hero_free"}

    def test_enemy_that_left_and_returned_still_counts_as_remained(self):
        """H2: the check is literal — current hex == snapshot hex."""
        state = _walk_state()
        state.place_entity("hero_enemy", Hex(q=2, r=0, s=-2))
        state.place_entity("hero_enemy", Hex(q=1, r=0, s=-1))

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_HEX").choose({"q": 3, "r": 0, "s": -3})
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=3, r=0, s=-3)

    def test_snapshot_reaches_into_the_previous_round(self):
        """H3: on turn 1 the 'last turn' is the last turn of the prior round."""
        from goa2.engine.handler import process_stack, push_steps
        from goa2.engine.steps import RoundResetStep

        state = _walk_state()
        state.last_turn_positions = {}
        push_steps(state, [RoundResetStep()])
        process_stack(state)
        assert state.round == 2

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_HEX").choose({"q": 3, "r": 0, "s": -3})
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=3, r=0, s=-3)

    def test_enemy_displaced_by_someone_else_is_not_a_valid_target(self):
        """U2: being pushed/swapped by another unit counts as having moved."""
        state = _walk_state()
        state.place_entity("hero_enemy", Hex(q=1, r=1, s=-2))  # shoved off its snapshot hex

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=1, r=1, s=-2)

    def test_respawned_enemy_with_no_snapshot_entry_is_not_offered(self):
        """U4/S2: no snapshot entry means 'remained' is false."""
        state = (
            EffectScenarioBuilder()
            .with_hexes(_disc_hexes(4))
            .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "time_walk"))
            .blue_hero("hero_respawned", at=(1, 0, -1))
            .blue_hero("hero_veteran", at=(0, 1, -1))
            .with_actor("hero_emmitt")
            .build()
        )
        _snapshot_now(state)
        del state.last_turn_positions["hero_respawned"]

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT")

        assert {o.id for o in run.latest_request.options} == {"hero_veteran"}

    def test_enemy_protected_from_forced_movement_is_not_offered(self):
        """U5: a Bulwark-style MOVE block makes the hero an invalid target."""
        from goa2.domain.models.effect import DisplacementType

        state = _walk_state()
        state.active_effects.append(
            _placement_prevention("hero_enemy", [DisplacementType.MOVE, DisplacementType.PLACE])
        )

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=1, r=0, s=-1)

    def test_enemy_protected_only_from_placement_can_still_be_shoved(self):
        """U5: Wasp-style protection blocks PLACE/SWAP, not a forced MOVE, so
        the hero stays a legal Time Walk target."""
        from goa2.domain.models.effect import DisplacementType

        state = _walk_state()
        state.active_effects.append(
            _placement_prevention("hero_enemy", [DisplacementType.PLACE, DisplacementType.SWAP])
        )

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_HEX").choose({"q": 3, "r": 0, "s": -3})
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=3, r=0, s=-3)

    def test_immune_enemy_is_not_offered(self):
        """U6."""
        state = _walk_state()
        state.active_effects.append(_self_immunity("hero_enemy"))

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=1, r=0, s=-1)

    def test_fast_forward_reaches_range_four(self):
        state = _walk_state("fast_forward", enemy_at=(4, 0, -4), board_radius=6)

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_HEX").choose({"q": 6, "r": 0, "s": -6})
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=6, r=0, s=-6)

    def test_time_walk_does_not_reach_range_four(self):
        state = _walk_state("time_walk", enemy_at=(4, 0, -4), board_radius=6)

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=4, r=0, s=-4)


# =============================================================================
# BACK TO THE FUTURE (§7, range 4) — "Choose one —
#  • Place a unit in range into the space where that unit was at the start of
#    this turn.
#  • Move an enemy hero in range, who remained in the same space since the last
#    turn, 2 spaces in a straight line."  (bullet B == Fast Forward)
# =============================================================================

_BTF_RECALL = 1
_BTF_SHOVE = 2


def _btf_state(**heroes):
    """Emmitt at the origin with Back to the Future in hand."""
    builder = (
        EffectScenarioBuilder()
        .with_hexes(_disc_hexes(4))
        .red_hero(
            "hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "back_to_the_future")
        )
    )
    return builder


@pytest.mark.effect_flow
class TestBackToTheFutureRecall:
    def _moved_enemy_state(self):
        state = (
            _btf_state().blue_hero("hero_enemy", at=(1, 1, -2)).with_actor("hero_emmitt").build()
        )
        _snapshot_now(state)
        state.place_entity("hero_enemy", Hex(q=2, r=0, s=-2))  # it moved this turn
        return state

    def test_recalls_moved_enemy_hero_to_its_turn_start_space(self):
        """H1."""
        state = self._moved_enemy_state()

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_RECALL)
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=1, r=1, s=-2)

    def test_recalls_friendly_hero(self):
        """H1: 'a unit' is not restricted to enemies."""
        state = _btf_state().red_hero("hero_ally", at=(1, 1, -2)).with_actor("hero_emmitt").build()
        _snapshot_now(state)
        state.place_entity("hero_ally", Hex(q=2, r=0, s=-2))

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_RECALL)
        run.expect_input("SELECT_UNIT").choose("hero_ally")
        run.finish()

        assert state.get_position("hero_ally") == Hex(q=1, r=1, s=-2)

    def test_recalls_minion(self):
        """H1."""
        state = (
            _btf_state().blue_minion("blue_minion", at=(1, 1, -2)).with_actor("hero_emmitt").build()
        )
        _snapshot_now(state)
        state.place_entity("blue_minion", Hex(q=2, r=0, s=-2))

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_RECALL)
        run.expect_input("SELECT_UNIT").choose("blue_minion")
        run.finish()

        assert state.get_position("blue_minion") == Hex(q=1, r=1, s=-2)

    def test_emmitt_cannot_recall_himself(self):
        """U2."""
        state = self._moved_enemy_state()

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_RECALL)
        run.expect_input("SELECT_UNIT")

        assert "hero_emmitt" not in {o.id for o in run.latest_request.options}

    def test_immune_unit_is_not_offered(self):
        """U3."""
        state = (
            _btf_state()
            .blue_hero("hero_immune", at=(1, 1, -2))
            .blue_hero("hero_plain", at=(0, 2, -2))
            .with_actor("hero_emmitt")
            .build()
        )
        _snapshot_now(state)
        state.place_entity("hero_immune", Hex(q=2, r=0, s=-2))
        state.place_entity("hero_plain", Hex(q=0, r=1, s=-1))
        state.active_effects.append(_self_immunity("hero_immune"))

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_RECALL)
        run.expect_input("SELECT_UNIT")

        assert {o.id for o in run.latest_request.options} == {"hero_plain"}

    def test_unit_spawned_this_turn_is_not_offered(self):
        """U4/S2: no snapshot entry means no defined start-of-turn space."""
        state = (
            _btf_state()
            .blue_hero("hero_spawned", at=(1, 1, -2))
            .blue_hero("hero_veteran", at=(0, 2, -2))
            .with_actor("hero_emmitt")
            .build()
        )
        _snapshot_now(state)
        del state.last_turn_positions["hero_spawned"]
        state.place_entity("hero_veteran", Hex(q=0, r=1, s=-1))

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_RECALL)
        run.expect_input("SELECT_UNIT")

        assert {o.id for o in run.latest_request.options} == {"hero_veteran"}

    def test_occupied_turn_start_space_is_selectable_but_aborts(self):
        """U1: the target stays selectable; the mandatory place then fails."""
        state = (
            _btf_state()
            .blue_hero("hero_enemy", at=(1, 1, -2))
            .blue_minion("squatter", at=(0, 1, -1))
            .with_actor("hero_emmitt")
            .build()
        )
        _snapshot_now(state)
        state.place_entity("hero_enemy", Hex(q=2, r=0, s=-2))
        state.place_entity("squatter", Hex(q=1, r=1, s=-2))  # sits on the snapshot hex

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_RECALL)
        run.expect_input("SELECT_UNIT")
        assert "hero_enemy" in {o.id for o in run.latest_request.options}
        run.choose("hero_enemy")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=2, r=0, s=-2)
        assert state.get_position("squatter") == Hex(q=1, r=1, s=-2)

    def test_unit_that_never_moved_occupies_its_own_turn_start_space(self):
        """U1: 'occupied by anyone, including the unit itself if it never
        moved' — selectable, but the place fails and the action aborts."""
        state = (
            _btf_state().blue_hero("hero_enemy", at=(1, 1, -2)).with_actor("hero_emmitt").build()
        )
        _snapshot_now(state)  # enemy never moved: snapshot hex == current hex

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_RECALL)
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=1, r=1, s=-2)


@pytest.mark.effect_flow
class TestBackToTheFutureShove:
    def test_bullet_b_moves_remained_enemy_two_spaces(self):
        """H2: bullet B is Fast Forward."""
        state = (
            _btf_state().blue_hero("hero_enemy", at=(1, 0, -1)).with_actor("hero_emmitt").build()
        )
        _snapshot_now(state)

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_SHOVE)
        run.expect_input("SELECT_UNIT").choose("hero_enemy")
        run.expect_input("SELECT_HEX").choose({"q": 3, "r": 0, "s": -3})
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=3, r=0, s=-3)

    def test_bullet_b_ignores_a_hero_that_moved_this_turn(self):
        """H2: bullet B keeps the 'remained in place' restriction."""
        state = (
            _btf_state().blue_hero("hero_enemy", at=(1, 0, -1)).with_actor("hero_emmitt").build()
        )
        _snapshot_now(state)
        state.last_turn_positions["hero_enemy"] = Hex(q=2, r=0, s=-2)

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_SHOVE)
        run.finish()

        assert state.get_position("hero_enemy") == Hex(q=1, r=0, s=-1)

    def test_bullet_b_does_not_target_minions(self):
        """H2: bullet B is 'an enemy hero', unlike bullet A's 'a unit'."""
        state = (
            _btf_state()
            .blue_minion("blue_minion", at=(1, 0, -1))
            .blue_hero("hero_enemy", at=(0, 1, -1))
            .with_actor("hero_emmitt")
            .build()
        )
        _snapshot_now(state)

        run = run_card(state, "hero_emmitt")
        run.expect_input("CHOOSE_ACTION").choose("SKILL")
        run.expect_input("SELECT_NUMBER").choose(_BTF_SHOVE)
        run.expect_input("SELECT_UNIT")

        assert {o.id for o in run.latest_request.options} == {"hero_enemy"}


# =============================================================================
# P2 primitive + Temporal Punch / Slam / Judgment (initiative-as-defense)
# =============================================================================

_TEMPORAL_ATTACK = 9  # temporal_punch primary_action_value


def _defense_card() -> Card:
    """A defense card with high Initiative and low Defense.

    The two values must differ so the block value reveals which stat was used.
    """
    return Card(
        id="def_card",
        name="Def Card",
        tier=CardTier.I,
        color=CardColor.GREEN,
        initiative=10,
        primary_action=ActionType.DEFENSE,
        primary_action_value=2,
        secondary_actions={},
        effect_id="",
        effect_text="",
        is_facedown=False,
    )


def _temporal_state(card_id: str = "temporal_punch"):
    """Emmitt attacks an adjacent enemy hero who holds one defense card."""
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (1, -1, 0), (0, 1, -1)])
        .red_hero("hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", card_id))
        .blue_hero("hero_enemy", at=(1, 0, -1))
        .with_actor("hero_emmitt")
        .build()
    )
    state.get_hero("hero_enemy").hand = [_defense_card()]
    return state


def _block_option(run):
    """The defender's reaction option for `def_card`."""
    assert run.latest_request is not None
    option = next(o for o in run.latest_request.options if o.id == "def_card")
    return option


def _drive_to_reaction(state):
    run = run_card(state, "hero_emmitt")
    run.expect_input("CHOOSE_ACTION").choose("ATTACK")
    run.expect_input("SELECT_UNIT").choose("hero_enemy")
    run.expect_input("SELECT_CARD_OR_PASS")
    return run


@pytest.mark.effect_flow
def test_temporal_punch_defender_blocks_with_initiative_not_defense() -> None:
    """H2: the block value is the card's Initiative (10), not its Defense (2)."""
    state = _temporal_state()

    run = _drive_to_reaction(state)
    option = _block_option(run)

    assert option.metadata["defense_value"] == 10


_TOKEN_HEX = Hex(q=2, r=0, s=-2)  # adjacent to the defender at (1,0,-1)


def _place_token_aura(
    state,
    *,
    token_id: str,
    token_type: TokenType,
    at: Hex,
    source_id: str,
    affects: AffectsFilter,
    stat_type: StatType,
    stat_value: int,
) -> None:
    """Place a token and hang an adjacent stat aura on it.

    Mirrors how Tali's Ice (INITIATIVE -1, enemy heroes) and Trinkets' Barrier
    (DEFENSE +1, self and friendly heroes) are built by their effects: a
    PASSIVE AREA_STAT_MODIFIER whose scope origin is the token.
    """
    token = Token(id=token_id, name=token_id, token_type=token_type)
    state.register_entity(token)
    state.place_entity(token_id, at)
    state.active_effects.append(
        ActiveEffect(
            id=f"{token_id}_aura",
            source_id=source_id,
            effect_type=EffectType.AREA_STAT_MODIFIER,
            scope=EffectScope(shape=Shape.ADJACENT, origin_id=token_id, affects=affects),
            duration=DurationType.PASSIVE,
            stat_type=stat_type,
            stat_value=stat_value,
            is_active=True,
            created_at_turn=1,
            created_at_round=1,
        )
    )


def _add_ice_token(state, at: Hex = _TOKEN_HEX) -> None:
    """Tali's Ice: -1 INITIATIVE to adjacent enemy heroes."""
    _place_token_aura(
        state,
        token_id="ice_1",
        token_type=TokenType.ICE,
        at=at,
        source_id="hero_emmitt",  # RED, so hero_enemy (BLUE) is an enemy hero
        affects=AffectsFilter.ENEMY_HEROES,
        stat_type=StatType.INITIATIVE,
        stat_value=-1,
    )


def _add_barrier_token(state, at: Hex = _TOKEN_HEX) -> None:
    """Trinkets' Barrier: +1 DEFENSE to itself and adjacent friendly heroes."""
    _place_token_aura(
        state,
        token_id="barrier_1",
        token_type=TokenType.BARRIER,
        at=at,
        source_id="hero_enemy",  # BLUE, so the defender is covered by SELF_AND_FRIENDLY
        affects=AffectsFilter.SELF_AND_FRIENDLY_HEROES,
        stat_type=StatType.DEFENSE,
        stat_value=1,
    )


@pytest.mark.effect_flow
def test_temporal_punch_initiative_aura_changes_the_block_value() -> None:
    """H4: an Initiative aura (Tali's Ice) lowers what the defender can block with.

    Base Initiative 10, one adjacent Ice token -> 9.
    """
    state = _temporal_state()
    _add_ice_token(state)

    run = _drive_to_reaction(state)

    assert _block_option(run).metadata["defense_value"] == 9


@pytest.mark.effect_flow
def test_defense_aura_helps_against_a_normal_attack() -> None:
    """Control for the Barrier test below: the aura really is wired up and adjacent.

    Under an ordinary attack the defender blocks with Defense 2 + Barrier 1 = 3.
    Without this control, the Barrier test could pass simply because the token
    was misplaced and contributed nothing to either stat.
    """
    state = _temporal_state(card_id="time_walk")  # any non-Temporal card
    _add_barrier_token(state)

    # Drive a plain attack rather than the Temporal effect.
    push_steps(state, [AttackSequenceStep(damage=_TEMPORAL_ATTACK, range_val=1)])
    result = process_stack(state)

    assert result.input_request is not None
    assert result.input_request.request_type.value == "SELECT_UNIT"
    state.execution_stack[-1].pending_input = {"selection": "hero_enemy"}
    result = process_stack(state)

    assert result.input_request.request_type.value == "SELECT_CARD_OR_PASS"
    option = next(o for o in result.input_request.options if o.id == "def_card")
    assert option.metadata["defense_value"] == 3


@pytest.mark.effect_flow
def test_temporal_punch_defense_aura_does_not_help() -> None:
    """A Defense aura (Trinkets' Barrier) is irrelevant when blocking with Initiative.

    The same Barrier that grants +1 in the control above contributes nothing
    here: the block value stays at the defender's bare Initiative of 10.
    """
    state = _temporal_state()
    _add_barrier_token(state)

    run = _drive_to_reaction(state)

    assert _block_option(run).metadata["defense_value"] == 10


@pytest.mark.effect_flow
def test_temporal_punch_reaction_option_is_labelled_with_initiative() -> None:
    """H5: the defending client sees the value it would actually block with."""
    state = _temporal_state()

    run = _drive_to_reaction(state)

    assert _block_option(run).text == "Def Card (Init: 10)"


@pytest.mark.effect_flow
def test_temporal_punch_minion_defense_modifiers_still_apply() -> None:
    """H3: "Minion defense modifiers are applied as normal."

    A friendly melee minion adjacent to the defender still lowers the attack
    they need to survive. The modifier is subtracted from the attack value
    rather than added to the defender's stat, so the Initiative swap must not
    disturb it.
    """
    state = _temporal_state()
    state.teams[TeamColor.BLUE].minions.append(
        Minion(id="blue_guard", name="Blue Guard", team=TeamColor.BLUE, type=MinionType.MELEE)
    )
    state.place_entity("blue_guard", Hex(q=1, r=-1, s=0))

    run = _drive_to_reaction(state)

    assert run.latest_request.context["minion_modifier"] == 1
    assert run.latest_request.context["defense_needed"] == _TEMPORAL_ATTACK - 1
    # ...and the block value is still Initiative, not Defense.
    assert _block_option(run).metadata["defense_value"] == 10


@pytest.mark.effect_flow
def test_initiative_as_defense_does_not_leak_into_a_later_attack() -> None:
    """U4 / S5: the flag is scoped to the attack sequence that set it.

    Emmitt's Temporal Punch is blocked (Initiative 10 >= attack 9), leaving the
    defender alive. A second, ordinary attack in the same turn must fall back
    to Defense (2) -- the execution context survives the first attack, so a
    flag that is only ever set to True would leak here.
    """
    second_card = _defense_card()
    second_card.id = "def_card_2"

    state = _temporal_state()
    state.get_hero("hero_enemy").hand = [_defense_card(), second_card]

    run = _drive_to_reaction(state)
    assert _block_option(run).metadata["defense_value"] == 10
    run.choose("def_card").finish()

    assert "hero_enemy" in state.entity_locations, "defender should have blocked and survived"

    push_steps(state, [AttackSequenceStep(damage=_TEMPORAL_ATTACK, range_val=1)])
    result = process_stack(state)
    assert result.input_request.request_type.value == "SELECT_UNIT"
    state.execution_stack[-1].pending_input = {"selection": "hero_enemy"}
    result = process_stack(state)

    assert result.input_request.request_type.value == "SELECT_CARD_OR_PASS"
    option = next(o for o in result.input_request.options if o.id == "def_card_2")
    assert option.metadata["defense_value"] == 2, "second attack must use Defense"


@pytest.mark.effect_flow
def test_temporal_punch_high_defense_low_initiative_card_fails_to_block() -> None:
    """H2 outcome: the swap decides the combat, not just the displayed number.

    Defense 12 would comfortably block the attack of 9; Initiative 5 does not.
    """
    state = _temporal_state()
    tank_card = _defense_card()
    tank_card.id = "tank_card"
    tank_card.initiative = 5
    tank_card.primary_action_value = 12
    state.get_hero("hero_enemy").hand = [tank_card]

    run = _drive_to_reaction(state)
    assert (
        next(o for o in run.latest_request.options if o.id == "tank_card").metadata["defense_value"]
        == 5
    )

    run.choose("tank_card").finish()

    defeated = [e for e in run.events if e.event_type == GameEventType.UNIT_DEFEATED]
    assert [e.target_id for e in defeated] == ["hero_enemy"]


@pytest.mark.effect_flow
@pytest.mark.parametrize(
    ("card_id", "attack"),
    [("temporal_punch", 9), ("temporal_slam", 11), ("temporal_judgment", 12)],
)
def test_all_three_temporal_tiers_use_initiative_as_defense(card_id: str, attack: int) -> None:
    """One effect, three cards: only the attack value differs."""
    state = _temporal_state(card_id=card_id)

    run = _drive_to_reaction(state)

    assert run.latest_request.context["attack_value"] == attack
    assert _block_option(run).metadata["defense_value"] == 10


# =============================================================================
# Card-granted "+N Defense" bonuses vs initiative-as-defense
# =============================================================================

_JUDGMENT_ATTACK = 12  # temporal_judgment primary_action_value


def _dodger_defense_state():
    """Emmitt attacks Dodger, who holds Shield of Decay (+2 Defense).

    Two empty spawn points sit within the card's radius 2 of the defender, so
    the bonus clause is satisfied and really does fire.
    """
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (2, 0, -2), (1, -1, 0), (0, 1, -1), (2, -1, -1)])
        .spawn_point((2, 0, -2))
        .spawn_point((1, -1, 0))
        .red_hero(
            "hero_emmitt", at=(0, 0, 0), current_card=hero_card("Emmitt", "temporal_judgment")
        )
        .blue_hero("hero_dodger", at=(1, 0, -1))
        .with_actor("hero_emmitt")
        .build()
    )
    state.battle_zones = {"lane1": "z1"}
    shield = hero_card("Dodger", "shield_of_decay")
    shield.is_facedown = False
    state.get_hero("hero_dodger").hand = [shield]
    return state


def _defender_is_alive(state) -> bool:
    return BoardEntityID("hero_dodger") in state.entity_locations


@pytest.mark.effect_flow
def test_defense_bonus_does_not_apply_when_blocking_with_initiative() -> None:
    """Shield of Decay's "+2 Defense" must not raise an Initiative block.

    Temporal Judgment attacks for 12. Shield of Decay blocks with Initiative 10.
    The card's +2 is a Defense bonus, so it contributes nothing here: 10 < 12
    and the defender falls. If the bonus leaked in, 10 + 2 = 12 would block.
    """
    state = _dodger_defense_state()

    run = run_card(state, "hero_emmitt")
    run.expect_input("CHOOSE_ACTION").choose("ATTACK")
    run.expect_input("SELECT_UNIT").choose("hero_dodger")
    run.expect_input("SELECT_CARD_OR_PASS").choose("shield_of_decay")
    run.finish()

    assert not _defender_is_alive(state)


@pytest.mark.effect_flow
def test_defense_bonus_still_applies_when_blocking_with_defense() -> None:
    """Control: under an ordinary attack the same +2 is load-bearing.

    Shield of Decay blocks with Defense 3. Against a 5-damage attack the bare
    card fails and only the +2 saves the defender, so this test fails if the
    bonus is dropped for every attack rather than only initiative ones.
    """
    state = _dodger_defense_state()

    push_steps(state, [AttackSequenceStep(damage=5, range_val=1)])
    result = process_stack(state)
    assert result.input_request.request_type.value == "SELECT_UNIT"
    state.execution_stack[-1].pending_input = {"selection": "hero_dodger"}
    result = process_stack(state)

    assert result.input_request.request_type.value == "SELECT_CARD_OR_PASS"
    state.execution_stack[-1].pending_input = {"selection": "shield_of_decay"}
    process_stack(state)

    assert _defender_is_alive(state)
