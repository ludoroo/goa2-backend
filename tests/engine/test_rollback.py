"""Tests for Action Resolution Rollback & Confirmation."""

import pytest

from goa2.domain.board import Board
from goa2.domain.hex import Hex
from goa2.domain.input import InputResponse
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardTier,
    Team,
    TeamColor,
)
from goa2.domain.models.enums import StepType, TargetType
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.filters import TeamFilter
from goa2.engine.handler import (
    process_stack,
    push_steps,
)
from goa2.engine.phases import start_resolution_phase
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.steps import (
    AskConfirmationStep,
    ConfirmResolutionStep,
    FinalizeHeroTurnStep,
    RevealHandCardStep,
    SelectStep,
)


def _make_card(card_id, initiative, action=ActionType.SKILL):
    return Card(
        id=card_id,
        name=f"Card {card_id}",
        tier=CardTier.I,
        color=CardColor.RED,
        initiative=initiative,
        primary_action=action,
        primary_action_value=None,
        secondary_actions={ActionType.HOLD: 0},
        effect_id="e",
        effect_text="t",
        is_facedown=False,
    )


def _filler_cards():
    return [
        Card(
            id=f"filler_{i}",
            name=f"Filler {i}",
            tier=CardTier.I,
            color=CardColor.RED,
            initiative=1,
            primary_action=ActionType.SKILL,
            primary_action_value=None,
            effect_id="e",
            effect_text="t",
        )
        for i in range(3)
    ]


def _make_state():
    """Two-hero state: hero_a (RED, init 20), hero_b (BLUE, init 10)."""
    hero_a = Hero(id=HeroID("hero_a"), name="A", team=TeamColor.RED, deck=[], hand=_filler_cards())
    hero_b = Hero(id=HeroID("hero_b"), name="B", team=TeamColor.BLUE, deck=[], hand=_filler_cards())
    board = Board()
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[hero_a], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[hero_b], minions=[]),
        },
    )
    state.place_entity("hero_a", Hex(q=0, r=0, s=0))
    state.place_entity("hero_b", Hex(q=2, r=0, s=-2))
    return state


def _setup_resolution(state):
    """Set up cards and start resolution phase."""
    state.get_hero("hero_a").current_turn_card = _make_card("card_a", 20)
    state.get_hero("hero_b").current_turn_card = _make_card("card_b", 10)
    state.unresolved_hero_ids = ["hero_a", "hero_b"]
    start_resolution_phase(state)


def _respond(session: GameSession, result, selection: object):
    assert result.input_request is not None
    return session.advance(InputResponse(request_id=result.input_request.id, selection=selection))


# ---- ConfirmResolutionStep basic behavior ----


class TestConfirmResolutionStep:
    def test_prompts_confirm_only(self):
        """Rollback uses the dedicated endpoint, not a misleading input option."""
        step = ConfirmResolutionStep(hero_id="hero_a")
        state = _make_state()
        state.current_actor_id = "hero_a"
        result = step.resolve(state, {})
        assert result.requires_input
        req = result.input_request
        assert req.player_id == "hero_a"
        option_ids = [o.id for o in req.options]
        assert option_ids == ["CONFIRM"]

    def test_auto_skips_when_rollback_frozen(self):
        """Confirm step auto-confirms when rollback is frozen."""
        step = ConfirmResolutionStep(hero_id="hero_a")
        state = _make_state()
        state.current_actor_id = "hero_a"
        result = step.resolve(state, {"rollback_frozen": True})
        assert result.is_finished
        assert not result.requires_input

    def test_confirm_input_finishes(self):
        """Submitting CONFIRM finishes the step."""
        step = ConfirmResolutionStep(hero_id="hero_a")
        step.pending_input = {"selection": "CONFIRM"}
        state = _make_state()
        state.current_actor_id = "hero_a"
        result = step.resolve(state, {})
        assert result.is_finished

    def test_non_confirm_input_is_rejected_and_re_requested(self):
        step = ConfirmResolutionStep(hero_id="hero_a")
        step.pending_input = {"selection": "ROLLBACK"}
        state = _make_state()
        state.current_actor_id = "hero_a"

        result = step.resolve(state, {})

        assert result.requires_input
        assert step.pending_input is None


# ---- Rollback disabled tracking ----


class TestRollbackSegmentBoundary:
    def test_card_tier_reveal_reanchors_at_next_owner_prompt(self):
        """A public hand-card reveal is a boundary, not a permanent freeze.

        The actor may reconsider a later decision without restoring the
        hidden card or crossing back to the pre-reveal prompt.
        """
        state = _make_state()
        state.current_actor_id = "hero_a"
        state.resolution_owner_id = HeroID("hero_a")
        state.execution_context["revealed_owner"] = "hero_b"
        state.execution_context["revealed_card"] = "filler_0"
        session = GameSession(state)

        push_steps(
            state,
            [
                AskConfirmationStep(player_id="hero_a", prompt="Pre-Reveal"),
                RevealHandCardStep(
                    owner_key="revealed_owner",
                    card_key="revealed_card",
                ),
                AskConfirmationStep(player_id="hero_a", prompt="Post-Reveal"),
                ConfirmResolutionStep(hero_id="hero_a"),
            ],
        )

        pre_reveal = session.advance()
        assert pre_reveal.input_request.prompt == "Pre-Reveal"
        pre_reveal_snapshot = session._rollback_snapshot
        assert pre_reveal_snapshot is not None

        post_reveal = _respond(session, pre_reveal, "YES")
        assert post_reveal.input_request.prompt == "Post-Reveal"
        assert post_reveal.input_request.can_rollback is True
        assert state.execution_context.get("rollback_frozen") is not True
        assert state.execution_context.get("rollback_reanchor_pending") is not True
        assert session._rollback_snapshot is not pre_reveal_snapshot
        assert state.card_reveal is not None
        assert state.card_reveal["card_id"] == "filler_0"

        confirm = _respond(session, post_reveal, "YES")
        assert confirm.input_request.can_rollback is True

        rolled_back = session.rollback()
        assert rolled_back.input_request.prompt == "Post-Reveal"
        assert rolled_back.input_request.can_rollback is True
        assert session.state.card_reveal is not None
        assert session.state.card_reveal["card_id"] == "filler_0"

    def test_foreign_input_clears_snapshot_without_freezing(self):
        """Foreign input drops the pre-foreign snapshot but does NOT set
        ``rollback_frozen`` — the resolution is a segment boundary, not
        permanently contaminated."""
        state = _make_state()
        state.current_actor_id = "hero_a"
        session = GameSession(state)

        push_steps(
            state,
            [
                AskConfirmationStep(player_id="hero_a", prompt="Continue?"),
                AskConfirmationStep(player_id="hero_b", prompt="Block?"),
            ],
        )

        res1 = session.advance()
        assert res1.input_request.player_id == "hero_a"
        assert session._rollback_snapshot is not None
        assert res1.input_request.can_rollback is True

        res2 = _respond(session, res1, "YES")
        assert res2.input_request.player_id == "hero_b"
        assert res2.input_request.can_rollback is False
        assert session._rollback_snapshot is None
        assert state.execution_context.get("rollback_frozen") is not True

    def test_owner_actionable_prompt_after_foreign_reanchors_and_rollback_returns_to_it(self):
        """A post-foreign owner actionable prompt re-anchors a fresh snapshot;
        rollback returns to that prompt and preserves the foreign player's
        committed side-effects."""
        state = _make_state()
        state.current_actor_id = "hero_a"
        session = GameSession(state)

        push_steps(
            state,
            [
                AskConfirmationStep(player_id="hero_a", prompt="Pre-Foreign"),
                AskConfirmationStep(player_id="hero_b", prompt="Foreign", output_key="foreign_ok"),
                AskConfirmationStep(player_id="hero_a", prompt="Post-Foreign"),
            ],
        )

        res1 = session.advance()
        assert res1.input_request.prompt == "Pre-Foreign"
        assert res1.input_request.can_rollback is True
        pre_foreign_snapshot = session._rollback_snapshot
        assert pre_foreign_snapshot is not None

        res2 = _respond(session, res1, "YES")
        assert res2.input_request.prompt == "Foreign"
        assert res2.input_request.can_rollback is False
        assert session._rollback_snapshot is None
        assert state.execution_context.get("rollback_frozen") is not True

        res3 = _respond(session, res2, "YES")
        assert res3.input_request.player_id == "hero_a"
        assert res3.input_request.prompt == "Post-Foreign"
        assert res3.input_request.can_rollback is True
        assert session._rollback_snapshot is not None
        assert session._rollback_snapshot is not pre_foreign_snapshot

        # Rollback returns to Post-Foreign, never past the foreign segment.
        res_rb = session.rollback()
        assert res_rb.input_request.prompt == "Post-Foreign"
        assert res_rb.input_request.can_rollback is True
        assert session.state.execution_context.get("foreign_ok") is True

    def test_same_player_input_does_not_clear_snapshot(self):
        """When a step prompts the current actor, the rollback snapshot is retained."""
        state = _make_state()
        state.current_actor_id = "hero_a"
        session = GameSession(state)

        own_step1 = AskConfirmationStep(player_id="hero_a", prompt="Continue 1?")
        own_step2 = AskConfirmationStep(player_id="hero_a", prompt="Continue 2?")
        push_steps(state, [own_step1, own_step2])

        res1 = session.advance()
        assert res1.input_request.player_id == "hero_a"
        assert session._rollback_snapshot is not None
        assert res1.input_request.can_rollback is True

        res2 = _respond(session, res1, "YES")
        assert res2.input_request.player_id == "hero_a"
        assert session._rollback_snapshot is not None
        assert res2.input_request.can_rollback is True


# ---- GameSession rollback ----


class TestSessionRollback:
    def test_rollback_raises_when_no_snapshot(self):
        """rollback() raises ValueError when there's no snapshot."""
        state = _make_state()
        session = GameSession(state)
        with pytest.raises(ValueError, match="No rollback snapshot"):
            session.rollback()

    def test_basic_rollback_flow(self):
        """Start resolution -> choose action -> rollback -> back to action choice."""
        state = _make_state()
        session = GameSession(state)
        _setup_resolution(state)

        # Process stack to get first action choice
        result = session.advance()
        assert result.result_type == SessionResultType.INPUT_NEEDED
        assert result.input_request is not None
        assert result.input_request.player_id == "hero_a"
        assert result.input_request.can_rollback is True
        # Snapshot should be taken
        assert session._rollback_snapshot is not None

        # Choose HOLD
        result2 = _respond(session, result, "HOLD")
        # Should be at ConfirmResolutionStep
        assert result2.result_type == SessionResultType.INPUT_NEEDED
        assert result2.input_request.can_rollback is True

        # Rollback
        result3 = session.rollback()
        assert result3.result_type == SessionResultType.INPUT_NEEDED
        assert result3.input_request is not None
        # Back to action choice
        assert result3.input_request.player_id == "hero_a"
        assert result3.input_request.can_rollback is True

    def test_multiple_rollbacks(self):
        """Rollback, choose differently, rollback again."""
        state = _make_state()
        session = GameSession(state)
        _setup_resolution(state)

        # Get first action choice
        result = session.advance()
        assert result.input_request.player_id == "hero_a"

        # Choose HOLD
        _respond(session, result, "HOLD")

        # Rollback
        r = session.rollback()
        assert r.input_request.player_id == "hero_a"

        # Choose HOLD again
        _respond(session, r, "HOLD")

        # Rollback again
        r2 = session.rollback()
        assert r2.input_request.player_id == "hero_a"

    def test_snapshot_cleared_after_turn(self):
        """After confirm -> finalize, snapshot is cleared."""
        state = _make_state()
        session = GameSession(state)
        _setup_resolution(state)

        # hero_a's action choice
        result = session.advance()
        assert session._rollback_snapshot is not None

        # Choose HOLD
        result = _respond(session, result, "HOLD")

        # Confirm
        result = _respond(session, result, "CONFIRM")

        # Now hero_b acts, hero_a's snapshot should be cleared and new one for hero_b
        if result.input_request and result.input_request.player_id == "hero_b":
            # Snapshot is now for hero_b
            assert session._rollback_snapshot is not None

    def test_can_rollback_false_for_other_players(self):
        """Input requests targeting non-actor players don't have can_rollback."""
        state = _make_state()
        state.current_actor_id = "hero_a"
        session = GameSession(state)
        session._rollback_snapshot = state.model_dump(mode="json")

        # Push a step that targets hero_b
        step = AskConfirmationStep(player_id="hero_b", prompt="Block?")
        push_steps(state, [step])
        result = session.advance()
        assert result.input_request is not None
        assert result.input_request.player_id == "hero_b"
        assert result.input_request.can_rollback is False

    def test_rollback_frozen_does_not_create_stale_snapshot(self):
        """Frozen rollback prompts must not become rollback targets later."""
        state = _make_state()
        state.current_actor_id = "hero_b"
        state.execution_context["rollback_frozen"] = True
        session = GameSession(state)

        # Simulates prompting hero_b
        push_steps(state, [AskConfirmationStep(player_id="hero_b", prompt="Action prompt?")])
        result = session.advance()
        assert result.input_request is not None
        assert result.input_request.player_id == "hero_b"
        assert result.input_request.can_rollback is False
        assert session._rollback_snapshot is None
        assert session._rollback_actor_id is None

        # Later hero_b becomes the actor with rollback unfrozen.
        state.execution_context.clear()
        push_steps(state, [AskConfirmationStep(player_id="hero_b", prompt="Hero B turn")])
        result = session.advance()
        assert result.input_request is not None
        assert result.input_request.player_id == "hero_b"
        assert result.input_request.can_rollback is True

        rollback = session.rollback()
        assert rollback.input_request is not None
        assert rollback.input_request.prompt == "Hero B turn"


# ---- Abort then rollback ----


class TestAbortThenRollback:
    def test_abort_clears_to_confirm_step(self):
        """Mandatory step failure aborts to ConfirmResolutionStep, not FinalizeHeroTurnStep."""
        state = _make_state()
        state.current_actor_id = "hero_a"

        # Use a mandatory select with filters that find no valid targets
        # TeamFilter(relation="ENEMY") requires enemies in range, but with
        # RangeFilter we can ensure none are found
        mandatory_select = SelectStep(
            target_type=TargetType.UNIT,
            prompt="Pick enemy",
            is_mandatory=True,
            filters=[
                TeamFilter(relation="ENEMY"),
                # hero_b is at distance 2 but range 0 means nothing in range
                {"type": "range_filter", "max_range": 0},
            ],
        )
        push_steps(
            state,
            [
                mandatory_select,
                ConfirmResolutionStep(hero_id="hero_a"),
                FinalizeHeroTurnStep(hero_id="hero_a"),
            ],
        )

        # Process: mandatory select fails (no valid targets), aborts to ConfirmResolutionStep
        stack_result = process_stack(state)

        # Should land on ConfirmResolutionStep
        assert stack_result.input_request is not None
        assert len(state.execution_stack) >= 1
        # The top of stack should be ConfirmResolutionStep
        top_step = state.execution_stack[-1]
        assert isinstance(top_step, ConfirmResolutionStep)


# ---- can_rollback flag in full flow ----


class TestCanRollbackFlag:
    def test_can_rollback_on_action_choice(self):
        """can_rollback is True on the initial action choice for the current actor."""
        state = _make_state()
        session = GameSession(state)
        _setup_resolution(state)

        result = session.advance()
        assert result.input_request is not None
        assert result.input_request.can_rollback is True

    def test_can_rollback_on_confirm_step(self):
        """can_rollback is True on the confirm step."""
        state = _make_state()
        session = GameSession(state)
        _setup_resolution(state)

        # Action choice
        initial = session.advance()
        # Choose HOLD
        result = _respond(session, initial, "HOLD")
        # Confirm step
        assert result.input_request is not None
        assert result.input_request.can_rollback is True


# ---- Per-actor rollback isolation ----


class TestRollbackPerActorIsolation:
    def test_rollback_does_not_restore_previous_actors_snapshot(self):
        """Rollback for player B should restore to B's turn start, not A's."""
        state = _make_state()
        session = GameSession(state)
        _setup_resolution(state)

        # hero_a's action choice (highest initiative goes first)
        r1 = session.advance()
        assert r1.input_request.player_id == "hero_a"
        assert r1.input_request.can_rollback is True
        snapshot_a = session._rollback_snapshot

        # hero_a chooses HOLD
        hold_result = _respond(session, r1, "HOLD")

        # hero_a confirms
        r_confirm = _respond(session, hold_result, "CONFIRM")

        # Now it's hero_b's turn
        assert r_confirm.input_request is not None
        assert r_confirm.input_request.player_id == "hero_b"
        assert r_confirm.input_request.can_rollback is True

        # Snapshot should have been replaced for hero_b
        assert session._rollback_actor_id == "hero_b"
        snapshot_b = session._rollback_snapshot
        assert snapshot_b is not snapshot_a

        # hero_b chooses HOLD
        _respond(session, r_confirm, "HOLD")

        # hero_b rolls back
        r_rollback = session.rollback()
        assert r_rollback.input_request is not None
        assert r_rollback.input_request.player_id == "hero_b"

        # The restored state should have hero_b as current actor, not hero_a
        assert session.state.current_actor_id == "hero_b"


# ---- Rollback during Hanu's ultimate action control ----


def _control_state():
    """blue_enemy is the actor resolving card_e, controlled by hero_hanu.

    Mirrors Hanu's ultimate: a CONTROL_NEXT_ACTION effect reroutes the
    controlled hero's inputs to Hanu, who confirms or rolls back the action.
    """
    from goa2.domain.models.effect import DurationType, EffectScope, EffectType, Shape
    from goa2.engine.effect_manager import EffectManager

    hero_hanu = Hero(
        id=HeroID("hero_hanu"), name="Hanu", team=TeamColor.RED, deck=[], hand=_filler_cards()
    )
    blue_enemy = Hero(
        id=HeroID("blue_enemy"), name="E", team=TeamColor.BLUE, deck=[], hand=_filler_cards()
    )
    board = Board()
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[hero_hanu], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[blue_enemy], minions=[]),
        },
    )
    state.place_entity("hero_hanu", Hex(q=0, r=0, s=0))
    state.place_entity("blue_enemy", Hex(q=2, r=0, s=-2))

    blue_enemy.current_turn_card = _make_card("card_e", 10)
    state.unresolved_hero_ids = ["blue_enemy"]
    start_resolution_phase(state)

    EffectManager.create_effect(
        state=state,
        source_id="hero_hanu",
        effect_type=EffectType.CONTROL_NEXT_ACTION,
        scope=EffectScope(shape=Shape.POINT, origin_id="blue_enemy"),
        duration=DurationType.THIS_ROUND,
        is_active=True,
        controlled_card_id="card_e",
    )
    return state


class TestRollbackDuringControl:
    def test_controller_gets_rollback_snapshot_and_flag(self):
        """During control, the remapped controller (Hanu) can roll back the
        controlled action even though the actor is the controlled hero."""
        state = _control_state()
        session = GameSession(state)

        result = session.advance()
        assert result.result_type == SessionResultType.INPUT_NEEDED
        assert result.input_request is not None
        # Input is remapped to the controller.
        assert result.input_request.player_id == "hero_hanu"
        assert result.input_request.context.get("controlled_hero_id") == "blue_enemy"
        # The controlled action must be rollback-able by the controller.
        assert result.input_request.can_rollback is True
        assert session._rollback_snapshot is not None

    def test_controller_can_actually_rollback(self):
        """rollback() restores the controlled action's start state."""
        state = _control_state()
        session = GameSession(state)

        result1 = session.advance()
        # Controller chooses HOLD for the controlled hero.
        result2 = _respond(session, result1, "HOLD")
        assert result2.input_request is not None
        assert result2.input_request.can_rollback is True

        result3 = session.rollback()
        assert result3.result_type == SessionResultType.INPUT_NEEDED
        assert result3.input_request is not None
        assert result3.input_request.player_id == "hero_hanu"
        assert result3.input_request.can_rollback is True


# ---- Snapshot board exclusion & persistence ----


def _loc(state, entity_id):
    """Look up an entity's hex regardless of key type coercion."""
    for k, v in state.entity_locations.items():
        if str(k) == entity_id:
            return v
    return None


class TestRollbackSnapshotBoardExclusion:
    def test_snapshot_excludes_board(self):
        """The rollback snapshot omits the static board to stay small."""
        state = _make_state()
        session = GameSession(state)
        _setup_resolution(state)

        result = session.advance()
        assert result.input_request.can_rollback is True
        assert session._rollback_snapshot is not None
        assert "board" not in session._rollback_snapshot

    def test_rollback_restores_positions_without_snapshotting_board(self):
        """Rolling back restores unit positions even though the board is excluded."""
        state = _make_state()
        session = GameSession(state)
        _setup_resolution(state)

        session.advance()  # snapshot taken at turn start; hero_a at (0,0,0)

        # Move hero_a after the snapshot
        session.state.place_entity("hero_a", Hex(q=1, r=-1, s=0))
        assert _loc(session.state, "hero_a") == Hex(q=1, r=-1, s=0)

        session.rollback()
        assert _loc(session.state, "hero_a") == Hex(q=0, r=0, s=0)


class TestRollbackSnapshotPersistence:
    @staticmethod
    def _save_and_load(tmp_path, session, *, game_id, snapshot, actor_id):
        """Persist ``(session.state, snapshot, actor_id)`` and reload."""
        from goa2.engine.persistence import load_game, save_game

        path = save_game(
            game_id=game_id,
            state=session.state,
            player_tokens={},
            spectator_token="s",
            hero_to_token={},
            created_at=0.0,
            save_dir=str(tmp_path),
            rollback_snapshot=snapshot,
            rollback_actor_id=actor_id,
        )
        return load_game(str(path))

    def test_snapshot_and_can_rollback_survive_save_load(self, tmp_path):
        """A mid-action rollback snapshot survives a save/reload cycle."""
        state = _make_state()
        session = GameSession(state)
        _setup_resolution(state)

        result = session.advance()  # action choice; snapshot taken
        assert result.input_request.can_rollback is True
        assert session._rollback_snapshot is not None

        data = self._save_and_load(
            tmp_path,
            session,
            game_id="g1",
            snapshot=session._rollback_snapshot,
            actor_id=session._rollback_actor_id,
        )
        restored = data["session"]

        assert restored._rollback_snapshot is not None
        assert restored._rollback_actor_id == "hero_a"
        assert data["last_result"].input_request.can_rollback is True

        rb = restored.rollback()
        assert rb.input_request.player_id == "hero_a"

    def test_frozen_resolution_never_advertises_can_rollback_after_reload(self, tmp_path):
        """A saved frozen resolution reloads without advertising rollback,
        even when a stale snapshot survives in the payload."""
        state = _make_state()
        state.current_actor_id = "hero_a"
        state.resolution_owner_id = HeroID("hero_a")
        state.execution_context["rollback_frozen"] = True
        push_steps(state, [AskConfirmationStep(player_id="hero_a", prompt="Owner Post-Freeze")])
        session = GameSession(state)

        data = self._save_and_load(
            tmp_path,
            session,
            game_id="frozen_persist",
            snapshot=session._make_snapshot(),  # stale non-null snapshot
            actor_id="hero_a",
        )
        restored = data["session"]

        assert restored.state.execution_context.get("rollback_frozen") is True
        assert data["last_result"].input_request.player_id == "hero_a"
        assert data["last_result"].input_request.can_rollback is False
        with pytest.raises(ValueError, match="No rollback snapshot"):
            restored.rollback()

    def test_hanu_controlled_request_preserves_can_rollback_after_reload(self, tmp_path):
        """A Hanu-remapped controlled request (``player_id == controller``,
        ``context.controlled_hero_id == controlled_actor``) still advertises
        rollback after save/load with a valid snapshot."""
        state = _control_state()
        session = GameSession(state)

        result = session.advance()
        assert result.input_request.player_id == "hero_hanu"
        assert result.input_request.context.get("controlled_hero_id") == "blue_enemy"
        assert result.input_request.can_rollback is True

        data = self._save_and_load(
            tmp_path,
            session,
            game_id="hanu_persist",
            snapshot=session._rollback_snapshot,
            actor_id=session._rollback_actor_id,
        )
        restored = data["session"]

        assert restored._rollback_actor_id == "blue_enemy"
        req = data["last_result"].input_request
        assert req.player_id == "hero_hanu"
        assert req.context.get("controlled_hero_id") == "blue_enemy"
        assert req.can_rollback is True

        rb = restored.rollback()
        assert rb.input_request.player_id == "hero_hanu"
        assert rb.input_request.context.get("controlled_hero_id") == "blue_enemy"
        assert rb.input_request.can_rollback is True

    def test_stale_actor_snapshot_is_rejected_after_reload(self, tmp_path):
        """A persisted snapshot whose ``rollback_actor_id`` belongs to a
        prior owner must not be usable in the new owner's resolution."""
        # hero_a's prior-turn snapshot (hero_a at (0,0,0)).
        prior_state = _make_state()
        prior_state.current_actor_id = "hero_a"
        prior_state.resolution_owner_id = HeroID("hero_a")
        stale_snapshot = GameSession(prior_state)._make_snapshot()

        # Current: hero_b owns; hero_a moved so non-restoration is provable.
        state = _make_state()
        state.current_actor_id = "hero_b"
        state.resolution_owner_id = HeroID("hero_b")
        current_hex = Hex(q=1, r=-1, s=0)
        state.place_entity("hero_a", current_hex)
        push_steps(state, [AskConfirmationStep(player_id="hero_b", prompt="Owner Prompt")])
        session = GameSession(state)

        data = self._save_and_load(
            tmp_path,
            session,
            game_id="stale_actor_persist",
            snapshot=stale_snapshot,
            actor_id="hero_a",  # mismatched with current owner hero_b
        )
        restored = data["session"]

        assert str(restored.state.resolution_owner_id) == "hero_b"
        req = data["last_result"].input_request
        assert req.player_id == "hero_b"
        assert req.can_rollback is False

        with pytest.raises(ValueError, match="No rollback snapshot"):
            restored.rollback()
        assert _loc(restored.state, "hero_a") == current_hex

    def test_reanchor_pending_snapshot_is_rejected_after_reload(self, tmp_path):
        """Save with ``rollback_reanchor_pending=True`` + a matching-owner
        snapshot: reload must not advertise ``can_rollback`` and direct
        rollback rejects, scrubbing the stale pair."""
        state = _make_state()
        state.current_actor_id = "hero_a"
        state.resolution_owner_id = HeroID("hero_a")
        pre_boundary_snapshot = GameSession(state)._make_snapshot()

        # Simulate the in-step boundary having fired.
        state.execution_context["rollback_reanchor_pending"] = True
        push_steps(state, [AskConfirmationStep(player_id="hero_a", prompt="Post-Boundary")])
        session = GameSession(state)

        data = self._save_and_load(
            tmp_path,
            session,
            game_id="reanchor_pending_persist",
            snapshot=pre_boundary_snapshot,
            actor_id="hero_a",  # matches current owner
        )
        restored = data["session"]

        assert restored.state.execution_context.get("rollback_reanchor_pending") is True
        req = data["last_result"].input_request
        assert req.player_id == "hero_a"
        assert req.can_rollback is False

        with pytest.raises(ValueError, match="No rollback snapshot"):
            restored.rollback()
        assert restored._rollback_snapshot is None
        assert restored._rollback_actor_id is None


# ---- StepType registration ----


class TestStepTypeRegistration:
    def test_confirm_resolution_step_type(self):
        """ConfirmResolutionStep has the correct StepType."""
        step = ConfirmResolutionStep(hero_id="hero_a")
        assert step.type == StepType.CONFIRM_RESOLUTION

    def test_serialization_roundtrip(self):
        """ConfirmResolutionStep can be serialized and deserialized."""
        step = ConfirmResolutionStep(hero_id="hero_a")
        data = step.model_dump(mode="json")
        assert data["type"] == "confirm_resolution"
        assert data["hero_id"] == "hero_a"

        restored = ConfirmResolutionStep.model_validate(data)
        assert restored.hero_id == "hero_a"
        assert restored.type == StepType.CONFIRM_RESOLUTION


# ---- Scenario C and Mine Blast checks ----


class TestScenarioCAndMineBlast:
    @staticmethod
    def _place_mine(state, mine_id, token_type, hex, owner_id="hero_b"):
        from goa2.domain.models import Token
        from goa2.domain.types import BoardEntityID

        mine = Token(
            id=BoardEntityID(mine_id),
            name="Mine",
            token_type=token_type,
            owner_id=owner_id,
            is_passable=True,
            is_facedown=True,
        )
        state.token_pool.setdefault(token_type, []).append(mine)
        state.misc_entities[BoardEntityID(mine_id)] = mine
        state.place_entity(BoardEntityID(mine_id), hex)
        return BoardEntityID(mine_id)

    @staticmethod
    def _push_mine_trigger(state, mine_id, *followup_steps, victim_id="hero_a"):
        """Push ``[TriggerMineStep, *followup_steps]`` with the context keys
        TriggerMineStep expects; the trigger fires first."""
        from goa2.engine.steps import TriggerMineStep

        state.execution_context["triggered_mine_ids"] = [mine_id]
        state.execution_context["mine_victim_id"] = victim_id
        push_steps(state, [TriggerMineStep(), *followup_steps])

    @classmethod
    def _make_mine_state(cls, token_type, mine_hex=None):
        """Return ``(state, session, mine_id, pre_snapshot)`` with hero_a as
        owner and an established pre-mine rollback anchor."""
        if mine_hex is None:
            mine_hex = Hex(q=1, r=0, s=-1)
        state = _make_state()
        state.current_actor_id = "hero_a"
        state.resolution_owner_id = HeroID("hero_a")
        mine_id = cls._place_mine(state, "mine_1", token_type, mine_hex)
        session = GameSession(state)
        push_steps(state, [AskConfirmationStep(player_id="hero_a", prompt="Pre-Mine")])
        res = session.advance()
        assert res.input_request.can_rollback is True
        pre_snapshot = session._rollback_snapshot
        _respond(session, res, "YES")
        return state, session, mine_id, pre_snapshot

    def test_mine_blast_forced_discard_reanchors_after_boundary(self):
        """Blast reveal is a boundary; the auto-spawned ForceDiscardStep is
        owner-addressed and re-anchors past the boundary."""
        from goa2.domain.models import TokenType

        state, session, _, _ = self._make_mine_state(TokenType.MINE_BLAST)
        self._push_mine_trigger(state, "mine_1")

        res = session.advance()
        assert "select a card to discard" in res.input_request.prompt
        assert state.execution_context.get("rollback_frozen") is not True
        assert res.input_request.can_rollback is True
        assert session._rollback_snapshot is not None
        assert state.execution_context.get("rollback_reanchor_pending") is not True

    def test_mine_dud_reveal_is_boundary_and_reanchors_at_target_selection(self):
        """Dud reveal + attack target select over two adjacent enemies →
        target select re-anchors; rollback returns there while preserving
        the mover's position and mine removal."""
        from goa2.domain.models import TokenType
        from goa2.domain.models.unit import Hero

        post_move_hex = Hex(q=1, r=-1, s=0)
        state, session, mine_id, pre_snap = self._make_mine_state(TokenType.MINE_DUD, post_move_hex)
        # Second adjacent enemy so target selection has two picks.
        state.teams[TeamColor.BLUE].heroes.append(
            Hero(
                id=HeroID("hero_c"),
                name="C",
                team=TeamColor.BLUE,
                deck=[],
                hand=_filler_cards(),
            )
        )
        state.place_entity("hero_a", post_move_hex)
        state.place_entity("hero_b", Hex(q=2, r=-1, s=-1))
        state.place_entity("hero_c", Hex(q=1, r=0, s=-1))

        self._push_mine_trigger(
            state,
            "mine_1",
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Pick target",
                is_mandatory=True,
                output_key="attack_target",
                filters=[
                    TeamFilter(relation="ENEMY"),
                    {"type": "range_filter", "max_range": 1},
                ],
            ),
            ConfirmResolutionStep(hero_id="hero_a"),
        )

        res = session.advance()
        assert res.input_request["type"] == "SELECT_UNIT"
        assert res.input_request.can_rollback is True
        assert session._rollback_snapshot is not pre_snap
        assert state.execution_context.get("rollback_frozen") is not True
        assert mine_id not in state.entity_locations
        assert {"hero_b", "hero_c"} <= {opt.id for opt in res.input_request.options}

        res_confirm = _respond(session, res, "hero_b")
        assert res_confirm.input_request.can_rollback is True

        res_rb = session.rollback()
        assert res_rb.input_request["type"] == "SELECT_UNIT"
        assert res_rb.input_request.can_rollback is True
        assert _loc(session.state, "hero_a") == post_move_hex
        assert mine_id not in session.state.entity_locations

    def test_mine_dud_reveal_alone_auto_completes_confirm(self):
        """Mine reveal + confirm alone → confirm auto-completes silently."""
        from goa2.domain.models import TokenType

        state, session, _, _ = self._make_mine_state(TokenType.MINE_DUD)
        self._push_mine_trigger(state, "mine_1", ConfirmResolutionStep(hero_id="hero_a"))

        res = session.advance()
        assert res.input_request is None
        assert res.result_type == SessionResultType.ACTION_COMPLETE
        assert state.execution_context.get("rollback_frozen") is not True
        with pytest.raises(ValueError, match="No rollback snapshot"):
            session.rollback()


# ---- Concrete attack: defender PASS freezes attacker's rollback ----


def _attack_state():
    """Attacker hero_a adjacent to defender hero_b; hero_a is the acting owner."""
    state = _make_state()
    state.current_actor_id = "hero_a"
    # hero_a already at (0,0,0), hero_b at (2,0,-2). Move hero_b adjacent for range=1.
    state.place_entity("hero_b", Hex(q=1, r=0, s=-1))
    return state


class TestAttackReactionRollback:
    """Attacker → foreign defense → ... rollback contract."""

    @staticmethod
    def _drive_through_defense(session, defender_response):
        """Attacker picks hero_b as target; defender responds. Returns the
        SessionResult on the prompt that lands next (or ACTION_COMPLETE)."""
        res = session.advance()
        assert res.input_request["type"] == "SELECT_UNIT"
        assert res.input_request.can_rollback is True
        assert session._rollback_snapshot is not None
        pre_attack_snap = session._rollback_snapshot

        res = _respond(session, res, "hero_b")
        assert res.input_request["type"] == "SELECT_CARD_OR_PASS"
        assert res.input_request.player_id == "hero_b"
        assert res.input_request.can_rollback is False
        assert session._rollback_snapshot is None
        assert session.state.execution_context.get("rollback_frozen") is not True

        return _respond(session, res, defender_response), pre_attack_snap

    @pytest.mark.parametrize("defender_response", ["PASS"])
    def test_confirm_only_after_defense_auto_completes(self, defender_response):
        """Attack → foreign defense → ConfirmResolutionStep alone: no
        actionable re-anchor → confirm auto-completes silently."""
        from goa2.engine.steps import AttackSequenceStep

        state = _attack_state()
        state.resolution_owner_id = HeroID("hero_a")
        session = GameSession(state)
        push_steps(
            state,
            [
                AttackSequenceStep(damage=3, range_val=1, is_ranged=False),
                ConfirmResolutionStep(hero_id="hero_a"),
            ],
        )

        res, _ = self._drive_through_defense(session, defender_response)
        assert res.input_request is None
        assert res.result_type == SessionResultType.ACTION_COMPLETE
        assert session._rollback_snapshot is None
        assert state.execution_context.get("rollback_frozen") is not True
        assert not any(
            getattr(step, "type", None) == StepType.CONFIRM_RESOLUTION
            for step in state.execution_stack
        )
        with pytest.raises(ValueError, match="No rollback snapshot"):
            session.rollback()

    @pytest.mark.parametrize("defender_response", ["PASS"])
    def test_post_attack_actionable_prompt_reanchors_and_confirm_retains_rollback(
        self, defender_response
    ):
        """Attack → foreign defense → post-attack actionable owner prompt →
        confirm. Actionable prompt re-anchors; rollback from confirm returns
        there, never past the defender's committed decision."""
        from goa2.engine.steps import AttackSequenceStep

        state = _attack_state()
        state.resolution_owner_id = HeroID("hero_a")
        session = GameSession(state)
        push_steps(
            state,
            [
                AttackSequenceStep(damage=3, range_val=1, is_ranged=False),
                AskConfirmationStep(player_id="hero_a", prompt="Attack again?"),
                ConfirmResolutionStep(hero_id="hero_a"),
            ],
        )

        res, pre_attack_snap = self._drive_through_defense(session, defender_response)
        assert res.input_request.player_id == "hero_a"
        assert res.input_request.prompt == "Attack again?"
        assert res.input_request.can_rollback is True
        assert session._rollback_snapshot is not None
        assert session._rollback_snapshot is not pre_attack_snap

        res_confirm = _respond(session, res, "YES")
        assert res_confirm.input_request.player_id == "hero_a"
        assert res_confirm.input_request.can_rollback is True

        res_rb = session.rollback()
        assert res_rb.input_request.prompt == "Attack again?"
        assert res_rb.input_request.can_rollback is True
