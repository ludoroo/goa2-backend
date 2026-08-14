"""Effects bind to the card whose text created them.

The binding happens at build time, in the CardEffect API, because that is the
only place that reliably knows which card is being performed: a re-performance
(Bullet Time, Reload, Mind Grip) resolves another card's text while the turn
context still names the granting card.
"""

from __future__ import annotations

import pytest

from goa2.domain.board import Board
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardState,
    CardTier,
    Hero,
    Team,
    TeamColor,
)
from goa2.domain.models.effect import DurationType, EffectScope, EffectType, Shape
from goa2.domain.state import GameState
from goa2.engine.effects import CardEffect, bind_effect_cards, register_effect
from goa2.engine.steps import (
    CreateEffectStep,
    MayRepeatOnceStep,
    PerformPrimaryActionStep,
)


def _card(card_id: str) -> Card:
    return Card(
        id=card_id,
        name=card_id.replace("_", " ").title(),
        tier=CardTier.I,
        color=CardColor.RED,
        initiative=5,
        primary_action=ActionType.SKILL,
        primary_action_value=None,
        effect_id=card_id,
        effect_text="",
        is_facedown=False,
    )


@pytest.fixture
def game_state() -> GameState:
    state = GameState(
        board=Board(),
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
        turn=1,
        round=1,
    )
    hero = Hero(id="hero_1", name="Hero 1", team=TeamColor.RED, deck=[])
    state.teams[TeamColor.RED].heroes.append(hero)
    state.current_actor_id = "hero_1"
    return state


def _effect_step(**overrides) -> CreateEffectStep:
    params: dict = dict(
        effect_type=EffectType.MOVEMENT_ZONE,
        scope=EffectScope(shape=Shape.ADJACENT),
        duration=DurationType.THIS_TURN,
    )
    params.update(overrides)
    return CreateEffectStep(**params)


class TestBindEffectCards:
    def test_binds_a_top_level_effect_step(self):
        step = _effect_step()

        bind_effect_cards([step], "card_1")

        assert step.source_card_id == "card_1"

    def test_binds_effect_steps_nested_in_a_repeat_template(self):
        step = MayRepeatOnceStep(steps_template=[_effect_step()])

        bind_effect_cards([step], "card_1")

        # Nested steps are isolated copies (see GameStep._isolate_nested_steps),
        # so assert through the container rather than the source object.
        assert step.steps_template[0].source_card_id == "card_1"

    def test_binds_effect_steps_nested_in_finishing_steps(self):
        step = _effect_step(finishing_steps=[_effect_step()])

        bind_effect_cards([step], "card_1")

        assert step.finishing_steps[0].source_card_id == "card_1"

    def test_leaves_an_explicit_card_alone(self):
        step = _effect_step(source_card_id="other_card")

        bind_effect_cards([step], "card_1")

        assert step.source_card_id == "other_card"

    def test_leaves_token_effects_unbound(self):
        step = _effect_step(is_token_effect=True)

        bind_effect_cards([step], "card_1")

        assert step.source_card_id is None


class _EffectCreatingEffect(CardEffect):
    def build_steps(self, state, hero, card, stats):
        return [_effect_step()]

    def build_defense_steps(self, state, defender, card, stats, context):
        return [_effect_step()]

    def build_on_block_steps(self, state, defender, card, stats, context):
        return [_effect_step()]


class TestCardEffectEntryPoints:
    def test_get_steps_binds_the_performed_card(self, game_state):
        hero = game_state.get_hero("hero_1")
        card = _card("card_1")

        steps = _EffectCreatingEffect().get_steps(game_state, hero, card)

        assert steps[0].source_card_id == "card_1"

    def test_get_steps_with_stats_binds_the_performed_card(self, game_state):
        from goa2.engine.stats import compute_card_stats

        hero = game_state.get_hero("hero_1")
        card = _card("card_1")
        stats = compute_card_stats(game_state, hero.id, card)

        steps = _EffectCreatingEffect().get_steps_with_stats(game_state, hero, card, stats)

        assert steps[0].source_card_id == "card_1"

    def test_get_defense_steps_binds_the_defense_card(self, game_state):
        hero = game_state.get_hero("hero_1")
        card = _card("defense_card")

        steps = _EffectCreatingEffect().get_defense_steps(game_state, hero, card, {})

        assert steps is not None
        assert steps[0].source_card_id == "defense_card"

    def test_get_on_block_steps_binds_the_defense_card(self, game_state):
        hero = game_state.get_hero("hero_1")
        card = _card("defense_card")

        steps = _EffectCreatingEffect().get_on_block_steps(game_state, hero, card, {})

        assert steps[0].source_card_id == "defense_card"


@register_effect("performed_card")
class _PerformedCardEffect(_EffectCreatingEffect):
    pass


class TestReperformance:
    """Re-performing another card's action attributes its effects to that card.

    Regression: the effect used to bind to whatever card the turn context named
    — the card that granted the re-performance — so a repeat both misattributed
    the effect and escaped the one-instance-per-card rule keyed on the card.
    """

    def test_performing_another_card_binds_effects_to_that_card(self, game_state):
        hero = game_state.get_hero("hero_1")
        granting = _card("granting_card")
        performed = _card("performed_card")
        hero.current_turn_card = granting
        hero.played_cards.append(performed)

        step = PerformPrimaryActionStep(hero_id="hero_1")
        context = {"current_card_id": granting.id, "selected_card": performed.id}
        result = step.resolve(game_state, context)

        created = [s for s in result.new_steps if isinstance(s, CreateEffectStep)]
        assert created
        assert created[0].source_card_id == "performed_card"

    def test_effects_from_an_already_resolved_card_start_active(self, game_state):
        """Dormancy models "played but not yet resolved".

        Effects wait for FinalizeHeroTurnStep to activate them, but that hook
        only ever fires for the hero's current turn card. An effect bound to a
        card that has already resolved would therefore stay dormant forever, so
        it must start in force.
        """
        hero = game_state.get_hero("hero_1")
        performed = _card("performed_card")
        performed.state = CardState.RESOLVED
        hero.current_turn_card = _card("granting_card")
        hero.played_cards.append(performed)

        step = PerformPrimaryActionStep(hero_id="hero_1")
        context = {"current_card_id": "granting_card", "selected_card": performed.id}
        create_step = next(
            s
            for s in step.resolve(game_state, context).new_steps
            if isinstance(s, CreateEffectStep)
        )
        create_step.resolve(game_state, context)

        effect = game_state.active_effects[0]
        assert effect.source_card_id == "performed_card"
        assert effect.is_active is True

    def test_effects_from_the_unresolved_current_card_stay_dormant(self, game_state):
        hero = game_state.get_hero("hero_1")
        current = _card("performed_card")
        current.state = CardState.UNRESOLVED
        hero.current_turn_card = current

        step = PerformPrimaryActionStep(hero_id="hero_1")
        context = {"current_card_id": current.id, "selected_card": current.id}
        create_step = next(
            s
            for s in step.resolve(game_state, context).new_steps
            if isinstance(s, CreateEffectStep)
        )
        create_step.resolve(game_state, context)

        assert game_state.active_effects[0].is_active is False


def test_engine_never_calls_build_steps_directly():
    """build_steps() skips the card binding, so only the CardEffect API may call it.

    Every engine caller must go through get_steps()/get_steps_with_stats(), which
    stamp the performed card onto the effect steps. A new direct call site would
    silently reintroduce misattributed effects.
    """
    import ast
    import pathlib

    engine_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "goa2" / "engine"
    allowed = {engine_dir / "effects.py"}

    offenders = []
    for path in engine_dir.rglob("*.py"):
        if path in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"build_steps", "build_defense_steps", "build_on_block_steps"}
            ):
                offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == []
