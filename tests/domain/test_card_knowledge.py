"""Behavioral contract for conservative, public card knowledge."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from pydantic import ValidationError

from goa2.data.heroes.registry import HeroRegistry
from goa2.domain.models import CardState, CardTier, StatType
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.domain.views import build_view
from goa2.engine.phases import commit_card, start_revelation_phase
from goa2.engine.setup import GameSetup
from goa2.engine.steps.cards import DiscardCardStep, RetrieveCardStep, RevealHandCardStep
from goa2.engine.steps.phases import EndPhaseCleanupStep
from goa2.engine.steps.selection import RevealAndResolveGuessStep

MAP = "src/goa2/data/maps/forgotten_island.json"


@pytest.fixture(scope="module")
def knowledge_api():
    """Keep a missing module a collected RED assertion, not an import error."""
    try:
        return importlib.import_module("goa2.domain.card_knowledge")
    except ModuleNotFoundError:

        class MissingKnowledgeAPI:
            def __getattr__(self, name: str):
                pytest.fail(f"missing public card-knowledge API: {name}")

        return MissingKnowledgeAPI()


def game(red: list[str], blue: list[str]) -> GameState:
    return GameSetup.create_game(MAP, red, blue, seed=11)


def hero_knowledge(result: Any, hero_id: str) -> Any:
    return result.heroes[hero_id]


def hypothesis_pairs(hero_result: Any) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        (tuple(h.active_card_ids), tuple(h.item_card_ids)) for h in hero_result.loadout_hypotheses
    }


def test_static_registry_defines_canonical_starting_cards(knowledge_api) -> None:
    """Every production definition can produce canonical public starting IDs."""
    for name in HeroRegistry.list_heroes():
        definition = HeroRegistry.get(name)
        assert definition is not None
        expected = {
            str(card.id)
            for card in definition.deck
            if card.tier in {CardTier.UNTIERED, CardTier.I} and not card.starts_in_deck
        }
        opponent = "Wasp" if name != "Wasp" else "Arien"
        result = knowledge_api.build_public_card_knowledge(game([name], [opponent]), None)
        assert set(hero_knowledge(result, str(definition.id)).starting_card_ids) == expected

    result = knowledge_api.build_public_card_knowledge(game(["Takahide"], ["Wasp"]), None)
    takahide = set(hero_knowledge(result, "hero_takahide").starting_card_ids)
    assert "float_like_a_butterfly" in takahide
    assert "sting_like_a_bee" not in takahide
    assert "strike_like_a_tiger" not in takahide


def test_commitment_identity_is_owner_only_and_hidden_change_invariant(knowledge_api) -> None:
    state = game(["Arien", "Wasp"], ["Brogan"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    first, second = arien.hand[:2]
    commit_card(state, arien.id, first)

    owner = knowledge_api.build_public_card_knowledge(state, "hero_arien")
    ally = knowledge_api.build_public_card_knowledge(state, "hero_wasp")
    enemy = knowledge_api.build_public_card_knowledge(state, "hero_brogan")
    spectator = knowledge_api.build_public_card_knowledge(state, None)
    assert hero_knowledge(owner, "hero_arien").committed_card_ids == (str(first.id),)
    assert hero_knowledge(ally, "hero_arien").committed_card_ids is None
    assert hero_knowledge(enemy, "hero_arien").committed_card_ids is None
    assert hero_knowledge(spectator, "hero_arien").committed_card_ids is None

    before_ally = ally
    before_enemy = enemy
    before_spectator = spectator
    arien.unplay_card(first)
    state.pending_inputs[arien.id] = second
    arien.play_card(second)
    assert knowledge_api.build_public_card_knowledge(state, "hero_wasp") == before_ally
    assert knowledge_api.build_public_card_knowledge(state, "hero_brogan") == before_enemy
    assert knowledge_api.build_public_card_knowledge(state, None) == before_spectator


def test_normal_revelation_is_permanent_persistent_and_legacy_safe(knowledge_api) -> None:
    state = game(["Arien"], ["Wasp"])
    arien = state.get_hero(HeroID("hero_arien"))
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert arien is not None and wasp is not None
    arien_card, wasp_card = arien.hand[0], wasp.hand[0]
    arien.play_card(arien_card)
    wasp.play_card(wasp_card)
    state.pending_inputs = {arien.id: arien_card, wasp.id: wasp_card}

    start_revelation_phase(state)
    assert ("hero_arien", str(arien_card.id)) in {
        (r.hero_id, r.card_id) for r in state.public_revealed_cards
    }
    restored = GameState.model_validate(state.model_dump(mode="json"))
    assert restored.public_revealed_cards == state.public_revealed_cards
    legacy = state.model_dump(mode="json", exclude={"public_revealed_cards"})
    assert GameState.model_validate(legacy).public_revealed_cards == ()

    public = knowledge_api.build_public_card_knowledge(state, None)
    assert str(arien_card.id) in hero_knowledge(public, "hero_arien").revealed_card_ids

    arien.retrieve_cards()
    EndPhaseCleanupStep()._retrieve_cards(state)
    public_after = knowledge_api.build_public_card_knowledge(state, None)
    assert str(arien_card.id) in hero_knowledge(public_after, "hero_arien").revealed_card_ids


def test_level_two_item_aggregate_keeps_both_arien_defense_hypotheses(
    knowledge_api,
) -> None:
    state = game(["Wasp"], ["Arien"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    arien.level = 2
    arien.items = {StatType.DEFENSE: 1}

    result = hero_knowledge(
        knowledge_api.build_public_card_knowledge(state, "hero_wasp"), "hero_arien"
    )
    assert result.status == knowledge_api.CardKnowledgeStatus.INFERRED
    assert hypothesis_pairs(result) == {
        (("magical_current",), ("arcane_whirlpool",)),
        (("raging_stream",), ("rogue_wave",)),
    }
    assert result.active_upgraded_card_ids is None


def test_owner_gets_exact_current_upgrades_not_hypotheses(knowledge_api) -> None:
    state = game(["Arien"], ["Wasp"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    active = next(c for c in arien.deck if str(c.id) == "magical_current")
    item = next(c for c in arien.deck if str(c.id) == "arcane_whirlpool")
    active.state = CardState.HAND
    item.state = CardState.ITEM
    arien.hand.append(active)
    arien.level = 2
    arien.items = {StatType.DEFENSE: 1}

    result = hero_knowledge(
        knowledge_api.build_public_card_knowledge(state, "hero_arien"), "hero_arien"
    )
    assert result.status == knowledge_api.CardKnowledgeStatus.EXACT
    assert result.active_upgraded_card_ids == ("magical_current",)
    assert result.loadout_hypotheses == ()


def test_impossible_item_aggregate_is_explicitly_inconsistent(knowledge_api) -> None:
    state = game(["Wasp"], ["Arien"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    arien.level = 2
    arien.items = {StatType.DEFENSE: 2}
    result = hero_knowledge(
        knowledge_api.build_public_card_knowledge(state, "hero_wasp"), "hero_arien"
    )
    assert result.status == knowledge_api.CardKnowledgeStatus.INCONSISTENT
    assert result.loadout_hypotheses == ()


def test_level_eight_caps_at_six_ordinary_upgrades_and_remains_inferable(
    knowledge_api,
) -> None:
    state = game(["Wasp"], ["Arien"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    arien.level = 8
    arien.items = {
        StatType.DEFENSE: 2,
        StatType.ATTACK: 1,
        StatType.RADIUS: 1,
        StatType.RANGE: 1,
        StatType.MOVEMENT: 1,
    }

    result = hero_knowledge(
        knowledge_api.build_public_card_knowledge(state, "hero_wasp"), "hero_arien"
    )

    assert result.status == knowledge_api.CardKnowledgeStatus.INFERRED
    assert result.loadout_hypotheses
    assert all(len(h.item_card_ids) == 6 for h in result.loadout_hypotheses)


def test_public_upgraded_reveals_filter_and_can_contradict_hypotheses(knowledge_api) -> None:
    state = game(["Wasp"], ["Arien"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    arien.level = 2
    arien.items = {StatType.DEFENSE: 1}

    state.record_public_revealed_card(arien.id, "magical_current")
    narrowed = hero_knowledge(
        knowledge_api.build_public_card_knowledge(state, "hero_wasp"), "hero_arien"
    )
    assert narrowed.status == knowledge_api.CardKnowledgeStatus.INFERRED
    assert hypothesis_pairs(narrowed) == {
        (("magical_current",), ("arcane_whirlpool",)),
    }

    state.record_public_revealed_card(arien.id, "raging_stream")
    contradicted = hero_knowledge(
        knowledge_api.build_public_card_knowledge(state, "hero_wasp"), "hero_arien"
    )
    assert contradicted.status == knowledge_api.CardKnowledgeStatus.INCONSISTENT
    assert contradicted.loadout_hypotheses == ()


def test_static_loadout_hypotheses_allow_optional_public_filters(knowledge_api) -> None:
    definition = HeroRegistry.get("Arien")
    assert definition is not None

    unfiltered = knowledge_api.enumerate_static_loadout_hypotheses(definition, 2)
    defense = knowledge_api.enumerate_static_loadout_hypotheses(
        definition, 2, public_items={StatType.DEFENSE: 1}
    )
    revealed = knowledge_api.enumerate_static_loadout_hypotheses(
        definition, 2, revealed_card_ids=("magical_current",)
    )

    assert unfiltered is not None and len(unfiltered) == 6
    assert {(h.active_card_ids, h.item_card_ids) for h in defense or ()} == {
        (("magical_current",), ("arcane_whirlpool",)),
        (("raging_stream",), ("rogue_wave",)),
    }
    assert revealed is not None
    assert {h.active_card_ids for h in revealed} == {("magical_current",)}


def test_faceup_discard_records_reveal_permanently_without_view_leak(knowledge_api) -> None:
    state = game(["Arien"], ["Wasp"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    card = arien.hand[0]
    card.is_facedown = False

    DiscardCardStep(card_id=str(card.id), hero_id=str(arien.id)).resolve(
        state, state.execution_context
    )
    assert (str(arien.id), str(card.id)) in {
        (record.hero_id, record.card_id) for record in state.public_revealed_cards
    }

    state.current_actor_id = arien.id
    state.execution_context["retrieved"] = str(card.id)
    RetrieveCardStep(card_key="retrieved").resolve(state, state.execution_context)
    public = knowledge_api.build_public_card_knowledge(state, None)
    assert str(card.id) in hero_knowledge(public, str(arien.id)).revealed_card_ids

    for viewer in (None, HeroID("hero_arien"), HeroID("hero_wasp")):
        assert "public_revealed_cards" not in build_view(state, for_hero_id=viewer)


def test_facedown_commit_discard_is_public_but_failed_discard_is_not() -> None:
    state = game(["Arien"], ["Wasp"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    committed = arien.hand[0]
    commit_card(state, arien.id, committed)
    assert committed.is_facedown is True

    DiscardCardStep(card_id=str(committed.id), hero_id=str(arien.id)).resolve(
        state, state.execution_context
    )

    assert committed.state is CardState.DISCARD
    assert (str(arien.id), str(committed.id)) in {
        (record.hero_id, record.card_id) for record in state.public_revealed_cards
    }

    before = state.public_revealed_cards
    DiscardCardStep(card_id="missing", hero_id=str(arien.id)).resolve(
        state, state.execution_context
    )
    assert state.public_revealed_cards == before


@pytest.mark.parametrize("name", ["Min", "Dodger", "Snorri"])
def test_nonstandard_item_heroes_fail_closed(knowledge_api, name: str) -> None:
    state = game(["Wasp"], [name])
    target = state.get_hero(HeroID(f"hero_{name.lower()}"))
    assert target is not None
    target.level = 2
    result = hero_knowledge(
        knowledge_api.build_public_card_knowledge(state, "hero_wasp"), str(target.id)
    )
    assert result.status == knowledge_api.CardKnowledgeStatus.UNAVAILABLE
    assert result.active_upgraded_card_ids is None
    assert result.loadout_hypotheses == ()


def test_direct_hand_reveal_remains_public_after_temporary_fields_clear(knowledge_api) -> None:
    state = game(["Wasp"], ["Arien"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    card = arien.hand[0]
    context = {"owner": str(arien.id), "card": str(card.id)}

    RevealHandCardStep(owner_key="owner", card_key="card").resolve(state, context)
    state.card_reveal = None
    context.clear()
    restored = GameState.model_validate(state.model_dump(mode="json"))

    result = knowledge_api.build_public_card_knowledge(restored, "hero_wasp")
    assert str(card.id) in hero_knowledge(result, str(arien.id)).revealed_card_ids


def test_wrong_color_guess_remains_public_after_temporary_fields_clear(knowledge_api) -> None:
    state = game(["Wasp"], ["Arien"])
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    card = arien.hand[0]
    guessed = "BLUE" if card.color is None or card.color.value != "BLUE" else "RED"
    context = {
        "card": str(card.id),
        "guess": guessed,
        "victim": str(arien.id),
    }

    RevealAndResolveGuessStep(
        card_key="card",
        guess_key="guess",
        victim_key="victim",
        correct_output_key="correct",
        wrong_output_key="wrong",
    ).resolve(state, context)
    assert context["wrong"] is True
    state.card_guess = None
    context.clear()
    restored = GameState.model_validate(state.model_dump(mode="json"))

    result = knowledge_api.build_public_card_knowledge(restored, "hero_wasp")
    assert str(card.id) in hero_knowledge(result, str(arien.id)).revealed_card_ids


def test_private_pending_level_up_progress_does_not_change_public_result(knowledge_api) -> None:
    state = game(["Wasp"], ["Arien"])
    before = knowledge_api.build_public_card_knowledge(state, "hero_wasp")
    state.pending_upgrades[HeroID("hero_arien")] = 2
    assert knowledge_api.build_public_card_knowledge(state, "hero_wasp") == before


def test_result_and_nested_models_are_immutable(knowledge_api) -> None:
    result = knowledge_api.build_public_card_knowledge(game(["Arien"], ["Wasp"]), None)
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        result.heroes = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.heroes["hero_arien"] = None
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        result.heroes["hero_arien"].status = knowledge_api.CardKnowledgeStatus.EXACT
