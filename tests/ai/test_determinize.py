"""Tests for ISMCTS determinization."""

from __future__ import annotations

import random

import pytest

from automata.evaluation.features import state_features
from automata.runtime.determinize import determinize
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from goa2.domain.models import CardState, CardTier, GamePhase, StatType, TeamColor
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.phases import commit_card, finish_planning
from goa2.engine.setup import GameSetup
from goa2.engine.steps.cards import apply_hero_upgrade


def _fresh() -> GameState:
    register_all_effects()
    return GameSetup.create_game(
        DEFAULT_MAP, ["Wasp", "Xargatha"], ["Arien", "Brogan"], game_type="QUICK", seed=1
    )


def test_determinize_resamples_enemy_commit_and_leaves_original() -> None:
    state = _fresh()
    assert state.phase == GamePhase.PLANNING
    blue_hero = state.teams[TeamColor.BLUE].heroes[0]
    hand_ids = {c.id for c in blue_hero.hand}
    committed = blue_hero.hand[0]
    commit_card(state, HeroID(blue_hero.id), committed)
    pending = state.pending_inputs[HeroID(blue_hero.id)]
    assert pending is not None and pending.id == committed.id

    # Perspective RED must not see BLUE's real commit; determinize resamples it.
    red_hero = state.teams[TeamColor.RED].heroes[0]
    clone = determinize(state, red_hero.id, random.Random(0))
    clone_commit = clone.pending_inputs[HeroID(blue_hero.id)]
    assert clone_commit is not None
    assert clone_commit.id in hand_ids  # a legal card from that hero's hand

    # Original is untouched.
    original_pending = state.pending_inputs[HeroID(blue_hero.id)]
    assert original_pending is not None and original_pending.id == committed.id


def test_determinize_resets_unselected_commit_lifecycle_and_feature_counts() -> None:
    state = _fresh()
    hidden = state.teams[TeamColor.BLUE].heroes[0]
    original = hidden.hand[0]
    replacement = hidden.hand[1]
    commit_card(state, HeroID(hidden.id), original)
    perspective = state.teams[TeamColor.RED].heroes[0]
    before = state_features(state, TeamColor.RED, "rich-v1")

    clone = determinize(state, perspective.id, _PickCards([replacement.id]))

    cloned_hidden = clone.get_hero(HeroID(hidden.id))
    assert cloned_hidden is not None
    returned = next(card for card in cloned_hidden.hand if card.id == original.id)
    assert returned.state is CardState.HAND
    assert returned.is_facedown is False
    assert returned.played_this_round is False
    after = state_features(clone, TeamColor.RED, "rich-v1")
    assert after["enemy_hand_cards"] == before["enemy_hand_cards"]
    assert after["enemy_played_cards"] == before["enemy_played_cards"]


def test_determinize_keeps_only_perspective_hero_commit() -> None:
    state = _fresh()
    red_hero = state.teams[TeamColor.RED].heroes[0]
    committed = red_hero.hand[0]
    commit_card(state, HeroID(red_hero.id), committed)

    clone = determinize(state, red_hero.id, random.Random(0))  # type: ignore[arg-type]
    # Only the deciding hero's private commitment is known.
    pending = clone.pending_inputs[HeroID(red_hero.id)]
    assert pending is not None and pending.id == committed.id


def test_determinize_is_deterministic_given_rng() -> None:
    def run() -> str | None:
        state = _fresh()
        bh = state.teams[TeamColor.BLUE].heroes[0]
        commit_card(state, HeroID(bh.id), bh.hand[0])
        perspective = state.teams[TeamColor.RED].heroes[0]
        clone = determinize(state, perspective.id, random.Random(42))  # type: ignore[arg-type]
        c = clone.pending_inputs[HeroID(bh.id)]
        return c.id if c else None

    assert run() == run()


def test_determinize_outside_planning_is_plain_clone() -> None:
    state = _fresh()
    state.phase = GamePhase.RESOLUTION
    perspective = state.teams[TeamColor.RED].heroes[0]
    clone = determinize(state, perspective.id, random.Random(0))  # type: ignore[arg-type]
    assert clone is not state
    assert clone.phase == GamePhase.RESOLUTION


class _PickCards(random.Random):
    """Deterministic fake that selects named cards from any offered sequence."""

    def __init__(self, card_ids: list[str]) -> None:
        super().__init__(0)
        self.card_ids = iter(card_ids)

    def choice(self, seq):  # type: ignore[no-untyped-def]
        if hasattr(seq[0], "active_card_ids"):
            return seq[0]
        wanted = next(self.card_ids)
        return next(card for card in seq if card.id == wanted)


def test_determinize_fully_planned_snapshot_does_not_reveal() -> None:
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=5)
    perspective = state.teams[TeamColor.RED].heroes[0]
    hidden = state.teams[TeamColor.BLUE].heroes[0]
    commit_card(state, HeroID(perspective.id), perspective.hand[0])

    # Model a planning snapshot captured after every slot was filled but before
    # the phase transition callback ran.
    original = hidden.hand[0]
    replacement = hidden.hand[1]
    hidden.play_card(original)
    state.pending_inputs[HeroID(hidden.id)] = original

    clone = determinize(state, perspective.id, _PickCards([replacement.id]))

    assert clone.phase is GamePhase.PLANNING
    assert set(clone.pending_inputs) == {HeroID(perspective.id), HeroID(hidden.id)}
    sampled = clone.pending_inputs[HeroID(hidden.id)]
    assert sampled is not None and sampled.id == replacement.id
    assert sampled.is_facedown is True


def _emmitt_state(emmitt_team: TeamColor) -> tuple[GameState, Hero]:
    register_all_effects()
    red = ["Emmitt"] if emmitt_team is TeamColor.RED else ["Wasp"]
    blue = ["Emmitt"] if emmitt_team is TeamColor.BLUE else ["Wasp"]
    state = GameSetup.create_game(DEFAULT_MAP, red, blue, game_type="QUICK", seed=4)
    emmitt = state.teams[emmitt_team].heroes[0]
    emmitt.level = 8
    return state, emmitt


@pytest.mark.parametrize("is_ally", [True, False])
def test_determinize_fully_resamples_hidden_two_card_commit(is_ally: bool) -> None:
    """An ally is no more visible than an enemy; neither first card is pinned."""
    if is_ally:
        register_all_effects()
        state = GameSetup.create_game(
            DEFAULT_MAP, ["Wasp", "Emmitt"], ["Arien"], game_type="QUICK", seed=4
        )
        perspective, hidden = state.teams[TeamColor.RED].heroes
        hidden.level = 8
    else:
        state, hidden = _emmitt_state(TeamColor.RED)
        perspective = state.teams[TeamColor.BLUE].heroes[0]
    original = [hidden.hand[0], hidden.hand[1]]
    alternatives = [
        next(card.id for card in hidden.hand if card.id == "unstable_timeline"),
        next(card.id for card in hidden.hand if card.id == "reverse_time"),
    ]
    commit_card(state, HeroID(hidden.id), original[0])
    commit_card(state, HeroID(hidden.id), original[1])

    clone = determinize(state, perspective.id, _PickCards(alternatives))  # type: ignore[arg-type]

    sampled_first = clone.pending_inputs[HeroID(hidden.id)]
    sampled_second = clone.pending_second_cards[HeroID(hidden.id)]
    assert sampled_first is not None and sampled_second is not None
    assert [sampled_first.id, sampled_second.id] == alternatives
    assert state.pending_inputs[HeroID(hidden.id)] is original[0]
    assert state.pending_second_cards[HeroID(hidden.id)] is original[1]


def test_determinize_preserves_emmitt_perspective_while_searching_second_card() -> None:
    state, emmitt = _emmitt_state(TeamColor.RED)
    first = emmitt.hand[0]
    commit_card(state, HeroID(emmitt.id), first)

    clone = determinize(state, emmitt.id, random.Random(9))  # type: ignore[arg-type]

    cloned_first = clone.pending_inputs[HeroID(emmitt.id)]
    assert cloned_first is not None
    assert cloned_first.id == first.id
    assert HeroID(emmitt.id) not in clone.pending_second_cards
    cloned_emmitt = clone.get_hero(HeroID(emmitt.id))
    assert cloned_emmitt is not None
    assert [card.id for card in cloned_emmitt.hand] == [card.id for card in emmitt.hand]


def test_determinize_preserves_hidden_commit_count_and_planning_done() -> None:
    state, hidden = _emmitt_state(TeamColor.RED)
    perspective = state.teams[TeamColor.BLUE].heroes[0]
    original = hidden.hand[0]
    replacement_id = next(card.id for card in hidden.hand if card.id == "reverse_time")
    commit_card(state, HeroID(hidden.id), original)
    finish_planning(state, HeroID(hidden.id))

    clone = determinize(  # type: ignore[arg-type]
        state, perspective.id, _PickCards([replacement_id])
    )

    sampled = clone.pending_inputs[HeroID(hidden.id)]
    assert sampled is not None and sampled.id == replacement_id
    assert HeroID(hidden.id) not in clone.pending_second_cards
    assert HeroID(hidden.id) in clone.planning_done


class _PickLoadout(random.Random):
    def __init__(self, active_card_id: str, commit_card_id: str | None = None) -> None:
        super().__init__(0)
        self.active_card_id = active_card_id
        self.commit_card_id = commit_card_id

    def choice(self, seq):  # type: ignore[no-untyped-def]
        if hasattr(seq[0], "active_card_ids"):
            return next((h for h in seq if self.active_card_id in h.active_card_ids), seq[0])
        if self.commit_card_id is not None:
            return next(card for card in seq if str(card.id) == self.commit_card_id)
        return seq[0]


def _level_two_arien() -> tuple[GameState, Hero, Hero]:
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=17)
    perspective = state.teams[TeamColor.RED].heroes[0]
    arien = state.teams[TeamColor.BLUE].heroes[0]
    arien.level = 2
    apply_hero_upgrade(state, str(arien.id), "magical_current")
    return state, perspective, arien


@pytest.mark.parametrize(
    ("sampled_id", "item_id"),
    [
        ("magical_current", "arcane_whirlpool"),
        ("raging_stream", "rogue_wave"),
    ],
)
def test_determinize_samples_each_public_compatible_upgrade_loadout(
    sampled_id: str, item_id: str
) -> None:
    state, perspective, arien = _level_two_arien()
    original = state.model_dump(mode="json")

    clone = determinize(state, perspective.id, _PickLoadout(sampled_id))
    sampled = clone.get_hero(HeroID(arien.id))
    assert sampled is not None
    assert {str(card.id) for card in sampled.hand if card.tier in {CardTier.II, CardTier.III}} == {
        sampled_id
    }
    assert next(card for card in sampled.deck if card.id == item_id).state is CardState.ITEM
    assert state.model_dump(mode="json") == original

    owner_clone = determinize(state, arien.id, _PickLoadout("raging_stream"))
    owner = owner_clone.get_hero(HeroID(arien.id))
    assert owner is not None
    assert [card.id for card in owner.hand] == [card.id for card in arien.hand]


def test_determinize_public_reveal_constrains_upgrade_hypothesis() -> None:
    state, perspective, arien = _level_two_arien()
    state.record_public_revealed_card(arien.id, "magical_current")

    clone = determinize(state, perspective.id, _PickLoadout("raging_stream"))
    sampled = clone.get_hero(HeroID(arien.id))
    assert sampled is not None
    assert "magical_current" in {str(card.id) for card in sampled.hand}
    assert "raging_stream" not in {str(card.id) for card in sampled.hand}


def test_commitment_is_resampled_from_sampled_upgrade_hand() -> None:
    state, perspective, arien = _level_two_arien()
    actual_upgrade = next(card for card in arien.hand if card.id == "magical_current")
    commit_card(state, HeroID(arien.id), actual_upgrade)

    clone = determinize(state, perspective.id, _PickLoadout("raging_stream", "raging_stream"))

    sampled_commit = clone.pending_inputs[HeroID(arien.id)]
    assert sampled_commit is not None and sampled_commit.id == "raging_stream"
    assert sampled_commit.id != actual_upgrade.id
    assert state.pending_inputs[HeroID(arien.id)] is actual_upgrade


def test_unsupported_item_provenance_uses_unconditioned_static_fallback() -> None:
    state, perspective, arien = _level_two_arien()
    arien.items = {StatType.DEFENSE: 2}

    clone = determinize(state, perspective.id, _PickLoadout("expert_duelist"))
    sampled = clone.get_hero(HeroID(arien.id))
    assert sampled is not None
    assert {str(card.id) for card in sampled.deck if card.state is CardState.ITEM} <= {
        "arcane_whirlpool",
        "rogue_wave",
    }
    assert sampled.items == {StatType.DEFENSE: 2}


def test_tier_three_determinization_preserves_public_lifecycle_references() -> None:
    state, perspective, arien = _level_two_arien()
    arien.level = 5
    apply_hero_upgrade(state, str(arien.id), "expert_duelist")
    apply_hero_upgrade(state, str(arien.id), "rogue_wave")
    apply_hero_upgrade(state, str(arien.id), "violent_torrent")
    resolved = next(card for card in arien.hand if card.id == "violent_torrent")
    arien.hand.remove(resolved)
    resolved.state = CardState.RESOLVED
    resolved.played_this_round = True
    arien.played_cards = [resolved]
    arien.resolved_turn_count = 1
    state.record_public_revealed_card(arien.id, str(resolved.id))

    clone = determinize(state, perspective.id, _PickLoadout("violent_torrent"))
    sampled = clone.get_hero(HeroID(arien.id))
    assert sampled is not None
    deck_card = next(card for card in sampled.deck if card.id == "violent_torrent")
    assert sampled.played_cards == [deck_card]
    assert sampled.played_cards[0] is deck_card
    assert deck_card.state is CardState.RESOLVED
    tier_two_red = {
        card.state
        for card in sampled.deck
        if card.color is not None and card.color.value == "RED" and card.tier is CardTier.II
    }
    assert tier_two_red == {CardState.ITEM, CardState.RETIRED}
    assert next(card for card in sampled.deck if card.id == "tidal_blast").state is CardState.ITEM


@pytest.mark.parametrize("name", ["Min", "Dodger", "Snorri"])
def test_nonstandard_hero_fallback_ignores_unsafe_historical_reveals(name: str) -> None:
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], [name], game_type="QUICK", seed=23)
    perspective = state.teams[TeamColor.RED].heroes[0]
    hidden = state.teams[TeamColor.BLUE].heroes[0]
    hidden.level = 2
    tier_two = [card for card in hidden.deck if card.tier is CardTier.II]
    revealed, sampled_card = tier_two[0], tier_two[1]
    assert sampled_card.id not in {card.id for card in hidden.hand}
    state.record_public_revealed_card(hidden.id, str(revealed.id))

    clone = determinize(state, perspective.id, _PickLoadout(str(sampled_card.id)))
    sampled = clone.get_hero(HeroID(hidden.id))
    assert sampled is not None
    assert sampled_card.id in {card.id for card in sampled.hand}
