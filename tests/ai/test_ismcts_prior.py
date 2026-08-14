"""Tests for the ISMCTS expansion prior (heuristic move ordering).

The prior only affects *which* legal child progressive widening reveals first,
never legality or value — so these assert ordering behavior, that a real search
stays legal/deterministic with the prior on, and that it can be switched off.
"""

from __future__ import annotations

import random

from automata.agents.heuristic_agent import HeuristicAgent
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from automata.search import ISMCTSAgent, SearchConfig
from automata.search.ismcts import Decision, legal_keys
from automata.search.node import Node
from automata.search.prior import HeuristicPrior
from goa2.engine.setup import GameSetup

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]


def _tiny_cfg(*, seed: int = 0, use_prior: bool = True) -> SearchConfig:
    return SearchConfig(iterations=2, cutoff_rounds=1, seed=seed, use_prior=use_prior)


# --- Node.expand ordering ------------------------------------------------- #


def test_expand_follows_prior_order() -> None:
    node = Node()
    legal = ["a", "b", "c", "d"]
    order = ["c", "a", "d", "b"]  # best-first
    rng = random.Random(0)
    # Reveal all children; each must be the next unexpanded key in `order`.
    revealed = [node.expand(legal, rng, order) for _ in range(len(legal))]
    assert revealed == order


def test_expand_falls_back_to_random_without_order() -> None:
    node = Node()
    legal = ["a", "b", "c"]
    rng = random.Random(1)
    revealed = {node.expand(legal, rng) for _ in range(len(legal))}
    assert revealed == set(legal)  # all revealed exactly once, no crash


def test_expand_ignores_order_entries_not_in_legal() -> None:
    node = Node()
    legal = ["a", "b"]
    order = ["z", "b", "a"]  # 'z' not legal
    rng = random.Random(0)
    first = node.expand(legal, rng, order)
    assert first == "b"  # first legal key in order


def test_expand_handles_order_missing_a_legal_key() -> None:
    # If the prior omits a legal key, expand must still be able to reveal it
    # (random fallback) rather than crash.
    node = Node()
    legal = ["a", "b", "c"]
    order = ["a"]  # incomplete
    rng = random.Random(0)
    revealed = {node.expand(legal, rng, order) for _ in range(3)}
    assert revealed == set(legal)


# --- HeuristicPrior ranking ----------------------------------------------- #


def test_heuristic_prior_ranks_cards_by_score() -> None:
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=2)
    hero = state.teams[next(iter(state.teams))].heroes[0]
    if not hero.hand:
        return  # nothing to rank in this setup
    h = HeuristicAgent(0)
    decision = Decision("CARD", hero=hero)
    legal = legal_keys(decision)
    result = HeuristicPrior(h)(state, decision, legal)
    order = result.order
    # Same set, and sorted by descending heuristic card score.
    assert set(order) == set(legal)
    by_id = {c.id: c for c in hero.hand}
    scores = [h.score_card(state, hero, by_id[k]) for k in order]
    assert scores == sorted(scores, reverse=True)


def test_heuristic_prior_exposes_weights_aligned_with_order() -> None:
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=2)
    hero = state.teams[next(iter(state.teams))].heroes[0]
    if len(hero.hand) < 2:
        return
    result = HeuristicPrior()(
        state, Decision("CARD", hero=hero), legal_keys(Decision("CARD", hero=hero))
    )
    # Weights present, cover the order, and are non-increasing along the order.
    assert result.weights is not None
    ws = [result.weights[k] for k in result.order]
    assert ws == sorted(ws, reverse=True)


def test_heuristic_prior_returns_permutation() -> None:
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=4)
    hero = state.teams[next(iter(state.teams))].heroes[0]
    decision = Decision("CARD", hero=hero)
    legal = legal_keys(decision)
    order = HeuristicPrior()(state, decision, legal).order
    assert sorted(map(str, order)) == sorted(map(str, legal))


# --- agent integration ----------------------------------------------------- #


def test_agent_with_prior_returns_legal_card() -> None:
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=2)
    agent = ISMCTSAgent(_tiny_cfg(use_prior=True))
    hero = state.teams[next(iter(state.teams))].heroes[0]
    card = agent.choose_card(state, hero)
    if hero.hand:
        assert card is not None
        assert card.id in {c.id for c in hero.hand}


def test_agent_prior_can_be_disabled() -> None:
    assert ISMCTSAgent(_tiny_cfg(use_prior=False))._prior is None
    assert ISMCTSAgent(_tiny_cfg(use_prior=True))._prior is not None


def test_agent_with_prior_is_deterministic() -> None:
    register_all_effects()

    def choose() -> object:
        state = GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=2)
        agent = ISMCTSAgent(_tiny_cfg(seed=7, use_prior=True))
        hero = state.teams[next(iter(state.teams))].heroes[0]
        c = agent.choose_card(state, hero)
        return c.id if c else None

    assert choose() == choose()
