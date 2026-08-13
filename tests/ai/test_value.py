"""Tests for the ValueFn seam (T1).

Covers the normalized ValueFn contract:

- :class:`HeuristicValue` returns ``tanh(evaluate_state / scale)`` in
  ``[-1, 1]`` with a configurable ``scale`` (default 300.0 preserving the
  previous semantics), and rejects non-positive / non-finite scales at
  construction time.
- A custom ``ValueFn`` is actually consulted by the search, injection
  preserves determinism, and the implicit default is behaviorally identical
  to an explicit ``HeuristicValue()``.
- ``SearchConfig`` no longer exposes a ``value_scale`` knob (the squash lives
  on the ValueFn implementation).
- The rollout maps a normalized value into ``[0, 1]`` reward exactly once via
  ``(v + 1) / 2`` and fails fast (``ValueError``) on non-finite / out-of-range
  ValueFn output.
- The terminal branch preserves 1.0 / 0.0 / 0.5 semantics via the public
  ``automata.search.ismcts.terminal_reward`` seam.
"""

from __future__ import annotations

import math

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.evaluation.features import evaluate_state
from automata.evaluation.value import HeuristicValue, ValueFn
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from automata.search import ISMCTSAgent, SearchConfig
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.engine.setup import GameSetup

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]


def _state(seed: int = 2) -> GameState:
    register_all_effects()
    return GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=seed)


def _tiny(*, seed: int = 0) -> SearchConfig:
    return SearchConfig(iterations=4, cutoff_rounds=1, seed=seed)


def _asymmetric_state(seed: int = 2) -> GameState:
    """Return a state where evaluate_state is nonzero for both teams, so
    tanh-squashing and identity produce measurably different outputs.
    """
    st = _state(seed=seed)
    st.teams[TeamColor.RED].life_counters = 3
    st.teams[TeamColor.BLUE].life_counters = 5
    return st


def test_heuristic_value_matches_tanh_of_evaluate_state() -> None:
    """HeuristicValue owns the squash: its output must equal
    ``tanh(evaluate_state / scale)`` with the default scale that preserves
    the previous scale=300 semantics.

    Uses an asymmetric state so ``evaluate_state`` is nonzero — otherwise
    ``tanh(0)`` and identity trivially agree and the assertion is vacuous.
    """
    st = _asymmetric_state()
    hv = HeuristicValue()
    for team in (TeamColor.RED, TeamColor.BLUE):
        raw = evaluate_state(st, team)
        assert raw != 0.0  # guard the discriminating precondition
        expected = math.tanh(raw / 300.0)
        assert hv(st, team) == pytest.approx(expected)
        # And the OLD contract (identity) must NOT hold — otherwise we'd
        # be re-testing the pre-refactor code path.
        assert hv(st, team) != pytest.approx(raw)


def test_custom_value_fn_is_consulted() -> None:
    st = _state()
    calls: list[TeamColor] = []

    class SpyValue:
        def __call__(self, state: GameState, team: TeamColor) -> float:
            calls.append(team)
            return 0.0

    spy: ValueFn = SpyValue()
    agent = ISMCTSAgent(_tiny(seed=1), value_fn=spy)
    hero = st.teams[next(iter(st.teams))].heroes[0]
    agent.choose_card(st, hero)
    # With a non-trivial hand and a cutoff, the search must reach leaves and
    # call the value fn at least once.
    if len(hero.hand) > 1:
        assert calls, "custom ValueFn was never consulted by search"


def test_injected_value_fn_preserves_determinism() -> None:
    def choose() -> object:
        st = _state()
        agent = ISMCTSAgent(_tiny(seed=7), value_fn=HeuristicValue())
        hero = st.teams[next(iter(st.teams))].heroes[0]
        c = agent.choose_card(st, hero)
        return c.id if c else None

    assert choose() == choose()


def test_default_value_fn_matches_explicit_heuristic_value() -> None:
    """The implicit default ValueFn must be behaviorally identical to an
    explicit ``HeuristicValue()``.

    Observed via the public search seam: two searches with the same seed,
    config, state, policy and prior — one omitting ``value_fn`` (default),
    one passing ``HeuristicValue()`` explicitly — must produce equal
    ``SearchResult``s. We compare the whole (best_key, per-child visits, per-
    child mean Q) triple rather than just the picked card, so the assertion
    catches any drift in the default even when the same top action would
    still be picked by chance.
    """
    from automata.search.ismcts import RootTarget, search

    st = _state(seed=2)
    hero = st.teams[TeamColor.RED].heroes[0]
    if len(hero.hand) < 2:
        pytest.skip("need multi-card hand for a meaningful distribution")
    legal = [c.id for c in hero.hand]
    cfg = SearchConfig(iterations=8, cutoff_rounds=1, seed=0)
    target = RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id}))

    def _run(value_fn: ValueFn | None) -> tuple[object, list[tuple[str, int, float]]]:
        result = search(
            st,
            TeamColor.RED,
            "CARD",
            legal,
            HeuristicAgent(0),
            cfg,
            None,  # no prior — isolates the value function's influence
            value_fn,
            root_target=target,
        )
        # Canonicalize child distribution by legal-key order for a stable
        # comparison independent of dict insertion order.
        dist = [
            (
                k,
                result.root.children[k].visits if k in result.root.children else 0,
                result.root.children[k].q if k in result.root.children else 0.0,
            )
            for k in legal
        ]
        return result.best_key, dist

    default_best, default_dist = _run(None)
    explicit_best, explicit_dist = _run(HeuristicValue())
    assert default_best == explicit_best
    assert default_dist == explicit_dist


# --------------------------------------------------------------------------- #
# Task 1 contract: normalized ValueFn output in [-1, 1], search-side mapping,
# SearchConfig no longer carries value_scale, invalid outputs fail fast, and
# terminal outcomes remain unchanged.
#
# Behavioral seams used below (no private coupling to _rollout / _Simulator /
# Decision):
#
# - ``HeuristicValue.__call__`` — public ValueFn, observed via output only.
# - ``ISMCTSAgent.choose_card`` — public agent boundary; propagates ValueError
#   raised inside the search when a ValueFn output is invalid.
# - ``search()`` + ``SearchResult.root`` — established search seam already used
#   by ``test_ismcts.py``; ``root.q`` is the tree's mean reward.
# - ``automata.search.ismcts.terminal_reward`` — the intended public
#   terminal-reward helper. Referenced via module attribute access so the
#   test fails cleanly with AttributeError until production exposes the seam.
# --------------------------------------------------------------------------- #


def test_heuristic_value_output_is_in_normalized_range() -> None:
    """``HeuristicValue`` must return a scalar in [-1, 1] for any state — that
    is the contract every ValueFn implementation obeys, and tanh's image is
    exactly (-1, 1) so the wrapper cannot escape the range.

    Uses an asymmetric state (nonzero raw score) so the range check is not
    trivially satisfied by 0.0.
    """
    st = _asymmetric_state()
    hv = HeuristicValue()
    for team in (TeamColor.RED, TeamColor.BLUE):
        v = hv(st, team)
        assert -1.0 <= v <= 1.0
        # And it must actually be a squashed value, not the raw (which for
        # this state is |200.0|, well outside [-1, 1]).
        raw = evaluate_state(st, team)
        assert abs(raw) > 1.0
        assert v != pytest.approx(raw)


def test_heuristic_value_default_scale_preserves_300_semantics() -> None:
    """The default scale must preserve the previous ``value_scale=300``
    semantics: a raw ``evaluate_state`` score equal to 300 should map to
    ``tanh(1) ≈ 0.7616`` — verified via output only.

    We arrange a state whose raw score is exactly ±300 (life differential of
    3 with weight 100) and assert the wrapper emits ``±tanh(1)``. If the
    default were, say, 100, the output would be ``±tanh(3) ≈ ±0.9951`` —
    distinguishable to many decimal places.
    """
    st = _state()
    st.teams[TeamColor.RED].life_counters = 3
    st.teams[TeamColor.BLUE].life_counters = 0
    # life_diff = 3 for RED → raw = 3 * 100 (life weight) = 300 exactly.
    assert evaluate_state(st, TeamColor.RED) == 300.0
    assert evaluate_state(st, TeamColor.BLUE) == -300.0
    hv = HeuristicValue()
    assert hv(st, TeamColor.RED) == pytest.approx(math.tanh(1.0))
    assert hv(st, TeamColor.BLUE) == pytest.approx(-math.tanh(1.0))


def test_heuristic_value_scale_is_overridable() -> None:
    """Callers must be able to construct a HeuristicValue with a custom
    scale (needed for experiments / re-tuning without touching search). The
    output must obey ``tanh(score / custom_scale)`` exactly — verified via
    output.
    """
    st = _asymmetric_state()
    hv = HeuristicValue(scale=100.0)
    for team in (TeamColor.RED, TeamColor.BLUE):
        raw = evaluate_state(st, team)
        assert raw != 0.0
        expected = math.tanh(raw / 100.0)
        assert hv(st, team) == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad_scale",
    [
        0.0,
        -1.0,
        -300.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
    ids=["zero", "negative_one", "negative_scale", "positive_inf", "negative_inf", "nan"],
)
def test_heuristic_value_rejects_invalid_scale(bad_scale: float) -> None:
    """A non-positive or non-finite ``scale`` is meaningless for ``tanh(x /
    scale)`` and would silently corrupt the normalized output (division by
    zero, NaN, or inverting the sign). Construction must fail fast with
    ``ValueError`` at each of the boundary cases rather than accepting the
    scale and producing garbage on first call.
    """
    with pytest.raises(ValueError):
        HeuristicValue(scale=bad_scale)


def test_search_config_no_longer_exposes_value_scale() -> None:
    """The value_scale knob has moved off SearchConfig onto HeuristicValue.
    A code path still reading ``SearchConfig.value_scale`` would be silently
    wrong under the new contract, so the attribute must be gone.
    """
    cfg = SearchConfig()
    assert not hasattr(cfg, "value_scale")


def test_ismcts_reward_from_fixed_value_fn_uses_v_plus_one_over_two() -> None:
    """Rollout reward = ``(value + 1) / 2``, applied exactly once.

    We observe this at the public search boundary: a ValueFn that returns a
    fixed ``v`` on every non-terminal state produces rollout rewards that are
    all ``(v+1)/2``, so the tree's mean Q at both the root and every child
    must equal ``(v+1)/2``. Rollouts do not reach terminals within the tiny
    cutoff, so terminal short-circuiting is not exercised here — that is
    covered separately by the terminal tests below.

    Any double-squash (e.g. the old ``0.5 * (1 + tanh(value / scale))``) would
    produce a different Q — for v = 1.0, the old path returned ≈ 0.5017
    instead of the contract's 1.0.

    Uses ``search`` (a public search-module function already used by the
    existing ISMCTS tests) directly so the observation reaches the tree's
    reward totals without depending on ``ISMCTSAgent`` private fields.
    """
    from automata.search.ismcts import RootTarget, search

    class _FixedValue:
        def __init__(self, v: float) -> None:
            self._v = v
            self.calls = 0

        def __call__(self, state: GameState, team: TeamColor) -> float:
            self.calls += 1
            return self._v

    for v, expected_q in ((0.0, 0.5), (1.0, 1.0), (-1.0, 0.0)):
        st = _state(seed=2)
        hero = st.teams[TeamColor.RED].heroes[0]
        if len(hero.hand) < 2:
            pytest.skip("need multi-card hand to force real rollouts")
        spy = _FixedValue(v)
        # 8 iterations expands the tree enough to gather multiple rewards;
        # cutoff_rounds=1 keeps each rollout very short.
        cfg = SearchConfig(iterations=8, cutoff_rounds=1, seed=0)
        target = RootTarget.card(hero_id=hero.id, owned_hero_ids=frozenset({hero.id}))
        result = search(
            st,
            TeamColor.RED,
            "CARD",
            [c.id for c in hero.hand],
            HeuristicAgent(0),  # default policy for opponents/rollouts
            cfg,
            None,  # no expansion prior — pure UCB1 exploration
            spy,
            root_target=target,
        )
        assert spy.calls > 0, f"value fn never consulted (v={v})"
        # Every visited node's mean Q must equal the mapping — no other reward
        # source contributed under this constant ValueFn.
        assert result.root.visits > 0
        assert result.root.q == pytest.approx(
            expected_q
        ), f"root Q for v={v}: {result.root.q} != {expected_q}"
        # And every expanded child inherits the same reward — no double-mapping
        # partway down the tree.
        for child in result.root.children.values():
            if child.visits > 0:
                assert child.q == pytest.approx(
                    expected_q
                ), f"child Q for v={v}: {child.q} != {expected_q}"


def test_ismcts_rejects_value_fn_output_above_range() -> None:
    """A ValueFn that returns a value > 1.0 is broken — the search must
    fail fast with ValueError, propagated through the public agent boundary.
    Silent clamping would hide a model bug (e.g. an un-normalized head).
    """

    def _too_high(state: GameState, team: TeamColor) -> float:
        return 1.5

    st = _state()
    hero = st.teams[TeamColor.RED].heroes[0]
    if len(hero.hand) < 1:
        pytest.skip("need at least one card in hand to drive a search")
    agent = ISMCTSAgent(_tiny(seed=0), value_fn=_too_high)
    with pytest.raises(ValueError):
        agent.choose_card(st, hero)


def test_ismcts_rejects_value_fn_output_below_range() -> None:
    """Symmetric: a value < -1.0 must surface a ValueError through the
    agent boundary.
    """

    def _too_low(state: GameState, team: TeamColor) -> float:
        return -1.5

    st = _state()
    hero = st.teams[TeamColor.RED].heroes[0]
    if len(hero.hand) < 1:
        pytest.skip("need at least one card in hand to drive a search")
    agent = ISMCTSAgent(_tiny(seed=0), value_fn=_too_low)
    with pytest.raises(ValueError):
        agent.choose_card(st, hero)


def test_ismcts_rejects_non_finite_value_fn_output() -> None:
    """NaN / +inf / -inf are all invalid ValueFn outputs — probably a model
    numerical failure. The search must surface a ValueError rather than
    poison the tree with a NaN backup.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):

        def _bad(state: GameState, team: TeamColor, _v: float = bad) -> float:
            return _v

        st = _state()
        hero = st.teams[TeamColor.RED].heroes[0]
        if len(hero.hand) < 1:
            pytest.skip("need at least one card in hand to drive a search")
        agent = ISMCTSAgent(_tiny(seed=0), value_fn=_bad)
        with pytest.raises(ValueError):
            agent.choose_card(st, hero)


# --------------------------------------------------------------------------- #
# Terminal-outcome preservation.
#
# The rollout's terminal branch resolves winner → reward BEFORE any ValueFn is
# consulted. Task 1 requires this branch to remain semantically unchanged and
# to be exposed on the search module as a stable public helper
# ``terminal_reward(winner: str | None, our_team: TeamColor) -> float`` so
# tests (and later a learned-value policy trainer) can reference it without
# touching the private simulator. We drive the tests through module attribute
# access — ``terminal_reward`` does not exist yet, so these fail cleanly with
# ``AttributeError`` until production exposes the seam.
# --------------------------------------------------------------------------- #


def test_terminal_reward_own_team_win_is_one() -> None:
    """Winner == our team → reward 1.0. The unchanged terminal semantic the
    rollout hits before consulting any ValueFn, exposed via the public
    ``terminal_reward`` seam on ``automata.search.ismcts``."""
    from automata.search import ismcts as ismcts_mod

    assert ismcts_mod.terminal_reward(TeamColor.RED.value, TeamColor.RED) == 1.0
    assert ismcts_mod.terminal_reward(TeamColor.BLUE.value, TeamColor.BLUE) == 1.0


def test_terminal_reward_opponent_win_is_zero() -> None:
    """Winner == opponent team → reward 0.0."""
    from automata.search import ismcts as ismcts_mod

    assert ismcts_mod.terminal_reward(TeamColor.BLUE.value, TeamColor.RED) == 0.0
    assert ismcts_mod.terminal_reward(TeamColor.RED.value, TeamColor.BLUE) == 0.0


def test_terminal_reward_no_winner_is_half() -> None:
    """Winner is None (draw / undecided) → reward 0.5."""
    from automata.search import ismcts as ismcts_mod

    assert ismcts_mod.terminal_reward(None, TeamColor.RED) == 0.5
    assert ismcts_mod.terminal_reward(None, TeamColor.BLUE) == 0.5
