"""Tests for the LearnedValue seam (T4).

`LearnedValue` is the portable, dependency-free inference wrapper for a
learned logistic model over the six differential features from
:mod:`automata.evaluation.features`. It obeys the same
:class:`~automata.evaluation.value.ValueFn` contract as
:class:`~automata.evaluation.value.HeuristicValue` (finite scalar in
``[-1, 1]``), so it drops into the search loop without changes.

This file pins the *public* contract only, through the module seam
``automata.evaluation.learned_value`` and the class ``LearnedValue``:

* Load from an on-disk JSON artifact path OR from an in-memory mapping.
* Portable, versioned schema (``model_version`` + ``schema_version``); the
  artifact declares its own ``red_roster`` / ``blue_roster`` — the trainer /
  generator picks the roster, LearnedValue does NOT hard-wire it.
* Structural + numeric validation surfaces as :class:`ValueError`: exact
  ``FEATURE_NAMES`` order, finite means / coefficients / intercept, strictly
  positive finite scales, matching vector lengths, correct versions.
* Runtime side-aware compatibility: a well-loaded artifact must refuse to
  score a :class:`~goa2.domain.state.GameState` whose rosters differ from
  the artifact's declared rosters.
* Inference math: standardize via ``z = (x - mean) / scale``, form
  ``logit = intercept + sum(coef * z)``, return ``tanh(logit / 2)`` — verified
  hand-calculably.
* Deterministic SHA-256 content digest so an evaluation run can pin model
  identity; the digest changes when any content field changes.
* Portable inference: neither ``numpy`` nor ``sklearn`` is imported.

All fixtures build a synthetic artifact aligned with a synthetic state so
failures point at behavior, not incidental setup.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from automata.evaluation.features import FEATURE_NAMES, feature_vector
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.engine.setup import GameSetup

# The seam under test is imported lazily inside each test so the file can be
# *collected* even before the module exists; each test then fails with a
# clear RED (missing module / API), not a collection error.

BENCH_RED = ["Wasp", "Xargatha"]
BENCH_BLUE = ["Arien", "Brogan"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _state(red: list[str], blue: list[str], seed: int = 2) -> GameState:
    register_all_effects()
    return GameSetup.create_game(DEFAULT_MAP, red, blue, game_type="QUICK", seed=seed)


def _bench_state(seed: int = 2) -> GameState:
    """State whose rosters match the canonical artifact below."""
    return _state(BENCH_RED, BENCH_BLUE, seed=seed)


def _mismatched_state(seed: int = 2) -> GameState:
    """State whose RED roster differs from the canonical artifact.

    A single-hero substitution on RED (``Xargatha`` → ``Bain``) is enough to
    force the runtime roster check to reject scoring: no assumption is made
    about which team is inspected first.
    """
    return _state(["Wasp", "Bain"], BENCH_BLUE, seed=seed)


def _canonical_artifact() -> dict[str, Any]:
    """Well-formed logistic artifact aligned with ``FEATURE_NAMES``.

    Numerics chosen for hand-calculability:

    * ``feature_means`` = zeros → ``z = x / scale``.
    * ``feature_scales`` = ones → ``z = x``.
    * ``coefficients`` = ``[1, 0, 0, 0, 0, 0]`` → only ``life_diff`` contributes.
    * ``intercept`` = 0.0.

    So on a state with ``life_diff == 1`` the logit is exactly ``1.0`` and
    the output must be ``tanh(0.5)``.
    """
    return {
        "model_version": "logistic-v1",
        "schema_version": 1,
        "red_roster": list(BENCH_RED),
        "blue_roster": list(BENCH_BLUE),
        "feature_names": list(FEATURE_NAMES),
        "feature_means": [0.0] * len(FEATURE_NAMES),
        "feature_scales": [1.0] * len(FEATURE_NAMES),
        "coefficients": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "intercept": 0.0,
    }


def _write_artifact(tmp_path: Path, payload: dict[str, Any]) -> Path:
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(payload))
    return p


def _one_life_diff_state() -> GameState:
    """Force ``life_diff = 1`` for RED (⇒ ``-1`` for BLUE) on a bench state.

    Paired with ``_canonical_artifact()``, this drives an exact logit of
    ``±1.0`` and a hand-calculable output of ``tanh(±0.5)``.
    """
    st = _bench_state()
    st.teams[TeamColor.RED].life_counters = 5
    st.teams[TeamColor.BLUE].life_counters = 4
    return st


# --------------------------------------------------------------------------- #
# Module seam & portability
# --------------------------------------------------------------------------- #


def test_module_import_stays_pure_python() -> None:
    """The public seam lives at ``automata.evaluation.learned_value`` and
    exposes ``LearnedValue`` without pulling in ``numpy`` or ``sklearn`` —
    portability to constrained runtimes (clients, ML-dep-less CI) is part
    of the contract.
    """
    for name in list(sys.modules):
        if name == "automata.evaluation.learned_value" or name.startswith(
            "automata.evaluation.learned_value."
        ):
            del sys.modules[name]
    had_numpy = "numpy" in sys.modules
    had_sklearn = "sklearn" in sys.modules

    from automata.evaluation import learned_value as lv_mod

    assert hasattr(lv_mod, "LearnedValue")
    if not had_numpy:
        assert "numpy" not in sys.modules, "learned_value must not import numpy"
    if not had_sklearn:
        assert "sklearn" not in sys.modules, "learned_value must not import sklearn"


# --------------------------------------------------------------------------- #
# Loading paths
# --------------------------------------------------------------------------- #


def test_load_from_path_and_mapping_produce_callable_value_fn(tmp_path: Path) -> None:
    """Both input shapes — filesystem path and in-memory mapping — must yield
    a fully-loaded, callable ``LearnedValue`` (the ValueFn boundary is the
    only observable behavior we care about after load)."""
    from automata.evaluation.learned_value import LearnedValue

    p = _write_artifact(tmp_path, _canonical_artifact())
    for lv in (LearnedValue(p), LearnedValue(_canonical_artifact())):
        v = lv(_bench_state(), TeamColor.RED)
        assert isinstance(v, float)
        assert math.isfinite(v)


def test_load_rejects_missing_path(tmp_path: Path) -> None:
    """A non-existent path must fail cleanly at load time — never silently
    degrade into a default artifact."""
    from automata.evaluation.learned_value import LearnedValue

    with pytest.raises(FileNotFoundError):
        LearnedValue(tmp_path / "does-not-exist.json")


def test_load_rejects_directory_path(tmp_path: Path) -> None:
    """A path that exists but is a directory (not a regular file) must fail
    at load time with a clear ``ValueError``. Silently reading garbage — or
    letting an ``IsADirectoryError`` bubble up unwrapped — would violate the
    loader-is-single-validation-seam contract the other tests rely on.
    """
    from automata.evaluation.learned_value import LearnedValue

    directory = tmp_path / "artifact_dir"
    directory.mkdir()
    with pytest.raises(ValueError):
        LearnedValue(directory)


# --------------------------------------------------------------------------- #
# Inference math — hand-calculable
# --------------------------------------------------------------------------- #


def test_inference_matches_hand_calculation() -> None:
    """Canonical artifact + ``life_diff = ±1`` state ⇒ output ``tanh(±0.5)``.

    Failure here points at broken standardization, logit assembly, or squash.
    """
    from automata.evaluation.learned_value import LearnedValue

    lv = LearnedValue(_canonical_artifact())
    st = _one_life_diff_state()

    life_idx = FEATURE_NAMES.index("life_diff")
    assert feature_vector(st, TeamColor.RED)[life_idx] == 1.0  # guard precondition
    assert feature_vector(st, TeamColor.BLUE)[life_idx] == -1.0

    assert lv(st, TeamColor.RED) == pytest.approx(math.tanh(0.5))
    assert lv(st, TeamColor.BLUE) == pytest.approx(math.tanh(-0.5))


def test_inference_applies_mean_scale_intercept_and_coefficients() -> None:
    """Non-trivial artifact: non-zero mean, non-unit scale, mixed-sign coefs,
    non-zero intercept. Catches implementations that skip standardization,
    drop the intercept, or misalign coefficients.

    Setup: only ``life_diff`` and ``push_diff`` weighted; state forces
    ``life_diff = 2`` for RED and ``push_diff = 0`` on a fresh setup, so
    ::

        logit = 0.25 + 2.0 * (2 - 0.5) / 2.0 = 1.75
        output = tanh(1.75 / 2) = tanh(0.875)
    """
    from automata.evaluation.learned_value import LearnedValue

    art = _canonical_artifact()
    art["feature_means"] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    art["feature_scales"] = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    art["coefficients"] = [2.0, -0.5, 0.0, 0.0, 0.0, 0.0]
    art["intercept"] = 0.25

    st = _bench_state()
    st.teams[TeamColor.RED].life_counters = 6
    st.teams[TeamColor.BLUE].life_counters = 4

    red_feats = feature_vector(st, TeamColor.RED)
    assert red_feats[FEATURE_NAMES.index("life_diff")] == 2.0
    assert red_feats[FEATURE_NAMES.index("push_diff")] == 0.0  # precondition

    lv = LearnedValue(art)
    assert lv(st, TeamColor.RED) == pytest.approx(math.tanh(0.875))


def test_inference_output_is_in_normalized_range() -> None:
    """Whatever the input state, the output must be a finite scalar in
    ``[-1, 1]`` — the ValueFn contract every implementation obeys.
    """
    from automata.evaluation.learned_value import LearnedValue

    lv = LearnedValue(_canonical_artifact())
    for seed in range(3):
        st = _bench_state(seed=seed)
        for team in (TeamColor.RED, TeamColor.BLUE):
            v = lv(st, team)
            assert math.isfinite(v)
            assert -1.0 <= v <= 1.0


# --------------------------------------------------------------------------- #
# Structural / numeric validation (schema is generic; trainer picks roster).
#
# Each parametrization patches ONE field of a valid artifact so the resulting
# failure isolates the specific rule under test. All errors must be
# ``ValueError`` — the artifact loader is the single validation seam.
# --------------------------------------------------------------------------- #


REQUIRED_FIELDS = (
    "model_version",
    "schema_version",
    "red_roster",
    "blue_roster",
    "feature_names",
    "feature_means",
    "feature_scales",
    "coefficients",
    "intercept",
)


def _mutate(patch: dict[str, Any]) -> dict[str, Any]:
    art = _canonical_artifact()
    art.update(patch)
    return art


def _drop(field: str) -> dict[str, Any]:
    art = _canonical_artifact()
    art.pop(field)
    return art


NON_FINITE = (float("inf"), float("-inf"), float("nan"))
BAD_SCALES = (0.0, -1.0, *NON_FINITE)


@pytest.mark.parametrize(
    "art",
    # Structural
    [_drop(f) for f in REQUIRED_FIELDS]
    + [
        # Feature-name ordering / membership
        _mutate({"feature_names": [FEATURE_NAMES[1], FEATURE_NAMES[0], *FEATURE_NAMES[2:]]}),
        _mutate(
            {
                "feature_names": list(FEATURE_NAMES[:-1]),
                "feature_means": [0.0] * (len(FEATURE_NAMES) - 1),
                "feature_scales": [1.0] * (len(FEATURE_NAMES) - 1),
                "coefficients": [0.0] * (len(FEATURE_NAMES) - 1),
            }
        ),
        # Vector length mismatches
        _mutate({"feature_means": [0.0] * (len(FEATURE_NAMES) + 1)}),
        _mutate({"feature_scales": [1.0] * (len(FEATURE_NAMES) + 1)}),
        _mutate({"coefficients": [0.0] * (len(FEATURE_NAMES) + 1)}),
        # Versioning
        _mutate({"model_version": "gbm-v1"}),
        _mutate({"schema_version": 999}),
    ]
    # Numeric — non-finite means / coefficients / intercept
    + [_mutate({"feature_means": [b] + [0.0] * (len(FEATURE_NAMES) - 1)}) for b in NON_FINITE]
    + [_mutate({"coefficients": [b] + [0.0] * (len(FEATURE_NAMES) - 1)}) for b in NON_FINITE]
    + [_mutate({"intercept": b}) for b in NON_FINITE]
    # Numeric — non-positive / non-finite scales
    + [_mutate({"feature_scales": [b] + [1.0] * (len(FEATURE_NAMES) - 1)}) for b in BAD_SCALES],
)
def test_malformed_artifact_raises_value_error(art: dict[str, Any]) -> None:
    """Every violation of the artifact schema — missing field, reordered
    features, wrong-length vector, wrong version, non-finite number,
    non-positive scale — surfaces as ``ValueError`` at load time. Silent
    acceptance is worse than a hard failure: it ships an artifact that
    scores nonsense.
    """
    from automata.evaluation.learned_value import LearnedValue

    with pytest.raises(ValueError):
        LearnedValue(art)


# --------------------------------------------------------------------------- #
# Roster compatibility at score time.
#
# The artifact schema is GENERIC over rosters — the trainer / Rung-2a
# generator picks the roster and stamps it into ``red_roster`` /
# ``blue_roster``. LearnedValue must accept any well-formed roster on load
# and only reject at call time when the *state's* rosters diverge from the
# ones declared in the artifact.
# --------------------------------------------------------------------------- #


def test_load_accepts_any_valid_declared_roster(tmp_path: Path) -> None:
    """The artifact declares its own rosters. LearnedValue must NOT hard-wire
    the benchmark matchup at load time — a different but well-formed roster
    (e.g. a future non-benchmark model) must load cleanly.
    """
    from automata.evaluation.learned_value import LearnedValue

    art = _mutate({"red_roster": ["Wasp", "Bain"], "blue_roster": ["Arien", "Dodger"]})
    lv = LearnedValue(art)  # must not raise

    # And a compatible state (matching the *declared* roster, not the
    # benchmark) must be scorable through it.
    st = _state(["Wasp", "Bain"], ["Arien", "Dodger"])
    v = lv(st, TeamColor.RED)
    assert math.isfinite(v) and -1.0 <= v <= 1.0


def test_state_matching_declared_roster_scores_cleanly() -> None:
    """Compatible state (matches the artifact's declared rosters) must NOT
    raise — guards against an over-eager compatibility check that rejects
    everything.
    """
    from automata.evaluation.learned_value import LearnedValue

    lv = LearnedValue(_canonical_artifact())
    lv(_bench_state(), TeamColor.RED)
    lv(_bench_state(), TeamColor.BLUE)


def test_state_with_wrong_roster_raises_at_call_time() -> None:
    """A well-loaded artifact must refuse to score a state whose rosters
    differ from the artifact's declared rosters (side-aware). Silent
    scoring would return an out-of-distribution value.
    """
    from automata.evaluation.learned_value import LearnedValue

    lv = LearnedValue(_canonical_artifact())
    with pytest.raises(ValueError):
        lv(_mismatched_state(), TeamColor.RED)


# --------------------------------------------------------------------------- #
# Content digest — model identity.
# --------------------------------------------------------------------------- #


def test_digest_is_stable_sha256_hex(tmp_path: Path) -> None:
    """The digest must be a deterministic 64-char lowercase hex SHA-256 that
    is independent of the load source (path vs. mapping) and stable across
    repeated loads of identical content — that's what makes it a valid
    evaluation-identity key.
    """
    from automata.evaluation.learned_value import LearnedValue

    art = _canonical_artifact()
    d_mem_1 = LearnedValue(art).digest
    d_mem_2 = LearnedValue(art).digest
    d_file = LearnedValue(_write_artifact(tmp_path, art)).digest

    assert isinstance(d_mem_1, str)
    assert len(d_mem_1) == 64
    assert all(c in "0123456789abcdef" for c in d_mem_1)
    assert d_mem_1 == d_mem_2 == d_file


@pytest.mark.parametrize(
    "patch",
    [
        {"intercept": 0.5},  # numeric field
        {"feature_scales": [2.0] + [1.0] * (len(FEATURE_NAMES) - 1)},  # vector field
        {"red_roster": ["Wasp", "Bain"]},  # categorical field
    ],
)
def test_digest_changes_when_any_content_field_changes(patch: dict[str, Any]) -> None:
    """Any content change must flip the digest. Parametrized over one
    numeric, one vector, and one categorical field so a hasher that only
    covers part of the artifact fails at least one case.
    """
    from automata.evaluation.learned_value import LearnedValue

    base = LearnedValue(_canonical_artifact()).digest
    assert LearnedValue(_mutate(patch)).digest != base
