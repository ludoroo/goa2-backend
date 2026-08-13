"""RED tests for the value dataset recorder + generator (Task 2).

Compact, deterministic per-decision training rows: schema_version, game_id,
world_seed, acting team, ordered features aligned with
:data:`automata.evaluation.features.FEATURE_NAMES`, fixed benchmark rosters,
source revision + dirty-tree hash, and the game's ``winner`` label. Rows are
buffered until ``game_over``; ``max_steps`` games and unknown-team decisions
are dropped. Generator CLI runs Heuristic self-play over an explicit seed
range via ``main(argv)``.

Tests exercise only the stable public surface:

- ``automata.evaluation.value_dataset.SCHEMA_VERSION``
- ``automata.evaluation.value_dataset.ValueDatasetRecorder``
- ``automata.evaluation.value_dataset.DatasetStats``
- ``automata.evaluation.value_dataset.load_examples``
- ``automata.scripts.generate_value_data.main``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.evaluation.features import FEATURE_NAMES, feature_vector
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP, RunResult
from goa2.domain.models import TeamColor
from goa2.engine.setup import GameSetup

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]
RED_HERO_IDS = {"hero_wasp", "hero_xargatha"}
BLUE_HERO_IDS = {"hero_arien", "hero_brogan"}


@pytest.fixture(scope="module", autouse=True)
def _effects_registered() -> None:
    register_all_effects()


def _state(seed: int = 0):
    return GameSetup.create_game(
        map_path=DEFAULT_MAP,
        red_heroes=RED,
        blue_heroes=BLUE,
        game_type="QUICK",
        seed=seed,
    )


def _identity(**overrides: Any) -> dict[str, Any]:
    """Standard identity kwargs for the recorder; override per test."""
    kw: dict[str, Any] = {
        "game_id": "g0",
        "world_seed": 0,
        "red_heroes": RED,
        "blue_heroes": BLUE,
        "source_revision": "rev-testfake",
        "dirty_tree_hash": "clean",
    }
    kw.update(overrides)
    return kw


def _decide(rec, team: str = "RED", **overrides: Any) -> None:
    """Record one decision; default is a legal RED CARD choice."""
    payload: dict[str, Any] = {
        "state": overrides.pop("state", None) or _state(),
        "team": team,
        "decision_kind": "CARD",
        "player_id": "hero_wasp",
        "legal_keys": ["a"],
        "chosen_key": "a",
    }
    payload.update(overrides)
    rec.record_decision(**payload)


# --------------------------------------------------------------------------- #
# Schema + row shape
# --------------------------------------------------------------------------- #


def test_schema_version_is_exposed_as_positive_integer() -> None:
    from automata.evaluation import value_dataset

    assert isinstance(value_dataset.SCHEMA_VERSION, int)
    assert value_dataset.SCHEMA_VERSION >= 1


def test_recorder_emits_one_row_per_recorded_decision(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import SCHEMA_VERSION, ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    with ValueDatasetRecorder(path, **_identity()) as rec:
        _decide(rec, team="RED")
        _decide(rec, team="BLUE", player_id="hero_arien")
        rec.record_outcome(winner="RED", rounds=7, reason="game_over")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    required = {
        "schema_version",
        "game_id",
        "world_seed",
        "team",
        "features",
        "winner",
        "red_heroes",
        "blue_heroes",
        "source_revision",
        "dirty_tree_hash",
    }
    for row in rows:
        assert required <= row.keys(), f"missing keys: {required - row.keys()}"
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["game_id"] == "g0"
        assert row["world_seed"] == 0
        assert row["red_heroes"] == RED
        assert row["blue_heroes"] == BLUE
        assert row["winner"] == "RED"
        assert row["source_revision"] == "rev-testfake"
        assert row["dirty_tree_hash"] == "clean"


def test_features_are_ordered_and_aligned_with_feature_names(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    state = _state(seed=1)
    expected = feature_vector(state, TeamColor.RED)
    with ValueDatasetRecorder(path, **_identity(world_seed=1)) as rec:
        _decide(rec, state=state, team="RED")
        rec.record_outcome(winner="BLUE", rounds=5, reason="game_over")

    row = json.loads(path.read_text().splitlines()[0])
    assert isinstance(row["features"], list)
    assert len(row["features"]) == len(FEATURE_NAMES)
    assert all(isinstance(v, (int, float)) for v in row["features"])
    assert row["features"] == pytest.approx(expected)


def test_row_omits_full_gamestate_snapshot(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    with ValueDatasetRecorder(path, **_identity()) as rec:
        _decide(rec)
        rec.record_outcome(winner="RED", rounds=3, reason="game_over")

    row = json.loads(path.read_text().splitlines()[0])
    for leaked in ("state", "board", "teams", "execution_stack"):
        assert leaked not in row


def test_recorder_ignores_selection_shape_of_legal_and_chosen_keys(tmp_path: Path) -> None:
    """``legal_keys`` / ``chosen_key`` carry engine-side selection shapes
    (integer, hex dict, ``"SKIP"`` string, plain id). The value dataset does
    not train on the action space, so the recorder must accept every shape
    without raising and must not persist them on the row."""
    from automata.evaluation.value_dataset import ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    with ValueDatasetRecorder(path, **_identity()) as rec:
        _decide(rec, team="RED", legal_keys=[1, 2, 3], chosen_key=2)
        _decide(
            rec,
            team="BLUE",
            player_id="hero_arien",
            legal_keys=[{"q": 0, "r": 0, "s": 0}, {"q": 1, "r": -1, "s": 0}],
            chosen_key={"q": 1, "r": -1, "s": 0},
        )
        _decide(rec, team="RED", legal_keys=["A", "SKIP"], chosen_key="SKIP")
        rec.record_outcome(winner="RED", rounds=4, reason="game_over")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    # Row does not carry the action-space payload; it is a value example, not
    # a policy example.
    for row in rows:
        assert "legal_keys" not in row
        assert "chosen_key" not in row


# --------------------------------------------------------------------------- #
# Buffering + completion semantics
# --------------------------------------------------------------------------- #


def test_incomplete_max_steps_game_is_discarded(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    with ValueDatasetRecorder(path, **_identity(game_id="incomplete")) as rec:
        _decide(rec)
        rec.record_outcome(winner=None, rounds=42, reason="max_steps")

    text = path.read_text() if path.exists() else ""
    assert [ln for ln in text.splitlines() if ln.strip()] == []


def test_decision_with_unknown_or_empty_team_is_skipped(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    with ValueDatasetRecorder(path, **_identity()) as rec:
        _decide(rec, team="", player_id="team:???")  # dropped
        _decide(rec, team="RED")  # kept
        rec.record_outcome(winner="RED", rounds=2, reason="game_over")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["team"] == "RED"


def test_buffered_decisions_do_not_stream_before_outcome(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    rec = ValueDatasetRecorder(path, **_identity())
    _decide(rec)
    if path.exists():
        assert [ln for ln in path.read_text().splitlines() if ln.strip()] == []
    rec.record_outcome(winner="RED", rounds=2, reason="game_over")
    assert len([ln for ln in path.read_text().splitlines() if ln.strip()]) == 1


# --------------------------------------------------------------------------- #
# Stats + multi-game + determinism
# --------------------------------------------------------------------------- #


def test_stats_report_recorded_and_skipped_counts(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import DatasetStats, ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    rec = ValueDatasetRecorder(path, **_identity())
    _decide(rec, team="RED")
    _decide(rec, team="BLUE", player_id="hero_arien")
    _decide(rec, team="", player_id="team:???")
    rec.record_outcome(winner="RED", rounds=4, reason="game_over")

    stats = rec.stats
    assert isinstance(stats, DatasetStats)
    assert stats.recorded_decisions == 2
    assert stats.skipped_decisions == 1
    assert stats.recorded_games == 1
    assert stats.skipped_games == 0


def test_stats_track_skipped_incomplete_games(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    rec = ValueDatasetRecorder(path, **_identity(game_id="g_incomplete"))
    _decide(rec)
    rec.record_outcome(winner=None, rounds=10, reason="max_steps")

    assert rec.stats.recorded_games == 0
    assert rec.stats.skipped_games == 1
    assert rec.stats.recorded_decisions == 0


def test_load_examples_roundtrips_written_rows(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import ValueDatasetRecorder, load_examples

    path = tmp_path / "ds.jsonl"
    with ValueDatasetRecorder(path, **_identity()) as rec:
        _decide(rec)
        rec.record_outcome(winner="RED", rounds=3, reason="game_over")

    examples = load_examples(path)
    assert len(examples) == 1
    ex = examples[0]
    # Whether returned as dataclass or dict, the public shape has these fields.
    features = ex["features"] if isinstance(ex, dict) else ex.features
    winner = ex["winner"] if isinstance(ex, dict) else ex.winner
    team = ex["team"] if isinstance(ex, dict) else ex.team
    assert len(features) == len(FEATURE_NAMES)
    assert winner == "RED"
    assert team == "RED"


def test_two_games_write_distinct_game_ids(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import ValueDatasetRecorder

    path = tmp_path / "ds.jsonl"
    for gid, seed in (("g0", 9), ("g1", 10)):
        with ValueDatasetRecorder(
            path, **_identity(game_id=gid, world_seed=seed), append=True
        ) as rec:
            _decide(rec)
            rec.record_outcome(winner="RED", rounds=2, reason="game_over")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert {r["game_id"] for r in rows} == {"g0", "g1"}
    assert {r["world_seed"] for r in rows} == {9, 10}


def test_recording_is_deterministic_across_two_runs(tmp_path: Path) -> None:
    from automata.evaluation.value_dataset import ValueDatasetRecorder

    def _write(target: Path) -> None:
        with ValueDatasetRecorder(target, **_identity(game_id="g11", world_seed=11)) as rec:
            _decide(rec, team="RED")
            _decide(rec, team="BLUE", player_id="hero_arien")
            rec.record_outcome(winner="RED", rounds=2, reason="game_over")

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a)
    _write(b)
    assert a.read_bytes() == b.read_bytes()


# --------------------------------------------------------------------------- #
# CLI: `generate_value_data.main(argv)` with a monkeypatched harness
# --------------------------------------------------------------------------- #


def _install_fake_run_game(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch ``run_game`` on the generator module so no real self-play runs."""
    from automata.scripts import generate_value_data as gen

    calls: list[dict[str, Any]] = []

    def _fake(
        red_heroes,
        blue_heroes,
        agents,
        *,
        map_path=None,
        game_type="QUICK",
        seed=0,
        max_steps=20_000,
        recorder=None,
    ):
        calls.append(
            {
                "red_heroes": list(red_heroes),
                "blue_heroes": list(blue_heroes),
                "agents": dict(agents),
                "seed": seed,
            }
        )
        if recorder is not None:
            state = _state(seed=seed)
            recorder.record_decision(
                state=state,
                team="RED",
                decision_kind="CARD",
                player_id="hero_wasp",
                legal_keys=["a"],
                chosen_key="a",
            )
            recorder.record_outcome(winner="RED", rounds=1, reason="game_over")
        return RunResult(winner="RED", rounds=1, turns=1, steps=1, reason="game_over")

    monkeypatch.setattr(gen, "run_game", _fake, raising=True)
    return calls


def test_generate_cli_main_uses_benchmark_rosters_and_heuristic_selfplay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generator CLI is Heuristic self-play on the benchmark roster:
    RED = Wasp+Xargatha, BLUE = Arien+Brogan, and every mapped agent is a
    :class:`HeuristicAgent` (both sides — self-play, not mixed)."""
    from automata.scripts import generate_value_data as gen

    calls = _install_fake_run_game(monkeypatch)

    out = tmp_path / "ds.jsonl"
    rc = gen.main(["--out", str(out), "--seed-start", "0", "--seed-end", "3"])
    assert rc in (None, 0)

    assert [c["seed"] for c in calls] == [0, 1, 2]
    for call in calls:
        assert call["red_heroes"] == RED
        assert call["blue_heroes"] == BLUE
        # Agent coverage: every benchmark hero mapped, and only those.
        assert set(call["agents"].keys()) == RED_HERO_IDS | BLUE_HERO_IDS
        # Self-play: every mapped agent is a HeuristicAgent. Insensitive to
        # instance identity (either one shared agent or per-hero instances).
        for agent in call["agents"].values():
            assert isinstance(agent, HeuristicAgent)

    assert out.exists()
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    for row in rows:
        assert row["red_heroes"] == RED
        assert row["blue_heroes"] == BLUE


def test_generate_cli_uses_explicit_seed_range_not_hidden_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--seed-start`` / ``--seed-end`` are used verbatim as a contiguous
    ``[start, end)`` range. No hidden skipping of "reserved" eval seeds."""
    from automata.scripts import generate_value_data as gen

    calls = _install_fake_run_game(monkeypatch)

    out = tmp_path / "ds.jsonl"
    gen.main(["--out", str(out), "--seed-start", "1000", "--seed-end", "1004"])
    assert [c["seed"] for c in calls] == [1000, 1001, 1002, 1003]


# --------------------------------------------------------------------------- #
# CLI: output-file lifecycle
# --------------------------------------------------------------------------- #


def _install_scriptable_run_game(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Patch ``run_game`` so each seed's recorder behavior is scriptable.

    ``outcomes[seed]`` is a dict with keys:

    - ``"decisions"`` (int) — number of RED CARD decisions to record.
    - ``"reason"`` (str) — ``"game_over"`` (flushes) or e.g. ``"max_steps"``
      (drops the buffer).
    - ``"winner"`` (str | None) — passed through to ``record_outcome``.

    Every call captures ``red_heroes``/``blue_heroes``/``seed``/``agents``
    so tests can inspect per-seed agent identity and RNG state without
    running a real game.
    """
    from automata.scripts import generate_value_data as gen

    calls: list[dict[str, Any]] = []

    def _fake(
        red_heroes: Any,
        blue_heroes: Any,
        agents: Any,
        *,
        map_path: Any = None,
        game_type: str = "QUICK",
        seed: int = 0,
        max_steps: int = 20_000,
        recorder: Any = None,
    ) -> RunResult:
        calls.append(
            {
                "red_heroes": list(red_heroes),
                "blue_heroes": list(blue_heroes),
                "agents": dict(agents),
                "seed": seed,
            }
        )
        spec = outcomes.get(seed, {"decisions": 1, "reason": "game_over", "winner": "RED"})
        if recorder is not None:
            state = _state(seed=seed)
            for _ in range(int(spec.get("decisions", 0))):
                recorder.record_decision(
                    state=state,
                    team="RED",
                    decision_kind="CARD",
                    player_id="hero_wasp",
                    legal_keys=["a"],
                    chosen_key="a",
                )
            recorder.record_outcome(
                winner=spec.get("winner"),
                rounds=1,
                reason=spec.get("reason", "game_over"),
            )
        return RunResult(
            winner=spec.get("winner"),
            rounds=1,
            turns=1,
            steps=1,
            reason=spec.get("reason", "game_over"),
        )

    monkeypatch.setattr(gen, "run_game", _fake, raising=True)
    return calls


def test_generate_cli_truncates_stale_output_even_when_no_games_produce_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale dataset on ``--out`` must NOT survive a new run, even when
    every game in the new range fails to produce rows (all ``max_steps``,
    unknown-team-only, etc.). The CLI must truncate ``--out`` up front
    rather than relying on the first recorder's flush."""
    from automata.scripts import generate_value_data as gen

    out = tmp_path / "ds.jsonl"
    # Pre-existing dataset from a previous run.
    stale = '{"schema_version":1,"game_id":"stale","team":"RED"}\n'
    out.write_text(stale, encoding="utf-8")

    # Every game hits max_steps → no rows flushed.
    _install_scriptable_run_game(
        monkeypatch,
        {
            0: {"decisions": 1, "reason": "max_steps", "winner": None},
            1: {"decisions": 1, "reason": "max_steps", "winner": None},
            2: {"decisions": 1, "reason": "max_steps", "winner": None},
        },
    )
    gen.main(["--out", str(out), "--seed-start", "0", "--seed-end", "3"])

    # Stale row is gone; file exists and is empty.
    assert out.exists()
    assert out.read_bytes() == b""


def test_generate_cli_preserves_later_games_when_early_ones_produce_no_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: with a preexisting output, seed 0 producing no rows
    (``max_steps``) and seed 1 completing normally, the final file must
    contain seed 1's rows only — no stale rows, and seed 0 must not
    prevent seed 1 from being written."""
    from automata.scripts import generate_value_data as gen

    out = tmp_path / "ds.jsonl"
    out.write_text('{"stale":true}\n', encoding="utf-8")

    _install_scriptable_run_game(
        monkeypatch,
        {
            0: {"decisions": 1, "reason": "max_steps", "winner": None},
            1: {"decisions": 2, "reason": "game_over", "winner": "RED"},
        },
    )
    gen.main(["--out", str(out), "--seed-start", "0", "--seed-end", "2"])

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    # Two rows from seed 1; no stale row survived.
    assert len(rows) == 2
    assert {r["world_seed"] for r in rows} == {1}
    assert all(r["winner"] == "RED" for r in rows)
    assert all("stale" not in r for r in rows)


# --------------------------------------------------------------------------- #
# CLI: per-seed agent construction
# --------------------------------------------------------------------------- #


def test_generate_cli_builds_fresh_agents_per_world_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every world seed must get a *fresh* HeuristicAgent instance — no
    single agent may be reused across games. Sharing an agent across games
    threads its tie-break RNG state through the whole run and breaks the
    concatenation property tested below."""
    from automata.scripts import generate_value_data as gen

    calls = _install_scriptable_run_game(
        monkeypatch,
        {seed: {"decisions": 1, "reason": "game_over", "winner": "RED"} for seed in range(4)},
    )
    out = tmp_path / "ds.jsonl"
    gen.main(["--out", str(out), "--seed-start", "0", "--seed-end", "4"])

    # Collect every HeuristicAgent instance passed into run_game across all
    # games. Fresh-per-seed means distinct object identities per game.
    per_game_ids: list[set[int]] = []
    for call in calls:
        agents_in_call = {id(a) for a in call["agents"].values()}
        per_game_ids.append(agents_in_call)
    # No agent id from game N is reused in game M != N.
    for i, ids_i in enumerate(per_game_ids):
        for j, ids_j in enumerate(per_game_ids):
            if i == j:
                continue
            assert ids_i.isdisjoint(
                ids_j
            ), f"HeuristicAgent instance reused between games {i} and {j}"


def test_generate_cli_splits_red_and_blue_agent_rngs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED and BLUE must get independent HeuristicAgent instances with
    distinct RNG seeds — otherwise their tie-break streams collide and the
    two sides of the same game share entropy."""
    from automata.scripts import generate_value_data as gen

    calls = _install_scriptable_run_game(
        monkeypatch,
        {0: {"decisions": 1, "reason": "game_over", "winner": "RED"}},
    )
    out = tmp_path / "ds.jsonl"
    gen.main(["--out", str(out), "--seed-start", "0", "--seed-end", "1"])
    assert len(calls) == 1
    agents = calls[0]["agents"]
    red_ids = {id(agents["hero_wasp"]), id(agents["hero_xargatha"])}
    blue_ids = {id(agents["hero_arien"]), id(agents["hero_brogan"])}
    # RED and BLUE agents are distinct instances.
    assert red_ids.isdisjoint(
        blue_ids
    ), "RED and BLUE share a HeuristicAgent instance — RNGs will collide"
    # Each side runs one shared agent across its heroes (the natural
    # "one policy per team" mapping) — reads as a single object identity.
    assert len(red_ids) == 1
    assert len(blue_ids) == 1


def test_generate_cli_agent_seed_is_pure_function_of_world_seed_and_side() -> None:
    """The per-agent seed must depend solely on ``(world_seed, side)`` —
    no per-game counter, no cross-run state. This is what makes the
    concatenation property below hold."""
    from automata.scripts.generate_value_data import agent_seed

    # Pure function: repeated calls return the same value.
    assert agent_seed(7, "RED") == agent_seed(7, "RED")
    # Side splits the stream.
    assert agent_seed(7, "RED") != agent_seed(7, "BLUE")
    # World seed splits the stream.
    assert agent_seed(7, "RED") != agent_seed(8, "RED")


def test_generate_cli_range_equals_concatenation_of_subranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Byte-identical concatenation property: generating ``[0, 6)`` in one
    call must equal concatenating ``[0, 3)`` and ``[3, 6)`` from separate
    calls. This holds only if per-game agent state is fully reconstructed
    from ``(world_seed, side)`` — no RNG carryover from prior games and
    no shared mutable agent instances."""
    from automata.scripts import generate_value_data as gen

    outcomes = {seed: {"decisions": 2, "reason": "game_over", "winner": "RED"} for seed in range(6)}

    # Full range in one run.
    _install_scriptable_run_game(monkeypatch, outcomes)
    full = tmp_path / "full.jsonl"
    gen.main(["--out", str(full), "--seed-start", "0", "--seed-end", "6"])

    # Two subranges written to separate files, then concatenated.
    _install_scriptable_run_game(monkeypatch, outcomes)
    part_a = tmp_path / "part_a.jsonl"
    gen.main(["--out", str(part_a), "--seed-start", "0", "--seed-end", "3"])
    _install_scriptable_run_game(monkeypatch, outcomes)
    part_b = tmp_path / "part_b.jsonl"
    gen.main(["--out", str(part_b), "--seed-start", "3", "--seed-end", "6"])

    assert full.read_bytes() == part_a.read_bytes() + part_b.read_bytes()
