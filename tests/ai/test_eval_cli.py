"""RED tests for the targeted / resumable evaluation CLI (Rung 1).

Stable boundary: :func:`automata.evaluation.cli.main` invoked with an explicit
``argv``. Narrow semi-public seams — :data:`cli.source_identity`,
:func:`cli.build_case_runner`, :func:`cli.build_agent`, and the module-level
:data:`cli.run_game` / :data:`cli.run_protocol` / :data:`cli.summarize`
re-exports — exist so tests can drive integration without paying for a real
game or ISMCTS search.

The scaffolds raise :class:`NotImplementedError` today, so every targeted test
here is behaviorally RED. Matrix backward-compat tests keep passing.
"""

from __future__ import annotations

import json
import pickle
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from automata.evaluation import cli as cli_module
from automata.evaluation.features import FEATURE_NAMES
from automata.evaluation.learned_value import LearnedValue
from automata.evaluation.matchup import MatchupResult
from automata.evaluation.protocol import (
    AgentSpec,
    EvaluationProtocol,
    EvaluationSummary,
    GameCase,
    GameObservation,
)
from automata.runtime.harness import RunResult

RED_HEROES = ("Wasp", "Xargatha")
BLUE_HEROES = ("Arien", "Brogan")


@pytest.fixture()
def fake_source_identity(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    ident = ("rev-testfake", "clean")
    monkeypatch.setattr(cli_module, "source_identity", lambda repo_root=None: ident)
    return ident


@dataclass
class _CapturedRun:
    protocol: EvaluationProtocol | None = None
    checkpoint_path: Path | None = None
    observations: list[GameObservation] | None = None


def _install_run_protocol(
    monkeypatch: pytest.MonkeyPatch,
    observations_fn: Callable[[EvaluationProtocol], list[GameObservation]] | None = None,
) -> _CapturedRun:
    captured = _CapturedRun()

    def _fake(protocol: EvaluationProtocol, *, checkpoint_path: Path, run_case):
        captured.protocol = protocol
        captured.checkpoint_path = Path(checkpoint_path)
        obs = observations_fn(protocol) if observations_fn else []
        captured.observations = obs
        return obs

    monkeypatch.setattr(cli_module, "run_protocol", _fake)
    monkeypatch.setattr(cli_module, "summarize", lambda obs: EvaluationSummary())
    return captured


@pytest.fixture()
def fake_run_protocol(monkeypatch: pytest.MonkeyPatch) -> _CapturedRun:
    return _install_run_protocol(monkeypatch)


@pytest.fixture()
def fake_matrix_evaluate(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Neutralize the legacy matrix path so untargeted runs never play games."""
    calls: list[dict[str, Any]] = []

    def _fake(a_factory, b_factory, *, label_a: str, label_b: str, games: int, **_: Any):
        calls.append({"label_a": label_a, "label_b": label_b, "games": games})
        return MatchupResult(
            label_a=label_a,
            label_b=label_b,
            games=games,
            a_wins=games,
            b_wins=0,
            draws=0,
            avg_rounds=10.0,
        )

    monkeypatch.setattr(cli_module, "evaluate", _fake)
    return calls


@pytest.mark.parametrize(
    "argv_str",
    [
        "--agent-a random",
        "--agent-b heuristic",
    ],
)
def test_parser_rejects_unpaired_agent_flag(
    fake_source_identity: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    argv_str: str,
) -> None:
    """One-agent-only invocations must fail with a message naming both flags,
    not the generic argparse "unrecognized arguments" error."""
    with pytest.raises(SystemExit):
        cli_module.main(argv_str.split())
    err = capsys.readouterr().err
    assert (
        "both --agent-a and --agent-b" in err
    ), f"expected a targeted-pairing validation message; got stderr:\n{err}"


@pytest.mark.parametrize(
    "argv_str",
    [
        "--agent-a random --agent-b heuristic --paired-seeds 0",
        "--agent-a ismcts --agent-b heuristic --a-uct-c 0",
    ],
)
def test_parser_rejects_non_positive_numeric_flag(
    fake_source_identity: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    argv_str: str,
) -> None:
    """Zero / negative numeric knobs must fail with a message flagging the
    ``positive`` constraint, not the generic argparse "unrecognized" error."""
    with pytest.raises(SystemExit):
        cli_module.main(argv_str.split())
    err = capsys.readouterr().err
    assert "positive" in err, f"expected a 'positive'-value validation message; got stderr:\n{err}"


def test_targeted_defaults_produce_canonical_screen(
    fake_source_identity: tuple[str, str],
    fake_run_protocol: _CapturedRun,
) -> None:
    """Canonical screen defaults: 6 paired seeds → 12 paired cases; ISMCTS
    defaults are iterations=4 and cutoff_rounds=1 in targeted mode."""
    cli_module.main(["--agent-a", "ismcts", "--agent-b", "ismcts"])

    proto = fake_run_protocol.protocol
    assert proto is not None
    cases = list(proto.cases())
    assert len(cases) == 12
    assert {c.world_seed for c in cases} == {0, 1, 2, 3, 4, 5}
    assert {c.a_side for c in cases} == {"RED", "BLUE"}

    assert proto.agent_a.kind == "ismcts"
    assert proto.agent_a.params.get("iterations") == 4
    assert proto.agent_a.params.get("cutoff_rounds") == 1
    assert proto.agent_b.params.get("iterations") == 4
    assert proto.agent_b.params.get("cutoff_rounds") == 1


@pytest.mark.parametrize(
    "flags",
    [
        ["--a-iterations", "8"],
        ["--b-iterations", "16"],
        ["--a-cutoff-rounds", "2"],
        ["--b-cutoff-rounds", "3"],
        ["--a-uct-c", "0.7"],
        ["--b-uct-c", "1.8"],
        ["--a-puct-c", "0.9"],
        ["--b-puct-c", "1.5"],
        ["--a-no-prior"],
    ],
)
def test_effective_search_settings_participate_in_protocol_identity(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
) -> None:
    """Behavioral identity: changing any effective search setting must
    change the resulting protocol's identity digest. This is stronger than
    asserting a specific ``AgentSpec.params`` field-set because it names the
    user-visible contract (identity/checkpoint) rather than an internal dict.
    Dynamic per-case seed is NOT a CLI flag, so it cannot leak into identity.
    """
    base = _install_run_protocol(monkeypatch)
    cli_module.main(["--agent-a", "ismcts", "--agent-b", "heuristic"])
    baseline_digest = base.protocol.identity_digest() if base.protocol else None
    assert baseline_digest is not None

    tweaked = _install_run_protocol(monkeypatch)
    cli_module.main(["--agent-a", "ismcts", "--agent-b", "ismcts", *flags])
    tweaked_digest = tweaked.protocol.identity_digest() if tweaked.protocol else None
    assert tweaked_digest is not None
    assert (
        tweaked_digest != baseline_digest
    ), f"changing {flags!r} must invalidate the checkpoint identity"


def test_source_identity_and_default_checkpoint_layout(
    fake_run_protocol: _CapturedRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module, "source_identity", lambda repo_root=None: ("rev-CAFEBABE", "deadbeef")
    )
    cli_module.main(["--agent-a", "random", "--agent-b", "heuristic"])

    proto = fake_run_protocol.protocol
    ckpt = fake_run_protocol.checkpoint_path
    assert proto is not None and ckpt is not None
    assert proto.source_revision == "rev-CAFEBABE"
    assert proto.dirty_tree_hash == "deadbeef"
    assert ckpt.name == f"{proto.identity_digest()}.jsonl"


def test_checkpoint_flag_overrides_and_seed_count_is_schedule(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--checkpoint overrides the default path; extending --paired-seeds
    reuses the same default identity (seed count is schedule, not identity)."""
    override = tmp_path / "override.jsonl"
    overridden = _install_run_protocol(monkeypatch)
    cli_module.main(
        [
            "--agent-a",
            "random",
            "--agent-b",
            "heuristic",
            "--checkpoint",
            str(override),
        ]
    )
    assert overridden.checkpoint_path == override

    small = _install_run_protocol(monkeypatch)
    cli_module.main(["--agent-a", "random", "--agent-b", "heuristic", "--paired-seeds", "6"])
    big = _install_run_protocol(monkeypatch)
    cli_module.main(["--agent-a", "random", "--agent-b", "heuristic", "--paired-seeds", "12"])
    assert (
        small.checkpoint_path == big.checkpoint_path
    ), "extending --paired-seeds must reuse the same default checkpoint identity"


def _make_protocol(**overrides: Any) -> EvaluationProtocol:
    base: dict[str, Any] = dict(
        agent_a=AgentSpec(name="a", kind="random", params={}),
        agent_b=AgentSpec(name="b", kind="heuristic", params={}),
        red_heroes=RED_HEROES,
        blue_heroes=BLUE_HEROES,
        world_seeds=(0, 1),
        map_path="src/goa2/data/maps/forgotten_island.json",
        game_type="QUICK",
        max_steps=12345,
        source_revision="rev-abc",
        dirty_tree_hash="clean",
    )
    base.update(overrides)
    return EvaluationProtocol(**base)


class _StubAgent:
    def __init__(self, spec: AgentSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed


def _install_case_runner_fakes(
    monkeypatch: pytest.MonkeyPatch,
    run_result: RunResult | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[AgentSpec, int]]]:
    run_calls: list[dict[str, Any]] = []
    agent_calls: list[tuple[AgentSpec, int]] = []
    result = run_result or RunResult(
        winner="RED", rounds=11, turns=44, steps=321, reason="game_over"
    )

    def _fake_run_game(red_heroes, blue_heroes, agents, **kwargs: Any) -> RunResult:
        run_calls.append(
            {
                "red_heroes": list(red_heroes),
                "blue_heroes": list(blue_heroes),
                "agents": dict(agents),
                **kwargs,
            }
        )
        return result

    def _fake_build_agent(
        spec: AgentSpec, seed: int, *, case_metadata: dict[str, Any] | None = None
    ) -> Any:
        agent_calls.append((spec, seed))
        return _StubAgent(spec, seed)

    monkeypatch.setattr(cli_module, "run_game", _fake_run_game)
    monkeypatch.setattr(cli_module, "build_agent", _fake_build_agent)
    return run_calls, agent_calls


def test_case_runner_dispatches_run_game_with_correct_config_and_side_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One case-runner test covering: run_game kwargs from protocol config,
    A/B side mapping for RED and BLUE cases, deterministic per-case seeds."""
    run_calls, agent_calls = _install_case_runner_fakes(monkeypatch)
    protocol = _make_protocol(world_seeds=(7,))
    runner = cli_module.build_case_runner(protocol)

    cases = {c.a_side: c for c in protocol.cases()}
    red_ids = {f"hero_{n.lower()}" for n in RED_HEROES}
    blue_ids = {f"hero_{n.lower()}" for n in BLUE_HEROES}

    runner(cases["RED"])
    red_call = run_calls[-1]
    assert red_call["map_path"] == protocol.map_path
    assert red_call["game_type"] == protocol.game_type
    assert red_call["max_steps"] == protocol.max_steps
    assert red_call["seed"] == cases["RED"].world_seed
    for hid in red_ids:
        assert red_call["agents"][hid].spec == protocol.agent_a
    for hid in blue_ids:
        assert red_call["agents"][hid].spec == protocol.agent_b

    runner(cases["BLUE"])
    blue_call = run_calls[-1]
    for hid in red_ids:
        assert blue_call["agents"][hid].spec == protocol.agent_b
    for hid in blue_ids:
        assert blue_call["agents"][hid].spec == protocol.agent_a

    # Deterministic: rerunning the same case yields the same agent-seed set.
    first_seeds = sorted(s for _, s in agent_calls)
    agent_calls.clear()
    runner(cases["RED"])
    runner(cases["BLUE"])
    replay_seeds = sorted(s for _, s in agent_calls)
    assert replay_seeds == first_seeds

    # Different world seed must produce a different seed vector.
    agent_calls.clear()
    other_proto = _make_protocol(world_seeds=(9999,))
    other_case = next(c for c in other_proto.cases() if c.a_side == "RED")
    cli_module.build_case_runner(other_proto)(other_case)
    assert sorted(s for _, s in agent_calls) != first_seeds


@pytest.mark.parametrize(
    "run_result,expected_winner,expected_reason",
    [
        (
            RunResult(winner="RED", rounds=10, turns=40, steps=500, reason="game_over"),
            "RED",
            "game_over",
        ),
        (
            RunResult(winner=None, rounds=99, turns=99, steps=20_000, reason="max_steps"),
            None,
            "max_steps",
        ),
    ],
)
def test_case_runner_translates_run_game_result_into_observation(
    monkeypatch: pytest.MonkeyPatch,
    run_result: RunResult,
    expected_winner: str | None,
    expected_reason: str,
) -> None:
    _install_case_runner_fakes(monkeypatch, run_result=run_result)
    protocol = _make_protocol()
    runner = cli_module.build_case_runner(protocol)
    case = next(iter(protocol.cases()))

    obs = runner(case)
    assert isinstance(obs, GameObservation)
    assert obs.case_id == case.case_id
    assert obs.world_seed == case.world_seed
    assert obs.a_side == case.a_side
    assert obs.winner_side == expected_winner
    assert obs.rounds == run_result.rounds
    assert obs.steps == run_result.steps
    assert obs.reason == expected_reason


def _obs(case: GameCase, winner: str | None, reason: str = "game_over") -> GameObservation:
    return GameObservation(
        case_id=case.case_id,
        world_seed=case.world_seed,
        a_side=case.a_side,
        winner_side=winner,
        rounds=8,
        steps=200,
        reason=reason,
    )


def _dominant_observations(protocol: EvaluationProtocol) -> list[GameObservation]:
    return [_obs(c, c.a_side) for c in protocol.cases()]


def _marginal_observations(protocol: EvaluationProtocol) -> list[GameObservation]:
    """7 A-wins, 4 B-wins, 1 max_steps → both gates FAIL."""
    out: list[GameObservation] = []
    for i, c in enumerate(protocol.cases()):
        if i < 7:
            out.append(_obs(c, c.a_side))
        elif i < 11:
            out.append(_obs(c, "BLUE" if c.a_side == "RED" else "RED"))
        else:
            out.append(_obs(c, None, reason="max_steps"))
    return out


@pytest.mark.parametrize(
    "obs_fn,expected_verdicts",
    [
        (_dominant_observations, {"PASS"}),
        (_marginal_observations, {"FAIL"}),
    ],
)
def test_targeted_output_reports_checkpoint_and_gate_verdicts(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    obs_fn: Callable[[EvaluationProtocol], list[GameObservation]],
    expected_verdicts: set[str],
) -> None:
    _install_run_protocol(monkeypatch, observations_fn=obs_fn)
    # Restore real summarize so gate verdicts reflect the observations.
    from automata.evaluation import protocol as protocol_module

    monkeypatch.setattr(cli_module, "summarize", protocol_module.summarize)

    override = tmp_path / "target.jsonl"
    cli_module.main(f"--agent-a ismcts --agent-b heuristic --checkpoint {override}".split())
    out = capsys.readouterr().out

    assert str(override) in out
    assert "SCREEN" in out and "PROMOTION" in out
    for verdict in expected_verdicts:
        assert verdict in out


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "init", "-q", str(root)])
    for k, v in [
        ("user.email", "t@example.com"),
        ("user.name", "Test"),
        ("commit.gpgsign", "false"),
    ]:
        subprocess.check_call(["git", "-C", str(root), "config", k, v])
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(root), "add", "seed.txt"])
    subprocess.check_call(["git", "-C", str(root), "commit", "-q", "-m", "seed"])


def test_source_identity_clean_tree_returns_head_and_stable_hash(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()

    rev_a, dirty_a = cli_module.source_identity(repo_root=repo)
    rev_b, dirty_b = cli_module.source_identity(repo_root=repo)
    assert rev_a == head == rev_b
    assert dirty_a == dirty_b  # deterministic on clean tree


def test_source_identity_dirty_hash_reflects_tracked_and_untracked_content(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _, clean = cli_module.source_identity(repo_root=repo)

    # Tracked-file diff → dirty differs.
    (repo / "seed.txt").write_text("seed changed\n", encoding="utf-8")
    _, dirty_tracked = cli_module.source_identity(repo_root=repo)
    assert dirty_tracked != clean

    # Add an untracked file → hash changes again.
    (repo / "new.txt").write_text("first\n", encoding="utf-8")
    _, dirty_untracked_a = cli_module.source_identity(repo_root=repo)
    assert dirty_untracked_a != dirty_tracked
    assert dirty_untracked_a != clean

    # Untracked file *content* must matter, not just presence.
    (repo / "new.txt").write_text("second\n", encoding="utf-8")
    _, dirty_untracked_b = cli_module.source_identity(repo_root=repo)
    assert dirty_untracked_b != dirty_untracked_a


def test_source_identity_and_checkpoint_are_independent_of_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_identity = cli_module.source_identity()
    expected_checkpoint = cli_module._REPO_ROOT / "data" / "evaluations" / "abc.jsonl"
    nested = tmp_path / "elsewhere" / "nested"
    nested.mkdir(parents=True)

    monkeypatch.chdir(nested)

    assert cli_module.source_identity() == expected_identity
    assert cli_module._default_checkpoint_path("abc") == expected_checkpoint


def test_matrix_mode_out_flag_writes_json(
    fake_matrix_evaluate: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    out = tmp_path / "baselines.json"
    cli_module.main(["--games", "4", "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "matchups" in payload and isinstance(payload["matchups"], list)


# --------------------------------------------------------------------------- #
# Regressions from the code-quality review                                    #
# --------------------------------------------------------------------------- #


def test_puct_flag_accepts_zero_but_rejects_negative(
    fake_source_identity: tuple[str, str],
    fake_run_protocol: _CapturedRun,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--a-puct-c 0`` / ``--b-puct-c 0`` is a valid disable-PUCT setting and
    must NOT be rejected as non-positive. Negative values must still be
    rejected with a clear ``"non-negative"`` diagnostic (distinct from the
    ``"positive"`` message used for strictly-positive knobs)."""
    # Zero is a legal value — disables PUCT selection.
    cli_module.main(
        [
            "--agent-a",
            "ismcts",
            "--agent-b",
            "ismcts",
            "--a-puct-c",
            "0",
            "--b-puct-c",
            "0",
        ]
    )
    proto = fake_run_protocol.protocol
    assert proto is not None
    assert proto.agent_a.params["puct_c"] == pytest.approx(0.0)
    assert proto.agent_b.params["puct_c"] == pytest.approx(0.0)

    # Negative values must be rejected with the "non-negative" constraint.
    for flag in ("--a-puct-c", "--b-puct-c"):
        with pytest.raises(SystemExit):
            cli_module.main(
                [
                    "--agent-a",
                    "ismcts",
                    "--agent-b",
                    "ismcts",
                    flag,
                    "-0.5",
                ]
            )
        err = capsys.readouterr().err
        assert (
            "non-negative" in err
        ), f"expected a 'non-negative' validation message for {flag}; got stderr:\n{err}"


def test_source_identity_ignores_gitignored_evaluations_dir(tmp_path: Path) -> None:
    """After committing a .gitignore that lists ``data/evaluations/``, dropping
    a checkpoint file there must NOT perturb the identity — the dirty-tree
    hash must stay ``"clean"``. This is the invariant that lets a targeted
    eval run reuse its own checkpoint across repeated invocations."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    (repo / ".gitignore").write_text("data/evaluations/\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(repo), "add", ".gitignore"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-q", "-m", "ignore evaluations dir"])

    _, dirty_before = cli_module.source_identity(repo_root=repo)
    assert dirty_before == "clean"

    (repo / "data" / "evaluations").mkdir(parents=True)
    (repo / "data" / "evaluations" / "result.jsonl").write_text(
        '{"case_id": "x"}\n', encoding="utf-8"
    )
    _, dirty_after = cli_module.source_identity(repo_root=repo)
    assert dirty_after == "clean", (
        "gitignored data/evaluations/ contents must not perturb source identity; "
        f"got dirty_after={dirty_after!r}"
    )


def test_module_execution_propagates_exit_code_on_targeted_failure(
    tmp_path: Path,
) -> None:
    """``python -m automata.evaluation.cli`` must propagate ``main()``'s int
    return as the process exit code. A deliberately tiny targeted run
    (random vs random, one paired seed, ``--max-steps 1``) must complete
    fast, fail the gates on max-steps termination, and exit with status 1.

    Kept robust: uses the caller's Python + PYTHONPATH env so the child
    can import the package the same way the parent test does.
    """
    import os
    import sys

    checkpoint = tmp_path / "targeted.jsonl"
    env = os.environ.copy()
    repo_src = str(Path(__file__).resolve().parents[2] / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_src + (os.pathsep + existing if existing else "")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "automata.evaluation.cli",
            "--agent-a",
            "random",
            "--agent-b",
            "random",
            "--paired-seeds",
            "1",
            "--max-steps",
            "1",
            "--checkpoint",
            str(checkpoint),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Rich failure context so the assertion output is self-explanatory when
    # this ever regresses.
    assert proc.returncode == 1, (
        f"expected exit code 1 (gates fail on max_steps); got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    # Should have printed the FAIL verdict and produced/updated the checkpoint.
    assert "FAIL" in proc.stdout, proc.stdout
    assert checkpoint.exists()


# --------------------------------------------------------------------------- #
# Per-case wall-clock timeout (targeted mode)                                 #
# --------------------------------------------------------------------------- #
#
# Contract: targeted mode gains ``--case-timeout-seconds`` (positive float,
# default 1800.0). The CLI must pass it through to the protocol, print the
# effective timeout config + a timeout count in its summary, and exit 1 when
# a timeout causes gates to fail. The runner it hands ``run_protocol`` must
# be pickle-safe so spawn subprocesses can import it.


def test_case_timeout_seconds_defaults_to_1800(
    fake_source_identity: tuple[str, str],
    fake_run_protocol: _CapturedRun,
) -> None:
    """Default targeted timeout must be 1800.0s (30 minutes)."""
    cli_module.main(["--agent-a", "random", "--agent-b", "heuristic"])
    proto = fake_run_protocol.protocol
    assert proto is not None
    assert getattr(proto, "case_timeout_seconds", "MISSING") == pytest.approx(1800.0)


def test_case_timeout_seconds_flag_is_passed_to_protocol(
    fake_source_identity: tuple[str, str],
    fake_run_protocol: _CapturedRun,
) -> None:
    cli_module.main(
        [
            "--agent-a",
            "random",
            "--agent-b",
            "heuristic",
            "--case-timeout-seconds",
            "45.5",
        ]
    )
    proto = fake_run_protocol.protocol
    assert proto is not None
    assert proto.case_timeout_seconds == pytest.approx(45.5)


@pytest.mark.parametrize(
    "value",
    ["0", "0.0", "-1", "-0.5"],
)
def test_case_timeout_seconds_rejects_non_positive(
    fake_source_identity: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        cli_module.main(
            [
                "--agent-a",
                "random",
                "--agent-b",
                "heuristic",
                "--case-timeout-seconds",
                value,
            ]
        )
    err = capsys.readouterr().err
    assert (
        "positive" in err
    ), f"expected a 'positive' diagnostic for --case-timeout-seconds {value!r}; got:\n{err}"


def test_timed_case_runner_is_picklable_when_timeout_configured(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner handed to run_protocol under a wall-clock timeout must be
    pickle-safe because it will be sent to a spawn subprocess.

    We drive the CLI, capture the runner it constructs, and pickle it. If
    the CLI still returns a closure (or any object referencing local
    scope), :func:`pickle.dumps` raises ``PicklingError`` / ``AttributeError``
    and this test fails.
    """
    captured: dict[str, Any] = {}

    def _capture(protocol: EvaluationProtocol, *, checkpoint_path: Path, run_case):
        captured["run_case"] = run_case
        captured["protocol"] = protocol
        return []

    monkeypatch.setattr(cli_module, "run_protocol", _capture)
    monkeypatch.setattr(cli_module, "summarize", lambda obs: EvaluationSummary())

    cli_module.main(
        [
            "--agent-a",
            "random",
            "--agent-b",
            "heuristic",
            "--case-timeout-seconds",
            "10",
        ]
    )
    runner = captured.get("run_case")
    assert runner is not None, "CLI did not invoke run_protocol"

    # pickle.dumps must succeed. The exact class is intentionally not
    # asserted — this is a picklability contract, not an internal-shape
    # contract. Both a top-level class instance and a functools.partial
    # bound to a top-level function satisfy it.
    pickle.dumps(runner)


# --------------------------------------------------------------------------- #
# Learned value artifacts in targeted mode                                    #
# --------------------------------------------------------------------------- #


def _value_artifact(coeff: float = 1.0) -> dict[str, Any]:
    return {
        "model_version": "logistic-v1",
        "schema_version": 1,
        "red_roster": list(RED_HEROES),
        "blue_roster": list(BLUE_HEROES),
        "feature_names": list(FEATURE_NAMES),
        "feature_means": [0.0] * len(FEATURE_NAMES),
        "feature_scales": [1.0] * len(FEATURE_NAMES),
        "coefficients": [coeff] + [0.0] * (len(FEATURE_NAMES) - 1),
        "intercept": 0.0,
    }


def _write_value_artifact(path: Path, coeff: float = 1.0) -> None:
    path.write_text(json.dumps(_value_artifact(coeff)))


@pytest.mark.parametrize("side", ["a", "b"])
def test_value_model_is_accepted_only_for_ismcts_side(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    side: str,
) -> None:
    artifact = tmp_path / "value.json"
    _write_value_artifact(artifact)
    kinds = {"a": "ismcts", "b": "heuristic"}
    kinds[side] = "ismcts"
    accepted = _install_run_protocol(monkeypatch)
    cli_module.main(
        [
            "--agent-a", kinds["a"], "--agent-b", kinds["b"],
            f"--{side}-value-model", str(artifact),
        ]
    )  # fmt: skip
    assert accepted.protocol is not None

    called = False

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("run_protocol must not run after invalid CLI input")

    monkeypatch.setattr(cli_module, "run_protocol", _forbidden)
    kinds[side] = "random"
    with pytest.raises(SystemExit):
        cli_module.main(
            [
                "--agent-a", kinds["a"], "--agent-b", kinds["b"],
                f"--{side}-value-model", str(artifact),
            ]
        )  # fmt: skip
    assert not called


def test_artifact_digest_participates_in_targeted_identity(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "value.json"
    _write_value_artifact(artifact)
    first = _install_run_protocol(monkeypatch)
    argv = ["--agent-a", "ismcts", "--agent-b", "heuristic", "--a-value-model", str(artifact)]
    cli_module.main(argv)
    assert first.protocol is not None
    expected = LearnedValue(artifact).digest
    assert re.fullmatch(r"[0-9a-f]{64}", expected)
    assert expected in json.dumps(first.protocol.agent_a.identity(), sort_keys=True)

    relocated = tmp_path / "relocated.json"
    _write_value_artifact(relocated)
    relocated_run = _install_run_protocol(monkeypatch)
    cli_module.main(
        ["--agent-a", "ismcts", "--agent-b", "heuristic", "--a-value-model", str(relocated)]
    )
    assert relocated_run.protocol is not None
    assert relocated_run.protocol.identity_digest() == first.protocol.identity_digest()

    _write_value_artifact(artifact, coeff=2.0)
    second = _install_run_protocol(monkeypatch)
    cli_module.main(argv)
    assert second.protocol is not None
    assert second.protocol.identity_digest() != first.protocol.identity_digest()


def test_build_agent_injects_captured_artifact_and_rejects_changed_content(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "value.json"
    _write_value_artifact(artifact)
    captured = _install_run_protocol(monkeypatch)
    cli_module.main(
        ["--agent-a", "ismcts", "--agent-b", "heuristic", "--a-value-model", str(artifact)]
    )
    assert captured.protocol is not None
    spec = captured.protocol.agent_a
    built: dict[str, Any] = {}

    class _AgentSpy:
        def __init__(
            self, config: Any, *, value_fn: Any = None, cutoff_observer: Any = None
        ) -> None:
            built.update(config=config, value_fn=value_fn, cutoff_observer=cutoff_observer)

    monkeypatch.setattr(cli_module, "ISMCTSAgent", _AgentSpy)
    agent = cli_module.build_agent(spec, seed=73)
    assert agent is not None
    assert isinstance(built["value_fn"], LearnedValue)
    assert built["config"].seed == 73

    _write_value_artifact(artifact, coeff=3.0)
    with pytest.raises(ValueError, match=r"(?i)(changed|digest|sha|artifact)"):
        cli_module.build_agent(spec, seed=73)


def test_learned_runner_is_pickleable_and_output_identifies_value_model(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "distinctive-value.json"
    _write_value_artifact(artifact)
    learned = _install_run_protocol(monkeypatch)
    cli_module.main(
        ["--agent-a", "ismcts", "--agent-b", "heuristic", "--a-value-model", str(artifact)]
    )
    learned_output = capsys.readouterr().out
    assert learned.protocol is not None
    pickle.dumps(cli_module.build_case_runner(learned.protocol))

    _install_run_protocol(monkeypatch)
    cli_module.main(["--agent-a", "ismcts", "--agent-b", "heuristic"])
    heuristic_output = capsys.readouterr().out
    digest = LearnedValue(artifact).digest
    markers = (artifact.name, str(artifact), digest, digest[:12], "learned")
    assert any(
        m.lower() in learned_output.lower() and m.lower() not in heuristic_output.lower()
        for m in markers
    )


def _observations_with_timeouts(protocol: EvaluationProtocol) -> list[GameObservation]:
    """All A-wins except one timeout row → dominant-but-timed-out schedule."""
    rows: list[GameObservation] = []
    cases = list(protocol.cases())
    for i, c in enumerate(cases):
        if i == 0:
            rows.append(
                GameObservation(
                    case_id=c.case_id,
                    world_seed=c.world_seed,
                    a_side=c.a_side,
                    winner_side=None,
                    rounds=0,
                    steps=0,
                    reason="wall_clock_timeout",
                )
            )
        else:
            rows.append(
                GameObservation(
                    case_id=c.case_id,
                    world_seed=c.world_seed,
                    a_side=c.a_side,
                    winner_side=c.a_side,
                    rounds=8,
                    steps=200,
                    reason="game_over",
                )
            )
    return rows


def test_targeted_prints_timeout_config_and_count(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Header/summary must include the effective timeout AND surface the
    timeout count in a user-visible summary line.

    The observations fixture produces exactly one wall_clock_timeout row
    (case index 0) and 11 game_over rows, so the count is 1.

    Format is not pinned. We accept any user-visible line that carries
    *both* a timeout label (case-insensitive substring ``timeout``) and
    the standalone integer ``1``, so all of the following satisfy:

    - ``Summary: A=... timeouts=1 ...``
    - ``timeouts: 1``
    - ``1 wall_clock_timeout``
    - ``timeout terminations: 1``
    - ``Summary: timeouts=1, case_timeout_seconds=42.5s`` (label+count and
      config all on one operator-friendly line)

    What is deliberately *not* sufficient:

    - a line that only reports the timeout *budget* / configuration
      (``case_timeout_seconds=42.5``) — that names timeouts but does not
      report the count, so the operator cannot tell how many cases
      actually hit the budget.

    Anti-forgery: the fixture's config value ``42.5`` contains no
    ``1``-digit, and we match the count with a digit-boundaried ``1``
    (``(?<!\\d)1(?!\\d)``) so it cannot be smuggled out of the config
    value. Merely printing the config value therefore cannot satisfy
    this assertion — a bare-count digit must additionally appear on a
    line that also names timeouts.
    """
    _install_run_protocol(monkeypatch, observations_fn=_observations_with_timeouts)
    from automata.evaluation import protocol as protocol_module

    monkeypatch.setattr(cli_module, "summarize", protocol_module.summarize)

    cli_module.main(
        [
            "--agent-a",
            "random",
            "--agent-b",
            "heuristic",
            "--case-timeout-seconds",
            "42.5",
        ]
    )
    out = capsys.readouterr().out

    # 1. Effective timeout must appear so the operator sees the budget.
    assert "42.5" in out, f"expected timeout config '42.5' in output; got:\n{out}"

    # 2. Some user-visible line must carry BOTH a timeout label and the
    #    numeric count ``1``. Because the config value ``42.5`` contains
    #    no ``1`` digit and we require a digit-boundaried match, a line
    #    that only reports the config cannot satisfy this — the count
    #    must be printed as its own value on a timeout-labelled line.
    count_lines = [
        ln
        for ln in out.splitlines()
        if "timeout" in ln.lower() and re.search(r"(?<!\d)1(?!\d)", ln)
    ]
    assert count_lines, (
        "expected a user-visible summary line containing both a timeout "
        "label and the numeric count 1; the timeout configuration value "
        f"alone is not sufficient. Got output:\n{out}"
    )


def test_targeted_exit_code_one_when_timeout_fails_gates(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A single timeout must flip the gates to FAIL and the exit code to 1."""
    _install_run_protocol(monkeypatch, observations_fn=_observations_with_timeouts)
    from automata.evaluation import protocol as protocol_module

    monkeypatch.setattr(cli_module, "summarize", protocol_module.summarize)

    rc = cli_module.main(
        [
            "--agent-a",
            "random",
            "--agent-b",
            "heuristic",
            "--case-timeout-seconds",
            "10",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit code 1 when a timeout fails gates; got {rc!r}\n{out}"
    assert "FAIL" in out


def test_ordinary_legacy_invocation_still_takes_matrix_path(
    fake_matrix_evaluate: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding the targeted timeout flag must not warp legacy matrix invocation.

    Ordinary legacy usage — no ``--agent-a``/``--agent-b``, just the pre-existing
    matrix flags (``--games`` / ``--seed``) — must still route through
    :func:`evaluate` (the matrix path) and must NOT touch :func:`run_protocol`
    (the targeted path). Regardless of whether ``--case-timeout-seconds`` has
    been added yet, an ordinary matrix invocation must:

    - call ``evaluate`` at least once (with the well-known matrix labels), AND
    - never call ``run_protocol``, AND
    - not raise ``SystemExit`` (the CLI accepts the invocation on the happy path).

    This is stronger than "does it reject the flag" because it names the
    contract we actually care about: legacy behavior is untouched.
    """
    run_protocol_calls: list[Any] = []

    def _forbidden_run_protocol(*args: Any, **kwargs: Any) -> Any:
        run_protocol_calls.append((args, kwargs))
        raise AssertionError("run_protocol must not be called from ordinary legacy matrix mode")

    monkeypatch.setattr(cli_module, "run_protocol", _forbidden_run_protocol)

    rc = cli_module.main(["--games", "3", "--seed", "0"])

    # Legacy mode returns None (see CLI docstring); no non-zero exit code.
    assert rc in (None, 0), f"legacy matrix mode returned unexpected rc={rc!r}"

    # The legacy evaluator must have been invoked at least once with the
    # canonical matrix labels — proof we took the matrix branch.
    labels = {(c["label_a"], c["label_b"]) for c in fake_matrix_evaluate}
    assert (
        "random",
        "random",
    ) in labels, f"expected legacy matrix evaluate() to be called; saw labels={labels}"

    # And run_protocol was never called.
    assert run_protocol_calls == [], (
        "ordinary legacy invocation must not stray into the targeted " "run_protocol path"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--agent-a", "random", "--agent-b", "ismcts", "--a-cutoff-telemetry", "a.jsonl"],
        ["--agent-a", "ismcts", "--agent-b", "heuristic", "--b-cutoff-telemetry", "b.jsonl"],
    ],
)
def test_cutoff_telemetry_requires_corresponding_ismcts_agent_before_run(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    called = False

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("invalid telemetry configuration reached run_protocol")

    monkeypatch.setattr(cli_module, "run_protocol", _forbidden)
    with pytest.raises(SystemExit):
        cli_module.main(argv)
    error = capsys.readouterr().err.lower()
    assert "unrecognized arguments" not in error
    assert "telemetry" in error and "ismcts" in error
    assert called is False


def test_telemetry_path_is_runtime_only_agent_spec_metadata(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = _install_run_protocol(monkeypatch)
    cli_module.main(["--agent-a", "ismcts", "--agent-b", "heuristic"])
    enabled = _install_run_protocol(monkeypatch)
    path = tmp_path / "cutoffs.jsonl"
    cli_module.main(
        [
            "--agent-a",
            "ismcts",
            "--agent-b",
            "heuristic",
            "--a-cutoff-telemetry",
            str(path),
        ]
    )

    assert baseline.protocol is not None and enabled.protocol is not None
    assert enabled.protocol.agent_a.cutoff_telemetry_path == str(path)
    assert enabled.protocol.identity_digest() == baseline.protocol.identity_digest()


@pytest.mark.parametrize("telemetry_path", [None, "cutoffs.jsonl"])
def test_build_agent_injects_cutoff_telemetry_observer(
    monkeypatch: pytest.MonkeyPatch,
    telemetry_path: str | None,
) -> None:
    constructed: dict[str, Any] = {}
    recorder = object()

    def _recorder_spy(path: str, case_metadata: dict[str, Any]) -> object:
        constructed["recorder_args"] = (path, case_metadata)
        return recorder

    class _AgentSpy:
        def __init__(self, config: Any, *, value_fn: Any, cutoff_observer: Any) -> None:
            constructed.update(config=config, cutoff_observer=cutoff_observer)

    monkeypatch.setattr(cli_module, "CutoffTelemetryRecorder", _recorder_spy, raising=False)
    monkeypatch.setattr(cli_module, "ISMCTSAgent", _AgentSpy)
    metadata = {"case_id": "case-7-RED", "world_seed": 7, "a_side": "RED", "agent_label": "A"}
    spec = AgentSpec(
        name="A",
        kind="ismcts",
        params={"iterations": 1},
        cutoff_telemetry_path=telemetry_path,
    )

    cli_module.build_agent(spec, seed=11, case_metadata=metadata)

    assert constructed["cutoff_observer"] is (recorder if telemetry_path else None)
    if telemetry_path:
        assert constructed["recorder_args"] == (telemetry_path, metadata)
    else:
        assert "recorder_args" not in constructed


def test_case_runner_supplies_agent_specific_telemetry_metadata_and_is_pickleable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def _build(spec: AgentSpec, seed: int, *, case_metadata: dict[str, Any]) -> _StubAgent:
        calls.append((spec.name, case_metadata))
        return _StubAgent(spec, seed)

    monkeypatch.setattr(cli_module, "build_agent", _build)
    monkeypatch.setattr(
        cli_module,
        "run_game",
        lambda *args, **kwargs: RunResult(
            winner="RED", rounds=1, turns=1, steps=1, reason="game_over"
        ),
    )
    protocol = _make_protocol(world_seeds=(7,))
    runner = cli_module.build_case_runner(protocol)
    pickle.dumps(runner)
    case = next(c for c in protocol.cases() if c.a_side == "BLUE")
    runner(case)

    common = {"case_id": case.case_id, "world_seed": 7, "a_side": "BLUE"}
    assert calls == [
        ("a", {**common, "agent_label": "A"}),
        ("b", {**common, "agent_label": "B"}),
    ]


def test_targeted_header_indicates_cutoff_telemetry_enabled(
    fake_source_identity: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _install_run_protocol(monkeypatch)
    args = f"--agent-a ismcts --agent-b heuristic --a-cutoff-telemetry {tmp_path / 'rows.jsonl'}"
    cli_module.main(args.split())
    output = capsys.readouterr().out.lower()
    assert "telemetry" in output and "enabled" in output
