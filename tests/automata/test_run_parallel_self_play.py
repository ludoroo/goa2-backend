from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from automata.scripts import run_parallel
from automata.training.generation import WORKER_COUNT


def _worker_options() -> list[str]:
    return [
        "--",
        "--artifact",
        "model",
        "--parent-model-digest",
        "a" * 64,
        "--parent-generation",
        "2",
        "--observation-schema-version",
        "4",
        "--generation-id",
        "generation-3",
        "--strategy-preset",
        "learned-policy-heuristic-value",
        "--search-config",
        '{"iterations":8}',
        "--variant-schedule",
        "balanced",
        "--visit-temperature",
        "0.5",
        "--max-steps",
        "10000",
        "--timeout-seconds",
        "3600",
    ]


def _argv(run_dir: Path) -> list[str]:
    return [
        "self-play",
        "--run-dir",
        str(run_dir),
        "--seed-start",
        "40200",
        "--seed-end",
        "40208",
        *_worker_options(),
    ]


def test_parallel_self_play_launches_four_pinned_workers_and_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "round"
    observed: dict[str, object] = {}
    monkeypatch.setattr(run_parallel, "source_identity", lambda **_kw: ("revision", "dirty"))

    def run_workers(commands, **kwargs):
        observed["commands"] = commands
        observed["worker_kwargs"] = kwargs

    def reconcile(**kwargs):
        observed["reconcile"] = kwargs
        return SimpleNamespace(
            game_count=8,
            row_count=123,
            destination=Path(kwargs["destination"]),
        )

    monkeypatch.setattr(run_parallel, "_run_self_play_workers", run_workers)
    monkeypatch.setattr(run_parallel, "reconcile_self_play", reconcile)

    assert run_parallel.main(_argv(run_dir)) == 0

    commands = observed["commands"]
    assert len(commands) == WORKER_COUNT
    for expected_worker, (worker_id, command, stdout_path, stderr_path) in enumerate(commands):
        assert worker_id == expected_worker
        assert command[:3] == [sys.executable, "-m", "automata.scripts.run_self_play"]
        assert command[command.index("--worker-id") + 1] == str(expected_worker)
        assert command[command.index("--seed-start") + 1] == "40200"
        assert command[command.index("--seed-end") + 1] == "40208"
        assert command[command.index("--source-revision") + 1] == "revision"
        assert command[command.index("--dirty-tree-hash") + 1] == "dirty"
        assert command.count("--no-progress") == 1
        assert stdout_path == run_dir / "logs" / f"worker-{worker_id}.stdout"
        assert stderr_path == run_dir / "logs" / f"worker-{worker_id}.stderr"
    assert observed["worker_kwargs"] == {
        "show_progress": True,
        "checkpoint_dir": run_dir / "checkpoints",
        "seed_start": 40200,
        "seed_end": 40208,
    }
    assert observed["reconcile"] == {
        "output_dir": run_dir / "output",
        "checkpoint_dir": run_dir / "checkpoints",
        "seed_start": 40200,
        "seed_end": 40208,
        "destination": run_dir / "generation.jsonl.zst",
        "worker_count": WORKER_COUNT,
    }
    assert (run_dir / "output").is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "logs").is_dir()


def test_parallel_self_play_worker_failure_never_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_parallel, "source_identity", lambda **_kw: ("revision", "dirty"))
    monkeypatch.setattr(
        run_parallel,
        "_run_self_play_workers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker 2 exited 7")),
    )
    monkeypatch.setattr(
        run_parallel,
        "reconcile_self_play",
        lambda **_kwargs: pytest.fail("failed workers must not publish"),
    )

    assert run_parallel.main(_argv(tmp_path / "round")) == 1


def test_parallel_self_play_rerun_builds_identical_resume_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "round"
    launches: list[object] = []
    monkeypatch.setattr(run_parallel, "source_identity", lambda **_kw: ("revision", "dirty"))
    monkeypatch.setattr(
        run_parallel,
        "_run_self_play_workers",
        lambda commands, **_kwargs: launches.append(commands),
    )
    monkeypatch.setattr(
        run_parallel,
        "reconcile_self_play",
        lambda **kwargs: SimpleNamespace(
            game_count=8, row_count=123, destination=Path(kwargs["destination"])
        ),
    )

    assert run_parallel.main(_argv(run_dir)) == 0
    assert run_parallel.main(_argv(run_dir)) == 0
    assert launches[0] == launches[1]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--worker-id", "0"], "wrapper-owned"),
        (["--output-dir", "elsewhere"], "wrapper-owned"),
        (["--worker-id=3"], "wrapper-owned"),
        (["--output-dir=elsewhere"], "wrapper-owned"),
    ],
)
def test_parallel_self_play_rejects_wrapper_owned_worker_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(run_parallel, "source_identity", lambda **_kw: ("revision", "dirty"))
    argv = [*_argv(tmp_path / "round"), *extra]

    assert run_parallel.main(argv) == 1
    assert message in capsys.readouterr().err


def test_parallel_self_play_rejects_invalid_nested_worker_options_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_parallel, "source_identity", lambda **_kw: ("revision", "dirty"))
    monkeypatch.setattr(
        run_parallel,
        "_run_self_play_workers",
        lambda *_args, **_kwargs: pytest.fail("invalid options must not launch workers"),
    )
    argv = _argv(tmp_path / "round")
    artifact = argv.index("--artifact")
    del argv[artifact : artifact + 2]

    assert run_parallel.main(argv) == 1
    assert "invalid run_self_play options" in capsys.readouterr().err
    assert not (tmp_path / "round").exists()


def test_parallel_self_play_rejects_unsafe_path_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        run_parallel,
        "_run_self_play_workers",
        lambda *_args, **_kwargs: pytest.fail("unsafe paths must not launch workers"),
    )

    assert run_parallel.main(_argv(linked)) == 1
    assert run_parallel.main(_argv(tmp_path / "parent" / ".." / "round")) == 1


def test_parallel_self_play_forwards_explicit_identity_and_disables_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        run_parallel,
        "_run_self_play_workers",
        lambda commands, **kwargs: observed.update(commands=commands, worker_kwargs=kwargs),
    )
    monkeypatch.setattr(
        run_parallel,
        "reconcile_self_play",
        lambda **kwargs: SimpleNamespace(
            game_count=8, row_count=1, destination=Path(kwargs["destination"])
        ),
    )
    argv = [
        "self-play",
        "--run-dir",
        str(tmp_path / "round"),
        "--seed-start",
        "40200",
        "--seed-end",
        "40208",
        "--source-revision",
        "pinned-revision",
        "--dirty-tree-hash",
        "pinned-dirty",
        "--no-progress",
        *_worker_options(),
    ]

    assert run_parallel.main(argv) == 0
    assert observed["worker_kwargs"]["show_progress"] is False
    for _, command, _, _ in observed["commands"]:
        assert command[command.index("--source-revision") + 1] == "pinned-revision"
        assert command[command.index("--dirty-tree-hash") + 1] == "pinned-dirty"


def test_parallel_self_play_rejects_partial_explicit_source_identity(tmp_path: Path) -> None:
    argv = [
        "self-play",
        "--run-dir",
        str(tmp_path / "round"),
        "--seed-start",
        "40200",
        "--seed-end",
        "40208",
        "--source-revision",
        "revision-only",
        *_worker_options(),
    ]
    assert run_parallel.main(argv) == 1
    assert not (tmp_path / "round").exists()


def test_parallel_self_play_requires_four_games(tmp_path: Path) -> None:
    argv = _argv(tmp_path / "round")
    argv[argv.index("--seed-end") + 1] = "40203"
    assert run_parallel.main(argv) == 1


def test_self_play_completed_seeds_filters_partial_wrong_and_unassigned_rows(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "worker-0.jsonl").write_text(
        '{"worker_id":0,"world_seed":40200,"reason":"game_over"}\n'
        '{"worker_id":0,"world_seed":40204,"reason":"game_over"}\n'
        '{"worker_id":0,"world_seed":40201,"reason":"game_over"}\n'
        '{"worker_id":0,"world_seed":40208,"reason":"game_over"}\n'
        '{"worker_id":0,"world_seed":40200,"reason":"game_over"}'
    )
    (checkpoint_dir / "worker-1.jsonl").write_text(
        '{"worker_id":1,"world_seed":40201,"reason":"game_over"}\n'
    )

    assert run_parallel.self_play_completed_seeds(
        checkpoint_dir, seed_start=40200, seed_end=40208
    ) == (40200, 40201, 40204)


def test_self_play_worker_failure_names_both_logs(tmp_path: Path) -> None:
    stdout_path = tmp_path / "worker.stdout"
    stderr_path = tmp_path / "worker.stderr"
    commands = [
        (
            2,
            [
                sys.executable,
                "-c",
                "import sys; print('bad', file=sys.stderr); raise SystemExit(7)",
            ],
            stdout_path,
            stderr_path,
        )
    ]

    with pytest.raises(RuntimeError, match=r"worker 2 exited 7") as raised:
        run_parallel._run_self_play_workers(commands, show_progress=False)
    assert str(stdout_path) in str(raised.value)
    assert str(stderr_path) in str(raised.value)


def test_self_play_worker_logs_are_separate_and_append_on_resume(tmp_path: Path) -> None:
    stdout_path = tmp_path / "worker.stdout"
    stderr_path = tmp_path / "worker.stderr"
    command = [
        sys.executable,
        "-c",
        "import sys; print('out'); print('err', file=sys.stderr)",
    ]
    commands = [(0, command, stdout_path, stderr_path)]

    run_parallel._run_self_play_workers(commands, show_progress=False)
    run_parallel._run_self_play_workers(commands, show_progress=False)

    assert stdout_path.read_text() == "out\nout\n"
    assert stderr_path.read_text() == "err\nerr\n"


def test_self_play_progress_postfix_does_not_force_duplicate_redraws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refresh_values: list[bool] = []

    class Progress:
        def update(self, _amount: int) -> None:
            pass

        def set_postfix(self, *, workers: str, refresh: bool = True) -> None:
            assert workers.endswith(f"/{WORKER_COUNT}")
            refresh_values.append(refresh)

        def close(self) -> None:
            pass

    monkeypatch.setattr(run_parallel, "tqdm", lambda **_kwargs: Progress())
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    command = [sys.executable, "-c", "pass"]

    run_parallel._run_self_play_workers(
        [(0, command, tmp_path / "worker.stdout", tmp_path / "worker.stderr")],
        show_progress=True,
        checkpoint_dir=checkpoint_dir,
        seed_start=0,
        seed_end=1,
        progress_interval=0.001,
    )

    assert refresh_values
    assert not any(refresh_values)
