from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from automata.models.contracts import (
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    Viewer,
    canonical_json_bytes,
)
from automata.scripts import run_parallel
from automata.scripts.generate_joint_bootstrap import CheckpointRow
from automata.scripts.run_parallel import Shard
from automata.training.dataset import (
    JointDatasetRow,
    joint_decision_id,
    load_joint_dataset,
    write_joint_dataset,
)


def _option(command: list[str], name: str) -> str:
    assert command.count(name) == 1
    return command[command.index(name) + 1]


def _row(seed: int) -> JointDatasetRow:
    game_id = f"game-{seed}"
    candidate = EncodedCandidate(
        schema_version=1,
        candidate_id=OptionCandidateID(schema_version=1, option_id="hold"),
        selection="hold",
    )
    observation = DecisionObservation(
        schema_version=3,
        state=LearnedObservation(
            schema_version=2,
            viewer=Viewer(schema_version=2, perspective_team="RED"),
            tokens=(
                ObservationToken(
                    schema_version=1,
                    local_ref="global:0",
                    kind="GLOBAL",
                    features={"map_id": "map", "game_type": "QUICK"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:red:0",
                    kind="HERO",
                    features={"name": "Wasp", "team_id": "RED"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:blue:0",
                    kind="HERO",
                    features={"name": "Arien", "team_id": "BLUE"},
                ),
            ),
        ),
        decision_kind="INPUT",
        candidates=(candidate,),
    )
    identity: dict[str, Any] = {
        "game_id": game_id,
        "world_seed": seed,
        "map_id": "map",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": "generation",
        "source_revision": "revision",
        "dirty_tree_hash": "dirty",
        "source_model_digest": None,
        "search_config_id": "search",
        "generator_config_id": "config",
    }
    return JointDatasetRow(
        **identity,
        decision_id=joint_decision_id(**identity, decision_index=0),
        decision_index=0,
        perspective_team="RED",
        observation=observation,
        policy_source="HEURISTIC",
        policy_target=(1.0,),
        selected_candidate_id=candidate.candidate_id,
        selected_selection="hold",
        terminal_winner="RED",
        value_target=1,
    )


def _checkpoint(path: Path, *rows: JointDatasetRow) -> None:
    path.write_bytes(
        b"".join(
            canonical_json_bytes(
                CheckpointRow(
                    config_id=row.generator_config_id,
                    game_id=row.game_id,
                    world_seed=row.world_seed,
                    completed=True,
                    reason="game_over",
                    winner="RED",
                    rounds=1,
                    turns=1,
                    steps=1,
                )
            )
            + b"\n"
            for row in rows
        )
    )


def test_joint_bootstrap_builds_four_balanced_isolated_workers_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "joint.jsonl"
    checkpoint = tmp_path / "games.jsonl"
    seen: list[tuple[Shard, list[str], Path]] = []

    monkeypatch.setattr(run_parallel, "source_identity", lambda **_kw: ("revision", "dirty"))

    def run_workers(commands: list[tuple[Shard, list[str], Path]], **_kwargs: object) -> None:
        seen.extend(commands)

    monkeypatch.setattr(run_parallel, "_run_joint_workers", run_workers)
    monkeypatch.setattr(
        run_parallel,
        "merge_joint_bootstrap",
        lambda parts, checkpoints, destination, **kwargs: 17,
    )

    status = run_parallel.main(
        [
            "joint-bootstrap",
            "--out",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--seed-start",
            "10000",
            "--seed-end",
            "10010",
            "--",
            "--target-source",
            "heuristic",
            "--target-recipe",
            "one-hot-exact-choice",
            "--max-steps",
            "123",
            "--timeout-seconds",
            "30",
        ]
    )

    assert status == 0
    assert [(shard.start, shard.count) for shard, _, _ in seen] == [
        (10000, 3),
        (10003, 3),
        (10006, 2),
        (10008, 2),
    ]
    for shard, command, _ in seen:
        assert command[1:3] == ["-m", "automata.scripts.generate_joint_bootstrap"]
        assert _option(command, "--seed-start") == str(shard.start)
        assert _option(command, "--seed-end") == str(shard.start + shard.count)
        assert _option(command, "--max-steps") == "123"
        assert _option(command, "--source-revision") == "revision"
        assert _option(command, "--dirty-tree-hash") == "dirty"
        assert Path(_option(command, "--out")).parent == Path(f"{output}.shards")
        assert Path(_option(command, "--checkpoint")).parent == Path(f"{checkpoint}.shards")


def test_joint_bootstrap_can_disable_parent_progress_without_forwarding_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(run_parallel, "source_identity", lambda **_kw: ("revision", "dirty"))

    def run_workers(commands: list[tuple[Shard, list[str], Path]], **kwargs: object) -> None:
        seen.update(kwargs)
        assert all("--no-progress" not in command for _, command, _ in commands)

    monkeypatch.setattr(run_parallel, "_run_joint_workers", run_workers)
    monkeypatch.setattr(run_parallel, "merge_joint_bootstrap", lambda *_args, **_kwargs: 0)

    status = run_parallel.main(
        [
            "joint-bootstrap",
            "--out",
            str(tmp_path / "out.jsonl"),
            "--checkpoint",
            str(tmp_path / "checkpoint.jsonl"),
            "--seed-start",
            "10",
            "--seed-end",
            "11",
            "--no-progress",
            "--progress-interval",
            "7.5",
            "--",
            "--target-source",
            "heuristic",
        ]
    )

    assert status == 0
    assert seen["show_progress"] is False
    assert seen["progress_interval"] == 7.5


def test_progress_reporter_renders_tty_progress_in_place_and_finishes_with_newline() -> None:
    stream = io.StringIO()
    reporter = run_parallel.ProgressReporter(
        total=8,
        workers=4,
        stream=stream,
        clock=lambda: 10.0,
        interactive=True,
    )
    reporter.report(run_parallel.ProgressCounts(success=2), active_workers=4)
    reporter.clock = lambda: 14.0
    reporter.report(run_parallel.ProgressCounts(success=5, failed=1), active_workers=2)
    reporter.close(run_parallel.ProgressCounts(success=5, failed=1), active_workers=0)

    text = stream.getvalue()
    assert "\r\033[2K[" in text
    assert "6/8" in text
    assert "75.0%" in text
    assert "workers 2/4" in text
    assert "1.00 games/s" in text
    assert "ETA 2s" in text
    assert "ok 5" in text
    assert "failed 1" in text
    assert text.endswith("\n")


def test_progress_reporter_emits_stable_plain_text_without_control_characters() -> None:
    stream = io.StringIO()
    reporter = run_parallel.ProgressReporter(
        total=10,
        workers=3,
        stream=stream,
        clock=lambda: 20.0,
        interactive=False,
    )

    reporter.report(run_parallel.ProgressCounts(success=4, failed=1, timed_out=1), active_workers=3)

    assert stream.getvalue() == (
        "joint-bootstrap progress [############--------] 6/10 60.0% "
        "workers 3/3 0.00 games/s ETA ? ok 4 failed 1 timeout 1\n"
    )
    assert "\033" not in stream.getvalue()
    assert "\r" not in stream.getvalue()


def test_checkpoint_progress_counts_resumed_and_latest_terminal_outcomes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "part.games.jsonl"
    rows = (
        CheckpointRow(
            config_id="config",
            game_id="success",
            world_seed=1,
            completed=True,
            reason="game_over",
            winner="RED",
            rounds=1,
            turns=1,
            steps=1,
        ),
        CheckpointRow(
            config_id="config",
            game_id="retried",
            world_seed=2,
            completed=False,
            reason="wall_clock_timeout",
            winner=None,
            rounds=None,
            turns=None,
            steps=None,
        ),
        CheckpointRow(
            config_id="config",
            game_id="failed",
            world_seed=3,
            completed=False,
            reason="max_steps",
            winner=None,
            rounds=2,
            turns=4,
            steps=100,
        ),
        CheckpointRow(
            config_id="config",
            game_id="retried",
            world_seed=2,
            completed=True,
            reason="game_over",
            winner="BLUE",
            rounds=3,
            turns=5,
            steps=90,
        ),
    )
    checkpoint.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))

    assert run_parallel.checkpoint_progress([checkpoint]) == run_parallel.ProgressCounts(
        success=2, failed=1, timed_out=0
    )


def test_joint_worker_reporting_starts_from_resumed_records_and_keeps_failure_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "part.games.jsonl"
    checkpoint.write_bytes(
        canonical_json_bytes(
            CheckpointRow(
                config_id="config",
                game_id="resumed",
                world_seed=10,
                completed=False,
                reason="wall_clock_timeout",
                winner=None,
                rounds=None,
                turns=None,
                steps=None,
            )
        )
        + b"\n"
    )

    class FailedProcess:
        pid = 123

        def poll(self) -> int:
            return 7

        def terminate(self) -> None:
            pytest.fail("finished process must not be terminated")

    monkeypatch.setattr(run_parallel.subprocess, "Popen", lambda *_args, **_kwargs: FailedProcess())
    stream = io.StringIO()
    command = (Shard(index=0, start=10, count=1), ["worker"], tmp_path / "worker.log")

    with pytest.raises(RuntimeError, match=r"shard 0 exited 7; inspect .*worker\.log"):
        run_parallel._run_joint_workers(
            [command],
            checkpoints=[checkpoint],
            total=1,
            workers=1,
            stream=stream,
            clock=lambda: 5.0,
            sleep=lambda _seconds: None,
        )

    progress_lines = [line for line in stream.getvalue().splitlines() if " progress " in line]
    assert len(progress_lines) == 2
    assert all("1/1" in line and "timeout 1" in line for line in progress_lines)
    assert stream.getvalue().endswith("\n")


def test_joint_merge_validates_checkpoints_deduplicates_and_sorts(tmp_path: Path) -> None:
    row_10, row_11 = _row(10), _row(11)
    part_a, part_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    part_a.write_bytes(canonical_json_bytes(row_11) + b"\n")
    part_b.write_bytes(canonical_json_bytes(row_10) + b"\n" + canonical_json_bytes(row_11) + b"\n")
    checkpoint_a, checkpoint_b = tmp_path / "a.games", tmp_path / "b.games"
    _checkpoint(checkpoint_a, row_11)
    _checkpoint(checkpoint_b, row_10, row_11)
    output = tmp_path / "merged.jsonl"

    assert (
        run_parallel.merge_joint_bootstrap(
            [part_a, part_b],
            [checkpoint_a, checkpoint_b],
            output,
            expected_seeds=range(10, 12),
        )
        == 2
    )
    assert output.read_bytes() == (
        canonical_json_bytes(row_10) + b"\n" + canonical_json_bytes(row_11) + b"\n"
    )


def test_joint_merge_reads_and_writes_compressed_datasets(tmp_path: Path) -> None:
    row_10, row_11 = _row(10), _row(11)
    part_a, part_b = tmp_path / "a.jsonl.zst", tmp_path / "b.jsonl.zst"
    write_joint_dataset(part_a, (row_11,))
    write_joint_dataset(part_b, (row_10,))
    checkpoint_a, checkpoint_b = tmp_path / "a.games.jsonl", tmp_path / "b.games.jsonl"
    _checkpoint(checkpoint_a, row_11)
    _checkpoint(checkpoint_b, row_10)
    output = tmp_path / "merged.jsonl.zst"

    assert (
        run_parallel.merge_joint_bootstrap(
            [part_a, part_b],
            [checkpoint_a, checkpoint_b],
            output,
            expected_seeds=range(10, 12),
        )
        == 2
    )
    assert load_joint_dataset(output).rows == (row_10, row_11)


def test_joint_merge_rejects_orphan_without_replacing_existing_output(tmp_path: Path) -> None:
    row = _row(10)
    part, checkpoint = tmp_path / "part.jsonl", tmp_path / "games.jsonl"
    part.write_bytes(canonical_json_bytes(row) + b"\n")
    checkpoint.write_bytes(b'{"truncated":')
    output = tmp_path / "merged.jsonl"
    output.write_bytes(b"previous\n")

    with pytest.raises(ValueError, match="orphan decision"):
        run_parallel.merge_joint_bootstrap(
            [part], [checkpoint], output, expected_seeds=range(10, 11)
        )
    assert output.read_bytes() == b"previous\n"


def test_joint_worker_failure_is_operational_and_preserves_resume_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "joint.jsonl"
    checkpoint = tmp_path / "games.jsonl"
    fragment = Path(f"{output}.shards") / ".part-00.jsonl.games" / "saved.jsonl"
    fragment.parent.mkdir(parents=True)
    fragment.write_text("useful")
    monkeypatch.setattr(run_parallel, "source_identity", lambda **_kw: ("revision", "dirty"))
    monkeypatch.setattr(
        run_parallel,
        "_run_joint_workers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("shard 0 exited 7")),
    )
    monkeypatch.setattr(
        run_parallel,
        "merge_joint_bootstrap",
        lambda *_args, **_kwargs: pytest.fail("failed workers must abort publication"),
    )

    status = run_parallel.main(
        [
            "joint-bootstrap",
            "--out",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--seed-start",
            "10000",
            "--seed-end",
            "10001",
            "--",
            "--target-source",
            "heuristic",
        ]
    )

    assert status == run_parallel.OPERATIONAL_FAILURE_EXIT_STATUS
    assert not output.exists()
    assert fragment.read_text() == "useful"


def test_joint_parallel_argument_errors_remain_argparse_status_two(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        run_parallel.main(
            [
                "joint-bootstrap",
                "--out",
                str(tmp_path / "out"),
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--seed-start",
                "10",
                "--seed-end",
                "10",
            ]
        )
    assert raised.value.code == 2
