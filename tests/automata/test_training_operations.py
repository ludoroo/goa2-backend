"""Keep documented training commands aligned with their CLI parsers."""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from automata.training import trainer
from automata.training.trainer import JointTrainingConfig

OPERATIONS = Path(__file__).parents[2] / "docs" / "LEARNED_TRAINING_OPERATIONS.md"


def _commands() -> list[list[str]]:
    blocks = OPERATIONS.read_text().split("```bash\n")[1:]
    return [
        shlex.split(line)
        for block in blocks
        for line in block.split("```", 1)[0].replace("\\\n", " ").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("command", _commands(), ids=lambda cmd: cmd[cmd.index("-m") + 1])
def test_documented_tool_help_commands_work(command: list[str]) -> None:
    # During the reset the operations guide deliberately documents discovery,
    # not runnable legacy generation recipes. Exercise the real CLI boundary.
    args = command[command.index("-m") :]
    assert args[-1] == "--help"
    result = subprocess.run(
        [sys.executable, *args],
        cwd=OPERATIONS.parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def _required_training_args(tmp_path: Path) -> list[str]:
    return [
        "--dataset",
        str(tmp_path / "dataset.jsonl"),
        "--split-manifest",
        str(tmp_path / "split.json"),
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--run-manifest",
        str(tmp_path / "run.json"),
        "--artifact",
        str(tmp_path / "artifact"),
        "--seed",
        "20000",
    ]


def test_training_cli_preserves_existing_config_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def capture(config: JointTrainingConfig, *, show_progress: bool) -> None:
        captured.update(config=config, show_progress=show_progress)

    monkeypatch.setattr(trainer, "train_joint", capture)

    assert trainer.main(_required_training_args(tmp_path)) == 0

    assert captured == {
        "config": JointTrainingConfig(
            dataset_path=tmp_path / "dataset.jsonl",
            split_manifest_path=tmp_path / "split.json",
            checkpoint_path=tmp_path / "checkpoint.pt",
            run_manifest_path=tmp_path / "run.json",
            artifact_path=tmp_path / "artifact",
            seed=20000,
        ),
        "show_progress": True,
    }


def test_training_cli_forwards_pilot_regularization_and_game_mode_holdouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def capture(config: JointTrainingConfig, *, show_progress: bool) -> None:
        captured.update(config=config, show_progress=show_progress)

    monkeypatch.setattr(trainer, "train_joint", capture)
    args = [
        *_required_training_args(tmp_path),
        "--dropout",
        "0.15",
        "--entropy-weight",
        "0.025",
        "--l2-weight",
        "0.0001",
        "--value-weight",
        "1.5",
        "--validation-fraction",
        "0.25",
        "--holdout-game-mode",
        "QUICK",
        "--holdout-game-mode",
        "LONG",
        "--no-progress",
    ]

    assert trainer.main(args) == 0

    config = captured["config"]
    assert isinstance(config, JointTrainingConfig)
    assert config.dropout == 0.15
    assert config.entropy_weight == 0.025
    assert config.l2_weight == 0.0001
    assert config.value_weight == 1.5
    assert config.validation_fraction == 0.25
    assert config.holdout_game_modes == ("QUICK", "LONG")
    assert captured["show_progress"] is False
