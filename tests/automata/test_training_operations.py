"""Keep documented training commands aligned with their CLI parsers."""

import shlex
from pathlib import Path
from typing import Any

import pytest

from automata.scripts.generate_joint_bootstrap import _parser as bootstrap_parser
from automata.training import trainer
from automata.training.trainer import JointTrainingConfig
from automata.training.trainer import _parser as training_parser

OPERATIONS = Path(__file__).parents[2] / "docs" / "LEARNED_TRAINING_OPERATIONS.md"


def _commands() -> list[list[str]]:
    blocks = OPERATIONS.read_text().split("```bash\n")[1:]
    return [shlex.split(block.split("```", 1)[0].replace("\\\n", " ")) for block in blocks]


def test_documented_bootstrap_and_training_commands_parse() -> None:
    commands = _commands()
    bootstrap = next(
        command for command in commands if "automata.scripts.generate_joint_bootstrap" in command
    )
    training_commands = [command for command in commands if "automata.training.trainer" in command]

    bootstrap_parser().parse_args(
        bootstrap[bootstrap.index("automata.scripts.generate_joint_bootstrap") + 1 :]
    )
    for training in training_commands:
        training_parser().parse_args(training[training.index("automata.training.trainer") + 1 :])


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
