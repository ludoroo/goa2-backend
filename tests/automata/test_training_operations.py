"""Keep documented training commands aligned with their CLI parsers."""

import shlex
from pathlib import Path

from automata.scripts.generate_joint_bootstrap import _parser as bootstrap_parser
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
    training = next(command for command in commands if "automata.training.trainer" in command)

    bootstrap_parser().parse_args(
        bootstrap[bootstrap.index("automata.scripts.generate_joint_bootstrap") + 1 :]
    )
    training_parser().parse_args(training[training.index("automata.training.trainer") + 1 :])
