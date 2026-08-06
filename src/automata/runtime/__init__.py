"""Runtime glue to the goa2 engine: effect registration, cloning,
determinization, and the headless self-play harness."""

from .clone import clone_state
from .determinize import determinize
from .driver import (
    BotDecision,
    DecisionKind,
    IllegalBotDecisionError,
    apply_decision,
    inspect_next_decision,
)
from .effects import register_all_effects
from .harness import DEFAULT_MAP, RunResult, run_game
from .trajectory import (
    InMemoryRecorder,
    JsonlRecorder,
    NullRecorder,
    TrajectoryRecorder,
)

__all__ = [
    "DEFAULT_MAP",
    "BotDecision",
    "DecisionKind",
    "IllegalBotDecisionError",
    "InMemoryRecorder",
    "JsonlRecorder",
    "NullRecorder",
    "RunResult",
    "TrajectoryRecorder",
    "apply_decision",
    "clone_state",
    "determinize",
    "inspect_next_decision",
    "register_all_effects",
    "run_game",
]
