"""Base classes for the step engine: StepResult and GameStep."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from goa2.domain.events import GameEvent
from goa2.domain.input import InputRequest
from goa2.domain.models.enums import StepType
from goa2.domain.state import GameState


class StepResult(BaseModel):
    """Result of a step execution."""

    is_finished: bool = True
    requires_input: bool = False
    input_request: InputRequest | None = None
    new_steps: Sequence[GameStep] = Field(default_factory=list)
    abort_action: bool = False
    events: list[GameEvent] = Field(default_factory=list)


class GameStep(BaseModel):
    """
    Base class for all atomic game operations.
    Each step performs a single logic unit and can manage its own state.
    """

    type: StepType = StepType.GENERIC
    # Nested actions normally discard every remaining effect step when a
    # mandatory step aborts. Stateful sequences that must finish after the
    # nested action resolves or fizzles may opt into being an abort boundary.
    survives_action_abort: ClassVar[bool] = False

    step_id: str = Field(default_factory=lambda: str(id(object())))
    pending_input: Any | None = None
    pending_request_id: str | None = None
    is_mandatory: bool = True
    active_if_key: str | None = None
    skip_if_key: str | None = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Fail loudly if a step subclass forgets to set a unique `type`.

        The `AnyStep` serialization union is auto-derived from `GameStep`
        subclasses but excludes `StepType.GENERIC`. A concrete step left at the
        default GENERIC would run fine yet silently drop out of the union, so it
        could never round-trip through JSON (persistence breaks). Turn that
        silent failure into a loud import-time error.
        """
        super().__pydantic_init_subclass__(**kwargs)
        if cls.model_fields["type"].default == StepType.GENERIC:
            raise TypeError(
                f"{cls.__name__} must set a unique `type: StepType = StepType.X` "
                f"default. Leaving it StepType.GENERIC excludes the step from the "
                f"serialization union, so it cannot round-trip through JSON "
                f"(persistence will silently break). Add a StepType enum value in "
                f"domain/models/enums.py and set it as this class's `type` default."
            )

    @model_validator(mode="after")
    def _isolate_nested_steps(self) -> GameStep:
        """Deep-copy nested steps so a template never aliases a live step.

        A GameStep is both the declarative recipe and the runtime state machine:
        `resolve()` writes back onto `self` (pending_input, counters, flags). The
        idiom `[*steps, MayRepeatNTimesStep(steps_template=steps)]` therefore hands
        the repeat the *same objects* that just ran, and the spawn-time deepcopy
        clones their dirty state. Copying on the way in makes the template pristine
        regardless of which fields a step happens to mutate.

        The walk covers lists, tuples, dicts and bare step fields so a container
        shape nobody has used yet (``Sequence[GameStep]`` validates to a *tuple*,
        ``dict[str, list[GameStep]]`` for a branching step) cannot silently opt out
        of the isolation.

        The guarantee is **construction-scoped**. ``model_copy`` and
        ``model_construct`` bypass validators by design, so a step rebuilt that way
        re-aliases whatever list it is handed, and ``model_copy`` without ``update``
        shallow-shares every container it did not replace. Code that rebuilds steps
        outside the constructor — or hands them to a non-GameStep model, such as
        ``Effect.finishing_steps`` (a plain ``list[Any]``) — must call
        :func:`isolate_steps` itself. ``CreateEffectStep.resolve()`` is the one
        place in the engine that does.
        """
        for name in type(self).model_fields:
            self.__dict__[name] = isolate_steps(self.__dict__.get(name))
        return self

    def should_skip(self, context: dict[str, Any]) -> bool:
        """Checks if the step should be skipped based on active_if_key or skip_if_key."""
        if self.active_if_key:
            val = context.get(self.active_if_key)
            if not val:
                return True
        if self.skip_if_key:
            val = context.get(self.skip_if_key)
            if val is not None:
                return True
        return False

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        """
        Executes the step.
        :param state: Global GameState.
        :param context: Shared transient memory for the current Action chain.
        :return: StepResult indicating if we are done or need input.
        """
        raise NotImplementedError


def isolate_steps(value: Any) -> Any:
    """Return `value` with every nested GameStep replaced by a deep copy.

    Containers without a step inside are returned unchanged (identity preserved),
    so this stays a no-op for the ordinary fields — ids, filters, hex lists.
    """
    if isinstance(value, GameStep):
        return copy.deepcopy(value)
    if isinstance(value, (list, tuple)):
        items = [isolate_steps(item) for item in value]
        if all(new is old for new, old in zip(items, value, strict=True)):
            return value
        return type(value)(items) if isinstance(value, tuple) else items
    if isinstance(value, dict):
        items_map = {key: isolate_steps(item) for key, item in value.items()}
        if all(new is value[key] for key, new in items_map.items()):
            return value
        return items_map
    return value
