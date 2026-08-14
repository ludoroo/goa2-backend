"""Steps nested inside another step must never alias a live step.

A GameStep is both the declarative recipe and the runtime state machine, so a
list handed to `steps_template`/`finishing_steps` while the same objects also run
inline would let a repeat inherit answered inputs and mid-flight counters.
"""

import pytest

from goa2.domain.models.effect import DurationType, EffectScope, EffectType
from goa2.domain.models.enums import TargetType
from goa2.engine.steps.base import isolate_steps
from goa2.engine.steps.effects import CreateEffectStep
from goa2.engine.steps.selection import SelectStep
from goa2.engine.steps.utility import MayRepeatOnceStep
from tests.engine.effects.builders import EffectScenarioBuilder


def _select() -> SelectStep:
    return SelectStep(target_type=TargetType.HEX, output_key="k", prompt="p")


def test_constructor_isolates_a_step_list() -> None:
    inner = [_select()]

    step = MayRepeatOnceStep(steps_template=inner)

    assert step.steps_template[0] is not inner[0]


@pytest.mark.parametrize(
    ("wrap", "read"),
    [
        (tuple, lambda out: out[0]),
        (lambda inner: {"yes": inner}, lambda out: out["yes"][0]),
    ],
    ids=["sequence-field-validates-to-a-tuple", "dict-of-step-lists"],
)
def test_isolation_covers_non_list_containers(wrap, read) -> None:
    """A step field typed `Sequence[GameStep]` validates to a *tuple* and a
    branching step would nest lists in a dict; neither may slip past the walk
    just because it is not a plain list.

    Exercised through the helper rather than a throwaway GameStep subclass:
    subclassing registers the class in the auto-derived `AnyStep` union for the
    whole session and breaks `test_step_registry_covers_concrete_step_classes`.
    """
    step = _select()

    out = isolate_steps(wrap([step]))

    assert read(out) is not step


def test_containers_without_steps_are_returned_unchanged() -> None:
    """The walk must not churn ordinary values — hex lists, id maps, filters."""
    data = [1, {"a": ("b", "c")}]

    assert isolate_steps(data) is data


def test_created_effect_does_not_share_steps_with_the_step_that_made_it() -> None:
    """`resolve_steps` rebuilds via `model_copy`, which skips validators, and
    `Effect.finishing_steps` is a plain `list[Any]` — so `CreateEffectStep`
    isolates at the boundary itself."""
    state = (
        EffectScenarioBuilder()
        .small_arena()
        .red_hero("hero_a", at=(0, 0, 0))
        .with_actor("hero_a")
        .build()
    )
    inner = _select()
    step = CreateEffectStep(
        effect_type=EffectType.DELAYED_TRIGGER,
        scope=EffectScope(shape="global"),
        duration=DurationType.THIS_TURN,
        finishing_steps=[inner, MayRepeatOnceStep(steps_template=[inner])],
    )

    step.resolve(state, {})

    made = state.active_effects[-1].finishing_steps
    own = step.finishing_steps
    assert made[0] is not own[0]
    assert made[1].steps_template[0] is not own[1].steps_template[0]
