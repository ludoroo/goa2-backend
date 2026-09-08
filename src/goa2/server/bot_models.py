"""Persisted bot metadata for server-managed AI heroes.

`BotSpec` is the serializable configuration for a bot-controlled hero. It lives
in `ManagedGame.bot_specs` (see `server/registry.py`) and is round-tripped
through the on-disk save (see `engine/persistence.py`). It intentionally does
**not** carry live agent instances, RNG state, or asyncio tasks — those are
runtime-only concerns owned by the bot coordinator.

Bounded ISMCTS settings
-----------------------

``SearchSettings`` exposes only the two knobs that meaningfully cap resource
use — ``iterations`` and ``decision_timeout_seconds`` — and validates each
against the production limits declared in
:mod:`automata.search.config` (``PROD_MIN_*`` / ``PROD_MAX_*``). The coordinator
uses these values to build a bounded :class:`ISMCTSAgent`; on queue timeout,
search timeout, exception, or invalid output the coordinator falls back to a
cached :class:`HeuristicAgent` on the cloned decision state.

Deeper algorithmic knobs (UCT c, widening, priors) stay internal to
:class:`automata.search.config.SearchConfig` — they are not part of the
persistence surface or the public request contract.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from automata.search.config import (
    PROD_DEFAULT_DECISION_TIMEOUT_SECONDS,
    PROD_DEFAULT_ITERATIONS,
    PROD_MAX_DECISION_TIMEOUT_SECONDS,
    PROD_MAX_ITERATIONS,
    PROD_MIN_DECISION_TIMEOUT_SECONDS,
    PROD_MIN_ITERATIONS,
)

# Supported agent kinds. Random and Heuristic are the always-on cheap agents;
# ISMCTS is opt-in with bounded execution.
BotKind = Literal["random", "heuristic", "ismcts"]


class ModelArtifactSpec(BaseModel):
    """Server-local immutable model identity; the model family is not persisted."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    reference: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("reference")
    @classmethod
    def _safe_relative_reference(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            value != value.strip()
            or "\\" in value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or (path.parts and ":" in path.parts[0])
        ):
            raise ValueError("artifact reference must be a safe relative path")
        return value


class SearchSettings(BaseModel):
    """Bounded ISMCTS knobs safe to persist and accept from clients.

    Only fields that meaningfully cap resource use are exposed here. Deeper
    algorithmic knobs (UCT c, widening, priors) stay internal to
    :class:`automata.search.config.SearchConfig`.

    Bounds are the production limits declared in
    :mod:`automata.search.config`; both the request-boundary schema
    (:class:`goa2.server.models.CreateBotSpec`) and this internal model use
    the same numbers so a restored save cannot violate them either.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    iterations: int = Field(
        default=PROD_DEFAULT_ITERATIONS,
        ge=PROD_MIN_ITERATIONS,
        le=PROD_MAX_ITERATIONS,
    )
    decision_timeout_seconds: float = Field(
        default=PROD_DEFAULT_DECISION_TIMEOUT_SECONDS,
        ge=PROD_MIN_DECISION_TIMEOUT_SECONDS,
        le=PROD_MAX_DECISION_TIMEOUT_SECONDS,
    )
    policy_source: Literal["heuristic", "learned"] = "heuristic"
    value_source: Literal["heuristic", "learned"] = "heuristic"
    leaf_mode: Literal["immediate", "bounded_continuation"] = "immediate"
    horizon: int = Field(default=2, ge=0, le=10)
    artifact: ModelArtifactSpec | None = None

    @model_validator(mode="after")
    def _learned_sources_require_artifact(self) -> SearchSettings:
        learned = "learned" in {self.policy_source, self.value_source}
        if learned != (self.artifact is not None):
            raise ValueError("artifact is required exactly when a learned source is configured")
        return self


class BotSpec(BaseModel):
    """Serializable declaration that a specific hero is bot-controlled.

    Attributes:
        kind: Which agent implementation controls the hero.
        search: Optional bounded search settings. Only meaningful for
            ``kind == "ismcts"``; supplying it for other kinds is rejected so
            that clients cannot accidentally imply cost that will not apply.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BotKind
    search: SearchSettings | None = None

    @model_validator(mode="after")
    def _search_only_for_ismcts(self) -> BotSpec:
        if self.search is not None and self.kind != "ismcts":
            raise ValueError("search settings are only valid for kind='ismcts'")
        return self
