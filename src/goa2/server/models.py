"""API request/response Pydantic models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from goa2.domain.time_control import TimeControlConfig
from goa2.server.bot_models import SearchSettings


def _reject_bots_key(data: Any) -> Any:
    """Reject a top-level ``bots`` key on draft request bodies.

    Draft-created games do **not** support bot configuration — bots are a
    ``POST /games`` concern only. Rather than silently dropping the field
    through the default ``extra="ignore"`` policy, we intercept it here so a
    client that thought it could bot-fill a draft gets a clear, targeted
    validation error naming the unsupported field. This is deliberately
    *not* a global ``extra="forbid"`` — unrelated forward-compat fields
    must continue to be silently ignored so a schema-additive client
    change does not break older servers.

    Called as a ``@model_validator(mode="before")`` on every draft request
    that accepts a body from the network.
    """
    if isinstance(data, dict) and "bots" in data:
        raise ValueError(
            "'bots' is not supported on draft requests; configure bots via " "POST /games instead"
        )
    return data


# -- Requests --


class CreateBotSpec(BaseModel):
    """Public, request-only bot configuration.

    Public schema:

    - ``kind`` accepts ``"random"``, ``"heuristic"``, and ``"ismcts"``. Any
      other value is rejected at Pydantic validation time (422) rather than
      at the route boundary, so the generated OpenAPI schema advertises only
      the kinds clients may actually use.
    - ``search`` is optional and is **only** valid when ``kind == "ismcts"``.
      Supplying it for any other kind fails validation with a
      clear 422. Omitting it on an ``"ismcts"`` request uses the production
      defaults declared in :mod:`automata.search.config`. The bounds on
      each field (``iterations`` / ``decision_timeout_seconds``) come from
      the shared :class:`goa2.server.bot_models.SearchSettings` model.
    - ``extra="forbid"`` catches typos and forward-compat guesses at the
      wire boundary.

    Routes translate a :class:`CreateBotSpec` into an internal
    :class:`goa2.server.bot_models.BotSpec` before persistence.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["random", "heuristic", "ismcts"]
    search: SearchSettings | None = None

    @model_validator(mode="after")
    def _search_only_for_ismcts(self) -> CreateBotSpec:
        if self.search is not None and self.kind != "ismcts":
            raise ValueError("search settings are only valid for kind='ismcts'")
        return self


class CreateGameRequest(BaseModel):
    map_name: str = "forgotten_island"
    red_heroes: list[str]
    blue_heroes: list[str]
    cheats_enabled: bool = False
    game_type: str = "LONG"
    time_control: TimeControlConfig | None = None
    # Optional server-managed bot configuration keyed by hero_id (e.g.
    # ``"hero_wasp"``). Every referenced hero must be present in
    # ``red_heroes`` / ``blue_heroes``. Uses the public :class:`CreateBotSpec`
    # schema, which advertises ``kind='random'``, ``kind='heuristic'``, and
    # ``kind='ismcts'``. Omitting the field or providing an empty mapping
    # leaves the game fully human-controlled — the response shape is
    # unchanged either way.
    bots: dict[str, CreateBotSpec] | None = None


class ReadyRequest(BaseModel):
    ready: bool = True


class CommitCardRequest(BaseModel):
    card_id: str


class SubmitInputRequest(BaseModel):
    request_id: str = Field(min_length=1)
    selection: Any = None


class GiveGoldRequest(BaseModel):
    hero_id: str
    amount: int


class SubmitBugReportRequest(BaseModel):
    title: str
    description: str = ""

    @field_validator("title")
    @classmethod
    def _validate_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be empty")
        if len(v) > 120:
            raise ValueError("title must be at most 120 characters")
        return v

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str) -> str:
        v = v.strip()
        if len(v) > 4000:
            raise ValueError("description must be at most 4000 characters")
        return v


# -- Responses --


class HeroMetadata(BaseModel):
    id: str
    difficulty_stars: int


class PlayerToken(BaseModel):
    hero_id: str
    token: str


class CreateGameResponse(BaseModel):
    game_id: str
    player_tokens: list[PlayerToken]
    spectator_token: str


class GameViewResponse(BaseModel):
    """Wraps the dict returned by build_view."""

    view: dict[str, Any]
    input_request: dict[str, Any] | None = None
    # Public identity of whoever the pending request is waiting on, even for a
    # recipient whose `input_request` was withheld as private.
    awaiting_input: list[str] = []
    winner: str | None = None


class ActionResultResponse(BaseModel):
    result_type: str
    current_phase: str
    events: list[dict[str, Any]] = []
    input_request: dict[str, Any] | None = None
    awaiting_input: list[str] = []
    winner: str | None = None


class ErrorResponse(BaseModel):
    detail: str


class BugReportResponse(BaseModel):
    id: str
    game_id: str
    title: str
    description: str
    reporter_hero: str | None
    decision_index: int | None
    round: int
    turn: int
    status: str
    created_at: float
    resolved_at: float | None


# -- Draft requests --


class CreateDraftRequest(BaseModel):
    """Open a lobby. All match settings (map, game type, draft mode, cheats) and team
    sizes are decided inside the lobby, not here. The optional fields seed the defaults."""

    host_name: str
    map_name: str = "forgotten_island"
    game_type: str = "LONG"
    draft_mode: str = "sequential_ban_pick"
    cheats_enabled: bool = False
    max_hero_stars: int = 4
    time_control: TimeControlConfig | None = None

    # Targeted rejection of a top-level ``bots`` key. See
    # :func:`_reject_bots_key`. Uses ``mode="before"`` so the raw payload is
    # inspected *before* Pydantic's default ``extra="ignore"`` policy would
    # otherwise silently drop the field.
    _reject_bots = model_validator(mode="before")(_reject_bots_key)


class UpdateDraftSettingsRequest(BaseModel):
    """Host-only, LOBBY-only. Any omitted field is left unchanged."""

    map_name: str | None = None
    game_type: str | None = None
    draft_mode: str | None = None
    cheats_enabled: bool | None = None
    max_hero_stars: int | None = None
    # Explicit null disables timers; omission leaves the setting unchanged.
    time_control: TimeControlConfig | None = None

    # Same targeted ``bots``-key rejection as :class:`CreateDraftRequest`.
    # Draft settings updates cannot introduce bot configuration either.
    _reject_bots = model_validator(mode="before")(_reject_bots_key)


class JoinDraftRequest(BaseModel):
    display_name: str


class SetTeamRequest(BaseModel):
    team: str  # "RED" | "BLUE"


class SetCaptainRequest(BaseModel):
    player_id: str


class DraftActionRequest(BaseModel):
    hero: str


class ClaimHeroRequest(BaseModel):
    hero: str


# -- Draft responses --


class DraftModeInfo(BaseModel):
    name: str
    description: str


class CreateDraftResponse(BaseModel):
    draft_id: str
    player_id: str
    player_token: str
    spectator_token: str


class JoinDraftResponse(BaseModel):
    draft_id: str
    player_id: str
    player_token: str


class DraftViewResponse(BaseModel):
    draft: dict[str, Any]
    you: dict[str, Any] | None = None
    game_id: str | None = None
    game_token: str | None = None


class OverrideOpSchema(BaseModel):
    name: str
    family: str  # "patch" | "unstick"
    label: str
    description: str
    args_schema: dict  # JSON Schema for the op's args


class OverrideSchemaResponse(BaseModel):
    ops: list[OverrideOpSchema]


class OverrideHistoryEntry(BaseModel):
    index: int  # raw decision index (rewind targets use this space)
    type: str  # replay record type (commit / input / ov_patch / ...)
    round: int | None = None
    turn: int | None = None
    hero_id: str | None = None
    label: str  # human-readable, viewer-scoped
    superseded: bool = False  # dead segment behind a rewind


class OverrideHistoryResponse(BaseModel):
    total: int
    decisions: list[OverrideHistoryEntry]
