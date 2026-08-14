"""Pure models and arithmetic for optional match time controls.

The server coordinator decides which clocks run and applies timeout fallbacks.
This module only owns persisted values and deterministic millisecond arithmetic,
so it is testable without asyncio or wall-clock sleeps.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

MAX_TIME_CONTROL_SECONDS = 24 * 60 * 60


class TimeControlConfig(BaseModel):
    planning_allowance_seconds: int = Field(ge=0, le=MAX_TIME_CONTROL_SECONDS)
    resolution_allowance_seconds: int = Field(ge=0, le=MAX_TIME_CONTROL_SECONDS)
    # One-shot allowance for the first primary Resolution actor of each
    # shared turn. A default keeps older saved configurations valid.
    initiative_bonus_seconds: int = Field(default=0, ge=0, le=MAX_TIME_CONTROL_SECONDS)
    response_grant_seconds: int = Field(ge=0, le=MAX_TIME_CONTROL_SECONDS)
    initial_time_bank_seconds: int = Field(ge=0, le=MAX_TIME_CONTROL_SECONDS)
    time_bank_increment_seconds: int = Field(ge=0, le=MAX_TIME_CONTROL_SECONDS)
    max_time_bank_seconds: int = Field(ge=0, le=MAX_TIME_CONTROL_SECONDS)
    upgrade_allowance_seconds: int = Field(ge=0, le=MAX_TIME_CONTROL_SECONDS)
    # Consecutive shared turns with no accepted human decision before the
    # match suspends. Zero deliberately disables abandonment suspension.
    automatic_turn_limit: int = Field(default=2, ge=0, le=100)

    @model_validator(mode="after")
    def validate_bank_cap(self) -> TimeControlConfig:
        if self.max_time_bank_seconds < self.initial_time_bank_seconds:
            raise ValueError(
                "max_time_bank_seconds must be greater than or equal to "
                "initial_time_bank_seconds"
            )
        return self

    @staticmethod
    def milliseconds(seconds: int) -> int:
        return seconds * 1000

    @property
    def planning_allowance_ms(self) -> int:
        return self.milliseconds(self.planning_allowance_seconds)

    @property
    def resolution_allowance_ms(self) -> int:
        return self.milliseconds(self.resolution_allowance_seconds)

    @property
    def initiative_bonus_ms(self) -> int:
        return self.milliseconds(self.initiative_bonus_seconds)

    @property
    def response_grant_ms(self) -> int:
        return self.milliseconds(self.response_grant_seconds)

    @property
    def initial_time_bank_ms(self) -> int:
        return self.milliseconds(self.initial_time_bank_seconds)

    @property
    def time_bank_increment_ms(self) -> int:
        return self.milliseconds(self.time_bank_increment_seconds)

    @property
    def max_time_bank_ms(self) -> int:
        return self.milliseconds(self.max_time_bank_seconds)

    @property
    def upgrade_allowance_ms(self) -> int:
        return self.milliseconds(self.upgrade_allowance_seconds)


class ClockStatus(StrEnum):
    WAITING_FOR_PLAYERS = "WAITING_FOR_PLAYERS"
    SUSPENDED_FOR_INACTIVITY = "SUSPENDED_FOR_INACTIVITY"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class ClockKind(StrEnum):
    PLANNING = "PLANNING"
    RESOLUTION = "RESOLUTION"
    RESPONSE = "RESPONSE"
    LEVEL_UP = "LEVEL_UP"


class PlayerClockState(BaseModel):
    planning_allowance_ms: int = Field(default=0, ge=0)
    resolution_allowance_ms: int = Field(default=0, ge=0)
    initiative_bonus_ms: int = Field(default=0, ge=0)
    response_time_ms: int = Field(default=0, ge=0)
    upgrade_allowance_ms: int = Field(default=0, ge=0)
    time_bank_ms: int = Field(default=0, ge=0)
    planning_complete: bool = False
    planning_locked_by_timeout: bool = False


class GameClockState(BaseModel):
    status: ClockStatus = ClockStatus.WAITING_FOR_PLAYERS
    ready_hero_ids: list[str] = Field(default_factory=list)
    turn_round: int
    turn_number: int
    initiative_bonus_hero_id: str | None = None
    level_up_round: int | None = None
    players: dict[str, PlayerClockState] = Field(default_factory=dict)
    active_kind: ClockKind | None = None
    active_hero_ids: list[str] = Field(default_factory=list)
    active_decision_hero_ids: list[str] = Field(default_factory=list)
    active_request_id: str | None = None
    credited_response_request_ids: list[str] = Field(default_factory=list)
    human_action_seen_this_turn: bool = False
    consecutive_automatic_turns: int = Field(default=0, ge=0)
    last_settled_at_ms: int | None = None
    revision: int = 0


def create_game_clock(
    config: TimeControlConfig,
    hero_ids: Iterable[str],
    *,
    round_number: int,
    turn_number: int,
) -> GameClockState:
    """Create the persisted waiting-state clock for a timed match."""
    return GameClockState(
        turn_round=round_number,
        turn_number=turn_number,
        players={
            str(hero_id): PlayerClockState(
                planning_allowance_ms=config.planning_allowance_ms,
                resolution_allowance_ms=config.resolution_allowance_ms,
                time_bank_ms=config.initial_time_bank_ms,
            )
            for hero_id in hero_ids
        },
    )


def start_game_clock(clock: GameClockState, now_ms: int) -> None:
    clock.status = ClockStatus.RUNNING
    clock.last_settled_at_ms = now_ms
    clock.revision += 1


def finish_game_clock(clock: GameClockState, now_ms: int) -> None:
    settle_clock(clock, now_ms)
    clock.status = ClockStatus.FINISHED
    clock.active_kind = None
    clock.active_hero_ids = []
    clock.active_decision_hero_ids = []
    clock.active_request_id = None
    clock.last_settled_at_ms = now_ms
    clock.revision += 1


def begin_shared_turn(
    clock: GameClockState,
    config: TimeControlConfig,
    *,
    round_number: int,
    turn_number: int,
    now_ms: int,
) -> bool:
    """Reset turn allowances and grant the fixed Time Bank increment once."""
    if (clock.turn_round, clock.turn_number) == (round_number, turn_number):
        return False

    settle_clock(clock, now_ms)
    clock.turn_round = round_number
    clock.turn_number = turn_number
    clock.level_up_round = None
    clock.initiative_bonus_hero_id = None
    clock.credited_response_request_ids = []
    clock.human_action_seen_this_turn = False
    for player in clock.players.values():
        player.planning_allowance_ms = config.planning_allowance_ms
        player.resolution_allowance_ms = config.resolution_allowance_ms
        player.initiative_bonus_ms = 0
        player.response_time_ms = 0
        player.upgrade_allowance_ms = 0
        player.planning_complete = False
        player.planning_locked_by_timeout = False
        player.time_bank_ms = min(
            config.max_time_bank_ms,
            player.time_bank_ms + config.time_bank_increment_ms,
        )
    clock.active_kind = None
    clock.active_hero_ids = []
    clock.active_decision_hero_ids = []
    clock.active_request_id = None
    clock.last_settled_at_ms = now_ms
    clock.revision += 1
    return True


def grant_initiative_bonus(
    clock: GameClockState,
    config: TimeControlConfig,
    hero_id: str,
) -> bool:
    """Grant the one-shot bonus to the first primary Resolution actor."""
    if clock.initiative_bonus_hero_id is not None:
        return False
    player = clock.players.get(str(hero_id))
    if player is None:
        return False
    clock.initiative_bonus_hero_id = str(hero_id)
    player.initiative_bonus_ms = config.initiative_bonus_ms
    clock.revision += 1
    return True


def suspend_game_clock_for_inactivity(clock: GameClockState, now_ms: int) -> None:
    """Pause an abandoned match at a shared-turn boundary."""
    settle_clock(clock, now_ms)
    clock.status = ClockStatus.SUSPENDED_FOR_INACTIVITY
    clock.ready_hero_ids = []
    clock.active_kind = None
    clock.active_hero_ids = []
    clock.active_decision_hero_ids = []
    clock.active_request_id = None
    clock.last_settled_at_ms = now_ms
    clock.revision += 1


def begin_level_up(
    clock: GameClockState,
    config: TimeControlConfig,
    *,
    round_number: int,
) -> bool:
    """Grant one Upgrade Allowance per player for this Level Up phase."""
    if clock.level_up_round == round_number:
        return False
    for player in clock.players.values():
        player.upgrade_allowance_ms = config.upgrade_allowance_ms
    clock.level_up_round = round_number
    clock.revision += 1
    return True


def grant_response_time(
    clock: GameClockState,
    config: TimeControlConfig,
    request_id: str,
    hero_ids: Iterable[str],
) -> bool:
    """Credit a genuine request once, never once per view or submission."""
    if request_id in clock.credited_response_request_ids:
        return False
    for hero_id in hero_ids:
        player = clock.players.get(str(hero_id))
        if player is not None:
            player.response_time_ms += config.response_grant_ms
    clock.credited_response_request_ids.append(request_id)
    clock.revision += 1
    return True


def activate_clocks(
    clock: GameClockState,
    kind: ClockKind | None,
    hero_ids: Iterable[str] = (),
    *,
    request_id: str | None,
    now_ms: int,
    decision_hero_ids: Iterable[str] | None = None,
) -> None:
    """Set current targets after settling the previous targets."""
    settle_clock(clock, now_ms)
    next_ids = [str(hero_id) for hero_id in hero_ids if str(hero_id) in clock.players]
    next_decision_ids = (
        [str(hero_id) for hero_id in decision_hero_ids if str(hero_id) in clock.players]
        if decision_hero_ids is not None
        else list(next_ids)
    )
    changed = (
        clock.active_kind != kind
        or clock.active_hero_ids != next_ids
        or clock.active_decision_hero_ids != next_decision_ids
        or clock.active_request_id != request_id
    )
    clock.active_kind = kind
    clock.active_hero_ids = next_ids
    clock.active_decision_hero_ids = next_decision_ids
    clock.active_request_id = request_id
    clock.last_settled_at_ms = now_ms
    if changed:
        clock.revision += 1


def _spending_fields(kind: ClockKind) -> tuple[str, ...]:
    if kind == ClockKind.PLANNING:
        return ("planning_allowance_ms", "time_bank_ms")
    if kind == ClockKind.RESOLUTION:
        return ("initiative_bonus_ms", "resolution_allowance_ms", "time_bank_ms")
    if kind == ClockKind.RESPONSE:
        return ("response_time_ms", "resolution_allowance_ms", "time_bank_ms")
    return ("upgrade_allowance_ms", "time_bank_ms")


def usable_time_ms(player: PlayerClockState, kind: ClockKind) -> int:
    return sum(int(getattr(player, field)) for field in _spending_fields(kind))


def spend_time(player: PlayerClockState, kind: ClockKind, elapsed_ms: int) -> None:
    remaining = max(0, elapsed_ms)
    for field in _spending_fields(kind):
        available = int(getattr(player, field))
        used = min(available, remaining)
        setattr(player, field, available - used)
        remaining -= used
        if remaining == 0:
            break


def settle_clock(clock: GameClockState, now_ms: int) -> int:
    """Charge every active personal clock to ``now_ms`` and return elapsed."""
    previous = clock.last_settled_at_ms
    elapsed = 0 if previous is None else max(0, now_ms - previous)
    if elapsed and clock.status == ClockStatus.RUNNING and clock.active_kind is not None:
        for hero_id in clock.active_hero_ids:
            player = clock.players.get(hero_id)
            if player is not None:
                spend_time(player, clock.active_kind, elapsed)
    clock.last_settled_at_ms = now_ms
    return elapsed


def exhausted_active_hero_ids(clock: GameClockState) -> list[str]:
    if clock.active_kind is None:
        return []
    return [
        hero_id
        for hero_id in clock.active_hero_ids
        if (player := clock.players.get(hero_id)) is not None
        and usable_time_ms(player, clock.active_kind) == 0
    ]


def milliseconds_until_next_exhaustion(clock: GameClockState) -> int | None:
    if (
        clock.status != ClockStatus.RUNNING
        or clock.active_kind is None
        or not clock.active_hero_ids
    ):
        return None
    values = [
        usable_time_ms(player, clock.active_kind)
        for hero_id in clock.active_hero_ids
        if (player := clock.players.get(hero_id)) is not None
    ]
    return min(values) if values else None


def public_clock_view(clock: GameClockState, now_ms: int) -> dict:
    """Return an extrapolated public snapshot without mutating persisted state."""
    projected = clock.model_copy(deep=True)
    settle_clock(projected, now_ms)
    active = None
    if projected.active_kind is not None:
        active = {
            "kind": projected.active_kind.value,
            "request_id": projected.active_request_id,
            "hero_ids": list(projected.active_decision_hero_ids),
            "running_hero_ids": list(projected.active_hero_ids),
        }
    active_ids = set(projected.active_hero_ids)
    return {
        "status": projected.status.value,
        "server_now_ms": now_ms,
        "turn_key": {"round": projected.turn_round, "turn": projected.turn_number},
        "ready_hero_ids": list(projected.ready_hero_ids),
        "active": active,
        "players": {
            hero_id: {
                **player.model_dump(mode="json"),
                "running": (
                    projected.status == ClockStatus.RUNNING
                    and projected.active_kind is not None
                    and hero_id in active_ids
                ),
            }
            for hero_id, player in projected.players.items()
        },
    }
