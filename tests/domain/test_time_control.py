import pytest
from pydantic import ValidationError

from goa2.domain.time_control import (
    ClockKind,
    ClockStatus,
    TimeControlConfig,
    activate_clocks,
    begin_level_up,
    begin_shared_turn,
    create_game_clock,
    exhausted_active_hero_ids,
    grant_initiative_bonus,
    grant_response_time,
    milliseconds_until_next_exhaustion,
    public_clock_view,
    settle_clock,
    start_game_clock,
    usable_time_ms,
)


@pytest.fixture
def config() -> TimeControlConfig:
    return TimeControlConfig(
        planning_allowance_seconds=10,
        resolution_allowance_seconds=20,
        response_grant_seconds=5,
        initial_time_bank_seconds=30,
        time_bank_increment_seconds=7,
        max_time_bank_seconds=40,
        upgrade_allowance_seconds=12,
    )


def _clock(config: TimeControlConfig):
    return create_game_clock(
        config,
        ["hero_a", "hero_b"],
        round_number=1,
        turn_number=1,
    )


def test_config_rejects_bank_cap_below_initial() -> None:
    with pytest.raises(ValidationError, match="max_time_bank_seconds"):
        TimeControlConfig(
            planning_allowance_seconds=1,
            resolution_allowance_seconds=1,
            response_grant_seconds=1,
            initial_time_bank_seconds=10,
            time_bank_increment_seconds=1,
            max_time_bank_seconds=9,
            upgrade_allowance_seconds=1,
        )


def test_automatic_turn_limit_defaults_for_older_saved_configurations() -> None:
    config = TimeControlConfig.model_validate(
        {
            "planning_allowance_seconds": 10,
            "resolution_allowance_seconds": 20,
            "response_grant_seconds": 5,
            "initial_time_bank_seconds": 30,
            "time_bank_increment_seconds": 7,
            "max_time_bank_seconds": 40,
            "upgrade_allowance_seconds": 12,
        }
    )
    assert config.automatic_turn_limit == 2
    assert config.initiative_bonus_seconds == 0


def test_initial_turn_has_allowances_and_no_increment(config: TimeControlConfig) -> None:
    clock = _clock(config)
    player = clock.players["hero_a"]
    assert clock.status == ClockStatus.WAITING_FOR_PLAYERS
    assert player.planning_allowance_ms == 10_000
    assert player.resolution_allowance_ms == 20_000
    assert player.time_bank_ms == 30_000

    assert not begin_shared_turn(
        clock,
        config,
        round_number=1,
        turn_number=1,
        now_ms=100,
    )
    assert player.time_bank_ms == 30_000


def test_new_turn_resets_allowances_clears_response_and_caps_bank(
    config: TimeControlConfig,
) -> None:
    clock = _clock(config)
    player = clock.players["hero_a"]
    player.planning_allowance_ms = 1
    player.resolution_allowance_ms = 2
    player.response_time_ms = 3
    player.time_bank_ms = 38_000
    player.planning_complete = True
    player.planning_locked_by_timeout = True
    player.initiative_bonus_ms = 15_000
    clock.initiative_bonus_hero_id = "hero_a"

    assert begin_shared_turn(
        clock,
        config,
        round_number=1,
        turn_number=2,
        now_ms=1_000,
    )
    assert player.planning_allowance_ms == 10_000
    assert player.resolution_allowance_ms == 20_000
    assert player.response_time_ms == 0
    assert player.initiative_bonus_ms == 0
    assert clock.initiative_bonus_hero_id is None
    assert player.time_bank_ms == 40_000
    assert not player.planning_complete
    assert not player.planning_locked_by_timeout


def test_planning_spends_allowance_then_bank_for_all_active_players(
    config: TimeControlConfig,
) -> None:
    clock = _clock(config)
    start_game_clock(clock, 1_000)
    activate_clocks(
        clock,
        ClockKind.PLANNING,
        ["hero_a", "hero_b"],
        request_id=None,
        now_ms=1_000,
    )

    settle_clock(clock, 13_000)
    for player in clock.players.values():
        assert player.planning_allowance_ms == 0
        assert player.time_bank_ms == 28_000
    assert milliseconds_until_next_exhaustion(clock) == 28_000


def test_response_spends_temporary_time_before_resolution_and_bank(
    config: TimeControlConfig,
) -> None:
    clock = _clock(config)
    start_game_clock(clock, 0)
    assert grant_response_time(clock, config, "req-1", ["hero_a"])
    assert not grant_response_time(clock, config, "req-1", ["hero_a"])
    activate_clocks(
        clock,
        ClockKind.RESPONSE,
        ["hero_a"],
        request_id="req-1",
        now_ms=0,
    )

    settle_clock(clock, 7_000)
    player = clock.players["hero_a"]
    assert player.response_time_ms == 0
    assert player.resolution_allowance_ms == 18_000
    assert player.time_bank_ms == 30_000
    assert usable_time_ms(player, ClockKind.RESPONSE) == 48_000


def test_initiative_bonus_is_primary_resolution_only(config: TimeControlConfig) -> None:
    config = config.model_copy(update={"initiative_bonus_seconds": 15})
    clock = _clock(config)
    player = clock.players["hero_a"]
    start_game_clock(clock, 0)

    assert grant_initiative_bonus(clock, config, "hero_a")
    assert player.initiative_bonus_ms == 15_000
    assert usable_time_ms(player, ClockKind.RESOLUTION) == 65_000

    grant_response_time(clock, config, "response-1", ["hero_a"])
    activate_clocks(
        clock,
        ClockKind.RESPONSE,
        ["hero_a"],
        request_id="response-1",
        now_ms=0,
    )
    settle_clock(clock, 20_000)

    # Response time can consume its grant and Resolution allowance, but never
    # the separate initiative bonus.
    assert player.response_time_ms == 0
    assert player.resolution_allowance_ms == 5_000
    assert player.initiative_bonus_ms == 15_000

    activate_clocks(
        clock,
        ClockKind.RESOLUTION,
        ["hero_a"],
        request_id="primary-1",
        now_ms=20_000,
    )
    settle_clock(clock, 35_000)
    assert player.initiative_bonus_ms == 0
    assert player.resolution_allowance_ms == 5_000


def test_team_targets_run_simultaneously_and_exhaust_independently(
    config: TimeControlConfig,
) -> None:
    clock = _clock(config)
    start_game_clock(clock, 0)
    a = clock.players["hero_a"]
    b = clock.players["hero_b"]
    a.resolution_allowance_ms = 1_000
    a.time_bank_ms = 0
    b.resolution_allowance_ms = 3_000
    b.time_bank_ms = 0
    activate_clocks(
        clock,
        ClockKind.RESPONSE,
        ["hero_a", "hero_b"],
        request_id="team-1",
        now_ms=0,
    )

    settle_clock(clock, 1_000)
    assert exhausted_active_hero_ids(clock) == ["hero_a"]
    assert milliseconds_until_next_exhaustion(clock) == 0

    # Removing the already exhausted target lets the coordinator wait for the
    # remaining teammate's personal deadline.
    activate_clocks(
        clock,
        ClockKind.RESPONSE,
        ["hero_b"],
        request_id="team-1",
        now_ms=1_000,
    )
    assert milliseconds_until_next_exhaustion(clock) == 2_000


def test_upgrade_allowance_is_one_pool_for_the_phase(config: TimeControlConfig) -> None:
    clock = _clock(config)
    begin_level_up(clock, config, round_number=1)
    assert clock.players["hero_a"].upgrade_allowance_ms == 12_000


def test_public_view_extrapolates_without_mutating_persisted_state(
    config: TimeControlConfig,
) -> None:
    clock = _clock(config)
    start_game_clock(clock, 10_000)
    activate_clocks(
        clock,
        ClockKind.PLANNING,
        ["hero_a"],
        request_id=None,
        now_ms=10_000,
    )

    view = public_clock_view(clock, 13_000)
    assert view["server_now_ms"] == 13_000
    assert view["players"]["hero_a"]["planning_allowance_ms"] == 7_000
    assert view["players"]["hero_a"]["running"] is True
    assert view["players"]["hero_b"]["running"] is False
    assert clock.players["hero_a"].planning_allowance_ms == 10_000
    assert clock.last_settled_at_ms == 10_000
