"""Version constants shared by artifacts, encoders, and runtimes."""

CURRENT_RUNTIME_COMPATIBILITY_VERSION = 1
CURRENT_MAP_SCHEMA_VERSION = 1
INITIAL_RELATIONSHIP_NAMES: tuple[str, ...] = (
    "delta_q",
    "delta_r",
    "delta_s",
    "hex_distance",
    "path_distance",
    "path_exists",
    "is_adjacent",
    "topology_is_adjacent",
    "is_straight_line",
    "has_line_of_sight",
    "same_zone",
    "same_lane",
    "lane_progress_delta",
    "relation",
    "reachable",
    "threatens",
    "supports",
    "path_distance_valid",
    "has_line_of_sight_valid",
    "reachable_valid",
    "threatens_valid",
    "supports_valid",
)

__all__ = [
    "CURRENT_MAP_SCHEMA_VERSION",
    "CURRENT_RUNTIME_COMPATIBILITY_VERSION",
    "INITIAL_RELATIONSHIP_NAMES",
]
