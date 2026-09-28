"""The initial learned-model experiment declaration."""

from automata.training.contracts.experiment import ExperimentScope, SeedRange, SeedRegistry

PHASE0_EXPERIMENT = ExperimentScope(
    map_id="forgotten_island",
    game_type="QUICK",
    red_heroes=("Wasp", "Xargatha"),
    blue_heroes=("Arien", "Brogan"),
    seed_registry=SeedRegistry(
        {
            "bootstrap": SeedRange(10_000, 20_000),
            "training": SeedRange(20_000, 1_000_000),
            "validation": SeedRange(1_000_000, 1_010_000),
            "screen": SeedRange(1_010_000, 1_020_000),
            "promotion": SeedRange(1_020_000, 1_100_000),
        }
    ),
)

__all__ = ["PHASE0_EXPERIMENT"]
