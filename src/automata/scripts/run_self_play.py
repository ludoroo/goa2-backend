"""Run one persistent worker from a deterministic four-worker self-play plan.

Process orchestration intentionally remains outside this command until the
``run_parallel`` integration lands. Launch this command once for each
``--worker-id``; every process receives a disjoint, reproducible seed shard.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

from automata.harness.game_runner import DEFAULT_MAP
from automata.models.contracts import LearnedModelRuntime, canonical_json_bytes
from automata.search.config import SearchConfig, parse_learned_lh_search_config
from automata.search.ismcts.strategy import SearchStrategy
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from automata.training.generation import (
    RANDOM_STREAM_NAMESPACE_MAX_LENGTH,
    VISIT_TEMPERATURE_SCHEDULES,
    WORKER_COUNT,
    GameSpec,
    GenerationConfig,
    LoadedChampionRuntime,
    RuntimeLoader,
    SelfPlayWorker,
    StrategyFactory,
    TelemetryEvent,
    build_worker_specs,
)

LEARNED_POLICY_HEURISTIC_VALUE_PRESET = "learned-policy-heuristic-value"
STRATEGY_PRESETS = (LEARNED_POLICY_HEURISTIC_VALUE_PRESET,)


@dataclass(frozen=True, slots=True)
class GameVariant:
    game_type: str
    red_composition: tuple[str, ...]
    blue_composition: tuple[str, ...]


def _resolved_variant_schedule(name: str) -> tuple[GameVariant, ...]:
    scope = PHASE0_EXPERIMENT
    fixed = GameVariant(scope.game_type, scope.red_heroes, scope.blue_heroes)
    if name == "fixed":
        return (fixed,)
    if name == "balanced":
        swapped = (scope.blue_heroes, scope.red_heroes)
        return (
            GameVariant("QUICK", scope.red_heroes, scope.blue_heroes),
            GameVariant("QUICK", *swapped),
            GameVariant("LONG", scope.red_heroes, scope.blue_heroes),
            GameVariant("LONG", *swapped),
        )
    raise ValueError(f"unknown variant schedule {name!r}")


def _serialized_variant_schedule(variants: Sequence[GameVariant]) -> list[dict[str, Any]]:
    return [
        {
            "game_type": variant.game_type,
            "red_composition": list(variant.red_composition),
            "blue_composition": list(variant.blue_composition),
        }
        for variant in variants
    ]


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def _random_stream_namespace(value: str) -> str:
    if not value.strip() or len(value) > RANDOM_STREAM_NAMESPACE_MAX_LENGTH:
        raise argparse.ArgumentTypeError(
            "must be nonempty and at most " f"{RANDOM_STREAM_NAMESPACE_MAX_LENGTH} characters"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--parent-model-digest", required=True)
    parser.add_argument("--parent-generation", required=True, type=int)
    parser.add_argument("--observation-schema-version", required=True, type=_positive)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-end", required=True, type=int)
    parser.add_argument("--worker-id", required=True, type=int, choices=range(WORKER_COUNT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    strategy = parser.add_mutually_exclusive_group()
    strategy.add_argument(
        "--strategy-preset",
        choices=STRATEGY_PRESETS,
        help="tracked built-in runtime strategy preset",
    )
    strategy.add_argument(
        "--strategy-factory", help="dotted module:callable runtime strategy factory"
    )
    parser.add_argument("--search-config", required=True, type=_json_object)
    parser.add_argument("--visit-temperature", type=_non_negative_float, default=0.0)
    parser.add_argument(
        "--visit-temperature-schedule",
        choices=VISIT_TEMPERATURE_SCHEDULES,
        default="constant",
        help="self-play visit-temperature schedule (default: constant)",
    )
    parser.add_argument(
        "--random-stream-namespace",
        type=_random_stream_namespace,
        default=None,
        help="share per-world/side agent RNG streams across comparable experiment arms",
    )
    parser.add_argument(
        "--variant-schedule",
        choices=("fixed", "balanced"),
        default="fixed",
        help="game mode/side schedule (default: fixed PHASE0 QUICK/original sides)",
    )
    parser.add_argument("--source-config", default="{}", type=_json_object)
    parser.add_argument("--max-steps", required=True, type=_positive)
    parser.add_argument("--timeout-seconds", required=True, type=_positive_float)
    parser.add_argument(
        "--decision-timeout-seconds",
        type=_positive_float,
        default=None,
        help="hard per-decision self-play timeout (disabled by default)",
    )
    parser.add_argument("--source-revision", help=argparse.SUPPRESS)
    parser.add_argument("--dirty-tree-hash", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false", help="disable progress output"
    )
    return parser


def _learned_policy_heuristic_value_config(
    raw: Mapping[str, Any],
) -> tuple[SearchConfig, dict[str, Any]]:
    """Backward-compatible wrapper around the shared production parser."""
    return parse_learned_lh_search_config(raw)


def _resolve_strategy_preset(
    name: str, raw_search_config: Mapping[str, Any]
) -> tuple[StrategyFactory, dict[str, Any], dict[str, Any]]:
    if name != LEARNED_POLICY_HEURISTIC_VALUE_PRESET:  # pragma: no cover - argparse guards
        raise ValueError(f"unknown strategy preset {name!r}")
    template, search_identity = _learned_policy_heuristic_value_config(raw_search_config)

    def factory(runtime: object, game: GameSpec, side: str, seed: int) -> SearchStrategy:
        del game, side
        from automata.agents.heuristic_agent import HeuristicAgent
        from automata.evaluation.learned_matrix import (
            LearnedMatrixCell,
            build_learned_ismcts,
        )

        heuristic = HeuristicAgent(seed=seed)
        return build_learned_ismcts(
            LearnedMatrixCell.LH,
            runtime=cast(LearnedModelRuntime, runtime),
            default_policy=heuristic,
            config=replace(template, seed=seed),
        )

    preset_identity = {
        "name": name,
        "matrix_cell": "L/H",
        "search_config": search_identity,
    }
    return cast(StrategyFactory, factory), search_identity, preset_identity


def _resolve_strategy_factory(reference: str) -> StrategyFactory:
    try:
        module_name, attribute = reference.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(f"invalid strategy factory reference {reference!r}") from exc
    if not callable(factory):
        raise ValueError(f"strategy factory {reference!r} is not callable")

    def checked(runtime: object, game: GameSpec, side: str, seed: int) -> SearchStrategy:
        strategy = factory(runtime, game, side, seed)
        if not callable(getattr(strategy, "select", None)) or not isinstance(
            getattr(strategy, "strategy_id", None), str
        ):
            raise TypeError("strategy factory did not return a SearchStrategy")
        return cast(SearchStrategy, strategy)

    return cast(StrategyFactory, checked)


def _load_learned_model_runtime(
    artifact_path: Path,
    config: GenerationConfig,
    variants: Sequence[GameVariant] | None = None,
) -> LoadedChampionRuntime:
    """Explicit optional-ML boundary: this is the only path that imports torch."""
    # Both imports transitively require torch and therefore deliberately remain
    # local to worker execution rather than command/contract import time.
    from automata.models.contracts import ArtifactError, RuntimeRequirements
    from automata.models.shared_encoder.runtime import SharedEncoderRuntime

    manifest_path = artifact_path / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read learned-model artifact manifest: {manifest_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("learned-model artifact manifest must be a JSON object")
    scope = PHASE0_EXPERIMENT
    scheduled = tuple(variants or _resolved_variant_schedule("fixed"))
    scheduled_heroes = frozenset(
        hero
        for variant in scheduled
        for composition in (variant.red_composition, variant.blue_composition)
        for hero in composition
    )
    scheduled_game_types = frozenset(variant.game_type for variant in scheduled)
    try:
        requirements = RuntimeRequirements(
            runtime_compatibility_version=int(raw["runtime_compatibility_version"]),
            observation_schema_version=config.observation_schema_version,
            map_schema_version=int(raw["map_schema_version"]),
            heroes=scheduled_heroes,
            map_id=scope.map_id,
            game_type=scheduled[0].game_type,
            hero_adapter_versions=dict(raw["hero_adapter_versions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("learned-model artifact manifest lacks runtime requirements") from exc
    runtime = SharedEncoderRuntime.from_artifact(artifact_path, requirements=requirements)
    missing_game_types = scheduled_game_types - runtime.supported_game_types
    if missing_game_types:
        raise ArtifactError(
            f"unsupported scheduled game type scope: {sorted(missing_game_types)!r}"
        )
    missing_heroes = scheduled_heroes - runtime.supported_heroes
    if missing_heroes:  # pragma: no cover - also enforced by artifact requirements
        raise ArtifactError(f"unsupported scheduled hero scope: {sorted(missing_heroes)!r}")
    return LoadedChampionRuntime(
        runtime=runtime,
        model_digest=str(raw.get("model_digest", "")),
        generation=config.parent_generation,
        observation_schema_version=runtime.schema.observation_schema_version,
    )


def _telemetry_to_stderr(event: TelemetryEvent) -> None:
    sys.stderr.buffer.write(canonical_json_bytes(event) + b"\n")
    sys.stderr.buffer.flush()


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_loader: RuntimeLoader | None = None,
    strategy_factory: StrategyFactory | None = None,
    telemetry: Callable[[TelemetryEvent], None] = _telemetry_to_stderr,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.seed_end <= args.seed_start:
        parser.error("seed range must be non-empty and half-open")
    if bool(args.source_revision) != bool(args.dirty_tree_hash):
        parser.error("--source-revision and --dirty-tree-hash must be supplied together")
    if strategy_factory is None and not (args.strategy_preset or args.strategy_factory):
        parser.error(
            "--strategy-preset or --strategy-factory is required unless a factory is injected"
        )
    preset_factory: StrategyFactory | None = None
    resolved_search_config = args.search_config
    preset_identity: Mapping[str, Any] | None = None
    if args.strategy_preset:
        try:
            preset_factory, resolved_search_config, preset_identity = _resolve_strategy_preset(
                args.strategy_preset, args.search_config
            )
        except ValueError as exc:
            parser.error(str(exc))

    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.source_revision:
        revision, dirty_hash = args.source_revision, args.dirty_tree_hash
    else:
        from automata.evaluation.provenance import repository_root, source_identity

        revision, dirty_hash = source_identity(
            exclude_paths=(repository_root() / "runs", output_dir, checkpoint_dir)
        )
    scope = PHASE0_EXPERIMENT
    variants = _resolved_variant_schedule(args.variant_schedule)
    schedule_seed_start = scope.seed_registry.range_for("training").start
    source_config: Mapping[str, Any] = {
        **args.source_config,
        **(
            {"random_stream_namespace": args.random_stream_namespace}
            if args.random_stream_namespace is not None
            else {}
        ),
        "artifact_path": str(Path(args.artifact).resolve()),
        "map_path": DEFAULT_MAP,
        "map_id": scope.map_id,
        "game_type": scope.game_type,
        "red_composition": list(scope.red_heroes),
        "blue_composition": list(scope.blue_heroes),
        "variant_schedule_name": args.variant_schedule,
        "variant_schedule_seed_start": schedule_seed_start,
        "variant_schedule": _serialized_variant_schedule(variants),
        "visit_temperature_schedule": args.visit_temperature_schedule,
        "worker_count": WORKER_COUNT,
        **({"strategy_preset": preset_identity} if preset_identity is not None else {}),
    }
    config = GenerationConfig(
        generation_id=args.generation_id,
        parent_model_digest=args.parent_model_digest,
        parent_generation=args.parent_generation,
        observation_schema_version=args.observation_schema_version,
        source_revision=revision,
        dirty_tree_hash=dirty_hash,
        search_config=resolved_search_config,
        source_config=source_config,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout_seconds,
        visit_temperature=args.visit_temperature,
        visit_temperature_schedule=args.visit_temperature_schedule,
        decision_timeout_seconds=args.decision_timeout_seconds,
        random_stream_namespace=args.random_stream_namespace,
    )
    games = tuple(
        GameSpec(
            world_seed=seed,
            map_id=scope.map_id,
            map_path=DEFAULT_MAP,
            game_type=variant.game_type,
            red_composition=variant.red_composition,
            blue_composition=variant.blue_composition,
        )
        for seed in range(args.seed_start, args.seed_end)
        for variant in (variants[(seed - schedule_seed_start) % len(variants)],)
    )
    specs = build_worker_specs(config, games)
    selected = specs[args.worker_id]

    def artifact_loader(config: GenerationConfig) -> LoadedChampionRuntime:
        return _load_learned_model_runtime(Path(args.artifact), config, variants)

    resolved_loader: RuntimeLoader = runtime_loader or artifact_loader
    resolved_strategy = (
        strategy_factory
        or preset_factory
        or _resolve_strategy_factory(cast(str, args.strategy_factory))
    )
    terminal_counts: Counter[str] = Counter()
    with tqdm(
        total=len(selected.games),
        desc=f"Self-play worker {selected.worker_id}",
        unit="game",
        disable=not args.progress,
    ) as progress:

        def report(event: TelemetryEvent) -> None:
            progress.clear()
            telemetry(event)
            if event.event == "game_complete":
                terminal_counts["complete"] += 1
            elif event.event in {"timeout", "decision_timeout"}:
                terminal_counts[event.reason or event.event] += 1
            else:
                progress.refresh()
                return
            progress.set_postfix(dict(terminal_counts))
            progress.update()

        worker = SelfPlayWorker(
            selected,
            output_dir=output_dir / f"worker-{selected.worker_id}",
            checkpoint_path=checkpoint_dir / f"worker-{selected.worker_id}.jsonl",
            runtime_loader=resolved_loader,
            strategy_factory=resolved_strategy,
            telemetry=report,
        )
        worker.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
