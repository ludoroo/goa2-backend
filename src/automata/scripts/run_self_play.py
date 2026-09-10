"""Run one persistent worker from a deterministic four-worker self-play plan.

Process orchestration intentionally remains outside this command until the
``run_parallel`` integration lands. Launch this command once for each
``--worker-id``; every process receives a disjoint, reproducible seed shard.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

from automata.harness.game_runner import DEFAULT_MAP
from automata.models.contracts import canonical_json_bytes
from automata.search.ismcts.strategy import SearchStrategy
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from automata.training.generation import (
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


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


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
    parser.add_argument(
        "--strategy-factory", help="dotted module:callable runtime strategy factory"
    )
    parser.add_argument("--search-config", required=True, type=_json_object)
    parser.add_argument("--source-config", default="{}", type=_json_object)
    parser.add_argument("--max-steps", required=True, type=_positive)
    parser.add_argument("--timeout-seconds", required=True, type=_positive_float)
    parser.add_argument("--source-revision", help=argparse.SUPPRESS)
    parser.add_argument("--dirty-tree-hash", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false", help="disable progress output"
    )
    return parser


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
    artifact_path: Path, config: GenerationConfig
) -> LoadedChampionRuntime:
    """Explicit optional-ML boundary: this is the only path that imports torch."""
    # Both imports transitively require torch and therefore deliberately remain
    # local to worker execution rather than command/contract import time.
    from automata.models.contracts import RuntimeRequirements
    from automata.models.shared_encoder.runtime import SharedEncoderRuntime

    manifest_path = artifact_path / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read learned-model artifact manifest: {manifest_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("learned-model artifact manifest must be a JSON object")
    scope = PHASE0_EXPERIMENT
    try:
        requirements = RuntimeRequirements(
            runtime_compatibility_version=int(raw["runtime_compatibility_version"]),
            observation_schema_version=config.observation_schema_version,
            map_schema_version=int(raw["map_schema_version"]),
            heroes=frozenset((*scope.red_heroes, *scope.blue_heroes)),
            map_id=scope.map_id,
            game_type=scope.game_type,
            hero_adapter_versions=dict(raw["hero_adapter_versions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("learned-model artifact manifest lacks runtime requirements") from exc
    runtime = SharedEncoderRuntime.from_artifact(artifact_path, requirements=requirements)
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
    if strategy_factory is None and not args.strategy_factory:
        parser.error("--strategy-factory is required unless one is injected")

    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.source_revision:
        revision, dirty_hash = args.source_revision, args.dirty_tree_hash
    else:
        from automata.evaluation.provenance import source_identity

        revision, dirty_hash = source_identity(exclude_paths=(output_dir, checkpoint_dir))
    scope = PHASE0_EXPERIMENT
    source_config: Mapping[str, Any] = {
        **args.source_config,
        "artifact_path": str(Path(args.artifact).resolve()),
        "map_path": DEFAULT_MAP,
        "map_id": scope.map_id,
        "game_type": scope.game_type,
        "red_composition": list(scope.red_heroes),
        "blue_composition": list(scope.blue_heroes),
        "worker_count": WORKER_COUNT,
    }
    config = GenerationConfig(
        generation_id=args.generation_id,
        parent_model_digest=args.parent_model_digest,
        parent_generation=args.parent_generation,
        observation_schema_version=args.observation_schema_version,
        source_revision=revision,
        dirty_tree_hash=dirty_hash,
        search_config=args.search_config,
        source_config=source_config,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout_seconds,
    )
    games = tuple(
        GameSpec(
            world_seed=seed,
            map_id=scope.map_id,
            map_path=DEFAULT_MAP,
            game_type=scope.game_type,
            red_composition=scope.red_heroes,
            blue_composition=scope.blue_heroes,
        )
        for seed in range(args.seed_start, args.seed_end)
    )
    specs = build_worker_specs(config, games)
    selected = specs[args.worker_id]

    def artifact_loader(config: GenerationConfig) -> LoadedChampionRuntime:
        return _load_learned_model_runtime(Path(args.artifact), config)

    resolved_loader: RuntimeLoader = runtime_loader or artifact_loader
    resolved_strategy = strategy_factory or _resolve_strategy_factory(args.strategy_factory)
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
            elif event.event == "timeout":
                terminal_counts[event.reason or "timeout"] += 1
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
