"""Run a resumable paired candidate-versus-parent learned L/H arena stratum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from automata.agents.contracts import Agent
from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.evaluation.learned_matrix import LearnedMatrixCell, build_learned_ismcts
from automata.evaluation.protocol import (
    AgentSpec,
    EvaluationGameResult,
    EvaluationProtocol,
    GameCase,
    run_protocol,
    summarize,
)
from automata.harness.game_runner import run_game
from automata.search.config import SearchConfig, parse_learned_lh_search_config

_RANDOM_STREAM_NAMESPACE_MAX_LENGTH = 128
_SEED_DERIVATION = "sha256(learned-arena-side-stream-v1,namespace,world-seed,side)"


def _hero_id(name: str) -> str:
    return f"hero_{name.lower().replace(' ', '_')}"


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def side_search_seed(namespace: str, world_seed: int, side: str) -> int:
    """Derive a side stream from only namespace, world seed, and board side."""
    if not namespace.strip() or len(namespace) > _RANDOM_STREAM_NAMESPACE_MAX_LENGTH:
        raise ValueError("random stream namespace must be nonempty and at most 128 characters")
    if side not in ("RED", "BLUE"):
        raise ValueError(f"invalid arena side {side!r}")
    return int(
        _canonical_digest(
            {
                "derivation": _SEED_DERIVATION,
                "namespace": namespace,
                "world_seed": world_seed,
                "side": side,
            }
        )[:16],
        16,
    )


def _search_identity(config: SearchConfig) -> dict[str, Any]:
    identity = asdict(config)
    identity["cutoff_unit"] = config.cutoff_unit.value
    identity["leaf_mode"] = config.leaf_mode.value
    identity["seed"] = "side_search_seed"
    return identity


def _runtime_requirements(*, map_id: str, game_type: str, heroes: frozenset[str]) -> Any:
    """Build current runtime requirements without trusting artifact-declared versions."""
    from automata.models.contracts import (
        CURRENT_MAP_SCHEMA_VERSION,
        CURRENT_RUNTIME_COMPATIBILITY_VERSION,
        RuntimeRequirements,
    )
    from automata.models.shared_encoder.schema import TensorFeatureSchema
    from automata.observation.hero_adapters import HeroObservationAdapterRegistry

    adapters = HeroObservationAdapterRegistry()
    registered = adapters.registered_versions
    return RuntimeRequirements(
        runtime_compatibility_version=CURRENT_RUNTIME_COMPATIBILITY_VERSION,
        observation_schema_version=TensorFeatureSchema.current().observation_schema_version,
        map_schema_version=CURRENT_MAP_SCHEMA_VERSION,
        heroes=heroes,
        map_id=map_id,
        game_type=game_type,
        hero_adapter_versions={
            "generic": adapters.generic_version,
            **{name: registered.get(name, adapters.generic_version) for name in heroes},
        },
    )


def _load_pinned_runtime(
    artifact: Path,
    expected_digest: str,
    *,
    map_id: str,
    game_type: str,
    heroes: frozenset[str],
) -> Any:
    """Load one fully verified artifact and require its exact pinned digest/scope."""
    try:
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        declared_digest = manifest["model_digest"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid learned artifact manifest: {artifact}") from exc
    if declared_digest != expected_digest:
        raise ValueError(f"artifact model digest does not match pinned digest: {artifact}")

    from automata.models.shared_encoder.serving import load_runtime

    requirements = _runtime_requirements(map_id=map_id, game_type=game_type, heroes=heroes)
    runtime = load_runtime(
        artifact,
        requirements=requirements,
        expected_digest=expected_digest,
    )
    if map_id not in runtime.supported_maps:
        raise ValueError(f"artifact does not support arena map {map_id!r}: {artifact}")
    if game_type not in runtime.supported_game_types:
        raise ValueError(f"artifact does not support arena game type {game_type!r}: {artifact}")
    missing_heroes = heroes - runtime.supported_heroes
    if missing_heroes:
        raise ValueError(
            f"artifact does not support arena heroes {sorted(missing_heroes)!r}: {artifact}"
        )
    return runtime


@dataclass(slots=True)
class LearnedArenaRunner:
    """Spawn-picklable top-level callable for one complete arena game."""

    candidate_artifact: Path
    candidate_digest: str
    parent_artifact: Path
    parent_digest: str
    map_path: Path
    map_id: str
    game_type: str
    red_heroes: tuple[str, ...]
    blue_heroes: tuple[str, ...]
    max_steps: int
    search_config: SearchConfig
    random_stream_namespace: str
    _progress_callback: Callable[[int, int], None] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        heroes = self.red_heroes + self.blue_heroes
        if not self.red_heroes or not self.blue_heroes:
            raise ValueError("red and blue hero rosters must be nonempty")
        if len(set(heroes)) != len(heroes):
            raise ValueError("arena hero names must be unique across both sides")
        if self.game_type not in ("QUICK", "LONG"):
            raise ValueError("game_type must be QUICK or LONG")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        for label, digest in (
            ("candidate", self.candidate_digest),
            ("parent", self.parent_digest),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{label} digest must be a lowercase SHA-256 digest")
        side_search_seed(self.random_stream_namespace, 0, "RED")

    @property
    def heroes(self) -> frozenset[str]:
        return frozenset((*self.red_heroes, *self.blue_heroes))

    def set_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """Receive the protocol child's progress-pipe callback."""
        self._progress_callback = callback

    def validate_artifacts(self) -> None:
        """Fail before scheduling unless both exact artifacts support this arena."""
        _load_pinned_runtime(
            self.candidate_artifact,
            self.candidate_digest,
            map_id=self.map_id,
            game_type=self.game_type,
            heroes=self.heroes,
        )
        _load_pinned_runtime(
            self.parent_artifact,
            self.parent_digest,
            map_id=self.map_id,
            game_type=self.game_type,
            heroes=self.heroes,
        )

    def _agent(self, runtime: Any, *, seed: int) -> ISMCTSAgent:
        config = replace(self.search_config, seed=seed)
        heuristic = HeuristicAgent(seed=seed)
        strategy = build_learned_ismcts(
            LearnedMatrixCell.LH,
            runtime=runtime,
            default_policy=heuristic,
            config=config,
        )
        return ISMCTSAgent(config, strategy=strategy)

    def __call__(self, case: GameCase) -> EvaluationGameResult:
        candidate_runtime = _load_pinned_runtime(
            self.candidate_artifact,
            self.candidate_digest,
            map_id=self.map_id,
            game_type=self.game_type,
            heroes=self.heroes,
        )
        parent_runtime = _load_pinned_runtime(
            self.parent_artifact,
            self.parent_digest,
            map_id=self.map_id,
            game_type=self.game_type,
            heroes=self.heroes,
        )

        side_runtimes = (
            {"RED": candidate_runtime, "BLUE": parent_runtime}
            if case.a_side == "RED"
            else {"RED": parent_runtime, "BLUE": candidate_runtime}
        )
        agents: dict[str, Agent] = {}
        for side, roster in (("RED", self.red_heroes), ("BLUE", self.blue_heroes)):
            seed = side_search_seed(self.random_stream_namespace, case.world_seed, side)
            agent = self._agent(side_runtimes[side], seed=seed)
            agents.update({_hero_id(name): agent for name in roster})

        outcome = run_game(
            list(self.red_heroes),
            list(self.blue_heroes),
            agents,
            map_path=str(self.map_path),
            game_type=self.game_type,
            seed=case.world_seed,
            max_steps=self.max_steps,
            progress_callback=self._progress_callback,
        )
        winner = None if outcome.winner is None else outcome.winner.upper()
        if winner not in (None, "RED", "BLUE"):
            raise ValueError(f"arena game returned invalid winner {outcome.winner!r}")
        return EvaluationGameResult(
            case_id=case.case_id,
            world_seed=case.world_seed,
            a_side=case.a_side,
            winner_side=winner,
            rounds=outcome.rounds,
            steps=outcome.steps,
            reason=outcome.reason,
        )


def build_protocol(
    runner: LearnedArenaRunner,
    *,
    world_seeds: tuple[int, ...],
    source_revision: str,
    dirty_tree_hash: str,
    case_timeout_seconds: float,
) -> EvaluationProtocol:
    """Bind a runner to the canonical candidate-as-A paired protocol."""
    shared = {
        "matrix_cell": LearnedMatrixCell.LH.value,
        "search_config": _search_identity(runner.search_config),
        "random_stream_namespace": runner.random_stream_namespace,
        "seed_derivation": _SEED_DERIVATION,
        "map_id": runner.map_id,
    }
    return EvaluationProtocol(
        agent_a=AgentSpec(
            name="candidate",
            kind="learned_ismcts_lh",
            params={**shared, "artifact_digest": runner.candidate_digest},
        ),
        agent_b=AgentSpec(
            name="parent",
            kind="learned_ismcts_lh",
            params={**shared, "artifact_digest": runner.parent_digest},
        ),
        red_heroes=runner.red_heroes,
        blue_heroes=runner.blue_heroes,
        world_seeds=world_seeds,
        map_path=str(runner.map_path),
        game_type=runner.game_type,
        max_steps=runner.max_steps,
        source_revision=source_revision,
        dirty_tree_hash=dirty_tree_hash,
        case_timeout_seconds=case_timeout_seconds,
    )


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


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


def _digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def _namespace(value: str) -> str:
    try:
        side_search_seed(value, 0, "RED")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--candidate-digest", required=True, type=_digest)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--parent-digest", required=True, type=_digest)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--game-type", required=True, choices=("QUICK", "LONG"))
    parser.add_argument("--red-heroes", required=True, nargs="+")
    parser.add_argument("--blue-heroes", required=True, nargs="+")
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-end", required=True, type=int)
    parser.add_argument("--max-steps", required=True, type=_positive)
    parser.add_argument("--timeout-seconds", required=True, type=_positive_float)
    parser.add_argument("--search-config", required=True, type=_json_object)
    parser.add_argument("--random-stream-namespace", required=True, type=_namespace)
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false", help="disable progress output"
    )
    return parser


def _summary_payload(
    observations: Sequence[EvaluationGameResult], protocol: EvaluationProtocol
) -> dict[str, Any]:
    aggregate = summarize(observations)
    lower, upper = aggregate.wilson_ci()
    return {
        "schema_version": 1,
        "protocol_identity": protocol.identity_digest(),
        "games": len(observations),
        "seed_pairs": len(observations) // 2,
        "candidate_wins": aggregate.a_wins,
        "parent_wins": aggregate.b_wins,
        "draws": aggregate.draws,
        "reasons": dict(sorted(Counter(row.reason for row in observations).items())),
        "rounds": {
            "total": sum(row.rounds for row in observations),
            "average_completed": aggregate.avg_rounds,
        },
        "steps": {
            "total": sum(row.steps for row in observations),
            "average_completed": aggregate.avg_steps,
        },
        "wilson_95": {
            "decisive_games": aggregate.decisive,
            "candidate_win_rate": aggregate.decisive_a_rate,
            "lower": lower,
            "upper": upper,
        },
        "max_step_terminations": aggregate.max_step_terminations,
        "timeout_terminations": aggregate.timeout_terminations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.seed_end <= args.seed_start:
        parser.error("seed range must be non-empty and half-open")
    try:
        search_config, _ = parse_learned_lh_search_config(args.search_config)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        candidate_artifact = Path(args.candidate_artifact).resolve(strict=True)
        parent_artifact = Path(args.parent_artifact).resolve(strict=True)
        map_path = Path(args.map_path).resolve(strict=True)
        if not candidate_artifact.is_dir() or not parent_artifact.is_dir():
            raise ValueError("candidate and parent artifacts must be directories")
        if not map_path.is_file():
            raise ValueError("map path must be a file")
        runner = LearnedArenaRunner(
            candidate_artifact=candidate_artifact,
            candidate_digest=args.candidate_digest,
            parent_artifact=parent_artifact,
            parent_digest=args.parent_digest,
            map_path=map_path,
            map_id=map_path.stem,
            game_type=args.game_type,
            red_heroes=tuple(args.red_heroes),
            blue_heroes=tuple(args.blue_heroes),
            max_steps=args.max_steps,
            search_config=search_config,
            random_stream_namespace=args.random_stream_namespace,
        )
        runner.validate_artifacts()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    from automata.evaluation.provenance import repository_root, source_tree_identity

    source_revision, dirty_tree_hash = source_tree_identity(
        exclude_paths=(repository_root() / "runs",)
    )
    protocol = build_protocol(
        runner,
        world_seeds=tuple(range(args.seed_start, args.seed_end)),
        source_revision=source_revision,
        dirty_tree_hash=dirty_tree_hash,
        case_timeout_seconds=args.timeout_seconds,
    )
    observations = run_protocol(
        protocol,
        checkpoint_path=Path(args.checkpoint),
        run_case=runner,
        show_progress=args.progress,
        progress_description="Learned arena cases",
    )
    payload = _summary_payload(observations, protocol)
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
