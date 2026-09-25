"""Classic server bot construction and runtime caching."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from automata.agents.contracts import Agent
from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.agents.random_agent import RandomAgent
from automata.search.config import (
    LEARNED_ROOT_PUCT_C,
    LEARNED_ROOT_WIDENING_ALPHA,
    LEARNED_ROOT_WIDENING_C,
    SearchConfig,
)
from automata.search.continuation import AgentContinuationPolicy, ArgmaxContinuationPolicy
from automata.search.contracts import LeafEvaluator, LeafMode, SearchPolicy
from automata.search.fallback import FallbackLeafEvaluator, FallbackSearchPolicy
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from goa2.domain.state import GameState
from goa2.server.bot_models import BotSpec, SearchSettings

logger = logging.getLogger(__name__)


class RuntimeCacheLike(Protocol):
    def get(self, artifact: str | Path, **kwargs: Any) -> Any: ...


class AgentGame(Protocol):
    game_id: str
    bot_specs: dict[str, BotSpec]
    _bot_agents: dict[str, Agent] | None
    session: Any


AgentFactory = Callable[..., Agent]


def _resolve_artifact_reference(root: Path, reference: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / reference).resolve(strict=True)
    if not candidate.is_relative_to(resolved_root) or not candidate.is_dir():
        from automata.models.contracts import ArtifactError

        raise ArtifactError("artifact reference must resolve to a directory under artifact root")
    return candidate


def _runtime_requirements(state: GameState) -> Any:
    from automata.models.contracts import (
        CURRENT_MAP_SCHEMA_VERSION,
        CURRENT_RUNTIME_COMPATIBILITY_VERSION,
        RuntimeRequirements,
    )
    from automata.models.shared_encoder.schema import TensorFeatureSchema
    from automata.observation.hero_adapters import HeroObservationAdapterRegistry

    heroes = frozenset(hero.name for team in state.teams.values() for hero in team.heroes)
    adapters = HeroObservationAdapterRegistry()
    versions = adapters.registered_versions
    schema = TensorFeatureSchema.current()
    return RuntimeRequirements(
        runtime_compatibility_version=CURRENT_RUNTIME_COMPATIBILITY_VERSION,
        observation_schema_version=schema.observation_schema_version,
        map_schema_version=CURRENT_MAP_SCHEMA_VERSION,
        heroes=heroes,
        map_id=state.board.map_id,
        game_type=state.game_type.value,
        hero_adapter_versions={
            "generic": adapters.generic_version,
            **{name: versions.get(name, adapters.generic_version) for name in heroes},
        },
    )


def agent_for_spec(
    spec: BotSpec,
    seed: int = 0,
    *,
    state: GameState | None = None,
    artifact_root: str | Path | None = None,
    runtime_cache: RuntimeCacheLike | None = None,
) -> Agent:
    """Build one of the classic random, heuristic, or ISMCTS agents."""
    if spec.kind == "random":
        return RandomAgent(seed=seed)
    if spec.kind == "heuristic":
        return HeuristicAgent(seed=seed)
    if spec.kind == "ismcts":
        settings = spec.search or SearchSettings()
        policy = HeuristicAgent(seed=seed)
        prior: SearchPolicy = HeuristicPrior(policy)
        leaf: LeafEvaluator = HeuristicLeafEvaluator()
        if settings.artifact is not None:
            if state is None or artifact_root is None:
                raise ValueError("learned ISMCTS requires game state and artifact root")
            try:
                artifact = _resolve_artifact_reference(
                    Path(artifact_root), settings.artifact.reference
                )
                requirements = _runtime_requirements(state)
                if runtime_cache is None:
                    # Torch is an optional serving dependency. Keep this import
                    # behind the learned configuration and inside the fallback
                    # boundary so classic bots still start when it is absent.
                    from automata.models.shared_encoder.serving import load_runtime

                    runtime = load_runtime(
                        artifact,
                        requirements=requirements,
                        expected_digest=settings.artifact.digest,
                    )
                else:
                    runtime = runtime_cache.get(
                        artifact,
                        requirements=requirements,
                        expected_digest=settings.artifact.digest,
                    )
                from automata.search.learned import LearnedLeafEvaluator, LearnedSearchPolicy
            except (ImportError, OSError, ValueError) as exc:
                if isinstance(exc, ValueError):
                    # ArtifactError intentionally stays behind the learned
                    # configuration boundary so classic H/H startup remains
                    # model-package neutral. Other ValueErrors are programmer
                    # misuse and must not silently become heuristic fallback.
                    from automata.models.contracts import ArtifactError

                    if not isinstance(exc, ArtifactError):
                        raise
                # Invalid/missing artifacts and an unavailable optional Torch
                # runtime are deployment availability failures, not reasons to
                # make the game unavailable. Do not construct a learned adapter
                # around a known-bad runtime: it would encode every observation
                # before discovering the same failure. Leaving each configured
                # component on its independently initialized heuristic avoids
                # that cost and preserves H/L and L/H behavior.
                logger.warning(
                    "Learned bot artifact %r unavailable (%r); falling back to heuristic",
                    settings.artifact.reference,
                    exc,
                )
            else:
                if settings.policy_source == "learned":
                    prior = FallbackSearchPolicy(LearnedSearchPolicy(runtime), prior)
                if settings.value_source == "learned":
                    leaf = FallbackLeafEvaluator(LearnedLeafEvaluator(runtime), leaf)
        learned_root = settings.policy_source == "learned"
        config = SearchConfig(
            iterations=settings.iterations,
            decision_timeout_seconds=settings.decision_timeout_seconds,
            seed=seed,
            cutoff_limit=settings.horizon,
            leaf_mode=(
                LeafMode.IMMEDIATE
                if settings.leaf_mode == "immediate"
                else LeafMode.BOUNDED_CONTINUATION
            ),
            root_puct_c=LEARNED_ROOT_PUCT_C if learned_root else None,
            root_widening_c=LEARNED_ROOT_WIDENING_C if learned_root else None,
            root_widening_alpha=LEARNED_ROOT_WIDENING_ALPHA if learned_root else None,
        )
        continuation = (
            ArgmaxContinuationPolicy(prior)
            if settings.policy_source == "learned"
            else AgentContinuationPolicy(policy)
        )
        return ISMCTSAgent(
            config,
            environment_policy=policy,
            prior=prior,
            leaf_evaluator=leaf,
            continuation_policy=continuation,
        )
    raise ValueError(f"unsupported bot kind: {spec.kind!r}")


def game_entropy(game_id: str) -> int:
    """Derive stable, non-negative int64 entropy from a game id."""
    digest = hashlib.sha1(game_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def get_or_build_agents(
    game: AgentGame, *, factory: AgentFactory = agent_for_spec
) -> dict[str, Agent]:
    """Build a deterministic per-hero agent map once per managed game."""
    if game._bot_agents is not None:
        return game._bot_agents
    base = game_entropy(game.game_id)
    agents: dict[str, Agent] = {}
    for index, (hero_id, spec) in enumerate(sorted(game.bot_specs.items())):
        seed = base ^ index
        learned = spec.search is not None and spec.search.artifact is not None
        if learned:
            agents[hero_id] = factory(
                spec,
                seed=seed,
                state=game.session.state,
                artifact_root=Path(os.getenv("GOA2_MODEL_ARTIFACT_ROOT", "model_artifacts")),
            )
        else:
            agents[hero_id] = factory(spec, seed=seed)
    game._bot_agents = agents
    return agents


__all__ = ["agent_for_spec", "game_entropy", "get_or_build_agents"]
