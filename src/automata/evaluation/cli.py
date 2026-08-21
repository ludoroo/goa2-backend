"""CLI: run the agent evaluation matrix and record baselines.

The eval matrix is the yardstick the whole AI effort is judged on: every
stronger agent must beat the previous baseline over a statistically meaningful
sample (Wilson CI). This runs a set of A-vs-B matchups and prints — and
optionally writes — the results.

Two modes:

- **Legacy matrix mode** (no ``--agent-a``/``--agent-b``): runs the fixed
  ladder of matchups defined in :data:`_MATCHUPS`. Prints per-matchup
  summaries; optionally writes a JSON baseline via ``--out``.
- **Targeted mode** (both ``--agent-a`` and ``--agent-b`` supplied): builds
  a deterministic resumable :class:`EvaluationProtocol` for A vs B, runs
  the paired schedule under a JSONL checkpoint, and prints screen /
  promotion gate verdicts.

Usage:
    # Quick baseline (small sample), print only:
    PYTHONPATH=src uv run python -m automata.evaluation.cli --games 20

    # Full matrix, write results JSON:
    PYTHONPATH=src uv run python -m automata.evaluation.cli \\
        --games 100 --out src/automata/evaluation/baselines.json

    # Targeted screen (ISMCTS vs heuristic, canonical 12-case screen):
    PYTHONPATH=src uv run python -m automata.evaluation.cli \\
        --agent-a ismcts --agent-b heuristic

    # Override the targeted per-case wall-clock limit (default: 1800 seconds):
    PYTHONPATH=src uv run python -m automata.evaluation.cli \
        --agent-a ismcts --agent-b ismcts --case-timeout-seconds 900
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agents.base import Agent
from ..agents.heuristic_agent import HeuristicAgent
from ..agents.random_agent import RandomAgent

# Re-imported into this module so tests can monkeypatch the seam names
# directly on ``automata.evaluation.cli`` without reaching into other
# packages (see :func:`build_case_runner` and :func:`main`).
from ..runtime.harness import run_game  # noqa: F401
from ..search import ISMCTSAgent, LearnedPolicy, SearchConfig
from .cutoff_telemetry import CutoffTelemetryRecorder
from .learned_value import LearnedValue
from .matchup import MatchupResult, evaluate, hero_id
from .protocol import (
    AgentSpec,
    EvaluationProtocol,
    EvaluationSummary,
    GameCase,
    GameObservation,
    load_observations,
    run_protocol,  # noqa: F401 — accessed via ``cli.run_protocol`` (monkeypatched)
    summarize,  # noqa: F401 — accessed via ``cli.summarize`` (monkeypatched)
)
from .value import HeuristicValue, ValueFn

# Quick-game recommended roster (2v2, single lane).
RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]

AgentFactory = Callable[[int], Agent]

# Named agent kinds accepted by ``--agent-a`` / ``--agent-b`` in targeted mode.
_AGENT_KINDS: tuple[str, ...] = ("random", "heuristic", "ismcts")

# Canonical targeted-mode screen defaults. 6 paired seeds → 12 cases; ISMCTS
# defaults deliberately cheap (iterations=4 / cutoff_rounds=1) so a screen
# completes in reasonable wall-clock time. Callers bump these for a real run.
_DEFAULT_PAIRED_SEEDS: int = 6
_DEFAULT_ISMCTS_ITERATIONS: int = 4
_DEFAULT_ISMCTS_CUTOFF_ROUNDS: int = 1

# Default checkpoint layout: ``data/evaluations/<identity_digest>.jsonl``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT_DIR = _REPO_ROOT / "data" / "evaluations"

# Clean-tree sentinel returned by :func:`source_identity`. Kept short and
# obviously non-hex so it never collides with a real dirty-hash value.
_CLEAN_TREE_SENTINEL = "clean"

# Named agent factories. Search agents use a small iteration budget so the
# matrix runs in reasonable time; strength tuning is a separate concern.
# NOTE: an ISMCTS *game* is expensive (~28s at 2 iters, ~160s at 16), since each
# decision runs many determinized playouts. So search matchups run at a small
# game count + low iteration budget by default; the fast (random/heuristic)
# matchups run a larger sample. Bump --games / SEARCH_ITERS for a real eval run.


def _factories(search_iters: int) -> dict[str, AgentFactory]:
    return {
        "random": lambda s: RandomAgent(s),
        "heuristic": lambda s: HeuristicAgent(s),
        "ismcts": lambda s: ISMCTSAgent(SearchConfig(iterations=search_iters, seed=s)),
        "ismcts_noprior": lambda s: ISMCTSAgent(
            SearchConfig(iterations=search_iters, seed=s, use_prior=False)
        ),
    }


# The matchups that define the ladder. Each later rung must beat the agent it
# claims to improve on here. ``search`` flags the slow (ISMCTS) matchups so the
# CLI can run them at a reduced sample.
_MATCHUPS: tuple[tuple[str, str, bool], ...] = (
    ("random", "random", False),  # sanity: ~50%
    ("heuristic", "random", False),  # heuristic must dominate random
    ("ismcts", "heuristic", True),  # search must beat its own default policy
    ("ismcts", "ismcts_noprior", True),  # does the prior help at equal budget?
)


def run_matrix(
    games: int,
    base_seed: int,
    *,
    search_games: int | None = None,
    search_iters: int = 8,
) -> list[MatchupResult]:
    """Run every matchup. Fast matchups use ``games``; slow (ISMCTS) matchups
    use ``search_games`` (default: min(games, 6)) at ``search_iters`` budget."""
    facts = _factories(search_iters)
    sg = search_games if search_games is not None else min(games, 6)
    results: list[MatchupResult] = []
    for a_name, b_name, is_search in _MATCHUPS:
        n = sg if is_search else games
        res = evaluate(
            facts[a_name],
            facts[b_name],
            red_heroes=RED,
            blue_heroes=BLUE,
            games=n,
            base_seed=base_seed,
            label_a=a_name,
            label_b=b_name,
        )
        results.append(res)
    return results


def _result_dict(r: MatchupResult) -> dict[str, object]:
    lo, hi = r.wilson_ci()
    return {
        "a": r.label_a,
        "b": r.label_b,
        "games": r.games,
        "a_wins": r.a_wins,
        "b_wins": r.b_wins,
        "draws": r.draws,
        "a_winrate": round(r.a_winrate, 4),
        "wilson_ci": [round(lo, 4), round(hi, 4)],
        "avg_rounds": round(r.avg_rounds, 2),
    }


# --------------------------------------------------------------------------- #
# Source identity                                                             #
# --------------------------------------------------------------------------- #


def _git(repo_root: Path, *args: str) -> str:
    """Run a git subcommand at ``repo_root`` and return stripped stdout.

    Uses explicit ``cwd`` (never mutates the process CWD) and captures both
    streams so a non-zero exit fails clearly with git's own stderr.
    """
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)!r} failed at {repo_root!s}: "
            f"exit={proc.returncode} stderr={proc.stderr.strip()!r}"
        )
    return proc.stdout


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    """Like :func:`_git` but returns raw bytes (for binary-safe diff output)."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)!r} failed at {repo_root!s}: "
            f"exit={proc.returncode} stderr={proc.stderr.decode(errors='replace').strip()!r}"
        )
    return proc.stdout


def source_identity(repo_root: Path | None = None) -> tuple[str, str]:
    """Return ``(head_revision, dirty_tree_hash)`` for the working tree.

    ``head_revision`` is the HEAD commit SHA. ``dirty_tree_hash`` is a
    deterministic content hash that covers **both** tracked-file diffs and
    untracked (non-ignored) file paths + contents; a clean tree returns the
    stable sentinel :data:`_CLEAN_TREE_SENTINEL`.

    Semi-public seam: tests monkeypatch this to keep experiment identity
    reproducible independent of the checkout state. ``repo_root`` overrides
    the default repository root so tests can drive a fixture
    repo without changing the process CWD.

    Raises :class:`RuntimeError` when git itself fails (e.g. ``repo_root`` is
    not a git repository) so callers see a clear error rather than a silent
    empty digest.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT

    head = _git(root, "rev-parse", "HEAD").strip()
    if not head:
        raise RuntimeError(f"git rev-parse HEAD returned empty at {root!s}")

    # Tracked changes: full patch vs HEAD. ``--no-color`` and a stable format
    # produce a byte-for-byte deterministic payload independent of user config.
    tracked_diff = _git_bytes(
        root,
        "-c",
        "core.quotepath=false",
        "-c",
        "diff.noprefix=false",
        "diff",
        "--no-color",
        "--no-ext-diff",
        "HEAD",
    )

    # Untracked, non-ignored files: names *and* content. A file appearing at
    # all must change the hash; changing its content must change the hash again.
    untracked_names = _git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked = [p for p in untracked_names.split("\x00") if p]
    untracked.sort()  # deterministic order regardless of git's enumeration

    if not tracked_diff and not untracked:
        return head, _CLEAN_TREE_SENTINEL

    hasher = hashlib.sha256()
    hasher.update(b"tracked-diff\x00")
    hasher.update(tracked_diff)
    for rel in untracked:
        hasher.update(b"untracked\x00")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\x00")
        file_path = root / rel
        try:
            hasher.update(file_path.read_bytes())
        except OSError:
            # Broken symlinks / permission errors: hash the failure marker
            # so the file's presence still perturbs the digest.
            hasher.update(b"<unreadable>")
        hasher.update(b"\x00")
    return head, hasher.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Agent building & case runner                                                #
# --------------------------------------------------------------------------- #


def build_agent(
    spec: AgentSpec, seed: int, *, case_metadata: dict[str, Any] | None = None
) -> Agent:
    """Construct a runtime :class:`Agent` from an :class:`AgentSpec`.

    Semi-public seam: the CLI's per-case runner calls this so tests can
    monkeypatch agent construction without having to build real ISMCTS
    agents or run games. ``seed`` is the deterministic per-agent seed the
    case runner derives from the world seed.

    Supports ``kind`` ∈ {``"random"``, ``"heuristic"``, ``"ismcts"``}. For
    ISMCTS, the :class:`SearchConfig` is reconstructed from ``spec.params``
    with the per-case ``seed`` merged in (dynamic; never in identity). Learned
    value and policy artifacts are reloaded and digest-checked in the case
    process before being injected into the agent.
    """
    kind = spec.kind
    if kind == "random":
        return RandomAgent(seed)
    if kind == "heuristic":
        return HeuristicAgent(seed)
    if kind == "ismcts":
        cfg_kwargs: dict[str, Any] = dict(spec.params)
        telemetry_path = cfg_kwargs.pop("cutoff_telemetry_path", None)
        # AgentSpec crosses the spawn boundary, so reload and revalidate
        # captured model metadata.
        model_path = cfg_kwargs.pop("value_model_path", None)
        expected_digest = cfg_kwargs.pop("value_model_digest", None)
        if (model_path is None) != (expected_digest is None):
            raise ValueError("learned value model metadata is incomplete")

        value_fn: ValueFn = HeuristicValue()
        if model_path is not None:
            learned = LearnedValue(str(model_path))
            if learned.digest != expected_digest:
                raise ValueError(
                    "learned value artifact changed after protocol construction: "
                    f"expected digest {expected_digest}, got {learned.digest}"
                )
            value_fn = learned

        policy_path = cfg_kwargs.pop("policy_model_path", None)
        expected_policy_digest = cfg_kwargs.pop("policy_model_digest", None)
        if (policy_path is None) != (expected_policy_digest is None):
            raise ValueError("learned policy model metadata is incomplete")

        prior: LearnedPolicy | None = None
        if policy_path is not None:
            prior = LearnedPolicy(str(policy_path))
            if prior.digest != expected_policy_digest:
                raise ValueError(
                    "learned policy artifact changed after protocol construction: "
                    f"expected digest {expected_policy_digest}, got {prior.digest}"
                )
        cfg_kwargs["seed"] = seed
        config = SearchConfig(**cfg_kwargs)
        if prior is not None and not config.use_prior:
            raise ValueError("learned policy model requires SearchConfig.use_prior=True")
        cutoff_observer = (
            CutoffTelemetryRecorder(str(telemetry_path), case_metadata or {})
            if telemetry_path is not None
            else None
        )
        agent_kwargs: dict[str, Any] = {
            "value_fn": value_fn,
            "cutoff_observer": cutoff_observer,
        }
        if prior is not None:
            agent_kwargs["prior"] = prior
        return ISMCTSAgent(config, **agent_kwargs)
    raise ValueError(f"unknown agent kind: {kind!r}")


def _agent_seed(world_seed: int, side: str) -> int:
    """Deterministic per-agent seed derived from ``(world_seed, side)``.

    Two agents in the same game must never share the seed we hand them
    (that would break tie-breaks across bot instances); mixing the side name
    into the world seed keeps the mapping deterministic and reproducible
    across reruns while cheaply splitting per-side streams.
    """
    # Fold the side label into a 32-bit stream so both A and B get a distinct
    # but reproducible seed from the same world seed. sha256 → 8-byte prefix
    # → int is overkill but stable across Python versions.
    blob = f"{world_seed}:{side}".encode()
    digest = hashlib.sha256(blob).digest()[:8]
    return int.from_bytes(digest, "big", signed=False)


class _CaseRunner:
    """Module-level picklable per-case runner for :func:`run_protocol`.

    Structured as a small dataclass-like class instead of a closure so it
    survives spawn-subprocess pickling — the timed protocol path requires
    a top-level callable it can ship to the child. Tests that monkeypatch
    ``cli.run_game`` / ``cli.build_agent`` still steer behavior because
    ``__call__`` re-resolves those names on this module on every invocation
    (see :func:`build_case_runner` docstring for the seam).
    """

    __slots__ = (
        "agent_a",
        "agent_b",
        "blue_heroes",
        "game_type",
        "map_path",
        "max_steps",
        "red_heroes",
    )

    def __init__(
        self,
        *,
        agent_a: AgentSpec,
        agent_b: AgentSpec,
        red_heroes: tuple[str, ...],
        blue_heroes: tuple[str, ...],
        map_path: str,
        game_type: str,
        max_steps: int,
    ) -> None:
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.red_heroes = red_heroes
        self.blue_heroes = blue_heroes
        self.map_path = map_path
        self.game_type = game_type
        self.max_steps = max_steps

    def __call__(self, case: GameCase) -> GameObservation:
        # Deterministic per-agent seeds. A/B split so the two bots in the
        # same game never collide even when the world seed is small.
        a_seed = _agent_seed(case.world_seed, "A")
        b_seed = _agent_seed(case.world_seed, "B")

        # Route module-level names through this module so tests can monkeypatch
        # ``cli.build_agent`` / ``cli.run_game`` without touching other packages.
        # Re-lookup on each call so test replacements installed after the
        # runner was constructed still take effect.
        import automata.evaluation.cli as _self

        common_metadata = {
            "case_id": case.case_id,
            "world_seed": case.world_seed,
            "a_side": case.a_side,
        }
        agent_a = _self.build_agent(
            self.agent_a,
            a_seed,
            case_metadata={**common_metadata, "agent_label": "A"},
        )
        agent_b = _self.build_agent(
            self.agent_b,
            b_seed,
            case_metadata={**common_metadata, "agent_label": "B"},
        )

        # a_side names the side A controls this game. Whichever side that is
        # gets A on all its heroes; the other side gets B.
        red_agent = agent_a if case.a_side == "RED" else agent_b
        blue_agent = agent_b if case.a_side == "RED" else agent_a

        agents: dict[str, Agent] = {}
        for name in self.red_heroes:
            agents[hero_id(name)] = red_agent
        for name in self.blue_heroes:
            agents[hero_id(name)] = blue_agent

        result = _self.run_game(
            list(self.red_heroes),
            list(self.blue_heroes),
            agents,
            map_path=self.map_path,
            game_type=self.game_type,
            seed=case.world_seed,
            max_steps=self.max_steps,
        )

        winner_side: str | None
        winner = (result.winner or "").upper()
        winner_side = winner if winner in ("RED", "BLUE") else None
        return GameObservation(
            case_id=case.case_id,
            world_seed=case.world_seed,
            a_side=case.a_side,
            winner_side=winner_side,
            rounds=result.rounds,
            steps=result.steps,
            reason=result.reason,
        )


def build_case_runner(
    protocol: EvaluationProtocol,
) -> Callable[[GameCase], GameObservation]:
    """Return a per-case runner suitable for :func:`run_protocol`.

    The returned callable, given a scheduled :class:`GameCase`, constructs A
    and B agents from the protocol's :class:`AgentSpec`s (via
    :func:`build_agent`) with deterministic per-agent seeds derived from
    the world seed, maps each agent to the heroes on its assigned side,
    calls :func:`run_game` with the protocol's map / game type / max_steps /
    world seed, and translates the harness result into a
    :class:`GameObservation` preserving winner / rounds / steps / reason.

    Returned as a top-level :class:`_CaseRunner` instance (not a closure)
    so it can be pickled and shipped to a spawn subprocess by the
    per-case wall-clock timeout supervisor in ``run_protocol``.

    Semi-public seam: tests inject fake ``run_game`` / ``build_agent`` on
    this module to exercise the runner behaviorally without playing games.
    The ``_CaseRunner.__call__`` re-resolves those names on every call so
    injections installed after ``build_case_runner`` returned still take
    effect.
    """
    return _CaseRunner(
        agent_a=protocol.agent_a,
        agent_b=protocol.agent_b,
        red_heroes=protocol.red_heroes,
        blue_heroes=protocol.blue_heroes,
        map_path=protocol.map_path,
        game_type=protocol.game_type,
        max_steps=protocol.max_steps,
    )


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def _positive_int(value: str) -> int:
    """argparse ``type`` for a strictly positive integer.

    Emits an error message containing the word ``"positive"`` so tests
    (and users) get a specific constraint diagnostic rather than the
    generic argparse "invalid int value" message.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    """argparse ``type`` for a strictly positive float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def _non_negative_float(value: str) -> float:
    """argparse ``type`` for a non-negative float (``0`` allowed).

    Distinct from :func:`_positive_float` for knobs where ``0`` is a
    meaningful "disabled" setting — most notably ``puct_c=0`` which
    disables PUCT selection in :class:`SearchConfig`. Emits
    ``"non-negative"`` in the error message so tests and users get a
    specific diagnostic that is textually distinguishable from the
    strictly-positive constraint.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value!r}") from exc
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value!r}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GoA2 agent evaluation matrix.")

    # Legacy matrix flags (unchanged).
    parser.add_argument("--games", type=int, default=20, help="Games per fast matchup.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-games", type=int, default=None, help="Games per ISMCTS matchup.")
    parser.add_argument(
        "--search-iters", type=int, default=8, help="ISMCTS iterations per decision."
    )
    parser.add_argument("--out", type=str, default=None, help="Write results JSON here.")

    # Targeted flags. Presence of both --agent-a and --agent-b switches modes.
    parser.add_argument("--agent-a", choices=_AGENT_KINDS, default=None)
    parser.add_argument("--agent-b", choices=_AGENT_KINDS, default=None)
    parser.add_argument(
        "--paired-seeds",
        type=_positive_int,
        default=_DEFAULT_PAIRED_SEEDS,
        help="Number of paired world seeds (each yields RED + BLUE cases).",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=20_000,
        help="Per-game engine step budget passed to run_game.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path (overrides the default identity layout).",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=_positive_float,
        default=1800.0,
        help=(
            "Per-case wall-clock timeout (seconds). Each case runs in a "
            "one-shot spawned subprocess and is terminated on timeout, "
            "producing a synthetic wall_clock_timeout observation."
        ),
    )

    # Asymmetric ISMCTS knobs.
    parser.add_argument("--a-iterations", type=_positive_int, default=None)
    parser.add_argument("--b-iterations", type=_positive_int, default=None)
    parser.add_argument("--a-cutoff-rounds", type=_positive_int, default=None)
    parser.add_argument("--b-cutoff-rounds", type=_positive_int, default=None)
    parser.add_argument("--a-uct-c", type=_positive_float, default=None)
    parser.add_argument("--b-uct-c", type=_positive_float, default=None)
    parser.add_argument("--a-puct-c", type=_non_negative_float, default=None)
    parser.add_argument("--b-puct-c", type=_non_negative_float, default=None)
    parser.add_argument("--a-no-prior", action="store_true", default=False)
    parser.add_argument("--b-no-prior", action="store_true", default=False)
    parser.add_argument(
        "--a-value-model",
        type=str,
        default=None,
        help="Learned value JSON artifact for agent A (ISMCTS only).",
    )
    parser.add_argument(
        "--b-value-model",
        type=str,
        default=None,
        help="Learned value JSON artifact for agent B (ISMCTS only).",
    )
    parser.add_argument(
        "--a-policy-model",
        type=str,
        default=None,
        help="Learned policy JSON artifact for agent A (ISMCTS with prior only).",
    )
    parser.add_argument(
        "--b-policy-model",
        type=str,
        default=None,
        help="Learned policy JSON artifact for agent B (ISMCTS with prior only).",
    )
    parser.add_argument(
        "--a-cutoff-telemetry",
        type=str,
        default=None,
        help="Append agent A's nonterminal ISMCTS cutoffs to this JSONL path.",
    )
    parser.add_argument(
        "--b-cutoff-telemetry",
        type=str,
        default=None,
        help="Append agent B's nonterminal ISMCTS cutoffs to this JSONL path.",
    )

    return parser


# --------------------------------------------------------------------------- #
# Targeted mode                                                               #
# --------------------------------------------------------------------------- #


def _build_agent_spec(
    *,
    label: str,
    kind: str,
    iterations: int | None,
    cutoff_rounds: int | None,
    uct_c: float | None,
    puct_c: float | None,
    no_prior: bool,
    value_model: str | None,
    cutoff_telemetry: str | None = None,
    policy_model: str | None = None,
) -> AgentSpec:
    """Build an :class:`AgentSpec` whose ``params`` reflect the effective
    SearchConfig (minus dynamic ``seed``) so identity captures every knob.

    Non-ISMCTS agents ignore the search knobs and carry an empty params dict
    — they have no reproducible-search state, so nothing to hash beyond
    ``name`` / ``kind``.
    """
    if kind != "ismcts":
        return AgentSpec(name=label, kind=kind, params={})
    if policy_model is not None and no_prior:
        raise ValueError("learned policy model is incompatible with no_prior")

    cfg_kwargs: dict[str, Any] = {
        "iterations": iterations if iterations is not None else _DEFAULT_ISMCTS_ITERATIONS,
        "cutoff_rounds": (
            cutoff_rounds if cutoff_rounds is not None else _DEFAULT_ISMCTS_CUTOFF_ROUNDS
        ),
        "use_prior": not no_prior,
    }
    if uct_c is not None:
        cfg_kwargs["uct_c"] = uct_c
    if puct_c is not None:
        cfg_kwargs["puct_c"] = puct_c
    cfg = SearchConfig(**cfg_kwargs)
    params = asdict(cfg)
    # Dynamic per-case seed is set by build_agent; it must not participate
    # in identity or the checkpoint would depend on run-order.
    params.pop("seed", None)
    if value_model is not None:
        model_path = str(Path(value_model))
        model = LearnedValue(model_path)
        params["value_model_path"] = model_path
        params["value_model_digest"] = model.digest
    if policy_model is not None:
        model_path = str(Path(policy_model))
        policy = LearnedPolicy(model_path)
        params["policy_model_path"] = model_path
        params["policy_model_digest"] = policy.digest
    if cutoff_telemetry is not None:
        params["cutoff_telemetry_path"] = str(Path(cutoff_telemetry))
    return AgentSpec(name=label, kind=kind, params=params)


def _default_checkpoint_path(digest: str) -> Path:
    return _CHECKPOINT_DIR / f"{digest}.jsonl"


def _print_targeted_header(
    *,
    protocol: EvaluationProtocol,
    checkpoint: Path,
    paired_seeds: int,
) -> None:
    def value_description(spec: AgentSpec) -> str:
        model_path = spec.params.get("value_model_path")
        model_digest = spec.params.get("value_model_digest")
        if model_path is None and model_digest is None:
            return "heuristic value"
        if model_path is None or model_digest is None:
            raise ValueError("learned value model metadata is incomplete")
        digest = str(model_digest)
        return f"learned value={model_path} ({digest[:12]})"

    def policy_description(spec: AgentSpec) -> str:
        model_path = spec.params.get("policy_model_path")
        model_digest = spec.params.get("policy_model_digest")
        if model_path is None and model_digest is None:
            return "heuristic policy" if spec.params.get("use_prior", True) else "policy disabled"
        if model_path is None or model_digest is None:
            raise ValueError("learned policy model metadata is incomplete")
        digest = str(model_digest)
        return f"learned policy={model_path} ({digest[:12]})"

    print(
        f"Protocol: {protocol.agent_a.name} ({protocol.agent_a.kind}, "
        f"{value_description(protocol.agent_a)}, {policy_description(protocol.agent_a)}) vs "
        f"{protocol.agent_b.name} ({protocol.agent_b.kind}, "
        f"{value_description(protocol.agent_b)}, {policy_description(protocol.agent_b)})"
    )
    print(f"  paired-seeds={paired_seeds} → {paired_seeds * 2} cases")
    print(f"  max_steps={protocol.max_steps}  identity={protocol.identity_digest()}")
    telemetry = [
        f"{label}={spec.cutoff_telemetry_path}"
        for label, spec in (("A", protocol.agent_a), ("B", protocol.agent_b))
        if spec.cutoff_telemetry_path is not None
    ]
    if telemetry:
        print(f"  cutoff telemetry enabled: {'  '.join(telemetry)}")
    if protocol.case_timeout_seconds is not None:
        # Include the effective per-case wall-clock budget so operators can
        # confirm what a run was configured with (the config value alone is
        # not sufficient to satisfy the "timeout count" summary line below —
        # see :func:`_print_targeted_summary`).
        print(f"  case_timeout_seconds={protocol.case_timeout_seconds}")
    print(f"Checkpoint: {checkpoint}")


def _print_targeted_summary(summary: EvaluationSummary) -> None:
    lo, hi = summary.wilson_ci()
    rate = summary.decisive_a_rate
    print(
        f"Summary: A={summary.a_wins}  B={summary.b_wins}  "
        f"draws={summary.draws}  max_steps={summary.max_step_terminations}  "
        f"timeouts={summary.timeout_terminations}"
    )
    print(
        f"  A decisive rate={rate:.1%}  Wilson 95% CI=[{lo:.1%}, {hi:.1%}]  "
        f"avg_rounds={summary.avg_rounds:.1f}  avg_steps={summary.avg_steps:.1f}"
    )
    screen = "PASS" if summary.screening_passes() else "FAIL"
    promo = "PASS" if summary.promotion_passes() else "FAIL"
    print(f"SCREEN: {screen}    PROMOTION: {promo}")


def _run_targeted(
    args: argparse.Namespace,
) -> int:
    """Run one targeted A-vs-B protocol; return the process exit code."""
    # source_identity is looked up on the module so tests can monkeypatch it.
    import automata.evaluation.cli as _self

    head, dirty = _self.source_identity()

    agent_a = _build_agent_spec(
        label="A",
        kind=args.agent_a,
        iterations=args.a_iterations,
        cutoff_rounds=args.a_cutoff_rounds,
        uct_c=args.a_uct_c,
        puct_c=args.a_puct_c,
        no_prior=args.a_no_prior,
        value_model=args.a_value_model,
        cutoff_telemetry=args.a_cutoff_telemetry,
        policy_model=args.a_policy_model,
    )
    agent_b = _build_agent_spec(
        label="B",
        kind=args.agent_b,
        iterations=args.b_iterations,
        cutoff_rounds=args.b_cutoff_rounds,
        uct_c=args.b_uct_c,
        puct_c=args.b_puct_c,
        no_prior=args.b_no_prior,
        value_model=args.b_value_model,
        cutoff_telemetry=args.b_cutoff_telemetry,
        policy_model=args.b_policy_model,
    )

    # World seeds: contiguous block starting at ``--seed`` so ``--paired-seeds``
    # only changes the schedule length, never the identity payload (protocol
    # identity excludes ``world_seeds`` by construction).
    world_seeds = tuple(range(args.seed, args.seed + args.paired_seeds))

    # ``run_game`` accepts an explicit map path; use the harness default.
    from ..runtime.harness import DEFAULT_MAP

    protocol = EvaluationProtocol(
        agent_a=agent_a,
        agent_b=agent_b,
        red_heroes=tuple(RED),
        blue_heroes=tuple(BLUE),
        world_seeds=world_seeds,
        map_path=DEFAULT_MAP,
        game_type="QUICK",
        max_steps=args.max_steps,
        source_revision=head,
        dirty_tree_hash=dirty,
        case_timeout_seconds=args.case_timeout_seconds,
    )

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else _default_checkpoint_path(protocol.identity_digest())
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    _print_targeted_header(
        protocol=protocol,
        checkpoint=checkpoint,
        paired_seeds=args.paired_seeds,
    )

    # Progress hint: how many cases are already checkpointed.
    scheduled_case_ids = {c.case_id for c in protocol.cases()}
    if checkpoint.exists():
        existing = [o for o in load_observations(checkpoint) if o.case_id in scheduled_case_ids]
        if existing:
            print(f"Cached: {len(existing)}/{len(scheduled_case_ids)} cases from checkpoint")

    runner = build_case_runner(protocol)
    observations = _self.run_protocol(protocol, checkpoint_path=checkpoint, run_case=runner)
    print(f"Completed: {len(observations)}/{len(scheduled_case_ids)} cases")

    # Use the module-level ``summarize`` so tests that monkeypatch it (fake
    # summary) can still steer the printed verdict.
    summary = _self.summarize(observations)
    _print_targeted_summary(summary)
    return 0 if (summary.screening_passes() and summary.promotion_passes()) else 1


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int | None:
    """CLI entry point. See module docstring.

    ``argv`` overrides ``sys.argv[1:]`` when provided (testable). Returns
    ``None`` on success in matrix mode (legacy behavior); targeted mode
    returns ``0`` on success and ``1`` when either gate fails.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Targeted-mode gate: both agent flags must appear together. We check
    # BEFORE dispatching so the message names the pairing constraint rather
    # than surfacing a downstream NoneType error.
    if (args.agent_a is None) != (args.agent_b is None):
        parser.error("targeted mode requires both --agent-a and --agent-b")

    if args.a_value_model is not None and args.agent_a != "ismcts":
        parser.error("--a-value-model is valid only when --agent-a is ismcts")
    if args.b_value_model is not None and args.agent_b != "ismcts":
        parser.error("--b-value-model is valid only when --agent-b is ismcts")
    if args.a_policy_model is not None and args.agent_a != "ismcts":
        parser.error("--a-policy-model is valid only when --agent-a is ismcts")
    if args.b_policy_model is not None and args.agent_b != "ismcts":
        parser.error("--b-policy-model is valid only when --agent-b is ismcts")
    if args.a_policy_model is not None and args.a_no_prior:
        parser.error("--a-policy-model is incompatible with --a-no-prior")
    if args.b_policy_model is not None and args.b_no_prior:
        parser.error("--b-policy-model is incompatible with --b-no-prior")
    if args.a_cutoff_telemetry is not None and args.agent_a != "ismcts":
        parser.error("--a-cutoff-telemetry is valid only when --agent-a is ismcts")
    if args.b_cutoff_telemetry is not None and args.agent_b != "ismcts":
        parser.error("--b-cutoff-telemetry is valid only when --agent-b is ismcts")

    if args.agent_a is not None and args.agent_b is not None:
        return _run_targeted(args)

    # --- legacy matrix mode --- #
    results = run_matrix(
        args.games,
        args.seed,
        search_games=args.search_games,
        search_iters=args.search_iters,
    )
    for r in results:
        print(r.summary())

    if args.out:
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "games": args.games,
            "search_games": (
                args.search_games if args.search_games is not None else min(args.games, 6)
            ),
            "base_seed": args.seed,
            "search_iterations": args.search_iters,
            "red": RED,
            "blue": BLUE,
            "notes": (
                "Rung-0 baseline. Fast matchups (random/heuristic) use --games; "
                "ISMCTS matchups use a small --search-games at low --search-iters "
                "because a single ISMCTS game is expensive (~28s at 2 iters). "
                "ISMCTS rows are therefore directional (wide Wilson CI), not "
                "conclusive; rerun with higher budgets for a real strength claim. "
                "Every later rung must beat the agent it improves on in these "
                "matchups over a meaningful sample."
            ),
            "matchups": [_result_dict(r) for r in results],
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nWrote baselines to {args.out}")
    return None


if __name__ == "__main__":
    # Propagate main()'s return as the process exit code. Legacy matrix mode
    # returns ``None`` (success — SystemExit(None) exits 0); targeted mode
    # returns 0 on both gates passing and 1 otherwise, which CI / callers
    # need to observe in the shell.
    raise SystemExit(main())
