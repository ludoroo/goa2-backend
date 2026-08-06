# AI Ladder Plan (automata)

Living plan for the Guards of Atlantis II AI (`src/automata`). The end goal is a
strong AI opponent: ISMCTS guided by a learned value + policy. We climb a ladder
where **every rung must beat the previous baseline on the eval matrix**
(`automata.evaluation.cli` → `baselines.json`). We escalate ML complexity only
when the evidence justifies it — linear/GBM before any neural net.

## Guiding constraints

- **ISMCTS is required** — simultaneous moves + hidden hands make minimax
  infeasible. Opponents are folded into a fixed default policy (cut B) and their
  hidden commits into `runtime.determinize`; every tree node is one of *our*
  decisions (a MAX node).
- **Full rollouts are too expensive** (~1.4 ms/clone, and an ISMCTS *game* costs
  ~28 s even at 2 iters). Rollouts truncate at a round-count cutoff and defer to
  a value function. Search eval is therefore small-sample and deliberate.
- **The eval matrix is the yardstick.** No strength claim without it.

## Architecture seams (all in place, backed by hand-crafted impls)

| Seam | Interface | Today's impl | Learned drop-in |
|------|-----------|--------------|-----------------|
| Value | `evaluation.value.ValueFn` | `HeuristicValue` (→ `evaluate_state`) | `LearnedValue(model)` |
| Features | `evaluation.features.state_features` / `feature_vector` | 6 differential features + `FEATURE_WEIGHTS` | NN encoder input |
| Policy | `search.prior.Policy` → `PolicyResult(order, weights\|None)` | `HeuristicPrior` (weights = scores) | `LearnedPolicy(model)` |
| Data | `runtime.trajectory.TrajectoryRecorder` | `Null`/`InMemory`/`Jsonl` | training-set source |

`ISMCTSAgent` takes an injectable `value_fn` and builds a `HeuristicPrior`; the
search loop reads `value_fn(...)` at leaves and `policy(...).order` at expansion.
The `weights` slot is unused today — it's reserved for PUCT (Rung 1) / learned
policy (Rung 3), so no signature will change when those land.

## Status

- **Rung 0 — instrumentation & baselines: DONE.** Eval matrix + committed
  `baselines.json`. Trajectory recording (full `GameState` snapshots, off by
  default). Feature extraction. Value/Policy protocols.
- **Heuristic fixed** (commits `7c16966`, `e88d683`): `_hex_score` gained an
  intra-zone enemy-approach gradient. 39.5% → **93.5%** vs random; games no
  longer stall (heur-vs-heur terminates ~22 rounds). This strengthens the
  ISMCTS default policy *and* prior.
- **Server integration (Tasks 1-9 of `plan_ai_backend_integration.md`): DONE.**
  Random, Heuristic, and (bounded) ISMCTS bots can now be created through the
  public `POST /games` API. See below.

### Server bot integration (delivered)

The `automata` search stack is now wired into `goa2.server` so a bot can share
the same engine, persistence, replay, clock, and broadcast paths as a human
opponent. What shipped:

| Piece | Where | Notes |
|-------|-------|-------|
| Server-neutral driver | `src/automata/runtime/driver.py` | One-decision-at-a-time API used by both headless self-play and the live server. Handles Emmitt FINISH, upgrade scoping, team-addressed input, and refuses to answer for a human. |
| Anchored ISMCTS | `src/automata/search/{agent,ismcts}.py` | Root decision-owner is explicit; a Wasp/Xargatha bot never speaks for its (potentially human) teammate. |
| Persisted bot metadata | `src/goa2/server/bot_models.py`, `registry.py`, `engine/persistence.py` | `ManagedGame.bot_specs` round-trips through disk; legacy save files without the field still load. |
| Coordinator | `src/goa2/server/bots.py` | Async worker: one decision per locked mutation, staleness revalidation, agent-cache seeding per game, tombstone-safe teardown, replay/log parity with human seams. |
| Lifecycle wiring | `server/app.py`, `routes_games.py`, `ws.py`, `time_control.py` | `start_bot_lifecycle` runs on game **creation** (`POST /games`) and on lifespan **restore**; it auto-readies bot heroes, persists + reconciles the clock, and hands off to the coordinator. Ongoing REST / WebSocket / timer-driven mutations call the lighter, idempotent `schedule_bot_drive` (via `timed_rest_mutation` for REST and via the WS handler / deadline worker directly) so the coordinator resumes without repeating the ready / clock reconciliation work. |
| Public request boundary | `server/models.py`, `docs/CLIENT_INTEGRATION_GUIDE.md` | Optional `bots` on `POST /games`; draft endpoints explicitly reject `bots` with a 422. |
| Bounded ISMCTS | `automata/search/config.py`, `server/bots.py` | Process-wide semaphore, per-decision timeout, queue-wait timeout, cached per-hero Heuristic fallback, invalid-decision fallback, shutdown drain. `ismcts_metrics` counter exposes fallback / latency / late-completion / queue-depth. |

Current supported production modes:

- **Random-vs-Random** (smoke / low-difficulty).
- **Heuristic-vs-Random** and **Heuristic-vs-Heuristic** (headline strength
  today — see baselines below).
- **Human-vs-Heuristic** and **Human-vs-Random** (single-player experience).
- **Bounded ISMCTS** opt-in via `bots.<hero>.search` (`iterations`, and
  `decision_timeout_seconds` capped at 5 s in production). Falls back to
  Heuristic on any bound violation.

Explicit non-goals for this integration (still parked):

- **Draft-created bot games.** `POST /drafts` / `PATCH /drafts/{id}/settings`
  reject a `bots` key with a 422; configure bots via a direct `POST /games`.
- **Learned value / policy models.** Rungs 2-3 below.
- **Distributed / process-pool search.** Thread-offload keeps the event loop
  responsive but does not defeat the GIL. Bounded concurrency + tight
  per-decision timeouts protect throughput today; a process pool is only
  justified once measured load requires it.
- **Client controls to swap bots mid-game.** Bot assignments are frozen at
  game creation.

### Current baseline (`baselines.json`, 2v2 Wasp/Xargatha vs Arien/Brogan)

| Matchup | A win-rate | Note |
|---------|-----------|------|
| random vs random | 50% | sanity ✓ |
| heuristic vs random | 95% | heuristic dominates ✓ |
| ismcts vs heuristic | 100% (4 games) | directional — search beats base policy |
| ismcts vs ismcts_noprior | 100% (4 games) | directional — prior helps |

ISMCTS rows are tiny-sample (games are expensive); treat as directional until a
larger deliberate run.

## Next rungs

Even though bots are now in production for Random / Heuristic / bounded
ISMCTS, the *strength* work below the ladder has not moved: the shipped bots
use the same hand-crafted `HeuristicValue` / `HeuristicPrior` established at
Rung 0. Improving strength further is orthogonal to server plumbing and stays
on the ladder below.

### Rung 1 — Squeeze the search (no learning). **← next**
Cheap, high-confidence wins before any ML:
1. **PUCT selection.** Use `PolicyResult.weights` as prior `P(a)` in the UCB
   term: `Q(a) + c·P(a)·√N_parent/(1+N(a))`. Today the prior only orders
   expansion; PUCT also biases selection. Requires: normalize weights →
   probabilities, thread into `Node.select`, add a `puct_c` config knob.
   **DONE but DEFAULT OFF.** Implemented (`Node.select` + `_normalize_weights` +
   `puct_c`). Measured **2-10 (16.7%) vs plain UCB1** at 8 iters / 12 games —
   PUCT *hurt* at low budget: a strong hand-crafted prior over-commits and
   under-explores, while UCB1's force-try-every-child explores better with so
   few playouts. So `puct_c` defaults to 0 (UCB1). Kept as a knob; expected to
   pay off with a *learned* policy (Rung 3) and/or higher iteration budgets —
   revisit then. Lesson: expansion-ordering prior helps (kept on); prior-in-
   selection does not, yet.
2. **Larger, deliberate ISMCTS eval.** Establish a *real* (not 4-game) win-rate
   for ismcts-vs-heuristic and prior-on-vs-off, so Rung-1 gains are measurable.
   Run in the background; record numbers. **PARTIAL** — 16-iter/40-game runs are
   >1h and impractical as a loop; the PUCT comparison used 8 iters/12 games
   (~30min). Need a cheaper, repeatable eval protocol (fewer iters, or cache).
3. **Tune** `iterations`, `cutoff_rounds`, `uct_c`/`puct_c`, widening via the
   matrix. Gate: search strength must not regress. **TODO.**

### Rung 2 — Learned value function
1. Generate self-play trajectories (`JsonlRecorder`) at scale.
2. Build a training-data loader that joins decision rows → game outcome, over
   `feature_vector` (or raw snapshot → features offline).
3. Fit a model (start **logistic / linear**, then GBM) predicting win prob;
   wrap as `LearnedValue(ValueFn)`.
4. Gate: `ISMCTSAgent(value_fn=LearnedValue)` must beat `HeuristicValue` search
   on the matrix.

### Rung 3 — Learned policy prior
1. From the same trajectories, learn `P(move | state)` (state → chosen key).
2. Wrap as `LearnedPolicy(Policy)` returning real `weights`; feeds PUCT directly.
3. Gate: beat `HeuristicPrior` at equal iteration budget.

### Rung 4 — NN + joint training (only if Rungs 2–3 plateau)
Single net with value + policy heads; AlphaZero-style self-play → train → eval
loop, ISMCTS-adapted. Big infra/compute step — justified only once linear/GBM
stops improving. `state_features` becomes the encoder input; `ValueFn`/`Policy`
become the net heads.

## Deliberate non-goals / parked

- **Openings book** (`openings-wip` branch): Round-1 opening data (colors +
  minions + partial hexes). Could later serve as a Round-1 policy prior, but
  positioning extraction stalled (~55% auto-resolved). Parked; revisit only if a
  Round-1-specific prior is wanted.
- No ML infra until Rung 2 evidence justifies it.

## Known issues

- None outstanding. The previously-noted pre-existing nits are fixed: the
  `heuristic_agent.py` `_qrs` mypy `assignment` error (disambiguated the dict vs
  object-attr locals) and the ruff `__all__`/import-sort issues in
  `agents/__init__.py` and older `tests/ai/*` files. `mypy src/automata` and
  `ruff check src/automata tests/ai` are clean.
