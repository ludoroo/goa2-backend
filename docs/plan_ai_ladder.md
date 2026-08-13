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
larger deliberate run. The targeted Rung-1 protocol now adds a promoted
ISMCTS-vs-Heuristic result: **12-0 over 6 paired seeds** at 4 iterations and a
1-round cutoff, with no max-step terminations and a 95% Wilson interval of
**[75.7%, 100%]**. The exact source and protocol identity are recorded in
`baselines.json`.

## Next rungs

Even though bots are now in production for Random / Heuristic / bounded
ISMCTS, the *strength* work below the ladder has not moved: the shipped bots
use the same hand-crafted `HeuristicValue` / `HeuristicPrior` established at
Rung 0. Improving strength further is orthogonal to server plumbing and stays
on the ladder below.

### Rung 1 — Squeeze the search (no learning)
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
   Run in the background; record numbers. **PARTIAL** — the deterministic,
   paired JSONL protocol now resumes exact missing cases and rejects stale
   source/config identities. The canonical 4-iteration/1-round screen completed
   ISMCTS-vs-Heuristic at **12-0**, passing both the point-estimate and Wilson
   promotion gates. Prior-on-vs-off remains incomplete at **5/12** (prior-on
   leads 4-1): the next game ran for more than 98 minutes without reaching the
   20,000-step cap, so no prior-strength claim is promoted from that run. Each
   targeted case now runs in a one-shot spawned process with a configurable
   wall-clock limit (`--case-timeout-seconds`, default **1800 s**). A timeout is
   hard-terminated, checkpointed as `wall_clock_timeout`, and blocks both gates;
   changing the limit creates a fresh protocol identity for an explicit retry.
3. **Tune** `iterations`, `cutoff_rounds`, `uct_c`/`puct_c`, widening via the
   matrix. Gate: search strength must not regress. **TODO.**

### Rung 2 — Learned value function. **← active**
1. **DONE — normalized value seam.** `ValueFn` now returns `[-1, 1]` from the
   acting team's perspective. `HeuristicValue` owns its `tanh(score / scale)`
   conversion; search maps any valid value exactly once to `[0, 1]`. This is
   compatible with logistic log-odds today and a direct tanh NN head later.
2. **DONE — compact data pipeline.** Benchmark-roster Heuristic self-play writes
   six-feature decision rows and terminal labels without full `GameState`
   snapshots. Generation is deterministic by world seed and discards incomplete
   games. The trainer splits by game, weights every game equally, and keeps
   scikit-learn in a training-only dependency group.
3. **DONE — portable model seam.** Logistic and gradient-boosted-tree training
   export versioned JSON; dependency-free `LearnedValue` validates
   feature/roster compatibility and returns `tanh(raw_score / 2)`. Targeted
   evaluation hashes the artifact into its checkpoint identity and verifies it
   again before each spawned game.
4. **FAILED — first logistic candidate.** Trained on 60 Heuristic-vs-Heuristic
   games (seeds 1000–1059; 70,500 decision rows; 42/9/9 game split). Held-out
   metrics looked strong: accuracy **93.2%**, log-loss **0.158**, Brier **0.041**,
   ECE **0.056**. At equal search budget against `HeuristicValue`, however, the
   learned model scored **3-9 (25%)** over 6 paired evaluation seeds, Wilson 95%
   CI **[8.9%, 53.2%]**, with no timeout or max-step games. It fails both gates
   and is not promoted.
5. **DONE — cutoff-state diagnosis.** A read-only observer recorded 6,521 exact
   non-terminal states where learned search queried its value function. The
   telemetry rerun reproduced the same 3-9 outcome, confirming no observer
   effect. Learned and heuristic values were negatively correlated
   (**-0.318**), disagreed in sign **54.0%** of the time, and differed by
   **0.993** on average despite the normalized `[-1, 1]` range. Learned outputs
   saturated at `|v| >= 0.95` on **65.8%** of queries versus **0%** for the
   heuristic. **58.5%** of cutoff rows had at least one feature beyond 3 training
   standard deviations; `level_diff` was the clearest shift (p95 `|z|` **5.28**,
   max **8.79**), followed by `push_diff` (p95 **3.32**). The problem was worse
   when learned A controlled RED: correlation **-0.581**, saturation **71.0%**,
   OOD **70.1%**, sign disagreement **61.1%**. This confirms severe distribution
   shift and logistic overconfidence, not merely insufficient row count.
6. **DONE — terminal-labelled cutoff data.** A deterministic sampler clones
   sparse cutoff states and continues each clone to terminal under fresh
   Heuristic policies, with independent RNG, step/round caps, durable sample IDs,
   and source-game-grouped splits. At 12 source games (84 labels), cutoff
   logistic still opened 1-3; cutoff GBM completed 2-9 with one timeout. At 36
   games (264 labels), logistic reached **6-6** over the full gate. At 72 source
   games (548 labels), offline logistic improved to accuracy **83.2%**, log-loss
   **0.461**, Brier **0.146**, ECE **0.099**; search reached **6-5** in decisive
   games but had one timeout, so both gates still failed. This learning curve
   confirms correct-distribution labels help substantially, but six features
   plateau near parity.
7. **FAILED — richer features and GBM.** Added an explicit backward-compatible
   `rich-v1` schema with 24 absolute composition, card-zone, battle-presence,
   progress, round, and wave features, plus portable pure-Python GBM inference.
   A review found and fixed an initial card-count bug that omitted each team's
   second hero; those first artifacts were invalidated and regenerated. On 36
   corrected rich-labelled source games (281 labels), held-out metrics were
   still nearly perfect (logistic accuracy **97.5%**, GBM **100%**). Corrected
   rich GBM opened 3-1 but collapsed to **3-9** over the full 12-case gate, with
   no timeout/max-step games. This is source-game overfit / noisy
   single-continuation targets, not evidence for more model capacity. No rich
   model was promoted.
8. **Next experiment.** Replace one-shot binary cutoff labels with multiple
   independent continuations per sampled state and train against empirical win
   probability (or a regressor on normalized expected value). Keep splits by
   source game and measure label variance before considering a neural network.
   Gate remains: equal-budget `ISMCTSAgent(value_fn=LearnedValue)` must beat
   `HeuristicValue` search with no timeout/max-step cases.

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
- No neural-network infrastructure until lower-capacity models have reliable
  multi-continuation targets and still plateau.

## Known issues

- The offline cutoff generator now has step/round caps, resumable source-game
  checkpoints, and a POSIX wall-clock timer. This was added after seed 2060
  stalled inside one search decision for more than seven hours.
- The previously-noted pre-existing nits are fixed: the
  `heuristic_agent.py` `_qrs` mypy `assignment` error (disambiguated the dict vs
  object-attr locals) and the ruff `__all__`/import-sort issues in
  `agents/__init__.py` and older `tests/ai/*` files. `mypy src/automata` and
  `ruff check src/automata tests/ai` are clean.
