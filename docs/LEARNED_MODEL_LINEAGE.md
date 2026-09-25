# Learned model lineage

> **Historical lineage, not the fresh Gen1 baseline.** Results and verdicts below
> are preserved unchanged. Legacy model execution/bridge code has been retired;
> reproducing those runs requires their original code snapshots. See
> [AI_EXPERIMENT_JOURNAL.md](AI_EXPERIMENT_JOURNAL.md) for the reset and current
> implementation status. No model below is automatically promoted into Gen1.

Append-only operator ledger for material learned-policy generations. Immutable
manifests, digests, split membership, metrics, and arena observations remain the
authoritative evidence under each listed `runs/` directory.

| Lineage | Parent | Material change | Dataset / split | Model | Decision |
|---|---|---|---|---|---|
| Generation 2 champion | Earlier lineage | Observation v3 / tensor v1 baseline | `runs/self-play-depth8-balanced-adaptive-t05-v1-64` | `d409b96c7993fba3356bbde82b2bb8b887e5e136dcd55db97f904f1a84eeee96` | Last arena-validated champion. |
| Rejected immediate-action v5 | Generation 2 | Earlier `IMMEDIATE_ACTION` target experiment | `runs/self-play-depth8-balanced-adaptive-t05-immediate-action-v5-64` | `6123e94c024a2fa8492f7aea69743bb171dcc09bb1f4f3b3ae8a54686271a382` | Rejected: paired arena lost 16–48 to generation 2 (`runs/arena-6123-vs-d409-depth8-balanced-64`). |
| Native-v4 generation 3 | Generation 2 through `decision-v4-to-v3-v1` | First decision-observation v4 / tensor v2 artifact; learned owned continuation, request-aware v1, and `public-consequence-v4` | Dataset `826aa943fc0b6b7612360ed72315814098156568349e2e3f5073a830d5d39bb0`; split `6e82c1e05a6d6cfd34b168488de5f9703c1d076b400846996b817c46ad295f4e` | `e7cae7b91d9356f9a97587b37ae02bea29910d11cafdb04b5056819831c73a56` | Bootstrap only; not arena-promoted. Establishes the homogeneous native-v4 parent lineage. |
| Native-v4 generation 4 | Native-v4 generation 3 | First fully native-v4 child trained from request-aware-v1 targets | Dataset `25b263d07875cf5b51c739a091231f976839c12a3a3aef91e0bd73a40ad8b2f1`; split `03897f48d8cb656d04c2dc2ff8ff84501c30b2fbcdb365c0fd999301b5c3658d` | `6efbbac6333f8f2ff5ee83037d370be3014f505cf03456952208f66b8584ca04` | Rejected. Exact-split pairwise gate: candidate `0.9964108134`, parent `0.9982667956`; failed overall plus `ATTACK_TARGET`, `MOVEMENT_DESTINATION`, and `PLANNING`. Exploratory paired screen then lost 0–16 to generation 2 (`runs/arena-stage5-gen4-vs-gen2-exploratory-screen-8`). |
| Stable-turn v2 diagnostic | Rejected generation 4, diagnostic use only | Eight-game mechanical/coverage check of `request-aware-v2` and `STABLE_TURN`; no training or promotion | Dataset `376f8a753bea2b4d3964702040b4483c37c0fd736cb80782fada144e9f77af9f`; seeds `[41100,41108)` | None | Passed operational diagnostics: 8/8 completed, 4–4 team outcomes, 8,833 decisions, 6,039 stable-turn leaves, no decision timeout/progression/visit failures; p95 `1.6285s`, max `10.6486s`. Does not make generation 4 an eligible parent. |
| Stable-turn v2 generation 5 | Generation 2 champion through `decision-v4-to-v3-v1` | First 64-game `request-aware-v2` / selective `STABLE_TURN` corpus and native-v4 candidate | Dataset `a3693aaa5530937abdb7c3d301abd222aefde479afcd31dfb015d10161a9d22b`; split `53d45ae81a6285bd2c5dc0778726f8d67dbba91fdba9e9bc586afdbbb7064e63`; seeds `[41200,41264)` | `3fd1d173683de03f7b16bd50a854cf5f1ea44bd9ada951bdb782fe04ecc8442d` | Rejected before arena. Against native-v4 generation 3 on identical validation membership, overall pairwise accuracy was `0.9071490790` versus `0.9073779108`; also regressed on `ATTACK_TARGET`, `DEFENSE_REACTION`, and `PLANNING`. `MOVEMENT_DESTINATION` had no informative evidence for either artifact. |
| Stable-turn v2 32-visit diagnostic | Generation 2 champion through `decision-v4-to-v3-v1` | Same-seed budget probe after generation 5 rejection; no training or promotion | Dataset `055ba7b44e4ae77758e73dd50dd479cee98514928c9cf48b3c494e7ad7fb5d3f`; seeds `[41200,41208)` | None | Operationally healthy: 8/8 completed, 5,197 decisions, no timeout/progression/visit failures; p95 `9.6659s`, max `13.7634s`. Compared with eight visits, candidate coverage and Q separation increased, and the expected broader visit target raised planning/action entropy. This is a compute tradeoff, not evidence that exploration is harmful; value learning proceeded by scaling independent eight-visit games instead. |
| Stable-turn v2 generation 6 | Generation 2 champion through `decision-v4-to-v3-v1` | Scaled value-learning corpus: 256 games at the runtime-sized eight-visit budget | Dataset `b3344d5b309582d08d123abf8d1e1386ec17bb68c8f0911194d7a9f021d858f0`; split `fea822839670985f2bc6345162e20240bbce144687e7a9cc8d4d372f4a29c763`; seeds `[41400,41656)` | `317fd604c666f358917841a41a3e888045ce104f6d334eb59ec4402cf2d50edc` | Rejected for promotion: exact-split policy ranking was `0.9055293777` versus generation 3 `0.9060024847`, with `ATTACK_TARGET` regression and no `MOVEMENT_DESTINATION` evidence. Value showed held-out signal (equal-game Brier `0.209143` versus neutral `0.25`), strongest after round 6. Same-artifact L/L-vs-L/H screen was halted after its first two cases both reached the 3,600-second wall-clock limit without a winner (`runs/arena-stage7-gen6-ll-vs-lh-value-ablation-screen-8`). The exact L/H-vs-L/H control also timed out 2/2 at rounds 84 and 86 (`runs/arena-stage7-gen6-lh-self-control-1pair`), removing learned value as the cause of long games. Against fixed gen2 on the same paired seed, both gen6 L/H and L/L lost 0–2 at 11.5 average rounds; L/H took 10m34s and L/L 17m52s (`runs/arena-stage7-gen6-{lh,ll}-vs-gen2-value-{control,test}-1pair`). Learned-value strength remains unresolved, with an observed ~69% wall-clock penalty in this probe. |

## Search implementation checkpoints

| Commit | Change | Generated model status |
|---|---|---|
| `171fed7` | Typed ranking evidence and executable offline promotion gate | Source checkpoint used for native-v4 generations 3 and 4. |
| `8ca30ac` | Empty baseline checkpoint after native-v4 generation/training | No model change. |
| `2ebce82` | Adds experimental `STABLE_TURN` and immutable `request-aware-v2` scheduling | No dataset/model yet; requires a fresh diagnostic and generation. |

Promotion remains fail-closed: offline gate evidence and paired arena evidence are
separate requirements. A bootstrap parent, lower loss, or exploratory arena does
not update the champion by itself.
