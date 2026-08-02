# STAGE 8 FORMAL QUALIFICATION GATE PROTOCOL

## Status

Frozen protocol. Tag: `stage8-gate-protocol-v1`. Parameters confirmed with
the user on 2026-08-02 (see `paper/STAGE8_PLAN_DRAFT.md` §5). Unlike every
prior Stage 8 arm (arm0/arm1/arm2a/arm2b: diagnostic pilots only, no
PASS/FAIL claim), this stage makes a formal competence-gate decision.

## 1. Purpose

Re-run Stage 7C-Q1's competence gate
(`STAGE1_TO_STAGE7_EXPERIMENT_FREEZE.md` §10.4.9, FAIL) at the identical
scale (20 seeds × 400,000 joint environment steps) and with the identical
gate threshold structure, training under the configuration selected by the
Stage 8 pilot chain:

- Double DQN, Base Reward V2 (`active_time_cost_per_step=0.0005`) — unchanged.
- `epsilon_decay_environment_steps=50,000` — arm0's value.
- Hard target-network copy every 250 updates — arm0's value, **not** arm2a's
  soft/Polyak update (arm2a's pilot showed a systematic late-stage collapse
  in both its seeds: 87.5%→37.5% and 87.5%→62.5% between the 75K and 100K
  pilot checkpoints).
- Linear learning-rate decay `0.0005 → 0.0001` — arm2b's fix, the only Stage
  8 pilot arm with a clear, evidence-backed improvement (91.4% mean success
  at the 100K pilot checkpoint across 8 seeds, vs 84.4%/81.2%/50.0% for
  arm0/arm1/arm2a). The decay window is extended from arm2b's pilot value
  (100,000, matching its own step budget) to span the full 400,000-step gate
  budget — **this specific decay shape was not itself pilot-validated**; it
  is a proportional-scaling extrapolation, confirmed explicitly with the
  user before freezing this protocol.

## 2. Threshold structure — copied verbatim, not re-derived

Every `GATE_*` constant, `GATE_CHECKPOINTS`, and `LEARNING_CURVE_CHECKPOINTS`
value is copied unchanged from `thesis.pilots.stage7c_q1_config`. The
decision function itself (`thesis.pilots.stage7c_q1_gate.evaluate_competence_gate`)
is reused unmodified, called with this stage's own 20-seed list. This gate's
PASS/FAIL decision is therefore directly comparable to Stage 7C-Q1's FAIL
under the identical rule set — the entire point of re-running the gate
rather than inventing a new one.

```
GATE_MEAN_SUCCESS_MIN = 0.95
GATE_COLLISION_MAX = 0.02
GATE_TRUNCATION_MAX = 0.03
GATE_SWAP_ELIGIBILITY_MIN = 0.75
GATE_SEED_SUCCESS_MIN = 61/64 (0.953125)
GATE_MIN_QUALIFIED_SEEDS = 16
GATE_ADJACENT_SUCCESS_DROP_MAX = 0.03
GATE_MATERIAL_REGRESSION = 0.20
GATE_MAX_MATERIAL_REGRESSION_SEEDS = 1
GATE_MAX_LATE_COLLAPSE_SEEDS = 1
```

## 3. Seeds

Master seeds: `65021`–`65040` (20 total). Forbidden: every historical block
(`61001`-`61010`, `62001`-`62020`, `63001`-`63020`, `64001`-`64020`) plus the
entire Stage 8 pilot block (`65001`-`65020`, fully consumed by
arm0/arm1/arm2a/arm2b).

## 4. Training budget and checkpoints

400,000 steps, checkpoints every 25,000 steps (17 total: `0, 25000, ...,
400000`). `GATE_CHECKPOINTS = (350000, 375000, 400000)` feed the pass/fail
decision directly; `LEARNING_CURVE_CHECKPOINTS = (200000, ..., 400000)` feed
the adjacent-drop / material-regression / late-collapse checks.

## 5. Evaluation — two evaluators, split by checkpoint

Confirmed with the user on 2026-08-02: full per-step diagnostic logging
(Q-values, front_gap, minimum_TTC, action sequence) is expensive at this
scale (20 seeds × up to 64 episodes/checkpoint × 17 checkpoints), so it is
restricted to the 4 checkpoints that actually matter for either the
decision or as an initial baseline:

- **`RICH_LOG_CHECKPOINTS = (0, 350000, 375000, 400000)`**: evaluated with
  `stage8_gate_eval.evaluate_checkpoint_stage8_gate` (per-step trajectory
  logging enabled).
- **All other 13 checkpoints**: evaluated with
  `stage7c_q1_eval.evaluate_checkpoint_stage7c`, reused **unmodified** from
  Stage 7C-Q1 (episode-level only — success/collision/truncation/swap
  eligibility, no per-step Q-value logging). This function already threads
  `active_time_cost_per_step` correctly and already switches between the
  8-block "early" plan (≤175,000) and 32-block "late" plan (>175,000) via
  its own config's `n_scenario_blocks`, numerically identical to this
  gate's for every checkpoint used here.

Both evaluators are called with `protocol_tag=stage8-gate-protocol-v1`,
keeping the SHA-256 evaluation-seed namespace disjoint from Stage 7C-Q1's.

## 6. Decision rule

Result status is `PASS` / `FAIL` / `INVALID` only
(`STAGE1_TO_STAGE7_EXPERIMENT_FREEZE.md` §10.4). `FAIL` → the Stage 8 fix
did not resolve competence at formal scale; report reverts to
competence-limited, no further parameter tuning on this protocol. `PASS` →
proceed to the final three-condition (baseline/mean_pbrs/min_pbrs)
confirmatory experiment (`paper/STAGE8_PLAN_DRAFT.md` §6) — but only after
that section's own prerequisites (non-inferiority margins, between-condition
power analysis, Chapter 2 λ/algorithm alignment) are separately resolved;
`PASS` on this gate alone does not authorize starting that experiment.

## 7. Pre-registered risk note (written before training, not after)

Based on arm2b's 8-seed pilot (tail-risk regression rate ≈12.5%, 1/8 seeds
showing a >10-point success-rate drop between the 75K and 100K pilot
checkpoints), if that rate holds at n=20 there is a **≈73.3% probability**
the `GATE_MAX_MATERIAL_REGRESSION_SEEDS ≤ 1` criterion alone still fails
(binomial: P(X>1 | n=20, p=0.125)). This does not mean the gate should not
be run — arm2b remains the best-evidenced configuration tested, and whether
400,000 steps (vs the pilot's 100,000) changes this tail-risk rate is
exactly what this gate is for — but a `FAIL` driven specifically by this
criterion should not be treated as a surprise requiring a new explanation;
it was anticipated here, before training started.

## 8. Governance

Consistent with `paper/STAGE8_PLAN_DRAFT.md`: this gate is a new,
independently frozen protocol, not a retroactive modification of Stage
7C-Q1 or any Stage 8 pilot arm. No parameter in this protocol may be
changed after gate results are observed.
