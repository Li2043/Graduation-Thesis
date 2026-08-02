# STAGE 8 ARM0 PROTOCOL — Diagnostic Reproduction with Per-Step Trajectory Logging

## Status

Frozen diagnostic protocol. Tag: `stage8-arm0-protocol-v1`.

## 1. Purpose

Stage 8 arm0 is a **diagnostic reproduction**, not a competence gate and not a
reward/algorithm ablation.

It reproduces exactly:

- Double DQN;
- Base Reward V2 (active-time cost `0.0005` per active policy step);
- Baseline condition (no PBRS, no shaping);
- the same G1-I1 environment, comfort lock, and DQN hyperparameters as
  Stage 7C-Q1 (`experiments/pilots/stage7c_q1_baseline_competence/`).

It differs from Stage 7C-Q1 only in:

- **scale**: 2 new seeds (`65001`, `65002`) instead of 20, 100,000 joint
  environment steps instead of 400,000;
- **instrumentation**: per-step trajectory logging (Q-values, front gap,
  minimum time-to-collision, action sequence) is recorded for every
  evaluation episode at every checkpoint, from checkpoint 0 onward.

It does **not**:

1. change any reward coefficient, including the active-time cost;
2. change the algorithm (Double DQN target only, vanilla forbidden, same as
   Stage 7C-Q1);
3. test Mean-PBRS or Min-PBRS;
4. claim a PASS/FAIL competence verdict;
5. attempt to fix or improve on Stage 7C-Q1's result.

## 2. Motivation

Stage 7C-Q1 FAILED its competence gate
(`E22_stage7c_q1_baseline_competence/final_new_stage7c_q1/analysis/stage7c_q1/v1/STAGE7C_Q1_DECISION.md`).
Analysis of its episode-level `evaluation_episodes.csv` showed the dominant
residual failure mode, `downstream_failure` (~13% of episodes even at 400K,
~93% of all remaining truncation at 400K), is neither a time-budget problem
(the survivor has ~70s of uncontested solo driving time after its peer exits,
far more than the ~13s a successful episode needs) nor a scenario-specific
problem (the failure rate is roughly uniform, 7.5%-17.5%, across all 8
validation blocks). The survivor's post-peer-exit route progress is only
~20-26% of the route on average, versus ~87% for successful episodes, while
accumulating a substantial hard-braking reward penalty — consistent with a
prolonged brake/accelerate oscillation rather than either a clean stall or a
clean (if slow) approach to the exit.

Stage 7C-Q1 itself never recorded per-step trajectory data (confirmed with
the user — not a data-access problem, it was never logged), and its trained
checkpoints live only on a separate, inaccessible machine. Arm0 exists solely
to get a per-step trajectory dataset that can directly confirm or refute the
oscillation-near-a-background-vehicle hypothesis.

## 3. Seeds

Master seeds: `65001`, `65002`. Reserved block for future Stage 8 arms:
`65003`-`65020`. Forbidden (all historical stages): `61001`-`61010`,
`62001`-`62020`, `63001`-`63020`, `64001`-`64020`.

## 4. Checkpoints and evaluation

Checkpoints: `0, 25000, 50000, 75000, 100000`. All are ≤175,000, so
evaluation always uses the 8-validation-block / 16-episode "early" plan
(unchanged from Stage 7C-Q1). Evaluation seeds are derived via
`stable_eval_seed()` (SHA-256, never Python `hash()`), keyed on the
`stage8-arm0-protocol-v1` tag — disjoint from Stage 7C-Q1's evaluation seed
namespace by construction.

Per-step trajectory logging is enabled for every evaluation episode at every
checkpoint (not just a subset) — the run is small enough (2 seeds × 5
checkpoints × 16 episodes = 160 episodes total) that full instrumentation is
cheap.

## 5. Success criterion (for this document only)

Arm0's only success criterion is: does the recorded per-step trajectory
evidence for `downstream_failure` episodes corroborate or refute the
post-peer-exit oscillation hypothesis? There is no PASS/FAIL/INVALID
machine gate — the output is a descriptive note
(`analysis/stage8_arm0/v1/STAGE8_ARM0_NOTE.md`) plus a per-episode diagnostics
CSV, read and interpreted manually.

## 6. Governance

Consistent with `paper/STAGE8_PLAN_DRAFT.md`: Stage 8 arm0 is a new,
independently frozen stage, not a retroactive modification of Stage 7C-Q1.
No parameter in this protocol may be changed after arm0 results are observed;
if the oscillation hypothesis needs a structural fix, that is a separate,
subsequently frozen arm (arm1), not a silent edit to arm0.
