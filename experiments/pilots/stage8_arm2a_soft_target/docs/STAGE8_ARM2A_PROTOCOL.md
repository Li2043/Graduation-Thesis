# STAGE 8 ARM2A PROTOCOL — Soft Target-Update Fix Targeting Training Instability

## Status

Frozen diagnostic pilot protocol. Tag: `stage8-arm2a-protocol-v1`. Parameters
confirmed with the user on 2026-08-01 (see `paper/STAGE8_PLAN_DRAFT.md` §4).

## 1. Purpose

Arm0 and arm1 both showed large, non-monotonic seed-to-seed swings in
success_rate / downstream_failure across checkpoints:

- arm0 seed 65001: success 56%→12.5%→37.5%→100% across 25K/50K/75K/100K.
- arm1 seed 65004: success 18.75%→6.25%→18.75%→68.75% across the same points.

This is consistent with the classic "hard periodic target-network copy
causes late-training instability" mechanism already flagged in
`STAGE1_TO_STAGE7_EXPERIMENT_FREEZE.md` §10.3 (Double DQN's late-collapse
seed count increased relative to Vanilla DQN). Arm2a tests this directly by
replacing the hard copy (every 250 updates) with a Polyak-averaged soft
update applied every learner update.

Arm1's own result (frozen_stall count at 100K checkpoint: 2→0, but overall
success_rate essentially unchanged at 0.813 vs arm0's 0.844, and seed 65004
still showed a deep mid-training trough worse than anything in arm0) argued
against exploration coverage being the dominant lever and strengthened the
case for testing training-stability mechanisms directly -- this is why arm2a
exists as a follow-up to arm1, not a replacement for it.

## 2. The single variable changed relative to arm0

`target_update_mode`: **"hard" (arm0/arm1) -> "soft" (arm2a)**, with
`target_soft_tau = 0.005` (effective averaging window ≈200 updates, chosen
as the closest smooth analogue to arm0's discrete every-250-updates hard
copy -- see `paper/STAGE8_PLAN_DRAFT.md` §4 for the τ discussion).

Everything else is unchanged from arm0: epsilon schedule
(`epsilon_decay_environment_steps=50,000`, same as arm0, NOT arm1's 75,000),
learning rate (constant 0.0005, same as arm0/arm1), reward, algorithm, and
checkpoint/evaluation protocol.

This is implemented via new opt-in fields on the shared
`FormalDQNConfig`/`IndependentDQNLearner` infrastructure
(`target_update_mode`, `target_soft_tau`,
`IndependentDQNLearner.soft_sync_target()`), added specifically for Stage 8
arm2a with defaults that preserve arm0/arm1's exact prior behaviour
(`target_update_mode="hard"` by default) -- confirmed via smoke-test
regression checks on both arm0 and arm1 after the shared-file change.

## 3. Seeds

Master seeds: `65005`, `65006` (within the reserved `65001`-`65020` block).
Forbidden (all historical stages plus arm0/arm1): `61001`-`61010`,
`62001`-`62020`, `63001`-`63020`, `64001`-`64020`, `65001`-`65004`.

## 4. Checkpoints and evaluation

Identical to arm0/arm1: checkpoints `0, 25000, 50000, 75000, 100000`,
8-block / 16-episode "early" evaluation plan, per-step trajectory logging
for every evaluation episode at every checkpoint. Evaluation reuses
`stage8_arm0_eval.py::evaluate_checkpoint_stage8_arm0` unchanged (only the
`protocol_tag` argument changes, to `stage8-arm2a-protocol-v1`) -- evaluation
is always greedy (epsilon=0), so it does not depend on arm2a's training-time
target-update mode.

## 5. Success criterion (for this document only)

Descriptive, not a PASS/FAIL machine gate. Two metrics, both reported in
`analysis/stage8_arm2a/v1/{comparison_frozen_stall_vs_arm0_arm1.csv,
comparison_success_volatility.csv, STAGE8_ARM2A_NOTE.md}`:

1. Per-seed success_rate volatility (max − min across the 5 checkpoints),
   compared against arm0's and arm1's -- this is arm2a's actual target
   mechanism (training-instability reduction), not a specific failure mode.
2. `frozen_stall` count at each checkpoint, for continuity with arm1's
   reporting, though arm2a does not specifically target this sub-mode.

Given n=2 seeds per arm, this is directional evidence, not a statistically
powered test.

## 6. Relationship to arm2b

Arm2a (soft target update) and arm2b (learning-rate decay) are deliberately
run as **separate single-variable arms**, not bundled into one, per the
project's single-variable-attribution discipline (`STAGE8_PLAN_DRAFT.md`
§4: "逐项测试，不要一次性打包") -- confirmed with the user on 2026-08-01
specifically because this reasoning needs to be defensible in the thesis's
experimental-design chapter. If both arms show reduced volatility, a
combined arm (both changes together) would be a legitimate *subsequent*,
separately-frozen arm -- not a retroactive merge of arm2a and arm2b.

## 7. Governance

Consistent with `paper/STAGE8_PLAN_DRAFT.md`: arm2a is a new, independently
frozen pilot arm. No parameter in this protocol may be changed after arm2a
results are observed.
