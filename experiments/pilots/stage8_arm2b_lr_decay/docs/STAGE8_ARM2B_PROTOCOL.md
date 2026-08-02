# STAGE 8 ARM2B PROTOCOL — Learning-Rate Decay Fix Targeting Training Instability

## Status

Frozen diagnostic pilot protocol. Tag: `stage8-arm2b-protocol-v1`. Parameters
confirmed with the user on 2026-08-01 (see `paper/STAGE8_PLAN_DRAFT.md` §4).

## 1. Purpose

Same motivation as arm2a (see `stage8_arm2a_soft_target/docs/STAGE8_ARM2A_PROTOCOL.md`
§1): arm0 and arm1 both showed large, non-monotonic seed-to-seed swings in
success_rate / downstream_failure across checkpoints, consistent with
training-instability mechanisms rather than a reward or exploration-coverage
problem. Arm2b tests the complementary hypothesis to arm2a's soft target
update: that a fixed 0.0005 learning rate causes overly large gradient steps
late in training, when the Q-function should be fine-tuning rather than
still taking large steps.

## 2. The single variable changed relative to arm0

Linear learning-rate decay: `learning_rate_end = 0.0001`,
`learning_rate_decay_environment_steps = 100,000` (decay spans the entire
training budget, never reaching 0 -- the training never fully stops
learning). Arm0's constant `learning_rate = 0.0005` is the schedule's
starting value.

Everything else is unchanged from arm0: epsilon schedule
(`epsilon_decay_environment_steps=50,000`, same as arm0), target-update mode
(hard copy every 250 updates, same as arm0 -- NOT arm2a's soft update),
reward, algorithm, and checkpoint/evaluation protocol.

This is implemented via new opt-in fields on the shared
`FormalDQNConfig`/`IndependentDQNLearner` infrastructure
(`learning_rate_end`, `learning_rate_decay_environment_steps`,
`formal_config.lr_at_step()`, `IndependentDQNLearner.set_learning_rate()`),
added specifically for Stage 8 arm2b with defaults that preserve
arm0/arm1/arm2a's exact prior behaviour (`learning_rate_end=None` means
constant LR by default) -- confirmed via smoke-test regression checks on
arm0 and arm1 after the shared-file change.

## 2a. 2026-08-02 robustness extension

The original 2-seed pilot (65007, 65008) reached 100% success / 0% collision
/ 0 `frozen_stall` at the 100K checkpoint for BOTH seeds, with zero residual
`downstream_failure` at 75K or 100K -- the cleanest result of any Stage 8 arm
so far (arm0: 49 downstream_failure episodes total; arm1: 44; arm2a: 37,
including a late-stage collapse at 100K for both its seeds). This is
promising but not trustworthy at n=2 given arm2a's opposite pattern, so 6
additional seeds (`65009`-`65014`) were added to `PILOT_SEEDS` under this
**same frozen protocol** -- no reward, algorithm, epsilon, or learning-rate
parameter changes. This is a scale-up of an already-frozen configuration,
not a new arm.

## 3. Seeds

Master seeds: `65007`-`65014` (8 total: `65007`/`65008` original pilot,
`65009`-`65014` the 2026-08-02 robustness extension; all within the reserved
`65001`-`65020` block). Forbidden (all historical stages plus arm0/arm1/arm2a): `61001`-`61010`,
`62001`-`62020`, `63001`-`63020`, `64001`-`64020`, `65001`-`65006`.

## 4. Checkpoints and evaluation

Identical to arm0/arm1/arm2a: checkpoints `0, 25000, 50000, 75000, 100000`,
8-block / 16-episode "early" evaluation plan, per-step trajectory logging
for every evaluation episode at every checkpoint. Evaluation reuses
`stage8_arm0_eval.py::evaluate_checkpoint_stage8_arm0` unchanged (only the
`protocol_tag` argument changes, to `stage8-arm2b-protocol-v1`) -- evaluation
is always greedy (epsilon=0), so it does not depend on arm2b's training-time
learning-rate schedule.

## 5. Success criterion (for this document only)

Descriptive, not a PASS/FAIL machine gate. Same two metrics as arm2a
(per-seed success_rate volatility compared against arm0/arm1, and
`frozen_stall` count for continuity), reported in
`analysis/stage8_arm2b/v1/{comparison_frozen_stall_vs_arm0_arm1.csv,
comparison_success_volatility.csv, STAGE8_ARM2B_NOTE.md}`. Given n=2 seeds
per arm, this is directional evidence, not a statistically powered test.

## 6. Relationship to arm2a

Arm2a (soft target update) and arm2b (learning-rate decay) are deliberately
run as **separate single-variable arms**, not bundled into one, per the
project's single-variable-attribution discipline -- confirmed with the user
on 2026-08-01 specifically because this reasoning needs to be defensible in
the thesis's experimental-design chapter. If both arms show reduced
volatility, a combined arm (both changes together) would be a legitimate
*subsequent*, separately-frozen arm -- not a retroactive merge of arm2a and
arm2b.

## 7. Governance

Consistent with `paper/STAGE8_PLAN_DRAFT.md`: arm2b is a new, independently
frozen pilot arm. No parameter in this protocol may be changed after arm2b
results are observed.
