# STAGE 8 ARM1 PROTOCOL — Exploration-Schedule Fix Targeting `frozen_stall`

## Status

Frozen diagnostic pilot protocol. Tag: `stage8-arm1-protocol-v1`. Parameters
confirmed with the user on 2026-08-01 (see `paper/STAGE8_PLAN_DRAFT.md` §4).

## 1. Purpose

Stage 8 arm0 (`stage8-arm0-protocol-v1`) confirmed via per-step trajectory
evidence that the dominant residual failure mode is not a brake/accelerate
oscillation near a background vehicle (the original hypothesis) but two
distinct sub-mechanisms:

- **`frozen_stall`** (69% of arm0's 49 downstream_failure episodes): the
  survivor is already at speed=0 before the peer exits, and then never
  resumes for the remainder of the episode (0 action switches, 0 route
  progress, ~0 hard-braking cost) -- a self-reinforcing absorbing state.
- **`moving_with_switches`** (31%): the survivor is still moving, with real
  action switching (9-39 switches) and a near-flat Q-value margin between
  the top-2 actions (mean 0.003) -- a genuine value-function-flatness
  oscillation, unrelated to background-vehicle proximity (`front_gap` /
  `minimum_TTC` were unpopulated -- i.e. no finite reading -- in every one
  of the 49 downstream_failure episodes in arm0).

Arm0 also showed that the same reward/algorithm combination CAN converge to
a fully successful policy: seed 65001 reached 100% success (both failure
modes at 0) by the 100K checkpoint, while seed 65002 still had 25%
downstream_failure at 100K (2 `frozen_stall` + 2 `moving_with_switches`).
This points to training convergence reliability, not reward misspecification.

**Arm1 targets `frozen_stall` specifically** via a single exploration-schedule
change, on the hypothesis that the "isolated survivor, needs to resume from a
stop" state region is under-visited during training, especially in the back
half of arm0's 100K-step run where epsilon had already reached its floor.

Arm1 does **not** target `moving_with_switches` -- that mechanism is arm2's
target (training-stability bundle: soft target update + LR decay), not
arm1's, and is expected to still be present in arm1 at a similar rate.

## 2. The single variable changed relative to arm0

`EPSILON_DECAY_STEPS`: **50,000 (arm0) -> 75,000 (arm1)**.

Everything else in `FormalExplorationConfig` is unchanged: `epsilon_start`
(1.0), `epsilon_end` (0.10), `epsilon_after_decay` (0.10), `schedule`
(`"linear"`). Arm0's epsilon already floors at 0.10 rather than decaying to
0 -- this was confirmed by reading `formal_config.py::FormalExplorationConfig`
and `epsilon_at_step()` before freezing this arm (an earlier plan-draft
candidate, "set an epsilon floor instead of decaying to 0", was dropped
because that floor already existed in arm0; see `STAGE8_PLAN_DRAFT.md` §4).

No reward coefficient, no algorithm (still Double DQN), and no checkpoint /
evaluation protocol changes relative to arm0.

## 3. Seeds

Master seeds: `65003`, `65004` (within the reserved `65001`-`65020` block,
immediately following arm0's `65001`/`65002`). Forbidden (all historical
stages plus arm0 itself): `61001`-`61010`, `62001`-`62020`, `63001`-`63020`,
`64001`-`64020`, `65001`-`65002`.

## 4. Checkpoints and evaluation

Identical to arm0: checkpoints `0, 25000, 50000, 75000, 100000`, 8-block /
16-episode "early" evaluation plan, per-step trajectory logging enabled for
every evaluation episode at every checkpoint. Evaluation reuses
`stage8_arm0_eval.py::evaluate_checkpoint_stage8_arm0` unchanged (only the
`protocol_tag` argument changes, to `stage8-arm1-protocol-v1`, keeping the
SHA-256 evaluation-seed namespace disjoint from arm0's) -- evaluation is
always greedy (epsilon=0), so it does not depend on arm1's training-time
exploration-schedule change.

## 5. Success criterion (for this document only)

Arm1's only success criterion is descriptive, not a PASS/FAIL machine gate:
does the `frozen_stall` count at the 100K checkpoint drop relative to arm0's
two seeds (65001=0, 65002=2)? Reported in
`analysis/stage8_arm1/v1/{comparison_vs_arm0.csv, STAGE8_ARM1_NOTE.md}`.
Given n=2 seeds per arm, this is directional evidence for whether the
exploration-coverage hypothesis is worth carrying into a larger-scale
confirmatory run -- not a statistically powered test.

## 6. Governance

Consistent with `paper/STAGE8_PLAN_DRAFT.md`: arm1 is a new, independently
frozen pilot arm, not a retroactive modification of arm0. No parameter in
this protocol may be changed after arm1 results are observed. If arm1 does
not reduce `frozen_stall`, the next step is arm2 (training-stability bundle)
or a re-examination of the exploration-coverage hypothesis itself -- not a
silent parameter edit to arm1.
