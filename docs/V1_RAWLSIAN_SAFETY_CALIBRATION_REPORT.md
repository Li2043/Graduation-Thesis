# V1 Rawlsian Safety & Calibration Report

Status: **pilot-debug phase (single seed = 1).** Not a full experiment. Findings
below are calibration signals only and must be confirmed with multi-seed runs
before any scientific claim.

---

## 1. Files changed

Only allowed modules were modified; no protected module changed.

| File | Change |
|---|---|
| `v1/rewards/merge_task_reward.py` | Added `terminal_collision_penalty` to `MergeTaskConfig`; added `terminal_collision_adjustment(...)`; added `classify_outcome(...)` (safe/unsafe merge split); refactored `outcome_from_state` to use it. |
| `v1/rewards/base_reward.py` | Added shared `terminal_collision_adjustment(self, ego_state)` method (inherited identically by both conditions). |
| `v1/rewards/rawlsian_reward.py` | Added `objective_scale` parameter; `compute` now returns `objective_scale * min_i E_i`. `rawlsian_objective` and `E_i` unchanged. |
| `v1/rewards/egoistic_reward.py` | Docstring only: documents that egoistic keeps its per-step collision aversion **and** receives the shared terminal collision constraint (deliberate, not double-counting). Per-step `compute` unchanged. |
| `v1/training/train.py` | Applies shared collision penalty after `env.step`; new safe/unsafe outcome metrics + eval aggregates; `RunConfig` gains `terminal_collision_penalty` and `rawlsian_objective_scale`; new CLI flags; calibration params echoed to config JSON + results CSV; schema-safe results writer. |
| `v1/diagnostics/run_diagnostics.py` | Mirrors production reward (adds shared collision penalty); reports safe/unsafe/collision-without-merge rates and terminal bonus/non-merge/collision penalty counts; prints calibration parameters used. |

Protected modules verified unchanged: environment dynamics, action semantics,
policy architecture, DQN update logic, `experience.py` mathematics,
`V1_SYSTEM_SPEC.md`, `prototype/`.

---

## 2. New reward formula

Per-step reward comes from `reward_fn.compute(...)`; the training/evaluation loop
then adds the **shared** terminal task and safety terms (identical function and
identical config for both conditions):

**Egoistic**
```
R_ego = progress_reward
        - collision_penalty            (per-step, individual aversion)
        - TTC_risk_penalty
        - waiting_penalty
        + merge_task_adjustment        (+bonus on merge / -penalty on non-merge)
        - terminal_collision_penalty   (shared, if crashed)
```

**Rawlsian**
```
R_rawls = rawlsian_objective_scale * min_i E_i
        + merge_task_adjustment        (shared, same as egoistic)
        - terminal_collision_penalty   (shared, same as egoistic)
```

Defaults (configurable, all logged): `merge_success_bonus = 1.0`,
`non_merge_failure_penalty = 1.0`, `terminal_collision_penalty = 10.0`,
`rawlsian_objective_scale = 1.0`. The egoistic mode ignores
`rawlsian_objective_scale`.

---

## 3. New outcome metrics

Per episode (mutually exclusive `termination_reason`: `safe_merge`,
`unsafe_merge`, `collision_without_merge`, `max_steps_unmerged`, `unknown`):

`merge_completed`, `collision`, `safe_merge`, `unsafe_merge`,
`collision_without_merge`, `non_merge_failure`, `time_to_merge` (actual step or
blank — never `max_steps`), `episode_length`, plus terminal flags
`merge_bonus_applied`, `non_merge_penalty_applied`, `collision_penalty_applied`.

Evaluation aggregates (in `results.csv`):
`eval_safe_merge_success_rate` (**primary success**), `eval_unsafe_merge_rate`,
`eval_collision_without_merge_rate`, `eval_non_merge_failure_rate`,
`eval_collision_rate`, `eval_merge_success_rate` (kept, **not** primary),
`eval_mean_time_to_merge_success_only`, `eval_episode_length_mean`, fairness
(`eval_min_experience`, `eval_gini_experience`, `eval_mean_experience`), and the
echoed calibration parameters.

---

## 4. Is safe vs. unsafe merge now distinguished?

**Yes.** `merge_success_rate` alone was misleading because an episode can both
merge and collide on the merge step. `safe_merge = merge ∧ ¬collision` and
`unsafe_merge = merge ∧ collision` are now separate columns, and
`eval_safe_merge_success_rate` is the designated primary success metric.

---

## 5. Is the collision penalty shared across both modes?

**Yes.** It is a single function (`terminal_collision_adjustment`) on the base
`RewardFunction`, built from the same `MergeTaskConfig`, inherited unchanged by
both `EgoisticReward` and `RawlsianReward`, and applied in the loop identically.
It lives in the reward/task layer — not in the environment and not in the
experience function.

---

## 6. Is the Rawlsian scale configurable and logged?

**Yes.** `--rawlsian-objective-scale` (default 1.0) sets `RunConfig.rawlsian_objective_scale`,
which scales only `min_i E_i` in Rawlsian mode (egoistic ignores it). The value
is written to the config JSON and to every `results.csv` row, alongside
`terminal_collision_penalty`, `merge_success_bonus`, and `non_merge_failure_penalty`.

---

## 7. Calibration pilot table (seed = 1, episodes = 100, max_steps = 100)

Shared constants: `terminal_collision_penalty = 10`, `merge_success_bonus = 1`,
`non_merge_failure_penalty = 1`.

| mode | seed | scale | coll_pen | safe_merge | unsafe_merge | collision | non_merge_fail | min_exp | gini | mean_exp |
|---|---|---|---|---|---|---|---|---|---|---|
| egoistic | 1 | (n/a) | 10 | **0.0** | 0.0 | 0.0 | 1.0 | −11.58 | 0.50 | −6.52 |
| rawlsian | 1 | 1.00 | 10 | **1.0** | 0.0 | 0.0 | 0.0 | 5.20 | 0.50 | 7.01 |
| rawlsian | 1 | 0.10 | 10 | 0.7 | 0.3 | 0.3 | 0.0 | 8.31 | 0.14 | 10.19 |
| rawlsian | 1 | 0.05 | 10 | **1.0** | 0.0 | 0.0 | 0.0 | 3.70 | 0.30 | 5.04 |
| rawlsian | 1 | 0.01 | 10 | 0.0 | 0.0 | 0.0 | 1.0 | −4.29 | 0.04 | −3.11 |

Observations (single seed — do not over-interpret):

- **The previously-reported Rawlsian unsafe-merge failure is gone at scale 1.0
  and 0.05** (collision 0.0, safe-merge 1.0). The shared terminal collision
  penalty appears to be the effective lever against unsafe merging.
- The sweep is **non-monotonic** (0.1 worse than both 0.05 and 1.0). This is a
  strong indication of single-seed variance, not a real ordering.
- At the extreme scale 0.01 the maximin signal is suppressed and the agent
  collapses to non-merge (the sparse ±1 task terms alone did not drive merging).
- **The egoistic baseline regressed to pure non-merge** at these constants
  (safe-merge 0.0, non-merge 1.0). With `collision_penalty = 10` vs
  `merge_bonus = non_merge_penalty = 1`, the asymmetry makes "never merge"
  (terminal −1) cheaper than risking an unsafe merge (−10 + 1) for an objective
  whose only forward pull is a small per-step progress term. The Rawlsian
  objective, by contrast, intrinsically rewards completing the merge because it
  raises the worst-off agent's experience. This asymmetry is itself an
  interesting candidate finding but **must not** be claimed from one seed, and a
  degenerate egoistic baseline is not a fair comparator.

---

## 8. Recommended calibration setting for the next pilot

- **Rawlsian:** `rawlsian_objective_scale = 0.05` (or `1.0`) with
  `terminal_collision_penalty = 10`, `merge_success_bonus = 1`,
  `non_merge_failure_penalty = 1`. Both gave safe-merge 1.0 / collision 0.0 on
  seed 1; `0.05` also brings the maximin magnitude closer to the task/safety
  scale, which is desirable for a balanced objective.
- **Egoistic (needs a fix before comparison):** raise `non_merge_failure_penalty`
  (e.g. 2–5) and/or lower `terminal_collision_penalty` (e.g. 5) so the egoistic
  baseline also learns to merge. The constants are now CLI-exposed, so this is a
  pure recalibration with no code change. Whatever values are chosen must be
  applied **identically** to both conditions and logged.

Next step should be a **multi-seed** mini-pilot (e.g. seeds 1–5) at the chosen
constants for both modes, reading `eval_safe_merge_success_rate` and
`eval_collision_rate` as the gate.

---

## 9. Is it safe to proceed to full experiments?

**Not yet.** The gating standard (Rawlsian collision acceptably low **and**
safe-merge non-trivial) is met on seed 1 at scales 1.0/0.05, which is
encouraging. However:

1. All results are **single-seed**; the non-monotonic sweep shows real variance.
2. The **egoistic baseline currently degenerates to non-merge** at the shared
   constants, so the two conditions are not yet a fair comparison.

**Recommendation: run one more calibration mini-pilot** — pick constants that
make *both* egoistic and Rawlsian merge safely, then confirm across several
seeds (1–5). Only if both conditions show acceptable collision rates and
non-trivial safe-merge rates across seeds should full multi-seed experiments
begin.

---

## Reproduce

```bash
# Baselines
python -m v1.training.train --mode egoistic --seed 1 --episodes 100 --max-steps 100 \
  --terminal-collision-penalty 10 --merge-success-bonus 1 --non-merge-failure-penalty 1
python -m v1.training.train --mode rawlsian --seed 1 --episodes 100 --max-steps 100 \
  --terminal-collision-penalty 10 --merge-success-bonus 1 --non-merge-failure-penalty 1 \
  --rawlsian-objective-scale 1.0

# Scale sweep (rawlsian): 0.1, 0.05, 0.01 (same task/safety constants)
python -m v1.training.train --mode rawlsian --seed 1 --episodes 100 --max-steps 100 \
  --terminal-collision-penalty 10 --merge-success-bonus 1 --non-merge-failure-penalty 1 \
  --rawlsian-objective-scale 0.05

python -m v1.diagnostics.run_diagnostics
```

All runs append to `experiments/results.csv` with calibration parameters in each row.
