# V1 Delta-Min Rawlsian Implementation Report

Status: **proposal-aligned implementation + smoke tests only (seed 1).** Full
experiments have **not** been run. Smoke results below are sanity checks, not
evidence of effectiveness.

---

## 1. Why raw `min_i E_i` was replaced

The earlier Rawlsian reward returned `objective_scale * min_i E_i` — the raw
*level* of the least-advantaged agent's experience each step. This was a useful
diagnostic for getting the pipeline working, but it is **not** what the revised
research proposal specifies, and it had two problems:

- Its magnitude (tens per step, hundreds per episode) dominated the shared
  task/safety constants, forcing fragile scale calibration.
- It rewards the *level* of `min E`, not its *improvement*, so it does not
  directly express "did the worst-off agent get better off this step?"

The proposal defines Rawlsian shaping as a **change-based** (difference) signal
on the least-advantaged agent's experience.

---

## 2. The exact new formula

Per step `t`:

```
E_min_t       = min_i E_i,t
delta_E_min_t = E_min_t - E_min_{t-1}

r_rawls_t = +lambda_R   if delta_E_min_t >  epsilon_R
            -lambda_R   if delta_E_min_t < -epsilon_R
             0.0        otherwise

R_rawls(per-step) = base_individual_reward + r_rawls_t
```

where `base_individual_reward` is exactly the EgoisticReward per-step driving
reward (progress − collision − TTC risk − waiting). The training loop then adds
the shared terminal terms (identical to egoistic):

```
reward  = reward_fn.compute(...)                 # base (+ delta-min for rawlsian)
reward += reward_fn.terminal_adjustment(...)     # +merge bonus / -non-merge penalty
reward += reward_fn.terminal_collision_adjustment(...)   # -collision penalty
```

Final learning rewards:

```
R_final_egoistic = base_individual_reward
                   + merge_task_adjustment
                   - terminal_collision_penalty

R_final_rawlsian = base_individual_reward + r_rawls_t
                   + merge_task_adjustment
                   - terminal_collision_penalty
```

So the **only** research difference between conditions is the additive
`r_rawls_t` shaping term.

**Documented approximation.** `compute(...)` only receives the post-step
`env_state` and pre-step `prev_env_state`. `E_min_t` is evaluated on
`(env_state, prev_env_state)`; `E_min_{t-1}` is approximated by evaluating the
experience function on `prev_env_state` with `None` as its own previous
reference (its mobility term is 0 for that evaluation). At episode start
(`prev_env_state is None`) and on empty/unavailable experiences, the signal
fails safe to `0.0`.

---

## 3. How this aligns with the research proposal

The proposal frames Rawlsian fairness as *improving the worst-off road user's
experience over time*. The discrete `±lambda_R / 0` signal directly rewards a
positive change in `min_i E_i` and penalises a negative one, with a dead-band
`epsilon_R` to ignore numerical noise. This is the change-based shaping the
proposal specifies, layered on top of an identical individual driving reward and
identical task/safety constraints so the comparison isolates the fairness
signal.

---

## 4. What files changed

| File | Change |
|---|---|
| `v1/rewards/rawlsian_reward.py` | Rewritten: composes an `EgoisticReward` for the base individual reward; adds `compute_rawlsian_signal(...)`; `compute(...)` returns `base + r_rawls_t`; stores delta-min diagnostics. |
| `v1/training/train.py` | `RunConfig` gains `rawlsian_lambda`, `rawlsian_epsilon`, `deprecated_rawlsian_objective_scale`; new CLI flags + deprecated alias mapping; `build_reward_function` passes them and `ego_agent`; per-episode delta-min diagnostics; results CSV echoes the new params. |
| `v1/diagnostics/run_diagnostics.py` | Tracks/aggregates Rawlsian signal counts; calibration block reports `lambda_R`/`epsilon_R`. |
| `docs/V1_DELTA_MIN_RAWLSIAN_IMPLEMENTATION_REPORT.md` | This report. |
| `docs/V1_DECISION_LOG.md`, `docs/V1_AI_USAGE_LOG.md` | Appended entries. |

---

## 5. What did NOT change

- `v1/env/` (environment dynamics, action semantics) — untouched.
- `v1/policies/` and DQN update logic — untouched.
- `v1/experience/experience.py` mathematics — untouched (`E_i`,
  `rawlsian_objective`, `least_advantaged_agent` all reused, not modified).
- `v1/rewards/egoistic_reward.py` per-step logic, `merge_task_reward.py`
  semantics, `base_reward.py` terminal adjustments — unchanged.
- `V1_SYSTEM_SPEC.md`, `prototype/` — untouched.
- No new RL algorithm; no full experiments run.

---

## 6. How `lambda_R` and `epsilon_R` are configured

- `RunConfig.rawlsian_lambda` (default `1.0`), `RunConfig.rawlsian_epsilon`
  (default `1e-6`).
- CLI: `--rawlsian-lambda`, `--rawlsian-epsilon`.
- Deprecated alias: `--rawlsian-objective-scale` still accepted; if provided it
  is mapped to `--rawlsian-lambda` with a printed warning, and the original
  value is logged as `deprecated_rawlsian_objective_scale`. Raw `min_i E_i`
  scaling is no longer used.
- All three values are written to the run config JSON and to every
  `results.csv` row for audit.
- Egoistic mode ignores both `lambda_R` and `epsilon_R`.

---

## 7. How task completion & collision constraints are still shared

The shared merge-task adjustment (`+merge_success_bonus` / `-non_merge_failure_penalty`)
and the shared terminal collision penalty (`-terminal_collision_penalty`) are
defined once on the base `RewardFunction` (`terminal_adjustment`,
`terminal_collision_adjustment`), built from the same `MergeTaskConfig`, and
applied identically by the training loop for both conditions. The Rawlsian base
individual reward is the same `EgoisticReward` instance behaviour, so both
conditions share: base driving reward, merge bonus/penalty, and collision
penalty. Only `r_rawls_t` is added for Rawlsian.

---

## 8. What diagnostics were added

Per training episode (`experiments/logs/<run_id>.csv`), diagnostic-only:
`mean_rawlsian_signal`, `rawlsian_positive_signal_count`,
`rawlsian_negative_signal_count`, `rawlsian_zero_signal_count`,
`mean_delta_min_experience`, `min_delta_min_experience`,
`max_delta_min_experience`. For egoistic mode these are blank/0 (the egoistic
reward exposes no shaping fields).

Diagnostics harness (`run_diagnostics.py`) aggregates
`rawlsian_{positive,negative,zero}_signal_count` and prints the calibration block.

These are **diagnostic only**. Primary comparison metrics remain:
`eval_safe_merge_success_rate`, `eval_collision_rate`,
`eval_non_merge_failure_rate`, `eval_min_experience`, `eval_mean_experience`,
`eval_gini_experience`, `eval_mean_time_to_merge_success_only`.
`eval_episode_reward` must **not** be compared across modes (different reward
composition).

---

## 9. Smoke-test results (seed 1, 20 episodes, max_steps 100)

Constants: `terminal_collision_penalty=5`, `merge_success_bonus=2`,
`non_merge_failure_penalty=3`.

| metric | egoistic | rawlsian (λ=1.0, ε=1e-6) |
|---|---|---|
| eval_safe_merge_success_rate | 1.0 | 1.0 |
| eval_unsafe_merge_rate | 0.0 | 0.0 |
| eval_collision_rate | 0.0 | 0.0 |
| eval_non_merge_failure_rate | 0.0 | 0.0 |
| eval_min_experience | 9.00 | 8.81 |
| eval_gini_experience | 0.041 | 0.038 |
| eval_mean_experience | 9.49 | 9.33 |

- The deprecated alias was verified: `--rawlsian-objective-scale 0.5` prints the
  deprecation warning, sets `rawlsian_lambda=0.5`, and logs
  `deprecated_rawlsian_objective_scale=0.5`.
- The shaping signal is active: in a short rawlsian run the per-episode log
  showed e.g. 13 positive signals, 0 negative, 0 zero, `mean_rawlsian_signal=λ`,
  confirming `delta_E_min` is computed and thresholded.
- `python -m v1.diagnostics.run_diagnostics` runs clean and prints the
  `lambda_R`/`epsilon_R` calibration block.

These are pipeline sanity checks on a single seed only.

---

## 10. Is it safe to proceed to a calibration pilot?

**Yes — to a calibration pilot, not to full experiments.** The implementation is
proposal-aligned, both conditions run through the identical loop with shared
task/safety terms, the shaping signal fires correctly, and protected modules are
untouched. The smoke run shows both conditions can merge safely at the test
constants, with no crashes.

Recommended next step: a short **multi-seed calibration pilot** (e.g. seeds 1–5)
sweeping `lambda_R` (e.g. 0.5, 1.0, 2.0) at fixed `epsilon_R` and fixed shared
task/safety constants, reading the primary metrics (`eval_safe_merge_success_rate`,
`eval_collision_rate`, `eval_min_experience`, `eval_gini_experience`). Do not draw
conclusions from seed 1 alone.
