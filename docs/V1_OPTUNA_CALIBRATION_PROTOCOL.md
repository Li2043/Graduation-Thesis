# V1 Optuna Calibration Protocol

Status: **implemented** (`v1/calibration/optuna_calibration.py`). This document
defines how hyperparameter calibration is run before any final experiment.
Optuna is **calibration only** — it does not produce final research results.

---

## 1. Why Optuna is used

The 300-episode pilot (seeds 1–3) shows both conditions are trainable but
Rawlsian still has elevated collision rate (~0.17 vs ~0.00 for egoistic).
Task/safety parameters (`terminal_collision_penalty`, merge bonuses/penalties)
and Rawlsian shaping parameters (`rawlsian_lambda`, `rawlsian_epsilon`) interact
non-linearly. Manual grid search is slow and hard to audit. Optuna provides a
reproducible, logged search over a conservative search space while enforcing the
seed protocol from `docs/V1_SEED_PROTOCOL.md`.

---

## 2. Parameters calibrated

### Shared task/safety (both egoistic and Rawlsian)

| parameter | search range |
|---|---|
| `terminal_collision_penalty` | [3.0, 8.0] |
| `merge_success_bonus` | [1.0, 5.0] |
| `non_merge_failure_penalty` | [2.0, 6.0] |

### Rawlsian-only (ignored by egoistic mode)

| parameter | search range |
|---|---|
| `rawlsian_lambda` | [0.5, 1.5] |
| `rawlsian_epsilon` | [1e-6, 1e-2] (log) |

Reference pilot configuration (not hard-coded winner):

- `terminal_collision_penalty = 5`
- `merge_success_bonus = 2`
- `non_merge_failure_penalty = 3`
- `rawlsian_lambda = 1.0`
- `rawlsian_epsilon = 1e-6`

Each trial samples **one** shared task/safety triple and Rawlsian pair, then
runs **both** modes on all calibration seeds with identical run seeds.

---

## 3. Seed sets

| role | seeds | used when |
|---|---|---|
| calibration | **1, 2, 3** | inside Optuna objective (search) |
| validation | **4, 5** | once, after study, on top configs only |
| final evaluation | **100, 101, 102, 103, 104** | **locked** — never used by this script |

Held-out **evaluation** during each run uses the default `EVAL_SEEDS`
(9001–9010) from `train.py` unless overridden. These are disjoint from run
seeds and measure policy quality; they are not the final-evaluation run seeds.

Optuna sampler seed: **42** (fixed, reproducible search).

---

## 4. Why `eval_episode_reward` is not optimized

Egoistic and Rawlsian use different reward compositions (delta-min shaping adds
discrete ±λ signals on top of the shared base reward). Comparing
`eval_episode_reward` across modes would confound objective design with task
outcomes. Calibration optimizes **outcome and experience metrics** that are
comparable across modes: safe merge success, collision rate, non-merge failure,
min/mean experience, and Gini.

---

## 5. Objective score formula

This is a **calibration score only**, not a final research metric.

Primary hard constraints (both modes must pass):

- mean `eval_safe_merge_success_rate` ≥ 0.6
- mean `eval_collision_rate` ≤ 0.3
- mean `eval_non_merge_failure_rate` ≤ 0.3

If either mode violates constraints, the trial receives a strong penalty
(`-1_000_000`).

Within valid trials (`direction="maximize"`):

```
score =
  + 2.0 * rawlsian_safe_merge_success_rate
  - 3.0 * rawlsian_collision_rate
  - 2.0 * rawlsian_non_merge_failure_rate
  + 0.2 * rawlsian_min_experience
  + 0.1 * rawlsian_mean_experience
  - 1.0 * rawlsian_gini_experience
  + 0.5 * max(0, rawlsian_min_experience - egoistic_min_experience)
  - 2.0 * max(0, 0.6 - egoistic_safe_merge_success_rate)
  - 2.0 * max(0, egoistic_collision_rate - 0.3)
  - 2.0 * max(0, egoistic_non_merge_failure_rate - 0.3)
```

All Rawlsian/Egoistic terms are **means across calibration seeds** for that
trial.

---

## 6. Hard constraints

Same thresholds for both modes (from `seed_protocol.HARD_CONSTRAINTS`):

| constraint | threshold |
|---|---|
| mean safe merge success | ≥ 0.6 |
| mean collision rate | ≤ 0.3 |
| mean non-merge failure | ≤ 0.3 |

Worst-seed robustness (≥ 0.3 safe merge) is tracked in seed protocol docs but
is not a hard gate in the Optuna script.

---

## 7. Freezing selected configs before final evaluation

1. Run full Optuna study on calibration seeds `[1, 2, 3]`.
2. Select top 3 constraint-passing trials by calibration score.
3. Run those configs on validation seeds `[4, 5]` (report only — no further tuning).
4. Human review of validation + calibration artifacts under `experiments/optuna/`.
5. **Freeze** one configuration in `docs/V1_DECISION_LOG.md` with exact parameter
   values and the date frozen.
6. Only then run final experiments on locked seeds `[100..104]` with
   `--seed-phase final`.

Final evaluation seeds must not appear in any calibration or validation run.

---

## 8. Warning

Optuna output is for **hyperparameter calibration**, not for claiming Rawlsian
superiority or publishing final numbers. Final claims require the locked
final-evaluation seed set only, after config freeze.

---

## Usage

```bash
# Smoke test (harness verification only)
python -m v1.calibration.optuna_calibration \
  --n-trials 2 --episodes 50 --max-steps 100 \
  --calibration-seeds 1 --validation-seeds 4 \
  --output-dir experiments/optuna/smoke

# Full calibration (default budgets)
python -m v1.calibration.optuna_calibration \
  --n-trials 30 --episodes 300 --max-steps 100 \
  --study-name v1_optuna_calibration \
  --output-dir experiments/optuna
```

Outputs: `study.db`, `study.pkl`, `trials.csv`, `best_trial.json`,
`top_configs.csv`, `calibration_summary.csv`, `validation_top_configs.csv`,
`validation_summary.md`.
