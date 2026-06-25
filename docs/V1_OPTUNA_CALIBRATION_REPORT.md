# V1 Optuna Calibration Report

Status: **smoke test completed**. Full Optuna study **not yet run**.

---

## 1. Number of trials run

| run | trials | episodes | calibration seeds | validation seeds | output dir |
|---|---|---|---|---|---|
| Smoke harness test | 2 | 50 | [1] | [4] | `experiments/optuna/smoke/` |

---

## 2. Best trial (smoke)

| field | value |
|---|---|
| trial_number | 0 |
| calibration_score | 4.2602 |
| constraint_passed | True |
| terminal_collision_penalty | 4.87 |
| merge_success_bonus | 4.80 |
| non_merge_failure_penalty | 4.93 |
| rawlsian_lambda | 1.10 |
| rawlsian_epsilon | 4.21e-06 |

Trial 1 failed constraints (egoistic safe_merge=0.6, non_merge_failure=0.4) and received penalty score −1_000_000.

Full detail: `experiments/optuna/smoke/best_trial.json`

---

## 3. Top candidate configs (smoke)

Only trial 0 passed constraints. See `experiments/optuna/smoke/top_configs.csv`.

Calibration metrics (seed 1, 50 episodes):

| mode | safe_merge | collision | non_merge_failure | min_experience |
|---|---|---|---|---|
| egoistic | 1.00 | 0.00 | 0.00 | 8.30 |
| rawlsian | 0.80 | 0.20 | 0.00 | 9.18 |

---

## 4. Validation results (smoke)

Top config (trial 0) validated on seed 4 (50 episodes). See
`experiments/optuna/smoke/validation_summary.md`.

| mode | safe_merge | collision |
|---|---|---|
| egoistic | 0.00 | 1.00 |
| rawlsian | 0.90 | 0.10 |

**Interpretation:** Smoke runs use only 50 episodes and a single seed per phase.
Validation instability here is expected and must not be used to select a final
config. The harness verified that calibration → validation pipeline executes.

---

## 5. Ready to freeze?

**No.** Smoke test only confirms the Optuna harness works. A full calibration
study is required before freezing any configuration.

---

## 6. Final experiments safe yet?

**No.** Do not run final evaluation seeds until:

1. Full Optuna calibration completes (default: 30 trials × 300 episodes × seeds [1,2,3]).
2. Top configs validate acceptably on seeds [4, 5].
3. One configuration is explicitly frozen in `V1_DECISION_LOG.md`.

---

## Smoke output files

Under `experiments/optuna/smoke/`:

- `study.db` — Optuna SQLite study
- `study.pkl` — pickled study object
- `trials.csv` — per-trial metrics
- `best_trial.json` — best trial + seed metadata
- `top_configs.csv` — constraint-passing top configs
- `calibration_summary.csv` — study summary
- `validation_top_configs.csv` — validation metrics
- `validation_summary.md` — human-readable validation report

---

## Next step: full calibration command

```bash
python -m v1.calibration.optuna_calibration \
  --n-trials 30 \
  --episodes 300 \
  --max-steps 100 \
  --calibration-seeds 1,2,3 \
  --validation-seeds 4,5 \
  --study-name v1_optuna_calibration \
  --output-dir experiments/optuna
```

This report will be updated after the full study.
