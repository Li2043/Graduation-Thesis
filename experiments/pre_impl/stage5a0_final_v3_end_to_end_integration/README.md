# Stage 5A-0 — Final V3 Reward–PBRS–DQN End-to-End Integration Regression

Connects locked G1-I1, locked comfort `(1.5, 3.5, 0.015)`, stakeholder-aware PBRS,
hardened Independent DQN, and the final V3 environment through one pre-training
pipeline.

## Not in scope

- Pilot / sustained policy training
- Hyperparameter tuning
- Final PBRS-lambda calibration
- Environment or comfort reselection

## Run

```bash
.\.venv_stage2b1\Scripts\python.exe experiments/pre_impl/stage5a0_final_v3_end_to_end_integration/scripts/run_stage5a0_tests.py
```

## Locks

- Environment: Stage 4A-R1 `20260730T003122Z_aee2d425`
- Comfort: Stage 3B-R1 `20260730T005639Z_c6992dd4`

Integration-test PBRS lambdas: `0.2` / `0.2` (`pbrs_parameters_final=false`).
