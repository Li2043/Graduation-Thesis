# Stage 3B-R1 — Final-Environment Joint Comfort Calibration

Calibrates `(a_comfort, a_hard, eta_H)` jointly on the locked Stage 4A-R1
environment (`G1-I1`). The original Stage 3B run remains historical FAIL evidence.

## Run

```bash
.\.venv_stage2b1\Scripts\python.exe experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/scripts/run_stage3b_r1_calibration.py
```

No DQN. No PBRS λ. Does not modify the final environment lock.
