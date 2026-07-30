# Stage 4A-R1 — Final Environment Reselection on Hardened V3 Geometry

Reruns environment candidate selection and genuine choice-state certification
after Stage 4A-0R (physics/obs hardening) and Stage 4A-0R2 (quintic merge geometry).

The historical Stage 4A lock `20260729T231946Z_c8d92bc3` is superseded and must
not be reused for training.

## Run

```bash
.\.venv_stage2b1\Scripts\python.exe experiments/pre_impl/stage4a_r1_final_environment_reselection/scripts/run_stage4a_r1_reselection.py
```

Requires `PYTHONPATH=src` (set by the runner).

## Constraints

- No DQN / optimiser updates
- No comfort calibration
- No PBRS λ calibration
- Calibration-only candidate selection; holdout validation once; no reselection
