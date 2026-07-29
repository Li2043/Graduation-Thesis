# Stage 3B — Comfort Threshold and Hard-Braking Weight Calibration

Offline calibration of `a_comfort`, `a_hard`, and `eta_hard_brake` from the
retained Stage 3A scripted transition trace.

## Scope

- No DQN training / no learned policies
- No PBRS λ tuning
- Progress / exit / collision weights frozen
- Source Stage 3A run is immutable (hashed)

## Source freeze

| Field | Value |
|-------|--------|
| Stage 3A `run_id` | `20260729T222933Z_3b07a818` |
| Stage 3A `git_commit` | `3b07a81879e913a175bfd05f8c985fc095841d34` |
| Primary file | `data/raw/.../transition_trace.jsonl` |
| `dt` | `0.2` s |

## Run

```powershell
$env:PYTHONPATH="src"
.\.venv_stage2b1\Scripts\python.exe experiments/pre_impl/stage3b_comfort_calibration/scripts/run_stage3b_calibration.py
```

## Notes

Selected parameters remain **provisional** until Stage 4 freezes geometry,
decision interval, vehicle dynamics, and background traffic
(`parameters_final=false`).
