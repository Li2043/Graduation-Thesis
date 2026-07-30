# Stage 5B-0 — Bounded Engineering Pilot Training

Small, strictly bounded engineering pilot over the final V3 training pipeline.

## Scope

Verifies sustained env/learner interaction, replay, optimiser updates, epsilon
schedule, target sync, checkpoint/resume, evaluation isolation, and artifacts.

**Not** formal dissertation training, hypothesis testing, or condition ranking.

## Run

```bash
.\.venv_stage2b1\Scripts\python.exe experiments/pilot/stage5b0_bounded_engineering_pilot/scripts/run_stage5b0_pilot.py
```

## Matrix

- Conditions: `baseline`, `mean_pbrs`, `min_pbrs`
- Seeds: `51001`, `51002`
- Steps/run: `5000`
- λ (pilot-only): `0.2` / `0.2` (`pbrs_parameters_final=false`)
