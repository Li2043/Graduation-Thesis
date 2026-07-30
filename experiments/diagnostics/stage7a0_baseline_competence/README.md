# Stage 7A-0 Baseline Competence Diagnostic Pilot

Exploratory, Baseline-only diagnostic. Not confirmatory.

## Status notes

- Intermediate full checkpoints (`ckpt_step_*.pt`) are **missing** from published Stage 6A.
- Only `final_online_target_weights.pt` at 100K is available for greedy reconstruction.
- Continuation probe 100K→200K is **BLOCKED**.

## Run

```powershell
$env:PYTHONPATH="src"
.\.venv_stage2b1\Scripts\python.exe experiments\diagnostics\stage7a0_baseline_competence\scripts\run_stage7a0.py
```
