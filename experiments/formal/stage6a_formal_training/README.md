# Stage 6A-0 — Formal 100K Training Infrastructure

Independent-run formal training runner and multi-job orchestrator.

**Do not start retained 30×100K training from this computer unless explicitly commanded.**

## Single job

```powershell
.\.venv_stage2b1\Scripts\python.exe experiments/formal/stage6a_formal_training/scripts/run_formal_job.py `
  --condition baseline --master-seed 61001 `
  --protocol-lock <path-to-final_training_protocol.yaml> `
  --output-root <output-root> --device cpu
```

## Matrix (process parallelism)

```powershell
.\.venv_stage2b1\Scripts\python.exe experiments/formal/stage6a_formal_training/scripts/run_formal_matrix.py `
  --run-matrix <formal_run_matrix.csv> `
  --protocol-lock <path-to-final_training_protocol.yaml> `
  --output-root <output-root> --workers 12 --threads-per-worker 1 --dry-run
```
