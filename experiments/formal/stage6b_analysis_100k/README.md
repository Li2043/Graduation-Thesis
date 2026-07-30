# Stage 6B — Formal 100K Results Analysis

Fetches/verifies Stage 6A results and runs preregistered seed-level analysis.

Does **not** retrain policies or alter the formal protocol.

## Run

```powershell
.\.venv_stage2b1\Scripts\python.exe experiments/formal/stage6b_analysis_100k/scripts/run_stage6b_analysis.py `
  --results-root <path-to-stage6a-execution-dir> `
  --result-tag formal-results-100k-complete `
  --result-commit c75845935a7fe9179b691298b2329208853773a6
```
