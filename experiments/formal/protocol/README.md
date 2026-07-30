# Stage 5C-0 — Final PBRS and Formal Training Protocol Lock

Freezes the dissertation training protocol before formal multi-seed training.

## Scope

- Final PBRS scales (`λ_mean = λ_min = 0.2`)
- Final Independent DQN protocol (`[64, 64]`, Adam, replay, ε-schedule)
- Formal seed matrix (3 × 10 = 30 runs, 20 000 steps each)
- Checkpoint / evaluation / analysis / failure policies

**Does not** execute sustained or formal training.

## Run

```bash
.\.venv_stage2b1\Scripts\python.exe experiments/formal/protocol/scripts/run_stage5c0_protocol_lock.py
```
