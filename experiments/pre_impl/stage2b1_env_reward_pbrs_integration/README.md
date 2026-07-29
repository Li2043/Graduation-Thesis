# Stage 2B-1 — Minimal Merge Environment and Reward/PBRS Integration

## Purpose

Create the **minimum executable merge environment** required to integrate and
test the already-validated Stage 1 base reward and Stage 2A PBRS modules.

**No DQN training. No policy optimisation. No final lambda / comfort calibration.**

## Python environment

Use the dedicated virtual environment:

```text
.venv_stage2b1
```

Install:

```bash
python -m venv .venv_stage2b1
.\.venv_stage2b1\Scripts\python.exe -m pip install -r requirements-stage2b1.txt
```

Do not modify system Python. Local Python 3.12 was unusable due to a corrupted
stdlib `json` module; Stage 2B-1 uses Python 3.14 with `highway-env==1.12.0`
(declared 3.14 support since 1.11).

## What is implemented

- `MergeEnvV2`: thesis-owned kinematic merge simulator (Gymnasium API)
- Fixed stakeholder set `V = {A, B, B_front, B_rear}`
- Controller identity separate from traffic role (`mainline` / `ramp`)
- Continuous route coordinates for mainline and ramp-through-join
- Base reward via `base_reward_v2` (not duplicated)
- Mean / Min PBRS diagnostics via `pbrs_v2` (not duplicated)
- Scripted scenarios for exits, collisions, truncation, hard braking

`highway-env` is installed for dependency reproducibility; Stage 2B-1 dynamics
are thesis-owned for deterministic fixtures (wrapping deferred).

## Geometry note

Road geometry in this stage is an **integration-test configuration**, not the
frozen dissertation geometry.

## Test-only parameters

| Parameter | Value | Status |
|-----------|-------|--------|
| `a_comfort`, `a_hard`, `eta_hard_brake` | 2 / 6 / 0.1 | TEST-ONLY |
| `lambda_mean`, `lambda_min` | 0.5 / 0.5 | TEST-ONLY |

## How to run

```bash
.\.venv_stage2b1\Scripts\python.exe experiments/pre_impl/stage2b1_env_reward_pbrs_integration/scripts/run_stage2b1_tests.py
```

## Outputs

Unique `run_id` directories under `data/`, `processed/`, `reports/`, `logs/`,
`artifacts/`. Never overwrites Stage 1 / Stage 2A / prior 2B-1 runs.

## Acceptance (summary)

Environment + integration tests pass; continuous routes; exact stakeholder
identity; exit ≤1; collision overrides exit; success needs both exits;
truncation ≠ termination; terminal φ=0; truncation φ retained; exact
decompositions; no NaN; isolated run folders.

## Explicitly unverified

- DQN training
- Replay-buffer integration
- Final dissertation road geometry
- Final λ / comfort calibration
