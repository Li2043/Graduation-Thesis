# Stage 3A — Scripted Base-Outcome and Incentive Audit

## Purpose

Audit behavioural incentives of the **frozen base reward** using deterministic
scripted trajectories.

**Scope: scripted audit only. No policy training. No DQN updates.**

PBRS may appear in environment diagnostics but does **not** determine PASS/FAIL
and is not used to select base-reward parameters.

## Frozen formula

```
r_base = 0.4*Δρ + 0.6*safe_exit - 1.0*collision - η*H
```

Comfort thresholds and `η` remain **TEST-ONLY** (`a_comfort=2`, `a_hard=6`, `η=0.1`).

## How to run

```bash
.\.venv_stage2b1\Scripts\python.exe experiments/pre_impl/stage3a_scripted_base_outcome_audit/scripts/run_stage3a_audit.py
```

## Matched blocks

Eight fixed initial-condition blocks (`block_001` … `block_008`) share geometry,
spawns, speeds, and seed within each block. Scripts differ; ICs are not tuned
after observing rankings.

## Fixture-only collisions

Teleport / fixture collisions are labelled `fixture_only=true` and excluded from
behavioural return ranking. Physical collision ranking requires dynamics-evolved
collisions.

## Explicitly unverified

- DQN policy performance
- Mean-PBRS vs Min-PBRS
- Final comfort / η calibration
