# Stage 2B-2 — Independent DQN, Replay, Masking, Bootstrap Integration

## Purpose

Integrate the validated merge environment, base reward, and PBRS outputs with a
**minimal Independent DQN data path**.

This stage verifies **correctness**, not performance.

**This is an integration test, not policy training.**

- No confirmatory training
- No multi-seed optimisation
- No architecture / exploration tuning
- No final λ calibration

## Framework

Uses dedicated `.venv_stage2b1` (Python 3.14.6) with pinned:

```text
torch==2.13.0+cpu
```

See `requirements-stage2b2.txt`.

## Independent learners

`learner_A` and `learner_B` each own:

- online Q-network
- target Q-network
- optimiser
- replay buffer
- exploration RNG

Weights are **not** shared. Masks follow **traffic role**, not controller id.

## Action set

`MAINTAIN`, `ACCELERATE`, `DECELERATE` (no MERGE — environment does not use one).

Masking applies at:

1. behaviour selection (greedy / ε-greedy over legal actions only)
2. Bellman target (`max` over legal next actions only)

## DQN target

```text
y = r + γ * (1 - terminated) * masked_max_next_Q
```

- True terminal → no bootstrap
- External truncation → **retains** bootstrap
- Never `1 - (terminated or truncated)`

## Completed controller (Option 1)

When A exits before B:

- `E_A = 1` remains in the fixed stakeholder potential
- A becomes inactive (env forces zero accel)
- Replay storage for A stops after the exit transition
- Joint step may still receive placeholder `MAINTAIN` for API compatibility
- No fictitious MERGE / absorbing NO_OP action is invented

## TEST-ONLY settings

Network `[32, 32]`, learning rate, ε, and PBRS λ values are **TEST-ONLY**.

## How to run

```bash
.\.venv_stage2b1\Scripts\python.exe experiments/pre_impl/stage2b2_dqn_replay_bootstrap/scripts/run_stage2b2_tests.py
```

## Explicitly unverified

Formal policy training has **not** started.
