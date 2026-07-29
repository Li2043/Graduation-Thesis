# Stage 1 — Base Reward Unit Tests

## Purpose

Validate the **frozen Stage-1 base reward** for learning controllers A and B
**before** any DQN / policy training and **before** Mean-PBRS or Min-PBRS.

This stage proves formula correctness, decomposition identity, exit/collision
rules, and finite-value validation with deterministic unit tests.

**No DQN training occurs in this stage.**

## Frozen base-reward formula

After every environment transition \(s_t \rightarrow a_t \rightarrow s_{t+1}\),
for each learning controller \(i \in \{A, B\}\):

```
r_base[i,t]
    = 0.4 * delta_route_progress[i,t]
    + 0.6 * safe_exit_event[i,t+1]
    - 1.0 * stakeholder_collision_event[t+1]
    - eta_hard_brake * hard_braking_cost[i,t+1]
```

- Route progress \(\rho\) uses **distance along the assigned route**, not raw world-x.
- Negative \(\Delta\rho\) is preserved (not clipped to zero).
- Safe-exit bonus is awarded **at most once** per vehicle per episode and is
  blocked if a stakeholder collision occurs on the same transition.
- Fixed stakeholder set \(V = \{A, B, B_{front}, B_{rear}\}\); any collision
  involving \(V\) gives both A and B the shared \(-1.0\) collision component.
- Hard braking uses SI acceleration (negative = braking).

## Final design values vs test-only placeholders

| Parameter | Value | Status |
|-----------|-------|--------|
| `progress_weight` | 0.4 | **Final design** |
| `exit_weight` | 0.6 | **Final design** |
| `collision_penalty` | 1.0 | **Final design** |
| `a_comfort` | 2.0 | **TEST-ONLY** placeholder |
| `a_hard` | 6.0 | **TEST-ONLY** placeholder |
| `eta_hard_brake` | 0.1 | **TEST-ONLY** placeholder |

Do **not** treat braking thresholds or \(\eta\) as calibrated experimental
parameters. They exist only so Stage-1 unit tests are deterministic.

## Code layout

| Path | Role |
|------|------|
| `src/thesis/rewards/base_reward_v2.py` | Reusable production reward module |
| `tests/rewards/test_base_reward_v2.py` | Pure unit tests (Tests 1–18) |
| `configs/reward_unit_tests.yaml` | Resolved Stage-1 config |
| `scripts/run_stage1_tests.py` | Isolated experiment runner |

Legacy / older reward implementations are **not** overwritten.

## How to run

From the `final_new` repository root:

```bash
python experiments/pre_impl/stage1_base_reward_unit_tests/scripts/run_stage1_tests.py
```

Optional:

```bash
python experiments/pre_impl/stage1_base_reward_unit_tests/scripts/run_stage1_tests.py \
  --config experiments/pre_impl/stage1_base_reward_unit_tests/configs/reward_unit_tests.yaml
```

Requires `pytest` and `PyYAML` (see repo `requirements-stage1.txt`).

## Output locations

Every execution creates a **unique** run directory (never overwritten):

```
data/raw/<run_id>/
data/processed/<run_id>/
reports/<run_id>/
logs/<run_id>/
artifacts/<run_id>/
```

`run_id` format: `YYYYMMDDTHHMMSSZ_<short_git_commit>`

Pointer (does not replace history):

```
experiments/pre_impl/stage1_base_reward_unit_tests/latest_run.json
```

## Acceptance criteria

Stage 1 **PASS** only when all of the following hold:

- All required pure unit tests pass
- NaN / infinity rejected with clear errors
- Exit bonus at most once per vehicle
- Stakeholder collision penalises both A and B
- Collision transition cannot receive a safe-exit bonus
- Ordinary truncation does not invent terminal rewards
- Decomposition sums exactly to `total_reward`
- Route progress is along-route, not raw x
- No previous experiment output overwritten
- Manifest + report produced for every run

Environment integration smoke may be **BLOCKED** only when the exact API
blocker is documented. A blocked smoke test does **not** validate environment
integration.

## Explicit non-goals

- No policy / DQN training
- No Mean-PBRS / Min-PBRS
- No calibration of final `a_comfort`, `a_hard`, or `eta_hard_brake`
- No modification of prior experiment outputs
