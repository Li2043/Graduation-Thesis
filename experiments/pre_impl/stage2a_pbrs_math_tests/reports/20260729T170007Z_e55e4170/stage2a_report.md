# Stage 2A Report — PBRS Mathematical Correctness

## 1. Overall: **PASS**

> **WARNING: git working tree is DIRTY (`git_dirty = true`).**
> For dissertation-grade acceptance, the final retained Stage 2A run should have `git_dirty = false`.

- Git commit: `e55e4170219dc6a33abb452abac36f822b49e3cd`
- Git dirty: `True`
- Configuration SHA-256: `a36a53d8a119106838583bcefa7fdfd52688cd7f52409fdee74ccf14e78a6e31`
- Learner / shaping gamma: `0.995`
- Test-only lambda_mean / lambda_min: `0.5` / `0.5` (NOT final experimental values)

## 2. Unit-test summary

| Metric | Value |
|--------|-------|
| Passed | 44 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Status | PASS |

## 3. Telescoping identities

- Terminal max |error|: `0.0`
- Truncation/open-segment max |error|: `1.1102230246251565e-16`
- All telescoping OK: `True`

| trajectory_id | abs_error | ok |
|---------------|-----------|----|
| smooth_improvement__mean | 1.110e-16 | True |
| smooth_improvement__min | 0.000e+00 | True |
| mean_improve_min_constant__mean | 0.000e+00 | True |
| mean_improve_min_constant__min | 5.204e-18 | True |
| worst_off_improvement__mean | 6.245e-17 | True |
| worst_off_improvement__min | 0.000e+00 | True |
| worst_off_identity_switch__mean | 1.214e-17 | True |
| worst_off_identity_switch__min | 2.776e-17 | True |
| true_terminal_success__mean | 0.000e+00 | True |
| true_terminal_success__min | 0.000e+00 | True |
| true_terminal_collision__mean | 0.000e+00 | True |
| true_terminal_collision__min | 0.000e+00 | True |
| external_truncation_nonzero__mean | 0.000e+00 | True |
| external_truncation_nonzero__min | 2.776e-17 | True |

## 4. Scope limits

- Environment integration: **UNVERIFIED** (not in Stage 2A scope).
- DQN / policy training: **UNVERIFIED** (not in Stage 2A scope).
- Lambda values: **TEST-ONLY**; do not treat as calibrated.

## 5. Recommendation

**PROCEED TO STAGE 2B (environment / learner wiring only; lambdas still uncalibrated)**
