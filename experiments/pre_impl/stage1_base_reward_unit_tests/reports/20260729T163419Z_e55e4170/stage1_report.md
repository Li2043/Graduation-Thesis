# Stage 1 Report — Base Reward Unit Tests

## 1. Overall: **PASS**

- Git commit: `e55e4170219dc6a33abb452abac36f822b49e3cd`
- Configuration SHA-256: `644971fd85390380a2f42f9e255cf740d55cca6db8db1205c221a25d639a319f`

## 2. Unit-test summary

| Metric | Value |
|--------|-------|
| Passed | 29 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Status | PASS |

Required coverage: Tests 1–18 in `tests/rewards/test_base_reward_v2.py`.

## 3. Integration smoke-test result

**Status:** `BLOCKED`

BLOCKED: environment integration smoke cannot run because `C:/Users/HP/Desktop/毕业项目/thesis/final_new/src/thesis/envs` does not exist in this repository. Tried imports: thesis.envs.long_repeated_merge (No module named 'thesis.envs'); thesis.envs (No module named 'thesis.envs'); thesis.environment (No module named 'thesis.environment'). No LongRepeatedMergeEnv (or equivalent) is available; refusing to fabricate an adapter. Pure unit tests remain authoritative.

> A blocked smoke test does **not** validate environment integration. Only pure unit tests have been validated.

## 4. Reward decomposition examples

| case_id | controller | progress | exit | collision | hard_brake | total |
|---------|------------|----------|------|------------|------------|-------|
| T01_stationary | A | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 0.000000 |
| T01_stationary | B | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 0.000000 |
| T02_positive_progress | A | 0.040000 | 0.000000 | -0.000000 | -0.000000 | 0.040000 |
| T02_positive_progress | B | 0.040000 | 0.000000 | -0.000000 | -0.000000 | 0.040000 |
| T04_safe_exit_A | A | 0.004000 | 0.600000 | -0.000000 | -0.000000 | 0.604000 |
| T04_safe_exit_A | B | 0.004000 | 0.000000 | -0.000000 | -0.000000 | 0.004000 |
| T08_collision_blocks_exit | A | 0.004000 | 0.000000 | -1.000000 | -0.000000 | -0.996000 |
| T08_collision_blocks_exit | B | 0.000000 | 0.000000 | -1.000000 | -0.000000 | -1.000000 |
| T10_intermediate_braking | A | 0.000000 | 0.000000 | -0.000000 | -0.025000 | -0.025000 |
| T10_intermediate_braking | B | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 0.000000 |
| T14_negative_progress | A | -0.020000 | 0.000000 | -0.000000 | -0.000000 | -0.020000 |
| T14_negative_progress | B | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 0.000000 |
| T16_decomposition_combo | A | 0.040000 | 0.000000 | -1.000000 | -0.025000 | -0.985000 |
| T16_decomposition_combo | B | 0.080000 | 0.000000 | -1.000000 | -0.100000 | -1.020000 |

## 5. Route-coordinate discontinuity

None detected in recorded unit-case examples.

## 6. Repeated exit bonus

Unit Test 5 asserts exit bonus cannot repeat when `already_exited=True`. No integration trajectory was available to scan for repeated exits.

## 7. NaN / invalid-state events

Unit Test 17 asserts NaN/infinity raise clear errors. No NaN accepted by the reward module under those cases.

## 8. Unresolved implementation issues

- Environment integration smoke is BLOCKED; Stage 1 pure unit tests do not validate LongRepeatedMergeEnv / simulator wiring.

## 9. Recommendation

**PROCEED TO STAGE 2 (with caveat: environment smoke BLOCKED; wire simulator before claiming env-integrated reward validation)**

Note: braking `a_comfort`, `a_hard`, and `eta_hard_brake` remain **TEST-ONLY placeholders** and must not be treated as final experiment values.
