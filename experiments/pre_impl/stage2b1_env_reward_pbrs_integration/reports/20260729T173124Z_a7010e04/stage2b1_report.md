# Stage 2B-1 Report — Env / Reward / PBRS Integration

## 1. Overall: **PASS**

 > **WARNING: git_dirty = true**. Dissertation-retained runs should be clean.

- Git commit: `a7010e049d7b09619929ba092d065de16fd7d7d5`
- Git dirty: `True`
- Python: `3.14.6`
- Config SHA-256: `f1a49d15b032983d4f7b54b18c57ef4e5373fffbd31a221b82a4d74c577471f8`
- Packages: `{"numpy": "2.3.2", "gymnasium": "1.2.0", "highway-env": "1.12.0", "pytest": "8.4.1", "PyYAML": "6.0.2"}`

## 2. Tests

| Metric | Value |
|--------|-------|
| Passed | 25 |
| Failed | 0 |
| Errors | 0 |
| Status | PASS |

## 3. Scenario metrics

| Metric | Value |
|--------|-------|
| Scenarios | 12 |
| NaN count | 0 |
| Route discontinuity count | 3 |
| Repeated exit count | 0 |
| Invalid term/trunc flags | 0 |

## 4. Unresolved limitations

- Dynamics are thesis-owned kinematics (not a live highway-env wrap); highway-env is installed for reproducibility.
- Geometry is integration-test configuration, not final dissertation geometry.
- DQN and replay-buffer integration remain **UNVERIFIED**.
- Lambda / comfort parameters remain **TEST-ONLY**.

## 5. Recommendation

**PROCEED TO STAGE 2B-2 (DQN/replay wiring) — DQN still unverified**
