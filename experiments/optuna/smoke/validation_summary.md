# Optuna Validation Summary

Validation runs are **report-only**. They do not continue tuning.

- Calibration seeds used during search: `[1]`
- Validation seeds used here: `[4]`
- Final evaluation seeds (locked, not used): `[100, 101, 102, 103, 104]`

## Top candidate configs (from calibration)

- Trial 0: score=4.2602, tcp=4.872700594236813, msb=4.802857225639665, nmfp=4.92797576724562, λ=1.0986584841970366, ε=4.2079886696066345e-06

## Validation results

### Rank 1 (trial 0)

- Egoistic safe_merge (mean): 0.000
- Egoistic collision (mean): 1.000
- Rawlsian safe_merge (mean): 0.900
- Rawlsian collision (mean): 0.100
- Rawlsian min_experience (mean): 8.806
