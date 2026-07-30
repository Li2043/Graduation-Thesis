# Stage 4A-R1 Report — Final Environment Reselection

## 1. Overall: **PASS**

## 2. Scope
Reselection on hardened V3 physics (4A-0R) and quintic merge geometry (4A-0R2).
No DQN. No comfort calibration. No PBRS λ calibration.

## 3. Git: `aee2d425319a0f1c917c98af8df1cb1b8d6550a9` dirty=`False`

## 4. Prior runs
- Stage 4A-0R: `20260729T235855Z_8ab30c89`
- Stage 4A-0R2: `20260730T000945Z_d0051f73`
- Superseded Stage 4A: `20260729T231946Z_c8d92bc3`
- Superseded lock SHA-256: `d5614d41d0c229db70b76973c55daa6905d7c5f07dc0781b81826b8891d76ded`

## 5. Tests
```json
{
  "passed": 50,
  "failed": 0,
  "errors": 0,
  "skipped": 0,
  "status": "PASS"
}
```

## 6. Candidates: 9; feasible: 9
## 7. Blocks: calibration=12, validation=8, duplicate_signatures=0
## 8. Selected: `G1-I1`
## 9. Calibration certified: 11/12
## 10. Validation certified: 7/8 pass=True
## 11. Order gaps: cal med/max=0.008227002151346608/0.015881944498357878; val med/max=0.007244225363168272/0.017713010288839894
## 12. Background relevance: cal=1.0 val=0.875
## 13. Spontaneous collisions: 0
## 14. Label-swap max error: 0.0

## 15. Geometry diagnostics
- **G1**: world_x=60.000, arc=60.189931, max|heading|=0.124355, max|κ|=0.00638577, a_lat@20=2.554308, recover_err_ramp=5.684e-14
- **G2**: world_x=80.000, arc=80.142627, max|heading|=0.093477, max|κ|=0.00359912, a_lat@20=1.439649, recover_err_ramp=2.842e-14
- **G3**: world_x=70.000, arc=70.162922, max|heading|=0.106736, max|κ|=0.00469721, a_lat@20=1.878883, recover_err_ramp=2.842e-14

## 16. Integrity: {"route_discontinuity_count": 0, "repeated_exit_count": 0, "invalid_flag_count": 0, "nan_inf_count": 0, "fixture_count": 0}
## 17. Final lock: `C:\Users\HP\Desktop\毕业项目\thesis\final_new\experiments\pre_impl\stage4a_r1_final_environment_reselection\artifacts\20260730T003122Z_aee2d425\final_environment_lock.yaml` sha256=`d2d82ac02feb5bb2f5217f8e399972b91bd56cce343f60861954f66d7f70bf12`
## 18. Flags: environment_parameters_final=True, comfort_parameters_final=false, policy_training_started=false
