# Stage 5A-0 Report — Final V3 End-to-End Integration

## Overall: **PASS**

- run_id: `20260730T015222Z_05e9613c`
- git: `05e9613cfe99731e0abb6550926f82cb58694f85` dirty=`False`
- tests: `{'passed': 206, 'failed': 0, 'errors': 0, 'skipped': 0, 'status': 'PASS'}`
- environment class: `MergeEnvCandidateV3`
- observation_dimension: 27
- algorithm: vanilla DQN (masked target max)
- integration_test_architecture: true
- reward conditions: baseline / mean_pbrs / min_pbrs
- integration lambdas: mean=0.2 min=0.2 (`pbrs_parameters_final=false`)
- environment lock: `d2d82ac02feb5bb2f5217f8e399972b91bd56cce343f60861954f66d7f70bf12` (unchanged=True)
- comfort lock: `1d9439c211955f9a8a177e455b6b5ff34aa98f85f3ff0677bbd62abd6d29b061` (unchanged=True)
- policy_training_started: false
- sustained_training_invoked: false
- isolated_optimizer_updates_only: true

## Integrity
```json
{
  "lock_hash_mismatch": 0,
  "physical_invariance_errors": 0,
  "decomposition_errors": 0,
  "telescoping_errors": 0,
  "determinism_errors": 0,
  "nan_inf_errors": 0,
  "v2_import_errors": 0
}
```

## Physical / PBRS
- max physical-trace diff: 0.0
- max decomposition error: 0.0
- mean telescoping error: 6.505213034913027e-17
- min telescoping error: 4.440892098500626e-16
- early-exit continuation cases: 1

## Isolated DQN updates
- count: 6
- max loss: 0.2827787399291992
- max |Q|: 0.23594829440116882
- target-network forward counts: [1, 1, 1, 1, 1, 1]
