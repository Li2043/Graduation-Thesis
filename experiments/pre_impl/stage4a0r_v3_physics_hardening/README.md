# Stage 4A-0R — Merge Environment V3 Hardening

Hardens `MergeEnvCandidateV3` physics and Markov observations.

- Does **not** rerun Stage 4A candidate selection
- Does **not** write `final_environment_lock.yaml`
- Marks prior Stage 4A run `20260729T231946Z_c8d92bc3` as `superseded_pending_v3_hardening`

## Run

```bash
python experiments/pre_impl/stage4a0r_v3_physics_hardening/scripts/run_stage4a0r_tests.py
```
