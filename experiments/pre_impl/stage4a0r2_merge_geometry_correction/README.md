# Stage 4A-0R2 — Realistic Merge Centreline Correction

Corrects `FinalRouteGeometry` so `merge_start`/`merge_end` define a quintic
ramp-to-mainline convergence (not a 90° quarter-circle join at `merge_start`).

## Run

```bash
python experiments/pre_impl/stage4a0r2_merge_geometry_correction/scripts/run_stage4a0r2_tests.py
```
