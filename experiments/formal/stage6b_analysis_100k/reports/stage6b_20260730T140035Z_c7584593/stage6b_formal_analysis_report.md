# Stage 6B Formal Analysis Report (100K)

## Overall: **PASS**

- analysis_id: `stage6b_20260730T140035Z_c7584593`
- result tag: `formal-results-100k-complete`
- result commit: `c75845935a7fe9179b691298b2329208853773a6`
- runner commit: `a89256db879f04d1e02782ff8dc1af00ff1d75b9`
- formal_execution_id: `stage6a_20260730T094829Z_a89256db_44d5e647`
- completed / failed runs: `30` / `0`
- validated evaluation episodes at step 100000: `480`
- missing convention consistency (seeds×conditions): `11`
- bootstrap: `10000` replicates, seed `91001`
- Wilcoxon defined / undefined: `15` / `0`
- Holm correction: within each primary endpoint across three contrasts
- PBRS comparison: **equal coefficient** (not magnitude-matched, not RMS-matched)

## Method notes

Performance is reported at the preregistered 100,000-step endpoint.
Episode-level evaluation fields were reconstructed from published final network
weights because Stage 6A retained only evaluation summary counts. This
reconstruction executes the locked greedy evaluation protocol and does **not**
train policies.

Intermediate evaluation episode payloads were not published; learning-curve
endpoint trajectories therefore cannot be recovered for steps other than 100000.
No interpolation of missing checkpoints was performed.

## Primary contrasts

See `tables/stage6b_20260730T140035Z_c7584593/primary_endpoint_contrasts.csv`.

Do not interpret p > 0.05 as proof of no effect.
