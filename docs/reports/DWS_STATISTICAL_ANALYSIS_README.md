# DWS Statistical Analysis — README

How to rerun `scripts/analyze_dws_formal.py`, and what each output means.
See `DWS_STATISTICAL_PROTOCOL_V1.md` for the frozen statistical protocol
this script implements; this file documents the implementation itself.

## Input files

| Cell | Source | Format |
|---|---|---|
| 1 (Maximin) | `F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_formal\outputs\wsc_v2_formal_seed_level.csv`, `condition=="maximin"`, `orig_*` columns | seed-level, pre-aggregated |
| 2 (Maximin+DWS) | `outputs/welfare_analysis/dense_interim_evaluation_maximin_dense_12seed_full.csv` (not yet produced) | episode-level, 256/seed |
| 3 (Maximin+WSC) | same CSV as Cell 1, `wsc_*` columns | seed-level, pre-aggregated |
| 4 (Maximin+WSC+DWS) | `outputs/welfare_analysis/dense_interim_evaluation_12seed_full.csv` | episode-level, 256/seed |

## Metric mapping

| Protocol name | Cell 1/3 CSV column | Cell 2/4 CSV column (per-episode, then averaged) |
|---|---|---|
| `U_min` | `orig_U_min` / `wsc_U_min` | `min_U` |
| `Gini` | `orig_gini` / `wsc_gini` | `gini` |
| `completion` | `orig_completion` / `wsc_completion` | `completion` |
| `collision` | `orig_collision` / `wsc_collision` | `collision` |
| `mean_U` | not available for Cell 1/3 | `mean_U` |
| `timeout` | not available for Cell 1/3 | `timeout` |

Seed-level aggregation for Cells 2/4: `U_min = mean(min_U over 256 episodes)`,
`Gini = mean(gini over 256 episodes)` — identical convention to the one
already used to produce the Cell 1/3 CSV (`wsc_v2_formal_analysis.py::
_seed_level_from_rows`).

## Exact formulas

```
Delta_DWS_Original(Y) = Y(Cell 2) - Y(Cell 1)
Delta_DWS_WSC(Y)      = Y(Cell 4) - Y(Cell 3)
I_DWSxWSC(Y)           = Delta_DWS_WSC(Y) - Delta_DWS_Original(Y)
```

for `Y in {U_min, Gini}`, computed per seed, then summarized across the 12
seeds via paired bootstrap.

## Bootstrap implementation

`n_boot=10,000`, RNG seed `0` (Python's `random.Random(0)`, per-contrast
independent stream — not shared/reused across contrasts, so results are
deterministic per contrast regardless of call order). Each iteration:
draw 12 indices into the 12 seed values with replacement, take the mean.
95% CI = 2.5th/97.5th percentile of the 10,000 resulting means (percentile
bootstrap, not BCa or normal-approximation).

## P-value method

`p = min(1, 2 * min(P(bootstrap_mean <= 0), P(bootstrap_mean >= 0)))`,
computed from the same 10,000-draw distribution as the CI. Two-sided.
Computed only for the 4 primary contrasts (never for mean_U/completion/
collision/timeout/GGI/burden/interaction).

## Holm families

- Family A (`U_min`): [Delta_DWS_Original, Delta_DWS_WSC] — 2 tests.
- Family B (`Gini`): [Delta_DWS_Original, Delta_DWS_WSC] — 2 tests.

Standard Holm step-down: sort ascending, multiply smallest raw p by 2
(family size), next by 1, take the running maximum to enforce monotonicity,
cap at 1.0.

## Sensitivity analysis

Leave-one-seed-out over all 4 primary contrasts + 2 interaction terms.
Never drops a seed from the primary n=12 result — this is a
robustness/diagnostic table only, in `dws_leave_one_seed_out.csv`.

## Output files (`outputs/dws_statistical_analysis/`)

- `dws_seed_level_metrics.csv` — one row per seed x cell, primary + secondary outcomes.
- `dws_primary_contrasts.csv` — one row per seed x contrast x outcome.
- `dws_primary_summary.csv` — one row per (outcome, contrast): mean/median effect, bootstrap CI, raw + Holm p-values, n_positive/negative/zero.
- `dws_interaction_summary.csv` — secondary DWS x WSC interaction, bootstrap CI only, no p-values.
- `dws_leave_one_seed_out.csv` — full-n12 estimate, min/max leave-one-out estimate and which seed produced each, per quantity.
- `dws_task_safety_descriptive.csv` — descriptive completion/collision (mean_U/timeout unavailable for Cells 1/3) DWS deltas under Original and WSC, CI only, no p-values.

## How to rerun

```powershell
cd "C:\dense reward"
.venv\Scripts\python.exe scripts\analyze_dws_formal.py --self-test   # validation suite only, synthetic data
.venv\Scripts\python.exe scripts\analyze_dws_formal.py               # real four-cell analysis
```

The real run fails loudly (non-zero exit, zero output files written) and
prints `WAITING FOR CELL 2` until
`outputs/welfare_analysis/dense_interim_evaluation_maximin_dense_12seed_full.csv`
exists with all 12 seeds x 256 episodes. Generate that file once Cell 2
training reaches step 2,000,000 for all 12 seeds:

```powershell
.venv\Scripts\python.exe scripts\evaluate_dense_interim.py --run-tag maximin_dense --condition maximin --no-include-welfare-state --seeds 900101 900102 900103 900104 910101 910102 920101 920102 920103 920104 920105 920106 --out-suffix maximin_dense_12seed_full
```

then rerun `analyze_dws_formal.py` with no arguments (the default
`--cell2-csv` path matches this exact `--out-suffix`).
