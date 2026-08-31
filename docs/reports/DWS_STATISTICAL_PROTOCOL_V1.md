# DWS Statistical Protocol V1 (Frozen)

Frozen statistical analysis protocol for the Dense Welfare Shaping (DWS)
follow-up, covering the four Maximin cells. This document is the
human-readable specification; `scripts/analyze_dws_formal.py` implements it
exactly. This protocol was frozen before Cell 2 (Maximin + DWS) formal
training completed — do not adjust any value in this document after seeing
Cell 2 results.

## Four Maximin cells

| Cell | Name | Observation | Welfare feedback |
|---|---|---|---|
| 1 | Maximin | Original 18D | terminal-only |
| 2 | Maximin + DWS | Original 18D | terminal + step-wise DWS |
| 3 | Maximin + WSC | WSC 22D | terminal-only |
| 4 | Maximin + WSC + DWS | WSC 22D | terminal + step-wise DWS |

Unit of replication: the independently trained seed (n=12), not the
individual evaluation episode. The 12 formal seed identities:
`900101, 900102, 900103, 900104, 910101, 910102, 920101, 920102, 920103, 920104, 920105, 920106`.

## 1. Primary outcomes

Exactly two primary fairness outcomes: worst-off episode utility `U_min` and
Utility Gini `G_U`. For each seed: compute the episode-level quantity per
held-out episode, then average across the full 256-episode held-out bank
within that seed. Only the resulting seed-level value enters inference. The
256 episodes are never treated as independent statistical replicates.
Mean utility, completion, collision, timeout, GGI, and burden are
secondary/descriptive, not part of the confirmatory family.

## 2. Primary DWS contrasts

Per seed:

```
Delta_DWS_Original(Y) = Y(Cell 2) - Y(Cell 1)
Delta_DWS_WSC(Y)      = Y(Cell 4) - Y(Cell 3)
```

for `Y in {U_min, Gini}`. Favourable direction: positive for `U_min`,
negative for `Gini`. These four outcome-by-information contrasts are the
primary confirmatory comparisons.

## 3. Secondary DWS x WSC interaction

Per seed, for `Y in {U_min, Gini}`:

```
I_DWSxWSC(Y) = [Y(Cell 4) - Y(Cell 3)] - [Y(Cell 2) - Y(Cell 1)]
             = Delta_DWS_WSC(Y) - Delta_DWS_Original(Y)
```

Favourable-under-WSC direction: positive for `U_min`, negative for `Gini`.
Secondary mechanism analysis only — excluded from both Holm families.

## 4. Paired bootstrap

Seed identity is the matched resampling unit. `n_boot=10,000`, RNG seed `0`,
95% CI via the percentile method: resample 12 seed indices with replacement,
compute the mean of the resampled per-seed contrast values, repeat 10,000
times, take the [2.5th, 97.5th] percentile of the resulting distribution.
Reported per contrast: observed mean effect, median effect, n_positive,
n_negative, n_zero (values exactly 0 reported separately, never forced into
a sign bucket), 95% CI.

## 5. P-values

Two-sided paired-bootstrap p-values for a null of zero, computed **only**
for the four primary contrasts (`Delta_DWS_Original`/`Delta_DWS_WSC` x
`U_min`/`Gini`). Formula (documented exactly, matches the code in
`analyze_dws_formal.py::paired_bootstrap`):

```
p = min(1, 2 * min(P(bootstrap_mean <= 0), P(bootstrap_mean >= 0)))
```

using the same 10,000 percentile-bootstrap draws as the CI (not a
null-shifted resampling scheme). No p-values are computed for mean utility,
completion, collision, timeout, GGI, burden, or the DWS x WSC interaction.

## 6. Multiplicity correction

Holm step-down correction, applied **separately** within two families of
exactly 2 p-values each:

- Family A (`U_min`): Delta_DWS_Original, Delta_DWS_WSC
- Family B (`Gini`): Delta_DWS_Original, Delta_DWS_WSC

`U_min` and `Gini` are never combined into one 4-test family. The DWS x WSC
interaction is excluded from both families. Both raw and Holm-adjusted
p-values are reported.

## 7. Task and safety outcomes

Descriptive only: mean utility, completion, collision, timeout for all four
cells, plus within-seed DWS differences under Original and WSC information,
with paired-bootstrap 95% CIs. No confirmatory p-values, no Holm correction.
The pre-existing WSC non-inferiority margins (completion -0.05, collision
+0.03) are **not** reused here — no frozen DWS decision adopts them (see
`DENSE_PROTOCOL_VERIFICATION_FOR_THESIS.md`). Raw contrasts and CIs only.

**Known data limitation**: `mean_U` and `timeout` are not available for
Cells 1/3 in the pre-existing seed-level source CSV (only `completion`,
`collision`, `U_min`, `Gini` are present there) — the task/safety descriptive
table is therefore limited to `completion` and `collision` in practice.

## 8. Sensitivity analysis

Leave-one-seed-out for all four primary contrasts and both interaction
terms (6 quantities total). For each: full n=12 estimate, then 12
recomputations each omitting one seed, reporting the min and max
leave-one-out estimate and which omitted seed produced each extreme. No
seed (910102 or any other) is excluded from the primary n=12 analysis under
any circumstance.

## 9. Input data (verified sources)

- **Cell 1 & Cell 3** (seed-level, pre-aggregated): `F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_formal\outputs\wsc_v2_formal_seed_level.csv`, filtered to `condition == "maximin"`; `orig_*` columns = Cell 1, `wsc_*` columns = Cell 3.
- **Cell 4** (episode-level, aggregated by this pipeline): `outputs/welfare_analysis/dense_interim_evaluation_12seed_full.csv`.
- **Cell 2** (episode-level, aggregated by this pipeline): `outputs/welfare_analysis/dense_interim_evaluation_maximin_dense_12seed_full.csv` — **not yet produced**; Cell 2 formal training was incomplete at the time this protocol was frozen and implemented. Generate it with `scripts/evaluate_dense_interim.py --run-tag maximin_dense --condition maximin --no-include-welfare-state --seeds <all 12> --out-suffix maximin_dense_12seed_full` once Cell 2 reaches step 2,000,000 for all 12 seeds.

**Metric-definition identity across cells — verified**:
`thesis/study_b/utility.py` is byte-identical between this bundle
(`C:\dense reward\project\src\...`) and the source repo
(`F:\正式训练_seed_replication_v1\project\src\...`) that produced the Cell
1/3 CSV. Both the Cell-1/3-producing pipeline (`wsc_v2_formal_analysis.py`
+ `evaluate_wsc_formal_v2.py`) and this bundle's Cell-2/4 pipeline
(`evaluate_dense_interim.py`, itself adapted from `evaluate_wsc_formal_v2.py`)
use: the same H1 scenario bank, the same 256-episode count
(`N_EPISODES_EXPECTED = 256` in `wsc_v2_formal_analysis.py`, independently
confirmed as `len(H1.json) == 256`), the same
`ensemble_window_for_stage_end(2_000_000)` checkpoint-window rule, and the
same seed-level aggregation convention (`U_min` = mean of per-episode
`min_U`; `Gini` = mean of per-episode `gini`, both verified in
`wsc_v2_formal_analysis.py::_seed_level_from_rows`, replicated exactly in
`analyze_dws_formal.py::_aggregate_episode_csv_to_seed_level`). No
discrepancy found; no deviation from Section 9's identity requirement was
necessary.

## 10/11. Implementation and outputs

Implemented at `scripts/analyze_dws_formal.py` (read-only w.r.t. all
training/eval data and all existing WSC scripts; deterministic; fails
loudly — non-zero exit, zero output files written — on any missing cell,
missing seed, or duplicate seed-condition rows). Outputs are written to
`outputs/dws_statistical_analysis/`: `dws_seed_level_metrics.csv`,
`dws_primary_contrasts.csv`, `dws_primary_summary.csv`,
`dws_interaction_summary.csv`, `dws_leave_one_seed_out.csv`,
`dws_task_safety_descriptive.csv`. See `DWS_STATISTICAL_ANALYSIS_README.md`
for exact column definitions and rerun instructions.

## 12. Validation

`scripts/analyze_dws_formal.py --self-test` runs 9 checks against synthetic
data only (paired-difference correctness, sign-flip on label swap, U_min/
Gini favourable-direction convention, Holm correction on a known pair,
bootstrap RNG-seed-0 reproducibility, hard failure on a missing seed,
interaction-identity algebra) — all 9 pass as of this protocol's freeze.
