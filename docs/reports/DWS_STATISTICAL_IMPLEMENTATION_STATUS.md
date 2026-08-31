# DWS Statistical Implementation Status

# Status

Pipeline implemented, validated, and dry-run against real Cell 1/3/4 data.
**Blocked on Cell 2** (Maximin + DWS, no WSC): as of this report, seeds
920105/920106 are at step 1,700,000 of 2,000,000 (300,000 steps remaining);
the other 10 seeds are already at 2,000,000. No final four-cell result has
been produced, and none was fabricated.

# Frozen statistical decisions

See `DWS_STATISTICAL_PROTOCOL_V1.md` for the full frozen protocol. Summary:
2 primary outcomes (`U_min`, `Gini`); 4 primary contrasts (Delta_DWS_Original
and Delta_DWS_WSC, each for both outcomes); 1 secondary interaction per
outcome; 10,000-resample percentile paired bootstrap, RNG seed 0, 95% CI;
two-sided bootstrap p-values for the 4 primary contrasts only; Holm
correction in two separate 2-test families (one per outcome); leave-one-
seed-out sensitivity on all 4 contrasts + 2 interactions, no seed ever
excluded from the primary n=12; task/safety (completion, collision — mean_U/
timeout unavailable for Cells 1/3) reported descriptively only, no p-values,
no reuse of the old WSC non-inferiority margins.

# Files created

- `scripts/analyze_dws_formal.py` — the analysis pipeline (read-only,
  deterministic, self-test mode, fails loudly).
- `reports/DWS_STATISTICAL_PROTOCOL_V1.md` — frozen protocol.
- `reports/DWS_STATISTICAL_ANALYSIS_README.md` — implementation/rerun guide.
- `reports/DWS_STATISTICAL_IMPLEMENTATION_STATUS.md` — this file.
- **Not yet created** (require Cell 2): `outputs/dws_statistical_analysis/dws_seed_level_metrics.csv`,
  `dws_primary_contrasts.csv`, `dws_primary_summary.csv`,
  `dws_interaction_summary.csv`, `dws_leave_one_seed_out.csv`,
  `dws_task_safety_descriptive.csv`. The script does not write any of these
  until all four cells load successfully — confirmed empty
  `outputs/dws_statistical_analysis/` directory (does not even exist yet)
  after the blocked real run.

# Validation performed

`python scripts/analyze_dws_formal.py --self-test` — all 9 Section-12 checks
PASS: paired-difference calculation on synthetic data; sign flip on cell-
label swap; U_min favourable-positive convention; Gini favourable-negative
convention; Holm correction against the known `[0.01, 0.04] -> [0.02, 0.04]`
pair; bootstrap reproducibility under RNG seed 0 (two independent runs
produced bit-identical results); hard failure on a missing seed; interaction
algebraically equals WSC-effect minus Original-effect (both outcomes).

Additionally verified (Section 9 metric-definition identity, real data, not
synthetic): `thesis/study_b/utility.py` is byte-identical between this
bundle and the source repo that produced the Cell 1/3 CSV; both evaluation
pipelines use the same H1 bank, the same 256-episode count, the same
`ensemble_window_for_stage_end(2_000_000)` rule, and the same seed-level
`U_min`/`Gini` averaging convention (verified directly in
`wsc_v2_formal_analysis.py::_seed_level_from_rows`, replicated exactly in
this pipeline). **No discrepancy found — no deviation from the literal
protocol was necessary.**

Real end-to-end run attempted against actual files (no arguments): Cell 1,
3, and 4 loaded and validated successfully (12/12 seeds each, no NaN/non-
finite values, no seed-set mismatch); Cell 2 CSV does not exist yet ->
script printed `WAITING FOR CELL 2`, exited non-zero, wrote zero output
files, as required.

# Input-data status by cell

| Cell | Status | Source |
|---|---|---|
| 1 (Maximin) | Available, loaded, verified | `wsc_v2_formal_seed_level.csv` (F: drive) |
| 2 (Maximin+DWS) | **NOT AVAILABLE** — training incomplete (10/12 seeds at 2,000,000; 920105/920106 at 1,700,000) | n/a until training finishes + evaluation is run |
| 3 (Maximin+WSC) | Available, loaded, verified | `wsc_v2_formal_seed_level.csv` (F: drive) |
| 4 (Maximin+WSC+DWS) | Available, loaded, verified, all 12 seeds, 3072/3072 episode rows | `dense_interim_evaluation_12seed_full.csv` |

# Remaining blocker

Cell 2 (Maximin + DWS) formal training has not reached step 2,000,000 for
all 12 seeds (920105/920106 remain, ~300,000 steps short as of this report).
No formal DWS evaluation for Cell 2 exists yet.

# Exact command to run the final analysis after Cell 2 evaluation is complete

```powershell
cd "C:\dense reward"

# 1. Confirm all 12 seeds reached 2,000,000 (all four ensemble-window
#    checkpoints — 1,850,000/1,900,000/1,950,000/2,000,000 — must exist
#    per seed under checkpoints\maximin_dense\).

# 2. Run the Cell 2 evaluation (produces the CSV analyze_dws_formal.py expects by default):
.venv\Scripts\python.exe scripts\evaluate_dense_interim.py --run-tag maximin_dense --condition maximin --no-include-welfare-state --seeds 900101 900102 900103 900104 910101 910102 920101 920102 920103 920104 920105 920106 --out-suffix maximin_dense_12seed_full

# 3. Run the frozen statistical analysis:
.venv\Scripts\python.exe scripts\analyze_dws_formal.py
```
