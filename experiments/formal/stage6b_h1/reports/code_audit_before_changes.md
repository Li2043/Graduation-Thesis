# Code audit before Stage 6B-H1 changes

## 1. Stage 6B entry script
`experiments/formal/stage6b_analysis_100k/scripts/run_stage6b_analysis.py`

## 2. Utility calculation function (pre-fix)
`src/thesis/analysis/reconstruct_eval.py` → `run_instrumented_evaluation_episode`
used `potential_state_from_v3_vehicles(info["vehicles_t1"], ...)` then
`compute_stakeholder_experiences` then `episode_stakeholder_utilities`.

## 3. Utility data source (pre-fix)
Final post-episode `info["vehicles_t1"]` experience values `E_i(s_T)`, including
completed/absorbing stakeholders mapped to experience `1.0`.

## 4. Final-state experience read?
Yes (pre-fix). This is the root defect addressed by H1.

## 5. Evaluation episodes
480 (= 30 checkpoints × 16 validation episodes).

## 6. Checkpoint count
30 (`final_online_target_weights.pt` under Stage 6A jobs).

## 7. Statistical unit
Formal training master seed (paired across conditions).

## 8. Output inventory (old Stage 6B)
Processed CSVs under `data/processed/<analysis_id>/`, tables, figures named
`*_learning_curve.*`, reports, `analysis_manifest.json`.

## 9. Manifest generation order (old)
Outputs written during run; manifest hashed afterward. Runner log mutation after
hashing was a known risk; H1 separates post-manifest verification log.

## 10. Learning-curve misnaming?
Yes: figures named `success_learning_curve`, etc., despite only the 100000-step
endpoint being available; `learning_curve_auc.csv` all-null.

## 11. Controller-swap formal computation?
Not in old Stage 6B primary package. H1 adds strict estimability diagnostics.

## 12. Requirements reproducibility?
Old package recorded library versions, but requirements text included ambiguous
entries. H1 writes frozen `analysis_requirements_h1.txt` + `pip_freeze.txt`.

## 13. Functions affected by this fix
- `run_instrumented_evaluation_episode`
- new `episode_utility_accumulator` module
- Stage 6B-H1 aggregation / contrast / convention / swap diagnostics
- figure naming for endpoint-only plots

## Audit materials note
Named audit package files (`100k_experiment_data_audit_and_fix_plan.md`,
`corrected_*.csv`, patch) were not present in the workspace at start; H1 uses
the locked mathematical definition and §19 numeric acceptance references.
