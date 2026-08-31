# Conflict-Exposure Diagnostic -- Reproducibility Log

## Files read (audit, section A)
- `project/src/thesis/study_b/envs/scenario_adapter.py`
- `project/src/thesis/study_b/envs/highwayenv_merge.py`
- `project/src/thesis/study_b/envs/highwayenv_wrapper.py`
- `project/src/thesis/study_b/envs/highwayenv_action.py`
- `project/src/thesis/study_b/envs/highwayenv_vehicle.py`
- `project/src/thesis/study_b/utility.py`
- `project/src/thesis/study_b/training_common.py`
- `project/src/thesis/study_b/q_ensemble.py`
- `project/experiments/pilots/study_b_fairness_mappo/scripts/evaluate_formal_welfare.py`
- `scripts/evaluate_behavioral_window.py` (F:\正式训练)
- `F:\正式训练\scenario_banks\H1.json` (all 256 scenarios inspected programmatically for role/ttc_slot consistency)

## Scripts created (additive only; no frozen file modified)
- `analysis_scripts/ch5_baseline/conflict_exposure_diagnostic_eval.py`
  -- new evaluation script, re-runs the 12 Baseline seeds x 256 H1
  scenarios through the existing checkpoint-Q ensemble, records full
  per-step (x,y,speed,acceleration,action) for all 4 vehicles.
- `analysis_scripts/ch5_baseline/conflict_exposure_analysis.py`
  -- statistical analysis (G1-G4, H, I) and figures A-F.

## Commands executed
```
F:\正式训练\.venv\Scripts\python.exe conflict_exposure_diagnostic_eval.py --shard-index 0 --num-shards 12   (smoke test, seed 900101)
F:\正式训练\.venv\Scripts\python.exe conflict_exposure_diagnostic_eval.py --shard-index {1..11} --num-shards 12   (parallel, remaining 11 seeds)
F:\正式训练\.venv\Scripts\python.exe conflict_exposure_analysis.py
```

## Exact seed list
900101, 900102, 900103, 900104, 910101, 910102, 920101, 920102, 920103, 920104, 920105, 920106 (n=12, Baseline condition only)

## Exact scenario bank
`F:\正式训练\scenario_banks\H1.json`, 256 scenarios, sha256-verified
identical to the copy under `F:\正式训练_seed_replication_v1\scenario_banks\H1.json`
in an earlier diagnostic this session. No new bank generated.

## Checkpoint ensemble used
`taskonly_arm` checkpoints, stage name `Formal_taskonly`, window
K(2,000,000) = {1,850,000; 1,900,000; 1,950,000; 2,000,000} -- identical
to the Chapter 5 formal Baseline result. epsilon_eval = 0 throughout.

## Outputs generated
- `outputs/conflict_exposure/conflict_exposure_shard{0..11}.csv` (raw, per shard)
- `outputs/conflict_exposure/conflict_exposure_episode_level.csv` (merged, 3072 rows)
- `outputs/conflict_exposure/conflict_exposure_seed_summary.csv` (12 rows)
- `outputs/conflict_exposure/conflict_exposure_g4_correlations.csv`
- `outputs/conflict_exposure/conflict_exposure_report_log.txt` (full console log)
- `outputs/conflict_exposure/conflict_exposure_report.md`
- `D:\obsidian\research\2. 论文写作\final thesis\figures\conflict_exposure_A_overlap_proportion.png`
- `.../conflict_exposure_B_crossing_gap_dist.png`
- `.../conflict_exposure_C_Utility_Gini_overlap_vs_no.png`
- `.../conflict_exposure_D_U_min_overlap_vs_no.png`
- `.../conflict_exposure_E_intensity_vs_gini.png`
- `.../conflict_exposure_F_decomposition.png`

No frozen training/environment/evaluation source file was modified. No
thesis text file was modified as part of this diagnostic.
