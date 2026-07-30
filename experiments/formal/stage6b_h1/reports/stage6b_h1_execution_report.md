# Stage 6B-H1 Execution Report

## 35.1 Executive status

**PASS**

- utility trajectory fix implemented and executed
- 480 evaluation episodes reconstructed
- 30 checkpoint hashes unchanged
- non-utility mismatches = 0
- acceptance reference checks passed
- old Stage 6B outputs not overwritten
- no thesis text files modified

## 35.2 Files changed

Code / tests / H1 package (not dissertation text):

- `src/thesis/analysis/episode_utility_accumulator.py`
- `src/thesis/analysis/reconstruct_eval.py`
- `experiments/formal/stage6b_h1/**`
- `tests/analysis/test_episode_utility_accumulator.py`
- `tests/analysis/test_stage6b_h1_*.py`

**No thesis or dissertation text files were modified.**

## 35.3 Root cause

Final-state stakeholder experience `E_i(s_T)` was used as episode utility,
rather than the mean of active-state trajectory attainment.

## 35.4 Code correction

- accumulator over active on-road states
- sample `s0` after reset; sample post-step states only while continuing
- exit/absorbing states excluded
- collision override only for stakeholders in collision pairs
- empty non-colliding sample raises `RuntimeError`
- utility-derived fields recomputed from corrected utilities

## 35.5 Test results

- `tests/analysis/test_episode_utility_accumulator.py` + H1 unit tests: **24 passed**
- related `tests/rewards` + `tests/envs` + `tests/audits`: **139 passed**
- post-run regression: episode count / acceptance checks verified

## 35.6 Evaluation results

- checkpoints: **30**
- evaluation episodes: **480**
- nonutility mismatches: **0**
- checkpoint hashes unchanged: **True**
- mean utility changed episodes: **480**
- minimum utility changed episodes: **477**
- worst-off identity changed episodes: **116**

## 35.7 Primary corrected values (condition means)

Mean stakeholder utility:

- Baseline: **0.605213**
- Mean-PBRS: **0.527772**
- Min-PBRS: **0.586206**

Minimum stakeholder utility:

- Baseline: **0.269496**
- Mean-PBRS: **0.151500**
- Min-PBRS: **0.287960**

Success / collision unchanged vs old Stage 6B:

- Success: 0.350 / 0.16875 / 0.3125
- Collision: 0.04375 / 0.100 / 0.0625

## 35.8 Controller-swap estimability

- Baseline: **4/10**
- Mean-PBRS: **0/10**
- Min-PBRS: **4/10**

Missing values remain NA (not zero-filled).

## 35.9 Unresolved design limitations

- equal coefficient is not RMS matching
- low task success
- controller-swap missingness
- non-inferiority not implemented
- one endpoint is not a learning curve

## 35.10 Reproducibility

- branch: `formal/analysis-100k-utility-corrected`
- stage6a root: `final_new_results_100k/.../stage6a_20260730T094829Z_a89256db_44d5e647`
- old stage6b root: `final_new_analysis_100k/.../stage6b_analysis_100k`
- output: `experiments/formal/stage6b_h1/output`
- manifest: `output/manifests/analysis_manifest.json`
- paper reminder: `reports/PAPER_CHANGES_REQUIRED_LATER.md`
