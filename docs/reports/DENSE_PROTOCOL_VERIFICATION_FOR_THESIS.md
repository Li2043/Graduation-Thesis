# Dense Welfare Shaping — Thesis Protocol Verification

Scope note: this report covers the four-cell **Maximin** design named in the audit prompts
(Original / +DWS / +WSC / +WSC+DWS). A separate GGI branch of the same Dense Reward Study
also exists (`ggi_wsc_dense`, `ggi_dense` under `F:\dense reward new\checkpoints\`, both
with all 12 seeds reaching `ckpt_step_2000000.pt` as of this audit) — it is **not** part of
the four-cell matrix below and is not further analyzed here.

## 1. Executive status

| Topic | Status | Key conclusion |
|---|---|---|
| Source checkpoint (1.2M C64) | VERIFIED PROTOCOL + VERIFIED FROM FORMAL RUN | All four cells share the identical C64 1,200,000-step checkpoint per seed; confirmed both in launcher code and in actual `--resume-from` paths used by completed/running processes. |
| DWS continuation budget (800,000 steps → final step 2,000,000) | VERIFIED PROTOCOL + VERIFIED FROM FORMAL RUN (cell 4 only) | Hardcoded constants in launcher; cell 4 (Maximin+WSC+DWS) formally completed at step 2,000,000 for all 12 seeds. Cell 2 (Maximin+DWS) is **still running**, not yet complete. |
| Checkpoint interval / final-four window | VERIFIED PROTOCOL + VERIFIED FROM FORMAL RUN (cell 4) | `checkpoint_every=50_000`; window function shared code-level with Original/WSC evaluation. |
| DWS frozen shaping parameters (mode/magnitude/epsilon/scope/cohort) | VERIFIED PROTOCOL | All five values traced to `configs/dense_reward_protocol_v1.json` with documented, outcome-independent derivation. |
| Inherited vs. changed training hyperparameters | VERIFIED PROTOCOL, with one CONFLICT-adjacent nuance | Core architecture/optimizer/γ/batch are code-shared (identical factory call) so cannot diverge; `--eps-decay-steps-absolute`/`--lr-decay-steps-absolute` are **not explicitly passed** by the DWS launcher (unlike the Original/WSC launcher, which reads them from the frozen config each time) — they currently resolve to the same numeric values only because the script defaults happen to equal the frozen config values. See Section 2D. |
| Held-out evaluation bank / scenario identity | VERIFIED FROM FORMAL RUN | H1 bank, byte-identical (SHA-256) across all three bundle roots used by the four cells; 256 scenarios. |
| Evaluation policy / checkpoint-selection rule | VERIFIED PROTOCOL + VERIFIED FROM FORMAL RUN (cell 4) | Greedy argmax over an equal-weight 4-checkpoint ensemble; identical `ensemble_window_for_stage_end()` function used by cell-3/cell-4 evaluation code. |
| Available fairness/task/safety metrics for DWS | VERIFIED FROM FORMAL RUN (cell 4) | U_min, mean_U, Gini, GGI, completion, collision, timeout, C_max/C_mean all present in the actual output CSV. |
| Behavioural/coordination metrics for DWS | NOT VERIFIED | No DWS-specific behavioural (yield-rate / priority / burden-transfer) script or output found anywhere in this bundle. |
| Statistical/inferential protocol (contrasts, bootstrap, Holm, sensitivity, margins) | NOT VERIFIED / NOT SPECIFIED | No DWS-specific analysis script, bootstrap output, or written statistical plan exists anywhere searched. |
| Cell 1 / Cell 3 baseline reachability | VERIFIED FROM FORMAL RUN (on a different drive, not in this bundle) | Reachable at `F:\正式训练_seed_replication_v1\...`, not copied into the DWS bundle itself. |
| Manifest `condition` field reliability | CONFLICT | `write_run_manifest(..., condition="maximin", ...)` is hardcoded in `launch_dense_priority.py`'s formal-launch loop regardless of the actual `--priority`/`--condition` used — run-state JSON for GGI runs also reads `"condition": "maximin"`. Manifests must NOT be used as evidence of scientific condition; the actual `--condition` CLI argument (visible in `--dry-run` output or process cmdline, not the manifest) is authoritative. |

## 2. Formal training protocol

### 2A. Source checkpoint

**VERIFIED PROTOCOL + VERIFIED FROM FORMAL RUN.**

- `scripts/launch_dense_priority.py` line 46: `C64_SOURCE_STEP = 1_200_000`.
- `_source_checkpoint()` (lines 55-59): `checkpoints/formal_init/<seed>/C64_R50/seed_<seed>_C64_R50/ckpt_step_1200000.pt`.
- This exact path pattern is used for **both** cell 2 and cell 4 (the `--resume-from` argument at initial launch is built from this function for every priority 1-4); `--continue-from-latest` (used mid-run) instead resumes from the seed's own latest dense checkpoint, not from C64 a second time — confirmed in `build_plan()` lines 127-134.
- Formal-run evidence: `outputs/run_state/maximin_wsc_dense_900101.json` records `"init_checkpoint": "...\\checkpoints\\maximin_wsc_dense\\...\\ckpt_step_1700000.pt"` for a *resumed* segment (not the original C64 launch); the original C64-anchored launch for this bundle's Priority 1/2 runs was verified via `--dry-run` output earlier in this engagement showing `--resume-from ...\formal_init\<seed>\C64_R50\...\ckpt_step_1200000.pt` for every one of the 12 seeds, and `logs/maximin_dense_900101.log` (Priority 2) contains `"resumed from ... (checkpoint step 1200000)"` at its first launch.

### 2B. Formal continuation budget

**VERIFIED PROTOCOL (both cells) + VERIFIED FROM FORMAL RUN (cell 4 only; cell 2 in progress).**

- `FORMAL_BUDGET_END_STEP = 2_000_000` (line 47). `max_additional_steps = FORMAL_BUDGET_END_STEP - start_step` (line 138) — for a fresh C64 launch this is `2,000,000 - 1,200,000 = 800,000`, identical for both DWS branches (cell 2 and cell 4). This is **not** assumed from Original/WSC's 800,000 — it is independently derived from the same two hardcoded constants that also define the DWS source step, and both numbers were read directly from this file.
- **Cell 4 (Maximin+WSC+DWS)**: VERIFIED FROM FORMAL RUN — all 12 seeds' logs (`logs/maximin_wsc_dense_<seed>.log`) end with a line of the shape `{"final_step": 2000000, "elapsed_seconds": ..., "manifest": ...}` and `outputs/run_state/maximin_wsc_dense_<seed>.json` shows `"completed": true, "technical_failure": false, "returncode": 0` for all 12.
- **Cell 2 (Maximin+DWS)**: NOT YET COMPLETE as of this audit. Actual checkpoint inspection (`checkpoints/maximin_dense/maximin_dense_<seed>/seed_<seed>_Dense_maximin_dense/ckpt_step_*.pt`):
  - `900101,900102,900103,900104,910101,910102,920101,920102,920103,920104` → latest checkpoint `1,900,000` (95% of budget).
  - `920105,920106` → latest checkpoint `1,250,000` (~6% of budget; these two seeds were restarted from C64 after an earlier machine-reboot interruption — see Section 5).
  - `psutil` process inspection at the time of this audit found 12 live `train_curriculum_stage_highwayenv.py` (non-`_wsc`) processes with growing CPU time — training is actively in progress, not stalled.

### 2C. Checkpoint-saving schedule

**VERIFIED PROTOCOL + VERIFIED FROM FORMAL RUN (cell 4).**

- `checkpoint_every = 50_000` for a formal (non-smoke) launch (line 117: `checkpoint_every = smoke_steps if live_smoke_test else 50_000`).
- Final-four-checkpoint window: `ensemble_window_for_stage_end(final_step)` in `project/src/thesis/study_b/q_ensemble.py` lines 55-58: `K(S) = {S-150_000, S-100_000, S-50_000, S}` — for `S=2,000,000` this is `{1,850,000, 1,900,000, 1,950,000, 2,000,000}`. This is the same function object imported and called unchanged by `evaluate_dense_interim.py`.
- Structural availability: for cell 4, all four window checkpoints exist for all 12 seeds (verified in this session's own evaluation run, which loaded all 48 window-checkpoint files without a `find_latest_checkpoint`/load failure). For cell 2, the window is **not yet fully available** for 920105/920106 (currently only at step 1,250,000, i.e. below even the window's lowest step 1,850,000); it is available for the other 10 seeds (already past 1,850,000).

### 2D. Training schedule and inherited hyperparameters

| Setting | Original/WSC value | DWS value | Same or changed | Evidence |
|---|---|---|---|---|
| Network architecture / obs_dim / n_actions / optimizer / γ / batch_size / replay capacity / target-sync interval / target mode | `FROZEN_EXPERIMENT_CONFIG.json` `dqn` block (hidden_sizes [64,64], Adam, γ=0.995, batch 64, replay 100000, target_sync_interval_updates 250, DOUBLE) | Same | **Same — code-shared, cannot diverge** | Both `train_curriculum_stage_highwayenv.py` (line 188: `build_study_b_dqn_config(reward_condition="baseline", device=args.device)`) and `train_curriculum_stage_highwayenv_wsc.py` (line 213: `build_study_b_dqn_config(reward_condition="baseline", device=args.device, obs_dim=WSC_OBS_DIM)`) call the identical shared factory `thesis.study_b.shared_local_dqn.build_study_b_dqn_config`. Neither the DWS launcher nor the CLI exposes any flag that would let these differ. |
| Replay warmup | `512` (`launch_formal.py` line 109: `"--replay-warmup", "512"` — hardcoded literal, not read from config) | `512` (`launch_dense_priority.py` line 153: `"--replay-warmup", str(replay_warmup)` where `replay_warmup = 512` for a formal, non-smoke run) | Same | Both hardcode the identical literal `512`; neither reads it from `FROZEN_EXPERIMENT_CONFIG.json`. |
| Welfare-lambda (`lambda_W`) | `cfg["welfare"]["lambda_W"]` read live from `FROZEN_EXPERIMENT_CONFIG.json` at launch time (`launch_formal.py` line 112) | Hardcoded literal `WELFARE_LAMBDA_MAXIMIN = 0.5` in `launch_dense_priority.py` (line 52) | Same **value** (0.5==0.5, confirmed: `FROZEN_EXPERIMENT_CONFIG.json`'s `welfare.lambda_W` is `0.5`), different **mechanism** | `launch_dense_priority.py` line 52 comment explicitly documents this: "Despite the name, this is FROZEN_EXPERIMENT_CONFIG.json's single, condition-agnostic welfare.lambda_W (0.5)". If the frozen config's `lambda_W` were ever edited, Original/WSC launches would pick up the change automatically and DWS launches would silently not. |
| `eps-decay-steps-absolute` | `cfg["dqn"]["eps_decay_steps_absolute"]` read live and **explicitly passed** (`launch_formal.py` line 110) = `640000` | **Not passed at all** by `launch_dense_priority.py` — the training script's own argparse default (`train_curriculum_stage_highwayenv.py` line 131 / `_wsc.py` line 161: `default=640_000`) is used instead | Same **numeric value currently** (640000==640000), different **mechanism — this is the one real, currently-latent divergence risk** | Confirmed by diffing the two launchers' CLI-construction code directly; no `--eps-decay-steps-absolute` token appears anywhere in `launch_dense_priority.py`'s `cmd` list. |
| `lr-decay-steps-absolute` | `cfg["dqn"]["lr_decay_steps_absolute"]` read live and explicitly passed (`launch_formal.py` line 111) = `800000` | Not passed; script default `default=800_000` used instead | Same numeric value currently, different mechanism (same caveat as above) | Same evidence method as the eps-decay row. |
| Dense shaping (mode/magnitude/epsilon) | N/A (Original/WSC have no dense shaping at all) | `--dense-welfare-shaping --dense-shaping-mode discrete --dense-shaping-magnitude 0.0005 --dense-shaping-epsilon 1e-06`, all three read live from `configs/dense_reward_protocol_v1.json` | **New, DWS-only** | `launch_dense_priority.py` lines 156-159; `dense_reward_protocol_v1.json` keys `dense_shaping_mode`, `dense_shaping_magnitude`, `dense_shaping_epsilon`. |
| Scenario distribution (training-time) | `scenario_bank=Q.json`, all 64 scenario IDs (`Q_00000`...`Q_00063`) | Identical — same `_scenario_ids()` helper reading the same `scenario_banks/Q.json` | Same | `launch_dense_priority.py` lines 62-64; `--scenario-bank`/`--scenario-ids` construction is identical text between the Priority-1/2/3/4 dry-run outputs observed this session. |

**CONFLICT flag (minor, currently inert)**: the eps-decay/lr-decay rows above are the one place where DWS and Original/WSC reach the *same number* through *different code paths* (one reads a config file live, the other relies on a script default that happens to match today). This is not a numeric discrepancy as of this audit, but it is not a single source of truth either — a future edit to `FROZEN_EXPERIMENT_CONFIG.json`'s `dqn.eps_decay_steps_absolute` or `dqn.lr_decay_steps_absolute` would silently desynchronize DWS from Original/WSC. Report as a latent risk, not an active conflict.

### 2E. Number of formal seeds and branches

**VERIFIED PROTOCOL + VERIFIED FROM FORMAL RUN (partial).**

- `FORMAL_SEEDS` (lines 42-45): `900101, 900102, 900103, 900104, 910101, 910102, 920101, 920102, 920103, 920104, 920105, 920106` — 12 seeds, identical tuple used by both DWS branches (and by the GGI branch, out of scope here).
- Both DWS branches (cell 2, cell 4) are implemented and have been formally launched for all 12 seeds (`build_plan()` defaults to `FORMAL_SEEDS` when no `--seeds` filter is given, and both `--priority 1` and `--priority 2` dry-runs produced exactly 12 rows in this session).
- Total DWS formal runs implied by the frozen protocol for the Maximin four-cell design: **2 DWS branches × 12 seeds = 24 new formal training runs** (cells 2 and 4; cells 1 and 3 are the pre-existing, already-completed Original/WSC campaign, not new runs).
- Cell 4: 12/12 formally complete. Cell 2: 12/12 launched, 10/12 past step 1,900,000, 2/12 restarted from C64 after an interruption and currently at step 1,250,000 (see Section 5) — **not** formally complete as of this audit.

### 2F. Formal-run evidence — summary

- Cell 4 (Maximin+WSC+DWS): **VERIFIED FROM FORMAL RUN.** All 12 seeds' `logs/maximin_wsc_dense_<seed>.log` end in a `final_step: 2000000` JSON line; `outputs/run_state/maximin_wsc_dense_<seed>.json` shows `completed:true, technical_failure:false, returncode:0` for all 12; independent checkpoint-file listing confirms `ckpt_step_2000000.pt` present for all 12 seeds.
- Cell 2 (Maximin+DWS): **PARTIALLY VERIFIED FROM FORMAL RUN, launch still in progress.** Do not treat this cell as complete for thesis purposes; re-check checkpoint files before citing final-step values.
- Cells 1/3 (Original/WSC Maximin): **VERIFIED FROM FORMAL RUN, but on a different drive**, not part of this DWS bundle. Cell 1: `F:\正式训练_seed_replication_v1\checkpoints\formal_runs\maximin_<seed>\` and `analysis_scripts\pooled12\outputs\pooled12_rq1_seed_level_metrics.csv`. Cell 3: `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\` and `analysis_scripts\wsc_v2_formal\outputs\wsc_v2_formal_seed_level.csv`.

### 2G. Dense-specific frozen parameters

All from `configs/dense_reward_protocol_v1.json` (authoritative machine-readable source; `reports/dense_reward_protocol_v1.md` is the human-readable record of the same freeze):

| Parameter | Value | Key |
|---|---|---|
| Shaping mode | `discrete` | `dense_shaping_mode` |
| Shaping magnitude (c) | `0.0005` | `dense_shaping_magnitude` |
| Epsilon / dead-zone | `1e-06` | `dense_shaping_epsilon` |
| Welfare objective | `maximin` | `primary_objective` |
| Shaping scope | `shared_global` (one shaping term added to every controlled vehicle, not per-agent) | `shaping_scope` |
| Fixed cohort rule | `fixed_four_vehicle` | `cohort` |
| Welfare source | `running_active_attainment` | `welfare_source` |

An earlier provisional `c=0.1` is explicitly recorded as revoked and **never used in any formal or Priority 1 training run** (`_provenance` key and `magnitude_recalibration.previous_c_never_used_in_training: true`).

## 3. Formal evaluation protocol

### 3A. Held-out scenario bank

**VERIFIED FROM FORMAL RUN.**

- `evaluate_dense_interim.py` line: `load_scenario_bank(BUNDLE_ROOT / "scenario_banks" / f"{bank_name}.json")` with `bank_name` defaulting to `"H1"` in `run_one()`'s signature.
- File: `scenario_banks/H1.json`, confirmed **256 scenarios** (`json.load(...)` length check performed directly this session).
- Scenario identity across all four cells: SHA-256 of `H1.json` is **byte-identical** across `C:\dense reward`, `F:\正式训练`, and `F:\正式训练_seed_replication_v1` (all three hash to `8345ba0d1141af223ea8629caa627889da50afaf9b5205de63d5cf89ffdcf5c8`). Since cell 1's evaluation and cell 3's evaluation (`evaluate_wsc_formal_v2.py`) read `H1.json` from `F:\正式训练\...` / read it via the `F:\正式训练_seed_replication_v1` bundle, and cells 2/4's evaluation reads it from `C:\dense reward\scenario_banks\H1.json`, all four cells use the identical scenario set.

### 3B. Evaluation action selection

**VERIFIED FROM FORMAL RUN / VERIFIED PROTOCOL.**

- `q_ensemble.py` `select_ensemble_action()`: `return int(np.argmax(q_ensemble_values(agents, obs)))` — pure greedy argmax, no epsilon parameter exists anywhere in this function or its caller `select_ensemble_actions()`. There is no `--eval-epsilon` or similar CLI flag in `evaluate_dense_interim.py`. Evaluation epsilon is **implicitly and unconditionally 0** — it is not a configurable value that could differ between cells, since the same `select_ensemble_action` function is called for the DWS evaluation and (per the shared-code-path evidence in `evaluate_wsc_formal_v2.py`, which imports the identical function) for the Original/WSC evaluation.

### 3C. Checkpoint selection

**VERIFIED PROTOCOL + VERIFIED FROM FORMAL RUN (cell 4).**

- Equal-weight final-four Q-value ensemble, not a single checkpoint: `load_ensemble_agents(..., expected_steps=WINDOW, ...)` where `WINDOW = ensemble_window_for_stage_end(2_000_000) = (1850000, 1900000, 1950000, 2000000)`.
- This is the **exact same function** (`thesis.study_b.q_ensemble.ensemble_window_for_stage_end`) imported unchanged in `evaluate_dense_interim.py`, in the Priority-1-era `evaluate_formal_welfare.py`, and (per the Priority-3 results report's own methodology section, itself citing `evaluate_wsc_formal_v2.py`) in the WSC-only evaluation — so cell 2, cell 3, and cell 4 all use an identical checkpoint-selection rule by construction, not merely by convention.

### 3D. Matching and pairing

**VERIFIED FROM FORMAL RUN (cell 4) / VERIFIED PROTOCOL (cells 2/3 by shared code).**

- `evaluate_dense_interim.py`'s `run_one(seed, ...)` is called once per seed in `FORMAL_SEEDS`, and `env.reset(seed=0, scenario=scenario)` uses a fixed `seed=0` for environment/scenario determinism (the *master* training seed only selects which trained agent/checkpoints are loaded; it does not reseed the held-out evaluation scenarios) — this means every seed's held-out evaluation walks through the identical 256 scenarios in the identical order, matched by scenario_id in the output CSV's `scenario_id` column.
- Cross-cell pairing is by training seed: cell 4's results CSV (`dense_interim_evaluation_12seed_full.csv`) and cell 3's baseline CSV (`wsc_v2_formal_seed_level.csv`) both key rows by the same 12 seed identities, confirmed used together in `reports/priority1_maximin_wsc_dense_results.md`'s per-seed delta table.

### 3E. Evaluation metrics

**VERIFIED FROM FORMAL RUN (cell 4); VERIFIED PROTOCOL (cell 2 — same script, not yet run to completion, so no cell-2 CSV exists yet).**

| Metric | Available for DWS? | Exact output field/file | Evidence |
|---|---|---|---|
| Worst-off utility U_min | Yes | `min_U` column | `outputs/welfare_analysis/dense_interim_evaluation_12seed_full.csv` header row. |
| Utility Gini | Yes | `gini` column | Same file. |
| Mean utility | Yes | `mean_U` column | Same file. |
| Completion | Yes | `completion` (0/1 per episode) | Same file. |
| Collision | Yes | `collision` (0/1 per episode) | Same file. |
| Timeout | Yes | `timeout` (0/1 per episode) | Same file. |
| GGI | Yes (present but not a Maximin-branch primary outcome) | `ggi` column | Same file — computed for every row regardless of the training condition's own welfare objective. |
| Burden (`C_max`/`C_mean`) | Yes | `C_max`, `C_mean` columns | Same file. |

Behavioural/coordination metrics (yield-rate `RY`, `P_priority_worse`, burden-transfer `BC`, `GapClosure_k25` — the metrics used in the existing WSC v2 behavioural analysis): **NOT VERIFIED.** No script in this bundle computes these for any DWS cell; `evaluate_dense_interim.py` does not produce merge-order or hard-brake fields at all (unlike `evaluate_formal_behavioral.py`, which is Original/WSC-only and is not invoked anywhere for DWS checkpoints).

### 3F. Actual evaluation evidence

- Cell 4: **VERIFIED FROM COMPLETED EVALUATION OUTPUT** — `outputs/welfare_analysis/dense_interim_evaluation_12seed_full.csv`, 3,072 rows (12 seeds × 256 episodes, row count confirmed by direct line count), all four ensemble-window checkpoints present for all 12 seeds at evaluation time.
- Cell 2: **NOT YET EVALUATED** — no `dense_interim_evaluation_*maximin_dense*` (or equivalently-named) output file exists in `outputs/welfare_analysis/` as of this audit, consistent with cell 2 training not yet being complete (the ensemble window is not yet fully available for 2 of the 12 seeds).

## 4. Statistical analysis protocol

**NOT VERIFIED / NOT SPECIFIED for every item below.** No DWS-specific statistical/inferential script, bootstrap output, or written analysis plan was found anywhere in this bundle. Searches performed: case-insensitive grep for `bootstrap` across `scripts/` and `project/` (matches found are all pre-existing, non-DWS modules — `thesis/study_b/analysis/bootstrap.py` is the general Study B bootstrap utility, imported only by the pre-existing `run_analysis.py`, which is not invoked anywhere for any `maximin_dense`/`maximin_wsc_dense`/`ggi_*_dense` output); filename search for `*dense*bootstrap*`, `*dense*analysis*`, `*dws*` (no matches beyond the training/evaluation scripts already covered in Sections 2-3).

| Item | Verified protocol | Evidence source | Evidence |
|---|---|---|---|
| Inferential unit | NOT SPECIFIED | — | No DWS analysis code exists to specify this. |
| Episode-level aggregation before inference | NOT SPECIFIED (but the descriptive report that does exist aggregates this way) | `reports/priority1_maximin_wsc_dense_results.md` §5 | That report computes "seed-level `U_min`/`Gini` = mean of the per-episode `min_U`/`gini` columns" as a descriptive convention, explicitly labeled non-bootstrap — this is a precedent, not a frozen inferential protocol. |
| Original-DWS effect, WSC-DWS effect formulas | NOT SPECIFIED as a formal contrast; an ad hoc simple-difference version exists | `reports/priority1_maximin_wsc_dense_results.md` §5-6 | The report computes `Δ = Dense_value − WSC_only_value` per seed and its arithmetic mean across seeds, explicitly caveated as "a simple arithmetic mean of the 12 per-seed deltas, not bootstrap." No `Y(Maximin+DWS) − Y(Maximin)` (Original-DWS effect) computation exists anywhere — the only baseline compared against so far is the WSC-only baseline (cell 3), not the Original baseline (cell 1). |
| DWS × WSC interaction formula | NOT VERIFIED | — | No interaction contrast has been computed for DWS anywhere in this bundle. |
| Primary vs. secondary outcomes | NOT SPECIFIED | — | No document declares U_min/Gini as confirmatory vs. completion/collision/timeout as descriptive for DWS specifically (the existing WSC v2 formal report does this for the WSC-only comparison, but that is a different, already-completed analysis, not a DWS-specific declaration). |
| Bootstrap resamples / CI level / RNG seed | NOT VERIFIED | — | No bootstrap has been run for DWS. |
| p-values | NOT VERIFIED / NOT SPECIFIED | — | — |
| Multiplicity correction | NOT VERIFIED / NOT SPECIFIED | — | — |
| Sensitivity analysis (leave-one-seed-out, n=11, etc.) | NOT VERIFIED / NOT SPECIFIED | — | The existing WSC v2 formal analysis has a seed-910102 sensitivity check (`wsc_v2_formal_sensitivity_910102.csv`), but nothing establishes this automatically applies to DWS, and no DWS-specific sensitivity file exists. |
| Task/safety non-inferiority margins (completion −0.05 / collision +0.03) | NOT VERIFIED for DWS | — | These margins were not found reused, restated, or referenced anywhere in the DWS bundle's config/reports/scripts. |

## Exact contrasts

No exact mathematical contrast formulas are frozen or implemented for DWS beyond the simple per-seed `Δ = cell4_value − cell3_value` already computed descriptively in `reports/priority1_maximin_wsc_dense_results.md`. The Original-DWS effect (`Y(Maximin+DWS) − Y(Maximin)`) and the DWS×WSC interaction contrast named in the audit prompt have **no implementation anywhere in this bundle**.

## 5. Remaining author decisions

These are genuinely unresolved — nothing in the repository specifies them for DWS:

1. **Whether to compute the Original-DWS effect at all** (cell 2 vs. cell 1), and against which cell-1 baseline file (`pooled12_rq1_seed_level_metrics.csv` appears to be the candidate, but this has not been decided or wired up anywhere for DWS).
2. **Whether/how to define and compute the DWS × WSC interaction contrast.**
3. **Whether U_min and Gini are the sole primary/confirmatory DWS outcomes**, with completion/collision/timeout as descriptive-only, mirroring the WSC v2 convention — or a different declaration.
4. **Bootstrap design for DWS**: resample count, CI level/construction, RNG seed, whether seeds are resampled as matched units. No default should be assumed from the WSC v2 procedure without an explicit decision, since the prompt instructs not to copy it silently.
5. **p-value and multiplicity-correction plan for DWS**, if confirmatory inference is intended at all.
6. **Whether the WSC v2 completion/collision non-inferiority margins (−0.05 / +0.03) apply to DWS**, or whether DWS needs its own margins.
7. **Whether a leave-one-seed-out or similar sensitivity check is needed for DWS**, and on which seed(s) — this session's own completed cell-4 analysis flagged large seed-to-seed variance (`priority1_maximin_wsc_dense_results.md` §6) including a training-time instability episode for seed 920106, which is exactly the kind of finding a formal sensitivity analysis is designed to characterize, but no such analysis has been run.
8. **Whether/when to complete cell 2 (Maximin+DWS)** and evaluate it, and whether to also close the currently-latent `eps-decay`/`lr-decay` mechanism gap (Section 2D) by making `launch_dense_priority.py` explicitly pass these from the frozen config, for future robustness.

## 6. Thesis-ready replacement text

**DWS continuation/training paragraph** (facts verified in Section 2 only):

> The Dense Welfare Shaping (DWS) follow-up continues each seed's existing Maximin agent from the same 1,200,000-step C64 checkpoint used by the Original and WSC formal campaigns, for an additional 800,000 environment steps to a final cumulative step of 2,000,000, checkpointed every 50,000 steps. Two DWS branches were run for all 12 formal seeds: Maximin+DWS (Original 18-dimensional observation plus step-wise dense welfare shaping) and Maximin+WSC+DWS (22-dimensional welfare-state-observable input plus the same shaping term). The shaping term is a discrete step-wise bonus/penalty of magnitude c=0.0005 applied whenever the per-step change in the shared Maximin welfare signal exceeds a dead-zone threshold of 1e-6, added identically to every controlled vehicle's reward (shared, not per-agent), computed over a fixed four-vehicle cohort to avoid an artifact from vehicles exiting the scenario. All core network, optimiser, and scheduling hyperparameters (architecture, γ, batch size, replay capacity, target-network update rule, learning-rate and epsilon decay schedules) are unchanged from the Original/WSC Maximin continuation. As of this verification, the Maximin+WSC+DWS branch formally completed for all 12 seeds; the Maximin+DWS (non-WSC) branch had reached step 1,900,000 for 10 of 12 seeds and step 1,250,000 for the remaining two, and was still in progress.

**DWS evaluation paragraph** (facts verified in Section 3 only):

> DWS checkpoints are evaluated on the same held-out H1 scenario bank (256 scenarios) used by the Original and WSC formal evaluations, confirmed byte-identical across bundles. Action selection is fully greedy (argmax over Q-values, no exploration), using the same equal-weight, four-checkpoint ensemble rule as the Original/WSC evaluation — the checkpoints at steps final−150,000, final−100,000, final−50,000, and final, i.e. 1,850,000/1,900,000/1,950,000/2,000,000 for a 2,000,000-step run. Evaluation is matched by training seed and reuses the identical scenario ordering across conditions. For the completed Maximin+WSC+DWS branch, per-episode worst-off utility, mean utility, Utility Gini, generalized Gini, completion, collision, timeout, and per-vehicle burden are all available; the existing behavioural/coordination metrics used in the WSC mechanism analysis (yield rate, priority allocation, burden transfer, worst-off recovery) have not been computed for any DWS condition.

**DWS statistical-analysis paragraph** (facts verified in Section 4 only):

> No dedicated statistical/inferential analysis pipeline exists yet for the Dense Welfare Shaping conditions. A descriptive comparison of the completed Maximin+WSC+DWS branch against the existing Maximin+WSC (no shaping) baseline, all 12 seeds, is available as simple per-seed differences and their arithmetic mean, explicitly not adjusted for multiple comparisons and not accompanied by bootstrap confidence intervals. The Original-DWS effect, the DWS×WSC interaction contrast, a bootstrap procedure, a multiplicity-correction plan, and a leave-one-seed-out sensitivity analysis remain to be specified before any confirmatory claim about Dense Welfare Shaping can be made.

## 7. Evidence index

| # | Claim | File | Location | Why it supports the claim |
|---|---|---|---|---|
| 1 | c=0.0005 frozen, epsilon=1e-6, mode=discrete, scope=shared_global, cohort=fixed_four_vehicle, objective=maximin, welfare_source=running_active_attainment | `C:\dense reward\configs\dense_reward_protocol_v1.json` | lines 3-9 | Direct JSON keys, the single authoritative machine-readable protocol file referenced by the launcher's `_load_dense_protocol()`. |
| 2 | Previous c=0.1 revoked, never used in training | same file | `magnitude_recalibration.previous_c_never_used_in_training` key, and `reports\dense_reward_protocol_v1.md` lines 6-19 | Explicit boolean flag plus a full human-readable derivation record. |
| 3 | C64 source step = 1,200,000; final step = 2,000,000; checkpoint interval = 50,000 | `C:\dense reward\scripts\launch_dense_priority.py` | lines 46-47, 117 | Hardcoded module-level constants and the `checkpoint_every` assignment inside `build_plan()`. |
| 4 | 12 formal seeds | same file | lines 42-45 (`FORMAL_SEEDS` tuple) | Direct tuple literal used by both DWS launches. |
| 5 | Priority 1 → cell 4 (WSC+DWS), Priority 2 → cell 2 (DWS-only) | same file | lines 78-85, 98-101 | `_PRIORITY_CONDITION`/`_PRIORITY_RUN_TAG` dicts and the `use_wsc = priority % 2 == 1` / script-selection line. |
| 6 | Dense CLI flags built from the frozen config at launch time | same file | lines 156-159 | `"--dense-shaping-mode", protocol["dense_shaping_mode"]` etc., reading the same JSON loaded in item 1. |
| 7 | `eps-decay`/`lr-decay` NOT passed by the DWS launcher | same file | full `cmd` list, lines 139-160 | No `--eps-decay-steps-absolute` or `--lr-decay-steps-absolute` token appears anywhere in the constructed command. |
| 8 | Original/WSC launcher DOES pass them, read live from config | `C:\dense reward\scripts\launch_formal.py` | lines 109-112 | `"--eps-decay-steps-absolute", str(cfg["dqn"]["eps_decay_steps_absolute"])` etc. |
| 9 | Those config values equal the training scripts' own argparse defaults | `C:\dense reward\configs\FROZEN_EXPERIMENT_CONFIG.json` vs. `train_curriculum_stage_highwayenv.py` / `_wsc.py` | `dqn.eps_decay_steps_absolute=640000`/`dqn.lr_decay_steps_absolute=800000` vs. `default=640_000`/`default=800_000` at lines 131-132 / 161-162 | Side-by-side value comparison; numeric equality confirmed. |
| 10 | Both training scripts share one DQN-config factory (architecture/optimizer/γ/batch cannot diverge) | `train_curriculum_stage_highwayenv.py` line 188; `train_curriculum_stage_highwayenv_wsc.py` line 213 | `build_study_b_dqn_config(reward_condition="baseline", ...)` | Identical function imported from `thesis.study_b.shared_local_dqn` in both scripts. |
| 11 | Ensemble window formula `K(S)={S-150K,S-100K,S-50K,S}` shared code path | `C:\dense reward\project\src\thesis\study_b\q_ensemble.py` | lines 55-58 | Function definition; imported unchanged by `evaluate_dense_interim.py` and (per its own methodology note) by the WSC evaluation. |
| 12 | Greedy, epsilon-free evaluation action selection | same file | lines 142-146 | `select_ensemble_action` body: `np.argmax(...)`, no epsilon parameter exists. |
| 13 | H1 bank byte-identical across all three bundles (256 scenarios) | `C:\dense reward\scenario_banks\H1.json`, `F:\正式训练\scenario_banks\H1.json`, `F:\正式训练_seed_replication_v1\scenario_banks\H1.json` | SHA-256 = `8345ba0d...` for all three; `len(json.load(...))==256` | Direct hash comparison and length check performed this session. |
| 14 | Cell 4 formal completion, all 12 seeds | `C:\dense reward\logs\maximin_wsc_dense_<seed>.log`, `C:\dense reward\outputs\run_state\maximin_wsc_dense_<seed>.json` | `{"final_step": 2000000, ...}` line; `completed:true, returncode:0` | Direct log/manifest inspection for all 12 seeds this session and in the prior session that produced `reports/priority1_maximin_wsc_dense_results.md`. |
| 15 | Cell 2 NOT complete as of this audit | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_<seed>\...\ckpt_step_*.pt` (directory listing) + live `psutil` process check | 10 seeds at 1,900,000; 2 seeds (920105/920106) at 1,250,000; 12 live non-`_wsc` training processes found | Direct filesystem check and process inspection performed at the start of this audit. |
| 16 | Manifest `condition` field is unreliable (hardcoded) | `C:\dense reward\scripts\launch_dense_priority.py` | line 237: `write_run_manifest(r["run_id"], seed=r["seed"], condition="maximin", ...)` | The literal string `"maximin"` is passed regardless of `args.priority`/`condition` variable; confirmed independently by the Priority-3 (GGI) results report's own note that its run-state JSON also reads `"condition": "maximin"`. |
| 17 | No DWS statistical/bootstrap code exists | repo-wide grep for `bootstrap` (case-insensitive) under `scripts/` and `project/`; filename search for `*dense*bootstrap*`, `*dense*analysis*`, `*dws*` | (search performed, no DWS-specific hits beyond the pre-existing, non-DWS `thesis/study_b/analysis/bootstrap.py`) | Absence of any matching file/reference after an explicit targeted search. |
| 18 | Cell 1 (Original Maximin) and cell 3 (WSC Maximin) data exist but are outside this bundle | `F:\正式训练_seed_replication_v1\checkpoints\formal_runs\maximin_<seed>\`, `analysis_scripts\pooled12\outputs\pooled12_rq1_seed_level_metrics.csv`; `analysis_scripts\wsc_v2_formal\outputs\wsc_v2_formal_seed_level.csv` | directory/file existence confirmed | Direct `find`/`ls` this session. |
| 19 | Existing cell-4-vs-cell-3 descriptive comparison, not a formal statistical result | `C:\dense reward\reports\priority1_maximin_wsc_dense_results.md` | full document, esp. §6-7 | Report's own explicit self-description as "descriptive... not bootstrap CI + Holm correction," produced and reviewed earlier this engagement. |

## Machine-readable summary

```json
{
  "formal_training": {
    "source_step": 1200000,
    "additional_steps": 800000,
    "final_step": 2000000,
    "checkpoint_interval": 50000,
    "seed_count": 12,
    "formal_run_status": "cell_4_(Maximin+WSC+DWS)_complete_all_12_seeds; cell_2_(Maximin+DWS)_in_progress_10_of_12_seeds_at_1900000_2_of_12_seeds_at_1250000"
  },
  "evaluation": {
    "scenario_bank": "H1",
    "scenario_count": 256,
    "epsilon_eval": 0.0,
    "checkpoint_rule": "equal_weight_ensemble_of_4_checkpoints_at_final-150000_final-100000_final-50000_final",
    "same_scenarios_across_four_cells": true,
    "formal_evaluation_status": "cell_4_evaluated_complete_3072_rows; cell_2_not_yet_evaluated_(training_incomplete); cells_1_and_3_evaluated_separately_on_a_different_drive"
  },
  "statistics": {
    "inferential_unit": null,
    "bootstrap_resamples": null,
    "ci_level": null,
    "bootstrap_rng_seed": null,
    "p_values": null,
    "multiplicity_correction": null,
    "sensitivity_analysis": null
  }
}
```
