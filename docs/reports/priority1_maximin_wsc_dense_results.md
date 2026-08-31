# Priority 1 (Maximin + WSC + Dense) — Results Report

**Status**: Training complete, all 12 formal seeds reached step 2,000,000. Evaluation
complete on the full 12-seed set. **This report is a descriptive comparison only —
no bootstrap CI / Holm correction has been applied.** See "Open next step" at the end.

---

## 1. What this condition is

Priority 1 tests whether adding step-wise dense welfare-shaping reward on top of an
already-WSC-observable (welfare-state-observable, 22D obs) Maximin agent improves
fairness outcomes, relative to Maximin+WSC alone (no dense shaping). This is the
first of the Dense Reward Study's planned conditions (see `README.md` Section 6).

- **Training script**: `train_curriculum_stage_highwayenv_wsc.py`
- **Condition**: `maximin`, `welfare-lambda=0.5`
- **Dense shaping**: `--dense-welfare-shaping --dense-shaping-mode discrete`
- **Formal seeds (12)**: `900101, 900102, 900103, 900104, 910101, 910102, 920101, 920102, 920103, 920104, 920105, 920106`
- **Budget**: continuation from the shared C64 1,200,000-step checkpoint to 2,000,000 steps (800,000 additional steps total per seed)

## 2. Frozen protocol values

From `configs/dense_reward_protocol_v1.json` (frozen before any training outcome was seen):

| Parameter | Value |
|---|---|
| `dense_shaping_mode` | `discrete` |
| `dense_shaping_magnitude` (c) | **0.0005** (final; an earlier provisional c=0.1 was explicitly revoked before any formal run and never used in training — see `reports/dense_reward_protocol_v1.md`) |
| `dense_shaping_epsilon` | `1e-6` |
| `shaping_scope` | `shared_global` (one shaping term added to every controlled vehicle's reward, not per-agent) |
| `cohort` | `fixed_four_vehicle` (welfare `Phi_t` always computed over the same 4-vehicle id set, never filtered to "still active," to avoid an exit-artifact spike) |
| `welfare_source` | `running_active_attainment` (unchanged from Study B's existing welfare definition) |
| `primary_objective` | `maximin` |

c and epsilon were calibrated only from reward-scale/frequency statistics in a real
smoke trace, never from any training outcome (see `reports/dense_reward_protocol_v1.md`
for the full derivation and the record of the revoked c=0.1).

## 3. Training run history

- **2026-08-26**: Priority 1 launched on a separate "experiment machine" (GPU-equipped),
  ran to seeds 900101–910102 @ step 1,700,000 and seeds 920101–920106 @ step 1,550,000,
  then paused (see `outputs/dense_priority1_pause.json`).
- **2026-08-27**: Resumed on this local machine (`C:\dense reward`, no GPU — training
  continued on `--device cpu`; the paused run's `--device cuda` instruction did not
  apply here). Environment was rebuilt locally (fresh `.venv` against `C:\Python314`,
  since the venv copied from the experiment machine had a hardcoded, non-portable
  interpreter path). Resumed via
  `scripts/launch_dense_priority.py --priority 1 --device cpu --continue-from-latest`,
  which correctly picked up each seed's latest real checkpoint (verified via a
  `--dry-run` plan check against the pause note before the real launch, and via
  `verify_checkpoints.py` confirming all 291 existing checkpoint files loaded
  cleanly on CPU with 0 load errors).
- All 12 seeds reached exactly step 2,000,000 and exited cleanly (each wrote a
  `{"final_step": 2000000, ...}` completion line and manifest; no unhandled
  exceptions in any of the 12 logs).

## 4. Evaluation methodology

- **Script**: `scripts/evaluate_dense_interim.py` (adapted, with identical episode
  logic, from the already-used `evaluate_wsc_formal_v2.py` template that produced
  the existing WSC-only baseline numbers — same ensemble-window rule, same bank,
  same utility/Gini functions, so the comparison is apples-to-apples).
- **Ensemble window**: `{1,850,000, 1,900,000, 1,950,000, 2,000,000}` (the frozen
  "Amendment-13" rule `K(S) = {S-150K, S-100K, S-50K, S}` for `S = 2,000,000`).
- **Scenario bank**: `H1`, 256 episodes per seed (same bank/count as the existing
  WSC-only baseline evaluation).
- **Policy**: greedy (`epsilon=0`), WSC observation (`include_welfare_state=True`, 22D).
- **Output**: `outputs/welfare_analysis/dense_interim_evaluation_12seed_full.csv`
  (3,072 rows = 12 seeds × 256 episodes).
- **Baseline for comparison**: Maximin+WSC (no dense), same 12 seeds, same bank/window/
  methodology, from the pre-existing `F:\正式训练_seed_replication_v1\analysis_scripts\
  wsc_v2_formal\outputs\wsc_v2_formal_seed_level.csv` (maximin rows).

## 5. Full 12-seed results

Seed-level `U_min`/`Gini` = mean of the per-episode `min_U`/`gini` columns over the
256 H1 episodes (same aggregation convention as the baseline CSV).

| Seed | Dense U_min | WSC-only U_min | Δ U_min | Dense Gini | WSC-only Gini | Δ Gini | Dense completion | WSC-only completion | Dense collision | WSC-only collision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 900101 | 0.660 | 0.902 | **-0.242** | 0.170 | 0.049 | +0.121 | 0.660 | 0.902 | 0.340 | 0.098 |
| 900102 | 0.984 | 0.953 | +0.031 | 0.008 | 0.023 | -0.016 | 0.980 | 0.953 | 0.020 | 0.047 |
| 900103 | 0.998 | 0.858 | +0.140 | 0.000 | 0.069 | -0.069 | 1.000 | 0.863 | 0.000 | 0.137 |
| 900104 | 0.869 | 0.967 | -0.098 | 0.063 | 0.016 | +0.047 | 0.879 | 0.969 | 0.121 | 0.031 |
| 910101 | 0.474 | 0.996 | **-0.522** | 0.265 | 0.002 | +0.263 | 0.477 | 0.996 | 0.523 | 0.004 |
| 910102 | 0.798 | 0.228 | **+0.570** | 0.100 | 0.386 | -0.286 | 0.805 | 0.230 | 0.195 | 0.770 |
| 920101 | 0.801 | 0.967 | -0.166 | 0.100 | 0.010 | +0.090 | 0.801 | 0.992 | 0.199 | 0.008 |
| 920102 | 0.630 | 0.339 | +0.291 | 0.182 | 0.332 | -0.150 | 0.645 | 0.340 | 0.355 | 0.660 |
| 920103 | 0.979 | 0.953 | +0.027 | 0.005 | 0.024 | -0.018 | 0.996 | 0.953 | 0.004 | 0.047 |
| 920104 | 0.917 | 0.904 | +0.013 | 0.040 | 0.047 | -0.007 | 0.922 | 0.906 | 0.078 | 0.094 |
| 920105 | 0.868 | 0.911 | -0.043 | 0.064 | 0.039 | +0.025 | 0.875 | 0.930 | 0.125 | 0.070 |
| 920106 | 0.608 | 0.961 | **-0.353** | 0.201 | 0.020 | +0.181 | 0.602 | 0.961 | 0.398 | 0.039 |

**Mean delta (simple arithmetic mean of the 12 per-seed deltas, not bootstrap):**

| Metric | Mean Δ (Dense − WSC-only) | Range across seeds |
|---|---:|---:|
| U_min | -0.029 | [-0.522, +0.570] |
| Gini | +0.015 | [-0.286, +0.263] |
| Completion | -0.030 | [-0.522, +0.575] |
| Collision | +0.030 | [-0.575, +0.522] |

**Seeds with U_min improved vs. worsened: 6 / 12 vs. 6 / 12 — an exact split, no directional signal.**

## 6. Honest interpretation

- **No systematic fairness improvement or regression is visible across the complete
  12-seed formal sample.** Mean deltas are all close to zero relative to the spread
  of individual seed effects.
- **Seed-to-seed variance is large and bidirectional**, comparable in kind (though
  numerically wider) to seed heterogeneity already documented elsewhere in this
  codebase for an unrelated comparison (WSC-vs-Original on the Baseline condition:
  completion delta range was [-0.219, +0.223] across the same 12 seeds — about
  2-3× narrower than this Dense-vs-WSC-only comparison's [-0.522, +0.575] range).
  This means part of the variance is a known property of this experimental setup
  (small 4-vehicle episodes, per-seed scenario sampling, DQN training stochasticity),
  but the wider spread here than in that other comparison is not fully explained by
  that alone and is not conclusively attributable to a specific code defect.
- **Two seeds show a striking safety regression**: 910101 (collision 0.004 → 0.523)
  and 920106 (collision 0.039 → 0.398).
  - **920106** shows *real, visible degradation during training itself* (not just
    at evaluation): the training-window completion rate declines steadily from
    0.874 (step 1,550,000) to 0.489 (step 2,000,000) while collision climbs from
    0.126 to 0.440, with `mean_Q(policy)` dropping sharply to 0.137 at step
    1,850,000. This looks like genuine late-training instability, not an
    evaluation artifact — worth further diagnosis (e.g. inspecting loss/grad-norm
    around step 1,800,000–1,850,000 if per-update diagnostics are re-run with
    `--dense-log-every`).
  - **910101** shows the opposite pattern: its training-window metrics stayed
    stable throughout (completion 0.80–0.86, collision 0.14–0.19 the whole way),
    but the final *greedy* evaluation policy performs much worse (completion 0.477,
    collision 0.523). This is a train-time (ε-greedy) vs. evaluation-time (greedy)
    generalization gap, not a training-log-visible collapse.
  - **Two seeds show a large *improvement*** in the opposite direction: 910102
    (collision 0.770 → 0.195) and 920102 (collision 0.660 → 0.355) — both seeds
    where the WSC-only baseline itself was already poor.
- No NaN/Inf, exception, or crash was found in any of the 12 training logs — there
  is no smoking-gun implementation bug signature. The dense-shaping implementation
  itself was independently unit-tested earlier in this study (20 tests including an
  ON/OFF byte-identical equivalence check and an active-set/exit-artifact guarantee),
  though only at small/short-run scale, not at this full 2,000,000-step scale.

## 7. Open next step

This report is a **descriptive per-seed-delta comparison only**. To turn this into a
formal, citable result, the natural next step (not performed here) is to re-run the
same bootstrap-CI + Holm-correction statistical treatment already used for the WSC
v2 formal analysis (`wsc_v2_formal_analysis.py`'s methodology) on this Dense-vs-
WSC-only comparison, and to further diagnose the 920106 training-instability episode
before treating any single seed's result as representative.
