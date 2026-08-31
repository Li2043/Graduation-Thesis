# WSC Formal Campaign Incident — Diagnosis Report

**Status: ROOT CAUSE CONFIRMED AND FIXED. Corrected code validated (see `WSC_CORRECTED_MAPPING_VALIDATION_AND_REFREEZE_REPORT.md`). A new, non-overlapping 48-run campaign has been prepared under a new scientific freeze but has NOT been launched pending explicit user approval.**

This report documents an incident discovered during routine monitoring of the formal WSC (Welfare-State Communication) training campaign launched per `wsc_formal_launch_report.md`, the diagnostic process used to isolate the cause, and the current state of the campaign.

## 0. FINAL DIAGNOSIS (read this first)

**The final, confirmed root cause is a semantic column-mapping bug in `wsc_checkpoint_expansion.py`, not a numerical-stability or optimizer issue.** The WSC 22D observation layout *interleaves* the 4 new welfare-state features into the vector (M_i inserted right after the self block at index 6; each neighbour's M_j appended at the end of its own 5-field block, indices 11/16/21) rather than appending them as a contiguous suffix. The original checkpoint-expansion code assumed a simple prefix/suffix split (`new[:, :18] = old; new[:, 18:] = 0`), which fed every original weight column from index 6 onward the WRONG, shifted input feature from the very first forward pass — a pure data-corruption bug, unrelated to gradient magnitudes, Adam bookkeeping, or Q-value margins. Full detail and evidence: §6b below.

Two earlier hypotheses were investigated in good faith, in this order, and are now **REFUTED / SUPERSEDED** — kept below as an audit trail, not deleted:
1. **Adam optimizer step-count inheritance** (§4) — refuted by direct A/B test (no difference between inherited step and step-reset-to-1).
2. **Small Q-value decision-margin fragility** (§6, originally written as "root cause identified") — this was the working diagnosis when a fix (Option 1: gradient warmup on the new columns) was selected and implemented. The fix was then tested and **also refuted** (§6a): ramping the new columns' gradient to as little as 10% strength for the first 50,000 steps produced collapse nearly identical to no ramp at all, which is inconsistent with a pure gradient-magnitude/margin-fragility mechanism. This refutation is what prompted re-examining the WSC layout itself and finding the actual bug (§6b).

The gradient-warmup mechanism (`wsc_gradient_warmup.py`) is **not used in the formal protocol going forward** — it did not solve the problem, and the corrected mapping alone (no warmup, full strength from step 0) reproduces healthy training dynamics (§6b).

---

## 1. Timeline

| Time | Event |
|---|---|
| 2026-08-21 19:54 | Formal WSC campaign launched (48 runs, concurrency=4) |
| 2026-08-22 ~11:00-14:00 | Routine progress check: 20/48 runs COMPLETED (seeds 900101, 900102, 900103, 900104, 910101, all 4 conditions each), 4/48 RUNNING (seed 910102) |
| 2026-08-22 (this session) | User requested an interim look at the 5 completed seeds to inform thesis-writing direction (explicitly flagged beforehand as high-risk given known seed heterogeneity — see §2) |
| 2026-08-22 | Interim evaluation revealed a severe, consistent task-competence collapse across **all four conditions, all five seeds** (§3) — not a fairness-specific effect |
| 2026-08-22 | Orchestrator stopped (no further new seeds launched); in-flight seed 910102 batch initially left running |
| 2026-08-22 | User explicitly requested pausing all training pending diagnosis | 
| 2026-08-22 | Seed 910102's 4 in-flight processes stopped cleanly (not crashed); registry updated to `BLOCKED` with last valid checkpoint recorded |
| 2026-08-22 | Diagnostic sequence run (§4-§6): Adam step-count hypothesis tested and refuted; Original-18D control confirmed the test harness itself is not at fault; Q-value margin analysis identified the actual mechanism |

---

## 2. Why an "interim look" was flagged as risky beforehand

Before running any interim analysis, the user was warned (and explicitly acknowledged) that a 5/12-seed read is a form of early "peeking" that this project's own frozen-protocol discipline is designed to avoid, given known seed-level heterogeneity in the **Original** (non-WSC) experiment — e.g. seed 910102's Mean-minus-Baseline ΔU_min was −0.678 while seed 920104's was +0.347, a sign-flipping range. The interim look was still performed (user's explicit choice), but this caveat is recorded here because it is why the interim analysis was framed as strictly non-formal from the outset — which turned out to be the right call, since it surfaced a data-validity problem, not a fairness result.

---

## 3. What the interim evaluation found

Evaluation script: `F:\正式训练_seed_replication_v1\analysis_scripts\ch5_baseline\evaluate_wsc_formal.py` (reuses the exact, unmodified `evaluate_formal_welfare.py`-style episode logic, pointed at the WSC checkpoint-Q ensemble, `include_welfare_state=True`, obs_dim=22). Evaluated: seeds 900101–900104 and 910101, all 4 conditions, frozen H1 bank, 256 scenarios, ε=0, same 4-checkpoint ensemble window {1,850,000; 1,900,000; 1,950,000; 2,000,000} as every other formal result in this project.

**Held-out completion / collision, Original vs. WSC, matched on the same 5 seeds:**

| Condition | Original completion | WSC completion | WSC collision |
|---|---:|---:|---:|
| Baseline | 0.974 | 0.520 | 0.48 |
| Mean | 0.983 | 0.322 | 0.68 |
| GGI | 0.961 | 0.617 | 0.38 |
| Maximin | 0.891 | 0.480 | 0.51 |

**Critical observation: Baseline+WSC — which has no welfare-shaping term at all (λ=0) — collapses just as badly as the welfare-shaped conditions.** This immediately rules out any explanation involving the welfare-shaping mechanism itself; the problem is in the WSC information channel/observation change, not in how the reward is used.

Full per-seed numbers: `F:\正式训练_seed_replication_v1\analysis_scripts\ch5_baseline\outputs\wsc_interim\wsc_interim_seed_level_comparison.csv` and `wsc_interim_evaluation_seed*.csv`.

**Training-window telemetry (from each run's own manifest, not just held-out eval) confirms this is a real training-time phenomenon, not an evaluation artifact.** Example, seed 900101:

| Run | step=1,400,000 completion/collision | step=2,000,000 (final) completion/collision |
|---|---|---|
| baseline_wsc_900101 | 0.109 / 0.868 | 0.316 / 0.653 |
| mean_wsc_900101 | 0.355 / 0.622 | 0.202 / 0.784 |
| ggi_wsc_900101 | 0.177 / 0.815 | 0.274 / 0.683 |
| maximin_wsc_900101 | 0.301 / 0.652 | 0.231 / 0.696 |

The collapse is already essentially complete by step 1,400,000 — only 200,000 steps (25%) into the 800,000-step continuation budget — and never recovers through step 2,000,000.

---

## 4. Diagnostic 1 — Adam optimizer step-count hypothesis (REFUTED)

**Hypothesis**: `wsc_checkpoint_expansion.expand_optimizer_state()` preserves the source checkpoint's Adam `step` counter (~1,199,339) for the *entire* `net.0.weight` tensor, including the 4 brand-new all-zero columns. Because Adam's bias-correction factor `1-β^t` is ≈1 at t≈1.2M, the new columns' first real gradient updates would receive none of the bias-correction "warmup" a genuinely fresh parameter gets, producing oversized early updates.

**Test**: two 50,000-step continuations from the identical real C64 checkpoint (seed 900101, Baseline/λ=0 — the cheapest reproduction case), identical in every respect except one variable:
- **Variant A** ("current"): `net.0.weight`'s optimizer step = inherited (~1.2M), exactly matching the real formal WSC runs.
- **Variant B** ("step-reset diagnostic"): same expansion, but `net.0.weight`'s optimizer step scalar overwritten to 1 before training (exp_avg/exp_avg_sq values themselves left untouched — isolates only the bias-correction mechanism).

Script: `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\diagnostic_adam_stepcount_test.py`. Output: `diagnostic_adam_stepcount\diagnostic_results.json`.

| Step | A (inherited step) completion/collision | B (step reset to 1) completion/collision |
|---|---|---|
| 1,205,000 (+5K) | 0.163 / 0.837 | 0.174 / 0.826 |
| 1,225,000 (+25K) | 0.098 / 0.902 | 0.165 / 0.824 |
| 1,250,000 (+50K) | 0.163 / 0.837 | 0.151 / 0.826 |

**Result: no meaningful difference between A and B.** Both collapse to the same severity within the first 5,000 steps. **The Adam step-count hypothesis is refuted** — resetting it does not prevent or reduce the collapse.

---

## 5. Diagnostic 2 — Original-18D control (harness verified sound)

Before concluding the problem is WSC-specific, ruled out the possibility that the lightweight diagnostic training loop itself (written fresh for this diagnostic, not the frozen formal script) had some discrepancy that would break even a plain Original continuation.

**Test**: identical lightweight harness, identical seed/checkpoint/schedule/budget, but **no WSC at all** — `include_welfare_state=False`, `obs_dim=18`, the real C64 checkpoint loaded directly with no expansion.

Script: `diagnostic_control_18d.py`. Output: `diagnostic_adam_stepcount\diagnostic_control_18d_results.json`.

| Step | Completion | Collision |
|---|---:|---:|
| 1,205,000 (+5K) | 0.784 | 0.216 |
| 1,220,000 (+20K) | 0.901 | 0.099 |
| 1,230,000 (+30K) | 1.000 | 0.000 |
| 1,250,000 (+50K) | 0.808 | 0.192 |

**Result: completely healthy throughout.** This confirms the diagnostic harness is faithful (no artifact of the diagnostic code itself), and **confirms the collapse is 100% attributable to the WSC/22D observation change**, not the optimizer-state handling nuance (already ruled out in §4) and not a test-harness bug.

---

## 6. Diagnostic 3 — Q-value decision-margin analysis (INITIAL HYPOTHESIS — later REFUTED, see §0/§6a/§6b)

With both prior hypotheses addressed, tested the project's own previously-documented failure mode — **small Q-value margin policy-boundary instability** (the same mechanism `q_ensemble.py`'s own design rationale cites as the reason a 4-checkpoint ensemble is used at evaluation time, per Amendment 12/13: `SMALL_Q_MARGIN_POLICY_BOUNDARY_INSTABILITY`, `C4_DIVERSITY_RECOVERY_SYNTHESIS.md`).

**Method**: loaded the real C64 online network (18D, unmodified), computed Q-values for 2,000 sampled observations, measured the gap between the best and second-best action's Q-value (the "decision margin") for each.

Read-only script executed inline this session (not saved as a separate file; reproducible from the description above using the source checkpoint and `QNetwork`).

| Statistic | Value |
|---|---:|
| Minimum margin | 7.15e-07 |
| 5th percentile | 0.00051 |
| Median | 0.0042 |
| Mean | 0.0093 |
| Maximum | 0.769 |
| Fraction of observations with margin < 0.01 | **91.7%** |
| Fraction with margin < 0.05 | 97.6% |
| Fraction with margin < a single Adam step (~lr=0.0005) | 4.85% |

**Conclusion — root cause**: the C64 checkpoint's decisions are made on razor-thin Q-value margins for the large majority of states (median margin 0.0042). Although the 4 new WSC input columns are exactly zero-weighted at initialization (verified elsewhere to be an exact no-op, `wsc_implementation_validation.md` TEST 2-4), **their gradient is not zero** — it is proportional to the (generally non-zero) M-feature input value and the ordinary backpropagated error signal. The very first real optimizer update therefore moves these new weights by a normal, unremarkable Adam step size (~lr), regardless of the bias-correction nuance tested in §4. Because these new weights feed the *same* 64 hidden units used by the carefully-converged old 18 columns, this ordinary-sized perturbation is large enough, relative to the wafer-thin decision margins measured above, to flip a large fraction of action choices almost immediately across all four vehicles simultaneously — cascading into the observed collision-dominated collapse within the first few thousand steps, well before any welfare-shaping or fairness-relevant learning could occur.

**This is not evidence against the WSC concept itself.** It is a numerical/optimization fragility of continuing single-network training from an already highly-converged, small-margin checkpoint when *any* new trainable capacity is introduced — largely independent of what that new capacity semantically represents.

> **STATUS UPDATE (superseding the paragraph above): this hypothesis was tested and REFUTED — see §6a.** The margin-fragility mechanism predicts that *reducing* the new columns' effective gradient magnitude should reduce or delay the collapse; it did not (§6a). The actual bug (§6b) is data corruption, not gradient magnitude, so it is invisible to any fix that only scales gradients.

---

## 6a. Diagnostic 4 — Gradient-ramp fix attempt (Option 1, selected based on §6 — REFUTED)

Based on the §6 diagnosis, the user selected **Option 1** (learning-rate warmup/freeze for the 4 new WSC input columns) from the candidate-fix list in §8. Implemented as `thesis.study_b.wsc_gradient_warmup.NewColumnGradientRamp`: a `Tensor.register_hook()` on `net.0.weight` that linearly scales the gradient of columns `[18:22]` from 0 to 1 over a configurable number of local steps, leaving the original 18 columns' gradient always at scale 1.0.

**Test**: same lightweight 50,000-step Baseline+WSC (λ=0) harness as every prior diagnostic, same seed/checkpoint (900101), two candidate warmup lengths.

Script: `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\diagnostic_gradient_ramp_fix.py`. Output: `diagnostic_adam_stepcount\diagnostic_gradient_ramp_fix_results.json`.

| Step | fix_warmup_20000 completion/collision | fix_warmup_50000 completion/collision |
|---|---|---|
| +5,000 | collapsed (comparable to no-ramp baseline) | collapsed (comparable to no-ramp baseline) |
| +50,000 | still collision-dominated | still collision-dominated |

**Result: both warmup lengths collapsed nearly identically to the un-fixed run, even though the new columns' gradient was scaled down to as little as 10% strength for most of the window.** This is the key negative result: if the mechanism were "an ordinary-sized Adam step to the new columns is too large relative to razor-thin Q-margins" (§6's hypothesis), scaling that step down by 10x should have measurably delayed or softened the collapse. It did not. **This refutes both the gradient-ramp fix and, by implication, the gradient-magnitude framing of the §6 hypothesis itself** — the problem does not shrink when the new columns' influence shrinks, which means the new columns are not merely "too strong," they are wired to the wrong input from the start. This result is what prompted a direct re-examination of the WSC 22D layout, leading to §6b.

---

## 6b. Diagnostic 5 — FINAL ROOT CAUSE: WSC observation-to-network semantic column-mapping bug (CONFIRMED)

Re-examined `local_observation.build_local_observation`'s actual WSC 22D layout directly, rather than assuming it. The layout **interleaves** the 4 new features rather than appending them as a suffix:

- Self block: `[role, speed, target_speed, acceleration, dist_to_merge, prev_action, M_i]` — indices `0–6` (M_i is new, at index 6, not 18).
- Neighbour 0: `[presence, Δd, Δv, lane_relation, M_j]` — indices `7–11` (M_j at 11).
- Neighbour 1: indices `12–16` (M_j at 16).
- Neighbour 2: indices `17–21` (M_j at 21).

So from index 6 onward, every ORIGINAL feature is shifted by +1, +2, or +3 relative to its old 18D index. The correct old→new column correspondence is:

```
OLD_TO_NEW_COLUMN_MAP = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20]
NEW_ONLY_COLUMNS      = [6, 11, 16, 21]
```

The pre-incident `wsc_checkpoint_expansion.py` instead did `new[:, :18] = old; new[:, 18:] = 0` — a naive prefix/suffix split. Under that WRONG mapping, e.g. the weight column trained (over 1.2M steps) to interpret "neighbour_0 presence" (old index 6) was, from the very first WSC forward pass, actually being fed **M_i** (the new index-6 feature) instead — and every feature after that was similarly misaligned. This is a **data-corruption bug**: the "converged" old weights were multiplying the wrong physical quantity from step 0, independent of gradient magnitude, optimizer step count, or Q-margins. It also explains why the gradient-ramp fix (§6a) had no effect: scaling down the gradient of the *new* columns does nothing to fix the *old* columns being fed the wrong inputs.

**This also means the original WSC implementation validation (`wsc_implementation_validation.md`, TEST 2–4) was itself invalid**: those tests built synthetic Q-equivalence checks using `np.concatenate([old_obs, new_features])`, which encodes the SAME wrong prefix/suffix assumption as the buggy expansion code — so the test and the code made the identical mistake and "agreed" with each other by construction, not because the mapping was actually correct.

**Fix**: rewrote `wsc_checkpoint_expansion.py` to derive `OLD_TO_NEW_COLUMN_MAP`/`NEW_ONLY_COLUMNS` programmatically from `local_observation.py`'s own layout constants (`SELF_OBS_DIM`, `NEIGHBOUR_OBS_DIM`, `NEIGHBOUR_SLOTS`, etc.), with module-level `assert` self-tests pinning the exact expected mapping above. `expand_state_dict` now does `new[:, OLD_TO_NEW_COLUMN_MAP] = old; new[:, NEW_ONLY_COLUMNS] = 0.0`. The same corrected mapping is applied to the Adam optimizer's `exp_avg`/`exp_avg_sq` moment tensors for `net.0.weight` (`expand_optimizer_state`); `step` and every other parameter's optimizer state are copied unchanged (no speculative reset). The online and target networks are expanded independently (never `target = online`).

**Re-verification 1 — genuinely matched real-observation Q-equivalence** (`reverify_q_equivalence_fixed.py`): runs an Original-18D env and a WSC-22D env in lockstep on identical scenarios/seeds/actions (both driven by the same Original-ensemble policy), so both environments visit the identical underlying vehicle states at every step — then compares `Q_old(obs_18d)` against `Q_new(obs_22d)` computed from the *actually corresponding* real observations (not a synthetic concatenation, closing the exact blind spot that hid the original bug):

```
n_compared = 2208
max_abs_diff(Q_old, Q_new) = 4.77e-07
greedy action agreement = 2208/2208 = 100.000%
```

**Re-verification 2 — corrected mapping, NO gradient warmup, full strength from step 0** (`diagnostic_fixed_expansion_test.py`): same 50,000-step Baseline+WSC (λ=0) harness as every prior diagnostic, seed 900101, corrected `expand_checkpoint`, no ramp/hook installed at all:

| Step | Completion | Collision |
|---|---:|---:|
| +5,000  | 0.458 | 0.542 |
| +10,000 | 0.868 | 0.132 |
| +15,000 | 0.915 | 0.085 |
| +20,000 | 0.942 | 0.058 |
| +25,000 | 0.861 | 0.139 |
| +30,000 | 0.763 | 0.237 |
| +35,000 | 0.684 | 0.316 |
| +40,000 | 0.662 | 0.338 |
| +45,000 | 0.863 | 0.137 |
| +50,000 | 0.903 | 0.097 |

This closely matches the shape of the §5 Original-18D control (healthy, fluctuating within a normal training-noise band, no persistent collision-dominated collapse) — **confirming the mapping fix alone, with no gradient tricks whatsoever, resolves the incident.** A permanent regression test (`tests/study_b/test_wsc_semantic_column_alignment.py`) now guards against this exact class of bug recurring, including a negative control that reintroduces the old prefix/suffix bug on purpose and confirms the test suite detects it.

**Conclusion**: the WSC concept, the terminal welfare-shaping mechanism, and the DQN training procedure were never the problem. The incident was caused entirely by an implementation bug in checkpoint expansion, now fixed and independently re-verified by two different methods.

---

## 7. Current state

- **Orchestrator**: stopped (was PID 36328/27496). No further runs were launched under the pre-fix campaign.
- **20 completed runs** (seeds 900101, 900102, 900103, 900104, 910101 × 4 conditions each): PRESERVED on disk for audit, registry status changed to `INVALID_PRE_FIX_COLUMN_MAPPING`. Confirmed invalid (used the buggy column mapping) — must never be used in formal WSC evaluation, fairness comparison, or thesis tables/figures.
- **4 in-flight runs** (seed 910102, all 4 conditions): stopped cleanly (not crashed). Registry status changed from `BLOCKED` to `INVALID_PRE_FIX_COLUMN_MAPPING`; last valid (but invalid-mapping) checkpoints preserved and **must NOT be resumed**:

  | Run | Last valid (pre-fix, invalid) checkpoint |
  |---|---:|
  | baseline_wsc_910102 | 1,750,000 |
  | mean_wsc_910102 | 1,750,000 |
  | ggi_wsc_910102 | 1,700,000 |
  | maximin_wsc_910102 | 1,600,000 |

- **24 pending runs** (seeds 920101–920106 × 4 conditions): never started; registry status changed to `INVALID_PRE_FIX_COLUMN_MAPPING` (superseded — see new matrix below).
- No Original (non-WSC) formal checkpoint or result was touched at any point, before or during this incident's diagnosis and fix.
- **Resolution**: the semantic column-mapping bug (§6b) has been fixed, independently re-verified, and covered by a permanent regression test. A NEW scientific freeze and a NEW 48-run campaign matrix (12 seeds × 4 conditions, all restarting from each seed's ORIGINAL 1.2M C64 checkpoint, corrected mapping, gradient warmup disabled, writing to a NEW output root that does not overlap the invalid pre-fix runs) have been prepared — see `WSC_CORRECTED_MAPPING_VALIDATION_AND_REFREEZE_REPORT.md`. **The new campaign has NOT been launched** and requires explicit user approval.

---

## 8. Candidate fixes considered (historical — superseded by §6b)

This section is preserved as originally written, as an audit trail of the decision process, even though the fix that was actually selected here (#1) turned out not to address the true root cause.

1. **Learning-rate warmup/freeze specifically for the 4 new WSC input weights** during an initial burn-in period. **This was the option selected by the user ("我选择方案1") and implemented as `wsc_gradient_warmup.NewColumnGradientRamp`.** Tested in §6a and **REFUTED** — it did not prevent or reduce the collapse at either warmup length tested. The true bug (§6b) was a data-alignment error, which gradient-magnitude scaling cannot fix. **This mechanism is DISABLED by default in the formal protocol** (`--wsc-weight-warmup-steps` now defaults to 0 in `train_curriculum_stage_highwayenv_wsc.py`); the code is retained only for possible future diagnostic use and must not be silently activated in any formal launch command.
2. **Gradient clipping** on the whole network. Never implemented or tested — superseded once §6b identified the actual (non-gradient-related) bug.
3. **Start WSC continuation from a less-converged / less margin-fragile checkpoint.** Never implemented — superseded for the same reason; the original C64 branch point was correct all along.

**Actual resolution (not listed above at the time this section was written): fix the semantic column mapping in `wsc_checkpoint_expansion.py` (§6b).** No gradient-based, learning-rate-based, or checkpoint-selection change was needed. New validation was performed (`WSC_CORRECTED_MAPPING_VALIDATION_AND_REFREEZE_REPORT.md`) and a new scientific freeze issued before any relaunch is prepared.

---

## 9. Files referenced in this report

- `F:\正式训练_seed_replication_v1\analysis_scripts\ch5_baseline\evaluate_wsc_formal.py`
- `F:\正式训练_seed_replication_v1\analysis_scripts\ch5_baseline\wsc_interim_comparison.py`
- `F:\正式训练_seed_replication_v1\analysis_scripts\ch5_baseline\outputs\wsc_interim\` (all interim evaluation CSVs)
- `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\diagnostic_adam_stepcount_test.py`
- `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\diagnostic_control_18d.py`
- `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\diagnostic_adam_stepcount\` (all diagnostic result JSONs/logs)
- `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\wsc_formal_run_registry.csv` (all 48 pre-fix rows now `INVALID_PRE_FIX_COLUMN_MAPPING`, preserved for audit)
- `F:\正式训练_seed_replication_v1\project\src\thesis\study_b\wsc_checkpoint_expansion.py` (rewritten with the corrected, programmatically-derived column mapping)
- `F:\正式训练_seed_replication_v1\project\src\thesis\study_b\wsc_gradient_warmup.py` (kept for diagnostic use only, disabled by default in the formal script)
- `F:\正式训练_seed_replication_v1\project\tests\study_b\test_wsc_semantic_column_alignment.py` (new permanent regression test, including a negative control)
- `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\diagnostic_gradient_ramp_fix.py` (§6a evidence)
- `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\reverify_q_equivalence_fixed.py` (§6b re-verification 1)
- `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\diagnostic_fixed_expansion_test.py` (§6b re-verification 2)
- Prior context: `wsc_implementation_validation.md` (note: its TEST 2–4 are now known invalid, see §6b), `wsc_formal_launch_report.md`, `wsc_scientific_freeze_manifest.json/.md` (superseded — see the new freeze package referenced in `WSC_CORRECTED_MAPPING_VALIDATION_AND_REFREEZE_REPORT.md`)
