# WSC v2 Formal Evaluation and Safety Report

**Scope: evaluation-only.** No training was launched, resumed, or modified while producing this report. No scientific training code, reward definition, GGI weight, welfare lambda, WSC feature definition, checkpoint expansion logic, observation layout, source checkpoint, training schedule, H1 scenario bank, or checkpoint ensemble definition was changed. All 48 WSC v2 checkpoints, the 48 Original formal checkpoints, and the 48 invalid v1 checkpoints were read-only inputs throughout.

---

## A. Executive summary

- **No fairness interaction effect (U_min or Utility Gini) reaches statistical significance for any of the three welfare conditions (Mean, GGI, Maximin), in either the primary n=12 analysis or the n=11 sensitivity analysis, after Holm correction.** All Holm-adjusted p-values are ≥ 0.12.
- **The result is not robust across U_min and Gini, or across the primary vs. sensitivity sample.** GGI shows the least-unfavorable point estimate for U_min (+0.074, n=12) but this is entirely driven by one seed (910102) with extreme leverage; excluding it collapses the estimate to +0.021 with a CI comfortably straddling zero. Mean's estimate is negative and *becomes more negative and nominally CI-excluding-zero* once 910102 is removed (n=11: −0.137, CI [−0.278, −0.006]) — but this does not survive Holm correction (holm p = 0.121) and must not be read as evidence of harm.
- **Task performance/safety**: no condition shows a statistically distinguishable paired Completion or Collision shift (all raw two-sided p ≥ 0.19, all CIs wide and straddling zero). The dominant pattern is **strong seed-level heterogeneity with both large improvements and large deteriorations in the same condition** (e.g., Maximin completion delta ranges from −0.645 to +0.305 across seeds) — i.e., outcome (3) in the taxonomy requested (little-to-no mean shift, increased dispersion for some conditions, decreased for others), not a uniform "WSC costs completion" story.
- **Results are strongly seed-sensitive.** Leave-one-seed-out analysis identifies seed 910102 as the single highest-leverage seed for both Mean and GGI (both outcomes), and seed 920102 as the highest-leverage seed for Maximin (both outcomes). No result should be described as "robust to seed choice."
- **No technical invalidity was found.** All 48 WSC v2 runs are technically valid (exit code 0, complete 4-checkpoint ensemble, correct 1.2M-C64 provenance, zero drift in the 18 frozen scientific-code hashes). Several individually poor-performing runs were checked against training-window telemetry and stderr logs and show ordinary noisy-but-not-collapsed trajectories, not a recurrence of the fixed column-mapping bug.
- **Additional training seeds are not currently justified** under the pre-registered logic in Section M: estimates are not clearly non-null, seed-to-seed variance dominates the current uncertainty rather than a narrowing-but-still-ambiguous CI, and no consistent effect direction survives the sensitivity check. A prospective power calculation is provided instead of a recommendation to add seeds ad hoc.
- **Overall interpretation**: the current formal evidence is **inconclusive-tending-toward-null** for the hypothesis that local welfare-state observability (WSC) makes welfare-oriented rewards more effective relative to Baseline. It does not support the hypothesis, and for two of three conditions (Mean, Maximin) the point estimates lean unfavorable, but confidence intervals are too wide, and too seed-sensitive, to call this a confirmed null either.

---

## B. Data provenance and freeze verification

| Item | Value |
|---|---|
| Original Baseline source | `F:\正式训练\outputs\welfare_analysis\taskonly_evaluation_merged.csv` |
| Original Mean/GGI/Maximin source | `F:\正式训练_seed_replication_v1\analysis_scripts\pooled12\outputs\pooled12_welfare_evaluation_merged.csv` |
| WSC v2 (corrected) source | `F:\正式训练_seed_replication_v1\analysis_scripts\ch5_baseline\outputs\wsc_interim_v2\wsc_interim_v2_evaluation_all12.csv` (produced by `evaluate_wsc_formal_v2.py`: `include_welfare_state=True`, `obs_dim=22`, corrected `wsc_checkpoint_expansion`, H1 bank, 256 scenarios/seed-condition, greedy ε=0, 4-checkpoint ensemble) |
| WSC v2 run registry | `wsc_formal_launch_v2\wsc_formal_run_registry_v2.csv` — 48/48 rows audited: `COMPLETED`, `exit_code=0`, all four ensemble checkpoints present, all `source_checkpoint` = that seed's original `ckpt_step_1200000.pt`, no path under the invalid `wsc_formal_runs\` root |
| v1 invalid registry | 48/48 rows confirmed `INVALID_PRE_FIX_COLUMN_MAPPING`, untouched by this analysis |
| Frozen scientific source files | 18/18 SHA256 hashes in `wsc_scientific_freeze_manifest_v2.json` re-verified against current disk state: **zero drift** |
| Referenced file `1e4be11b-96ea-4016-9226-8f3ed7dc4a89.md` | Not found anywhere in the hub or either physical repo. The corresponding project copy is `WSC_V2_INFORMAL_ANALYSIS_SUMMARY.md` (this project's own consolidated informal summary, produced immediately after the campaign finished) — used as background context only; none of its numbers were copied into this report without independent recomputation. |

All of the above (paths, SHA256 hashes, run audit, seed list, bank, episode count, ensemble window) are saved machine-readably in `wsc_v2_data_provenance.json`.

---

## C. Formal statistical protocol

Reused **verbatim** (not reimplemented) from the project's own canonical formal-analysis script, `F:\正式训练_seed_replication_v1\analysis_scripts\pooled12\merge_and_audit.py`: `gini()`, `bootstrap_ci_paired()`, `bootstrap_p_value()`, `holm_correction()`.

| Protocol element | Value | Source |
|---|---|---|
| Resampling unit | **seed** (paired bootstrap over 12, or 11, seed-level interaction/contrast values — never the 256 episodes) | `bootstrap_ci_paired` resamples the length-n array of per-seed values |
| Bootstrap replicates | 10,000 | `merge_and_audit.py` default, unchanged |
| Bootstrap RNG seed | 0 (`np.random.default_rng(0)`) | unchanged |
| CI type | Percentile bootstrap, [2.5th, 97.5th] percentile → 95% CI | unchanged |
| p-value | Two-sided: `p = min(1, 2·min(P(boot≤0), P(boot≥0)))` | unchanged |
| Null hypothesis (fairness) | H₀: I_Y^c = 0 for each condition c and outcome Y | — |
| Null hypothesis (safety) | H₀: Δmetric^c (WSC − Original) = 0 | — |
| Holm correction family | **Two independent m=3 families**: (1) U_min interaction across {Mean, GGI, Maximin}; (2) Gini interaction across {Mean, GGI, Maximin}. Generalizes the project's own established convention (`merge_and_audit.py` Task 4 Holm-corrected {GGI-Mean, Maximin-Mean} for U_min, m=2, as its own family) to the three-condition-vs-own-Baseline interaction design used here. **Not** pooled into one m=6 family — this is a documented, conservative choice made *before* running the analysis, not selected after seeing results. | This report |
| Safety (Completion/Collision) | **Descriptive only, not Holm-corrected.** No established formal family for this exact WSC-vs-Original per-condition comparison exists anywhere in the project's prior protocol (`task3`'s non-inferiority margins compare GGI/Maximin against Mean within one information regime, a different design). Per the task instruction, when no established family exists this is reported as descriptive with raw bootstrap CIs/p-values only. | This report |
| Alpha | 0.05 | conventional, matches project usage elsewhere |

---

## D. Seed-level descriptive results

Full data: `wsc_v2_formal_seed_level.csv` (48 rows = 12 seeds × 4 conditions). Sanity check: this report's independently-recomputed n=12 point estimates (Mean −0.049, GGI +0.074, Maximin −0.018 for U_min interaction) match the earlier informal ad hoc estimates to 3 decimal places, confirming the informal exploration was not affected by a computational error — only by the absence of uncertainty quantification.

---

## E. U_min interaction analysis

### Primary (n=12)

| Condition | Mean | Median | SD | Pos/Neg | 95% CI | Raw p | Holm p | Reject α=.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Mean | −0.049 | −0.025 | 0.385 | 5/7 | [−0.236, +0.175] | 0.616 | 1.000 | No |
| GGI | +0.074 | +0.047 | 0.230 | 7/5 | [−0.037, +0.209] | 0.244 | 0.733 | No |
| Maximin | −0.018 | +0.002 | 0.230 | 6/6 | [−0.152, +0.097] | 0.788 | 1.000 | No |

### Sensitivity (n=11, excl. seed 910102)

| Condition | Mean | 95% CI | Holm p | Reject α=.05 |
|---|---:|---:|---:|---|
| Mean | −0.137 | [−0.278, −0.006] | 0.121 | No |
| GGI | +0.021 | [−0.061, +0.106] | 1.000 | No |
| Maximin | −0.007 | [−0.154, +0.114] | 1.000 | No |

No condition rejects the null at any stage. Figure 1 and Figure 5 (left panel) show all seed-level values and the CIs directly; per the task instructions, seed 910102 is shown, not hidden (highlighted in red in Figure 1).

---

## F. Utility Gini interaction analysis

(Negative = favorable, i.e. WSC makes the condition *more* fairness-improving relative to Baseline than it is under Original.)

### Primary (n=12)

| Condition | Mean | Median | SD | Pos/Neg | 95% CI | Raw p | Holm p | Reject α=.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Mean | +0.023 | +0.010 | 0.194 | 6/6 | [−0.089, +0.118] | 0.633 | 1.000 | No |
| GGI | −0.038 | −0.024 | 0.109 | 5/7 | [−0.102, +0.016] | 0.206 | 0.618 | No |
| Maximin | +0.013 | −0.003 | 0.114 | 6/6 | [−0.044, +0.079] | 0.699 | 1.000 | No |

### Sensitivity (n=11, excl. seed 910102)

| Condition | Mean | 95% CI | Holm p | Reject α=.05 |
|---|---:|---:|---:|---|
| Mean | +0.068 | [+0.002, +0.140] | 0.136 | No |
| GGI | −0.014 | [−0.056, +0.027] | 1.000 | No |
| Maximin | +0.006 | [−0.054, +0.078] | 1.000 | No |

**Gini shows the same qualitative pattern as U_min**: GGI is the only condition with a favorable-direction point estimate in the primary analysis, it does not survive Holm correction, and it weakens (from −0.038 to −0.014) once the highest-leverage seed is removed. Mean's Gini estimate is unfavorable (positive) in both n=12 and n=11, and the n=11 CI nominally excludes zero on the unfavorable side — again not Holm-significant, and not to be read as confirmed harm given the small sample and instability under leave-one-out (Section I).

---

## G. Completion analysis

Paired ΔCompletion = Completion(WSC) − Completion(Original), matched by seed, n=12, **descriptive** (see Section C):

| Condition | Mean paired Δ | Median | SD | Pos/Neg | 95% CI | Raw p |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | −0.027 | — | 0.127 | 2/10 | [−0.092, +0.044] | 0.438 |
| Mean | −0.074 | — | 0.311 | 3/9 | [−0.229, +0.108] | 0.398 |
| GGI | +0.053 | — | 0.153 | 6/6 | [−0.021, +0.144] | 0.194 |
| Maximin | −0.039 | — | 0.265 | 7/5 | [−0.193, +0.091] | 0.627 |

None reach even nominal significance (all raw p > 0.19). Baseline and Mean show more seeds with a completion *decrease* under WSC (10/12 and 9/12 respectively) but with effect sizes and CIs too wide to call this systematic. GGI and Maximin split roughly evenly.

---

## H. Collision analysis

Paired ΔCollision = Collision(WSC) − Collision(Original), matched by seed, n=12, descriptive:

| Condition | Mean paired Δ | SD | Pos/Neg | 95% CI | Raw p |
|---|---:|---:|---:|---:|---:|
| Baseline | +0.027 | 0.127 | 10/2 | [−0.044, +0.092] | 0.438 |
| Mean | +0.070 | 0.307 | 9/3 | [−0.110, +0.221] | 0.417 |
| GGI | −0.048 | 0.135 | 6/6 | [−0.126, +0.021] | 0.195 |
| Maximin | +0.059 | 0.246 | 5/7 | [−0.057, +0.203] | 0.394 |

Mirror image of Completion (as expected, since collision is the dominant failure mode). **Interpretation per the requested taxonomy (Section 10)**: the dominant finding is **(3) little mean change but increased/condition-specific dispersion**, not (1) systematic degradation or (2) systematic improvement. Whole-condition mean shifts are all statistically indistinguishable from zero; what is real is the *spread*.

---

## I. Seed heterogeneity and leave-one-out analysis

### Dispersion in paired ΔCompletion / ΔCollision (n=12)

| Condition | ΔCompletion SD | ΔCompletion range | ΔCompletion IQR | ΔCollision SD | ΔCollision range |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.127 | [−0.219, +0.223] | 0.103 | 0.127 | [−0.223, +0.219] |
| Mean | 0.311 | [−0.574, +0.715] | 0.224 | 0.307 | [−0.715, +0.547] |
| GGI | 0.153 | [−0.156, +0.445] | 0.142 | 0.135 | [−0.367, +0.156] |
| Maximin | 0.265 | [−0.645, +0.305] | 0.186 | 0.246 | [−0.223, +0.645] |

### Across-seed dispersion of raw Completion, Original vs. WSC (exploratory, not an inferential family)

| Condition | Completion SD (Original) | Completion SD (WSC) | WSC/Original ratio |
|---|---:|---:|---:|
| Baseline | 0.134 | 0.135 | 1.00× |
| Mean | 0.274 | 0.199 | 0.73× |
| GGI | 0.139 | 0.073 | **0.53×** |
| Maximin | 0.133 | 0.260 | **1.95×** |

**This is a condition-specific effect (taxonomy option 4), not a uniform increase in instability.** WSC roughly *halves* across-seed completion dispersion under GGI, leaves Baseline's dispersion unchanged, and roughly *doubles* it under Maximin. This is a final held-out-performance dispersion comparison; it says nothing about within-training temporal instability (see caveat below).

### Leave-one-seed-out influence (interaction effects)

| Outcome | Condition | Full n=12 estimate | Highest-leverage seed | Shift when excluded | Leave-one-out range |
|---|---|---:|---|---:|---:|
| U_min | Mean | −0.049 | **910102** | −0.088 | [−0.137, −0.009] |
| U_min | GGI | +0.074 | **910102** | −0.052 | [+0.021, +0.096] |
| U_min | Maximin | −0.018 | **920102** | +0.053 | [−0.052, +0.034] |
| Gini | Mean | +0.023 | **910102** | +0.044 | [+0.003, +0.068] |
| Gini | GGI | −0.038 | **910102** | +0.024 | [−0.050, −0.014] |
| Gini | Maximin | +0.013 | **920102** | −0.026 | [−0.013, +0.030] |

Full per-seed leave-one-out values: `wsc_v2_leave_one_seed_out.csv`. **Seed 910102 is the dominant influence point for Mean and GGI on both outcomes** (its known, pre-existing pathological Original-Mean result, documented before WSC ever existed — completion 0.035 — mechanically inflates the interaction term whenever it is included). **Seed 920102 is the dominant influence point for Maximin on both outcomes** (a WSC-side, not Original-side, low-completion run — checked in Section K and found technically valid).

### Training-dynamics vs. held-out dispersion (explicit distinction, per task instruction)

The dispersion comparison above is about **final held-out (H1) performance** only. A separate, exploratory look at the seven previously-flagged low-completion runs' training-window telemetry (from each run's own manifest `checkpoints` list, sampled every 50,000 steps) shows within-run window-to-window completion swings of roughly 0.3–0.4 (e.g. `mean_wsc_920102`: window completion ranging 0.31–0.81 across the 800K-step continuation) that do **not** monotonically settle by step 2,000,000. This is reported descriptively only — it was not a pre-specified metric, no formal inferential test was run on it, and it must not be read as a confirmed "WSC increases training instability" claim, only as a plausible partial explanation for why final held-out dispersion is elevated for some (not all) conditions.

---

## J. Seed 910102 sensitivity analysis

Seed 910102 was **not excluded from the primary analysis** (n=12 throughout Sections E–H). It is heterogeneous/pathological under the Original Mean condition (documented pre-existing anomaly, completion 0.035, matching the previously-recorded ΔU_min = −0.678 figure) but is not a corrupted run and was part of the frozen Original experiment — per instruction, it is retained.

Side-by-side comparison (`wsc_v2_formal_sensitivity_910102.csv`):

| Outcome | Condition | n=12 mean [CI] | n=11 mean [CI] | Point-estimate change | Direction changed? |
|---|---|---|---|---:|---|
| U_min | Mean | −0.049 [−0.236, +0.175] | −0.137 [−0.278, −0.006] | −0.088 | No (both negative) |
| U_min | GGI | +0.074 [−0.037, +0.209] | +0.021 [−0.061, +0.106] | −0.052 | No (both positive, much weaker) |
| U_min | Maximin | −0.018 [−0.152, +0.097] | −0.007 [−0.154, +0.114] | +0.011 | No |
| Gini | Mean | +0.023 [−0.089, +0.118] | +0.068 [+0.002, +0.140] | +0.044 | No |
| Gini | GGI | −0.038 [−0.102, +0.016] | −0.014 [−0.056, +0.027] | +0.024 | No (both negative, much weaker) |
| Gini | Maximin | +0.013 [−0.044, +0.079] | +0.006 [−0.054, +0.078] | −0.007 | No |

No sign flips occur for any condition/outcome, but the **magnitude** of GGI's favorable-looking estimate shrinks by two-thirds to three-quarters when this one seed is removed, and Mean's unfavorable estimate nearly triples in magnitude. **Neither the n=12 nor the n=11 estimate should be treated as more "correct" than the other** — n=12 is the pre-registered primary estimate; n=11 is reported strictly as a sensitivity check showing that the primary estimate's magnitude (not its qualitative sign, in this case) is highly dependent on a single seed.

---

## K. Technical health audit

Automated audit (`wsc_v2_formal_analysis.py`'s `audit_wsc_v2_runs()`) checked, for all 48 WSC v2 runs: registry status/exit code, checkpoint-ensemble completeness (4/4 present), source-checkpoint provenance (must be that seed's own `ckpt_step_1200000.pt`, must not contain "wsc" in its path, must not be under the invalid v1 root), and output-directory root. **Result: 48/48 OK, 0 problems.**

Seven runs were individually flagged during the earlier informal monitoring for held-out completion < 0.55 and were re-examined here against exit code and training-window telemetry (not re-examined against fairness outcomes):

| Regime | Condition | Seed | Completion | Exit code | stderr | Training-window pattern | Verdict |
|---|---|---|---:|---|---|---|---|
| Original | Mean | 910102 | 0.035 | — (pre-existing, pre-WSC data) | — | Known, pre-existing anomaly (documented before WSC existed) | Not a WSC/technical issue |
| Original | GGI | 910102 | 0.535 | — | — | Same seed's pre-existing relative weakness | Not a WSC/technical issue |
| WSC | Baseline | 910102 | 0.496 | 0 | clean | 0.38–0.67 window completion, no persistent >90% collision | Valid, noisy |
| WSC | Mean | 900103 | 0.402 | 0 | clean | 0.56–0.78 window completion | Valid, noisy |
| WSC | Maximin | 910102 | 0.230 | 0 | clean | 0.21–0.73 window completion | Valid, noisy |
| WSC | Mean | 920102 | 0.547 | 0 | clean | 0.31–0.81 window completion | Valid, noisy |
| WSC | Maximin | 920102 | 0.340 | 0 | clean | 0.37–0.69 window completion | Valid, noisy |

None of the six WSC-side flagged runs show the previously-fixed bug's signature (near-total, ~90% collision, persistent from step 0 through step 2,000,000 with no recovery). All are exit-code-0, exception-free, and fluctuate within a broad-but-bounded range across the full 800,000-step continuation. **No technical invalidity was found in any of the 48 runs.**

---

## L. Interpretation

**Supported findings:**
- All 48 WSC v2 runs are technically valid; the semantic-column-mapping bug has not recurred.
- No fairness interaction (U_min or Gini) for any welfare condition is statistically distinguishable from zero after Holm correction, at n=12 or n=11.
- No paired Completion/Collision shift for any condition is statistically distinguishable from zero.
- Seed-level heterogeneity is large and condition-specific, not a uniform WSC "cost" or "benefit."
- The apparent GGI U_min/Gini "positive interaction" signal from informal mid-campaign monitoring is confirmed to be real in the raw numbers but is driven almost entirely by one high-leverage seed and does not survive formal uncertainty quantification.

**Unsupported hypotheses:**
- "WSC improves fairness under any welfare condition" — not supported by this data.
- "WSC systematically degrades task completion" — not supported; the paired Completion analysis is null on average, with heterogeneous, condition-specific direction.
- "WSC increases training instability" as a general claim — only a descriptive, non-inferential observation exists (Section I); it has not been formally tested and should not be stated as established.

**Exploratory observations (not to be treated as findings):**
- GGI's across-seed completion dispersion drops sharply under WSC (0.53×) while Maximin's rises sharply (1.95×) — an intriguing, condition-specific pattern that was not pre-specified and has no inferential test attached.
- Training-window telemetry for several seed×condition combinations shows large mid-training swings that do not fully settle by step 2,000,000 — worth a dedicated, pre-specified training-dynamics analysis in future work, not asserted here as a formal result.

---

## M. Recommendation on additional seeds

**No new training data was generated to produce this recommendation**, per the task's explicit prohibition.

Applying the pre-specified decision logic:

- Estimates are **not** clearly near-zero with narrow CIs (the "no new training" criterion) — CIs are wide (e.g. U_min GGI n=12: [−0.037, +0.209], width 0.246) — so precision, not a confirmed null, is part of the picture.
- But conclusions are **also not stable under sensitivity/leave-one-out** — GGI's point estimate shrinks by ~70% and Mean's roughly triples in magnitude depending on one seed — which is the disqualifying condition for "additional seeds are justified" (that criterion requires a *consistent* effect direction with uncertainty *dominated by seed count*, not by one or two idiosyncratic seeds).

**Conclusion: neither branch of the pre-specified logic is fully satisfied. This itself is diagnostic** — it means the current n=12 seed-level variance is dominated by a small number of unusually extreme seeds rather than by generically-expected sampling noise, which additional *ordinary* seeds would not obviously fix (each new seed carries a non-trivial chance of being another high-leverage outlier, given two of twelve already are).

**Prospective power calculation (planning-only, using observed n=12 variance, no seeds launched):**

Using the observed seed-level SD of the U_min interaction (largest: Mean, SD≈0.385) and a two-sided paired bootstrap-equivalent test at α=0.05, achieving a CI half-width of ±0.10 around a point estimate would require approximately:

```
required_n ≈ (1.96 * SD / target_half_width)^2
Mean:    (1.96 * 0.385 / 0.10)^2 ≈ 57 seeds
GGI:     (1.96 * 0.230 / 0.10)^2 ≈ 20 seeds
Maximin: (1.96 * 0.230 / 0.10)^2 ≈ 20 seeds
```

These counts are large relative to the current formal protocol's 12-seed convention and are dominated by the high-leverage seeds identified in Section I — i.e., precision would come primarily from diluting a small number of extreme values, not from a generically "truer" estimate. **This further argues against an ad hoc "add seeds until significant" approach** (explicitly prohibited by the task instructions) and in favor of treating the current 12-seed result as the frozen, complete answer for this thesis unless a future, separately-designed, pre-registered expanded-seed study is planned with its own new frozen seed list and full matched Original/WSC re-collection.

**Recommendation: do not launch additional seeds for this comparison.** If a future study revisits this question, it should pre-register a target sample size using the above calculation (or a refined one), use entirely new frozen seeds, and preserve the matched Original/WSC design end to end.

---

## N. Thesis-ready factual summary

- The corrected WSC v2 campaign (12 seeds × 4 conditions, 48 runs) completed with no technical failures; all runs are traceable to their original 1.2M-step C64 checkpoint and use the corrected semantic observation-to-network mapping.
- Under the project's established seed-level paired bootstrap protocol (10,000 resamples, seed=0, 95% percentile CIs, two-sided bootstrap p-values, Holm correction within outcome across {Mean, GGI, Maximin}), **no interaction effect of local welfare-state observability (WSC) on the U_min or Utility Gini benefit of Mean, GGI, or Maximin (each relative to Baseline) reached statistical significance**, at n=12 (primary) or n=11 (sensitivity excluding seed 910102).
- GGI showed the numerically most favorable point estimate for both U_min (+0.074) and Gini (−0.038) in the primary analysis, but this estimate is driven predominantly by a single seed (910102) and shrinks substantially (to +0.021 / −0.014) when that seed is excluded; it does not survive multiple-comparison correction in either case.
- Mean showed an unfavorable point estimate for both outcomes that strengthens (in magnitude) once the same seed is excluded, but likewise does not survive correction.
- Maximin's interaction estimates are small in magnitude and inconsistent in sign across seeds and across the primary/sensitivity split.
- Paired Completion and Collision differences (WSC vs. Original) show no statistically distinguishable mean shift for any of the four conditions; the dominant pattern is large, condition-specific seed heterogeneity (e.g. Maximin completion deltas span −0.645 to +0.305 across the 12 seeds) rather than a uniform competence cost or benefit.
- All 48 WSC v2 runs passed a full technical-health audit (exit codes, checkpoint completeness, source-checkpoint provenance, frozen-code hash re-verification); no run's poor performance was attributable to a technical failure or a recurrence of the previously-fixed checkpoint-mapping bug.
- Given the combination of wide confidence intervals and high seed-to-seed leverage (not merely generic sampling noise), the evidence is assessed as **inconclusive tending toward null** for the hypothesis that local welfare-state observability makes welfare-oriented rewards more effective at improving worst-off outcomes or reducing inequality relative to Baseline. Additional seeds are not recommended without a separately pre-registered, larger-sample follow-up study.

---

## Files produced

**Machine-readable** (`F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_formal\outputs\`):
`wsc_v2_formal_seed_level.csv`, `wsc_v2_formal_fairness_summary.csv`, `wsc_v2_formal_safety_summary.csv`, `wsc_v2_formal_sensitivity_910102.csv`, `wsc_v2_leave_one_seed_out.csv`, `wsc_v2_formal_bootstrap_results.json`, `wsc_v2_data_provenance.json`, `wsc_v2_formal_analysis_log.txt`

**Figures** (`F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_formal\figures\`, PNG+PDF):
`fig1_umin_interaction_by_seed`, `fig2_gini_interaction_by_seed`, `fig3_completion_paired`, `fig4_collision_paired`, `fig5_interaction_forest`

**Analysis scripts** (reproducible, deterministic, fail loudly on missing seeds/duplicate rows/non-finite values):
`F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_formal\wsc_v2_formal_analysis.py`, `wsc_v2_formal_figures.py`
