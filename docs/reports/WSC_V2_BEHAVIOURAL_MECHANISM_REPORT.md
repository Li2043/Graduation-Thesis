# WSC v2 Behavioural Mechanism Report

**Evaluation-only.** No training was launched, resumed, or modified. All 96 (seed × condition × regime) checkpoint combinations were read-only inputs. WSC-specific gradient warmup, reward definitions, GGI weights, welfare lambda, checkpoint expansion, and the H1 scenario bank were all unchanged.

---

## A. Executive summary

- **Welfare-responsive yielding (RY) is the one primary mechanism metric where a corrected implementation bug now yields a real, non-trivial signal**: an early version of the event-extraction script never incremented the "yielded while neighbour is not worse-off" counter, making RY uncomputable (all-NaN) in a first pass. After the fix, RY interaction point estimates are **positive for all three welfare conditions** (Mean +0.96, GGI +0.80, Maximin +0.12) — i.e. WSC is associated with *more* selective yielding toward worse-off neighbours specifically for Mean and GGI — but **none survive Holm correction** (Mean holm p=0.57, GGI holm p=0.27, Maximin holm p=0.90).
- **Merge-priority allocation (P_priority_worse)** could only be estimated for GGI and Maximin (Baseline and Mean had too few seeds with a resolvable non-tied priority pair for a bootstrap CI); neither shows a supported WSC effect.
- **Burden transfer (BC)** and **worst-off gap closure (GapClosure_k25)** show no supported WSC effect for any condition; point estimates are noisy and CIs are wide.
- **Behavioural uptake is present but weak and condition-dependent**: even under the Original (non-WSC) regime, yielding is already somewhat welfare-responsive (median RY > 1 for most seed×condition cells), consistent with welfare state correlating with the already-observable relative-speed feature (Δv) rather than requiring the explicit M_i/M_j channel. WSC's marginal contribution on top of that baseline is the (unsupported) RY interaction described above.
- **High-leverage seeds for behavioural metrics differ from the high-leverage seeds identified in the outcome-level formal report.** Seed 910102 and 920102 (influential for the outcome-level U_min/Gini interactions) are **not** the dominant influence points here; RY/BC leave-one-out is instead dominated by seeds 900103, 900102, 920104/920105.
- **Best-supported classification (Section M): Pattern A/B mixture — weak, condition-dependent behavioural uptake without a statistically supported aggregate fairness improvement**, consistent with, and not contradicting, the outcome-level formal report's inconclusive-tending-null finding.

---

## B. Scientific motivation

The formal outcome-level WSC report found no statistically supported reward × information interaction on U_min or Utility Gini for any of Mean/GGI/Maximin (all Holm-adjusted p ≥ 0.12, primary n=12). This mechanism analysis asks the logically prior question: did WSC change *how* agents behave toward locally disadvantaged neighbours at all, even absent a supported aggregate fairness effect? Three candidate readings were pre-specified: (1) no behavioural uptake, (2) uptake without outcome improvement, (3) localized/condition-specific redistribution.

---

## C. Behavioural metric definitions (frozen before any result was inspected)

Full definitions in `wsc_behavioural_metric_definitions.json`. Summary: yielding opportunity = both vehicles active, |Δx|≤50m (R_OBS), i not yet past the merge zone; yield = discrete action==DECELERATE while an opportunity exists; welfare gap = M_i(t)−M_j(t) using the exact `running_active_attainment` function (identical for Original and WSC regimes, computed offline for Original); merge priority = who exits x=380 first among a pair, relative to their welfare state at first co-occurrence; costly cooperative action = the project's existing frozen hard-brake definition (accel ≤ −3.0 m/s², contiguous events); worst-off identity = tie-corrected argmin of M_i(t) (reusing the project's existing 1e-9 tie tolerance and fractional-weighting convention); recovery horizons k∈{10,25,50} steps, pre-specified from the environment's own merge-window traversal time. No "gap creation" metric is defined (not found anywhere in the repository; out of scope per the definitions file).

**Correction note**: the yielding-opportunity counters (`opp_better`/`yield_better`, i.e. events where the neighbour is *not* worse-off) were implemented but the "yielded" increment for that branch was accidentally omitted in the first version of `wsc_v2_behavioural_run.py`, making `P(yield|not worse-off)` identically zero and RY undefined (NaN) for all 96 combinations. This was caught during aggregation (implausible: `opp_better` counts were large — tens of thousands per combination — while `yield_better` was exactly zero in every single row), fixed, verified on a 1-seed smoke test (yield_better became non-zero, consistent with baseline expectations), and all 4 shards (12 seeds) were re-run from scratch under the corrected code before any number in this report was computed.

---

## D. Data provenance and trajectory logging

- Original checkpoints: `checkpoint_paths_for()` copied verbatim from `F:\正式训练\scripts\evaluate_behavioral_window.py`. WSC v2 checkpoints: `wsc_formal_runs_v2\{cond}_wsc_{seed}\...` (the corrected, formally-frozen v2 campaign).
- M_i/M_j for **both** regimes are read via `running_active_attainment(env._traces[vid])` — the wrapper's own internal trace, populated unconditionally regardless of `include_welfare_state`. Original policies never receive M_i/M_j as input (`include_welfare_state=False` for every Original run); M_i/M_j are computed identically for both regimes purely for analysis.
- Performance: a naive per-step call to `running_active_attainment` (which rebuilds the full trace history each call, O(T) per call) made the naive implementation impractically slow (>350 CPU-seconds without finishing even one 256-episode combination). Replaced with a mathematically identical O(1) incremental accumulator, verified against the reference function on the first episode of the first combination of every run (`[validation] incremental M_i matched running_active_attainment() exactly for episode 0` — printed and confirmed in every shard's log).
- **Sanity check**: completion/collision rates recomputed by this script for seed 900101 exactly match the already-validated formal evaluation numbers (e.g. Baseline: 0.996/0.004 Original, 0.859/0.141 WSC — identical to the formal report), confirming the trajectory logger reproduces the established evaluation pipeline correctly before any new behavioural quantity is trusted.
- No raw per-timestep table is persisted (see engineering-note in `wsc_v2_behavioural_run.py`'s docstring): opportunity/yield/priority/burden/recovery counters are aggregated online per (seed, condition, regime, group) to avoid ~10-15 million pseudo-replicated rows; `wsc_behavioural_events.csv` (delivered as 4 shard files, `wsc_behavioural_events_shard{A,B,C,D}.csv`) contains these aggregated counts, one row per (seed, condition, regime, group).

---

## E. Welfare-responsive yielding (RY)

Primary (n=12), WSC − Original paired delta, ALL group:

| Condition | Delta (WSC−Original) | 95% CI | raw p |
|---|---:|---|---:|
| Baseline | −0.464 | [−1.427, +0.530] | 0.357 |
| Mean | +0.751 | [−0.105, +1.816] | 0.101 |
| GGI | −0.006 | [−0.338, +0.344] | 0.920 |
| Maximin | −0.403 | [−1.246, +0.566] | 0.376 |

Interaction (relative to Baseline's own WSC−Original delta), primary n=12:

| Condition | Interaction mean | Holm p |
|---|---:|---:|
| Mean | +0.964 | 0.570 |
| GGI | +0.804 | 0.269 |
| Maximin | +0.125 | 0.897 |

No condition survives Holm correction. Direction is positive for all three welfare conditions (unlike the outcome-level analysis, where Mean was unfavourable) — an interesting but statistically unsupported divergence between the behavioural and outcome-level pictures.

**Event sufficiency**: opportunity counts are large (tens of thousands per combination); yield counts range from single digits to several thousand. See `wsc_behavioural_seed_summary.csv` for exact per-seed counts — no seed was dropped.

---

## F. Merge-priority allocation

Baseline and Mean had fewer than 3 seeds with a usable (non-tied, both-exited) resolvable pair for at least one regime, so no primary bootstrap CI could be formed for those two conditions (reported as `insufficient finite seeds` in `wsc_behavioural_primary_effects.csv`, not silently dropped). For GGI and Maximin:

| Condition | Delta (WSC−Original) | 95% CI | raw p |
|---|---:|---|---:|
| GGI | −0.010 | [−1.000, +0.971] | 0.811 |
| Maximin | +0.111 | [0.000, +0.333] | 0.591 |

Both wide, both cross zero. No interaction Holm family could be validly formed (requires all 3 conditions to have ≥3 finite seeds; Mean/Baseline fail this).

---

## G. Burden transfer / cooperative sacrifice (BC)

| Condition | Delta (WSC−Original) | 95% CI | raw p |
|---|---:|---|---:|
| Baseline | +10.98 | [−1.85, +32.76] | 0.273 |
| Mean | +2.79 | [−0.62, +8.07] | 0.212 |
| GGI | −1.64 | [−4.15, +0.55] | 0.158 |
| Maximin | −0.24 | [−1.99, +1.74] | 0.762 |

Interaction (n=12): Mean −1.54 (holm p=0.898), GGI −2.91 (holm p=0.898), Maximin −14.78 (holm p=0.537). Extremely wide CIs driven by sparse hard-brake-event counts for some seeds; no supported effect for any condition.

---

## H. Worst-off recovery (GapClosure, k=25 primary)

| Condition | Delta (WSC−Original) | 95% CI | raw p |
|---|---:|---|---:|
| Baseline | −0.002 | [−0.014, +0.010] | 0.718 |
| Mean | +0.002 | [−0.005, +0.007] | 0.527 |
| GGI | +0.0001 | [−0.014, +0.010] | 0.912 |
| Maximin | −0.0002 | [−0.016, +0.017] | 0.980 |

All near zero, all CIs narrow-ish and centred on zero — the closest this analysis comes to a **precise null** (Section 15-style classification: option B, precise null / practically small effect) rather than an inconclusive-due-to-imprecision result. k=10 and k=50 (secondary/exploratory, not Holm-corrected) show the same pattern; see `wsc_behavioural_primary_effects.csv`.

---

## I. Group-specific behavioural redistribution (descriptive)

Full per-group breakdowns (role-speed_class × yielding performed/received, priority received, burden borne, worst-off persistence) are in `wsc_behavioural_group_analysis.csv` (384 rows). Given the scale of this report and that none of the aggregate (ALL-group) primary effects are statistically supported, group-level splits are reported as exploratory data only; no group-level number is treated as confirmatory here, consistent with Section 16's instruction not to apply significance language to exploratory findings.

---

## J. Reward × information behavioural interactions

Summarised in Sections E–H above; full table in `wsc_behavioural_interactions.csv`. Holm families: one per primary metric (RY, P_priority_worse, BC, GapClosure_k25), across {Mean, GGI, Maximin}, mirroring the outcome-level report's convention. **No interaction is supported for any primary behavioural metric after correction.**

---

## K. Seed heterogeneity and leave-one-out

| Metric | Condition | Full n=12 estimate | Highest-leverage seed | Shift |
|---|---|---:|---|---:|
| RY | Mean | +0.751 | 900103 | −0.472 |
| RY | GGI | −0.006 | 920105 | −0.118 |
| RY | Maximin | −0.403 | 900102 | −0.354 |
| BC | Mean | +2.785 | 900103 | −2.506 |
| BC | GGI | −1.640 | 920104 | +1.031 |
| BC | Maximin | −0.237 | 900103 | −0.711 |
| GapClosure_k25 | Maximin | −0.0002 | 910102 | −0.0065 |

**910102 and 920102 (the outcome-level high-leverage seeds) are generally NOT the dominant influence points for the behavioural metrics** — 900103, 900102, and 920104/920105 dominate instead. This is itself a notable finding: whatever drives the outcome-level fairness-interaction sensitivity to 910102/920102 is not the same mechanism driving behavioural-metric sensitivity, suggesting the two levels of analysis are picking up at least partially distinct sources of seed variability. Full leave-one-out table: `wsc_behavioural_leave_one_seed_out.csv`.

---

## L. Behaviour-to-outcome linkage

Descriptive only, no causal language: the direction of the RY interaction (favourable for Mean and GGI) is **not** consistent with the outcome-level report's Mean interaction (unfavourable point estimate for U_min/Gini) — i.e. more welfare-responsive yielding under WSC for Mean is *associated with* a worse, not better, aggregate outcome estimate in this data, though neither is statistically supported. This is consistent with Pattern B/C in the pre-registered taxonomy: local behavioural responsiveness, where it appears at all, does not translate cleanly into the aggregate fairness direction.

---

## M. Interpretation

**Classification: mixed / inconclusive mechanism evidence, leaning toward "behavioural uptake without aggregate fairness improvement" for yielding specifically, and "no detectable uptake" for priority allocation, burden transfer, and dynamic recovery.**

- Not "no behavioural uptake" outright: RY shows a consistent positive direction for Mean and GGI, and even Original-regime yielding is already somewhat welfare-correlated (a real, if unsurprising, behavioural fact — speed differences are directly observable without WSC).
- Not "uptake with outcome improvement": none of the four primary behavioural mechanisms is statistically supported after correction, and the one directionally interesting result (RY) points the opposite way from Mean's outcome-level estimate.
- Not "localized redistribution" in a clean sense: the group-level breakdowns (Section I) are exploratory and were not used to select or emphasize any particular group finding.
- Seed heterogeneity is again the dominant feature of the data, but interestingly with a **different set of high-leverage seeds** than at the outcome level.

---

## N. Thesis-ready factual summary

The behavioural mechanism analysis, covering welfare-responsive yielding, merge-priority allocation, cooperative burden transfer, and worst-off recovery across the twelve matched Original/WSC seed pairs, finds no statistically supported reward × information interaction for any of the four pre-specified primary mechanisms after Holm correction. Welfare-responsive yielding shows a positive (favourable) point estimate under WSC for the Mean and GGI conditions, but with wide, zero-crossing confidence intervals; worst-off gap closure is close to a precise null; merge-priority allocation and burden transfer are imprecisely estimated, in part due to sparse qualifying events for some seeds. The seeds with the greatest influence on the behavioural estimates differ from those most influential for the outcome-level fairness interactions, indicating that seed-level heterogeneity in this experiment operates at multiple, only partially overlapping levels. Taken together with the outcome-level formal report, the evidence does not support the hypothesis that local welfare-state observability produces a reliable behavioural or distributive fairness benefit in this decentralized merging setting.

---

## Files produced

**Machine-readable** (`F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_behavioural\outputs\`): `wsc_behavioural_events_shard{A,B,C,D}.csv`, `wsc_behavioural_episode_counts_shard{A,B,C,D}.csv`, `wsc_behavioural_seed_summary.csv`, `wsc_behavioural_group_analysis.csv`, `wsc_behavioural_primary_effects.csv`, `wsc_behavioural_interactions.csv`, `wsc_behavioural_leave_one_seed_out.csv`, `wsc_behavioural_bootstrap_results.json`, `wsc_behavioural_metric_definitions.json`, `wsc_behavioural_data_provenance.json`.

**Figures** (`F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_behavioural\figures\`): `fig1_RY_paired`, `fig2_priority_paired`, `fig3_burden_paired`, `fig4_gapclosure_paired`, `fig5_behavioural_interaction_forest` (PNG+PDF).

**Scripts**: `wsc_v2_behavioural_run.py` (trajectory/event extraction, corrected), `wsc_v2_behavioural_aggregate.py` (formal statistics), `wsc_v2_behavioural_figures.py`.
