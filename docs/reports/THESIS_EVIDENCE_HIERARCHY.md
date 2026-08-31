# Thesis Evidence Hierarchy

Isolated analysis root: `C:\dense reward\outputs\whole_thesis_evidence_synthesis_v1\`.
Inferential unit throughout: **training seed** (n=12 unless noted). Episodes are repeated evaluations of one policy, not replicates.

---

## Tier 1 — Core claims

Safe for Abstract / Conclusion. Each claim uses a pre-specified primary contrast or a descriptive fact that does not depend on post-hoc metric search.

### T1.1 Successful coordination can still leave a worse-off vehicle

| Field | Content |
|---|---|
| Claim | Under Original Baseline, mean held-out worst-off utility is below mean utility (U_min = 0.882 vs mean U = 0.941; Utility Gini = 0.058). |
| Exact numbers | Table `condition_absolute_means.csv`; thesis Table 5.5 (same sources: `taskonly_evaluation_merged.csv`, H1). |
| Source | `F:\正式训练\outputs\welfare_analysis\taskonly_evaluation_merged.csv` |
| Uncertainty | Seed range is wide (U_min from 0.652 to 1.000). Collision rate 0.116. |
| Caveat | Seed-level U_min/Gini mix successful and failed episodes. Failed episodes set U_i=0 and inflate Gini. Successful-episode below-target burden is smaller (Baseline success burden 0.024; `table5_9_outcome_decomposition_seedlevel.md`). |
| Recommended wording | “Task-only coordination leaves a lower tail: mean U_min is below mean utility, and some seeds fail a large fraction of held-out merges.” |
| Avoid | “Baseline is systematically unfair to Slow (or Fast) vehicles even in successful merges.” Class gaps on mixed episodes are small (Fast 0.938 vs Slow 0.944). |

### T1.2 Stronger inequality aversion does not improve realised fairness monotonically

| Field | Content |
|---|---|
| Claim | Mean, GGI, and Maximin vs matched Baseline: all U_min and Gini CIs include 0. Across-seed means are Mean 0.902 / GGI 0.895 / Maximin 0.874 (U_min), not increasing with inequality aversion. 0/12 seeds are fully monotonic on the hypothesised order. |
| Exact numbers | Mean−Baseline ΔU_min +0.020, 95% CI [−0.129, +0.139], Holm p = 1.00. GGI−Baseline +0.013 [−0.054, +0.093]. Maximin−Baseline −0.008 [−0.078, +0.062]. GGI−Mean ΔU_min −0.007 [−0.087, +0.103] (matches `pooled12_umin_holm.csv`). Adjacent-order agreement 16/36 = 44%. |
| Source | `cross_experiment_contrasts.csv`, `objective_strength_ordering.csv`; thesis Tables 5.4–5.5. |
| Uncertainty | Seed 910102 is an extreme Mean collapse (completion 0.035). Leave-one-out can flip Mean−Baseline sign. |
| Caveat | RQ2 in the thesis did not pre-register Holm on the three vs-Baseline contrasts; Holm p=1.00 is a synthesis check, not a new confirmatory family invented to rescue a result. |
| Recommended wording | “Increasing the objective’s inequality aversion from Mean to GGI to Maximin did not produce a matching improvement in held-out U_min or Utility Gini.” |
| Avoid | “Maximin is worse than Mean” as a confirmed ranking. The pairwise CIs include 0. |

### T1.3 Welfare-state observability does not reliably improve final fairness

| Field | Content |
|---|---|
| Claim | Reward-by-information interactions for Mean, GGI, Maximin on U_min and Gini are all Holm-nonsignificant (all Holm p ≥ 0.62). |
| Exact numbers | U_min I: Mean −0.049 [−0.236, +0.175] Holm 1.00; GGI +0.074 [−0.037, +0.209] Holm 0.73; Maximin −0.018 [−0.152, +0.097] Holm 1.00. Gini I: Mean +0.023; GGI −0.038; Maximin +0.013 (same source). |
| Source | `F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_formal\outputs\wsc_v2_formal_fairness_summary.csv`; `WSC_V2_FORMAL_EVALUATION_AND_SAFETY_REPORT.md`. |
| Uncertainty | GGI’s favourable points shrink toward 0 if seed 910102 is dropped. Mean’s n=11 intervals exclude 0 but Holm p = 0.12 / 0.14. |
| Caveat | Do not call this equivalence. CIs are wide. Behavioural yielding point estimates can move without an outcome interaction. |
| Recommended wording | “Adding local running-welfare features did not produce a Holm-supported increase in the fairness effect of Mean, GGI, or Maximin.” |
| Avoid | “WSC does nothing.” “WSC harms Mean” (n=11 nominal intervals are sensitivity, not primary). |

### T1.4 Dense shared Maximin feedback can worsen fairness and task performance when welfare is hidden

| Field | Content |
|---|---|
| Claim | Original: Maximin+DWS vs terminal Maximin. Holm-significant harm on U_min and Utility Gini; completion down and collision up; direction stable in every leave-one-seed-out. |
| Exact numbers | ΔU_min −0.140 [−0.258, −0.033], raw p 0.0068, Holm p **0.0136**, 3/12 favourable. ΔGini +0.073 [+0.020, +0.133], Holm p **0.0096**. Δcompletion −0.153 [−0.279, −0.033]. Δcollision +0.172 [+0.058, +0.290]. |
| Source | `C:\dense reward\outputs\dws_final_reevaluation_v1\dws_primary_fairness_summary.csv`, `dws_task_safety_summary.csv`, `DWS_FINAL_REEVALUATION_REPORT.md`. |
| Uncertainty | n=12. Harm is joint with task collapse, not a pure redistribution. |
| Caveat | DWS is not potential-based shaping and is not agent-specific credit. WSC+DWS is **not** a confirmed benefit (Holm p ≈ 0.70). DWS×WSC interaction CIs include 0. |
| Recommended wording | “A shared, step-wise Maximin signal reduced worst-off utility and raised Utility Gini when the policy could not observe the welfare state, and it did so together with more collisions.” |
| Avoid | “DWS always fails.” “WSC protects against DWS” as a confirmed interaction. “This proves credit assignment is the cause.” |

---

## Tier 2 — Supporting mechanism claims

Suitable for Discussion with cautious wording.

### T2.1 Information and reward timing look coupled, but the interaction is not confirmed

DWS×WSC I(U_min) = +0.110 [−0.068, +0.306]; I(Gini) = −0.058 [−0.158, +0.033]. Direction is leave-one-out stable. **Do not call moderation confirmed.**

### T2.2 Running Maximin is not the same object as terminal failure-aware U_min

Under WSC, reconstructed net shaping-event balance moves favourably for DWS-trained trajectories (+0.159) while final U_min/Gini stay null. Mid-episode Phi (p50/p90) does not significantly track the Original terminal harm (`running_vs_terminal_contrasts.csv`). M_i(t) does not zero on collision the way U_i does.

### T2.3 Shared F_t cannot name the responsible vehicle

On negative-DeltaPhi steps, one vehicle often accounts for the M drop (Cell 1 single-decliner shares commonly 0.4–0.8) while all four receive F_t. The largest M decline frequently is not the strongest control action. DWS does not change this structure (`dws_shared_credit_summary.md`).

### T2.4 Reconstructed (x, M) neighbourhoods have less DWS-sign disagreement when M is included

Predeclared proxy, k ∈ {10,25,50} not tuned. At k=25, Cell 1 sign-disagreement 0.190 (x only) vs 0.070 (x+M). Same direction in all four cells. **This is not the 18D/22D policy-observation analysis** — observation vectors were not dumped. Figure: `figures/aliasing_original_vs_wsc_proxy.png`.

### T2.5 WSC can change local yielding point estimates without changing U_min

WSC behavioural interactions for RY: Mean +0.75 (raw p 0.10), GGI ≈ 0, Maximin −0.40; Holm-nonsignificant. Thesis text: Baseline-referenced RY interactions +0.964 / +0.804 / +0.125, Holm p 0.57 / 0.27 / 0.90.

### T2.6 Evaluation-time brakes are not more often paired with F<0 after Original DWS

P(F<0|BRAKE) falls by −0.182 [−0.354, −0.051]. Do not use this as a confirmatory anti-yield proof, and do not flip it into “DWS protects yielding.”

---

## Tier 3 — Exploratory observations

Limitations / future work only.

- Merge-priority allocation is too sparse to test in WSC and DWS (often 2/12 finite seeds).
- Braking-burden (acceleration-integral) was not recovered as a reusable per-seed column; left blank in the master table.
- DWS Phi/event rates were never computed for Baseline/Mean/GGI.
- Learning-dynamics along 1.2–2.0M were not re-run for this synthesis.
- Seed 910102 is a substantive instability example, not a reason to drop a valid seed.
- High-completion cells (completion ≥ 0.90, n=70 seed-condition cells) include only **one** with Gini ≥ 0.05. “Unfair but competent” is rare at this aggregation; most Gini movement is collision-tied (Spearman Gini–collision = 0.99).
- 18D vs 22D observation-space aliasing: **NOT ESTIMABLE** from stored trajectories.

---

## Ranking of candidate contributions (prompt §13)

| Rank | Candidate | Empirical | Novelty | Theory | Robustness | Overclaim risk |
|---:|---|---|---|---|---|---|
| 1 | Dense shared feedback can harm under hidden welfare | Strong | High | High | LOO-stable, Holm-sig | Calling it proven credit failure |
| 2 | Stronger aversion ≠ better realised fairness | Strong | Medium | High | Matches frozen RQ2 | Ranking Maximin as inferior |
| 3 | Observability alone is insufficient | Strong (null) | Medium | High | Holm family pre-specified | Calling a confirmed null / equivalence |
| 4 | Objective, information, and timing must align | Moderate | High | High | Interaction CI includes 0 | “Confirmed complementarity” |
| 5 | Structured inequality despite success | Moderate | Medium | High | Success-only burden is small | Treating collision-Gini as unfairness |
| 6 | Running welfare ≠ terminal U_min | Moderate | Medium | High | WSC signal vs outcome split | Causal mediation |
| 7 | Shared signal lacks agent-specific credit | Suggestive | Medium | High | True by design | “Experimentally proven cause” |
