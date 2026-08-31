# Thesis Evidence for ChatGPT

Use this file with `THESIS_EVIDENCE_HIERARCHY.md` and `DWS_THESIS_RESULTS_INPUT.md`. Do not reopen the codebase unless a number below is missing.

## A. Overall thesis finding

The thesis’s strongest *positive* empirical result is negative: **Original Maximin + Dense Welfare Shaping harms worst-off utility, Utility Gini, completion, and collision rate** (Holm-significant, leave-one-seed-out stable). The strongest *nulls* are also real: **Mean/GGI/Maximin do not beat Baseline on primary fairness**, and **WSC does not Holm-significantly amplify those objectives**. Reframe the thesis as an alignment problem (objective × observability × timing), not as a winning fairness reward.

## B. Tier-1 claims

| Claim | Evidence | Strength | Caveat |
|---|---|---|---|
| Baseline leaves a worse-off tail | U_min 0.882 vs mean U 0.941, Gini 0.058 (H1, 12 seeds) | Strong descriptive | Heavily mixed with collisions |
| Stronger aversion ≠ better U_min/Gini | Mean/GGI/Maximin vs Baseline CIs all include 0; means 0.902 / 0.895 / 0.874 | Strong null | 910102 collapses Mean |
| WSC does not amplify fairness outcomes | Interactions Holm p ≥ 0.62 for all 3×2 | Strong null | Wide CIs; not equivalence |
| Original+DWS harms fairness and safety | ΔU_min −0.140 Holm p=0.0136; ΔGini +0.073 Holm p=0.0096; Δcomp −0.153; Δcoll +0.172 | Strong | Original only; joint with crashes |

## C. Tier-2 mechanism claims

- DWS×WSC interaction +0.110 U_min / −0.058 Gini; CIs include 0; LOO-stable direction. Suggestive buffering, **not confirmed**.
- Running Phi / event balance can move without terminal U_min (especially WSC+DWS).
- Shared F_t never names the decliner; single-vehicle M drops are common.
- (x,M) k-NN proxy: adding M reduces F-sign disagreement. **Not** 18D/22D.
- WSC yielding point estimates can move; Holm-nonsignificant.
- Evaluation P(F<0\|BRAKE) falls under Original DWS — do not sell as anti-yield.

## D. Findings to de-emphasise

GGI+WSC as a “best cell”; n=11 Mean-WSC harm; merge priority; DWS RY/burden/recovery; learning curves (not redone); 18D/22D aliasing; unconditional Maximin burden.

## E. Key numbers

**Original absolute (12-seed H1 means)**

| Condition | U_min | Gini | Mean U | Completion | Collision |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.882 | 0.058 | 0.941 | 0.884 | 0.116 |
| Mean | 0.902 | 0.049 | 0.951 | 0.903 | 0.097 |
| GGI | 0.895 | 0.050 | 0.944 | 0.895 | 0.098 |
| Maximin | 0.874 | 0.058 | 0.938 | 0.872 | 0.108 |

**RQ2 paired vs Baseline (U_min):** Mean +0.020 [−0.129, +0.139]; GGI +0.013 [−0.054, +0.093]; Maximin −0.008 [−0.078, +0.062].

**WSC interactions (U_min):** Mean −0.049 Holm 1.00; GGI +0.074 Holm 0.73; Maximin −0.018 Holm 1.00.

**DWS four-cell:** Maximin 0.874 / 0.058 / 0.872 / 0.108; +DWS 0.734 / 0.132 / 0.718 / 0.280; +WSC 0.828 / 0.085 / 0.833 / 0.167; +WSC+DWS 0.799 / 0.100 / 0.803 / 0.197.

Seeds: 900101–900104, 910101–910102, 920101–920106. Bank H1, 256 scenarios, greedy final-four ensemble {1.85, 1.90, 1.95, 2.00}M.

## F. Mechanism audit results

- **Aliasing:** 18D/22D not estimable. Proxy: k=25 disagreement Cell1 0.190 (x) vs 0.070 (x+M).
- **Concession:** Original DWS ΔP(F<0\|BRAKE)= −0.182 [−0.354, −0.051]. Opposite of a simple anti-yield story at evaluation time.
- **Shared credit:** F shared by design. Cell 1 single-decliner often 0.4–0.8; actor≠largest decline often 0.3–0.8. DWS does not change this.
- **Running vs terminal:** mid-episode Phi not a significant Original-DWS drop; terminal U_min is. WSC signal can improve while U_min does not.
- **Actions:** Original DWS episodes 6.4 steps shorter; WSC DWS HOLD rate −0.17.
- **Seeds:** 910102 levers Mean and WSC-GGI; Original DWS harm is not that seed.

## G. Best theoretical contribution

**Strong:** shared welfare rewards are not sufficient local control signals; denser shared Maximin can hurt when welfare is hidden.

**Conservative fallback:** objective, observability, and timing do not automatically substitute; the only Holm-significant follow-up effect is Original+DWS harm.

## H. Suggested thesis narrative (7 sentences)

The study first asks whether a task-only four-agent merge leaves a worse-off vehicle. It does, but seed-level U_min and Gini move almost with completion and collision, so RQ1 must separate crashes from successful-episode burden. It then replaces the terminal objective with Mean, GGI, or Maximin on matched seeds; none reliably beats Baseline, and more inequality-averse objectives are not more fair. A WSC follow-up adds local running-welfare features without changing the reward; the reward-by-information interactions are Holm-nonsignificant. A DWS follow-up then densifies Maximin feedback as one shared step-wise signal. Under Original observations that signal lowers U_min and raises collisions; under WSC it does not significantly change either outcome. The programme therefore supports an alignment claim — stronger or denser fairness signals are not automatically beneficial — and does not support a claim that any one reward “worked.”

## I. Remaining limitations

n=12 for interactions; no observation-vector dump; U_i failure-aware by construction; one geometry; shared-global DWS never contrasted with agent-specific DWS; braking-burden column missing from the master table; DWS learning curves not re-run.

## J. Files to send ChatGPT

1. `C:\dense reward\outputs\whole_thesis_evidence_synthesis_v1\THESIS_EVIDENCE_FOR_CHATGPT.md` (this file)
2. `...\THESIS_EVIDENCE_HIERARCHY.md`
3. `...\THESIS_GLOBAL_EVIDENCE_SYNTHESIS.md`
4. `C:\dense reward\reports\DWS_THESIS_RESULTS_INPUT.md`
5. `C:\dense reward\reports\DWS_FINAL_REEVALUATION_REPORT.md`
6. Optional: `whole_thesis_seed_level_evidence.csv`, `cross_experiment_contrasts.csv`, `condition_absolute_means.csv`
7. Thesis Chapter 5 source you are editing (do not let the model invent numbers not in these files)
