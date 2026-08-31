# Whole-Thesis Experimental Evidence Synthesis

**Analysis directory:** `C:\dense reward\outputs\whole_thesis_evidence_synthesis_v1\`
**Date:** 2026-08-30
**Rule:** no retraining, no checkpoint edits, no overwrite of prior official analyses. New files only in this directory.

---

## 1. Executive judgement

**Verdict: B — yes, but only if reframed.**

The completed experiments support a coherent thesis if the story is *not* “we found a fairness reward that works.” The defensible story is:

> Decentralised merge policies can complete the task while still leaving a worse-off vehicle, but seed-level U_min/Gini largely track collisions. Changing the terminal welfare objective (Mean / GGI / Maximin) does not monotonically improve those outcomes. Making running welfare locally observable (WSC) does not Holm-significantly raise the fairness effect of those objectives. Making the same Maximin signal temporally dense and globally shared **does** change outcomes — it **harms** fairness and task competence when the policy cannot see the welfare state. The same dense signal is compatible with a null under WSC; the interaction is suggestive, not confirmed.

That is an alignment result (objective × information × timing), not a bake-off among rewards.

Examiner-style label for empirical contribution: **one robust negative intervention result (Original+DWS) plus two well-controlled near-nulls (RQ2, WSC)**. That is enough for a thesis if the writing stays inside those bounds.

---

## 2. Strongest individual findings

Ranked by a combination of pre-specification, effect size, CI, seed consistency, and technical validity.

1. **Original Maximin+DWS harms U_min** (−0.140, 95% CI [−0.258, −0.033], Holm p = 0.0136, 3/12 favourable, LOO direction never flips). Source: `dws_final_reevaluation_v1/dws_primary_fairness_summary.csv`. **STRONG.**
2. **Original Maximin+DWS raises Utility Gini** (+0.073, [+0.020, +0.133], Holm p = 0.0096). Same source. **STRONG.**
3. **That harm is joint with task failure**, not a hidden fairness–efficiency trade-off: Δcompletion −0.153 [−0.279, −0.033]; Δcollision +0.172 [+0.058, +0.290]; Δmean U −0.071. **STRONG.**
4. **RQ2: no welfare objective beats Baseline on U_min or Gini** (all CIs include 0; Holm p = 1.00 on a synthesis check). Thesis Table 5.4 / `cross_experiment_contrasts.csv`. **STRONG (null).**
5. **No Mean → GGI → Maximin fairness ordering.** Across-seed U_min: 0.902 / 0.895 / 0.874. 0/12 seeds fully monotonic; adjacent agreement 44%. `objective_strength_ordering.csv`. **STRONG.**
6. **WSC reward×information interactions are Holm-nonsignificant** for all three welfare conditions on both primary outcomes. `wsc_v2_formal_fairness_summary.csv`. **STRONG (null).**
7. **WSC+DWS vs Maximin+WSC is null** (U_min −0.029, Holm p = 0.70; Gini +0.015, Holm p = 0.70). **STRONG (null), weaker robustness** (LOO can flip sign).
8. **Seed-level U_min is almost a completion score** (Spearman 0.988 across 120 seed-condition cells). Gini–collision 0.992. `task_vs_fairness_associations.csv`. **STRONG descriptive; limits how much “unfairness” can be separated from crashes at this aggregation.**
9. **Successful-episode mobility burden is not the large unconditional Maximin/GGI numbers.** Baseline success burden 0.024 vs collision 0.333; Maximin success 0.094 vs timeout 24.9 (`table5_9_outcome_decomposition_seedlevel.md`). **STRONG as a warning against unconditional burden.**
10. **Shared F_t cannot attribute a Phi drop to one agent** (design + high single-decliner rates). `dws_shared_credit_summary.md`. **MODERATE / by construction.**

---

## 3. Weak findings that should not be emphasised

- Any “GGI + WSC is the winner” reading of absolute means (GGI+WSC U_min 0.941). That cell is not a pre-specified primary contrast against Original GGI in the WSC Holm family (the family is the *interaction*, which is n.s.).
- n=11 Mean-WSC intervals that exclude 0 (Holm p 0.12 / 0.14). Sensitivity only.
- Merge-priority allocation (too sparse).
- Confirmatory DWS behavioural mechanisms (RY, burden transfer, recovery k=25): none Holm-significant.
- P(F<0|BRAKE) falling under Original DWS — exploratory, easy to over-interpret.
- 18D/22D observation aliasing — **not estimable**.
- Monotonic “Fast vehicles are the worst-off class” as a Baseline law. Mixed-episode Fast/Slow utilities are 0.938 / 0.944.

---

## 4. Cross-experiment pattern

### 4.1 Objective strength vs realised fairness

Hypothesised fairness-improving order Baseline < Mean < GGI < Maximin is **not** the across-seed mean order. Original means:

| | Baseline | Mean | GGI | Maximin |
|---|---:|---:|---:|---:|
| U_min | 0.882 | 0.902 | 0.895 | 0.874 |
| Gini | 0.058 | 0.049 | 0.050 | 0.058 |
| Completion | 0.884 | 0.903 | 0.895 | 0.872 |
| Collision | 0.116 | 0.097 | 0.098 | 0.108 |

WSC absolute means are dominated by GGI+WSC looking strong and Mean+WSC / Maximin+WSC looking weak — still not a monotonic aversion gradient, and not a substitute for the interaction test.

### 4.2 Information × objective

Use the official WSC interactions, not raw WSC−Original differences. No Holm-supported interaction. GGI’s favourable points are 910102-levered.

### 4.3 Information × reward timing

Four-cell Maximin design. Original DWS harm confirmed. WSC DWS null. Interaction point estimates say “WSC buffers harm” but both CIs include 0. **Moderation is not confirmed.**

### 4.4 Task vs fairness

Association, not causation: at the seed-condition cell, U_min and Gini move with completion/collision. High-completion cells (70) include only one with Gini ≥ 0.05. Original+DWS is **joint degradation**, not a fairness gain paid for with safety.

High-completion but unfair policies exist in principle (that one cell) and should not be treated as the typical Baseline story.

---

## 5. Mechanism evidence

### 5.1 Reward-observation aliasing

**NOT ESTIMABLE in policy observation space.** Trajectories store x, M, action, Phi, F_t — not 18D/22D vectors.

Predeclared proxy (k = 10, 25, 50; cap 4000 points/action/seed; RNG 0): standardised [x] vs [x,M], same-action k-NN. At k=25, adding M cuts sign-disagreement by roughly half to two-thirds in every cell (e.g. Cell 1: 0.190 → 0.070). This is consistent with “hidden welfare history aliases a shared F_t in kinematic neighbourhoods.” It is **not** a test of what the network sees.

### 5.2 Temporary concession and DWS signal

Exploratory. Original DWS **lowers** P(F<0|BRAKE) (−0.182, CI excludes 0), including on successful episodes (−0.083). This does **not** confirm an anti-yield training penalty. Evaluation policies that collide earlier can simply generate a different (state, brake, F) joint.

### 5.3 Shared-credit ambiguity

By design, F_t is identical for all four vehicles. On negative-DeltaPhi steps, a single decliner is common; the largest M drop often is not the strongest brake. DWS contrasts on these shares are null. Supported wording: the signal does not identify the agent. Not supported: this is proven to be *why* Original+DWS fails.

### 5.4 Running vs terminal welfare

Phi at 25/50/75/90% of the episode does not significantly fall under Original DWS, while terminal U_min does. Under WSC, reconstructed net event balance can improve without a final U_min gain (already in the DWS re-evaluation). Running attainment and failure-aware U_i are different random variables.

### 5.5 Behavioural policy changes

Official four mechanisms: no Holm-significant DWS effect in either regime; merge-priority too sparse. Action rates: Original DWS shortens episodes (−6.4 steps); WSC DWS reduces merge-window HOLD rate (−0.172, CI excludes 0). Action frequency is not fairness.

WSC official RY interactions are directionally mixed and Holm-nonsignificant.

### 5.6 Seed heterogeneity

Treat as substantive. Seed 910102 collapses Original Mean (completion 0.035, Mean−Baseline ΔU_min ≈ −0.678) and levers WSC GGI. Original DWS harm is **not** a 910102 artefact (LOO-stable). WSC DWS and several mechanism estimates **are** seed-fragile. Fairness interventions here often look like **policy-selection instability** across lineages more than a uniform shift. Phrase as “consistent with,” not as a proven alternative DGP.

---

## 6. Where the fairness pipeline breaks

| Arrow | Classification | Why |
|---|---|---|
| objective → information available | **supported as a design fact** | WSC adds local M; Original does not. Not itself an outcome. |
| information available → reward feedback | **not tested / broken for Original+DWS** | Reward is computed from hidden M/Phi either way. WSC does not change F_t; it only changes the observation. Original+DWS therefore pairs a dense reward with an unobserved welfare state. |
| reward feedback → action adaptation | **weak** | No Holm-significant confirmatory mechanism. Some exploratory action-rate movement. |
| action adaptation → running welfare | **weak / mixed** | WSC+DWS: net event balance up; Original+DWS: net balance down. Recovery k=25 Holm-nonsignificant. |
| running welfare → terminal fairness | **broken under WSC+DWS; jointly worse under Original+DWS** | WSC: better reconstructed signal, null U_min. Original: worse signal *and* worse U_min/completion. Terminal U zeros on collision; M does not. |

---

## 7. Best-supported theoretical contribution

**Strongest formulation.**
In this decentralised four-agent merge, a shared welfare reward is not a sufficient statistic for fair local control. Making that reward more inequality-averse does not monotonically improve held-out worst-off utility. Making the welfare state locally visible does not Holm-significantly amplify the objective. Making the same Maximin signal temporally dense **worsens** fairness and safety when the welfare state remains hidden.

**Cautious formulation.**
Objective specification, welfare observability, and reward timing are separable design choices. In this sample they do not automatically substitute for one another. The only Holm-significant intervention effect in the follow-up programme is a **negative** Original+DWS effect.

**One-sentence abstract version.**
Changing how welfare is valued, or whether it is locally observed, did not reliably improve worst-off merge outcomes; adding a dense shared Maximin signal reduced worst-off utility and completion when agents could not observe that welfare state.

---

## 8. What the thesis should NOT claim

- MAPPO is inferior to DQN, or DQN is a new fairness algorithm.
- The old two-agent PBRS experiment proves the four-agent hypothesis.
- Mean → GGI → Maximin is an interval scale of “more fairness.”
- WSC is equivalent to no effect, or WSC is confirmed to harm Mean.
- DWS×WSC complementarity is confirmed.
- DWS is potential-based / policy-invariant.
- Shared DWS is agent-specific credit assignment, or credit failure is causally proven.
- Episode n = 256 × 12 is the inferential n.
- Best-checkpoint or dropped-seed results.
- Unconditional mobility burden as Maximin’s “fairness cost” without the success/collision/timeout split.
- 18D/22D aliasing as a completed test.

---

## 9. Examiner-style critique

**Verdict: B.**

### Strongest defence points

- Matched seeds, frozen H1 bank (256), greedy final-four ensemble, SHA-tracked scientific code on WSC v2.
- Primary outcomes were not swapped after seeing DWS.
- The strongest result is negative and LOO-stable — hard to dismiss as p-hacking toward a desired fairness win.
- Outcome decomposition already exists and blocks a naive burden story.

### Likely examiner objections, and the answer

| Objection | Answer | Residual gap |
|---|---|---|
| “U_min is just completion.” | Spearman 0.99 at cell level. Report it. Use success-only burden and identity tables for RQ1. | RQ1 “inequality despite success” is weaker than the mixed-episode tables suggest. |
| “n=12 is underpowered.” | True for WSC and for DWS×WSC. Not an excuse to call those confirmed. Original DWS still excludes 0. | Do not add seeds post hoc. |
| “910102 drives everything.” | It drives Mean and WSC-GGI. It does **not** drive Original DWS harm. | Keep the seed; show LOO. |
| “Why not agent-specific DWS?” | Out of scope; would be a new experiment. Shared-global was frozen. | Cannot claim you tested credit assignment. |
| “Protocol was amended (local sensing, DWS).” | Amendments are documented; DWS protocol verification already exists. | Cite `DENSE_PROTOCOL_VERIFICATION_FOR_THESIS.md`. |
| “No mechanism for the DWS harm.” | Honest: confirmatory mechanisms are null; joint collision increase is the clearest companion. | Do not invent a yielding story. |

### Genuine limitations

- Observation vectors were not stored, so the highest-priority aliasing test cannot be finished without a new dump-only rollout.
- Inferential power for interactions is low.
- Failure-aware U_i couples fairness and safety by construction.
- One road geometry, one sensing range, one discrete action set.

---

## 10. Recommended Chapter 6 Discussion structure

1. Restate the three design axes (objective / information / timing).
2. RQ1: what is unequal, and what is just a crash.
3. RQ2: why a more Rawlsian terminal objective did not buy U_min.
4. WSC: observability without a Holm-supported outcome gain; cautious behavioural note.
5. DWS: the robust negative Original result; null under WSC; unconfirmed interaction.
6. Pipeline diagram (Section 6 above).
7. What a next experiment would manipulate (agent-specific vs shared F; dump observations; success-conditional primary).
8. Claims the chapter refuses.

---

## 11. Recommended Abstract / Conclusion claims

Use only Tier 1 wording from `THESIS_EVIDENCE_HIERARCHY.md`.

Minimum abstract core:

> Twelve matched training seeds, a shared local DQN, and a held-out heterogeneous merge bank were used to separate welfare objectives, welfare-state observability, and reward timing. Mean, GGI, and Maximin did not significantly raise worst-off utility or reduce Utility Gini relative to an equal-budget Baseline, and stronger inequality aversion did not produce better realised fairness. Local welfare-state features (WSC) did not Holm-significantly increase those objective effects. Supplementing terminal Maximin with a shared step-wise Maximin signal reduced worst-off utility by 0.14 and raised collision rate when welfare remained hidden; the same dense signal was not significant under WSC. The results support treating objective, observability, and timing as aligned design choices rather than interchangeable fairness upgrades.

---

## Candidate synthesis (prompt §5) — clause-by-clause

| Claim | Support | Contrary / limit | Strength |
|---|---|---|---|
| Successful decentralised coordination can contain structured lower-tail inequality | Baseline U_min 0.882 < mean U 0.941; some seeds collision-heavy | Cell-level Gini is collision-tied; success-only burden is small | **Rewrite:** “can leave a worse-off vehicle; much of the seed-level gap is failure-aware.” |
| Changing the terminal social-welfare objective does not reliably remove it | RQ2 all CIs include 0; no monotonic order | Mean point estimate is favourable | **Keep**, with the point-estimate caveat |
| Welfare observability changes some local responses but not final fairness | RY point estimates; Holm-null outcomes | RY itself is Holm-nonsignificant | **Soften** “changes some local responses” to “showed only uncorrected, condition-dependent yielding point estimates” |
| Denser shared welfare reward can worsen fairness and task when welfare is hidden | Original DWS Holm-sig + collision | Not true (or not shown) under WSC | **Keep**, explicitly Original-only |
| Objective, observability, and timing must be aligned | Joint pattern of the three studies | DWS×WSC CI includes 0 | **Soften** to “are not interchangeable; alignment is a hypothesis consistent with the pattern, not a confirmed interaction” |

Rewritten candidate (use this, not the original paragraph):

> Failure-aware worst-off utility under a competent-looking mean can still sit well below mean utility, largely because collisions zero individual utilities. Replacing the terminal objective with Mean, GGI, or Maximin does not reliably raise that floor. Adding local running-welfare features does not Holm-significantly enlarge the objective’s effect. Adding a dense shared Maximin signal **lowers** the floor and raises collisions when that welfare state is hidden. The studies together show that a stronger or denser fairness signal is not automatically beneficial; they do not prove a confirmed three-way statistical interaction.
