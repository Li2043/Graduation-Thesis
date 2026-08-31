# DWS Thesis Results Input

## A. One-paragraph result summary

Across the complete formal 12-seed sample, supplementing terminal Maximin
welfare feedback with step-wise Dense Welfare Shaping (DWS) produced a
statistically robust, direction-consistent **harmful** effect on both
primary fairness outcomes — worst-off utility (U_min) and Utility Gini —
when the policy could not observe the welfare state (Original 18D
observation): Holm-adjusted p = 0.0136 (U_min) and p = 0.0096 (Gini), with
effect direction unchanged across every leave-one-seed-out re-estimate.
This harm co-occurred with lower task completion and higher collision rate,
not an isolated fairness-metric artefact. Under WSC (22D, welfare-state
observable), the same DWS intervention produced no significant effect on
either outcome. The DWS×WSC interaction point estimates for both outcomes
point toward WSC buffering the Original-regime harm and are robust in
direction to leave-one-seed-out, but the interaction confidence intervals
cross zero at n=12 and are not independently significant. None of the four
pre-defined confirmatory behavioural mechanisms (welfare-responsive
yielding, merge-priority allocation, cooperative burden transfer, 25-step
worst-off recovery) reached Holm significance in either regime, so no
single mechanism explains the Original-regime harm at this sample size,
though the offline-reconstructed running-welfare signal moved in the same
unfavourable direction as the final outcome under Original and in the
opposite (favourable) direction under WSC — where it still did not rescue
the null final result.

## B. Primary fairness table

| Contrast | Outcome | Effect | 95% CI | Raw p | Holm p | Seeds favourable / 12 | Interpretation |
|---|---|---:|---|---:|---:|---:|---|
| Original: Cell2−Cell1 | U_min | -0.140 | [-0.258, -0.033] | 0.0068 | **0.0136** | 3 / 12 | Significant, unfavourable |
| WSC: Cell4−Cell3 | U_min | -0.029 | [-0.181, +0.128] | 0.7042 | 0.7042 | 6 / 12 | Null |
| Original: Cell2−Cell1 | Utility Gini | +0.073 | [+0.020, +0.133] | 0.0048 | **0.0096** | 3 / 12 | Significant, unfavourable |
| WSC: Cell4−Cell3 | Utility Gini | +0.015 | [-0.065, +0.092] | 0.7020 | 0.7020 | 6 / 12 | Null |

(For U_min, "favourable" = positive effect; for Gini, "favourable" =
negative effect. Holm correction applied within two separate 2-test
families — {U_min: Original, WSC} and {Gini: Original, WSC} — never
combined.)

## C. Four-cell absolute outcomes

12-seed means, H1 held-out bank, frozen final-four checkpoint ensemble.

| Cell | U_min | Utility Gini | Mean utility | Completion | Collision | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| Cell 1 — Maximin | 0.874 | 0.058 | 0.938 | 0.872 | 0.108 | 0.020 |
| Cell 2 — Maximin + DWS | 0.734 | 0.132 | 0.867 | 0.718 | 0.280 | 0.001 |
| Cell 3 — Maximin + WSC | 0.828 | 0.085 | 0.914 | 0.833 | 0.167 | 0.000 |
| Cell 4 — Maximin + WSC + DWS | 0.799 | 0.100 | 0.899 | 0.803 | 0.197 | 0.000 |

## D. DWS × WSC interaction

| Outcome | Mean interaction | 95% CI | Leave-one-out range | Direction stable? |
|---|---:|---|---|---|
| U_min | +0.110 | [-0.068, +0.306] | [+0.055 (omit 910102), +0.148 (omit 910101)] | Yes |
| Utility Gini | -0.058 | [-0.158, +0.033] | [-0.078 (omit 910101), -0.030 (omit 910102)] | Yes |

Positive U_min interaction / negative Gini interaction both mean "DWS is
more favourable (or less harmful) under WSC than under Original" — both
point estimates agree with this reading and are robust to omitting any
single seed, but neither CI excludes zero, so this is **suggestive, not
confirmed**.

## E. Mechanism table

| Mechanism | Original DWS effect | WSC DWS effect | CI / support | Interpretation |
|---|---:|---:|---|---|
| Welfare-responsive yielding (RY) | +0.116 (Holm p=0.893) | -0.401 (Holm p=0.688) | Wide CIs, sparse event counts | Not significant either regime |
| Merge-priority allocation | not estimable | not estimable | 2/12 seeds finite each regime | Too sparse to test (reported honestly, not forced) |
| Cooperative burden transfer | -0.074 (Holm p=0.326) | +0.116 (Holm p=0.326) | Opposite-signed point estimates, both wide | Not significant either regime |
| Worst-off recovery (k=25) | +0.011 (Holm p=0.211) | +0.019 (raw p=0.049, Holm p=0.098) | WSC closest to nominal significance, does not survive correction | Not significant either regime |

## F. Dense-signal diagnostics

- Positive/negative dense-shaping event rates were reconstructed offline
  (counterfactually, for all four cells) from the frozen formula
  (`Phi_t=min(M_1..M_4)`, discrete ±0.0005 shaping at epsilon=1e-6).
- **Net event balance** (positive-event share minus negative-event share),
  Cell2−Cell1 vs Cell4−Cell3: **Original -0.052, WSC +0.159.**
- This pattern **matches** the primary-outcome direction under Original
  (both worse) and **diverges** from it under WSC (signal better, final
  outcome null) — is consistent with "improving the running welfare process
  is not equivalent to improving the final failure-aware welfare
  distribution," since M_i(t) and terminal U_i are different constructs by
  definition.
- DWS-trained trajectories do not show a uniformly "more sustained
  improvement" pattern across regimes; the direction of the shift itself
  reverses between Original and WSC.

## G. Class/distribution findings

- Original-regime mean-utility harm is **broadly distributed**, not
  concentrated in one vehicle class: Fast -0.077, Slow -0.066, Ramp -0.070,
  Mainline -0.072 (all four groups shift unfavourably by a similar
  magnitude).
- Under WSC the same breakdown is small and mixed in sign (Fast +0.001,
  Slow -0.031, Ramp -0.008, Mainline -0.023), consistent with the null
  aggregate result.
- No evidence that DWS reallocates burden toward or away from a specific
  role/speed class in either regime at this sample size.

## H. Seed heterogeneity

- The Original-regime primary harm (U_min and Gini) is the **most robust**
  quantity in the whole analysis: direction never changes under
  leave-one-seed-out for either outcome (`dws_leave_one_seed_out.csv`).
- The **least robust** quantities are the WSC-regime primary effects and
  the Original-regime RY mechanism estimate — their direction flips
  depending on which seed is omitted, consistent with them being
  genuinely close to zero/imprecise rather than a real effect obscured by
  one seed.
- The interaction estimates (both outcomes) are directionally robust to
  leave-one-out even though not independently significant.
- The seeds most influential for the *primary fairness* leave-one-out range
  (900103/920104 region) are not the same seeds most influential for the
  *interaction* leave-one-out range (910101/910102) — a genuine
  cross-cutting heterogeneity pattern, consistent with this thesis's
  broader, previously-documented finding (in the WSC follow-up) that
  outcome-level and mechanism/interaction-level seed sensitivity are not
  the same phenomenon.

## I. Mechanism-chain conclusion

1. **Did DWS change learned behaviour?** Not confirmed by any of the four
   pre-defined confirmatory mechanisms (none Holm-significant in either
   regime); this does not positively demonstrate the absence of a
   behavioural change, only that these four specific quantities did not
   detect one reliably at n=12.
2. **Did DWS improve running welfare / worst-off recovery?** Mixed:
   worst-off recovery (k=25) point estimates are directionally favourable
   in both regimes but neither is Holm-significant; the exploratory
   dense-signal net-event-balance diagnostic moved unfavourably under
   Original and favourably under WSC.
3. **Did DWS improve final U_min/Gini?** No — under Original it
   significantly *worsened* both; under WSC it had no significant effect on
   either.
4. **Did WSC change the DWS effect?** The interaction point estimates
   consistently suggest yes (WSC buffers the harm), and this direction is
   robust to leave-one-seed-out, but the interaction is not independently
   significant at n=12.
5. **Where does the objective-to-behaviour-to-welfare chain appear to
   break?** At (or before) the behaviour-to-running-welfare link under
   Original — the confirmatory mechanisms could not localize a specific
   behavioural cause even though the running-welfare and final-outcome
   signals moved together unfavourably; under WSC, the chain breaks between
   running welfare and final outcome — the signal moved favourably but this
   did not survive into a significant final-fairness benefit.

## J. Best-supported theoretical contribution

Ranked 1 (strongest) to 3 (most cautious alternative). Each statement below
is fully licensed by the numbers in Sections B–I; none extrapolates beyond
them.

**1. (Strongest supported.)** Under Original (non-WSC-observable)
conditions, supplementing terminal Maximin welfare feedback with a
step-wise, shared, global dense welfare signal produces a statistically
robust reduction in realised distributive fairness (lower worst-off
utility, higher Utility Gini), accompanied by a broader decline in task
completion and safety — identifying a design boundary in which denser
temporal availability of a *shared* welfare signal, on its own, is
harmful rather than merely insufficient when the policy cannot observe the
welfare state the signal refers to. This is a negative but genuine and
robust (seed-consistent) empirical result, not an absence of a finding.

**2. (Second-best supported.)** The data are consistent with welfare-state
observability and reward-timing being non-independent design dimensions:
the harmful effect identified in Statement 1 is not observed when the
policy can also observe the welfare state (WSC), and the DWS×WSC
interaction point estimate is directionally consistent with WSC buffering
this harm across both fairness outcomes and robust to leave-one-seed-out —
this suggests, but at n=12 does not statistically confirm, that
observability can offset a harmful reward-timing effect.

**3. (More cautious alternative, if the thesis prefers to make no interaction
claim at all.)** Denser reward timing changes both task and fairness
outcomes under Original observation, but this study's four pre-registered
behavioural mechanisms (welfare-responsive yielding, merge-priority
allocation, cooperative burden transfer, 25-step worst-off recovery) do not
individually reach statistical significance at n=12 to explain how the
change arises — the mechanism-level evidence identifies a limitation of
this analysis's power to localize the pathway, rather than evidence against
a pathway existing. No claim is made here about *why* WSC removes the
effect, or about agent-specific credit assignment (this reward is a single
shared value added identically to every controlled vehicle's reward, never
an agent-specific signal), and no causal-mediation claim is made from the
observed within-evaluation behavioural/signal correlations to the final
outcome.

## K. Thesis-ready Results subsection

### Dense Welfare Shaping Follow-Up

#### Fairness Outcomes

The Dense Welfare Shaping (DWS) follow-up tested whether supplementing the
terminal Maximin welfare reward with step-wise welfare feedback changes
realised fairness, evaluated across the same 12 formal seeds and the same
held-out H1 scenario bank used throughout this thesis. Under the Original
(non-WSC) observation design, DWS produced a Holm-adjusted, statistically
significant reduction in worst-off utility (mean effect -0.140, 95% CI
[-0.258, -0.033], Holm p = 0.0136) and a corresponding significant increase
in Utility Gini (mean effect +0.073, 95% CI [+0.020, +0.133], Holm p =
0.0096); both effects were directionally consistent across every
leave-one-seed-out re-estimate. Under the WSC observation design, the same
DWS intervention produced no significant change in either outcome (Holm p
= 0.704 and 0.702 respectively), and the effect direction was not stable
under leave-one-seed-out, consistent with a genuinely near-null effect. The
DWS × WSC interaction — whether observability changes the DWS effect —
pointed consistently toward WSC attenuating the Original-regime harm for
both outcomes (U_min interaction +0.110, Gini interaction -0.058) and was
directionally robust to leave-one-seed-out, but its 95% confidence
intervals crossed zero and it is reported as suggestive rather than
confirmed.

#### Task Performance and Safety

The Original-regime fairness harm co-occurred with a broader, descriptive
decline in task performance and safety: completion fell by 0.153 (95% CI
[-0.279, -0.033]) and collision rose by 0.172 (95% CI [+0.058, +0.290]),
indicating the fairness result is not an isolated metric artefact produced
by an otherwise-unchanged or improved task policy. Under WSC, neither
completion nor collision shifted outside a CI that excluded zero.

#### Behavioural Mechanisms

Four pre-registered behavioural mechanisms — welfare-responsive yielding,
merge-priority allocation, cooperative burden transfer, and 25-step
worst-off recovery — were tested with the same paired-bootstrap and
within-mechanism Holm correction as the primary outcomes. None reached
Holm significance in either observation regime (yielding Holm p = 0.893 /
0.688; burden transfer Holm p = 0.326 / 0.326; recovery Holm p = 0.211 /
0.098; merge-priority allocation was too sparse to test, with only 2 of 12
seeds contributing a resolvable pair in either regime). These null
mechanism-level results do not explain the significant Original-regime
outcome-level harm; they indicate that this study's four pre-specified,
event-level behavioural quantities lack the power, at 12 seeds, to localize
its behavioural origin.

#### Reward-Timing Mechanism Finding

An exploratory, offline reconstruction of the running Maximin welfare
signal (Phi_t) from every evaluated trajectory — necessarily counterfactual
for the terminal-only cells, since the evaluation policy never receives
this signal as a reward — showed a net shaping-event balance that moved
unfavourably under Original (-0.052) and favourably under WSC (+0.159)
relative to each regime's terminal-only baseline. The Original-regime shift
matches the direction of the significant final-outcome harm; the
WSC-regime shift does not translate into a significant final-outcome
benefit, illustrating that the running welfare process and the final
failure-aware welfare distribution are related but distinct constructs.
Taken together, the DWS follow-up identifies a genuine, seed-consistent
design boundary — a shared, temporally denser welfare signal can be
harmful, not merely insufficient, when the policy cannot observe the
welfare state to which it refers — while leaving open, as a directionally
suggestive but statistically unconfirmed possibility, that welfare-state
observability buffers this harm.

## L. Required Chapter 5 edits outside the new subsection

- **5.1 Results Overview**: add one sentence noting that a follow-up Dense
  Welfare Shaping (DWS) study was conducted on the Maximin objective (the
  four-cell Original/WSC × terminal/DWS design), with its own subsection
  later in the chapter; do not alter any existing RQ1/RQ2/WSC sentence.
- **Research-question/hypothesis summary table**: add one row for the DWS
  follow-up's own question ("does temporally denser welfare feedback change
  realised fairness, and does welfare-state observability moderate this?")
  with outcome "Harmful under Original (Holm-significant); null under WSC;
  interaction suggestive, not confirmed" — do not modify the existing
  RQ1/RQ2/WSC rows.
- **Chapter-level synthesis** (wherever the thesis currently synthesizes the
  WSC follow-up's "no reliable aggregate benefit" finding): add that the
  DWS follow-up extends this by showing that *some* interventions in this
  design space are not merely ineffective but actively harmful under
  certain information conditions, and that the two follow-up studies
  together (WSC, DWS) point at information and reward-timing as
  interacting, not independent, design axes for this decentralised
  fairness problem.
- **5.9 Chapter Summary**: add one paragraph (drawing directly from the "One
  paragraph result summary" in Section A above) alongside — not replacing —
  the existing WSC follow-up summary paragraph.
- Do not rewrite any existing RQ1, RQ2, or WSC numerical result, table, or
  figure caption.
