# Dense Welfare Shaping — Final Re-evaluation

## 1. Evaluation integrity

All four Maximin cells passed the training audit (`DWS_FINAL_TRAINING_AUDIT.md`,
48/48 cell x seed combinations PASS: final checkpoint at 2,000,000 exists,
the frozen final-four ensemble window {1,850,000; 1,900,000; 1,950,000;
2,000,000} exists, no technical-failure signature found). The held-out
evaluation was re-run from scratch (not reused from any prior interim CSV):
same H1 bank (256 scenarios, byte-identical SHA256 across the two drives
the four cells' checkpoints live on), same scenario IDs/order across all
four cells, deterministic greedy action selection, equal-weight Q-ensemble
across the frozen four-checkpoint window. Total: 12 seeds x 256 scenarios x
4 cells = 12,288 episode rows (`dws_final_episode_level.csv`), verified with
zero duplicates, all distinct scenario IDs per cell x seed, zero NaN in any
primary field. A same-checkpoint cross-check against the prior (pre-existing)
Cell 4 evaluation CSV from earlier in this session produced an exact match
(max abs diff = 0.0000 across all 12 seeds x {U_min, completion, collision}),
confirming the new pipeline reproduces the established evaluation exactly.
Wall-clock: ~15m51s for all 48 rollout jobs (10 parallel workers).

## 2. Primary fairness result

Inferential unit: training seed (n=12, matched). Per-episode outcomes were
first averaged within seed before any inference (256 episodes are NOT
treated as independent replicates). 10,000 paired-bootstrap resamples,
RNG seed 0, 95% percentile CI, two-sided bootstrap p-values, Holm correction
applied separately within two 2-test families (U_min family; Gini family —
never combined into one 4-test family, per protocol).

### 2.1 Original: DWS vs terminal-only Maximin

- **U_min**: mean effect **-0.140**, 95% CI **[-0.258, -0.033]**, raw p =
  0.0068, **Holm-adjusted p = 0.0136**. 3/12 seeds favourable, 9/12
  unfavourable.
- **Gini**: mean effect **+0.073** (positive = unfavourable for Gini), 95% CI
  **[+0.020, +0.133]**, raw p = 0.0048, **Holm-adjusted p = 0.0096**. 9/12
  seeds unfavourable, 3/12 favourable.
- Both effects are **Holm-significant and unfavourable**, and both are
  **robust to leave-one-seed-out**: the direction never flips across any of
  the 12 leave-one-out estimates for either outcome (`dws_leave_one_seed_out.csv`).

### 2.2 WSC: DWS vs terminal-only Maximin

- **U_min**: mean effect -0.029, 95% CI [-0.181, +0.128], Holm-adjusted p =
  0.704. Null.
- **Gini**: mean effect +0.015, 95% CI [-0.065, +0.092], Holm-adjusted p =
  0.702. Null.
- Both are **not** robust to leave-one-seed-out — the effect direction flips
  depending on which seed is omitted, consistent with a genuinely
  near-zero, noisy effect rather than a real null hiding a fragile signal.

### 2.3 DWS × WSC interaction

- **U_min**: mean +0.110, 95% CI [-0.068, +0.306]. Point estimate direction
  (favourable-under-WSC) is robust to leave-one-seed-out.
- **Gini**: mean -0.058, 95% CI [-0.158, +0.033]. Same directional
  robustness (favourable-under-WSC direction for Gini too).
- Both CIs cross zero — the interaction is **suggestive but not
  independently significant** at n=12. Per protocol, this is a secondary
  mechanism estimate, not in either primary Holm family.

## 3. Task performance and safety

Descriptive only (no confirmatory p-values or multiplicity correction
assigned, per protocol; the pre-existing WSC study's ±0.05/+0.03
non-inferiority margins are **not** reused here since no frozen DWS decision
adopts them).

| Contrast | Completion | Collision | Timeout | Mean utility | Episode length |
|---|---:|---:|---:|---:|---:|
| Original: Cell2−Cell1 | **-0.153** [-0.279,-0.033] | **+0.172** [+0.058,+0.290] | -0.019 [-0.059,+0.003] | **-0.071** [-0.131,-0.017] | **-6.39** [-14.22,-0.29] |
| WSC: Cell4−Cell3 | -0.030 [-0.182,+0.131] | +0.030 [-0.131,+0.182] | 0.000 [0,0] | -0.015 [-0.092,+0.064] | -0.21 [-4.20,+3.71] |

Under Original, completion, collision, mean utility, and episode length all
shift unfavourably alongside the fairness harm (CIs excluding zero for
completion, collision, and mean utility) — the fairness effect above is
**not** an isolated metric artefact produced by a hidden task-competence
gain; if anything, task competence and safety degrade together with fairness
under Original+DWS. Under WSC, none of these descriptive contrasts exclude
zero.

Absolute per-cell levels (12-seed means): Cell 1 completion 0.872/collision
0.108; Cell 2 completion 0.718/collision 0.280; Cell 3 completion
0.833/collision 0.167; Cell 4 completion 0.803/collision 0.197.

## 4. Behavioural mechanism analysis

Pre-defined confirmatory family (Section 20): welfare-responsive yielding
(RY), merge-priority allocation, cooperative burden transfer, worst-off
recovery (k=25 primary). Holm correction applied separately within each
mechanism's own 2-test family (Original vs WSC), never combined across
mechanisms or with the primary U_min/Gini families.

### 4.1 Welfare-responsive yielding

RY = P(BRAKE | visible neighbour worse off) / P(BRAKE | neighbour not
worse off). Original: mean effect +0.116, Holm p = 0.893. WSC: mean effect
-0.401, Holm p = 0.688. **Neither Holm-significant.** CIs are extremely
wide (RY is a ratio of sparse per-seed event counts) — this is a genuinely
imprecise estimate at n=12, not strong evidence of "no yielding effect."

### 4.2 Merge-priority allocation

**Too sparse to test in either regime** — only 2/12 seeds had a resolvable
(non-tied, both-exit-observed) priority pair in each of the Original and
WSC contrasts. Per Section 11's own rule, no bootstrap was forced; this is
reported as sparsity, not as a null result.

### 4.3 Cooperative burden transfer

Share of hard-brake events occurring under a "neighbour worse-off"
opportunity. Original: mean effect -0.074, Holm p = 0.326. WSC: mean effect
+0.116, Holm p = 0.326. **Neither Holm-significant**; point estimates point
in opposite directions between regimes, but both CIs are wide and cross
zero.

### 4.4 Worst-off recovery (k=25)

Mean gap-closure at 25 steps. Original: mean effect +0.011, Holm p = 0.211.
WSC: mean effect +0.019, raw p = 0.049, **Holm p = 0.098 — does not survive
correction.** Both point estimates are directionally favourable (closer
recovery) but neither is Holm-significant; the WSC estimate is the closest
of the four confirmatory mechanisms to nominal significance but is
correctly not reported as significant after the pre-specified correction.

**None of the four pre-defined confirmatory mechanisms reached
Holm-significance in either information regime.** This does not mean DWS
had "no behavioural effect" — it means none of these four specific,
pre-registered event-level quantities detected a reliable effect at n=12
seeds with the event counts actually observed.

## 5. Dense welfare-signal dynamics

Exploratory/descriptive (Section 14), offline-reconstructed via the exact
frozen formula (`M_i(t)=running_active_attainment`, `Phi_t=min(M_1..M_4)`,
discrete ±0.0005 shaping at epsilon=1e-6) from every evaluation trajectory,
for all four cells (for the terminal-only cells this is a **counterfactual**
reconstruction — the evaluated policy did not receive this signal as a
reward, since rewards are not inputs to the evaluation policy).

Net dense-signal event balance (fraction of steps with `DeltaPhi_t >
epsilon` minus fraction with `DeltaPhi_t < -epsilon`), Cell2−Cell1 vs
Cell4−Cell3:

- Original: mean diff **-0.052** (net balance shifts negative for the
  DWS-trained policy vs terminal-only, at evaluation time).
- WSC: mean diff **+0.159** (net balance shifts positive for the DWS-trained
  policy vs terminal-only).

This is directionally consistent with the outcome-level story: under
Original, the running-welfare trajectory is measurably worse for the
DWS-trained policy, matching its worse final U_min/Gini. Under WSC, the
running-welfare trajectory is measurably *better* for the DWS-trained
policy, yet this did **not** translate into a significant final U_min/Gini
improvement — an instance of "improving the running welfare process is not
equivalent to improving the final failure-aware welfare distribution"
(M_i(t) and terminal U_i are different constructs by definition).

## 6. Distributional changes across vehicle classes

Original-regime DWS harm to mean utility is broadly distributed, not
concentrated in one group: Fast -0.077, Slow -0.066, Ramp -0.070, Mainline
-0.072 (`dws_class_distribution_summary.csv`). No group is spared and none
is disproportionately singled out — consistent with a shared, global
shaping signal producing a broad effect rather than a class-targeted one.
Under WSC the same breakdown is small and mixed (Fast +0.001, Slow -0.031,
Ramp -0.008, Mainline -0.023), consistent with the null aggregate finding.

## 7. Seed heterogeneity and sensitivity

The Original-regime primary harm (U_min, Gini) is the most robust
quantity in this analysis: it never changes direction under
leave-one-seed-out for either outcome. The WSC-regime primary effects and
the RY mechanism (Original) are the least robust — direction flips
depending on which seed is held out, consistent with their already-null /
imprecise status. The interaction estimates (both outcomes) are directionally
robust but not independently significant. No seed was excluded from any
primary analysis at any point.

## 8. Outcome decomposition

Timeout/truncation episodes are rare and asymmetric across cells (Cell 1:
2/12 seeds ever timeout; Cell 2: 1/12; Cells 3/4: 0/12) —
`dws_outcome_decomposition.csv` reports these as descriptive/sparse rather
than a basis for any burden-magnitude claim conditioned on timeout.

## 9. Learning dynamics

**Not performed.** Section 19 explicitly marks this analysis optional
("If required checkpoints/logs are available and the computation is
reasonable") and lower priority than the final 2.0M evaluation; given the
scope already completed, this was deliberately skipped rather than run
partially, and is flagged here rather than silently omitted.

## 10. Mechanism chain

See `dws_mechanism_chain_table.csv` for the full table with evidence
citations. Summary: the reward-timing → behaviour link is not confirmed by
any single pre-defined mechanism (none Holm-significant), but the
behaviour/running-welfare → final-fairness link **is** confirmed in the
Original regime (both move together, unfavourably) and diverges in the WSC
regime (running welfare improves, final fairness does not) — see Section 5
above and Section 11 below.

## 11. Main theoretical interpretation

The evidence is most consistent with two linked findings, not one:

**(a) Under Original observation, denser reward timing produces a real,
Holm-significant, leave-one-out-robust *harmful* effect on both primary
fairness outcomes** (U_min and Gini), accompanied by a broader task/safety
degradation (lower completion, higher collision, lower mean utility) and a
matching negative shift in the reconstructed running-welfare signal. This
is not "no effect" and not "insufficient effect" — it is a directionally
consistent, mechanistically-plausible harm at the outcome level, even
though none of the four pre-defined confirmatory behavioural mechanisms
individually reached significance at n=12 to explain *how* it arises.

**(b) Under WSC observation, this harm is absent** (null, fragile-direction
primary effects), and the DWS×WSC interaction point estimates for both
outcomes are directionally consistent with "WSC buffers/mitigates the
Original-regime harm" and robust to leave-one-seed-out — but the interaction
CIs cross zero, so this is suggestive, not independently confirmed.

This pattern **is not** "no effect anywhere" (Pattern F/no-effect reading is
ruled out by the Original-regime Holm-significant result) and it **is not**
a favourable-DWS finding (Pattern A) — there is no cell/outcome combination
where DWS shows a Holm-significant *favourable* effect. The data are most
consistent with a harmful-reward-timing-effect-under-Original interpretation
that is buffered (not eliminated with confidence) by welfare-state
observability, i.e. a hybrid of "reward timing changes outcomes in a
specific, non-null direction under one information regime" and "information
and timing are not independent design dimensions" — see the ranked
candidates in `DWS_THESIS_RESULTS_INPUT.md` Section J for the precise,
hedged phrasing this licenses.

## 12. Claims that ARE supported

- DWS under Original observation produces a Holm-significant, direction-robust
  *unfavourable* effect on both U_min and Utility Gini.
- This Original-regime harm co-occurs with a broader task/safety degradation
  (completion, collision, mean utility), not an isolated fairness-metric
  artefact.
- DWS under WSC observation produces no significant effect on either primary
  outcome.
- The DWS×WSC interaction point estimates are directionally consistent
  (across both outcomes and robust to leave-one-seed-out) with WSC
  attenuating the Original-regime harm, though not independently significant.
- The reconstructed running-welfare signal moves in the same direction as
  the final outcome under Original (worse/worse) and in the opposite
  direction under WSC (better running signal, null final outcome).
- The Original-regime harm is broadly distributed across vehicle role/speed
  classes, not concentrated in one group.

## 13. Claims that are NOT supported

- That DWS improves fairness under any regime (no cell shows a favourable
  Holm-significant primary effect).
- That any single pre-defined behavioural mechanism (yielding, priority,
  burden transfer, or 25-step recovery) explains the Original-regime harm —
  none reached Holm significance.
- That the DWS×WSC interaction is a confirmed effect — CIs cross zero for
  both outcomes.
- Any claim of agent-specific credit assignment: DWS is a single value
  added identically to every controlled vehicle's reward each step
  (`shaping_scope="shared_global"`), never an agent-specific signal.
- Any causal-mediation claim linking the observed behavioural/signal
  correlations to the final fairness outcome — these are observational
  associations within the same evaluation, not an independent causal test.
- That DWS constitutes policy-invariant potential-based reward shaping —
  it is explicitly not that (see `dense_shaping.py`'s own design record).

## 14. Limitations of the DWS follow-up

- n=12 seeds is the same sample size used throughout this thesis's other
  formal comparisons; several mechanism-level estimates (RY, merge-priority)
  are visibly underpowered/sparse at this n, even though the primary
  fairness contrasts were well-resolved.
- Merge-priority allocation could not be tested in either regime (2/12
  seeds finite) — this gap is inherent to how rarely a fully resolvable
  (non-tied, both-vehicles-exit-observed) pair occurs in 256 held-out
  episodes at this environment's traffic density, not a methodological
  choice.
- The Dense-signal reconstruction (Section 5 here / Section 14 of the
  prompt) is inherently descriptive for the terminal-only cells (Cells 1
  and 3), since the evaluated greedy policy never received the shaping
  signal as a reward during evaluation for any cell.
- Learning-dynamics analysis (checkpoint-by-checkpoint trajectory of the
  primary effects) was not performed (Section 19, explicitly optional).

## 15. Exact source files and reproducibility commands

- Episode-level data: `outputs/dws_final_reevaluation_v1/dws_final_episode_level.csv`
  (produced by `scripts/dws_final_eval_launcher.py` orchestrating
  `scripts/dws_eval_worker.py`, 48 independent cell x seed jobs).
- Trajectory data: `outputs/dws_final_reevaluation_v1/trajectories/<cell>_<seed>.jsonl.gz`
  (48 files, ~65MB total).
- Primary/interaction/task-safety statistics: `scripts/dws_analyze_primary.py`
  (uses `scripts/dws_stats_lib.py` for the paired-bootstrap/Holm/leave-one-out
  primitives — RNG seed 0, 10,000 resamples, hardcoded).
- Behavioural mechanisms: `scripts/dws_analyze_mechanisms.py`.
- Dense-signal diagnostics and action-policy rates: `scripts/dws_analyze_signal_and_actions.py`.
- Figures: `scripts/dws_make_figures.py` -> `outputs/dws_final_reevaluation_v1/figures/`.
- Training audit: `reports/DWS_FINAL_TRAINING_AUDIT.md`.
- To reproduce end to end: run the five scripts above in that order from
  `C:\dense reward` using `.venv\Scripts\python.exe`; all are deterministic
  and read-only with respect to training/checkpoints.
