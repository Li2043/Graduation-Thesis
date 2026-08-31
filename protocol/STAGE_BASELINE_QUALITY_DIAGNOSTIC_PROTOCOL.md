# Baseline-Quality Dependence of Welfare Fine-Tuning — Diagnostic Protocol

**Status: DRAFT, NOT FROZEN.** Several parameters (new-seed count, exact
seed IDs, intermediate checkpoint-freeze steps) are still open — see
§6 "Open decisions" — and must be fixed *before* any new training run
starts, per this project's frozen-protocol discipline. Do not launch
training against this document until those are filled in and it is
re-saved with a `stageNX-protocol-v1` tag.

## 0. Why this stage exists

The formal 6-seed RQ1/RQ2 analysis (`05 results.md` §5.6–5.8) found a
strong descriptive correlation (Pearson $r=-0.93$ to $-0.98$, $n=6$
seeds) between how well a seed's Mean-condition policy performs and how
much GGI/Maximin change it, relative to Mean, on the same shared
$C64_{R50}$ starting checkpoint (§5.7.1). The single largest positive
welfare-intervention effect in the entire formal dataset — GGI/Maximin
rescuing seed 910102 from a near-total Mean-policy collapse — sits at
one extreme of this correlation.

This correlation is **observational and confounded**: each of the six
formal seeds contributes exactly one baseline-quality data point, so
seed identity and baseline quality are perfectly entangled. It is
impossible, from the existing data, to tell "welfare fine-tuning helps
more when the baseline is worse" apart from "910102 is idiosyncratic for
some unrelated reason." §5.7.4 of the results chapter explicitly flags
this as unresolved and recommends a targeted diagnostic — this document
designs that diagnostic.

**This stage is not designed to make GGI/Maximin look more effective.**
It is designed to test whether the baseline-quality correlation is a
real, general phenomenon or a single-seed artifact, and it is required to
report the result in either direction. See §5 (governance) for the
explicit anti-cherry-picking rules this implies.

## 1. Theoretical grounding

Three independent literatures give plausible mechanisms for the pattern
already observed, and — together — a specific, falsifiable prediction for
this diagnostic (§1.4). None of these are proof; they are cited here to
motivate the design and to justify what the diagnostic will be read
against, decided *before* seeing its results.

### 1.1 Max-min (Rawlsian) welfare is a harder optimization target than GGI

Siddique, Weng & Zimmer, *"Learning Fair Policies in Multi-Objective
(Deep) Reinforcement Learning with Average and Discounted Rewards,"*
ICML 2020 — introduces the Generalized Gini Index (GGI) welfare function
(the same GGI used throughout this thesis) explicitly as a **smoothed
alternative to the max-min/Rawlsian objective**, motivated by the fact
that the pure max-min objective is concave but **not differentiable**,
which makes it difficult to optimize with gradient-based methods. GGI's
sorted-weighted-sum formulation gives a smoother gradient signal across
several low-ranked agents rather than concentrating all credit on a
single worst-off agent.

Follow-up work in the fair multi-objective RL literature confirms this is
a known, general difficulty: max-min-style fairness objectives require
specialized (e.g. subgradient-based) optimization machinery precisely
because of this non-smoothness — see the max-min MORL formulation and
fairness-in-MARL surveys found alongside the Siddique et al. paper
(sources below).

**Connection to this thesis's own data:** this directly predicts that
Maximin fine-tuning should show more optimization instability than GGI
fine-tuning from the same starting point — which is exactly what
`05 results.md` §5.7.2 already found (900104's late-training collapse
under Maximin — rising timeout, falling `mean_Q(policy)` — is far more
severe than anything observed under GGI for the same or other seeds; the
GGI-side burden/task-performance anomalies mostly vanish under the
successful-episodes-only sensitivity check in §5.6.3, while Maximin's do
not fully).

### 1.2 "Alignment tax": secondary-objective fine-tuning degrades an already-capable policy

Ouyang et al., *"Training Language Models to Follow Instructions with
Human Feedback,"* NeurIPS 2022 (the InstructGPT paper) — reports that
RLHF fine-tuning of an already-capable pretrained model, using a
secondary (alignment) objective layered on top of the original one,
produced measurable **performance regressions on the model's original
capabilities** (standard QA/translation benchmarks), an effect the
subsequent literature has come to call the "alignment tax." Follow-up
work (e.g. Lin et al., *"Mitigating the Alignment Tax of RLHF,"* EMNLP
2024) treats this as a general, expected cost of secondary-objective
fine-tuning, not an implementation bug.

**Connection to this thesis's own data:** structurally, Mean → GGI/Maximin
fine-tuning in this thesis is the same shape — an already-converged,
task-competent $C64_{R50}$ checkpoint is fine-tuned further under a
second (welfare) objective layered on the original task reward. Several
seeds show exactly this pattern: 900102's and 900104's task completion
measurably *drops* under GGI/Maximin relative to their own Mean result
(Table 5.2), even though the shared starting checkpoint was identical.

### 1.3 Warm-starting from a converged checkpoint can generalize worse than training fresh

Ash & Adams, *"On Warm-Starting Neural Network Training,"* NeurIPS 2020 —
shows that continuing training ("warm-starting") a network from a
previously-converged checkpoint, even on data from the same distribution,
often generalizes *worse* than training an identically-configured network
from a fresh random initialization, despite similar final training loss —
attributed to the warm-started network being confined to a harder-to-escape
region of the loss landscape shaped by its prior training.

**Connection to this thesis's own data:** this offers a candidate
mechanism for *why* fine-tuning under a new (welfare) objective from an
already-converged checkpoint might be inherently unstable, independent of
whether the new objective is "harder" in the Siddique et al. sense — the
checkpoint's prior convergence itself may be part of the problem. This
reframes the baseline-quality correlation: instead of "worse baselines
benefit more from welfare objectives," the more precise hypothesis may be
"any further fine-tuning of an already-converged checkpoint under a new
objective is inherently somewhat destabilizing, and this shows up as a
*net improvement* only when the starting policy had little to lose (as
with 910102's already-collapsed Mean baseline) and as a *net regression*
when the starting policy had a lot to lose (as with 900104's near-perfect
Mean baseline)."

### 1.4 The falsifiable prediction this diagnostic tests

If §1.1–1.3 together explain the observed pattern, this diagnostic
should find:

1. **Within one seed (900103, §2.1), $\Delta_{\text{GGI/Maximin}-\text{Mean}}$
   becomes more positive (or less negative) as the starting checkpoint's
   pre-fine-tune quality gets worse** (C64 < C16 < C4 in 900103's own
   curriculum) — replicating the cross-seed correlation within a single
   seed, removing the seed-identity confound.
2. **Maximin's deltas should be more variable (larger magnitude swings in
   both directions) than GGI's**, holding starting quality fixed, per
   §1.1.
3. **If the new-seed curriculum runs (§2.2) turn up any seed with a
   genuinely intermediate or poor pre-fine-tune competence**, the same
   two patterns (1) and (2) should replicate in that independent seed.

If instead $\Delta$ shows no relationship to starting quality within
900103, or the new-seed replications don't show pattern (1)/(2), the
correct conclusion is that 910102's rescue is a single-seed artifact, not
a general baseline-quality mechanism — and the thesis should say so
plainly rather than retrofitting the theory to the null result.

## 2. Design

Two independent, parallel tracks. Track A is cheap (reuses existing
checkpoints); Track B is expensive (full curriculum from scratch) and
exists to test whether Track A's within-seed finding (if any) replicates
in an independent seed.

### 2.1 Track A — within-seed dose-response (seed 900103)

Seed 900103 is the only one of the six formal seeds whose existing
curriculum checkpoints span a genuine competence gradient (Table 5.1a):

| Checkpoint | Pre-fine-tune completion (Q-bank gate, already measured, frozen) |
|---|---:|
| `C4_R50` | 0.750 |
| `C16_R50` | 0.625 |
| `C64_R50` | 0.453 |

$C64_{R50} \to$ Mean/GGI/Maximin **already exist** (part of the 18 formal
runs) and are **not rerun**. New work: fine-tune $C4_{R50}$ and
$C16_{R50}$ independently under Mean, GGI, and Maximin —

$$
2 \text{ new starting checkpoints} \times 3 \text{ conditions} = 6 \text{ new 800{,}000-step runs}
$$

— using the exact frozen hyperparameters, $\lambda_W=0.5$, checkpoint
schedule, and $R{=}50\,\mathrm{m}$/$K(2{,}000{,}000)$/$\epsilon=0$
evaluation protocol already used for the 18 formal runs. No coefficient,
threshold, or reward term changes.

### 2.2 Track B — new independent seeds, full curriculum

Train $N$ new seeds through the complete $M6_{R50} \to C4_{R50} \to
C16_{R50} \to C64_{R50}$ curriculum from scratch, exactly as the original
six formal seeds were trained (same hyperparameters, same stage budgets,
same qualification-gate machinery). At each of the three curriculum
checkpoints (C4/C16/C64), fine-tune under Mean/GGI/Maximin for 800,000
steps, using the same protocol as Track A.

$$
N \text{ seeds} \times 3 \text{ curriculum checkpoints} \times 3 \text{ conditions} = 9N \text{ new formal-budget runs}
$$

(plus the $N$ curriculum runs themselves, each $\approx 1{,}200{,}000$
steps).

**All $N$ seeds are used and reported regardless of what quality they turn
out to have** — including if all $N$ turn out as uniformly high-competence
as the existing five (900101, 900102, 900104, 910101, 910102 all show
flat, near-ceiling C4/C16/C64 gates). That outcome would itself be
informative (it would suggest 900103/910102-style mid-training difficulty
is rare, not something to go looking for) and must not be used as a
reason to discard the seeds or draw more.

## 3. Compute scope

| Track | New training volume |
|---|---|
| A (900103) | $6 \times 800{,}000 = 4{,}800{,}000$ steps |
| B (per new seed) | $\approx 1{,}200{,}000$ (curriculum) $+ 9 \times 800{,}000 = 8{,}400{,}000$ steps |
| B total ($N$ seeds) | $N \times 8{,}400{,}000$ steps |

For reference, the original 18-run formal stage was $18 \times 800{,}000
= 14{,}400{,}000$ steps of welfare fine-tuning (plus the pre-existing
curriculum). Track A alone is roughly $1/3$ of that. Track B's cost scales
linearly and steeply with $N$ — this is the main reason $N$ must be
decided deliberately (§6), not defaulted to something large.

Both tracks can run with the same process-level parallelism already used
for the 18-run formal launch (32-core machine, one process per run, no
GPU benefit per the earlier CPU/GPU benchmarking finding) and are mutually
independent — Track A and Track B do not block each other, and Track B's
$N$ seeds' curriculum stages can themselves run in parallel with each
other.

## 4. Evaluation

Identical to the existing formal protocol, no exceptions:

- Curriculum qualification gates: checkpoint-Q-ensemble
  $K(S)=\{S-150{,}000,S-100{,}000,S-50{,}000,S\}$, $\epsilon=0$, 64-scenario
  `Q.json` bank (Track B only — Track A reuses 900103's existing gate
  results for C4/C16/C64).
- Welfare fine-tune evaluation: checkpoint-Q-ensemble $K(2{,}000{,}000)$,
  $\epsilon=0$, $R{=}50\,\mathrm{m}$, H0 (Mean only) and H1 (all three
  conditions) banks — same as `evaluate_formal_welfare.py`.
- Behavioural measures (hard-brake at the $-3.0\,\mathrm{m/s^2}$ physical
  floor, merge order) computed the same way as
  `evaluate_formal_behavioral.py`, for consistency with §5.6.5's existing
  analysis.

## 5. Governance (frozen once this document is finalized)

1. No selection of which new seeds (Track B) to keep based on their
   curriculum or fine-tuning outcomes. All $N$ pre-registered seeds are
   reported.
2. No changes to $\lambda_W$, reward coefficients, network architecture,
   or evaluation protocol anywhere in this diagnostic.
3. No choosing the intermediate checkpoint-freeze step *after* seeing how
   good it turns out to be — Track A uses the curriculum's own existing,
   already-frozen C4/C16 checkpoints (chosen for existing, not new,
   reasons); Track B's checkpoint schedule is the same fixed schedule
   already used for the six formal seeds' curricula.
4. This diagnostic's results do not retroactively alter the frozen 6-seed
   primary RQ1/RQ2 analysis (`05 results.md` §5.5–5.6). They are reported
   as a separate, clearly-labelled mechanistic diagnostic, exactly as
   §5.7 already is.
5. The falsifiable prediction in §1.4 is written down before any new run
   starts. If the data doesn't match it, that is reported as a
   disconfirmation of the baseline-quality-dependence hypothesis, not
   quietly reinterpreted.

## 6. Open decisions (must be filled in before freezing)

- [ ] $N$ (number of new Track B seeds) — proposed default: 3.
- [ ] Exact new seed IDs for Track B (must not overlap any existing
  pilot/smoke/formal seed range).
- [ ] Confirm Track B's curriculum checkpoint schedule is identical to the
  original six seeds' (no deviation), or state explicitly if anything
  differs and why.
- [ ] Confirm compute/wall-clock budget is acceptable given §3's totals.

## Sources

- Siddique, U., Weng, P., & Zimmer, M. (2020). *Learning Fair Policies in
  Multi-Objective (Deep) Reinforcement Learning with Average and
  Discounted Rewards.* ICML 2020.
  http://proceedings.mlr.press/v119/siddique20a/siddique20a.pdf
- Yu, X., Siddique, U., & Weng, P. *Fair Deep Reinforcement Learning with
  Generalized Gini Welfare Functions.*
  https://alaworkshop2023.github.io/papers/ALA2023_paper_34.pdf
- Ouyang, L., et al. (2022). *Training Language Models to Follow
  Instructions with Human Feedback.* NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf
- Lin, Y., et al. (2024). *Mitigating the Alignment Tax of RLHF.* EMNLP
  2024. https://arxiv.org/html/2309.06256v4
- Ash, J. T., & Adams, R. P. (2020). *On Warm-Starting Neural Network
  Training.* NeurIPS 2020.
  https://papers.neurips.cc/paper_files/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf
