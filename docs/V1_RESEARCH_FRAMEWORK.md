# V1 Research Framework

> Status: **DRAFT SKELETON** — structure only. No results are claimed here.
> This document defines the scope and research questions for V1. It does not
> report any experimental outcome. Fill placeholders (`_TBD_`) as decisions are
> made and recorded in `V1_DECISION_LOG.md`.

---

## 1. V0 / V1 boundary

### 1.1 What V0 was (Prototype)

All prior work (v0.1 – v0.6.3, as summarised in the root `README.md`) is treated
as **V0 / Prototype**. V0 is frozen as historical reference and is **not** part
of the V1 evidence base. Key V0 characteristics:

- Single controlled ego vehicle + environment-controlled background vehicles.
- Iterative, exploratory changes to scope, experience function, and parameters.
- Single-seed and small multi-seed runs used mainly for pipeline validation.
- Results in `results/` are prototype artefacts, not V1 deliverables.

### 1.2 What V1 is

V1 establishes a **controlled, reproducible, auditable** experiment framework for
**dual controlled-vehicle** Rawlsian reward shaping. V1 does not inherit V0
results; it re-runs experiments under a fixed protocol once that protocol is
frozen.

V1 boundary rules:

- V1 fixes its design before final runs (see `V1_EXPERIMENT_PROTOCOL.md`).
- Every V1 experiment is registered in `experiments/registry.csv`.
- Every notable design choice is recorded in `V1_DECISION_LOG.md`.
- Any AI-assisted change is recorded in `V1_AI_USAGE_LOG.md`.

### 1.3 V0 → V1 differences (to be finalised)

| Aspect | V0 (Prototype) | V1 (Target) |
| --- | --- | --- |
| Controlled vehicles | Single ego | _TBD_ (dual controlled) |
| Fairness scope | global / ego_neighbourhood | _TBD_ |
| Experience function | safety_mobility proxy | _TBD_ (see `V1_EXPERIENCE_DEFINITION.md`) |
| Seed protocol | ad hoc | Frozen train/eval split (see protocol) |
| Auditability | partial | Full registry + decision/AI logs |

---

## 2. Main research question

**_TBD_** — single overarching question for V1.

Working placeholder:

> Does Rawlsian maximin reward shaping, applied to a dual controlled-vehicle
> setting, change learned driving behaviour in a way that improves fairness for
> the least-advantaged vehicle without unacceptable safety or efficiency cost?

---

## 3. Research questions

### RQ1 — Fairness

_TBD_. Placeholder framing:

- Does Rawlsian shaping improve the experience of the least-advantaged
  (worst-off) controlled vehicle relative to the egoistic baseline?
- Primary fairness construct and metric: see `V1_METRIC_DEFINITIONS.md`.

### RQ2 — Safety / efficiency trade-off

_TBD_. Placeholder framing:

- What is the trade-off between safety (collision / risk) and efficiency
  (mobility / throughput / delay) introduced by Rawlsian shaping?
- How is the trade-off characterised and reported?

### RQ3 — Behavioural failure diagnosis

_TBD_. Placeholder framing:

- When Rawlsian shaping fails (no improvement or regression), what behavioural
  mechanism explains it (e.g. least-advantaged vehicle uncontrollable, reason
  attribution, seed sensitivity)?
- Which diagnostic signals are used to explain failures?

---

## 4. Core contribution

_TBD_. Placeholder:

- A controlled, reproducible, auditable evaluation framework for Rawlsian
  reward shaping in a dual controlled-vehicle merge setting, plus the empirical
  answer to RQ1–RQ3 under a frozen protocol.

---

## 5. Optional extension

_TBD_. Candidates (not committed):

- Scaling beyond two controlled vehicles.
- Richer experience function (waiting time, TTC, headway, merge success).
- Human-in-the-loop weight calibration (SYMPLEX-style).
- Symbolic / interpretable norm extraction.

---

## 6. Out of scope (V1)

The following are explicitly **out of scope** for V1 unless promoted via the
decision log:

- Full multi-agent RL (beyond the dual controlled-vehicle setting).
- Human-in-the-loop experiments with recruited participants.
- Runtime norm synthesis.
- Algorithms other than the agreed training algorithm.
- Environments other than the agreed environment.
- Re-using or re-interpreting V0 results as V1 evidence.
