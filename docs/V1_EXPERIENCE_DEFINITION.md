# V1 Experience Definition

> Status: **DRAFT SKELETON** — placeholder sections only. This document will
> formally define the per-vehicle "experience" used by the Rawlsian maximin
> term and by fairness metrics. Nothing here is final; do **not** treat these as
> the implemented definitions. Code is not modified by this document — any change
> to the experience function must go through the decision log first.

---

## 1. Mobility

_TBD_ — definition of the mobility component of experience.

- Construct: _TBD_
- Inputs / signals: _TBD_
- Normalisation: _TBD_
- Range: _TBD_

## 2. TTC / collision risk

_TBD_ — definition of the time-to-collision / proximity risk component.

- Construct: _TBD_
- Inputs / signals: _TBD_
- Threshold(s): _TBD_
- Range: _TBD_

## 3. Accumulated waiting

_TBD_ — definition of accumulated waiting / delay component.

- Construct: _TBD_
- Inputs / signals: _TBD_
- Accumulation rule: _TBD_
- Range: _TBD_

## 4. Collision outcome

_TBD_ — definition of the terminal collision outcome component.

- Construct: _TBD_
- Trigger condition: _TBD_
- Penalty treatment: _TBD_

## 5. Combined experience

_TBD_ — how the components above are combined into a single scalar experience
per vehicle.

- Combination form: _TBD_ (e.g. weighted sum)
- Weights: _TBD_
- Sign convention (higher = better): _TBD_
- Edge cases: _TBD_

## 6. Least-advantaged vehicle

_TBD_ — how the least-advantaged (worst-off) vehicle is selected for the
Rawlsian maximin term in the dual controlled-vehicle setting.

- Candidate set / scope: _TBD_
- Selection rule (min experience): _TBD_
- Tie-breaking: _TBD_
- Diagnostic attribution (reason for disadvantage): _TBD_
