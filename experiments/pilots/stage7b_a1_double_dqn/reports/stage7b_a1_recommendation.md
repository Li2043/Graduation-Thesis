# Stage 7B-A1 Recommendation

## Decision

**B — algorithm stabilisation was beneficial but insufficient**

## Gate

- Vanilla passed: False
- Double passed: False
- Consecutive checkpoint confirmation: False
- Stable sufficient budget: `none`

## Stability

- Vanilla late collapses: 2
- Double late collapses: 4
- Seed bifurcation reduced: True
- Unilateral stall reduced (paired mean Δ<0): True

## Safety

competence improvement accompanied by safety degradation

## Next experiment

1. Do **not** declare Double competence-qualified.
2. Prefer a **reward / active-time-cost / stall-resolution** single-factor pilot next.
3. If Double is retained as the learner default, freeze it explicitly in the next protocol and draw **new** seeds outside 610xx/620xx/630xx.
4. Do not extend budget alone as the primary next step.
