# Stage 2A — PBRS Pure-Function and Mathematical Correctness Tests

## Purpose

Validate **Mean-PBRS** and **Min-PBRS** mathematics in isolation:

- stakeholder experience definitions;
- fixed four-member stakeholder set \(V\);
- raw vs actual potentials;
- true terminal vs external truncation;
- PBRS transition signal \(F_t = \gamma\,\phi(s_{t+1}) - \phi(s_t)\);
- shaped reward \(r_{\text{shaped}} = r_{\text{base}} + \lambda\,F\) (no base rescaling);
- terminal and truncated telescoping identities.

**No environment, replay buffer, DQN, or policy training is tested in this stage.**

## Fixed stakeholder membership

\[
V = \{A,\, B,\, B_{\text{front}},\, B_{\text{rear}}\}
\]

Learning controllers:

\[
N = \{A,\, B\}
\]

Membership is fixed for the whole episode. Completed stakeholders are **not**
removed. The mean denominator remains **4**.

## Active and completed experience

Active:

\[
e_i(s) = \mathrm{clip}(v_i(s) / v_{\text{target},i},\, 0,\, 1)
\]

Safely completed (episode continues):

\[
E_i(s) = 1
\]

Negative speed → 0; overspeed → 1 (never above 1).

## Mean and Min potentials

Non-terminal:

\[
\mathrm{raw}\,\phi_{\text{mean}}(s) = \tfrac{1}{4}\sum_{i\in V} E_i(s),\qquad
\mathrm{raw}\,\phi_{\text{min}}(s) = \min_{i\in V} E_i(s)
\]

Actual:

\[
\phi_c(s) =
\begin{cases}
0 & \text{if } s \text{ is a true terminal state}\\
\mathrm{raw}\,\phi_c(s) & \text{otherwise}
\end{cases}
\]

External truncation (`terminated=False`, `truncated=True`) does **not** zero
\(\phi\). Simultaneous `terminated=True` and `truncated=True` is rejected.

## PBRS transition timing

\[
F_{c,t} = \gamma\,\phi_c(s_{t+1}) - \phi_c(s_t)
\]

\[
r_{\text{shaped}}[i,t] = r_{\text{base}}[i,t] + \lambda_c\, F_{c,t}
\]

Same common \(F\) for A and B; each keeps its own base reward.
Shaping \(\gamma\) must equal learner \(\gamma\).

`lambda_mean` / `lambda_min` in Stage 2A configs are **TEST-ONLY placeholders**.

## Telescoping identities

True terminal (\(\phi_T = 0\)):

\[
\sum_{t=0}^{T-1} \gamma^t F_t = -\phi_0
\]

External truncation (\(\phi_K \neq 0\)):

\[
\sum_{t=0}^{K-1} \gamma^t F_t = -\phi_0 + \gamma^K \phi_K
\]

Tolerance: `1e-12`.

## How to run

From the `final_new` repository root:

```bash
python experiments/pre_impl/stage2a_pbrs_math_tests/scripts/run_stage2a_tests.py
```

## Output locations

Unique run directories (never overwritten):

```
data/raw/<run_id>/
data/processed/<run_id>/
reports/<run_id>/
logs/<run_id>/
artifacts/<run_id>/
```

Pointer only: `latest_run.json` (history preserved).

## Acceptance criteria

- All required pure tests pass
- Stakeholder mapping is exactly the fixed four-member set
- Completed stakeholders remain with \(E_i = 1\); mean denominator stays 4
- True terminal potential is zero; truncation potential is not zeroed
- Both telescoping identities pass within `1e-12`
- Shaping gamma equals learner gamma
- Base reward is not rescaled; \(\lambda=0\) reproduces base
- No non-finite values accepted
- Complete manifest/report every run; Stage 1 outputs untouched

A dirty git tree is reported prominently. Dissertation-grade retained runs
should have `git_dirty = false`.

## What remains for Stage 2B

- Environment / simulator wiring of potentials
- Replay / learner integration of shaped rewards
- Any later calibration of \(\lambda\) (not done here)
- DQN training (explicitly out of scope for 2A)
