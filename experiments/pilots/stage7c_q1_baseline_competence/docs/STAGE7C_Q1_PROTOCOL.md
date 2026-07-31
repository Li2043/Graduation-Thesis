# STAGE 7C-Q1 PROTOCOL — Base Reward V2 Baseline Competence Qualification Pilot

## Status

Frozen qualification protocol. Annotated tag: `stage7c-q1-protocol-v1`.

## 1. Purpose

Stage 7C-Q1 is a **single-condition qualification pilot**.

It trains only:

- Double Independent DQN;
- Baseline reward condition (no Mean-PBRS, no Min-PBRS);
- Base Reward V2 (active-time cost);
- master seeds `64001`–`64020`;
- maximum `400000` **joint** environment timesteps.

It does **not**:

1. compare Base Reward V1 vs V2 causal differences;
2. treat Stage 7B old-reward outcomes as anything other than **historical motivation**;
3. test Mean-PBRS or Min-PBRS;
4. search reward coefficients;
5. allow early stopping or best-checkpoint selection.

## 2. Base Reward V2

\[
r^{V2}_{i,t}=r^{V1}_{i,t}-0.0005\,I_i^{\mathrm{active}}(s_t)
\]

Equivalent form: \(-c_T\Delta t\,I_i^{\mathrm{active}}(s_t)\) with
\(c_T=0.0025\) per simulated second and \(\Delta t=0.20\) s.

Active indicator uses the **transition-start** state:

- `I_active(s_t)=1` if the learner is still on-road / controlled at `s_t`;
- `I_active(s_t)=0` if the learner already safely exited before `s_t`.

Frozen coefficient: `active_time_cost_per_step = 0.0005` (no coefficient search).

## 3. Algorithm

Double DQN target only:

\[
a^*=\arg\max_a Q_{\mathrm{online}}(s',a),\qquad
y=r+\gamma Q_{\mathrm{target}}(s',a^*).
\]

Vanilla DQN is forbidden. Observation, action space, network, optimizer, learning
rate, replay, epsilon schedule, gamma, target sync, geometry, collision logic, and
evaluation policy are unchanged from Stage 6 / Stage 7B except for the reward term
and the 400K budget.

## 4. Timestep definition

One **joint** environment transition counts as one timestep, even though two
learning agents act. Agents are not double-counted.

## 5. Evaluation

Greedy evaluation (`epsilon=0`).

Eval seeds use stable SHA-256 (never Python `hash()`):

```text
eval_seed = SHA256(protocol_tag | master_seed | checkpoint | scenario_block)
```

Assignments A and B of the same scenario block share the same base eval seed.
Different master seeds must not share overlapping eval seeds.

Episode counts:

- checkpoints `≤175K`: 8 scenario blocks × 2 assignments = 16 episodes;
- checkpoints `≥200K`: 32 scenario blocks × 2 assignments = 64 episodes.

Blocks `0..7` are the frozen standard subset comparable to historical 16-episode
protocols. Late 64-episode sets are complete; do not generate a second independent
16-episode corpus.

## 6. Checkpoints

Exact schedule: `0,25K,...,400K` (17 points). All 20 seeds train to 400K.
No early stop. No best-checkpoint selection. Gate uses **350K, 375K, 400K**.

## 7. Competence gate outcomes

Machine gate may emit only:

- `PASS` — data complete and all criteria satisfied;
- `FAIL` — data complete but at least one criterion fails;
- `INVALID` — integrity / protocol / seed / overlap / completeness failure.

There is no formal “near pass”.

PASS requires (among other frozen rules in `stage7c_q1_gate.py`):

- at 350K, 375K, and 400K: mean success ≥ 0.95; collision ≤ 0.02; truncation ≤ 0.03;
  swap eligibility ≥ 0.75;
- the **intersection** of seeds with success ≥ 61/64 across those three checkpoints
  has size ≥ 16;
- learning-curve adjacent success drops ≤ 0.03 on 200K…400K;
- material regressions and late collapses within frozen caps.

## 8. Governance (frozen)

1. Stage 7C-Q1 is a single-condition qualification pilot.
2. Do not compare V1 vs V2 causal differences in this stage.
3. Stage 7B old-reward results are historical motivation only.
4. This stage does not test Mean-PBRS or Min-PBRS.
5. The coefficient `0.0005` is frozen before training.
6. Reward coefficient search is forbidden.
7. All 20 seeds must train to 400K joint environment steps.
8. Early stopping is forbidden.
9. Best-checkpoint selection is forbidden.
10. The competence gate uses fixed checkpoints 350K, 375K, and 400K.
11. PASS is required before any final three-condition experiment.
12. FAIL stops further algorithm or reward modification under this pilot’s intent.
13. Pilot / smoke seeds must not enter formal experiment seed blocks.
14. Checkpoints remain on the experiment machine (or its independent backup).
15. GitHub transports only code, CSV, JSON, manifests, reports, and figures — never
    full checkpoints / replay / weights archives.

## 9. Pre-training scripted audit

A scripted Base Reward V2 audit must pass before formal long-run training.
Failures must report trajectory decompositions. The coefficient `0.0005` must not be
changed to force a pass.
