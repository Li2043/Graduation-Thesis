# STAGE 9 — FORMAL THREE-CONDITION CONFIRMATORY EXPERIMENT PROTOCOL

## Status

**Frozen, activated.** Tag: `stage9-confirmatory-v1`. Parameters drafted
2026-08-03, following the six prerequisites sketched in
`paper/STAGE8_PLAN_DRAFT.md` §6 (written 2026-08-02, before Stage 8's gate
result was known). The four open items originally listed in §10 were
resolved with the user on 2026-08-03 and are recorded there (not deleted) —
this document is now implementable. No parameter below may change after
mean_pbrs/min_pbrs results are observed.

This protocol supersedes Stage 6A/6B/6B-H1 (hub `E16`–`E18`) as the
dissertation's baseline/mean_pbrs/min_pbrs comparison. It does not modify or
re-litigate the Stage 8 formal qualification gate (`stage8-gate-protocol-v1`,
FAIL, `final_new_stage8/analysis/stage8_gate/v1/STAGE8_GATE_DECISION.md`) —
that result stands as recorded.

## 1. Purpose

Formally address Chapter 2 §2.10's RQ1–RQ3 at the same scale and
methodological rigor as the Stage 8 gate (400K steps, Double DQN, full
learning-curve checkpoint schedule, corrected utility accumulation), rather
than at Stage 6A's 100K/single-endpoint scale.

**Why this supersedes E16, stated explicitly (for the dissertation text,
not only this protocol):**

- E16's baseline success rate was 35.0% — far below any competence
  threshold, and `stage6b_h1_execution_report.md` §35.9 already records
  "the baseline competence gate was not reliably passed."
- E16's mean_pbrs condition had 0/10 swap-estimable blocks — convention
  analysis (RQ1/RQ2) was not computable for that condition.
- E16 used a single 100K training endpoint, not a learning curve — no
  adjacent-checkpoint stability, material-regression, or late-collapse
  check was possible.
- E16's episode-utility accumulation had a bug (final-state value used
  instead of active-state trajectory mean), corrected in Stage 6B-H1 but
  never re-run at the training level (only the existing 100K checkpoints
  were re-evaluated).
- Chapter 2 §2.5/§2.9 currently describes the two PBRS potentials as
  "RMS-magnitude calibrated"; the λ values actually used
  (`stage5c0_h1_r1_100k_protocol/artifacts/.../final_pbrs_parameters.yaml`)
  are `lambda_mean=0.2, lambda_min=0.2, rms_matched: false,
  comparison_type: equal_coefficient`. This is a text/implementation
  mismatch already flagged in `PAPER_CHANGES_REQUIRED_LATER.md` items 1–2,
  never applied to dissertation text.

**Explicitly not reopened by this protocol:**

- The equal-coefficient λ design (λ_mean = λ_min = 0.2) — reused verbatim
  from Stage 5C0-H1-R1, not re-derived. See §2 and §10 for the associated
  open confirmation item.
- Double DQN / Base Reward V2 / LR-decay hyperparameters — reused verbatim
  from the Stage 8 gate protocol.
- Environment, comfort, and PBRS-math locks (Stage 3B-R1, 4A-R1, 2A) —
  unchanged.

**Explicitly does not require a fresh baseline competence-gate PASS** — see
§6 for the substitute interpretive constraint, which is Chapter 2 §2.9's own
stated design, not a new relaxation invented for this protocol.

## 2. Algorithm / reward spec (frozen, identical across all 3 conditions except reward)

Shared DQN configuration (verbatim from `stage8-gate-protocol-v1`):

```
algorithm_mode              = double_dqn
target_update_mode          = hard
target_sync_interval_updates = 250
learning_rate                = 0.0005
learning_rate_end            = 0.0001
learning_rate_decay_environment_steps = 400000
epsilon_decay_environment_steps = 50000
epsilon_after_decay          = 0.10
base_reward_version           = v2_active_time
active_time_cost_per_step     = 0.0005
```

Per-condition reward:

- **baseline**: base reward only, λ = 0.0 (no PBRS term). No new training —
  see §4.
- **mean_pbrs**: `r' = r_base + λ_mean·[γ·Φ_mean(s_{t+1}) − Φ_mean(s_t)]`,
  λ_mean = 0.2
- **min_pbrs**: `r' = r_base + λ_min·[γ·Φ_min(s_{t+1}) − Φ_min(s_t)]`,
  λ_min = 0.2

Potentials (verbatim from `src/thesis/rewards/pbrs_v2.py`, verified against
Chapter 2 §2.6/§2.9 line-for-line in this session — no discrepancy found in
functional form):

```
V = {A, B, B_front, B_rear}
e_i(s) = clip01(v_i(s) / v_target_i)
E_i(s) = 1 if stakeholder i has safely exited, else e_i(s)
Phi_mean(s) = (1/4) * sum_{i in V} E_i(s)
Phi_min(s)  = min_{i in V} E_i(s)
Phi_c(s) = 0 if s is true terminal (success or collision), else raw potential
           (truncation preserves the raw potential — does not zero it)
gamma_shaping = 0.995 = gamma_learner (must match within 1e-12, per PBRSConfig.validate())
```

**Design note, FROZEN 2026-08-03 (per governance §9):** λ_mean = λ_min =
0.2 is an *equal-coefficient* comparison, not RMS-magnitude-matched, and is
reused verbatim from the frozen `final_pbrs_parameters.yaml`
(`stage5c0_h1_r1_100k_protocol`, `pbrs_parameters_final: true`,
`pilot_comparative_outcomes_used_for_selection: false` — confirming the
original choice was not itself the product of post-hoc pilot tuning).
Decision: equal-coefficient is retained; a new RMS-magnitude calibration
(policy-independent transition library + calibration procedure, neither of
which exist anywhere in this codebase) is rejected as new, unvalidated
scope this late in the project. This is defensible, not merely expedient:
Chapter 2 §2.5 already frames D-mean as "the necessary control for D-min...
retains the stakeholder information and PBRS structure while changing only
the aggregation rule" — with λ held equal across D-mean and D-min, the
D-mean-vs-D-min comparison already isolates the aggregation-rule variable;
RMS-matching would additionally rescale away a difference in realized
signal magnitude that is arguably intrinsic to the aggregation rule itself
(mean vs. min), not an artificial confound. Chapter 2's text describing RMS
calibration is inaccurate and must be corrected to describe
equal-coefficient comparison — a dissertation text fix, not a protocol
parameter change.

## 3. Timestep / episode / environment

Identical to the Stage 8 gate protocol: one joint environment transition = 1
timestep (controllers are not double-counted). Environment, comfort, and
PBRS-math locks unchanged from Stage 3B-R1 / 4A-R1 / Stage 2A.

## 4. Seeds

| Condition | Seeds | Training |
|---|---|---|
| baseline | `65021`–`65040` (n=20) | **reused verbatim** from the Stage 8 gate (`stage8-gate-protocol-v1`) — not retrained |
| mean_pbrs | `66001`–`66020` (n=20) | new |
| min_pbrs | `67001`–`67020` (n=20) | new |

**Baseline-reuse justification:** the Stage 8 gate was already a complete,
protocol-frozen 20×400K run under the exact algorithm/reward spec this
protocol also uses for baseline. Retraining would reproduce statistically
similar, not materially more informative, data at full additional compute
cost — a violation of the project's own "don't waste a completed formal run"
practice already established when Stage 6B-H1 re-evaluated Stage 6A's
existing checkpoints instead of retraining.

Forbidden seed blocks (cumulative, all prior stages plus this stage's own
mean_pbrs/min_pbrs blocks): `61001`–`67020` inclusive.

`protocol_tag` for the two newly trained conditions: `stage9-confirmatory-v1`
(keeps the eval-seed SHA-256 namespace disjoint from `stage8-gate-protocol-v1`
and all prior stages). Baseline's evaluation data keeps its original
`stage8-gate-protocol-v1` tag — it is not re-evaluated under the new tag.

## 5. Training budget & checkpoints

Identical to the Stage 8 gate: 400,000 steps; checkpoints every 25,000 steps
(17 total, `0..400000`); early (≤175,000: 8 scenario blocks, 16
episodes/seed-checkpoint) / late (>175,000: 32 scenario blocks, 64
episodes/seed-checkpoint) evaluation-plan split, identical mechanism
(`n_scenario_blocks`, `template_validation_index = scenario_block % 8`).
`RICH_LOG_CHECKPOINTS = (0, 350000, 375000, 400000)` for per-step
Q-value/trajectory logging; all other checkpoints use the lightweight
episode-level evaluator. Only applies to the two newly trained conditions —
baseline's checkpoints already exist.

## 6. Interpretive constraint used in place of a fresh competence gate

This stage does **not** re-run a PASS/FAIL competence gate on baseline
(already run and recorded FAIL under `stage8-gate-protocol-v1`;
re-litigating it here would violate this project's frozen-protocol
discipline — a stage does not get to re-grade an earlier stage's already-
observed result). Instead, per Chapter 2 §2.9's own stated design:

> "Convention analysis then examines q, p_MF and D_swap. Welfare comparison
> follows only after the policies resolve a sufficient proportion of
> certified conflicts."

this protocol applies that constraint literally:

- RQ1–RQ3 convention analysis (`q`, `p_MF`, `D_swap`) and welfare comparison
  (`U_mean`, `U_min`, per Chapter 2 §2.6's episode-level formula) are
  computed **only over certified choice states**
  (`src/thesis/certification/choice_state_scenarios.py` — states where a
  pre-registered script proves both mainline-first and ramp-first are
  feasible), not the raw aggregate episode pool used for the Stage 8 gate's
  competence criteria.
- The realized resolution proportion within certified choice states, for
  all three conditions, is reported as a primary descriptive quantity
  **before** any welfare number is interpreted. If it is low for any
  condition (e.g., comparable to E16's degenerate 0/10 swap-estimable
  blocks for mean_pbrs), the report states plainly that welfare comparison
  is not interpretable for that condition — it does not present a welfare
  number without this caveat attached.
- Data-completeness integrity (seed × checkpoint × episode counts present,
  no NaN) is still checked mechanically and yields a binary VALID/INVALID
  flag, same discipline as every prior stage — this is separate from, and
  does not substitute for, the resolution-proportion caveat above.

## 7. Statistical design (sample size / power) — FROZEN 2026-08-03

Two different statistical questions require two different frameworks. Using
one framework (difference-detection power) for both was an error in the
original draft of this section, corrected here before freezing.

### 7.1 RQ1/RQ2 — does the condition change the outcome (difference detection)

Computed from the Stage 8 gate's own observed seed-level variance (n=20,
pooled across the 3 gate checkpoints — 350K/375K/400K, 60 seed-checkpoint
observations, this session):

| metric | observed SD (pooled, baseline) |
|---|---|
| success_rate | 0.209 |
| collision_rate | 0.071 |
| swap_eligibility | 0.311 |

Required n per group, two-sample two-sided test, α=0.05, power=0.80, equal n:

| metric | minimum effect size (MES) | required n/group |
|---|---|---|
| success_rate | 0.10 | 69 |
| success_rate | 0.15 | 31 |
| success_rate | 0.20 | 18 |
| collision_rate | 0.05 | 33 |
| collision_rate | 0.10 | 9 |
| swap_eligibility | 0.15 | 68 |
| swap_eligibility | 0.20 | 38 |

**Frozen decision:** n=20 per condition (matching baseline's already-fixed
seed count, for budget parity with every prior formal stage — not
independently derived by this power analysis). Success-rate MES = 0.20
(20 points) and swap_eligibility MES = 0.20 are accepted as this design's
detection floor for RQ1/RQ2 superiority-style claims. Effects smaller than
this must be reported as "not detected — underpowered at this n," never as
"no effect" or "equivalent" (`PAPER_CHANGES_REQUIRED_LATER.md` item 9).
Re-running with a larger n to chase a smaller MES was considered and
rejected: it would roughly double the compute/time cost of an already
400K×20-per-condition experiment, which is not available in this project's
remaining timeline.

### 7.2 RQ3 — does min_pbrs avoid material loss (non-inferiority, precision framing)

Non-inferiority does not need difference-detection power; it needs the
confidence interval around the observed difference to be narrow enough to
rule out a loss larger than a pre-specified margin. At n=20/group, the 95%
CI half-width for a two-sample mean difference is
`1.96 * sqrt(2) * SD / sqrt(20) ≈ 0.62 * SD`. Using the observed SDs above,
n=20 is already adequate precision for the margins frozen below (each
margin exceeds the corresponding CI half-width).

**Frozen non-inferiority margins** (2026-08-03, set from a mix of this
project's own existing safety threshold and a freshly computed baseline
anchor — not derived from the comparison data itself):

| outcome | margin | basis |
|---|---|---|
| collision rate | ≤ 0.03 (3 points) absolute increase vs. baseline | matches the scale of this project's own `GATE_COLLISION_MAX=0.02` gate threshold plus slack; CI half-width at n=20 ≈ 0.044, so this margin is only marginally resolvable — treat any RQ3 collision non-inferiority claim at this n as provisional, not a strong claim |
| mean learner mobility (U̅, A/B only — see note) | ≤ 10% relative reduction vs. baseline | baseline anchor computed this session directly from Stage 8 gate's rich trajectories (350K/375K/400K, Stage-6B-H1-style active-state accounting, collision override, A/B controllers only): **U̅_baseline ≈ 0.897** (pooled SD ≈ 0.248, n=7680 episode-controller rows) → floor ≈ **0.807** |
| coordination resolution (q, certified choice states) | ≤ 0.05 (5 points) absolute decrease vs. baseline | same scale logic as collision, one notch looser since q is a coarser-grained safe-resolution measure |

**Note on the mobility anchor's scope:** this is the mean of `U_i` for the
two *learning* controllers (A, B) only — Chapter 2's full potential is
defined over all four stakeholders `V={A,B,B_front,B_rear}`, but
`B_front`/`B_rear` speed/target-speed telemetry is not present in the
current per-step trajectory logs (only `controller∈{A,B}` rows exist;
`target_speed` itself is a single fixed constant, 20.0 m/s, shared by all
stakeholders — see `merge_env_v2.py::MergeEnvConfig.target_speed`). The A/B
proxy is accepted as the anchor for this margin because A/B are the
stakeholders the shaping term most directly targets; computing the full
4-stakeholder value for mean_pbrs/min_pbrs's own confirmatory analysis (not
just this margin-setting step) requires extending the evaluator's per-step
logging to include background-vehicle telemetry — flagged as an
implementation requirement for the Stage 9 evaluator, not merely for this
margin.

## 8. Evaluation

Reuse unchanged: `stage8_gate_eval.evaluate_checkpoint_stage8_gate` (rich,
the 4 `RICH_LOG_CHECKPOINTS`) and `stage7c_q1_eval.evaluate_checkpoint_stage7c`
(lightweight, the other 13 checkpoints), called with
`protocol_tag=stage9-confirmatory-v1` for mean_pbrs and min_pbrs only.
Baseline's existing Stage 8 gate evaluation output
(`final_new_stage8/results/stage8_gate/v1/`) is reused as-is — no
re-evaluation.

## 9. Governance

1. No parameter in this protocol (seeds, λ values, checkpoint schedule, MES
   targets once set) may change after any condition's results are observed.
2. No early stopping, no best-checkpoint selection — all three conditions
   train to the full 400,000-step budget and are evaluated at every
   pre-declared checkpoint.
3. Pilot/smoke-test seeds must not overlap `61001`–`67020`.
4. This stage does not re-derive or adjust the Stage 8 gate's already-
   recorded FAIL; it is cited as historical motivation only (per this
   project's established norm against treating an earlier stage's result as
   more than that).
5. λ_mean = λ_min = 0.2 (equal-coefficient) is retained from Stage
   5C0-H1-R1 verbatim for this run — do not silently substitute an
   RMS-matched value without a new, separately frozen protocol.
6. Full checkpoints/replay buffers stay off git (project `.gitignore`
   convention); code, configs, protocol docs, evaluation CSV/JSON outputs
   go in version control, matching the code/results/analysis branch
   separation used for Stage 7C-Q1 and Stage 8.

## 10. Open items — RESOLVED 2026-08-03 (kept as an audit trail, not deleted)

- [x] **MES values in §7.1** — accepted as the honest detection floor at
  n=20 (success/swap MES=0.20), not increased; re-running at a larger n to
  chase a smaller MES was explicitly rejected on time/compute grounds. See
  §7.1's "Frozen decision."
- [x] **Non-inferiority margins for RQ3** — set (§7.2): collision ≤0.03
  absolute, mean learner mobility ≤10% relative (anchored to a freshly
  computed baseline value, U̅≈0.897, not E16's superseded number),
  resolution (q) ≤0.05 absolute. Flagged explicitly as this session's
  proposed values, to be revisited with the advisor if a different standard
  of "acceptable loss" is required — but the protocol does not proceed with
  the margins unset.
- [x] **λ = 0.2/0.2 equal-coefficient vs. RMS-magnitude calibration** — 
  equal-coefficient retained (§2), reasoned justification recorded there,
  not merely deferred.
- [x] **E16-supersession framing** — accepted: E16 is framed as a
  formative/exploratory-stage result whose documented limitations motivated
  Stage 7-8's diagnostic work and this confirmatory redesign, not as "E16
  was wrong" — matching how Stage 6B was already marked superseded by
  Stage 6B-H1. E16's numbers remain in the dissertation, captioned with
  their limitations, rather than removed.
- [ ] **Certified-choice-state definition** (§6) — still not yet
  cross-checked against `src/thesis/certification/choice_state_scenarios.py`
  in this session. This is a correctness check on existing code, not a
  design decision, and does not block freezing this protocol's parameters —
  but must be done before the Stage 9 evaluator is implemented and trusted.
