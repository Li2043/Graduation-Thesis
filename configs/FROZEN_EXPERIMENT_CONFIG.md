# Frozen Experiment Configuration (source of truth)

Snapshot date: 2026-08-18. This document is a distilled, cross-checked
summary of the currently frozen Study B (N=4 heterogeneous highway
merge, R=50m locality amendment) protocol, verified directly against
the running source machine's code and tracking files at bundle-build
time. It is a **convenience reference**, not the authority itself --
the authority is `experiment_records/RUNBOOK.md` +
`experiment_records/autonomous_highwayenv/GATE_RESULTS.json` +
`experiment_records/autonomous_highwayenv/AUTONOMOUS_EXPERIMENT_STATE.json`
+ `experiment_records/autonomous_highwayenv/AUTONOMOUS_EXPERIMENT_LOG.md`.
If this document and those files ever disagree, the tracking files win.

## 1. Environment

- Backend: HighwayEnv 1.12.0 (`ThesisHighwayMergeEnv`, subclasses
  HighwayEnv's own unmodified `ConnectedLaneMergeGenericEnv`).
- Gymnasium 1.2.0.
- Road geometry (frozen): `before_merge_length=220.0m`,
  `converge_merge_length=80.0m`, `parallel_merge_length=80.0m`,
  `after_merge_length=120.0m`, `route_exit_margin=90.0m`,
  `route_exit_x=470.0m` (completion threshold),
  `route_total_x=500.0m` (progress-fraction denominator only),
  `lanes_count=2`, lane width `4.0m` (HighwayEnv
  `AbstractLane.DEFAULT_WIDTH`).
- Vehicle: `LENGTH=5.0m`, `WIDTH=2.0m` (rendering/kinematics only);
  `collidable=False`; project's own collision definition:
  `collision_distance_longitudinal_m=4.0`,
  `collision_lateral_threshold_m=1.5` (a logical flag check, not
  HighwayEnv's native position-perturbing collision response).
- Action representation: `meta_speed` (desired-speed controller,
  physical acceleration clipped to `[-3.0, +2.0] m/s^2`). 3 actions:
  HOLD / ACCELERATE / BRAKE.
- N=4 vehicles: 2 ramp + 2 mainline roles, heterogeneous target speeds
  18/22 m/s, matched-TTC spawn init.
- `episode_max_steps=200` for all curriculum/qualification/formal runs
  (5Hz policy rate).

## 2. Observation (R=50m locality amendment, FROZEN 2026-08-17)

- `LOCAL_OBS_DIM=18` = `SELF_OBS_DIM=6` + `NEIGHBOUR_SLOTS=3` x
  `NEIGHBOUR_OBS_DIM=4`.
- Self features: `[role, speed, target_speed, acceleration,
  dist_to_merge, prev_action]`.
- Neighbour features (per visible slot, nearest-first):
  `[presence, delta_d_norm, delta_v_norm, lane_relation]`. Never reads
  a neighbour's `target_speed`/exact position/reward/utility/value/
  intended action.
- `local_sensing_range_m = 50.0` (FROZEN): a genuine finite
  visibility range, distinct from `r_obs=50.0` (a pre-existing,
  unrelated `delta_d_norm` clip/normalization divisor -- same numeric
  value, different concept, NOT the reason R=50m was chosen). Distance
  metric: road-relative `|distance_to_merge(other) -
  distance_to_merge(ego)|` (not Euclidean). Boundary inclusive
  (`<= R` is visible). Vehicles outside range are excluded from
  selection entirely (not merely deprioritized) -- masked slots have
  `presence=0.0` and all-zero features, no farther-vehicle fallback.
  Chosen for information-structure/geometry reasons (Phase-8 spawn-time
  audit over the `Q.json` bank: R>=75m is ~unlimited, R=30m is a much
  stronger restriction, R=50m gives meaningful partial visibility).
- `v_scale=10.0`.
- Implementation: `src/thesis/study_b/local_observation.py`
  (`build_neighbour_observations`/`build_local_observation`,
  `local_sensing_range_m: float | None = None` parameter, default
  `None` preserves pre-amendment behaviour byte-for-byte) and
  `src/thesis/study_b/envs/highwayenv_wrapper.py`
  (`StudyBHighwayWrapperConfig.local_sensing_range_m`).

## 3. Solver: parameter-shared local DQN

- `QNetwork`: 2 hidden ReLU layers, `hidden_sizes=(64,64)`,
  `obs_dim=18`, `n_actions=3`, linear output.
- Optimizer: `torch.optim.Adam`.
- Learning rate: `0.0005` -> `0.0001`, linear decay,
  `lr_decay_steps_absolute=800000`.
- Epsilon: `1.0` -> `0.10`, linear decay,
  `eps_decay_steps_absolute=640000`.
- Replay: **plain uniform `ReplayBuffer`, NOT PER** (no alpha/beta).
  Capacity `100000`, batch size `64`.
- TD target: **1-step**, NOT n-step (`compute_td_targets()`'s own
  docstring: "1-step TD targets").
- Target network: hard sync every `250` gradient updates, Double DQN
  (`DQNTargetMode.DOUBLE`).
- `gamma=0.995`.
- Do not reopen: PER, n-step, batch size, architecture, LR schedule,
  epsilon schedule -- all previously verified directly against code,
  not to be re-derived or "improved."

## 4. Reward and welfare

- Per-step task reward: exit/collision/hard-brake/time-cost/progress
  terms (`ThesisHighwayMergeEnv._reward()`); `time_cost_per_step` is
  gated by `is_active_at_start`; the progress-delta term is computed
  unconditionally every step (documented as `AUDIT-2`, severity 0 --
  never enters training since the training loop skips transitions for
  already-inactive vehicles; only a ~0.003-unit diagnostic-logging
  inflation, zero effect on the learned policy).
- Direct terminal welfare bonus: `R_c^W = lambda_W * (W_c(U) - 1)`,
  applied once at episode end.
- `lambda_W = 0.5` -- **FROZEN**, do not reopen the 1.0/0.25 ladder.
- Mean: `W_mean(U) = mean_i(U_i)`.
- GGI: sort utilities ascending, weighted sum with weights
  `[0.4, 0.3, 0.2, 0.1]`.
- Maximin: `W_min(U) = min_i(U_i)`.

## 5. R=50m curriculum (rebuilt from scratch after the zero-training
   C64 bridge failed under the new observation)

| stage | steps | scenario bank | gate |
|---|---|---|---|
| `M6_R50_audited` | 0 -> 400,000 | `Q_00000` (single fixed scenario) | four-tier rule (STRONG/LEARNABLE_WITH_VARIANCE/INSUFFICIENT/FAIL) |
| `C4_R50` | 400,000 -> 600,000 (+100,000 ext. if SOFT_PASS+improving, stage name `C4_R50ext`) | `C4.json` (4 scenarios) | per-lineage: PASS>=0.90, SOFT 0.75-0.90+improving->+100K, FAIL<0.75->DR1-DR4; final decision via checkpoint-Q-ensemble gate |
| `C16_R50` | 700,000 -> 950,000 | `C16.json` (16 scenarios) | checkpoint-Q-ensemble gate, >=3/4 strict PASS |
| `C64_R50` | 950,000 -> 1,200,000 | `Q.json` (64 scenarios) | checkpoint-Q-ensemble gate, >=3/4 strict PASS |

Checkpoint-Q-ensemble gate (authoritative, used for C4/C16/C64/Mean/
formal): for stage ending at absolute step `S`, window
`K(S) = {S-150000, S-100000, S-50000, S}`, equal-weight arithmetic-mean
Q-ensemble across those 4 checkpoints, `epsilon=0` greedy action
selection, no training/replay/optimizer updates. PASS band
`completion>=0.90`; SOFT_PASS `0.75<=completion<0.90`; FAIL
`completion<0.75` (note: the gate script's `apply_original_gate()`
only checks the completion threshold, not collision/timeout directly,
though those are always logged alongside it).

## 6. Mean qualification (R=50m, lambda_W=0.5) -- QUALIFIED

- Seeds 900101/900102 (ordinal first two, not outcome-selected),
  resumed from own `C64_R50` `ckpt_step_1200000.pt`.
- Budget: 1,200,000 -> 2,000,000 (800,000 initial), one extension to
  2,200,000 only if clearly improving near the 800K mark (honest
  non-monotonic-oscillation-does-not-count standard).
- Gate: checkpoint-Q-ensemble at `K(2000000)`, frozen 2/2 strict-PASS
  rule for N=2 (NOT the 4-seed `>=3` rule -- `stage_q_ensemble_gate.py`'s
  own built-in CASE_A/CASE_B_OR_C label is miscalibrated for N=2 and
  must be manually corrected; the script's printed "NOT_QUALIFIED" for
  a 2/2 result is WRONG, ignore it).
- **Result: 900101=1.000/0.000, 900102=1.000/0.000. QUALIFIED, perfect
  2/2.** This closes the R=50m+lambda_W=0.5 validation loop.

## 7. Formal seed block (pre-outcome amendment, 2026-08-18)

Frozen BEFORE any formal Mean/GGI/Maximin result exists:

```
formal_seeds = [900101, 900102, 900103, 900104, 910101, 910102]
```

- **Same 6 seed IDs reused across all 3 conditions** (Mean, GGI,
  Maximin) -- a matched seed-block design.
- 900101-900104: reuse their own qualified `C64_R50`
  `ckpt_step_1200000.pt` as the SAME starting point for all 3
  conditions. Kept despite 900103's C64_R50 gate result being an
  outright FAIL (0.453 completion, 0.547 collision) -- do NOT drop it,
  do NOT select seeds by outcome.
- 910101/910102: brand-new seeds, never used for anything else in this
  project; require their own fresh `M6_R50_audited -> C4_R50 -> C16_R50
  -> C64_R50` build (identical procedure/budgets/gates to
  900101-900104's build) before they can serve as a formal starting
  point. Their curriculum build was IN PROGRESS at bundle-build time --
  see `checkpoints/curriculum_910101_910102/` and
  `MIGRATION_MANIFEST.json`'s `curriculum_in_progress` block for the
  exact step/checkpoint copied into this bundle.
- `lambda_W=0.5` fixed for all 3 conditions -- do not re-search, do not
  tune per condition.
- Formal training budget: identical to Mean qualification's own budget
  (800,000 steps from the C64 checkpoint, i.e.
  `1,200,000 -> 2,000,000`, same extension-if-clearly-improving rule),
  since RUNBOOK sec 49 says "identical frozen configuration except
  welfare objective."
- Keep ALL 18 formal runs in the final analysis regardless of outcome
  unless a documented genuine TECHNICAL failure occurs (crash, NaN,
  corrupted checkpoint) -- poor welfare/fairness performance is a
  scientific result, never grounds for exclusion or a rerun.

## 8. Held-out evaluation banks

`H0.json` (256 homogeneous scenarios), `H1.json` (256 heterogeneous
scenarios) -- already exist in `scenario_banks/`, do NOT regenerate.
SHA-256 hashes recorded in `scenario_banks/scenario_hashes.txt` and
`CHECKSUMS.sha256`.

## 9. What must NEVER change on the destination machine

Sensing range (R=50m), `lambda_W=0.5`, GGI weights `[0.4,0.3,0.2,0.1]`,
DQN architecture/hyperparameters (Section 3), reward/welfare formulas
(Section 4), the 6-seed formal block and its matched-block design,
episode length, action space, curriculum stage definitions and gate
rules, held-out banks. GPU execution and CPU process-parallelism
(Sections 3-4 of `README.md`) are the ONLY things this migration is
allowed to change, and only as a hardware execution detail.
