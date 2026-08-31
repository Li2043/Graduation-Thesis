# Conflict-Exposure Diagnostic

Diagnostic analysis only. No environment, reward, observation, training,
checkpoint, or scenario-bank change was made. No seed or episode was
filtered from the primary analysis. No causal claim is made.

## 1. Implementation audit

Source verified before writing any analysis code (not assumed from prose):

- **Vehicle roles**: `H1.json` gives each vehicle explicit `role`
  ("ramp"/"mainline") and `ttc_slot` ("front"/"rear") fields per scenario.
  Checked programmatically across all 256 H1 scenarios: every scenario has
  exactly 2 ramp + 2 mainline vehicles, and exactly one front + one rear
  `ttc_slot` per role. Empirically V0/V1 = ramp and V2/V3 = mainline in
  every H1 scenario, but role is read dynamically from scenario metadata
  throughout this diagnostic, never hardcoded by vehicle ID.
- **Front/rear wave identity**: directly encoded by `ttc_slot`, not
  invented. Front pair = (ramp vehicle with `ttc_slot=front`, mainline
  vehicle with `ttc_slot=front`); rear pair analogous. `nominal_ttc` in
  the scenario file confirms these are matched-TTC spawn pairs by design.
- **Interaction-region coordinates**, verified from
  `ThesisHighwayMergeEnvConfig` (`highwayenv_merge.py`), not assumed:
  `before_merge_length=220.0` -> convergence begins at **x=220**;
  `+ converge_merge_length=80.0` -> parallel merge begins at **x=300**;
  `+ parallel_merge_length=80.0` -> merge zone ends at **x=380**. The
  previously documented 220/300/380 m values are exactly correct.
  (For reference, `route_exit_x` = 220+80+80+90 = 470, a different,
  further boundary used elsewhere for episode completion -- not the
  interaction-region boundary used here.)
- **Decision frequency**: `policy_frequency=5` (dt=0.2s),
  `simulation_frequency=15` (3 physics substeps/policy step) -- unchanged
  from the Chapter 5 behavioural-cadence audit.
- **Continuous crossing-time estimation**: no pre-existing implementation
  was found anywhere in the codebase. `evaluate_behavioral_window.py` and
  `illustrative_episode.py` only record the discrete policy-step index of
  first crossing (resolution = dt = 0.2 s). This diagnostic adds a new,
  simple linear interpolation between the last pre-threshold and first
  post-threshold recorded step position to estimate a continuous crossing
  time. This is a NEW addition for this diagnostic, not a reuse of an
  existing "continuous" method (none existed).
- **Per-step state availability**: `world_xy(vehicle)` (x,y),
  `vehicle.speed`, `vehicle.action["acceleration"]`, and the discrete
  action returned by `select_ensemble_actions()` are all available per
  policy step -- same access pattern already used by
  `evaluate_behavioral_window.py`.
- **Utility/burden implementation**: `thesis.study_b.utility`'s
  `episode_utilities`, `episode_burdens`, `gini_coefficient`,
  `utility_range` -- read and reused completely unmodified. No new
  fairness metric was created for this diagnostic.
- **Hard-brake/BRAKE definitions**: reused exactly as frozen elsewhere in
  Chapter 5 -- BRAKE = discrete action index 2; hard-brake event =
  contiguous run of recorded policy-step accelerations at or below
  -3.0 m/s^2, counted once per contiguous run; acceleration-based braking
  burden `C_brake` = the same `DT * max(0, -accel) / 3` window
  accumulation already used for Table 5.7/5.10.
- **Vehicle physical length** (5.0 m, used only for the optional clearance
  column): stated directly in `highwayenv_vehicle.py`'s own comments
  (`Vehicle.LENGTH=5.0`), not guessed.
- **Formal Baseline evaluation procedure**, verified unchanged: 12 seeds
  (900101-104, 910101-102, 920101-106), `taskonly_arm` checkpoints,
  ensemble window K(2,000,000) = {1,850,000; 1,900,000; 1,950,000;
  2,000,000}, stage = `Formal_taskonly`, frozen H1 bank (256 scenarios),
  epsilon_eval = 0 -- identical to the Chapter 5 Baseline result.

**Data source decision**: the existing `behavioral_window` merged CSV
(Phase 2) stores per-vehicle window entry/exit steps and aggregated
window statistics, but not the full per-step (x,y) trajectory of all 4
vehicles simultaneously. That is insufficient for D3 (minimum physical
separation) and D4 (offline TTC-style diagnostic), so a new additive
evaluation script (`conflict_exposure_diagnostic_eval.py`) was written,
re-running exactly the 12 Baseline seeds x 256 H1 scenarios through the
identical checkpoint-Q ensemble, this time recording full per-step
position/speed/acceleration/action for all 4 vehicles.

## 2. Does the environment create conflict opportunities?

All numbers below are seed-level (12 seeds), 10,000-resample bootstrap,
percentile 95% CI, unless marked "pooled (descriptive)".

- **Cross-road merge-zone overlap**: 100% of episodes, in every one of
  the 12 seeds (seed-level mean = 1.0000, 95% CI [1.0000, 1.0000]).
  Every single Baseline H1 episode has at least one ramp vehicle and at
  least one mainline vehicle simultaneously inside [220, 380) m at some
  point.
- **Overlap duration**: cross-seed mean 6.88 s per episode (95% CI
  [6.69, 7.09]), pooled IQR [6.4, 7.2] s (out of ~14-18 s total time in
  the interaction region per vehicle).
- **max_simultaneous_merge_vehicles**: exactly 4 in 100% of episodes,
  every seed. All four controlled vehicles are inside the interaction
  region at the same policy step at some point in essentially every
  episode.
- **Minimum Ramp-Mainline crossing-time gap** (continuous, interpolated):
  - at x=380 (merge-zone end): median-of-seed-medians 0.236 s; proportion
    with gap <= 0.5 s: 80.4% (95% CI [62.3%, 94.2%]); <= 1.0 s: 99.8%
    (95% CI [99.5%, 100%]); <= 1.5 s: 100%. 36/3072 episodes have no
    valid pair (typically a collision before either vehicle reaches the
    boundary).
  - at x=300 (parallel-merge start): materially the same pattern (median
    0.225 s; <=0.5s: 82.4%; <=1.0s: 99.9%; <=1.5s: 100%).
  - These three thresholds are reported only as descriptive sensitivity
    checks; none was chosen after seeing which looked strongest -- all
    three are reported together and agree on the same conclusion.
- **Minimum cross-road physical separation**: pooled median 1.84 m
  (5%-95%: [0.02, 12.8] m); seed-level mean 3.76 m (95% CI [2.05, 5.68]),
  with substantial seed-to-seed spread. A non-trivial share of episodes
  show very small minimum separation (5th percentile = 0.02 m), i.e.
  near-miss configurations.

**Conclusion for this section**: the H1 held-out bank does create
essentially universal spatial overlap and very tight arrival-time
proximity between ramp and mainline vehicles. This is consistent with the
scenario generator's own "matched-TTC" spawn design (`nominal_ttc` field).

## 3. Do exposed episodes produce behavioural coordination?

Because cross-road overlap occurs in 100% of episodes in every seed, the
overlap-vs-no-overlap contrast specified in Sections G2/G3/H of the
protocol is **degenerate**: there are zero no-overlap episodes in any of
the 12 seeds, so no within-seed contrast or bootstrap CI can be computed
for that comparison (reported explicitly, not silently manufactured).

Descriptive pooled behavioural prevalence across all 3,072 episodes
(since a with/without-overlap split is not available):

| Measure | Proportion / mean |
|---|---:|
| P(any BRAKE action in window) | 55.1% |
| P(any hard-brake event in window) | 39.4% |
| P(any below-target mobility burden) | 28.7% |
| mean C_brake (acceleration-based) | 0.267 |
| mean C_mean (below-target burden) | 0.038 |

Seed-level P(any BRAKE action) ranges from 0.4% to 100% depending on the
seed -- very large seed-to-seed heterogeneity in how often the Baseline
policy actually brakes inside the interaction window, despite universal
spatial overlap.

Because the binary contrast is unavailable, Section G4's continuous
conflict-intensity measures are the substantive test for this diagnostic
(see Section 5).

## 4. Is inequality stronger during conflict exposure?

Not directly testable as an overlap-vs-no-overlap contrast, for the same
reason as Section 3 (zero no-overlap episodes in every seed). This
question is answered instead via conflict **intensity** (continuous),
reported in Section 5.

## 5. Conflict intensity and welfare inequality (exploratory)

Per-seed Spearman rho, all 12 seeds, no p-values:

| X (intensity) | Y | median rho | range across seeds |
|---|---|---:|---|
| overlap duration (s) | Utility Gini | +0.34 | [+0.01, +0.81] (11/11 defined seeds positive; 1 seed's Gini rho undefined) |
| overlap duration (s) | U_min | −0.34 | [−0.81, −0.02] (all 12 seeds negative) |
| overlap duration (s) | utility range | +0.34 | [+0.01, +0.81] |
| min crossing-time gap @x380 (s) | Utility Gini | −0.19 | [−0.54, +0.51] (mixed sign across seeds) |
| min crossing-time gap @x380 (s) | U_min | +0.19 | [−0.51, +0.53] (mixed sign) |
| min cross-road separation (m) | Utility Gini | −0.25 | [−0.77, +0.37] (mixed sign) |
| min cross-road separation (m) | U_min | +0.25 | [−0.36, +0.74] (mixed sign) |

**Overlap duration** is the one conflict-intensity variable with a fully
consistent direction across seeds: longer simultaneous ramp/mainline
occupancy of the interaction region is associated with higher Utility
Gini and lower U_min in essentially every seed, though the strength of
this association varies a great deal by seed (rho from near-zero to
~0.8). The crossing-time-gap and physical-separation variables do not
show a seed-consistent direction; with n=12 seeds this is reported as an
uncertain/mixed relationship for those two variables, not as evidence of
"no relationship."

## 6. Success-only sensitivity (SECONDARY)

Not computable as an overlap-vs-no-overlap contrast for the same
structural reason (zero no-overlap episodes, in both the full sample and
the 88.4% success-only subsample, in every seed). Reported explicitly
rather than working around it. The G4 intensity relationships above
already use the full (primary) sample; restricting to success-only would
not change the fact that no binary contrast is available.

## 7. Interpretation

The exposure/behaviour/welfare-sacrifice decomposition (Section I),
cross-seed mean proportions with 95% CI:

| Category | Proportion |
|---|---:|
| no overlap | 0.0% (structurally absent) |
| overlap, no BRAKE action | 44.9% (95% CI [25.7%, 65.1%]) |
| overlap + BRAKE, zero below-target burden | 36.1% (95% CI [19.0%, 54.9%]) |
| overlap + BRAKE, positive below-target burden | 19.0% (95% CI [7.0%, 35.8%]) |

This pattern is closest to a **mixture of Case 3 and Case 4** in the
protocol's interpretation menu:

- Physical/timing overlap is essentially universal (Case 1's exposure
  premise is fully met), and conflict *duration* is consistently
  associated with more inequality across seeds (partial Case 1 support).
- But actual behavioural adjustment is far from universal (BRAKE action
  in only 55% of episodes despite 100% spatial overlap), and even when
  braking does occur, the below-target burden metric registers zero in
  roughly two-thirds of those braking episodes (36.1% of all episodes) --
  confirming, again and independently, the already-known measurement
  scope limitation of `C_i` (Case 4): braking from an overspeed state
  back to at-or-above target speed records zero burden by that metric's
  own definition. This is not evidence of an absence of interaction.
- The crossing-time-gap and physical-separation intensity variables do
  not show a seed-consistent relationship with inequality, so the
  strongest, most consistent conflict-related driver identified here is
  simply *how long* ramp and mainline vehicles simultaneously occupy the
  interaction region, not how close in time or space they pass.

This is consistent with, not proof of, Case 5's motivation: conflict
exposure is not rare or sparse in this environment (ruling out Case 2 as
the primary story), so the RQ2 null result for terminal welfare shaping
is not well explained by "the environment rarely presents fairness-
relevant situations." The data are more consistent with the fairness-
relevant signal being diluted by the mixture of low-consequence overlaps
(45% no braking at all) and the terminal/sparse delivery of the welfare
bonus discussed in the earlier training-dynamics analysis, rather than by
an absence of conflict opportunities.
