"""Stage 11 pilot (E30) frozen-draft constants and guards.

Status: DESIGN/DRAFT -- not yet git-tagged/frozen. See
E30_stage11_dyad_merge_pilot/docs/ (hub) for the protocol writeup and
STRUCTURE.md's E30 section for the full scope-change rationale and the
Stage-9-to-Stage-11 welfare-formula mapping.

Scope (2026-08-06, user-directed pivot away from Stage 10's 6-vehicle
target): this is a NEW stage, not "Stage 10 v8" -- the terminal target
itself changes from 6 vehicles to 2 (no more 2->4->6 curriculum), no
scripted background vehicles are added (both vehicles remain learners,
sharing stage10_shared_dqn.SharedDQNLearner + explicit role/zone features
exactly as in Stage 10 v5-v7), and this pilot adds a stakeholder-welfare
measurement layer (stage11_welfare.py) ported from chap02.tex's Stage 9
theoretical framework -- measured only, NOT added to the training reward
(CONDITION="baseline", no PBRS this round).

Why 2 vehicles, not 6: Stage 10 v5/v6/v7 (6-vehicle terminal target)
plateaued at 3-13% completion across three reward-formula iterations, each
trading one failure mode (collision-heavy) for another (over-cautious/
passive collapse). Literature/GitHub research this session found reported
95%+ success rates on merge-type RL tasks come exclusively from
single-learner-plus-scripted-traffic settings; genuinely multi-agent (all
vehicles learning) merge tasks top out at 60-75% even in published work.
But this project's own Stage 9 (2 independently-learned persistent
controllers A/B + 2 scripted background vehicles, base reward only)
already demonstrated ~80% completion with TWO simultaneously-learning
agents -- exceeding that external 60-75% ceiling -- which points to the
6-vehicle jump itself (not multi-agent learning in general) as the harder
variable. Stage 10 v7's own trajectory logs corroborate this: at its
2-vehicle curriculum stage, well-performing seeds already reached ~76-78%
completion, close to Stage 9's number, while stalled seeds showed the same
over-cautious collapse -- contaminated by a collision-penalty curriculum
ramp keyed to global step count rather than to curriculum-stage progress.
Stage 11 removes that confound by making 2 vehicles the terminal target
outright (no curriculum stage to stall out of) and re-timing the
reward-magnitude ramp accordingly (see REWARD_CURRICULUM_RAMP_STEPS below).

Why no spawn-position staggering despite the identical-spawn-position
diagnosis: an audit of Stage 10 v7's spawn code found both vehicles are
generated at an EXACTLY identical route_position and speed every episode
(a single shared jitter draw applied to both roles) -- a plausible
contributor to the merging-zone collisions seen in v7's trajectory data (no
structural head start for either party, forcing the shared policy to
invent asymmetric yielding from scratch every episode). Staggering spawn
positions so one role structurally leads was considered and explicitly
REJECTED by the user: it would eliminate genuine conflict (mirrors
chap02.tex's own definition of an implicit merge convention -- "where road
geometry ... leaves only one safe order, the order that appears is a
physical constraint, not a selected convention"), destroying exactly the
thing Stage 10/11 needs to study. Spawn positions are therefore left
IDENTICAL across roles (unchanged from Stage 10); only the spawn ORIGIN
moves (spawn_route_lead 40 -> 0, giving more pre-merge reaction distance,
independent of the identical-position property).
"""

from __future__ import annotations

from thesis.agents.dqn_bootstrap import DQNTargetMode
from thesis.pilots.stage10_role_phase_subpolicy_config import (
    collision_penalty_at_step as _collision_penalty_at_step_generic,
    ttc_weight_at_step as _ttc_weight_at_step_generic,
)

PROTOCOL_TAG = "stage11-dyad-pilot-protocol-v1"  # planned tag, NOT yet applied -- see module docstring
STAGE = "stage11_dyad_merge_pilot"
PURPOSE = "dyad_merge_completion_baseline_with_welfare_detection"
ALGORITHM = "double_dqn"
CONDITION = "baseline"  # base reward only, no PBRS -- this pilot's whole point (see module docstring)
ARCHITECTURE_SHARED_PARAMETER = "shared_parameter"  # same architecture as Stage 10 v5-v7, unchanged
# Pilot v6 (2026-08-07): ramp and mainline get their OWN independent DQN
# (own weights, own replay buffer, own optimiser/target network) instead of
# one network shared across both roles. NOT the same as Stage 10 v1-v4's
# sub-DQN architecture (see stage11_dyad_merge_pilot_config.py's
# PILOT_V6_SEEDS docstring below for the precise distinction): that split on
# (role, zone) jointly (6 networks, needing cross-network bootstrap at zone
# boundaries since zone changes mid-episode); this splits on role ALONE (2
# networks, zone stays an input feature within each), and needs no
# cross-network bootstrap since role is fixed for a vehicle's whole episode.
ARCHITECTURE_ROLE_SPLIT = "role_split_parameter"

# Pilot v1 (Stage 11's own first round): 8 seeds, matching the Stage 10
# v4-v7 convention. Fresh block, disjoint from every prior stage's seeds
# (see FORBIDDEN_* below, including Stage 10's own 68001-68044).
PILOT_V1_SEEDS: tuple[int, ...] = (69001, 69002, 69003, 69004, 69005, 69006, 69007, 69008)
# Pilot v2 (round 2, 2026-08-06): per-step trajectory analysis of v1 found
# that TTC essentially never fired (fire rate ~0.001) while the curriculum-
# ramped COLLISION_PENALTY_MAGNITUDE, once locked at 5.0 for the training
# tail, drove BOTH vehicles to near-zero speed and permanent pre-merge-zone
# stalling -- a fixed, position-independent penalty makes total avoidance
# reward-dominant once perceived collision risk crosses a modest threshold,
# regardless of how slowly it was ramped in. v2 replaces the fixed/curriculum
# magnitude with a per-vehicle "clawback": the collision penalty exactly
# cancels whatever positive reward that vehicle has already earned this
# episode (see stage10_symmetric_merge_env.py's step() docstring,
# collision_penalty_override, and stage11_dyad_merge_runner.py for the
# bookkeeping) -- bounding the worst case at what was actually put at risk,
# not an arbitrary constant. TTC weight is set to 0.0 this round (confirmed
# inert in v1, dropped to isolate the clawback's own effect cleanly). New
# seeds, disjoint from v1 and every prior stage.
PILOT_V2_SEEDS: tuple[int, ...] = (69009, 69010, 69011, 69012, 69013, 69014, 69015, 69016)
# Pilot v3 (round 3, 2026-08-06): two fixes on top of v2, bundled together.
# (a) Scoping fix: v2's per-step trajectory data showed collision detection
# checking every pair of active vehicle ids regardless of completed status,
# so an already-exited vehicle (still cruising at constant speed) could
# register a LATER, physically real collision (e.g. a trailing vehicle
# catching up) and have its already-banked exit bonus erroneously erased by
# the clawback (confirmed empirically: one v2 episode where the clawback
# penalty hit ~1.0, the full progress+exit ceiling, on a vehicle that had
# already exited) -- and the same blanket episode-level "collided" boolean
# also incorrectly zeroed that vehicle's episode-level U_i welfare outcome.
# Fixed via stage11_welfare.clawback_tracking_contribution/
# episode_collided_for_vehicle (exit bonus permanently banked the moment
# it's earned; an already-completed vehicle's U_i is never zeroed by a
# collision that occurs after its own journey ended). Note: this bug never
# affected actual TRAINING -- the runner already skips storing a new replay
# transition for any vehicle once `was_completed`, so it only corrupted
# logged diagnostics/welfare metrics, not the learned policy itself.
# (b) Incentive fix: v2's per-step data showed several seeds settling into a
# stable "crawl slowly, avoid all risk, accept a partial-progress timeout"
# local optimum -- estimated payoff ~0.3-0.35 (most of route_exit's progress
# reward, no bonus, but also no clawback risk), which only loses to
# "attempt completion" once the perceived success probability exceeds
# roughly 35% under the original EXIT_REWARD_MAGNITUDE=0.6 (breakeven:
# p*(0.4+0.6) > 0.35). Raising the completion bonus to 1.8 lowers that
# breakeven to roughly p>16% (p*(0.4+1.8) > 0.35), giving a much stronger,
# earlier-triggering incentive to attempt completion rather than settle for
# safe partial credit. New seeds, disjoint from v1/v2 and every prior stage.
PILOT_V3_SEEDS: tuple[int, ...] = (69017, 69018, 69019, 69020, 69021, 69022, 69023, 69024)
EXIT_REWARD_MAGNITUDE_V3: float = 1.8
# Pilot v4 (round 4, 2026-08-07): full reward redesign, superseding v1-v3's
# curriculum/clawback/TTC experimentation entirely. Per-step audits of v1-v3
# repeatedly showed the SAME "total avoidance" collapse (near-zero speed,
# permanently stuck pre-merge) across three different collision-penalty
# mechanisms -- the user identified the actual root cause: this pilot never
# ported Stage 9's own already-proven (80% completion, HARDER 4-vehicle/2-
# scripted-background setup) base reward, which has two terms Stage 10/11's
# reward always lacked -- a hard-braking cost and a per-step active-time
# cost -- and additionally used a collision penalty (5.0, ~8.33:1 vs exit)
# far above Stage 9's own proven 1.0 (~1.67:1). v4 ports Stage 9's formula
# (progress/exit/hard-braking/time-cost terms and collision magnitude,
# verbatim) while KEEPING this pilot's own perpetrator-only collision
# attribution (already the env default, an improvement over Stage 9's
# blanket 4-stakeholder attribution -- see stage10_symmetric_merge_env.py's
# module docstring "Stage 9-based reward redesign" section for the full
# rationale). Deliberately FLAT, non-curriculum magnitudes throughout (no
# clawback, no TTC, no collision/TTC ramp) -- Stage 9's own reward needed
# none of these to reach 80%, and mixing them back in would re-conflate "is
# the reward right" with "does the curriculum help". The 2->4->6 vehicle-
# count curriculum remains deferred to a later round for the same reason.
# New seeds, disjoint from v1/v2/v3 and every prior stage.
PILOT_V4_SEEDS: tuple[int, ...] = (69025, 69026, 69027, 69028, 69029, 69030, 69031, 69032)
COLLISION_PENALTY_MAGNITUDE_V4: float = 1.0  # Stage 9's proven value, NOT v6/v7's 5.0
TTC_PENALTY_WEIGHT_V4: float = 0.0  # confirmed inert in v1; Stage 9 has no TTC analog
EXIT_REWARD_MAGNITUDE_V4: float = 0.6  # Stage 9's value, NOT v3's experimental 1.8
HARD_BRAKING_ETA_V4: float = 0.015  # Stage 9's eta_hard_brake, ported verbatim
TIME_COST_PER_STEP_V4: float = 0.0005  # Stage 9's active_time_cost_per_step, ported verbatim
# Pilot v5 (round 5, 2026-08-07): two targeted fixes on top of v4, built on
# a CORRECTED reading of v4's results -- a runner metrics bug
# (``EpisodeWindowStats`` double-counting episodes, fixed separately, see
# ``stage11_dyad_merge_runner.py``'s ``record_episode`` docstring) meant
# v4's TRUE average final completion was ~43.8%, not the ~21.9% first
# reported, and revealed 4/8 seeds reaching 82.5-95.2% true completion
# mid-training (already at/above Stage 9's 80%) before 3 of them declined.
# Per-step timing: the "total avoidance" failure (1/8 seeds, permanently
# zero from step ~30K) happens EARLY, while epsilon is still ~0.66 (ruling
# out insufficient exploration as the direct cause); the "increasingly
# reckless" failure (declining from a good peak, collision rate rising
# 2-3x) happens LATE, step ~70K onward. Two fixes, targeted at each:
#   (a) Collision penalty becomes a SHORT ramp (1.0 -> 2.0 over the first
#       ``COLLISION_PENALTY_RAMP_STEPS_V5`` steps, then held at 2.0) instead
#       of v4's flat 1.0. A flat higher value from step 0 risks WORSENING
#       the early total-avoidance failure (DQN004's "floor becomes ceiling"
#       lesson, already cited for v7); a short ramp keeps the early penalty
#       lenient (protects against early collapse) while reaching full
#       strength well before step 70K (targets the late-training drift
#       toward recklessness).
#   (b) Replay buffer capacity 20,000 -> 100,000. Verified from v4's own
#       checkpoint data (seed 69025: replay_size hits its 20,000 cap by
#       step ~20K and stays there) that the buffer only ever covers the
#       most recent ~13,000 env-steps (~13% of the 100K run) once full --
#       by the time the late-training drift happens (step 70K+), the
#       buffer has completely evicted the good step-40-80K "peak" period,
#       leaving nothing in the training mix to counteract the drift.
#       100,000 (~1.5 transitions/env-step observed) covers roughly the
#       most recent 65-70K steps, reaching back into the peak period so
#       that good experience stays in view during the decline window.
# New seeds, disjoint from v1/v2/v3/v4 and every prior stage.
PILOT_V5_SEEDS: tuple[int, ...] = (69033, 69034, 69035, 69036, 69037, 69038, 69039, 69040)
COLLISION_PENALTY_RAMP_START_V5: float = 1.0  # same starting point as v4's flat value
COLLISION_PENALTY_RAMP_END_V5: float = 2.0  # user-directed target, NOT v6/v7's 5.0
COLLISION_PENALTY_RAMP_STEPS_V5: int = 40_000  # reaches 2.0 well before the observed step-70K drift onset
REPLAY_CAPACITY_V5: int = 100_000
# Pilot v6 (round 6, 2026-08-07): architecture-only change on top of v5's
# best config (Stage 9-based reward, collision penalty ramp 1.0->2.0,
# 100,000-capacity replay buffer -- all unchanged). Per-step audits of v4/v5
# found role's first-layer weights grow far slower than zone/speed features
# (~2-2.7x by step 100K vs. zone_pre's ~6-8x), and direct Q-value probing
# showed flipping role NEVER changes the greedy action across 32 tested
# conflict states/checkpoints -- the network is not using the one feature
# that could break the ramp/mainline symmetry. `p_ramp_first` accordingly
# never converges away from ~0.5 across the whole 100K-step run, in every
# seed checked. Literature check (Terry et al. 2020 arXiv:2005.13625;
# HyperMARL, Kang et al. 2024 arXiv:2412.04233) converged on "cross-agent
# gradient interference" as the likely mechanism: concatenating a role
# indicator into a SHARED network's input lets the two roles' gradients
# fight over the same weights when the task doesn't otherwise force
# differentiation. Mashayekhi et al. (2022, "Prosocial Norm Emergence in
# Multi-agent Systems" -- local PDF in this hub's E28 folder) independently
# corroborates the fix direction: their Cha framework gets genuine,
# undesigned convention emergence (each of 1,000 independent simulation
# runs converges to ONE stable asymmetric norm, not a persistent coin flip)
# precisely because same-type agents share experience only WITHIN their own
# type's utility table -- never across types. v6 ports that same structural
# idea: ramp and mainline each get their OWN independent DQN (own weights,
# replay buffer, optimiser, target network) instead of one shared network.
#
# Explicitly NOT the same as Stage 10 v1-v4's sub-DQN architecture, which
# split on (role, zone) JOINTLY -- 6 independent networks, needing
# cross-network bootstrap support at zone boundaries because zone changes
# mid-episode. v6 splits on role ALONE -- 2 independent networks, zone stays
# an ordinary input feature inside each network -- and needs no
# cross-network bootstrap at all, since a vehicle's role never changes
# within an episode (bootstrapping always stays inside the same role's
# network for that vehicle's whole episode, same as the single-network
# case). This is also a much smaller data-fragmentation hit than the old
# 6-way split (~1/2 of transitions per network here, vs. ~1/6 there).
#
# No reward/curriculum/buffer changes this round -- this is deliberately an
# architecture-only test, isolated from v5's own two simultaneous changes,
# to see whether it resolves what magnitude tuning alone (v4->v5) did not:
# v5's own per-step audit found collision rate STILL climbing through the
# decline window even after the collision penalty had been fully ramped to
# 2.0 by step 40K, i.e. already ruling out "penalty not yet strong enough"
# as the explanation for that specific pattern.
PILOT_V6_SEEDS: tuple[int, ...] = (69041, 69042, 69043, 69044, 69045, 69046, 69047, 69048)
# Pilot v7 (round 7, 2026-08-07): builds on v6's role-split architecture
# (reward/curriculum/buffer/architecture all unchanged from v6) with ONE new
# mechanism -- periodic partial resets of each role's network, targeting the
# specific collapse trigger found by auditing v6's own per-role trajectory
# data. Per-step diagnosis: in 6/8 v6 seeds, one role's independent network
# permanently collapsed into near-zero-speed avoidance (~2 m/s, stuck
# pre/merging) while the other dominated (~25-29 m/s, mostly post) -- and in
# every collapsed case, the collapse follows within ~5,000 steps of a small
# (3-6 event) cluster of collisions specifically charged to that role's own
# network, occurring around step 14,000-20,000 while epsilon was still
# 0.78-0.84 (ruling out insufficient exploration). The collapse never
# recovered for the remaining ~80,000 steps of any affected seed.
#
# Literature match: Nikishin et al. (2022, "The Primacy Bias in Deep
# Reinforcement Learning", ICML, arXiv:2205.07802) -- deep RL agents overfit
# to early experience in a way that permanently damages the rest of
# training ("primacy bias"); their fix, verified on discrete-action (Atari
# 100k) domains, is to periodically re-initialise the LAST FEW LAYERS of the
# network while KEEPING the replay buffer intact (their own repository,
# evgenii-nikishin/rl_with_resets, confirms this exact recipe). v7 ports
# this: reset each role's last 2 Linear layers (keeping the first layer's
# raw-feature encoding untouched), rebuild the Adam optimiser at the
# CURRENT learning rate (not the initial one), and immediately hard-sync
# the target network so it isn't left stale relative to the reset online
# weights -- the replay buffer itself is never touched.
#
# Schedule: NOT copied from Nikishin et al.'s own hyperparameters (their
# reset_interval=100,000-200,000 was sized for a much larger total budget
# than our own 100,000-step run -- copying it verbatim would mean at most
# one reset, right at the very end, useless here). Instead sized off our
# OWN empirically-found collapse window: resets at steps 20,000 / 40,000 /
# 60,000 / 80,000 -- the first reset lands just after the observed
# ~14,000-20,000 collapse onset (giving an already-collapsed role a chance
# to recover with ~80,000 steps of budget still left), with three further
# resets to also catch any later-onset collapse (e.g. v6 seed 69047's
# mainline collapsed around step 48,000).
RESET_INTERVAL_V7: int = 20_000
PILOT_V7_SEEDS: tuple[int, ...] = (69049, 69050, 69051, 69052, 69053, 69054, 69055, 69056)
# Pilot v8 (round 8, 2026-08-07): builds on v6's role-split architecture
# (reward/curriculum/buffer/architecture all unchanged from v6) -- explicitly
# NOT v7's periodic reset (user-directed: v7 fixed the collapse rate (6/8 ->
# 1/8 seeds) but at the cost of continually wiping out whatever
# ramp/mainline convention had formed, so p_ramp_first never stabilised in
# any v7 seed -- in tension with studying convention emergence itself).
# Instead ports a DIFFERENT mechanism from Mashayekhi et al. 2022 (the "Cha"
# framework already used for v6's diagnosis): Cha's exploration rate decays
# as e^{-E*m}, where m is "the number of times the same [conflict] situation
# has arisen before" for THAT agent -- not a global interaction count. This
# means an agent that starts avoiding conflict situations automatically gets
# a SLOWER epsilon decay for exactly those situations (visit count stays
# low), giving it more chances to gather disconfirming evidence before
# committing to permanent avoidance -- addressing the same v6-diagnosed
# trigger (a small early collision cluster locking in avoidance while
# global-step epsilon keeps decaying regardless) WITHOUT ever discarding
# learned weights the way v7's resets do.
#
# Adaptation for this DQN setting (not a literal port of Cha's tabular
# e^{-E*m} formula): epsilon for each role is driven by that role's own
# CUMULATIVE count of steps spent in the "merging" zone (this env's actual
# conflict zone -- collisions concentrate almost entirely in
# (merging, merging) zone pairs, confirmed in the v5 per-step collision-
# signature audit) while not yet completed, using the SAME linear-decay
# shape as this file's own epsilon_at_step (for stylistic consistency with
# the rest of the schedule, rather than introducing an unrelated exponential
# form) -- just driven by this per-role visit count instead of global step
# count. VISIT_COUNT_DECAY_TARGET_V8 is calibrated from real v7 trajectory
# data (seed 69053): a healthy, actively-participating role accumulates
# roughly 15,000-18,500 cumulative merging-zone visits by step 80,000 (where
# the existing global schedule reaches its EPSILON_END floor) -- 16,000 is
# picked as a round number in that range, so a NORMALLY-behaving role
# decays on roughly the same overall timescale as before, while a role that
# starts avoiding the merging zone decays far more slowly (visit count
# barely grows), keeping its exploration elevated until it actually
# gathers more conflict-zone experience.
VISIT_COUNT_DECAY_TARGET_V8: int = 16_000
PILOT_V8_SEEDS: tuple[int, ...] = (69057, 69058, 69059, 69060, 69061, 69062, 69063, 69064)

# ---------------------------------------------------------------- v9 (2026-08-08)
# v8's direct recomputation of real per-role visit counts (not the epsilon
# formula, the actual trajectory data) proved the collapse bottleneck is NOT
# exploration rate: in the two most-collapsed v8 seeds, NEITHER role ever
# reached VISIT_COUNT_DECAY_TARGET_V8 for the whole 100,000-step run --
# epsilon stayed well above EPSILON_END the entire time -- yet both still
# collapsed permanently into the same degenerate zero-completion convention
# seen in v6. So two exploration-axis fixes (v7's step-based schedule
# untouched, v8's visit-count schedule) have now both been tried; the
# bottleneck must be in the TRAINING SIGNAL itself, not exploration.
#
# v9 instead applies HYSTERETIC Q-LEARNING to each role's independent DQN
# update: asymmetric weighting of the loss based on the sign of the
# TD-error, so a transition where the outcome was WORSE than the network's
# current Q-estimate (negative TD-error -- e.g. an unlucky collision, quite
# plausibly caused by the OTHER independent role's exploratory action, not a
# real property of this role's own policy at this state) gets DOWN-WEIGHTED
# relative to a transition that was better than expected (positive
# TD-error, full weight). This targets "relative overgeneralization" -- the
# named pathology in the independent-multi-agent-learner literature for
# exactly this failure: an agent permanently over-generalizing from a
# partner-caused negative surprise. Crucially: no transition is EXCLUDED
# from training (collisions are still learned from, danger-avoidance is not
# lost) and neither the replay buffer nor the exploration schedule is
# touched -- this is a pure training-signal-weighting change, orthogonal to
# both v7 and v8's axis.
#
# Literature (confidence: search-summary level for all four -- no PDF
# text-extraction tool was available in this environment to verify any of
# these firsthand against the original documents):
#   - Matignon, Laurent & Le Fort-Piat (2007), "Hysteretic Q-learning: an
#     algorithm for decentralized reinforcement learning in cooperative
#     multi-agent teams" (IEEE/RSJ IROS 2007). Original TABULAR
#     formulation: two learning rates alpha (positive TD-error) and beta
#     (negative TD-error), beta < alpha, explicitly to be "cautious about
#     penalizing actions that appear suboptimal due to a teammate's
#     exploratory/noisy behavior." Their own reported values: alpha=0.1,
#     beta=0.01 (ratio beta/alpha=0.1) -- used below as HYSTERESIS_RATIO_V9's
#     starting point, NOT as a validated-for-this-domain constant.
#   - Omidshafiei et al. (2017), "Deep Decentralized Multi-task Multi-Agent
#     Reinforcement Learning under Partial Observability" (ICML 2017).
#     Extends hysteretic Q-learning to deep RL (Dec-HDRQN): generalizes the
#     two-learning-rate mechanism to DQN/DRQN -- our own algorithm family --
#     plus a concurrent replay buffer (CERTs), which is NOT being ported
#     here; only the core hysteretic loss-weighting idea is used.
#   - Palmer, Tuyls, Bloembergen & Savani (2018), "Lenient Multi-Agent Deep
#     Reinforcement Learning" (AAMAS 2018, arXiv:1707.04402). Refines with
#     per-state-action DECAYING "leniency" (temperature-based, decaying with
#     visitation) instead of one fixed ratio -- noted as a possible future
#     refinement, explicitly NOT implemented in v9 (v9 uses ONE fixed ratio
#     for the whole run, no decay schedule -- that is the single new
#     variable this round, matching this project's "one variable at a time"
#     discipline used at every prior version).
#   - Sana, Piovesan, De Domenico, Ayed & Zhang (2026), "HPO: Hysteretic
#     Policy Optimization for Stable and Efficient Training under
#     Sparse-Reward Regime" (arXiv:2605.30201, May 2026). Independently
#     reinvents the identical core idea (asymmetric weighting, negative-
#     advantage updates down-weighted by a factor in [0,1], positive kept at
#     full weight) in a completely different modern context (single-agent
#     LLM RLVR fine-tuning via GRPO) -- cited ONLY as evidence the
#     underlying principle resurfaces independently across eras/domains, NOT
#     as a MARL or DQN precedent.
HYSTERESIS_RATIO_V9: float = 0.1
PILOT_V9_SEEDS: tuple[int, ...] = (69065, 69066, 69067, 69068, 69069, 69070, 69071, 69072)

# ---------------------------------------------------------------- v10 (2026-08-08)
# SAME mechanism as v9 (hysteretic Q-learning), DIFFERENT calibration. v9's
# own per-step diagnosis (this session's direct trajectory analysis, bucketed
# into 2000-step windows -- an empirical finding of THIS project, not a
# literature claim) found that hysteresis_ratio=0.1 creates a roughly
# 10,000-12,000-step LAG between a real, informative collision cluster and
# its full consolidation into the policy: seed 69071 had a genuine 5-event
# collision cluster in steps 66,000-68,000, completion stayed healthy
# (0.717) through step 70,000, collision events dropped to exactly zero from
# step 68,000 onward, yet the full behavioural collapse into permanent
# avoidance (completion -> 0, p_ramp_first locked to 1.0) only landed around
# steps 76,000-80,000. That lag consistently falls in the FINAL stretch of
# the 100,000-step budget, after epsilon (floors at step 80,000, per
# EPSILON_DECAY_STEPS=0.8*MAX_STEPS) and the learning rate (floors at step
# 100,000) have already mostly decayed -- leaving no remaining budget for
# the policy to actually stabilise once the correction finally lands.
#
# Hypothesis: raising the ratio toward normal DQN (1.0, i.e. no hysteretic
# protection at all) should shorten this consolidation lag, so a real
# negative pattern gets fully learned earlier in the budget -- while still
# damping isolated one-off bad luck meaningfully more than plain DQN would.
# HYSTERESIS_RATIO_V10=0.3 is a 3x increase from v9's 0.1 -- THIS PROJECT'S
# OWN calibration choice for this round (not taken from a specific paper's
# grid search the way v9's 0.1 was lifted directly from Matignon et al.
# 2007's own alpha=0.1/beta=0.01). It is the ONE new variable this round:
# same v6 role-split architecture, same v5 reward/collision-curriculum,
# same replay buffer, same step-based epsilon schedule, no periodic reset,
# no visit-count epsilon -- see stage11_dyad_merge_runner.py's
# REWARD_VERSION_STAGE11_V10_HYSTERETIC_RATIO_0_3 comment for the flag
# wiring (mutually exclusive with v9's flag -- these are two calibrations
# of the same single variable being A/B tested, not a combination).
HYSTERESIS_RATIO_V10: float = 0.3
PILOT_V10_SEEDS: tuple[int, ...] = (69073, 69074, 69075, 69076, 69077, 69078, 69079, 69080)

# ---------------------------------------------------------------- v11 (2026-08-08)
# Two rounds of fixed hysteretic-ratio tuning (v9=0.1, v10=0.3) both failed to
# produce a version that is both collapse-resistant AND settles into a stable
# final state within the 100,000-step budget (this project's own analysis,
# see versions/v9_.../README.md and versions/v10_.../README.md). v11 tries a
# mechanistically DIFFERENT fix rather than a third fixed ratio.
#
# Literature (confidence: read via arXiv HTML fetch this session, NOT the
# full PDF -- stated honestly): Patil et al. 2026, "Randomness is sometimes
# necessary for coordination" / "Diamond Attention" (arXiv:2605.06825).
# Core argument: under permutation-symmetric/repeating observations, a
# DETERMINISTIC policy structurally cannot break a tie -- facing a
# literally-identical recurring situation, it can only ever produce ONE
# fixed response, never "sometimes A goes first, sometimes B does." Their
# fix: a fresh random scalar per decision point, visible to BOTH agents (NOT
# private per-agent noise -- their own ablation shows private/unstructured
# noise gets 0% success on their symmetric XOR coordination game: "protocol-
# space structure, not stochastic noise, is the operative ingredient").
# Deterministic baselines plateau at exactly 0.5 (random-chance) on that
# game; their mechanism reaches 1.0.
#
# This project's own pre-v6 per-step collision-signature audit (this
# session's finding, not a literature claim) found "matching action pair"
# dominance at collision time -- both vehicles picking the SAME action at a
# recurring symmetric decision point (both "go" -> collision, or both "wait"
# -> mutual stall/truncation). Stage 11's vehicles are deliberately spawned
# at identical position/speed every episode (preserved, not a bug -- see
# this file's own docstring above), so within a given role's network the
# recurring geometric situation stays close to identical episode-to-episode,
# and a trained (low-epsilon) network can only produce one fixed response to
# it -- a plausible instance of exactly the pathology Patil et al. describe.
#
# Mechanism (Stage10SymmetricMergeEnv.include_priority_signal, adapted for
# this DQN setting -- NOT a literal port): one fresh random draw per active
# vehicle per EPISODE (not per-timestep the way Patil et al. do it -- Stage
# 11 is a single one-shot conflict resolution, not their continuous
# multi-step setting; redrawing every step would make an approaching
# vehicle's "priority" flicker mid-approach). Each vehicle's observation
# gets its OWN draw and its PEER's draw appended (both raw values -- the
# policy must learn any comparison itself). This is a coordination DEVICE
# the agents may learn to exploit, NOT a hardcoded yield rule: nothing in
# the environment or reward forces any particular use of the signal: it is
# just two more numbers in the observation vector, exactly like every other
# feature, and only the existing, UNCHANGED reward can make the network
# learn to condition on it.
#
# This round deliberately COMBINES the new signal with v9's
# enable_hysteretic_v9/HYSTERESIS_RATIO_V9 (0.1, not v10's 0.3 -- v10's
# slightly-worse results suggest v9's calibration is the better base) rather
# than testing the priority signal alone in isolation: v9 protects against
# training-dynamics lock-in from noisy negative TD-errors, while the
# priority signal targets the more fundamental "how do two agents break an
# otherwise-identical tie" problem -- these are complementary mechanisms,
# not alternate calibrations of the same thing (unlike v9 vs v10, which ARE
# mutually exclusive). This is a deliberate, discussed, LAST-ROUND choice
# to maximise the chance of a real result over clean attribution (if this
# combination works, credit cannot be cleanly split between the two
# mechanisms) -- documented here as a tradeoff, not an oversight.
#
# Only the FIRST 2-3 seeds of PILOT_V11_SEEDS (69081-69083) are being
# launched for an initial small-scale verification before deciding whether
# to run the full 8 -- a future reader of this file should not be surprised
# if some of these seeds have no corresponding trajectory/manifest data yet.
OBS_DIM_V11: int = 15  # Stage10SymmetricMergeEnv.OBS_DIM_WITH_ROLE_ZONE_AND_PRIORITY
PILOT_V11_SEEDS: tuple[int, ...] = (69081, 69082, 69083, 69084, 69085, 69086, 69087, 69088)

# ---------------------------------------------------------------- v12 (2026-08-09)
# Eleven rounds (v1-v11) all used TWO learners per run -- either one network
# shared across both roles (v1-v5) or two fully independent per-role
# networks (v6-v11) -- and none reliably resolved a non-stationarity-driven
# training-instability problem: because both learners update simultaneously,
# each treats the other's changing policy as part of a non-stationary
# environment (chap02.tex Sec. 2.1's own framing, quoted in
# stage11_dyad_merge_runner.py's REWARD_VERSION_STAGE11_V9_HYSTERETIC-through
# -V11 comments). Researched four MARL-specific fixes for exactly this
# problem this session (WoLF variable learning rate, multi-timescale/
# alternating updates, centralised-training-decentralised-execution,
# LOLA/LOQA opponent-learning-awareness) -- the user explicitly decided
# AGAINST continuing to patch the two-independent-learner premise (a faithful
# LOQA port was estimated at ~1.5-2x v11's own implementation size, with no
# guarantee of resolving the problem) and chose instead to abandon that
# premise entirely: ONE joint network sees both vehicles' concatenated
# observations and outputs two per-role action-value heads, eliminating
# non-stationarity by construction (only one set of parameters is ever being
# updated -- there is no "other learner" from this network's own
# perspective). See src/thesis/agents/joint_dqn.py's module docstring for the
# full architecture and the explicit trade-off this makes against Stage 9's
# own "two independent, persistent, identity-swappable controllers" theory-
# chapter premise (chap02.tex Sec. 2.1-2.2, the D_swap machinery) -- a
# deliberate, user-directed choice in favour of task success rate, not an
# oversight.
#
# Paired with this architecture change: v12 re-runs this project's own core
# research design -- baseline / mean_pbrs / min_pbrs, the three-condition
# PBRS welfare comparison chap02.tex's theory chapter defines (Condition C /
# D-mean / D-min) and chapter 4 already ran once on Stage 9's own, different,
# 4-stakeholder environment (where min_pbrs's improvement claim was tested
# and not detected, with a diagnosed causal-reach explanation: the worst-off
# stakeholder was overwhelmingly an uncontrollable scripted vehicle) -- on
# this project's Stage 11 2-vehicle dyad environment instead, where BOTH
# stakeholders are the learners' own controllable vehicles (no background
# vehicles at all), removing that specific causal-reach confound by
# construction. This is this project's OWN research design, not an external
# literature adaptation the way v8/v9's mechanisms were -- cited here as
# such, not manufactured against an outside source.
#
# baseline: Condition C, v5's reward structure (Stage 9-based: hard-braking
# cost + time cost + progress + exit + collision-penalty curriculum
# 1.0->2.0) with NO PBRS shaping -- corresponds to enable_stage9_based_reward_v5
# alone (no role-split, no hysteretic, no priority-signal: v12 is a from-
# scratch architecture, not layered on any of v6-v11's per-role mechanisms).
#
# mean_pbrs / min_pbrs: baseline reward + PBRS shaping using this project's
# own Stage-11-scoped 2-stakeholder potential functions
# (stage11_welfare.actual_potential/pbrs_shaping_signal, built on the
# already-N-agnostic mean_welfare/min_welfare -- NOT thesis.rewards.pbrs_v2,
# which hardcodes Stage 9's 4-stakeholder set and is left untouched).
# PBRS_LAMBDA_V12 reuses Stage 9's own already-validated EQUAL-COEFFICIENT
# design (lambda_mean = lambda_min = 0.2, chap04.tex's "Formative Result"
# section -- the correction that the two PBRS potentials are NOT RMS-matched
# in this project's actual implementation, an intentional choice because
# D-mean is specifically the control D-min needs to isolate the aggregation-
# rule effect from the mere presence of a dense stakeholder signal, chap02.tex
# Sec. 2.6) -- this is this project's own already-tested choice, not
# re-derived or tuned for Stage 11 specifically.
#
# Execution order (chap02.tex Sec. 2.9's own gate ordering, not extra
# caution invented here): baseline must first demonstrate the joint-network
# architecture can raise completion at all before mean_pbrs/min_pbrs's
# welfare comparison is interpretable -- see the original project's
# staged-rollout planning notes for the decision (baseline's 8 seeds
# launched and analysed first; mean_pbrs/
# min_pbrs's 16 seeds launched only after baseline is confirmed viable).
ARCHITECTURE_JOINT_V12 = "joint_centralised_v12"
PBRS_LAMBDA_V12: float = 0.2
PILOT_V12_BASELINE_SEEDS: tuple[int, ...] = (69089, 69090, 69091, 69092, 69093, 69094, 69095, 69096)
PILOT_V12_MEAN_PBRS_SEEDS: tuple[int, ...] = (69097, 69098, 69099, 69100, 69101, 69102, 69103, 69104)
PILOT_V12_MIN_PBRS_SEEDS: tuple[int, ...] = (69105, 69106, 69107, 69108, 69109, 69110, 69111, 69112)

# ------------------------------------------------------------- v12 PER + n-step
# Round 2 of v12 (2026-08-09), same joint-network architecture, same baseline
# reward -- adds two of Rainbow DQN's six components (Hessel et al. 2018,
# "Rainbow: Combining Improvements in Deep Reinforcement Learning", AAAI,
# arXiv:1710.02298 -- read directly from the paper's own LaTeX source this
# session, not a paraphrase). Chosen specifically because Rainbow's OWN
# ablation study found these two to be the most load-bearing of the six:
# removing either Prioritized Experience Replay or multi-step (n-step)
# returns caused a large, broadly consistent drop in median performance (the
# full agent beat either single-component-removed ablation in 53 of 57 Atari
# games); Dueling showed no significant aggregate difference when removed,
# and Double Q-learning's marginal value in that paper was entangled with
# their own distributional-return clipping, an Atari/domain-specific detail
# -- neither is added this round. Motivated by this project's own empirical
# "frozen_stall" finding in the completed PILOT_V12_BASELINE_SEEDS run
# (seed 69089, steps 60,000-100,000): one vehicle's speed collapses to near
# zero and never recovers for the rest of a 400-600-step episode, verified
# directly from real per-step trajectory data -- n-step's faster credit
# propagation is this project's hypothesis for why this might be preventable
# (a genuinely different mechanism from simply extending episode length,
# which real trajectory data already ruled out: the stuck vehicle's speed
# never recovers regardless of remaining time within an episode).
#
# Both flags are additive and default-off -- PILOT_V12_BASELINE_SEEDS's
# already-completed run remains exactly reproducible under the new code with
# both left at default (enable_prioritized_replay_v12=False, n_step_v12=1).
#
# PER hyperparameters below are Hessel et al. (2018)'s own published defaults
# (their Table 1) -- NOT re-tuned for this project's much smaller/shorter
# environment, per this project's "no hyperparameter search unless the pilot
# shows a clear need" discipline. Rainbow's own published multi-step n=3 is
# DELIBERATELY NOT reused as a constant here -- it was tuned for Atari's
# ~thousands-of-steps episodes, whereas this project's episodes run roughly
# 100-600 steps; n_step_v12 is instead a plain runner kwarg (default 1,
# unchanged behaviour) left for the caller/launch script to set per-run.
PER_ALPHA_V12: float = 0.5  # Hessel et al. 2018 Table 1: prioritization exponent omega
PER_BETA_START_V12: float = 0.4  # Hessel et al. 2018 Table 1: importance-sampling beta, start
PER_BETA_END_V12: float = 1.0  # Hessel et al. 2018 Table 1: importance-sampling beta, end (annealed linearly)
PER_EPSILON_V12: float = 1e-6  # small constant added to |TD-error| so no transition ever has zero priority
PILOT_V12_BASELINE_V2_SEEDS: tuple[int, ...] = (69113, 69114, 69115, 69116, 69117, 69118, 69119, 69120)

PILOT_SEEDS: tuple[int, ...] = (
    PILOT_V1_SEEDS
    + PILOT_V2_SEEDS
    + PILOT_V3_SEEDS
    + PILOT_V4_SEEDS
    + PILOT_V5_SEEDS
    + PILOT_V6_SEEDS
    + PILOT_V7_SEEDS
    + PILOT_V8_SEEDS
    + PILOT_V9_SEEDS
    + PILOT_V10_SEEDS
    + PILOT_V11_SEEDS
    + PILOT_V12_BASELINE_SEEDS
    + PILOT_V12_MEAN_PBRS_SEEDS
    + PILOT_V12_MIN_PBRS_SEEDS
    + PILOT_V12_BASELINE_V2_SEEDS
)
STAGE11_RESERVED_SEED_BLOCK: tuple[int, ...] = tuple(range(69001, 69121))

FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORBIDDEN_STAGE7A1_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
FORBIDDEN_STAGE7B_SEEDS: tuple[int, ...] = tuple(range(63001, 63021))
FORBIDDEN_STAGE7C_Q1_SEEDS: tuple[int, ...] = tuple(range(64001, 64021))
FORBIDDEN_STAGE8_SEEDS: tuple[int, ...] = tuple(range(65001, 65041))
FORBIDDEN_STAGE9_MEAN_PBRS_SEEDS: tuple[int, ...] = tuple(range(66001, 66021))
FORBIDDEN_STAGE9_MIN_PBRS_SEEDS: tuple[int, ...] = tuple(range(67001, 67021))
FORBIDDEN_STAGE10_SEEDS: tuple[int, ...] = tuple(range(68001, 68045))

MAX_STEPS: int = 100_000  # single-stage budget -- no vehicle-count curriculum this round
CHECKPOINT_STEPS: tuple[int, ...] = tuple(range(0, MAX_STEPS + 1, 10_000))  # 0,10K,...,100K

# Stage 11 v12 round 2 (PER + n-step, 2026-08-09): a SEPARATE, v12-scoped
# step budget -- NOT a mutation of MAX_STEPS above, which v1-v11's already-
# completed, frozen-protocol runs depend on via assert_stage11_pilot_guards's
# exact-match check. User's explicit requirement ("我希望步数尽量长，但是不要
# 太长导致之前实验出现的在一定timestep之后衰减"): as many steps as useful
# without reintroducing the "frozen_stall" primacy-bias tail-decay pattern
# (Stage 8's own diagnostic finding, re-confirmed directly in v12 baseline's
# own per-step trajectory data -- completion collapses to 0 once epsilon/LR
# have decayed near their floor and the network loses the plasticity to
# self-correct a locked-in bad convention). 400,000 steps mirrors Stage 9's
# own already-validated recipe scale (Double DQN + linear LR decay, 400K
# steps, 89.8-91.9% success on its own environment) -- not an arbitrary
# escalation. LEARNING_RATE_DECAY_STEPS_V12/EPSILON_DECAY_STEPS_V12 below
# scale PROPORTIONALLY with MAX_STEPS_V12 (same 100%/80% fractions as the
# v1-v11 schedule), so the low-plasticity tail is pushed out to the same
# relative point in a longer run, not left fixed in absolute step count.
MAX_STEPS_V12: int = 400_000
CHECKPOINT_STEPS_V12: tuple[int, ...] = tuple(range(0, MAX_STEPS_V12 + 1, 10_000))  # 0,10K,...,400K

# DQN hyperparameters and observation/action dims: reused UNCHANGED from
# Stage 10 v5-v7 (same architecture, same shared-DQN learner) -- only the
# TASK (vehicle count, spawn origin) and the reward-curriculum TIMING
# change this round, per protocol S2.3-style "no architecture/hyperparameter
# search unless the pilot shows a clear need."
HIDDEN_SIZES: tuple[int, ...] = (64, 64)
LEARNING_RATE_START: float = 0.0005
LEARNING_RATE_END: float = 0.0001
LEARNING_RATE_DECAY_STEPS: int = MAX_STEPS
GAMMA: float = 0.995
REPLAY_CAPACITY: int = 20_000
BATCH_SIZE: int = 64
REPLAY_WARMUP: int = 512
TARGET_SYNC_INTERVAL_UPDATES: int = 250
OBS_DIM: int = 13  # Stage10SymmetricMergeEnv.OBS_DIM_WITH_ROLE_ZONE (role + zone one-hot), unchanged
N_ACTIONS: int = 3

EPSILON_START: float = 1.0
EPSILON_END: float = 0.10
# 80% of the budget, not the ~20% Stage 10 v1/v2 originally used -- DQN004's
# own documented lesson (Alhazza 2026, cited in Stage 10 v7): decaying
# epsilon too early locks in a policy before a later-tightening reward
# signal (here, the collision/TTC curriculum below) has anything left to
# reshape.
EPSILON_DECAY_STEPS: int = int(0.8 * MAX_STEPS)

# Same 100%/80% fractions as LEARNING_RATE_DECAY_STEPS/EPSILON_DECAY_STEPS
# above, applied to MAX_STEPS_V12 instead of MAX_STEPS -- see MAX_STEPS_V12's
# docstring for why this proportional (not fixed-absolute) scaling is the
# load-bearing part of the 400K extension.
LEARNING_RATE_DECAY_STEPS_V12: int = MAX_STEPS_V12
EPSILON_DECAY_STEPS_V12: int = int(0.8 * MAX_STEPS_V12)

# Reward-magnitude curriculum: SAME shape (linear ramp, same start/end
# values) as Stage 10 v7's collision_penalty_at_step/ttc_weight_at_step, but
# a NEW, shorter ramp window sized for this single 2-vehicle stage instead
# of v7's "assume Stage1+2 safety valves sum to 100K steps" timing (which no
# longer applies -- there is no Stage 2/3 to size against). 40% of the
# budget is used for the ramp, leaving 60% at full strength for
# consolidation -- loosely mirroring Alhazza (2026) DQN004's Phase A+B /
# Phase C split (~56%/44%), not claimed identical.
REWARD_CURRICULUM_RAMP_STEPS: int = int(0.4 * MAX_STEPS)


def collision_penalty_at_step(step: int) -> float:
    """Stage 11's own ramp window wrapping stage10_role_phase_subpolicy_config's
    generic collision_penalty_at_step (same start/end values, shorter ramp_steps)."""
    return _collision_penalty_at_step_generic(step, ramp_steps=REWARD_CURRICULUM_RAMP_STEPS)


def collision_penalty_at_step_v5(step: int) -> float:
    """Stage 11 v5's own collision-penalty ramp: 1.0 -> 2.0 linearly over
    ``COLLISION_PENALTY_RAMP_STEPS_V5`` steps, then held at 2.0. Distinct
    from v1's ``collision_penalty_at_step`` (which ramps to the generic
    1.0->5.0 target) -- wraps the SAME generic
    ``stage10_role_phase_subpolicy_config.collision_penalty_at_step`` pure
    function, just with v5's own start/end/window."""
    return _collision_penalty_at_step_generic(
        step,
        start=COLLISION_PENALTY_RAMP_START_V5,
        end=COLLISION_PENALTY_RAMP_END_V5,
        ramp_steps=COLLISION_PENALTY_RAMP_STEPS_V5,
    )


def ttc_weight_at_step(step: int) -> float:
    """Stage 11's own ramp window wrapping stage10_role_phase_subpolicy_config's
    generic ttc_weight_at_step (same start/end values, shorter ramp_steps)."""
    return _ttc_weight_at_step_generic(step, ramp_steps=REWARD_CURRICULUM_RAMP_STEPS)


def epsilon_at_step(step: int) -> float:
    """Linear decay EPSILON_START -> EPSILON_END over EPSILON_DECAY_STEPS,
    then constant. No stage-transition bump needed (unlike Stage 10 v4-v7)
    -- there is only one stage, so no re-exploration reset is required."""
    if step >= EPSILON_DECAY_STEPS:
        return float(EPSILON_END)
    frac = float(step) / float(EPSILON_DECAY_STEPS)
    return float(EPSILON_START + frac * (EPSILON_END - EPSILON_START))


def epsilon_at_visit_count_v8(visit_count: int) -> float:
    """Stage 11 v8: same linear-decay shape as epsilon_at_step, but driven by
    a role's own cumulative merging-zone visit count instead of global step
    count -- see PILOT_V8_SEEDS docstring above for the full rationale
    (Mashayekhi et al. 2022's visit-count-based exploration decay, adapted
    for this DQN's continuous observations)."""
    if visit_count >= VISIT_COUNT_DECAY_TARGET_V8:
        return float(EPSILON_END)
    frac = float(visit_count) / float(VISIT_COUNT_DECAY_TARGET_V8)
    return float(EPSILON_START + frac * (EPSILON_END - EPSILON_START))


def lr_at_step(step: int) -> float:
    """Linear LR decay LEARNING_RATE_START -> LEARNING_RATE_END over
    LEARNING_RATE_DECAY_STEPS, then constant."""
    if step >= LEARNING_RATE_DECAY_STEPS:
        return float(LEARNING_RATE_END)
    frac = float(step) / float(LEARNING_RATE_DECAY_STEPS)
    return float(LEARNING_RATE_START + frac * (LEARNING_RATE_END - LEARNING_RATE_START))


def epsilon_at_step_v12(step: int, decay_steps: int = EPSILON_DECAY_STEPS_V12) -> float:
    """Stage 11 v12 round 2's own epsilon schedule: same shape/start/end as
    epsilon_at_step, but decayed over EPSILON_DECAY_STEPS_V12 (80% of the
    400K v12 budget, i.e. 320,000 steps) instead of the v1-v11 100K budget's
    80,000 -- so a v12 run reaches the EPSILON_END floor at the same
    RELATIVE point in training, not the same absolute step count. Exists as
    a separate function (rather than parameterising epsilon_at_step) so
    v1-v11's frozen call sites are byte-identical and unaffected.

    ``decay_steps`` defaults to the fixed 400K-budget constant (so every
    existing caller is unaffected) but can be overridden -- e.g. by the N=4
    extended-training run, which computes ``round(0.8 * max_steps)`` from
    its own, larger step ceiling so exploration keeps annealing across the
    whole run instead of flooring at EPSILON_END for everything past step
    320,000."""
    if step >= decay_steps:
        return float(EPSILON_END)
    frac = float(step) / float(decay_steps)
    return float(EPSILON_START + frac * (EPSILON_END - EPSILON_START))


def lr_at_step_v12(step: int, decay_steps: int = LEARNING_RATE_DECAY_STEPS_V12) -> float:
    """Stage 11 v12 round 2's own LR schedule: same shape/start/end as
    lr_at_step, but decayed over LEARNING_RATE_DECAY_STEPS_V12 (the full
    400K v12 budget) instead of the v1-v11 100K budget -- see
    epsilon_at_step_v12's docstring for the same proportional-scaling
    rationale and for what overriding ``decay_steps`` is for."""
    if step >= decay_steps:
        return float(LEARNING_RATE_END)
    frac = float(step) / float(decay_steps)
    return float(LEARNING_RATE_START + frac * (LEARNING_RATE_END - LEARNING_RATE_START))


def target_mode() -> DQNTargetMode:
    return DQNTargetMode.DOUBLE


def assert_stage11_pilot_guards(
    *, master_seed: int, max_steps: int, allowed_max_steps: int | None = None
) -> None:
    """Hard-fail on any deviation from the frozen-draft pilot scope.

    ``allowed_max_steps``: defaults to ``None``, which preserves the
    original behaviour exactly -- ``max_steps`` must equal ``MAX_STEPS``
    (100,000, v1-v11's frozen protocol). Callers running Stage 11 v12 round
    2's separate 400,000-step budget pass ``allowed_max_steps=MAX_STEPS_V12``
    explicitly to check against that value instead, WITHOUT touching what
    v1-v11 checks against -- v1-v11 call sites are unaffected by this
    parameter's existence as long as they omit it."""
    if master_seed not in PILOT_SEEDS:
        raise RuntimeError(f"master_seed {master_seed} not in frozen pilot seeds {PILOT_SEEDS}")
    for name, block in (
        ("formal", FORBIDDEN_FORMAL_SEEDS),
        ("stage7a1", FORBIDDEN_STAGE7A1_SEEDS),
        ("stage7b", FORBIDDEN_STAGE7B_SEEDS),
        ("stage7c_q1", FORBIDDEN_STAGE7C_Q1_SEEDS),
        ("stage8", FORBIDDEN_STAGE8_SEEDS),
        ("stage9_mean_pbrs", FORBIDDEN_STAGE9_MEAN_PBRS_SEEDS),
        ("stage9_min_pbrs", FORBIDDEN_STAGE9_MIN_PBRS_SEEDS),
        ("stage10", FORBIDDEN_STAGE10_SEEDS),
    ):
        if master_seed in block:
            raise RuntimeError(f"seed {master_seed} collides with {name} seed block")
    expected_max_steps = MAX_STEPS if allowed_max_steps is None else allowed_max_steps
    if int(max_steps) != expected_max_steps:
        raise RuntimeError(f"max_steps must be {expected_max_steps} for Stage 11, got {max_steps}")


__all__ = [
    "PROTOCOL_TAG",
    "STAGE",
    "PURPOSE",
    "ALGORITHM",
    "CONDITION",
    "ARCHITECTURE_SHARED_PARAMETER",
    "ARCHITECTURE_ROLE_SPLIT",
    "PILOT_V1_SEEDS",
    "PILOT_V2_SEEDS",
    "PILOT_V3_SEEDS",
    "EXIT_REWARD_MAGNITUDE_V3",
    "PILOT_V4_SEEDS",
    "COLLISION_PENALTY_MAGNITUDE_V4",
    "TTC_PENALTY_WEIGHT_V4",
    "EXIT_REWARD_MAGNITUDE_V4",
    "HARD_BRAKING_ETA_V4",
    "TIME_COST_PER_STEP_V4",
    "PILOT_V5_SEEDS",
    "COLLISION_PENALTY_RAMP_START_V5",
    "COLLISION_PENALTY_RAMP_END_V5",
    "COLLISION_PENALTY_RAMP_STEPS_V5",
    "REPLAY_CAPACITY_V5",
    "PILOT_V6_SEEDS",
    "RESET_INTERVAL_V7",
    "PILOT_V7_SEEDS",
    "VISIT_COUNT_DECAY_TARGET_V8",
    "PILOT_V8_SEEDS",
    "HYSTERESIS_RATIO_V9",
    "PILOT_V9_SEEDS",
    "HYSTERESIS_RATIO_V10",
    "PILOT_V10_SEEDS",
    "OBS_DIM_V11",
    "PILOT_V11_SEEDS",
    "ARCHITECTURE_JOINT_V12",
    "PBRS_LAMBDA_V12",
    "PILOT_V12_BASELINE_SEEDS",
    "PILOT_V12_MEAN_PBRS_SEEDS",
    "PILOT_V12_MIN_PBRS_SEEDS",
    "PER_ALPHA_V12",
    "PER_BETA_START_V12",
    "PER_BETA_END_V12",
    "PER_EPSILON_V12",
    "PILOT_V12_BASELINE_V2_SEEDS",
    "PILOT_SEEDS",
    "STAGE11_RESERVED_SEED_BLOCK",
    "FORBIDDEN_FORMAL_SEEDS",
    "FORBIDDEN_STAGE7A1_SEEDS",
    "FORBIDDEN_STAGE7B_SEEDS",
    "FORBIDDEN_STAGE7C_Q1_SEEDS",
    "FORBIDDEN_STAGE8_SEEDS",
    "FORBIDDEN_STAGE9_MEAN_PBRS_SEEDS",
    "FORBIDDEN_STAGE9_MIN_PBRS_SEEDS",
    "FORBIDDEN_STAGE10_SEEDS",
    "MAX_STEPS",
    "CHECKPOINT_STEPS",
    "MAX_STEPS_V12",
    "CHECKPOINT_STEPS_V12",
    "HIDDEN_SIZES",
    "LEARNING_RATE_START",
    "LEARNING_RATE_END",
    "LEARNING_RATE_DECAY_STEPS",
    "GAMMA",
    "REPLAY_CAPACITY",
    "BATCH_SIZE",
    "REPLAY_WARMUP",
    "TARGET_SYNC_INTERVAL_UPDATES",
    "OBS_DIM",
    "N_ACTIONS",
    "EPSILON_START",
    "EPSILON_END",
    "EPSILON_DECAY_STEPS",
    "LEARNING_RATE_DECAY_STEPS_V12",
    "EPSILON_DECAY_STEPS_V12",
    "REWARD_CURRICULUM_RAMP_STEPS",
    "collision_penalty_at_step",
    "collision_penalty_at_step_v5",
    "ttc_weight_at_step",
    "epsilon_at_step",
    "epsilon_at_step_v12",
    "epsilon_at_visit_count_v8",
    "lr_at_step",
    "lr_at_step_v12",
    "target_mode",
    "assert_stage11_pilot_guards",
]
