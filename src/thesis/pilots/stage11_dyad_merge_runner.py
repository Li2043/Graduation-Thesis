"""Stage 11 pilot (E30) single-seed training runner -- 2-vehicle dyad merge,
shared-parameter architecture, stakeholder-welfare detection (no PBRS).

Structurally a simplified cousin of Stage 10's
``stage10_shared_dqn_runner.py``: same ``SharedDQNLearner`` architecture,
same environment class (``Stage10SymmetricMergeEnv``, reused unmodified --
Stage 11 only changes its CONFIG: ``n_vehicles=2`` fixed for the whole run,
``spawn_route_lead=0.0``), same curriculum-ramped collision-penalty/TTC-weight
mechanism (reusing Stage 10 v7's ramp SHAPE via
``stage11_dyad_merge_pilot_config``'s wrappers, with a new, shorter ramp
window). What is genuinely new here:

    1. No vehicle-count curriculum at all -- ``n_vehicles`` is fixed at 2
       for the entire run, so none of Stage 10 v4-v7's threshold-triggered
       stage-advancement / epsilon-bump machinery is needed.
    2. Per-step stakeholder-welfare measurement (``stage11_welfare.py``,
       ported from ``chap02.tex``'s Stage 9 theoretical framework): target-
       speed attainment, the absorbing completed-status experience, and the
       mean/min welfare potentials are computed and logged every step --
       NOT added to the training reward (this pilot's ``CONDITION`` is
       ``"baseline"``, no PBRS).
    3. Episode-level mobility outcome U_i(tau) (collision -> 0, else the
       time-average of attainment over on-road steps) and a simple
       convention-style "who passed the conflict point first" bookkeeping
       (per role AND per physical vehicle identity, since Stage 11's shared
       network has no separate per-identity weights for a true controller-
       identity-swap test the way Stage 9's independent A/B networks do).

See ``STRUCTURE.md``'s E30 section for the full Stage-9-to-Stage-11 formula
mapping and the reasoning behind each design choice.

Clawback scoping fix (2026-08-06, same day as v2): collision detection in
``stage10_symmetric_merge_env.py`` checks every pair of currently-active
vehicle ids regardless of completed status, so a vehicle that has already
exited (and keeps cruising at constant speed) can still register a later,
physically real collision (e.g. a trailing vehicle catching up to it) in
the SAME episode. Found empirically in v2's own data: one episode where the
clawback penalty hit ~1.0 (the full progress+exit ceiling) on a vehicle
that had already exited and banked its exit bonus. A collision must only
ever cancel reward/welfare this vehicle's CURRENT, still-ongoing attempt
put at risk -- not an already-completed success belonging to a different
vehicle's later incident. Fixed via
``stage11_welfare.clawback_tracking_contribution``/``episode_collided_for_vehicle``
(exit bonus is permanently banked out of the clawback-tracking running
total the moment it is earned; an already-completed vehicle's episode-level
U_i is never zeroed by a collision that occurs after its own journey ended)
-- see those functions' docstrings for the full mechanism.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis.agents.independent_dqn_v2 import DQNConfig
from thesis.agents.joint_dqn import ROLE_ORDER, JointDQNConfig, JointDQNLearner, JointReplayTransition
from thesis.agents.replay_buffer_v2 import ReplayTransition
from thesis.agents.stage10_shared_dqn import SharedDQNLearner
from thesis.envs.stage10_symmetric_merge_env import (
    EXIT_REWARD_MAGNITUDE,
    REWARD_VERSION_STAGE9_BASED,
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage11_dyad_merge_pilot_config import (
    ALGORITHM,
    ARCHITECTURE_JOINT_V12,
    ARCHITECTURE_ROLE_SPLIT,
    ARCHITECTURE_SHARED_PARAMETER,
    BATCH_SIZE,
    CHECKPOINT_STEPS,
    COLLISION_PENALTY_MAGNITUDE_V4,
    CONDITION,
    EXIT_REWARD_MAGNITUDE_V4,
    GAMMA,
    HARD_BRAKING_ETA_V4,
    HIDDEN_SIZES,
    HYSTERESIS_RATIO_V9,
    HYSTERESIS_RATIO_V10,
    LEARNING_RATE_START,
    MAX_STEPS,
    MAX_STEPS_V12,
    CHECKPOINT_STEPS_V12,
    N_ACTIONS,
    OBS_DIM,
    OBS_DIM_V11,
    PBRS_LAMBDA_V12,
    PER_ALPHA_V12,
    PER_BETA_END_V12,
    PER_BETA_START_V12,
    PROTOCOL_TAG,
    REPLAY_CAPACITY,
    REPLAY_CAPACITY_V5,
    REPLAY_WARMUP,
    RESET_INTERVAL_V7,
    TARGET_SYNC_INTERVAL_UPDATES,
    TIME_COST_PER_STEP_V4,
    TTC_PENALTY_WEIGHT_V4,
    VISIT_COUNT_DECAY_TARGET_V8,
    assert_stage11_pilot_guards,
    collision_penalty_at_step,
    collision_penalty_at_step_v5,
    epsilon_at_step,
    epsilon_at_step_v12,
    epsilon_at_visit_count_v8,
    lr_at_step,
    lr_at_step_v12,
    target_mode,
    ttc_weight_at_step,
)
from thesis.pilots.stage11_welfare import (
    actual_potential,
    clawback_tracking_contribution,
    episode_collided_for_vehicle,
    episode_mobility_outcome,
    mean_welfare,
    min_welfare,
    pbrs_shaping_signal,
    stakeholder_experience,
    target_speed_attainment,
)

REWARD_VERSION_STAGE11_V1 = "dyad_baseline_no_pbrs_welfare_detection"
# Stage 11 v2 (2026-08-06): v1's per-step trajectory analysis found TTC
# essentially never fired (fire rate ~0.001) while the curriculum-ramped
# fixed collision penalty, once locked at 5.0, drove both vehicles to
# permanent near-zero-speed pre-merge-zone stalling -- total avoidance, not
# merely over-caution. v2 replaces the fixed/curriculum magnitude with a
# per-vehicle "clawback" (collision penalty exactly cancels whatever
# positive reward that vehicle has already earned this episode) and drops
# the now-confirmed-inert TTC term entirely, to isolate the clawback's own
# effect cleanly. See stage10_symmetric_merge_env.py's module docstring
# ("Collision-penalty clawback" section) and STRUCTURE.md's E30 section.
REWARD_VERSION_STAGE11_V2_CLAWBACK = "dyad_reward_clawback_collision_penalty_no_ttc"
# Stage 11 v3 (2026-08-06): two fixes bundled on top of v2 -- (a) a scoping
# bug fix (an already-completed vehicle's banked exit bonus / episode-level
# U_i could be erroneously erased by a LATER collision it was no longer
# meaningfully at risk from -- see stage11_welfare.py's
# clawback_tracking_contribution/episode_collided_for_vehicle), and (b) the
# completion bonus raised 0.6 -> 1.8 to make "attempt completion" reward-
# dominant over a "crawl slowly, accept partial-progress timeout" local
# optimum found in v2's per-step data at a much lower assumed success
# probability. See stage11_dyad_merge_pilot_config.py's PILOT_V3_SEEDS
# docstring and STRUCTURE.md's E30 section for the full reasoning.
REWARD_VERSION_STAGE11_V3_CLAWBACK_FIXED_HIGHER_EXIT = "dyad_reward_clawback_scoping_fixed_exit_1_8_no_ttc"
# Stage 11 v4 (2026-08-07): full reward redesign, superseding v1-v3's
# curriculum/clawback/TTC experimentation. Ports Stage 9's formally-locked
# hard-braking-cost and active-time-cost terms verbatim, keeps this pilot's
# own perpetrator-only collision attribution, and lowers the collision
# penalty back to Stage 9's proven 1.0 (from v6/v7's 5.0). Flat, non-
# curriculum magnitudes throughout -- no clawback, no TTC. See
# stage11_dyad_merge_pilot_config.py's PILOT_V4_SEEDS docstring and
# stage10_symmetric_merge_env.py's module docstring ("Stage 9-based reward
# redesign" section) for the full diagnosis/rationale.
REWARD_VERSION_STAGE11_V4_STAGE9_BASED = "dyad_" + REWARD_VERSION_STAGE9_BASED
# Stage 11 v5 (2026-08-07): two targeted fixes on top of v4, driven by a
# CORRECTED reading of v4's results (a runner metrics bug --
# EpisodeWindowStats double-counting episodes, see record_episode's
# docstring -- meant v4's true average final completion was ~43.8%, not the
# ~21.9% first reported; 4/8 seeds reached 82.5-95.2% true completion
# mid-training, already at/above Stage 9's 80%, before 3 of them declined).
# (a) Collision penalty is now a SHORT ramp 1.0 -> 2.0 (not v4's flat 1.0),
#     reaching 2.0 well before the observed step-70K "reckless drift" onset
#     while staying lenient through the step-30K window where v4's total-
#     avoidance collapse locked in (DQN004's "floor becomes ceiling" lesson
#     -- a flat higher value from step 0 risks worsening early collapse).
# (b) Replay buffer capacity 20,000 -> 100,000 (REPLAY_CAPACITY_V5) -- v4's
#     own checkpoint data showed the 20,000 buffer fills and FIFO-evicts by
#     step ~20K, covering only the most recent ~13% of the run once full;
#     by the time the late-training decline happens the buffer had entirely
#     forgotten the good step-40-80K "peak" period.
# See stage11_dyad_merge_pilot_config.py's PILOT_V5_SEEDS docstring for the
# full reasoning/numbers behind both changes.
REWARD_VERSION_STAGE11_V5_COLLISION_CURRICULUM_BIGGER_REPLAY = (
    "dyad_stage9_based_collision_curriculum_1to2_replay100k"
)
# Stage 11 v6 (2026-08-07): architecture-only change on top of v5's reward/
# curriculum/buffer settings (all unchanged) -- ramp and mainline each get
# their OWN independent DQN instead of one network shared across both
# roles. See stage11_dyad_merge_pilot_config.py's PILOT_V6_SEEDS docstring
# for the full diagnosis (role's Q-value sensitivity/weight-growth findings,
# literature grounding, and the precise distinction from Stage 10 v1-v4's
# (role, zone)-keyed sub-DQN).
REWARD_VERSION_STAGE11_V6_ROLE_SPLIT = "dyad_stage9_based_collision_curriculum_role_split_architecture"
# Stage 11 v7 (2026-08-07): builds on v6's role-split architecture (reward/
# curriculum/buffer/architecture all unchanged) with ONE new mechanism --
# periodic partial resets of each role's own network (Nikishin et al. 2022's
# primacy-bias fix), targeting the specific collapse trigger found auditing
# v6's per-role trajectory data: in 6/8 v6 seeds, one role's network
# permanently collapsed into near-zero-speed avoidance within ~5,000 steps
# of a small (3-6 event) collision cluster around step 14,000-20,000, while
# epsilon was still 0.78-0.84 (ruling out insufficient exploration), and
# never recovered. See ``reset_last_layers``'s docstring and
# ``stage11_dyad_merge_pilot_config.py``'s ``PILOT_V7_SEEDS`` docstring for
# the full mechanism and schedule.
REWARD_VERSION_STAGE11_V7_PERIODIC_RESET = (
    "dyad_stage9_based_collision_curriculum_role_split_periodic_reset"
)
# Stage 11 v8 (2026-08-07): builds on v6's role-split architecture --
# explicitly NOT v7's periodic reset (user-directed: v7 fixed the collapse
# rate but at the cost of never letting a ramp/mainline convention
# stabilise). Instead replaces the GLOBAL step-based epsilon schedule with a
# PER-ROLE schedule driven by that role's own cumulative merging-zone visit
# count -- adapted from Mashayekhi et al. 2022 ("Cha")'s exploration decay,
# which is a function of how many times that specific agent has encountered
# a conflict situation, not of global elapsed interactions. See
# ``stage11_dyad_merge_pilot_config.py``'s ``PILOT_V8_SEEDS`` docstring for
# the full mechanism and calibration.
REWARD_VERSION_STAGE11_V8_VISIT_COUNT_EPSILON = (
    "dyad_stage9_based_collision_curriculum_role_split_visit_count_epsilon"
)
# Stage 11 v9 (2026-08-08): builds on v6's role-split architecture --
# independent of (and not combined with, this round) v7's periodic reset or
# v8's visit-count epsilon. v8 proved the collapse bottleneck is NOT
# exploration rate (the two worst-collapsed v8 seeds never reached the
# visit-count decay target for either role -- epsilon stayed well above
# floor the entire run, yet still collapsed permanently). v9 instead targets
# the training SIGNAL itself: hysteretic Q-learning (Matignon, Laurent & Le
# Fort-Piat 2007; deep-RL extension Omidshafiei et al. 2017's Dec-HDRQN)
# down-weights negative-TD-error updates relative to positive ones, so an
# isolated early collision (plausibly caused by the OTHER independent role's
# exploratory action, not a real property of that state) cannot single-
# handedly drag a role's Q-values down to permanent avoidance the way a
# uniformly-weighted TD update can. See
# ``stage11_dyad_merge_pilot_config.py``'s ``HYSTERESIS_RATIO_V9`` and
# ``PILOT_V9_SEEDS`` docstrings for the full mechanism, literature, and
# confidence levels.
REWARD_VERSION_STAGE11_V9_HYSTERETIC = (
    "dyad_stage9_based_collision_curriculum_role_split_hysteretic"
)
# Stage 11 v10 (2026-08-08): SAME mechanism as v9 (hysteretic Q-learning),
# DIFFERENT calibration -- v9's own per-step diagnosis found that
# hysteresis_ratio=0.1 creates a ~10,000-12,000-step lag between a real,
# informative collision cluster and its full consolidation into the policy
# (seed 69071: collision cluster at step 66-68K, completion stayed healthy
# through step 70K, full collapse into permanent avoidance only landed
# around step 76-80K) -- and that lag consistently falls in the FINAL
# stretch of the 100K-step budget, after epsilon/LR have already floored,
# leaving no remaining steps to stabilise afterward. v10 raises the ratio
# to HYSTERESIS_RATIO_V10 (0.3, a 3x increase -- this project's own
# calibration choice, not taken from a specific paper the way v9's 0.1
# was) to shorten that consolidation lag while still damping isolated
# one-off bad luck more than plain DQN (ratio=1.0). Mutually exclusive
# with ``enable_hysteretic_v9`` -- these are two calibrations of the same
# single variable being A/B tested, not a combination. See
# ``stage11_dyad_merge_pilot_config.py``'s ``HYSTERESIS_RATIO_V10`` and
# ``PILOT_V10_SEEDS`` docstrings for the full rationale.
REWARD_VERSION_STAGE11_V10_HYSTERETIC_RATIO_0_3 = (
    "dyad_stage9_based_collision_curriculum_role_split_hysteretic_ratio_0p3"
)
# Stage 11 v11 (2026-08-08): a mechanistically DIFFERENT fix from v9/v10's
# fixed hysteretic ratios -- a per-episode "priority signal" (symmetry-
# breaking device, Patil et al. 2026 "Diamond Attention" /
# arXiv:2605.06825, adapted for this single-conflict DQN setting) combined
# with v9's hysteretic Q-learning (ratio=0.1). Each vehicle's observation
# gains its own and its peer's fresh per-episode random draw -- a
# coordination DEVICE the agents may learn to exploit, NOT a hardcoded
# yield rule (nothing forces any particular use of it; the unchanged
# reward is the only thing that can make the network condition on it).
# Targets this project's own pre-v6 "matching action pair" collision-
# signature finding: two vehicles picking the SAME action at a recurring
# symmetric decision point, because Stage 11's vehicles are deliberately
# spawned identically every episode and a deterministic, low-epsilon
# network can only produce one fixed response to a literally-repeating
# input. Deliberately combined with v9 (not tested alone) as a LAST-ROUND
# choice to maximise the chance of a real result over clean attribution --
# see stage11_dyad_merge_pilot_config.py's OBS_DIM_V11/PILOT_V11_SEEDS
# docstring for the full rationale and this tradeoff, stated explicitly.
REWARD_VERSION_STAGE11_V11_PRIORITY_SIGNAL = (
    "dyad_stage9_based_collision_curriculum_role_split_hysteretic_priority_signal"
)
# Stage 11 v12 (2026-08-09): abandons the two-independent-learner premise
# entirely (see stage11_dyad_merge_pilot_config.py's PILOT_V12_BASELINE_SEEDS
# docstring block and src/thesis/agents/joint_dqn.py's module docstring for
# the full architecture/rationale) in favour of ONE joint network with two
# per-role output heads, and re-runs this project's own core research design
# -- baseline / mean_pbrs / min_pbrs, the three-condition PBRS welfare
# comparison chap02.tex's theory chapter defines -- on this project's own
# Stage 11 2-vehicle environment. Three distinct tags, selected by
# ``pbrs_condition_v12``.
REWARD_VERSION_STAGE11_V12_JOINT_BASELINE = "dyad_v12_joint_network_baseline_no_pbrs"
REWARD_VERSION_STAGE11_V12_JOINT_MEAN_PBRS = "dyad_v12_joint_network_mean_pbrs"
REWARD_VERSION_STAGE11_V12_JOINT_MIN_PBRS = "dyad_v12_joint_network_min_pbrs"
REWARD_VERSION_STAGE11 = REWARD_VERSION_STAGE11_V1  # kept for backward-compat imports


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class EpisodeWindowStats:
    """Rolling counters since the previous checkpoint (protocol S5/S9 style)."""

    episodes: int = 0
    completions: int = 0
    collisions: int = 0
    truncations: int = 0
    # Welfare aggregates (mean over episodes in this window).
    mean_U_sum: float = 0.0
    min_U_sum: float = 0.0
    # Convention-style bookkeeping: among episodes with a recorded first
    # crossing (either vehicle reached the post-merge zone), how often was
    # it the ramp vs the mainline role, stratified also by physical vehicle
    # identity for the role-vs-identity cross-check (see module docstring).
    first_crosser_role_counts: dict[str, int] = field(default_factory=lambda: {"ramp": 0, "mainline": 0})
    first_crosser_identity_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_episode(self, *, welfare_by_stakeholder: dict[str, float], first_crosser: tuple[str, str] | None) -> None:
        """``welfare_by_stakeholder``: any dict whose VALUES are the
        episode's per-stakeholder U_i (only ``.values()`` is read, by
        ``mean_welfare``/``min_welfare`` -- both already N-agnostic). Keyed
        by role for the v1-v11 callers (always exactly one vehicle per
        role, so role-keyed and vehicle-id-keyed are equivalent there) and
        by vehicle id for the v12 N-vehicle caller, where role-keying would
        silently collide same-role peers -- was named ``U_by_role`` before
        this rename, kept neutral now that it's no longer always role-keyed."""
        # NOTE: does NOT increment self.episodes -- the caller (the training
        # loop's terminal-episode block) already does `window.episodes += 1`
        # right next to the completions/collisions/truncations counters. This
        # method used to also increment it, silently double-counting every
        # episode (episodes ended up exactly 2x the true count, verified via
        # completions+collisions+truncations != episodes on real Stage 11 v4
        # data -- found 2026-08-07 auditing v4's results). That inflated
        # denominator halved every reported rate (completion_rate,
        # collision_free_rate, truncation_rate, mean/min_U_mean) across the
        # ENTIRE Stage 11 pilot (v1-v4) -- true v4 average final completion
        # is ~43.8%, not the ~21.9% originally reported.
        self.mean_U_sum += mean_welfare(list(welfare_by_stakeholder.values()))
        self.min_U_sum += min_welfare(list(welfare_by_stakeholder.values()))
        if first_crosser is not None:
            role, vid = first_crosser
            self.first_crosser_role_counts[role] += 1
            self.first_crosser_identity_counts.setdefault(vid, {"ramp": 0, "mainline": 0})
            self.first_crosser_identity_counts[vid][role] += 1

    def as_dict(self) -> dict[str, Any]:
        n = max(self.episodes, 1)
        n_resolved = sum(self.first_crosser_role_counts.values())
        p_ramp_first = (self.first_crosser_role_counts["ramp"] / n_resolved) if n_resolved > 0 else None
        return {
            "episodes": self.episodes,
            "completion_rate": self.completions / n,
            "collision_free_rate": (self.episodes - self.collisions) / n,
            "truncation_rate": self.truncations / n,
            "mean_U_mean": self.mean_U_sum / n,
            "min_U_mean": self.min_U_sum / n,
            "n_resolved_conflicts": n_resolved,
            "p_ramp_first": p_ramp_first,
            "first_crosser_role_counts": dict(self.first_crosser_role_counts),
            "first_crosser_identity_counts": {k: dict(v) for k, v in self.first_crosser_identity_counts.items()},
        }

    def reset(self) -> None:
        self.episodes = 0
        self.completions = 0
        self.collisions = 0
        self.truncations = 0
        self.mean_U_sum = 0.0
        self.min_U_sum = 0.0
        self.first_crosser_role_counts = {"ramp": 0, "mainline": 0}
        self.first_crosser_identity_counts = {}


def build_shared_dqn_config(*, replay_capacity: int = REPLAY_CAPACITY, obs_dim: int = OBS_DIM) -> DQNConfig:
    """``replay_capacity`` defaults to Stage 11 v1-v4's shared 20,000 --
    Stage 11 v5 passes ``REPLAY_CAPACITY_V5`` (100,000) instead. See
    ``stage11_dyad_merge_pilot_config.py``'s ``PILOT_V5_SEEDS`` docstring for
    why: v4's own checkpoint data showed the 20,000-capacity buffer fills and
    starts FIFO-evicting by step ~20K, meaning it only ever covers the most
    recent ~13% of a 100K-step run -- by the time v4's late-training
    performance decline happens (step 70K+), the buffer had completely
    evicted the good step-40-80K "peak" period, leaving nothing in the
    training mix to counteract the drift.

    ``obs_dim`` defaults to ``OBS_DIM`` (13, role/zone features) -- Stage 11
    v11 passes ``OBS_DIM_V11`` (15) instead when the priority-signal feature
    is enabled. Every other caller is unaffected."""
    cfg = DQNConfig(
        obs_dim=obs_dim,
        n_actions=N_ACTIONS,
        hidden_sizes=HIDDEN_SIZES,
        learning_rate=LEARNING_RATE_START,
        gamma=GAMMA,
        epsilon=1.0,  # overwritten per-step by epsilon_at_step; placeholder only
        replay_capacity=replay_capacity,
        batch_size=BATCH_SIZE,
        device="cpu",
        reward_condition="baseline",
        target_mode=target_mode(),
    )
    cfg.validate()
    return cfg


def reset_last_layers(learner: SharedDQNLearner) -> None:
    """Stage 11 v7: Nikishin et al. (2022)'s primacy-bias fix -- re-initialise
    the last TWO Linear layers of the online network (``QNetwork.net`` is
    ``Sequential(Linear, ReLU, Linear, ReLU, Linear)`` for our 2-hidden-layer
    64x64 config -- this resets indices 2 and 4, the second hidden layer and
    the output layer, leaving index 0's raw-feature encoding untouched),
    rebuild the Adam optimiser at the CURRENT learning rate (not the
    original ``config.learning_rate`` -- the schedule has likely already
    decayed by the time a reset fires), and immediately hard-sync the target
    network so it is not left stale relative to the freshly-reset online
    weights. The replay buffer is deliberately NEVER touched -- this forgets
    what the network learned FROM the data, not the data itself. See
    ``stage11_dyad_merge_pilot_config.py``'s ``PILOT_V7_SEEDS`` docstring for
    the full rationale and schedule."""
    current_lr = float(learner.optimiser.param_groups[0]["lr"])
    for layer in (learner.online.net[2], learner.online.net[4]):
        layer.reset_parameters()
    learner.optimiser = torch.optim.Adam(learner.online.parameters(), lr=current_lr)
    learner.hard_sync_target()


def _resolve_v12_checkpoint_steps(checkpoint_steps: tuple[int, ...], max_steps: int) -> tuple[int, ...]:
    """CHECKPOINT_STEPS (0..100K) is the module-level default for
    ``run_stage11_pilot_training_job``'s ``checkpoint_steps`` parameter,
    sized for v1-v11's 100K budget; it would silently stop checkpointing a
    v12 run partway through, so whenever the caller left ``checkpoint_steps``
    at that shared default, derive a full 0..``max_steps`` range (every
    10,000 steps, ``CHECKPOINT_STEPS_V12``'s own spacing) instead of a fixed
    constant -- at ``max_steps=400_000`` (the original v12 budget) this
    produces exactly ``CHECKPOINT_STEPS_V12`` byte-for-byte; at a larger
    ``max_steps`` (e.g. the N=4 extended-training run) it now checkpoints
    all the way through instead of stopping at 400K. An explicit
    caller-supplied value (identity differs from the ``CHECKPOINT_STEPS``
    constant -- e.g. a short test run's own small tuple) is still honoured
    as-is."""
    if checkpoint_steps is not CHECKPOINT_STEPS:
        return checkpoint_steps
    return tuple(range(0, max_steps + 1, 10_000))


def _stage11_extended_stop_check(
    checkpoint_records: list[dict[str, Any]],
    *,
    converge_window: int = 5,
    converge_completion_mean: float = 0.95,
    converge_completion_std: float = 0.02,
    fail_window: int = 10,
    fail_completion_floor: float = 0.05,
    fail_collision_floor: float = 0.5,
) -> str | None:
    """Auto-stop check for the N=4 extended-training run, evaluated after
    every checkpoint save. Looks only at the WINDOW metrics of completed
    checkpoints (i.e. excludes the step-0 record, which has zero episodes
    and would otherwise read as a spurious 0.0 completion_rate) --
    ``checkpoint_records`` already accumulates one entry per
    ``save_checkpoint`` call in save-order, so the tail of the list is the
    most recent history.

    Returns ``"converged"`` if the last ``converge_window`` windows' mean
    completion_rate is >= ``converge_completion_mean`` AND its population
    std across those windows is <= ``converge_completion_std`` (both high
    AND stable, not a lucky single-window spike). Returns
    ``"failure_frozen_stall"`` if the last ``fail_window`` windows all have
    completion_rate <= ``fail_completion_floor`` (the project's known
    frozen_stall failure mode). Returns ``"failure_collision"`` if the last
    ``fail_window`` windows all have collision_free_rate < ``fail_collision_floor``.
    Returns ``None`` if none of these trigger yet (including simply not
    having enough windows of history yet) -- the caller keeps training up
    to its own max_steps ceiling in that case."""
    windows = [rec["window"] for rec in checkpoint_records if rec["window"]["episodes"] > 0]

    if len(windows) >= converge_window:
        tail = windows[-converge_window:]
        rates = [w["completion_rate"] for w in tail]
        mean_rate = statistics.mean(rates)
        std_rate = statistics.pstdev(rates)
        if mean_rate >= converge_completion_mean and std_rate <= converge_completion_std:
            return "converged"

    if len(windows) >= fail_window:
        tail = windows[-fail_window:]
        if all(w["completion_rate"] <= fail_completion_floor for w in tail):
            return "failure_frozen_stall"
        if all(w["collision_free_rate"] < fail_collision_floor for w in tail):
            return "failure_collision"

    return None


def _run_stage11_v12_joint_network_job(
    *,
    master_seed: int,
    output_root: Path,
    checkpoint_root: Path,
    max_steps: int,
    checkpoint_steps: tuple[int, ...],
    episode_max_steps: int | None,
    enable_trajectory_logging: bool,
    pbrs_condition_v12: str | None,
    reward_version: str,
    n_step_v12: int = 1,
    enable_prioritized_replay_v12: bool = False,
    target_speed_ramp: float | None = None,
    spawn_speed_ramp: float | None = None,
    n_vehicles: int = 2,
    converge_window: int = 5,
    converge_completion_mean: float = 0.95,
    converge_completion_std: float = 0.02,
    fail_window: int = 10,
    fail_completion_floor: float = 0.05,
    fail_collision_floor: float = 0.5,
) -> dict[str, Any]:
    """Stage 11 v12: separate, largely self-contained training loop for the
    joint (centralised) network -- kept apart from the v1-v11 per-vid loop
    above rather than threading a fourth architecture through it, since v12
    genuinely does not share that loop's per-role/per-learner structure
    (one joint transition per STEP, not one per vehicle; one learner, not a
    ``learner_map``). Shares only the surrounding scaffolding (env
    construction, git-commit lookup, checkpoint/manifest shape, trajectory
    logging, episode-outcome/welfare bookkeeping) with the main function,
    reusing the exact same helper functions/classes
    (``stage11_welfare.*``, ``EpisodeWindowStats``, ``_git_head``).

    ``n_vehicles`` (default 2, Study A byte-identical): total vehicle
    count, ``per_role = n_vehicles // 2``. N>2 is Study B (see
    ``STAGE11B_MULTI_VEHICLE_PILOT_DESIGN.md``) -- a pilot, not yet a
    frozen formal protocol the way Study A is.

    ``pbrs_condition_v12``: ``None``/``"baseline"`` -> Condition C, no
    shaping. ``"mean"``/``"min"`` -> add ``PBRS_LAMBDA_V12 * F`` (F computed
    via ``stage11_welfare.pbrs_shaping_signal`` from this project's own
    2-stakeholder ``actual_potential``-wrapped mean/min welfare) to BOTH
    vehicles' base reward every step -- see
    ``stage11_dyad_merge_pilot_config.py``'s v12 docstring block for the
    full rationale (Stage 9's own equal-coefficient design, lambda=0.2, and
    the staged baseline-first rollout).

    ``n_step_v12`` (default 1, byte-identical to pre-n-step behaviour) /
    ``enable_prioritized_replay_v12`` (default False, byte-identical to
    pre-PER behaviour): this round's two Rainbow-DQN-inspired additions
    (Hessel et al. 2018) -- see ``joint_dqn.py``'s module docstring for the
    full rationale, literature grounding, and this project's own
    adaptations (per-transition n_steps for episode-boundary correctness;
    max-of-two-heads TD-error combination for PER priorities)."""
    output_root = Path(output_root).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve()
    while repo_root != repo_root.parent and not (repo_root / ".git").exists():
        repo_root = repo_root.parent
    code_commit = _git_head(repo_root)

    if n_vehicles not in (2, 4, 6):
        raise ValueError(f"n_vehicles must be 2, 4, or 6, got {n_vehicles}")
    per_role = n_vehicles // 2

    env_config_kwargs: dict[str, Any] = {
        "seed": master_seed,
        "n_vehicles": n_vehicles,
        "spawn_route_lead": 0.0,
        "include_role_zone_features": True,
    }
    if episode_max_steps is not None:
        env_config_kwargs["max_steps"] = episode_max_steps
    if target_speed_ramp is not None:
        env_config_kwargs["target_speed_ramp"] = target_speed_ramp
    if spawn_speed_ramp is not None:
        env_config_kwargs["spawn_speed_ramp"] = spawn_speed_ramp
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(**env_config_kwargs))
    # Speed-asymmetry option (STAGE11_PROTOCOL.md Sec 12.4): role-keyed, not
    # vehicle-id-keyed, since role assignment is re-randomised every episode
    # (see `roles = dict(info["roles"])` below) while this is fixed for the
    # whole run -- computed once here, looked up per-vehicle-per-episode at
    # each `target_speed_attainment` call site via `target_speed_by_role[roles[vid]]`.
    target_speed_by_role = {
        "ramp": (
            float(env.config.target_speed_ramp)
            if env.config.target_speed_ramp is not None
            else float(env.config.target_speed)
        ),
        "mainline": float(env.config.target_speed),
    }

    joint_config = JointDQNConfig(
        per_vehicle_obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_sizes=HIDDEN_SIZES,
        learning_rate=LEARNING_RATE_START,
        gamma=GAMMA,
        epsilon=1.0,  # overwritten per-step by epsilon_at_step; placeholder only
        replay_capacity=REPLAY_CAPACITY_V5,
        batch_size=BATCH_SIZE,
        device="cpu",
        target_mode=target_mode(),
        prioritized_replay=enable_prioritized_replay_v12,
        per_alpha=PER_ALPHA_V12,
        per_beta_start=PER_BETA_START_V12,
        per_beta_end=PER_BETA_END_V12,
        n_vehicles=n_vehicles,
    )
    joint_learner = JointDQNLearner(joint_config, seed=master_seed)

    traj_file = None
    if enable_trajectory_logging:
        traj_dir = output_root / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_file = open(traj_dir / f"seed_{master_seed}.jsonl", "a", encoding="utf-8")

    # Epsilon/LR decay steps, scaled proportionally to THIS run's own
    # max_steps rather than the fixed 400K v12 budget -- same 80%/100%
    # fractions as EPSILON_DECAY_STEPS_V12/LEARNING_RATE_DECAY_STEPS_V12,
    # so at max_steps=400_000 these evaluate to exactly those constants
    # (320_000 / 400_000) and every existing 400K run is unaffected. At a
    # larger max_steps (the N=4 extended-training run) decay now spans the
    # whole run instead of flooring at the old absolute step counts.
    eps_decay_steps = max(1, round(0.8 * max_steps))
    lr_decay_steps = max(1, max_steps)

    window = EpisodeWindowStats()
    checkpoint_records: list[dict[str, Any]] = []
    checkpoint_targets = sorted(s for s in checkpoint_steps if 0 <= s <= max_steps)
    stop_reason = "max_steps_reached"

    def save_checkpoint(step: int) -> dict[str, Any]:
        payload = {
            "step": step,
            "master_seed": master_seed,
            "protocol_tag": PROTOCOL_TAG,
            "algorithm": ALGORITHM,
            "condition": CONDITION,
            "architecture": ARCHITECTURE_JOINT_V12,
            "reward_version": reward_version,
            "code_commit": code_commit,
            "learner": {
                "online": joint_learner.online.state_dict(),
                "target": joint_learner.target.state_dict(),
                "optimiser": joint_learner.optimiser.state_dict(),
                "update_count": joint_learner._update_count,
                "replay_size": len(joint_learner.replay),
            },
        }
        path = checkpoint_root / f"seed_{master_seed}" / f"ckpt_step_{step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

        metrics: dict[str, Any] = {
            "step": step,
            "window": window.as_dict(),
            "learner": {
                "replay_size": len(joint_learner.replay),
                "update_count": joint_learner._update_count,
            },
        }
        window.reset()
        checkpoint_records.append(metrics)
        return metrics

    obs_dict, info = env.reset(seed=master_seed)
    roles = dict(info["roles"])
    active_ids = tuple(roles.keys())
    step = 0
    if 0 in checkpoint_targets:
        save_checkpoint(0)

    on_road_attainment: dict[str, list[float]] = {vid: [] for vid in active_ids}
    first_crosser: tuple[str, str] | None = None
    already_completed_before_collision: dict[str, bool] = {vid: False for vid in active_ids}
    # N-step windowing (n_step_v12=1 default: this always holds at most one
    # pending entry, emitted+popped every single step -- see
    # _emit_n_step_transition's call sites below for the exact mechanism
    # and joint_dqn.JointReplayTransition's docstring for why n_steps must
    # be tracked PER transition rather than as one global constant).
    n_step_window: list[dict[str, Any]] = []

    def slot_vids(current_roles: dict[str, str]) -> tuple[str, ...]:
        """Role-major slot order (all ramp slots, then all mainline slots),
        each role's members sorted by vehicle id -- reuses
        stage10_symmetric_merge_env.py's own spawn-queue ordering rule, so
        slot (role, i) has a stable RULE-based meaning every episode even
        though the physical vehicle occupying it varies (roles are
        re-randomised every reset). At n_vehicles=2 this is exactly
        (ramp_vid, mainline_vid), the original role_vid_map order."""
        out: list[str] = []
        for role in ROLE_ORDER:
            members = sorted(vid for vid, r in current_roles.items() if r == role)
            if len(members) != per_role:
                raise ValueError(f"expected {per_role} vehicles with role={role!r}, got {members}")
            out.extend(members)
        return tuple(out)

    slots = slot_vids(roles)

    def joint_obs_from(obs_d: dict[str, np.ndarray], current_slots: tuple[str, ...]) -> np.ndarray:
        return np.concatenate([obs_d[vid] for vid in current_slots])

    def _emit_n_step_transition(
        anchor: dict[str, Any],
        rewards_slice: list[dict[str, Any]],
        *,
        n_steps: int,
        next_joint_obs: np.ndarray | None,
        next_action_masks: tuple[np.ndarray, ...] | None,
        controller_terminal: bool,
        terminated_flag: bool,
        truncated_flag: bool,
    ) -> None:
        """Emit ONE transition anchored at ``anchor`` (the oldest pending
        window entry), whose reward is the discounted sum of ``rewards_slice``
        (``anchor`` followed by however many later real steps are still
        available -- ``len(rewards_slice) == n_steps`` always) and whose
        bootstrap horizon is ``gamma ** n_steps``. ``rewards_slice`` shorter
        than the caller's configured n-step window happens only when an
        episode ends before the window could fill -- see the two call sites
        below (mid-episode full-window emit vs. episode-end flush)."""
        discounted = tuple(
            float(sum((GAMMA**k) * r["rewards"][slot] for k, r in enumerate(rewards_slice)))
            for slot in range(n_vehicles)
        )
        transition = JointReplayTransition(
            joint_observation=anchor["joint_observation"],
            actions=anchor["actions"],
            rewards=discounted,
            next_joint_observation=next_joint_obs,
            terminated=terminated_flag,
            truncated=truncated_flag,
            action_masks=anchor["action_masks"],
            next_action_masks=next_action_masks,
            controller_terminal=controller_terminal,
            n_steps=n_steps,
        )
        joint_learner.store_transition(transition)

    try:
        while step < max_steps:
            # v12-exclusive schedule functions (not the shared v1-v11
            # epsilon_at_step/lr_at_step), decayed over THIS run's own
            # eps_decay_steps/lr_decay_steps (== the 400K-budget constants
            # when max_steps=400_000, computed above) rather than a fixed
            # 400K assumption.
            lr = lr_at_step_v12(step, decay_steps=lr_decay_steps)
            joint_learner.set_learning_rate(lr)

            masks = {vid: env.action_mask(vid) for vid in active_ids}
            was_completed = {vid: env.is_completed(vid) for vid in active_ids}
            zone_t = {vid: env.zone_of(vid) for vid in active_ids}

            joint_obs_t = joint_obs_from(obs_dict, slots)
            eps = epsilon_at_step_v12(step, decay_steps=eps_decay_steps)
            masks_tuple = tuple(masks[vid] for vid in slots)
            actions_tuple = joint_learner.select_action(joint_obs_t, masks_tuple, epsilon=eps)
            actions = dict(zip(slots, actions_tuple))

            # PBRS: Phi at t, from PRE-step vehicle state. A pre-step state
            # is by construction never a true-terminal state (the loop would
            # not have continued into this iteration otherwise), so the
            # terminal-zero rule never applies here -- phi_t is always the
            # raw potential.
            phi_t = 0.0
            if pbrs_condition_v12 in ("mean", "min"):
                attainment_t = {
                    vid: target_speed_attainment(env._vehicles[vid].speed, target_speed_by_role[roles[vid]]) for vid in active_ids
                }
                experience_t = {
                    vid: stakeholder_experience(attainment_t[vid], completed=was_completed[vid])
                    for vid in active_ids
                }
                phi_t = (mean_welfare if pbrs_condition_v12 == "mean" else min_welfare)(
                    list(experience_t.values())
                )

            # Stage 11 v5's reward structure (Condition-C baseline for v12):
            # collision-penalty curriculum 1.0->2.0, hard-braking + time
            # cost, Stage 9's exit magnitude. No clawback, no TTC (confirmed
            # inert), no per-role overrides -- v12 has no role-split.
            collision_penalty_mag = collision_penalty_at_step_v5(step)

            next_obs_dict, reward, terminated, truncated, step_info = env.step(
                actions,
                collision_penalty_magnitude=collision_penalty_mag,
                ttc_penalty_weight=TTC_PENALTY_WEIGHT_V4,
                collision_penalty_override=None,
                exit_reward_magnitude=EXIT_REWARD_MAGNITUDE_V4,
                hard_braking_eta=HARD_BRAKING_ETA_V4,
                time_cost_per_step=TIME_COST_PER_STEP_V4,
            )
            step += 1
            zone_t1 = step_info["zone_t1"]

            if step_info["collision_event"]:
                for vid in active_ids:
                    if was_completed[vid]:
                        already_completed_before_collision[vid] = True

            # --- Stakeholder welfare (measured every step regardless of
            # condition, exactly as v1-v11 already did) ---
            attainment = {
                vid: target_speed_attainment(env._vehicles[vid].speed, target_speed_by_role[roles[vid]]) for vid in active_ids
            }
            completed_t1 = step_info["completed"]
            experience = {
                vid: stakeholder_experience(attainment[vid], completed=bool(completed_t1[vid]))
                for vid in active_ids
            }
            phi_mean = mean_welfare(list(experience.values()))
            phi_min = min_welfare(list(experience.values()))
            for vid in active_ids:
                if not was_completed[vid]:
                    on_road_attainment[vid].append(attainment[vid])

            reward_shaped = dict(reward)
            shaping_amount = 0.0
            if pbrs_condition_v12 in ("mean", "min"):
                raw_phi_t1 = phi_mean if pbrs_condition_v12 == "mean" else phi_min
                phi_t1 = actual_potential(raw_phi_t1, terminated=bool(terminated), truncated=bool(truncated))
                shaping_f = pbrs_shaping_signal(phi_t, phi_t1, gamma=GAMMA)
                shaping_amount = PBRS_LAMBDA_V12 * shaping_f
                for vid in active_ids:
                    reward_shaped[vid] = float(reward_shaped[vid]) + shaping_amount

            if first_crosser is None:
                for vid in active_ids:
                    if zone_t[vid] == "merging" and zone_t1[vid] == "post":
                        first_crosser = (roles[vid], vid)
                        break

            if traj_file is not None:
                traj_file.write(
                    json.dumps(
                        {
                            "step": step,
                            "seed": master_seed,
                            "architecture": ARCHITECTURE_JOINT_V12,
                            "pbrs_condition": pbrs_condition_v12,
                            "collision_penalty_magnitude": collision_penalty_mag,
                            "pbrs_shaping_amount": float(shaping_amount),
                            "welfare_mean": phi_mean,
                            "welfare_min": phi_min,
                            "term_reason": step_info["term_reason"],
                            "collision_event": bool(step_info["collision_event"]),
                            "vehicles": [
                                {
                                    "id": vid,
                                    "role": roles[vid],
                                    "zone": zone_t[vid],
                                    "position": float(env._vehicles[vid].route_position),
                                    "speed": float(env._vehicles[vid].speed),
                                    "action": int(actions[vid]),
                                    "attainment": float(attainment[vid]),
                                    "experience": float(experience[vid]),
                                    "base_reward": float(reward[vid]),
                                    "shaped_reward": float(reward_shaped[vid]),
                                    "collision_penalty_applied": bool(
                                        step_info["collision_penalty_applied"][vid]
                                    ),
                                    "collision_penalty_used": float(
                                        step_info["collision_penalty_used_per_vehicle"][vid]
                                    ),
                                    "hard_braking_cost": float(step_info["hard_braking_cost_used"][vid]),
                                    "time_cost_applied": bool(step_info["time_cost_applied"][vid]),
                                }
                                for vid in active_ids
                            ],
                        }
                    )
                    + "\n"
                )

            # --- N-step transition windowing (n_step_v12=1 default: exactly
            # one JOINT transition emitted+popped every single step, byte-
            # identical to the pre-n-step per-step-transition behaviour) ---
            episode_ended = bool(terminated or truncated)
            n_step_window.append(
                {
                    "joint_observation": joint_obs_t,
                    "actions": actions_tuple,
                    "rewards": tuple(float(reward_shaped[vid]) for vid in slots),
                    "action_masks": masks_tuple,
                }
            )
            if episode_ended:
                # Flush every entry still pending: each gets its own
                # SHORTENED n_steps (the real number of steps from that
                # entry's position to this true episode end) and a
                # correspondingly truncated discounted-reward sum, with NO
                # bootstrap term.
                while n_step_window:
                    anchor = n_step_window.pop(0)
                    rewards_slice = [anchor] + n_step_window
                    _emit_n_step_transition(
                        anchor,
                        rewards_slice,
                        n_steps=len(rewards_slice),
                        next_joint_obs=None,
                        next_action_masks=None,
                        controller_terminal=True,
                        terminated_flag=bool(terminated),
                        truncated_flag=bool(truncated),
                    )
            elif len(n_step_window) == n_step_v12:
                anchor = n_step_window.pop(0)
                rewards_slice = [anchor] + n_step_window
                _emit_n_step_transition(
                    anchor,
                    rewards_slice,
                    n_steps=n_step_v12,
                    next_joint_obs=joint_obs_from(next_obs_dict, slots),
                    next_action_masks=tuple(env.action_mask(vid) for vid in slots),
                    controller_terminal=False,
                    terminated_flag=False,
                    truncated_flag=False,
                )

            if len(joint_learner.replay) >= REPLAY_WARMUP:
                if enable_prioritized_replay_v12:
                    joint_learner.update(current_step=step, total_steps=max_steps)
                else:
                    joint_learner.update()
                if joint_learner._update_count % TARGET_SYNC_INTERVAL_UPDATES == 0:
                    joint_learner.hard_sync_target()

            obs_dict = next_obs_dict

            if terminated or truncated:
                episode_collided = step_info["term_reason"] == "collision"
                # Keyed by VEHICLE ID, not role -- at n_vehicles>2 two
                # same-role vehicles would otherwise silently overwrite each
                # other under a role-keyed dict (only the last-iterated
                # same-role vehicle's outcome would survive), corrupting
                # mean_welfare/min_welfare without raising any error. Safe
                # at n_vehicles=2 either way (exactly one vid per role) --
                # kept vid-keyed uniformly rather than branching on N.
                U_by_vid = {
                    vid: episode_mobility_outcome(
                        collided=episode_collided_for_vehicle(
                            episode_collided=episode_collided,
                            was_already_completed_before_collision=already_completed_before_collision[vid],
                        ),
                        attainment_samples=on_road_attainment[vid],
                    )
                    for vid in active_ids
                }
                window.episodes += 1
                success = step_info["term_reason"] == "success"
                if success:
                    window.completions += 1
                elif step_info["term_reason"] == "collision":
                    window.collisions += 1
                elif step_info["term_reason"] == "truncation":
                    window.truncations += 1
                window.record_episode(welfare_by_stakeholder=U_by_vid, first_crosser=first_crosser)

                obs_dict, info = env.reset(seed=master_seed * 1_000_003 + step)
                roles = dict(info["roles"])
                active_ids = tuple(roles.keys())
                slots = slot_vids(roles)
                on_road_attainment = {vid: [] for vid in active_ids}
                first_crosser = None
                already_completed_before_collision = {vid: False for vid in active_ids}
                n_step_window = []

            if step in checkpoint_targets:
                save_checkpoint(step)
                stop_check = _stage11_extended_stop_check(
                    checkpoint_records,
                    converge_window=converge_window,
                    converge_completion_mean=converge_completion_mean,
                    converge_completion_std=converge_completion_std,
                    fail_window=fail_window,
                    fail_completion_floor=fail_completion_floor,
                    fail_collision_floor=fail_collision_floor,
                )
                if stop_check is not None:
                    stop_reason = stop_check
                    break
    finally:
        if traj_file is not None:
            traj_file.close()

        if step in checkpoint_targets and (not checkpoint_records or checkpoint_records[-1]["step"] != step):
            save_checkpoint(step)

    manifest = {
        "stage": "stage11_dyad_merge_pilot",
        "protocol_tag": PROTOCOL_TAG,
        "seed": master_seed,
        "algorithm": ALGORITHM,
        "condition": CONDITION,
        "architecture": ARCHITECTURE_JOINT_V12,
        "reward_version": reward_version,
        "n_vehicles": n_vehicles,
        "final_step": step,
        "stop_reason": stop_reason,
        "code_commit": code_commit,
        "checkpoint_steps": checkpoint_targets,
        "checkpoints": checkpoint_records,
    }
    manifest_path = output_root / f"seed_{master_seed}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_stage11_pilot_training_job(
    *,
    master_seed: int,
    output_root: Path,
    checkpoint_root: Path,
    max_steps: int = MAX_STEPS,
    strict: bool = True,
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS,
    episode_max_steps: int | None = None,
    enable_trajectory_logging: bool = True,
    enable_clawback_collision_penalty: bool = False,
    exit_reward_magnitude: float | None = None,
    enable_stage9_based_reward: bool = False,
    enable_stage9_based_reward_v5: bool = False,
    enable_role_split_v6: bool = False,
    enable_periodic_reset_v7: bool = False,
    enable_visit_count_epsilon_v8: bool = False,
    enable_hysteretic_v9: bool = False,
    enable_hysteretic_v10: bool = False,
    enable_priority_signal_v11: bool = False,
    enable_joint_network_v12: bool = False,
    pbrs_condition_v12: str | None = None,
    n_step_v12: int = 1,
    enable_prioritized_replay_v12: bool = False,
    target_speed_ramp: float | None = None,
    spawn_speed_ramp: float | None = None,
    n_vehicles: int = 2,
    converge_window: int = 5,
    converge_completion_mean: float = 0.95,
    converge_completion_std: float = 0.02,
    fail_window: int = 10,
    fail_completion_floor: float = 0.05,
    fail_collision_floor: float = 0.5,
) -> dict[str, Any]:
    """Single-seed Stage 11 dyad-merge training job: 2 vehicles fixed for
    the whole run, shared-parameter DQN + explicit role/zone features
    (unchanged from Stage 10 v5-v7), and per-step stakeholder-welfare
    logging (measured only, never added to reward).

    ``enable_clawback_collision_penalty`` selects the reward mechanism:
    False (default, Stage 11 v1's original behaviour, unchanged) uses the
    curriculum-ramped fixed collision-penalty/TTC-weight magnitudes (same
    shape as Stage 10 v7, new shorter window). True (Stage 11 v2/v3) instead
    uses a per-vehicle "clawback" collision penalty -- exactly cancelling
    whatever positive reward that vehicle has already earned this episode
    (v3: with the already-completed-vehicle scoping fix, see
    ``stage11_welfare.clawback_tracking_contribution``) -- and disables TTC
    entirely (confirmed inert in v1's per-step data).

    ``exit_reward_magnitude`` (Stage 11 v3): optional override of the
    completion bonus, defaulting to ``None`` (module constant, 0.6 -- v1/v2
    behaviour unaffected). Independent of ``enable_clawback_collision_penalty``
    so either lever can be tested on its own.
    See ``REWARD_VERSION_STAGE11_V2_CLAWBACK``'s module-level comment and
    ``stage10_symmetric_merge_env.py``'s module docstring for the full
    diagnosis and mechanism.

    ``enable_stage9_based_reward`` (Stage 11 v4): supersedes v1-v3's reward
    mechanism entirely -- flat (non-curriculum) collision penalty at Stage
    9's proven 1.0 (``COLLISION_PENALTY_MAGNITUDE_V4``, not v6/v7's 5.0),
    TTC disabled, no clawback, plus the two NEW terms ported verbatim from
    Stage 9's locked base reward (hard-braking cost, active-time cost -- see
    ``HARD_BRAKING_ETA_V4``/``TIME_COST_PER_STEP_V4``). Mutually exclusive
    with ``enable_clawback_collision_penalty`` (raises ``ValueError`` if both
    are True). When ``exit_reward_magnitude`` is also left ``None``, defaults
    to Stage 9's own 0.6 (``EXIT_REWARD_MAGNITUDE_V4``), not v3's
    experimental 1.8. See ``REWARD_VERSION_STAGE11_V4_STAGE9_BASED``'s
    module-level comment for the full rationale.

    ``enable_stage9_based_reward_v5`` (Stage 11 v5): builds on v4's reward
    (same hard-braking cost, time cost, perpetrator-only collision
    attribution, exit magnitude) with two changes -- (a) the collision
    penalty becomes a short ramp 1.0 -> 2.0
    (``collision_penalty_at_step_v5``) instead of v4's flat 1.0, and (b) the
    replay buffer capacity becomes ``REPLAY_CAPACITY_V5`` (100,000) instead
    of the shared 20,000. Mutually exclusive with both
    ``enable_stage9_based_reward`` and ``enable_clawback_collision_penalty``
    (raises ``ValueError`` if either is also True) -- it is a strict
    superset/alternative of v4's flag, not composable with it. See
    ``REWARD_VERSION_STAGE11_V5_COLLISION_CURRICULUM_BIGGER_REPLAY``'s
    module-level comment and ``stage11_dyad_merge_pilot_config.py``'s
    ``PILOT_V5_SEEDS`` docstring for the full rationale.

    ``enable_role_split_v6`` (Stage 11 v6): architecture-only change, layered
    ON TOP OF v5's reward (requires ``enable_stage9_based_reward_v5=True``;
    raises ``ValueError`` otherwise) -- ramp and mainline each get their own
    independent ``SharedDQNLearner`` (own weights, replay buffer, optimiser,
    target network) instead of one network shared across both roles. See
    ``REWARD_VERSION_STAGE11_V6_ROLE_SPLIT``'s module-level comment and
    ``stage11_dyad_merge_pilot_config.py``'s ``PILOT_V6_SEEDS`` docstring for
    the full diagnosis and the precise distinction from Stage 10 v1-v4's
    (role, zone)-keyed sub-DQN architecture.

    ``enable_periodic_reset_v7`` (Stage 11 v7): layered ON TOP of v6's
    role-split architecture (requires ``enable_role_split_v6=True``; raises
    ``ValueError`` otherwise) -- every ``RESET_INTERVAL_V7`` steps, each
    role's network has its last two layers re-initialised (replay buffer
    untouched) via ``reset_last_layers``. See that function's docstring and
    ``REWARD_VERSION_STAGE11_V7_PERIODIC_RESET``'s module-level comment for
    the full rationale.

    ``enable_visit_count_epsilon_v8`` (Stage 11 v8): layered ON TOP of v6's
    role-split architecture (requires ``enable_role_split_v6=True``; raises
    ``ValueError`` otherwise) -- independent of (and combinable with, though
    not validated together) ``enable_periodic_reset_v7``. Replaces the
    global step-based epsilon with a per-role schedule driven by that role's
    own cumulative merging-zone visit count (``epsilon_at_visit_count_v8``).
    See ``REWARD_VERSION_STAGE11_V8_VISIT_COUNT_EPSILON``'s module-level
    comment and ``stage11_dyad_merge_pilot_config.py``'s ``PILOT_V8_SEEDS``
    docstring for the full rationale.

    ``enable_hysteretic_v9`` (Stage 11 v9): layered ON TOP of v6's
    role-split architecture (requires ``enable_role_split_v6=True``; raises
    ``ValueError`` otherwise) -- independent of both ``enable_periodic_reset_v7``
    and ``enable_visit_count_epsilon_v8`` (v8 proved the collapse bottleneck
    is not exploration rate). Applies hysteretic Q-learning to each role's
    DQN update: negative-TD-error transitions are down-weighted by
    ``HYSTERESIS_RATIO_V9`` relative to positive-TD-error ones, so an
    isolated early collision cannot single-handedly drag a role's Q-values
    into permanent avoidance. See ``REWARD_VERSION_STAGE11_V9_HYSTERETIC``'s
    module-level comment and ``stage11_dyad_merge_pilot_config.py``'s
    ``HYSTERESIS_RATIO_V9``/``PILOT_V9_SEEDS`` docstrings for the full
    mechanism and literature.

    ``enable_hysteretic_v10`` (Stage 11 v10): SAME mechanism as v9, a
    DIFFERENT calibration -- layered ON TOP of v6's role-split architecture
    (requires ``enable_role_split_v6=True``; raises ``ValueError``
    otherwise), mutually exclusive with ``enable_hysteretic_v9`` (raises
    ``ValueError`` if both are True -- these are two ratios of the same
    single variable being A/B tested, not a combination). Negative-TD-error
    transitions are down-weighted by ``HYSTERESIS_RATIO_V10`` (0.3, vs v9's
    0.1) instead. Motivated by v9's own per-step diagnosis: a
    ~10,000-12,000-step lag between a real collision cluster and its full
    consolidation into the policy, landing after epsilon/LR had already
    floored, leaving no budget to stabilise. See
    ``REWARD_VERSION_STAGE11_V10_HYSTERETIC_RATIO_0_3``'s module-level
    comment and ``stage11_dyad_merge_pilot_config.py``'s
    ``HYSTERESIS_RATIO_V10``/``PILOT_V10_SEEDS`` docstrings for the full
    rationale.

    ``enable_priority_signal_v11`` (Stage 11 v11): a mechanistically
    DIFFERENT mechanism from v9/v10 -- layered ON TOP of v6's role-split
    architecture (requires ``enable_role_split_v6=True``; raises
    ``ValueError`` otherwise), but INDEPENDENT of / combinable with
    ``enable_hysteretic_v9``/``enable_hysteretic_v10`` (NOT mutually
    exclusive the way v9/v10 are with each other -- this is a genuinely
    different mechanism, not an alternate calibration). Adds a per-episode
    "priority signal" observation feature (own + peer random draw, see
    ``stage10_symmetric_merge_env.Stage10MergeEnvConfig.include_priority_signal``)
    -- a coordination device the agents may learn to exploit, not a
    hardcoded yield rule. This round's actual launch combines it with
    ``enable_hysteretic_v9=True``. See
    ``REWARD_VERSION_STAGE11_V11_PRIORITY_SIGNAL``'s module-level comment
    and ``stage11_dyad_merge_pilot_config.py``'s
    ``OBS_DIM_V11``/``PILOT_V11_SEEDS`` docstrings for the full rationale.

    ``enable_joint_network_v12`` (Stage 11 v12): abandons every one of
    v6-v11's independent-per-role-learner mechanisms in favour of ONE joint
    network with two per-role action-value heads (``thesis.agents.joint_dqn``)
    -- MUTUALLY EXCLUSIVE with ``enable_role_split_v6``,
    ``enable_periodic_reset_v7``, ``enable_visit_count_epsilon_v8``,
    ``enable_hysteretic_v9``, ``enable_hysteretic_v10``,
    ``enable_priority_signal_v11`` (all of those are independent-learner-
    architecture mechanisms; raises ``ValueError`` if any is combined with
    v12). Still requires ``enable_stage9_based_reward_v5=True`` (v12 reuses
    v5's underlying reward structure -- hard-braking cost, time cost,
    collision-penalty curriculum -- as its Condition-C baseline, just as v6
    did). Dispatches to a separate, largely self-contained training-loop
    implementation (``_run_stage11_v12_joint_network_job``) rather than
    threading a fourth architecture through the v1-v11 per-vid loop above.

    ``pbrs_condition_v12`` (Stage 11 v12): one of ``"baseline"``, ``"mean"``,
    ``"min"``, or ``None`` -- selects which of this project's own three
    Condition C / D-mean / D-min PBRS conditions to run (re-running
    chap02.tex's theory-chapter design and chapter 4's Stage 9 comparison on
    this project's own Stage 11 environment). Only meaningful when
    ``enable_joint_network_v12=True`` (raises ``ValueError`` if set
    otherwise). ``None``/``"baseline"`` adds no PBRS shaping; ``"mean"``/
    ``"min"`` add ``PBRS_LAMBDA_V12 * F`` (Stage 9's own already-validated
    equal-coefficient lambda=0.2) to every stakeholder's base reward each
    step, using this project's own 2-stakeholder
    ``stage11_welfare.actual_potential``/``pbrs_shaping_signal`` (NOT
    ``thesis.rewards.pbrs_v2``, which hardcodes Stage 9's 4-stakeholder set
    and is left untouched). See ``stage11_dyad_merge_pilot_config.py``'s
    v12 docstring block and ``joint_dqn.py``'s module docstring for the full
    rationale and the staged (baseline-first) rollout this design follows.

    ``n_step_v12`` (default 1, byte-identical to pre-n-step behaviour) /
    ``enable_prioritized_replay_v12`` (default False, byte-identical to
    pre-PER behaviour): only meaningful when ``enable_joint_network_v12=True``
    (raises ``ValueError`` otherwise, and ``n_step_v12`` must be >= 1) --
    Rainbow DQN's (Hessel et al. 2018) two most load-bearing ablated
    components by that paper's own ablation study. See
    ``joint_dqn.py``'s module docstring and
    ``stage11_dyad_merge_pilot_config.py``'s ``PER_ALPHA_V12`` /
    ``PER_BETA_START_V12`` / ``PER_BETA_END_V12`` docstrings for the full
    rationale, literature grounding, and this project's own adaptations.

    ``converge_window``/``converge_completion_mean``/``converge_completion_std``/
    ``fail_window``/``fail_completion_floor``/``fail_collision_floor``: only
    meaningful when ``enable_joint_network_v12=True`` -- auto-stop
    thresholds for extended-budget runs (e.g. the N=4 pilot's
    3,000,000-step extension) that would otherwise need a human watching
    checkpoints to decide when a run has converged, is clearly failing, or
    has gone on long enough without either. Evaluated by
    ``_stage11_extended_stop_check`` after every checkpoint; see its
    docstring for the exact rule. Defaults reproduce a reasonable stopping
    rule but do nothing to change behaviour before it triggers or after
    ``max_steps`` is reached -- a run that neither converges nor fails
    simply trains to ``max_steps`` exactly as before these parameters
    existed."""
    if sum([enable_stage9_based_reward, enable_stage9_based_reward_v5, enable_clawback_collision_penalty]) > 1:
        raise ValueError(
            "enable_stage9_based_reward, enable_stage9_based_reward_v5, and "
            "enable_clawback_collision_penalty are mutually exclusive reward mechanisms"
        )
    if enable_role_split_v6 and not enable_stage9_based_reward_v5:
        raise ValueError(
            "enable_role_split_v6 requires enable_stage9_based_reward_v5=True "
            "(v6 is v5's reward on top of the role-split architecture, not a "
            "standalone combination)"
        )
    if enable_periodic_reset_v7 and not enable_role_split_v6:
        raise ValueError(
            "enable_periodic_reset_v7 requires enable_role_split_v6=True "
            "(v7 is periodic resets on top of v6's role-split architecture, "
            "not a standalone combination)"
        )
    if enable_visit_count_epsilon_v8 and not enable_role_split_v6:
        raise ValueError(
            "enable_visit_count_epsilon_v8 requires enable_role_split_v6=True "
            "(v8's epsilon schedule is inherently per-role, not a standalone "
            "combination)"
        )
    if enable_hysteretic_v9 and not enable_role_split_v6:
        raise ValueError(
            "enable_hysteretic_v9 requires enable_role_split_v6=True "
            "(v9's hysteretic loss weighting is applied per-role learner, not a "
            "standalone combination)"
        )
    if enable_hysteretic_v10 and not enable_role_split_v6:
        raise ValueError(
            "enable_hysteretic_v10 requires enable_role_split_v6=True "
            "(v10's hysteretic loss weighting is applied per-role learner, not a "
            "standalone combination)"
        )
    if enable_hysteretic_v9 and enable_hysteretic_v10:
        raise ValueError(
            "enable_hysteretic_v9 and enable_hysteretic_v10 are alternate "
            "calibrations of the same mechanism (ratio 0.1 vs 0.3) -- mutually "
            "exclusive, not combinable"
        )
    if enable_priority_signal_v11 and not enable_role_split_v6:
        raise ValueError(
            "enable_priority_signal_v11 requires enable_role_split_v6=True "
            "(v11's priority-signal observation feature is meaningful only "
            "for independently-learned per-role networks, not a standalone "
            "combination)"
        )
    if enable_joint_network_v12 and not enable_stage9_based_reward_v5:
        raise ValueError(
            "enable_joint_network_v12 requires enable_stage9_based_reward_v5=True "
            "(v12 reuses v5's reward structure as its Condition-C baseline, "
            "same as v6 required it)"
        )
    if enable_joint_network_v12 and any(
        [
            enable_role_split_v6,
            enable_periodic_reset_v7,
            enable_visit_count_epsilon_v8,
            enable_hysteretic_v9,
            enable_hysteretic_v10,
            enable_priority_signal_v11,
        ]
    ):
        raise ValueError(
            "enable_joint_network_v12 is mutually exclusive with v6-v11's "
            "independent-per-role-learner mechanisms (role-split, periodic "
            "reset, visit-count epsilon, hysteretic weighting, priority "
            "signal) -- v12 abandons that architecture entirely, it does not "
            "layer on top of it"
        )
    if pbrs_condition_v12 is not None and not enable_joint_network_v12:
        raise ValueError("pbrs_condition_v12 may only be set when enable_joint_network_v12=True")
    if pbrs_condition_v12 is not None and pbrs_condition_v12 not in ("baseline", "mean", "min"):
        raise ValueError(f"pbrs_condition_v12 must be one of 'baseline'/'mean'/'min'/None, got {pbrs_condition_v12!r}")
    if int(n_step_v12) < 1:
        raise ValueError(f"n_step_v12 must be >= 1, got {n_step_v12}")
    if n_step_v12 != 1 and not enable_joint_network_v12:
        raise ValueError("n_step_v12 may only be set (!= 1) when enable_joint_network_v12=True")
    if enable_prioritized_replay_v12 and not enable_joint_network_v12:
        raise ValueError("enable_prioritized_replay_v12 may only be set when enable_joint_network_v12=True")
    if target_speed_ramp is not None and not enable_joint_network_v12:
        raise ValueError("target_speed_ramp may only be set when enable_joint_network_v12=True")
    if spawn_speed_ramp is not None and not enable_joint_network_v12:
        raise ValueError("spawn_speed_ramp may only be set when enable_joint_network_v12=True")
    if n_vehicles != 2 and not enable_joint_network_v12:
        raise ValueError("n_vehicles may only be set (!= 2) when enable_joint_network_v12=True")
    if n_vehicles not in (2, 4, 6):
        raise ValueError(f"n_vehicles must be 2, 4, or 6, got {n_vehicles}")
    if strict:
        # v12's own separate 400K budget (MAX_STEPS_V12) is checked against
        # instead of v1-v11's frozen 100K MAX_STEPS when v12 is active --
        # v1-v11 callers never pass allowed_max_steps, so their guard
        # behaviour is exactly unchanged (see assert_stage11_pilot_guards's
        # docstring).
        assert_stage11_pilot_guards(
            master_seed=master_seed,
            max_steps=max_steps,
            allowed_max_steps=MAX_STEPS_V12 if enable_joint_network_v12 else None,
        )

    if enable_joint_network_v12:
        if pbrs_condition_v12 == "mean":
            reward_version_v12 = REWARD_VERSION_STAGE11_V12_JOINT_MEAN_PBRS
        elif pbrs_condition_v12 == "min":
            reward_version_v12 = REWARD_VERSION_STAGE11_V12_JOINT_MIN_PBRS
        else:
            reward_version_v12 = REWARD_VERSION_STAGE11_V12_JOINT_BASELINE
        return _run_stage11_v12_joint_network_job(
            master_seed=master_seed,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            max_steps=max_steps,
            checkpoint_steps=_resolve_v12_checkpoint_steps(checkpoint_steps, max_steps),
            episode_max_steps=episode_max_steps,
            enable_trajectory_logging=enable_trajectory_logging,
            pbrs_condition_v12=pbrs_condition_v12,
            reward_version=reward_version_v12,
            n_step_v12=int(n_step_v12),
            enable_prioritized_replay_v12=enable_prioritized_replay_v12,
            target_speed_ramp=target_speed_ramp,
            spawn_speed_ramp=spawn_speed_ramp,
            n_vehicles=n_vehicles,
            converge_window=converge_window,
            converge_completion_mean=converge_completion_mean,
            converge_completion_std=converge_completion_std,
            fail_window=fail_window,
            fail_completion_floor=fail_completion_floor,
            fail_collision_floor=fail_collision_floor,
        )

    if (enable_stage9_based_reward or enable_stage9_based_reward_v5) and exit_reward_magnitude is None:
        exit_mag_used = EXIT_REWARD_MAGNITUDE_V4
    else:
        exit_mag_used = EXIT_REWARD_MAGNITUDE if exit_reward_magnitude is None else float(exit_reward_magnitude)

    if enable_priority_signal_v11 and enable_hysteretic_v9:
        reward_version = REWARD_VERSION_STAGE11_V11_PRIORITY_SIGNAL
    elif enable_periodic_reset_v7 and enable_visit_count_epsilon_v8 and enable_hysteretic_v9:
        reward_version = (
            REWARD_VERSION_STAGE11_V7_PERIODIC_RESET
            + "_and_v8_visit_count_epsilon_and_v9_hysteretic"
        )
    elif enable_periodic_reset_v7 and enable_visit_count_epsilon_v8 and enable_hysteretic_v10:
        reward_version = (
            REWARD_VERSION_STAGE11_V7_PERIODIC_RESET
            + "_and_v8_visit_count_epsilon_and_v10_hysteretic_ratio_0p3"
        )
    elif enable_periodic_reset_v7 and enable_hysteretic_v9:
        reward_version = REWARD_VERSION_STAGE11_V7_PERIODIC_RESET + "_and_v9_hysteretic"
    elif enable_periodic_reset_v7 and enable_hysteretic_v10:
        reward_version = REWARD_VERSION_STAGE11_V7_PERIODIC_RESET + "_and_v10_hysteretic_ratio_0p3"
    elif enable_visit_count_epsilon_v8 and enable_hysteretic_v9:
        reward_version = REWARD_VERSION_STAGE11_V8_VISIT_COUNT_EPSILON + "_and_v9_hysteretic"
    elif enable_visit_count_epsilon_v8 and enable_hysteretic_v10:
        reward_version = REWARD_VERSION_STAGE11_V8_VISIT_COUNT_EPSILON + "_and_v10_hysteretic_ratio_0p3"
    elif enable_periodic_reset_v7 and enable_visit_count_epsilon_v8:
        reward_version = REWARD_VERSION_STAGE11_V7_PERIODIC_RESET + "_and_v8_visit_count_epsilon"
    elif enable_hysteretic_v9:
        reward_version = REWARD_VERSION_STAGE11_V9_HYSTERETIC
    elif enable_hysteretic_v10:
        reward_version = REWARD_VERSION_STAGE11_V10_HYSTERETIC_RATIO_0_3
    elif enable_priority_signal_v11:
        reward_version = REWARD_VERSION_STAGE11_V11_PRIORITY_SIGNAL
    elif enable_visit_count_epsilon_v8:
        reward_version = REWARD_VERSION_STAGE11_V8_VISIT_COUNT_EPSILON
    elif enable_periodic_reset_v7:
        reward_version = REWARD_VERSION_STAGE11_V7_PERIODIC_RESET
    elif enable_role_split_v6:
        reward_version = REWARD_VERSION_STAGE11_V6_ROLE_SPLIT
    elif enable_stage9_based_reward_v5:
        reward_version = REWARD_VERSION_STAGE11_V5_COLLISION_CURRICULUM_BIGGER_REPLAY
    elif enable_stage9_based_reward:
        reward_version = REWARD_VERSION_STAGE11_V4_STAGE9_BASED
    elif enable_clawback_collision_penalty and exit_reward_magnitude is not None:
        reward_version = REWARD_VERSION_STAGE11_V3_CLAWBACK_FIXED_HIGHER_EXIT
    elif enable_clawback_collision_penalty:
        reward_version = REWARD_VERSION_STAGE11_V2_CLAWBACK
    else:
        reward_version = REWARD_VERSION_STAGE11_V1

    output_root = Path(output_root).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve()
    while repo_root != repo_root.parent and not (repo_root / ".git").exists():
        repo_root = repo_root.parent
    code_commit = _git_head(repo_root)

    env_config_kwargs: dict[str, Any] = {
        "seed": master_seed,
        "n_vehicles": 2,
        "spawn_route_lead": 0.0,  # Stage 11: spawn from route_start, not 40m in -- see module docstring
        "include_role_zone_features": True,
    }
    if enable_priority_signal_v11:
        env_config_kwargs["include_priority_signal"] = True
    if episode_max_steps is not None:
        env_config_kwargs["max_steps"] = episode_max_steps
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(**env_config_kwargs))
    target_speed = float(env.config.target_speed)
    replay_capacity_used = REPLAY_CAPACITY_V5 if enable_stage9_based_reward_v5 else REPLAY_CAPACITY
    obs_dim_used = OBS_DIM_V11 if enable_priority_signal_v11 else OBS_DIM
    architecture_used = ARCHITECTURE_ROLE_SPLIT if enable_role_split_v6 else ARCHITECTURE_SHARED_PARAMETER
    # Stage 11 v6: ramp and mainline each get their OWN SharedDQNLearner
    # instance (own weights/replay/optimiser/target network) -- keyed by
    # role in learner_map. v1-v5 (enable_role_split_v6=False) keep a single
    # learner under the "shared" key, and learner_for() below always
    # resolves to it regardless of vid/role, so those call sites are
    # unaffected. Distinct seeds per role avoid identical initialisation.
    if enable_role_split_v6:
        learner_map: dict[str, SharedDQNLearner] = {
            "ramp": SharedDQNLearner(
                build_shared_dqn_config(replay_capacity=replay_capacity_used, obs_dim=obs_dim_used),
                seed=master_seed,
            ),
            "mainline": SharedDQNLearner(
                build_shared_dqn_config(replay_capacity=replay_capacity_used, obs_dim=obs_dim_used),
                seed=master_seed + 500_003,
            ),
        }
    else:
        learner_map = {
            "shared": SharedDQNLearner(
                build_shared_dqn_config(replay_capacity=replay_capacity_used, obs_dim=obs_dim_used),
                seed=master_seed,
            )
        }

    def learner_for(vid: str) -> SharedDQNLearner:
        return learner_map[roles[vid]] if enable_role_split_v6 else learner_map["shared"]

    traj_file = None
    if enable_trajectory_logging:
        traj_dir = output_root / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_file = open(traj_dir / f"seed_{master_seed}.jsonl", "a", encoding="utf-8")

    window = EpisodeWindowStats()
    checkpoint_records: list[dict[str, Any]] = []
    checkpoint_targets = sorted(s for s in checkpoint_steps if 0 <= s <= max_steps)

    def save_checkpoint(step: int) -> dict[str, Any]:
        # v1-v5 (single "shared" learner): payload["learner"] stays a flat
        # dict, byte-for-byte identical shape to before. v6 (role-split):
        # payload["learner"] becomes {"ramp": {...}, "mainline": {...}}, same
        # per-learner sub-dict shape either way.
        learner_state = {
            key: {
                "online": lnr.online.state_dict(),
                "target": lnr.target.state_dict(),
                "optimiser": lnr.optimiser.state_dict(),
                "update_count": lnr._update_count,
                "replay_size": len(lnr.replay),
            }
            for key, lnr in learner_map.items()
        }
        payload = {
            "step": step,
            "master_seed": master_seed,
            "protocol_tag": PROTOCOL_TAG,
            "algorithm": ALGORITHM,
            "condition": CONDITION,
            "architecture": architecture_used,
            "reward_version": reward_version,
            "code_commit": code_commit,
            "learner": learner_state if enable_role_split_v6 else learner_state["shared"],
        }
        path = checkpoint_root / f"seed_{master_seed}" / f"ckpt_step_{step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

        # metrics["learner"] stays a flat {"replay_size", "update_count"}
        # scalar-valued dict in EVERY case (existing v1-v5 consumers rely on
        # this exact shape) -- for v6, replay_size/update_count are the SUM
        # across both role learners; metrics["learner_by_role"] carries the
        # per-role breakdown additionally, only when split.
        metrics: dict[str, Any] = {
            "step": step,
            "window": window.as_dict(),
            "learner": {
                "replay_size": sum(len(lnr.replay) for lnr in learner_map.values()),
                "update_count": sum(lnr._update_count for lnr in learner_map.values()),
            },
        }
        if enable_role_split_v6:
            metrics["learner_by_role"] = {
                role: {"replay_size": len(lnr.replay), "update_count": lnr._update_count}
                for role, lnr in learner_map.items()
            }
        window.reset()
        checkpoint_records.append(metrics)
        return metrics

    obs_dict, info = env.reset(seed=master_seed)
    roles = dict(info["roles"])
    active_ids = tuple(roles.keys())
    step = 0
    if 0 in checkpoint_targets:
        save_checkpoint(0)

    # Per-episode welfare accumulators, reset at every env.reset().
    on_road_attainment: dict[str, list[float]] = {vid: [] for vid in active_ids}
    first_crosser: tuple[str, str] | None = None  # (role, vid) of whoever first reaches "post" zone
    # Stage 11 v2 only: running per-vehicle sum of positive reward earned so
    # far THIS episode (progress + exit; never includes a collision penalty,
    # since collision ends the episode the moment it happens) -- this is the
    # basis for the clawback collision penalty. Snapshotted BEFORE each
    # step's env.step() call, so it deliberately excludes that same step's
    # own progress delta (a small, defensible simplification -- see
    # module docstring).
    cumulative_reward: dict[str, float] = {vid: 0.0 for vid in active_ids}
    # Stage 11 v3 fix: per-vehicle "had this vehicle already exited before
    # any collision occurred this episode" -- see stage11_welfare.py's
    # clawback_tracking_contribution/episode_collided_for_vehicle docstrings
    # for the bug this guards against (an already-completed vehicle's
    # banked exit bonus / episode-level U_i getting erased by a LATER
    # collision it is no longer meaningfully at risk from).
    already_completed_before_collision: dict[str, bool] = {vid: False for vid in active_ids}
    # Stage 11 v8: cumulative count, per role, of steps spent in the
    # "merging" zone (this env's actual conflict zone) while not yet
    # completed -- persists across the WHOLE run, never reset at episode
    # boundaries (mirrors Cha's own "m = number of times this situation has
    # arisen before", a lifetime count, not a per-episode one). Only
    # meaningful when enable_visit_count_epsilon_v8 is True; otherwise
    # unused.
    role_visit_counts: dict[str, int] = {"ramp": 0, "mainline": 0}

    try:
        while step < max_steps:
            lr = lr_at_step(step)
            for lnr in learner_map.values():
                lnr.set_learning_rate(lr)

            masks = {vid: env.action_mask(vid) for vid in active_ids}
            was_completed = {vid: env.is_completed(vid) for vid in active_ids}
            zone_t = {vid: env.zone_of(vid) for vid in active_ids}

            if enable_visit_count_epsilon_v8:
                for vid in active_ids:
                    if not was_completed[vid] and zone_t[vid] == "merging":
                        role_visit_counts[roles[vid]] += 1
                eps_by_vid = {
                    vid: epsilon_at_visit_count_v8(role_visit_counts[roles[vid]]) for vid in active_ids
                }
            else:
                eps_by_vid = {vid: epsilon_at_step(step) for vid in active_ids}

            actions = {
                vid: learner_for(vid).select_action(obs_dict[vid], masks[vid], epsilon=eps_by_vid[vid])
                for vid in active_ids
            }

            if enable_stage9_based_reward_v5:
                collision_penalty_mag = collision_penalty_at_step_v5(step)  # ramped 1.0 -> 2.0
                ttc_weight = TTC_PENALTY_WEIGHT_V4  # 0.0 -- still off, same as v4
                collision_penalty_override = None  # no clawback this version
                hard_braking_eta = HARD_BRAKING_ETA_V4
                time_cost_per_step = TIME_COST_PER_STEP_V4
            elif enable_stage9_based_reward:
                collision_penalty_mag = COLLISION_PENALTY_MAGNITUDE_V4  # flat, Stage 9's proven 1.0
                ttc_weight = TTC_PENALTY_WEIGHT_V4  # 0.0 -- confirmed inert; no Stage 9 analog
                collision_penalty_override = None  # no clawback this version
                hard_braking_eta = HARD_BRAKING_ETA_V4
                time_cost_per_step = TIME_COST_PER_STEP_V4
            elif enable_clawback_collision_penalty:
                collision_penalty_mag = 0.0  # unused: override below covers every active vehicle
                ttc_weight = 0.0  # confirmed inert in v1; dropped to isolate the clawback's own effect
                collision_penalty_override = dict(cumulative_reward)
                hard_braking_eta = None
                time_cost_per_step = None
            else:
                collision_penalty_mag = collision_penalty_at_step(step)
                ttc_weight = ttc_weight_at_step(step)
                collision_penalty_override = None
                hard_braking_eta = None
                time_cost_per_step = None

            next_obs_dict, reward, terminated, truncated, step_info = env.step(
                actions,
                collision_penalty_magnitude=collision_penalty_mag,
                ttc_penalty_weight=ttc_weight,
                collision_penalty_override=collision_penalty_override,
                exit_reward_magnitude=exit_mag_used,
                hard_braking_eta=hard_braking_eta,
                time_cost_per_step=time_cost_per_step,
            )
            step += 1
            zone_t1 = step_info["zone_t1"]

            if enable_clawback_collision_penalty:
                for vid in active_ids:
                    if not step_info["collision_penalty_applied"][vid]:
                        cumulative_reward[vid] += clawback_tracking_contribution(
                            step_reward=reward[vid],
                            exit_event=bool(step_info["exit_event"][vid]),
                            exit_reward_magnitude=exit_mag_used,
                        )

            if step_info["collision_event"]:
                for vid in active_ids:
                    if was_completed[vid]:
                        already_completed_before_collision[vid] = True

            # --- Stakeholder welfare (chap02.tex-ported, measured only) ---
            attainment = {
                vid: target_speed_attainment(env._vehicles[vid].speed, target_speed)
                for vid in active_ids
            }
            completed_t1 = step_info["completed"]
            experience = {
                vid: stakeholder_experience(attainment[vid], completed=bool(completed_t1[vid]))
                for vid in active_ids
            }
            phi_mean = mean_welfare(list(experience.values()))
            phi_min = min_welfare(list(experience.values()))
            for vid in active_ids:
                if not was_completed[vid]:
                    on_road_attainment[vid].append(attainment[vid])

            # --- Convention-style crossing-order bookkeeping ---
            if first_crosser is None:
                for vid in active_ids:
                    if zone_t[vid] == "merging" and zone_t1[vid] == "post":
                        first_crosser = (roles[vid], vid)
                        break

            if traj_file is not None:
                traj_file.write(
                    json.dumps(
                        {
                            "step": step,
                            "seed": master_seed,
                            "architecture": architecture_used,
                            "collision_penalty_magnitude": collision_penalty_mag,
                            "ttc_penalty_weight": ttc_weight,
                            "hard_braking_eta": step_info["hard_braking_eta_used"],
                            "time_cost_per_step": step_info["time_cost_per_step_used"],
                            "welfare_mean": phi_mean,
                            "welfare_min": phi_min,
                            "term_reason": step_info["term_reason"],
                            "collision_event": bool(step_info["collision_event"]),
                            "vehicles": [
                                {
                                    "id": vid,
                                    "role": roles[vid],
                                    "zone": zone_t[vid],
                                    "position": float(env._vehicles[vid].route_position),
                                    "speed": float(env._vehicles[vid].speed),
                                    "action": int(actions[vid]),
                                    "attainment": float(attainment[vid]),
                                    "experience": float(experience[vid]),
                                    "ttc_penalty": float(step_info["ttc_penalty"][vid]),
                                    "collision_penalty_applied": bool(
                                        step_info["collision_penalty_applied"][vid]
                                    ),
                                    "collision_penalty_used": float(
                                        step_info["collision_penalty_used_per_vehicle"][vid]
                                    ),
                                    "hard_braking_cost": float(step_info["hard_braking_cost_used"][vid]),
                                    "time_cost_applied": bool(step_info["time_cost_applied"][vid]),
                                }
                                for vid in active_ids
                            ],
                        }
                    )
                    + "\n"
                )

            for vid in active_ids:
                if was_completed[vid]:
                    continue
                exited_now = bool(step_info["exit_event"][vid])
                collided = bool(step_info["collision_event"])
                controller_terminal = exited_now or collided
                if controller_terminal:
                    next_obs = None
                    next_mask = None
                else:
                    next_obs = next_obs_dict[vid]
                    next_mask = masks[vid]
                transition = ReplayTransition(
                    observation=obs_dict[vid],
                    action=int(actions[vid]),
                    shaped_reward=float(reward[vid]),
                    next_observation=next_obs,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    action_mask=masks[vid],
                    next_action_mask=next_mask,
                    controller_terminal=controller_terminal,
                    learner_completed=exited_now,
                )
                learner_for(vid).store_transition(transition)

            for lnr in learner_map.values():
                if len(lnr.replay) >= REPLAY_WARMUP:
                    if enable_hysteretic_v10:
                        ratio = HYSTERESIS_RATIO_V10
                    elif enable_hysteretic_v9:
                        ratio = HYSTERESIS_RATIO_V9
                    else:
                        ratio = None
                    lnr.update(hysteresis_ratio=ratio)
                    if lnr._update_count % TARGET_SYNC_INTERVAL_UPDATES == 0:
                        lnr.hard_sync_target()

            obs_dict = next_obs_dict

            if terminated or truncated:
                episode_collided = step_info["term_reason"] == "collision"
                U_by_role = {
                    roles[vid]: episode_mobility_outcome(
                        collided=episode_collided_for_vehicle(
                            episode_collided=episode_collided,
                            was_already_completed_before_collision=already_completed_before_collision[vid],
                        ),
                        attainment_samples=on_road_attainment[vid],
                    )
                    for vid in active_ids
                }
                window.episodes += 1
                success = step_info["term_reason"] == "success"
                if success:
                    window.completions += 1
                elif step_info["term_reason"] == "collision":
                    window.collisions += 1
                elif step_info["term_reason"] == "truncation":
                    window.truncations += 1
                window.record_episode(welfare_by_stakeholder=U_by_role, first_crosser=first_crosser)

                obs_dict, info = env.reset(seed=master_seed * 1_000_003 + step)
                roles = dict(info["roles"])
                active_ids = tuple(roles.keys())
                on_road_attainment = {vid: [] for vid in active_ids}
                first_crosser = None
                cumulative_reward = {vid: 0.0 for vid in active_ids}
                already_completed_before_collision = {vid: False for vid in active_ids}

            if enable_periodic_reset_v7 and 0 < step < max_steps and step % RESET_INTERVAL_V7 == 0:
                for lnr in learner_map.values():
                    reset_last_layers(lnr)

            if step in checkpoint_targets:
                save_checkpoint(step)
    finally:
        if traj_file is not None:
            traj_file.close()

        if step in checkpoint_targets and (not checkpoint_records or checkpoint_records[-1]["step"] != step):
            save_checkpoint(step)

    manifest = {
        "stage": "stage11_dyad_merge_pilot",
        "protocol_tag": PROTOCOL_TAG,
        "seed": master_seed,
        "algorithm": ALGORITHM,
        "condition": CONDITION,
        "architecture": architecture_used,
        "reward_version": reward_version,
        "final_step": step,
        "code_commit": code_commit,
        "checkpoint_steps": checkpoint_targets,
        "checkpoints": checkpoint_records,
    }
    manifest_path = output_root / f"seed_{master_seed}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "REWARD_VERSION_STAGE11",
    "REWARD_VERSION_STAGE11_V1",
    "REWARD_VERSION_STAGE11_V2_CLAWBACK",
    "REWARD_VERSION_STAGE11_V3_CLAWBACK_FIXED_HIGHER_EXIT",
    "REWARD_VERSION_STAGE11_V4_STAGE9_BASED",
    "REWARD_VERSION_STAGE11_V5_COLLISION_CURRICULUM_BIGGER_REPLAY",
    "REWARD_VERSION_STAGE11_V6_ROLE_SPLIT",
    "REWARD_VERSION_STAGE11_V7_PERIODIC_RESET",
    "REWARD_VERSION_STAGE11_V8_VISIT_COUNT_EPSILON",
    "REWARD_VERSION_STAGE11_V9_HYSTERETIC",
    "REWARD_VERSION_STAGE11_V10_HYSTERETIC_RATIO_0_3",
    "REWARD_VERSION_STAGE11_V11_PRIORITY_SIGNAL",
    "REWARD_VERSION_STAGE11_V12_JOINT_BASELINE",
    "REWARD_VERSION_STAGE11_V12_JOINT_MEAN_PBRS",
    "REWARD_VERSION_STAGE11_V12_JOINT_MIN_PBRS",
    "reset_last_layers",
    "EpisodeWindowStats",
    "build_shared_dqn_config",
    "run_stage11_pilot_training_job",
]
