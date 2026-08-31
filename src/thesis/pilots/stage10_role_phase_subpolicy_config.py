"""Stage 10 pilot (E28) frozen-draft constants and guards.

Status: DRAFT protocol (E28_STAGE10_PILOT_PROTOCOL_DRAFT.md) -- not yet
git-tagged/frozen per that document's S7 governance item 1. PROTOCOL_TAG
below is the *planned* tag name, recorded here so code and doc agree, but
this pilot must not be represented as a frozen/tagged experiment until a
human has signed off.

Purpose: diagnostic-only feasibility pilot for the role x zone sub-DQN
architecture on the new symmetric 4-vehicle merge geometry. No PASS/FAIL
gate, no PBRS, base reward only (protocol S1).
"""

from __future__ import annotations

from thesis.agents.dqn_bootstrap import DQNTargetMode

PROTOCOL_TAG = "stage10-pilot-protocol-v1"  # planned tag, NOT yet applied -- see module docstring
STAGE = "stage10_role_phase_subpolicy_pilot"
PURPOSE = "feasibility_pilot_role_zone_subpolicy_architecture"
ALGORITHM = "double_dqn"
CONDITION = "baseline"  # base reward only, no PBRS (protocol S1 item 3)

# Pilot v1 (round 1): base architecture, no LR decay, 100m post-merge zone,
# no curriculum. Frozen -- do not rerun/overwrite these seeds.
PILOT_V1_SEEDS: tuple[int, ...] = (68001, 68002, 68003, 68004)
# Pilot v2 (round 2): 2-stage vehicle-count curriculum (hard cutover) + LR
# decay + 150m post-merge zone + trajectory logging. New seeds, disjoint
# from v1.
PILOT_V2_SEEDS: tuple[int, ...] = (68005, 68006, 68007, 68008)
# Pilot v3 (round 3): reverts the 150m post-merge zone back to 100m,
# replaces the hard curriculum cutover with an episode-level probability
# blend, and adds a peer-intent observation feature (OBS_DIM 7->8). New
# seeds, disjoint from v1/v2. RESULT: made things worse (~2.7% vs v2's
# ~10.3%) -- root cause was a missing scene-context observation feature,
# fixed in pilot v4 below.
PILOT_V3_SEEDS: tuple[int, ...] = (68009, 68010, 68011, 68012)
# Pilot v4 (round 4): scope change -- E28's final target is now a symmetric
# 6-vehicle merge (2026-08-05, see protocol doc S0.1), reached via a
# threshold-triggered 2->4->6 curriculum (Gupta et al. 2017 Algorithm 2
# style, replacing v3's probability blend), a new scene-vehicle-count
# observation feature (OBS_DIM 8->9, fixes v3's identified partial-
# observability gap), and an easier baseline geometry (wider pre-merge zone
# and spawn spacing, longer per-episode step budget). New seeds, disjoint
# from v1/v2/v3. Extended from 4 to 8 seeds (2026-08-05, same session as the
# epsilon fix above) given the demonstrated high seed-to-seed variance in
# this pilot (e.g. pilot v2's completion ranged 0%-28.9% across just 4
# seeds at the same checkpoint) -- 4 seeds risks an unrepresentative read;
# 8 is still pilot-scale, not Stage 8/9's 20-seed formal-gate scale. Cheap
# to add given seeds now run as separate concurrent processes on a 32-core
# machine rather than sequentially -- wall-clock is bounded by the slowest
# single seed, not by seed count.
PILOT_V4_SEEDS: tuple[int, ...] = (68013, 68014, 68015, 68016, 68017, 68018, 68019, 68020)
# Pilot v5 (round 5): architectural pivot -- ONE shared Q-network for every
# vehicle (Gupta et al. 2017's parameter-sharing approach) instead of six
# independent (role,zone)-keyed networks, with role and zone now explicit
# input features (stage10_symmetric_merge_env.OBS_DIM_WITH_ROLE_ZONE, 13-dim)
# since routing no longer implicitly encodes them. Deliberately a CONTROLLED
# comparison against pilot v4: same curriculum stages/thresholds/safety
# valves, same LR/epsilon schedules, same geometry -- only the architecture
# changes, so any completion-rate difference is attributable to parameter
# sharing specifically. See stage10_shared_dqn.py and protocol doc S0.1.
PILOT_V5_SEEDS: tuple[int, ...] = (68021, 68022, 68023, 68024, 68025, 68026, 68027, 68028)
# Pilot v6 (round 6): reward-function revision, keeping v5's shared-parameter
# architecture completely unchanged -- (a) adds a continuous TTC-based
# proximity-risk term, (b) makes the collision penalty perpetrator-only
# rather than uniform to every active vehicle, (c) raises the collision
# penalty magnitude from 1.0 to 3.0 (5:1 vs the 0.6 completion reward). See
# stage10_symmetric_merge_env.py's module docstring "Reward revision"
# section for the full literature-grounded rationale. Deliberately a
# CONTROLLED comparison against pilot v5: same architecture, curriculum,
# LR/epsilon schedules, geometry, and budget -- only the reward formula
# changes. New seeds, disjoint from v1-v5.
PILOT_V6_SEEDS: tuple[int, ...] = (68029, 68030, 68031, 68032, 68033, 68034, 68035, 68036)
# Pilot v7 (round 7): reward-MAGNITUDE curriculum, keeping v6's reward
# FORMULA (TTC shaping + perpetrator-only collision penalty) and v5's
# architecture completely unchanged -- only the collision-penalty magnitude
# and TTC weight now ramp over training instead of being static from step 0.
# Directly modelled on Alhazza (2026)'s DQN004 (continuous-ramp collision
# penalty) and DQN006 (a second, independently-ramped shaping term); v6's
# static-full-strength-from-step-0 approach reproduced the same "too
# cautious, farms survival instead of completing" collapse Alhazza's DQN002/
# DQN005 hit before their own curriculum fix. See
# stage10_symmetric_merge_env.py's module docstring "Reward-magnitude
# curriculum" section for the full rationale/citations, and
# ``collision_penalty_at_step``/``ttc_weight_at_step`` below for the ramp
# functions. Deliberately a CONTROLLED comparison against pilot v6: same
# architecture, curriculum, LR/epsilon schedules, geometry, and budget --
# only the reward magnitudes' time-dependence changes. New seeds, disjoint
# from v1-v6.
PILOT_V7_SEEDS: tuple[int, ...] = (68037, 68038, 68039, 68040, 68041, 68042, 68043, 68044)
# Union -- any arm's seeds pass the strict guard below (all are frozen
# pilot data, just from different arms of the same E28 Stage 10 pilot).
PILOT_SEEDS: tuple[int, ...] = (
    PILOT_V1_SEEDS
    + PILOT_V2_SEEDS
    + PILOT_V3_SEEDS
    + PILOT_V4_SEEDS
    + PILOT_V5_SEEDS
    + PILOT_V6_SEEDS
    + PILOT_V7_SEEDS
)
STAGE10_RESERVED_SEED_BLOCK: tuple[int, ...] = tuple(range(68001, 68045))

# Architecture tags (pilot v5) -- threaded into checkpoints/manifests so
# output data is self-describing about which architecture produced it,
# without needing to cross-reference code history.
ARCHITECTURE_INDEPENDENT_SUBPOLICY = "independent_subpolicy"  # pilots v1-v4
ARCHITECTURE_SHARED_PARAMETER = "shared_parameter"  # pilot v5

# Every prior stage's seed block -- pilot seeds must never overlap these
# (rl-thesis-protocol anti-pattern: pilot/formal seed overlap).
FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORBIDDEN_STAGE7A1_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
FORBIDDEN_STAGE7B_SEEDS: tuple[int, ...] = tuple(range(63001, 63021))
FORBIDDEN_STAGE7C_Q1_SEEDS: tuple[int, ...] = tuple(range(64001, 64021))
FORBIDDEN_STAGE8_SEEDS: tuple[int, ...] = tuple(range(65001, 65041))
FORBIDDEN_STAGE9_MEAN_PBRS_SEEDS: tuple[int, ...] = tuple(range(66001, 66021))
FORBIDDEN_STAGE9_MIN_PBRS_SEEDS: tuple[int, ...] = tuple(range(67001, 67021))

MAX_STEPS = 100_000  # frozen v1-v3 total budget -- do not change (historical runs depend on this)
CHECKPOINT_STEPS: tuple[int, ...] = (0, 20_000, 40_000, 60_000, 80_000, 100_000)
EPSILON_START = 1.0
EPSILON_END = 0.10
EPSILON_DECAY_STEPS = 50_000  # reused unchanged from Stage 8/9 formal defaults

# DQN hyperparameters: reused unchanged from FormalDQNConfig (Stage 8/9), per
# protocol S2.3 ("复用现有Stage 8/9已验证的Double DQN架构与训练设置... 除非pilot结果
# 显示明确需要调整，否则不做架构/超参搜索"). Pilot v1 had no LR decay -- diagnosed
# afterwards (with the user, citing Gupta et al. 2017) as a likely contributor to
# v1's oscillating/non-monotonic metrics. Pilot v2 reintroduces it, matching
# Stage 8 arm2b's schedule exactly (0.0005->0.0001 linear).
HIDDEN_SIZES: tuple[int, ...] = (64, 64)
LEARNING_RATE = 0.0005  # pilot v1: constant; pilot v2: this is the START of the decay
GAMMA = 0.995
REPLAY_CAPACITY_PER_SUBPOLICY = 20_000
BATCH_SIZE = 64
REPLAY_WARMUP_PER_SUBPOLICY = 512
TARGET_SYNC_INTERVAL_UPDATES = 250

OBS_DIM = 9  # Stage10SymmetricMergeEnv.OBS_DIM (v3: +1 peer-intent; v4: +1 scene-vehicle-count)
# Pilot v5 only: Stage10SymmetricMergeEnv.OBS_DIM_WITH_ROLE_ZONE (+1 role, +3 zone one-hot).
OBS_DIM_V5 = 13
N_ACTIONS = 3

# --- Pilot v2 additions -----------------------------------------------------

# LR decay (Stage 8 arm2b schedule, reused verbatim): linear over the FULL
# 100K-step budget, continuous through the curriculum transition below -- no
# reset at the stage boundary (untested combination, flagged to the user as
# a judgement call, not a validated interaction).
LEARNING_RATE_START = LEARNING_RATE  # 0.0005
LEARNING_RATE_END = 0.0001
LEARNING_RATE_DECAY_STEPS = 100_000

# Two-stage vehicle-count curriculum (Gupta, Egorov & Kochenderfer 2017's own
# mitigation for DQN/PS-DQN degrading as simultaneous-learner count rises --
# their Multi-Walker curriculum scaled agent count the same way). Stage A is
# 1 ramp + 1 mainline (the minimal interacting pair -- not a literal single
# vehicle, which would have no merge conflict at all), Stage B is the full
# symmetric 2+2. Fixed split, not Gupta's adaptive Dirichlet-weighted
# threshold-triggered version (that's a materially larger engineering lift;
# a fixed split is the deliberately simpler choice for this diagnostic pilot).
CURRICULUM_STAGE_A_VEHICLES = 2
CURRICULUM_STAGE_A_STEPS = 40_000  # pilot v2's hard-cutover step -- kept for provenance/its own tests, unused by pilot v3's runner
CURRICULUM_STAGE_B_VEHICLES = 4

# --- Pilot v3 blend curriculum: REMOVED in pilot v4 -------------------------
#
# Round 3 replaced round 2's hard cutover with an episode-level probability
# blend (p_four_vehicle_at_step/n_vehicles_for_draw, a simplified analogue of
# Portelas et al.'s ALP-GMM, arXiv:1910.07224). RESULT: this made things
# WORSE (~2.7% vs round 2's ~10.3% completion). Root cause: no observation
# feature ever told the network how many vehicles were active in the current
# episode, so blending 2- and 4-vehicle episodes in the same time window fed
# contradictory optimal-action targets for near-identical observations,
# corrupting the training signal throughout the whole blend window (not just
# at one shock moment, as round 2's hard cutover did). Removed rather than
# left dead alongside the v4 mechanism below; see stage10_symmetric_merge_env.py's
# module docstring and protocol doc S0.1 for the full account.
# (tests referencing these removed functions were updated in
# test_stage10_pilot_v3.py, not silently broken.)

# --- Pilot v4 additions ------------------------------------------------------
#
# Scope change (2026-08-05, protocol doc S0.1): E28's final target is now a
# symmetric 6-vehicle merge. Threshold-triggered 3-stage curriculum modelled
# on Gupta, Egorov & Kochenderfer (2017)'s actual Algorithm 2 mechanism
# (train the current difficulty until a performance threshold clears, then
# advance -- NOT a fixed step split like round 2, NOT a continuous blend like
# round 3): Stage 1 (2 vehicles) -> Stage 2 (4 vehicles) -> Stage 3 (6
# vehicles, terminal). Each non-terminal stage advances once a rolling
# completion rate over the trailing CURRICULUM_V4_ROLLING_WINDOW_EPISODES
# episodes reaches CURRICULUM_V4_ADVANCE_THRESHOLD, OR once that stage's own
# safety-valve step cap is reached, whichever comes first (the safety valve
# exists so training can't stall forever on an unreachable threshold -- Gupta
# et al.'s own version has an analogous role, though implemented via
# Dirichlet-weight circular-shifting rather than a hard step cap; a full
# Dirichlet-weighted multi-task sampler would be disproportionate engineering
# for a diagnostic pilot).
CURRICULUM_V4_STAGE_VEHICLE_COUNTS: tuple[int, int, int] = (2, 4, 6)
CURRICULUM_V4_STAGE_MAX_STEPS: tuple[int, int, int] = (40_000, 60_000, 80_000)  # per-stage safety valves
CURRICULUM_V4_ADVANCE_THRESHOLD: float = 0.90  # NOT 100% -- see protocol discussion, even the easiest
# regime can have rare genuine failures; requiring literal 100% risks never advancing.
CURRICULUM_V4_ROLLING_WINDOW_EPISODES: int = 100  # trailing-episode window for the advancement decision;
# must be FULL (>= this many episodes since the current stage started) before the threshold can trigger,
# to avoid a tiny-sample fluke advancing (or failing to advance) a stage.

# Stage-transition re-exploration bump (2026-08-05 fix, same session as the
# 8-seed extension below): epsilon_at_step's fixed 50,000-step decay window
# means Stage 3 (6 vehicles, usually starting well past step 100,000 given
# Stage 1+2's 40K+60K safety valves) previously began fully floored at
# EPSILON_END with no dedicated exploration boost for the harder task. Fix
# is scoped, not a global rewrite of epsilon_at_step (which keeps governing
# Stage 1 unchanged, since Stage 1 already starts at EPSILON_START by
# definition and needs no bump): on entering Stage 2 or Stage 3, epsilon
# jumps to EPSILON_STAGE_BUMP_VALUE and decays linearly back to EPSILON_END
# over the first EPSILON_STAGE_BUMP_DECAY_FRACTION of that stage's own
# CURRICULUM_V4_STAGE_MAX_STEPS allocation, floor for the remainder. See
# ``epsilon_for_step`` below, which dispatches between the two schedules.
EPSILON_STAGE_BUMP_VALUE: float = 0.5
EPSILON_STAGE_BUMP_DECAY_FRACTION: float = 0.20

MAX_STEPS_V4: int = sum(CURRICULUM_V4_STAGE_MAX_STEPS)  # 180_000: the safety-valve CEILING, not a guarantee
# -- a seed whose early stages clear their threshold quickly uses fewer total steps; this is only reached
# if every stage exhausts its own safety valve.
LEARNING_RATE_DECAY_STEPS_V4: int = MAX_STEPS_V4  # LR decays across the full 180K ceiling, not the old 100K
CHECKPOINT_STEPS_V4: tuple[int, ...] = tuple(range(0, MAX_STEPS_V4 + 1, 20_000))  # 0,20K,...,180K


def epsilon_at_step(step: int) -> float:
    """Linear decay EPSILON_START -> EPSILON_END over EPSILON_DECAY_STEPS, then constant.
    NOTE (pilot v4): left unchanged (still decays to minimum by step 50,000) despite the
    curriculum's now much longer (up to 180,000-step) total budget -- this was not one of
    the four confirmed round-4 changes, and is flagged here as a known limitation rather
    than silently extended: Stage 3 (6 vehicles, the hardest/final stage) will in most
    runs begin well after epsilon has already reached its floor, so it gets no dedicated
    extra-exploration boost for the harder task specifically."""
    if step >= EPSILON_DECAY_STEPS:
        return float(EPSILON_END)
    frac = float(step) / float(EPSILON_DECAY_STEPS)
    return float(EPSILON_START + frac * (EPSILON_END - EPSILON_START))


def epsilon_for_stage_transition(
    steps_in_stage: int,
    *,
    stage_max_steps: int,
    bump_value: float = EPSILON_STAGE_BUMP_VALUE,
    decay_fraction: float = EPSILON_STAGE_BUMP_DECAY_FRACTION,
    floor: float = EPSILON_END,
) -> float:
    """Pilot v4 fix: the re-exploration bump used for non-initial curriculum
    stages (Stage 2, Stage 3). ``steps_in_stage`` is steps elapsed since THIS
    stage began (0 at the instant the stage starts, not the global step
    count) -- epsilon starts at ``bump_value`` and decays linearly back to
    ``floor`` over the first ``decay_fraction`` of ``stage_max_steps``, then
    stays at ``floor`` for the remainder of the stage. Pure function of its
    arguments, no hidden state, so it's directly unit-testable against
    hand-picked (steps_in_stage, stage_max_steps) pairs."""
    decay_steps = decay_fraction * stage_max_steps
    if decay_steps <= 0 or steps_in_stage >= decay_steps:
        return float(floor)
    frac = float(steps_in_stage) / float(decay_steps)
    return float(bump_value + frac * (floor - bump_value))


def epsilon_for_step(
    *,
    step: int,
    stage_idx: int,
    steps_in_stage: int,
    stage_max_steps: tuple[int, ...] = CURRICULUM_V4_STAGE_MAX_STEPS,
) -> float:
    """Combined pilot-v4 epsilon schedule: Stage 1 (``stage_idx == 0``) uses
    the original global ``epsilon_at_step(step)`` schedule completely
    unchanged (it already starts at ``EPSILON_START`` by definition, no bump
    needed); Stage 2/3 (``stage_idx >= 1``) use the stage-transition
    re-exploration bump instead, keyed off ``steps_in_stage`` (which resets
    to 0 at every curriculum transition) rather than the global step count,
    so exploration genuinely resets fresh each time the task gets harder."""
    if stage_idx == 0:
        return epsilon_at_step(step)
    return epsilon_for_stage_transition(steps_in_stage, stage_max_steps=stage_max_steps[stage_idx])


def lr_at_step(step: int, *, decay_steps: int = LEARNING_RATE_DECAY_STEPS) -> float:
    """Linear LR decay LEARNING_RATE_START -> LEARNING_RATE_END over
    ``decay_steps``, then constant -- mirrors epsilon_at_step's shape and
    formal_config.lr_at_step's formula (Stage 8 arm2b precedent). Purely a
    function of absolute step count, so it is automatically continuous
    across any curriculum vehicle-count transition -- there is no separate
    schedule per stage to reset. ``decay_steps`` defaults to the frozen v1-v3
    value (LEARNING_RATE_DECAY_STEPS, 100_000) so existing v1-v3 call sites
    and tests are unaffected; pilot v4's runner passes
    ``decay_steps=LEARNING_RATE_DECAY_STEPS_V4`` (180_000) explicitly."""
    if step >= decay_steps:
        return float(LEARNING_RATE_END)
    frac = float(step) / float(decay_steps)
    return float(LEARNING_RATE_START + frac * (LEARNING_RATE_END - LEARNING_RATE_START))


# --- Pilot v7 additions: reward-magnitude curriculum ------------------------
#
# Ramp schedule for the collision-penalty magnitude and TTC-shaping weight
# (see stage10_symmetric_merge_env.py's module docstring "Reward-magnitude
# curriculum" section for the full literature rationale: Alhazza (2026)
# dissertation's DQN004/DQN006, itself building on Freitag et al. (2025)'s
# reward-curriculum concept -- secondary citation, not independently verified
# against the Freitag primary source). Single continuous linear ramp across
# steps 0 -> REWARD_CURRICULUM_RAMP_STEPS, then locked at the target value --
# same "pure function of step" shape as epsilon_at_step/lr_at_step above, and
# deliberately NOT reset/re-ramped at curriculum vehicle-count transitions
# (Alhazza's own DQN004 ramp was likewise a single pass across the whole
# run, not interleaved with another curriculum).
COLLISION_PENALTY_CURRICULUM_START: float = 1.0  # this pilot's own pre-v6 magnitude (v1-v5)
COLLISION_PENALTY_CURRICULUM_END: float = 5.0  # matches stage10_symmetric_merge_env.COLLISION_PENALTY_MAGNITUDE (v6's target)
TTC_WEIGHT_CURRICULUM_START: float = 0.02  # lenient start, same proportional shrink as the collision-penalty ramp
TTC_WEIGHT_CURRICULUM_END: float = 0.1  # matches stage10_symmetric_merge_env.TTC_PENALTY_WEIGHT (v6's target)
REWARD_CURRICULUM_RAMP_STEPS: int = 100_000  # = CURRICULUM_V4_STAGE_MAX_STEPS[0] + [1] (Stage 1+2 combined safety
# valve budget) -- ramp completes right as Stage 3 (6 vehicles, final/hardest) begins, so Stage 3 gets the full
# target penalty for its entire consolidation period, mirroring Alhazza's DQN004 Phase C.


def collision_penalty_at_step(
    step: int,
    *,
    start: float = COLLISION_PENALTY_CURRICULUM_START,
    end: float = COLLISION_PENALTY_CURRICULUM_END,
    ramp_steps: int = REWARD_CURRICULUM_RAMP_STEPS,
) -> float:
    """Linear ramp start -> end over ramp_steps, then constant at end.
    Pure function of its arguments, directly unit-testable."""
    if step >= ramp_steps:
        return float(end)
    frac = float(step) / float(ramp_steps)
    return float(start + frac * (end - start))


def ttc_weight_at_step(
    step: int,
    *,
    start: float = TTC_WEIGHT_CURRICULUM_START,
    end: float = TTC_WEIGHT_CURRICULUM_END,
    ramp_steps: int = REWARD_CURRICULUM_RAMP_STEPS,
) -> float:
    """Linear ramp start -> end over ramp_steps, then constant at end. Same
    shape/timing as collision_penalty_at_step, ramped independently (own
    start/end values) per Alhazza's DQN006 precedent of curriculum-scaling a
    second shaping term separately from the primary collision penalty."""
    if step >= ramp_steps:
        return float(end)
    frac = float(step) / float(ramp_steps)
    return float(start + frac * (end - start))


def vehicles_at_step(step: int) -> int:
    """Pilot v2's hard-cutover curriculum lookup: Stage A (2 vehicles) below
    the transition step, Stage B (4 vehicles) at/after it. Kept for
    provenance/its own tests; superseded by pilot v4's
    ``stage_index_for_advance`` threshold-triggered mechanism below."""
    return CURRICULUM_STAGE_B_VEHICLES if step >= CURRICULUM_STAGE_A_STEPS else CURRICULUM_STAGE_A_VEHICLES


def stage_index_for_advance(
    *,
    current_stage_idx: int,
    rolling_completion_rate: float | None,
    steps_in_current_stage: int,
    n_stages: int = len(CURRICULUM_V4_STAGE_VEHICLE_COUNTS),
    advance_threshold: float = CURRICULUM_V4_ADVANCE_THRESHOLD,
    stage_max_steps: tuple[int, ...] = CURRICULUM_V4_STAGE_MAX_STEPS,
) -> int:
    """Pure decision function for pilot v4's threshold-triggered curriculum:
    given the current stage and its rolling performance/step-count so far,
    return the stage index to use for the NEXT episode (equal to
    ``current_stage_idx`` if no advancement should happen this episode).

    ``rolling_completion_rate`` should be ``None`` if the trailing-episode
    window isn't yet full (caller's responsibility -- keeps this function
    free of any windowing/RNG concerns, same separation-of-concerns pattern
    as pilot v3's ``n_vehicles_for_draw``). The terminal stage
    (``current_stage_idx == n_stages - 1``) never advances further.
    """
    if current_stage_idx >= n_stages - 1:
        return current_stage_idx
    threshold_met = rolling_completion_rate is not None and rolling_completion_rate >= advance_threshold
    safety_valve_hit = steps_in_current_stage >= stage_max_steps[current_stage_idx]
    if threshold_met or safety_valve_hit:
        return current_stage_idx + 1
    return current_stage_idx


def target_mode() -> DQNTargetMode:
    return DQNTargetMode.DOUBLE


def assert_stage10_pilot_guards(*, master_seed: int, max_steps: int) -> None:
    """Hard-fail on any deviation from the frozen-draft pilot scope.
    Pilot v4 seeds require the new MAX_STEPS_V4 (180_000) ceiling; v1-v3
    seeds still require the original frozen MAX_STEPS (100_000) -- each
    arm's own historical budget stays immutable."""
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
    ):
        if master_seed in block:
            raise RuntimeError(f"seed {master_seed} collides with {name} seed block")
    # Pilots v5, v6, and v7 all deliberately reuse MAX_STEPS_V4's budget/
    # curriculum unchanged (controlled comparisons, see PILOT_V5_SEEDS/
    # PILOT_V6_SEEDS/PILOT_V7_SEEDS docstrings above) -- there is no separate
    # MAX_STEPS_V5/V6/V7 constant because the budget is intentionally
    # identical, not coincidentally so.
    expected_max_steps = (
        MAX_STEPS_V4
        if master_seed in (PILOT_V4_SEEDS + PILOT_V5_SEEDS + PILOT_V6_SEEDS + PILOT_V7_SEEDS)
        else MAX_STEPS
    )
    if int(max_steps) != expected_max_steps:
        raise RuntimeError(
            f"max_steps must be {expected_max_steps} for seed {master_seed}, got {max_steps}"
        )


__all__ = [
    "ALGORITHM",
    "BATCH_SIZE",
    "CHECKPOINT_STEPS",
    "CONDITION",
    "EPSILON_DECAY_STEPS",
    "EPSILON_END",
    "EPSILON_START",
    "FORBIDDEN_FORMAL_SEEDS",
    "FORBIDDEN_STAGE7A1_SEEDS",
    "FORBIDDEN_STAGE7B_SEEDS",
    "FORBIDDEN_STAGE7C_Q1_SEEDS",
    "FORBIDDEN_STAGE8_SEEDS",
    "FORBIDDEN_STAGE9_MEAN_PBRS_SEEDS",
    "FORBIDDEN_STAGE9_MIN_PBRS_SEEDS",
    "GAMMA",
    "HIDDEN_SIZES",
    "LEARNING_RATE",
    "LEARNING_RATE_START",
    "LEARNING_RATE_END",
    "LEARNING_RATE_DECAY_STEPS",
    "CURRICULUM_STAGE_A_VEHICLES",
    "CURRICULUM_STAGE_A_STEPS",
    "CURRICULUM_STAGE_B_VEHICLES",
    "CURRICULUM_V4_STAGE_VEHICLE_COUNTS",
    "CURRICULUM_V4_STAGE_MAX_STEPS",
    "CURRICULUM_V4_ADVANCE_THRESHOLD",
    "CURRICULUM_V4_ROLLING_WINDOW_EPISODES",
    "CHECKPOINT_STEPS_V4",
    "MAX_STEPS",
    "MAX_STEPS_V4",
    "LEARNING_RATE_DECAY_STEPS_V4",
    "N_ACTIONS",
    "OBS_DIM",
    "OBS_DIM_V5",
    "PILOT_SEEDS",
    "PILOT_V1_SEEDS",
    "PILOT_V2_SEEDS",
    "PILOT_V3_SEEDS",
    "PILOT_V4_SEEDS",
    "PILOT_V5_SEEDS",
    "PILOT_V6_SEEDS",
    "PILOT_V7_SEEDS",
    "COLLISION_PENALTY_CURRICULUM_START",
    "COLLISION_PENALTY_CURRICULUM_END",
    "TTC_WEIGHT_CURRICULUM_START",
    "TTC_WEIGHT_CURRICULUM_END",
    "REWARD_CURRICULUM_RAMP_STEPS",
    "collision_penalty_at_step",
    "ttc_weight_at_step",
    "ARCHITECTURE_INDEPENDENT_SUBPOLICY",
    "ARCHITECTURE_SHARED_PARAMETER",
    "PROTOCOL_TAG",
    "PURPOSE",
    "REPLAY_CAPACITY_PER_SUBPOLICY",
    "REPLAY_WARMUP_PER_SUBPOLICY",
    "STAGE",
    "STAGE10_RESERVED_SEED_BLOCK",
    "TARGET_SYNC_INTERVAL_UPDATES",
    "assert_stage10_pilot_guards",
    "epsilon_at_step",
    "epsilon_for_stage_transition",
    "epsilon_for_step",
    "EPSILON_STAGE_BUMP_VALUE",
    "EPSILON_STAGE_BUMP_DECAY_FRACTION",
    "lr_at_step",
    "vehicles_at_step",
    "stage_index_for_advance",
    "target_mode",
]
