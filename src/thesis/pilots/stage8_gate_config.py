"""Stage 8 formal qualification gate: frozen constants and guards.

This is Stage 8's own re-run of the Stage 7C-Q1 competence gate
(`STAGE1_TO_STAGE7_EXPERIMENT_FREEZE.md` SS10.4.9, FAIL), at the SAME scale
(20 seeds x 400,000 joint environment steps) and with the SAME gate
threshold structure, but training under the configuration selected by the
Stage 8 pilot chain (arm0 -> arm1 -> arm2a -> arm2b, see
`paper/STAGE8_PLAN_DRAFT.md` SS2.3/SS4/SS5):

- Double DQN, Base Reward V2 (`active_time_cost_per_step=0.0005`) -- unchanged.
- epsilon schedule: `epsilon_decay_environment_steps=50,000` (arm0's value,
  NOT arm1's 75,000 -- arm1 did not show a clear advantage).
- target-network update: hard copy every 250 updates (arm0's default, NOT
  arm2a's soft/Polyak update -- arm2a showed a *systematic* late-stage
  collapse in both of its pilot seeds, evidence pointing the wrong way).
- learning rate: linear decay 0.0005 -> 0.0001 (arm2b's fix, the only pilot
  arm with a clear, evidence-backed improvement: 91.4% mean success at the
  100K pilot checkpoint across 8 seeds vs 84.4%/81.2%/50.0% for
  arm0/arm1/arm2a respectively). The decay window is extended from arm2b's
  pilot value (100,000, matching the pilot's own step budget) to span the
  full 400,000-step gate budget -- this specific decay *shape* was NOT
  pilot-validated and is a proportional-scaling extrapolation, confirmed
  with the user on 2026-08-02 (see STAGE8_PLAN_DRAFT.md SS5.1 for the
  explicit acknowledgement that this is a new, unverified assumption).

It makes a PASS / FAIL / INVALID competence-gate claim, unlike every prior
Stage 8 arm (arm0/arm1/arm2a/arm2b were diagnostic pilots only).

Gate threshold constants (GATE_*, GATE_CHECKPOINTS, LEARNING_CURVE_CHECKPOINTS,
EARLY/LATE_SCENARIO_BLOCKS, EARLY_EVAL_MAX_CHECKPOINT) are copied VERBATIM
from `thesis.pilots.stage7c_q1_config` -- not re-derived -- so this gate's
PASS/FAIL decision is directly comparable to Stage 7C-Q1's own FAIL under
the identical rule set (`STAGE8_PLAN_DRAFT.md` SS5.3: "沿用 Stage 7C-Q1 的
门槛结构，不重新推导"). The gate-decision function itself
(`thesis.pilots.stage7c_q1_gate.evaluate_competence_gate`) is reused
unchanged, called with this module's own `PILOT_SEEDS`.
"""

from __future__ import annotations

from thesis.agents.dqn_bootstrap import DQNTargetMode

PROTOCOL_TAG = "stage8-gate-protocol-v1"
STAGE = "stage8_gate"
PURPOSE = "formal_qualification_gate_lr_decay_fix"
ALGORITHM = "double_dqn"
CONDITION = "baseline"
BASE_REWARD_VERSION = "v2_active_time"

# Identical to Stage 7C-Q1 / every Stage 8 pilot arm -- reward is not varied.
ACTIVE_TIME_COST_PER_STEP = 0.0005
ACTIVE_TIME_COST_PER_SECOND = 0.0025
POLICY_INTERVAL_SECONDS = 0.20

# 20 new seeds, disjoint from every historical block including the Stage 8
# pilot block (65001-65020, fully consumed by arm0/arm1/arm2a/arm2b).
PILOT_SEEDS: tuple[int, ...] = tuple(range(65021, 65041))

FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORBIDDEN_STAGE7A1_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
FORBIDDEN_STAGE7B_SEEDS: tuple[int, ...] = tuple(range(63001, 63021))
FORBIDDEN_STAGE7C_Q1_SEEDS: tuple[int, ...] = tuple(range(64001, 64021))
FORBIDDEN_STAGE8_PILOT_SEEDS: tuple[int, ...] = tuple(range(65001, 65021))

MAX_STEPS = 400_000
CHECKPOINT_STEPS: tuple[int, ...] = tuple(range(0, 400_001, 25_000))
GATE_CHECKPOINTS: tuple[int, ...] = (350_000, 375_000, 400_000)
LEARNING_CURVE_CHECKPOINTS: tuple[int, ...] = tuple(range(200_000, 400_001, 25_000))
EARLY_EVAL_MAX_CHECKPOINT = 175_000
LATE_EVAL_MIN_CHECKPOINT = 200_000
EARLY_SCENARIO_BLOCKS = 8
LATE_SCENARIO_BLOCKS = 32
ASSIGNMENTS_PER_BLOCK = 2

# Checkpoints that get full per-step diagnostic logging (Q-values, front_gap,
# minimum_TTC, action sequence). Confirmed with the user on 2026-08-02:
# only the checkpoints that feed the actual gate DECISION get rich logging
# (0 as the untrained baseline + the 3 GATE_CHECKPOINTS); the other 13
# checkpoints (needed only for learning-curve-continuity / material-regression
# success-rate deltas) use the lightweight evaluator (episode-level only, no
# per-step Q-value logging) to keep data volume tractable at 20 seeds x
# 400K steps x up to 64 episodes/checkpoint.
RICH_LOG_CHECKPOINTS: tuple[int, ...] = (0, 350_000, 375_000, 400_000)

# Unchanged from arm0 (NOT arm1's 75,000 -- arm1 showed no clear advantage).
EPSILON_DECAY_STEPS = 50_000

# Unchanged from arm0 (NOT arm2a's soft update -- arm2a showed a systematic
# late-stage collapse in pilot). target_sync_interval_updates stays at
# FormalDQNConfig's default (250).
TARGET_UPDATE_MODE = "hard"

# arm2b's fix, decay window extended to span the full 400K gate budget
# (NOT pilot-validated at this length -- see module docstring).
LEARNING_RATE_START = 0.0005
LEARNING_RATE_END = 0.0001
LEARNING_RATE_DECAY_STEPS = 400_000

TRAINING_RUN_COUNT = 20

# Competence gate thresholds -- copied VERBATIM from stage7c_q1_config.py,
# not re-derived. Direct comparability to the Stage 7C-Q1 FAIL decision under
# the identical rule set is the point of this gate.
GATE_MEAN_SUCCESS_MIN = 0.95
GATE_COLLISION_MAX = 0.02
GATE_TRUNCATION_MAX = 0.03
GATE_SWAP_ELIGIBILITY_MIN = 0.75
GATE_SEED_SUCCESS_MIN = 61 / 64  # 0.953125
GATE_MIN_QUALIFIED_SEEDS = 16
GATE_ADJACENT_SUCCESS_DROP_MAX = 0.03
GATE_MATERIAL_REGRESSION = 0.20
GATE_MAX_MATERIAL_REGRESSION_SEEDS = 1
GATE_MAX_LATE_COLLAPSE_SEEDS = 1


def n_scenario_blocks(checkpoint_step: int) -> int:
    """Identical logic to stage7c_q1_config.n_scenario_blocks: 8 physical
    validation-block templates cycled once (early) or 4x with distinct
    eval_seeds (late, checkpoint > 175_000)."""
    if int(checkpoint_step) <= EARLY_EVAL_MAX_CHECKPOINT:
        return EARLY_SCENARIO_BLOCKS
    return LATE_SCENARIO_BLOCKS


def episodes_per_seed_checkpoint(checkpoint_step: int) -> int:
    return n_scenario_blocks(checkpoint_step) * ASSIGNMENTS_PER_BLOCK


def assert_stage8_gate_guards(
    *,
    algorithm: str,
    condition: str,
    reward_shaping_enabled: bool,
    shaping_coefficient: float,
    master_seed: int,
    max_steps: int,
    active_time_cost_per_step: float,
    target_update_mode: str,
    learning_rate_end: float,
    learning_rate_decay_environment_steps: int,
    allow_mean_pbrs: bool = False,
    allow_min_pbrs: bool = False,
) -> None:
    """Hard-fail on any deviation from the frozen gate protocol."""
    if algorithm != ALGORITHM:
        raise RuntimeError(f"algorithm must be {ALGORITHM!r}, got {algorithm!r}")
    if condition != CONDITION:
        raise RuntimeError(f"condition must be {CONDITION!r}, got {condition!r}")
    if reward_shaping_enabled or allow_mean_pbrs or allow_min_pbrs:
        raise RuntimeError("Mean-PBRS / Min-PBRS / shaping forbidden in the Stage 8 gate")
    if float(shaping_coefficient) != 0.0:
        raise RuntimeError("shaping coefficient must be 0")
    if abs(float(active_time_cost_per_step) - ACTIVE_TIME_COST_PER_STEP) > 1e-15:
        raise RuntimeError(
            f"active_time_cost_per_step must be {ACTIVE_TIME_COST_PER_STEP}, "
            f"got {active_time_cost_per_step}"
        )
    if target_update_mode != TARGET_UPDATE_MODE:
        raise RuntimeError(f"target_update_mode must be {TARGET_UPDATE_MODE!r}, got {target_update_mode!r}")
    if abs(float(learning_rate_end) - LEARNING_RATE_END) > 1e-15:
        raise RuntimeError(f"learning_rate_end must be {LEARNING_RATE_END}, got {learning_rate_end}")
    if int(learning_rate_decay_environment_steps) != LEARNING_RATE_DECAY_STEPS:
        raise RuntimeError(
            f"learning_rate_decay_environment_steps must be {LEARNING_RATE_DECAY_STEPS}, "
            f"got {learning_rate_decay_environment_steps}"
        )
    if master_seed not in PILOT_SEEDS:
        raise RuntimeError(f"master_seed {master_seed} not in frozen 65021-65040")
    if master_seed in FORBIDDEN_FORMAL_SEEDS:
        raise RuntimeError("formal seeds 61001-61010 forbidden")
    if master_seed in FORBIDDEN_STAGE7A1_SEEDS:
        raise RuntimeError("Stage 7A-1 seeds 62001-62020 forbidden")
    if master_seed in FORBIDDEN_STAGE7B_SEEDS:
        raise RuntimeError("Stage 7B seeds 63001-63020 forbidden")
    if master_seed in FORBIDDEN_STAGE7C_Q1_SEEDS:
        raise RuntimeError("Stage 7C-Q1 seeds 64001-64020 forbidden")
    if master_seed in FORBIDDEN_STAGE8_PILOT_SEEDS:
        raise RuntimeError("Stage 8 pilot seeds 65001-65020 forbidden in the gate")
    if int(max_steps) != MAX_STEPS:
        raise RuntimeError(f"max_steps must be {MAX_STEPS}, got {max_steps}")
    if int(max_steps) > MAX_STEPS:
        raise RuntimeError("training beyond 400K forbidden")


def target_mode() -> DQNTargetMode:
    return DQNTargetMode.DOUBLE


__all__ = [
    "ACTIVE_TIME_COST_PER_SECOND",
    "ACTIVE_TIME_COST_PER_STEP",
    "ALGORITHM",
    "ASSIGNMENTS_PER_BLOCK",
    "BASE_REWARD_VERSION",
    "CHECKPOINT_STEPS",
    "CONDITION",
    "EARLY_EVAL_MAX_CHECKPOINT",
    "EARLY_SCENARIO_BLOCKS",
    "EPSILON_DECAY_STEPS",
    "FORBIDDEN_FORMAL_SEEDS",
    "FORBIDDEN_STAGE7A1_SEEDS",
    "FORBIDDEN_STAGE7B_SEEDS",
    "FORBIDDEN_STAGE7C_Q1_SEEDS",
    "FORBIDDEN_STAGE8_PILOT_SEEDS",
    "GATE_ADJACENT_SUCCESS_DROP_MAX",
    "GATE_CHECKPOINTS",
    "GATE_COLLISION_MAX",
    "GATE_MATERIAL_REGRESSION",
    "GATE_MAX_LATE_COLLAPSE_SEEDS",
    "GATE_MAX_MATERIAL_REGRESSION_SEEDS",
    "GATE_MEAN_SUCCESS_MIN",
    "GATE_MIN_QUALIFIED_SEEDS",
    "GATE_SEED_SUCCESS_MIN",
    "GATE_SWAP_ELIGIBILITY_MIN",
    "GATE_TRUNCATION_MAX",
    "LATE_EVAL_MIN_CHECKPOINT",
    "LATE_SCENARIO_BLOCKS",
    "LEARNING_CURVE_CHECKPOINTS",
    "LEARNING_RATE_DECAY_STEPS",
    "LEARNING_RATE_END",
    "LEARNING_RATE_START",
    "MAX_STEPS",
    "PILOT_SEEDS",
    "POLICY_INTERVAL_SECONDS",
    "PROTOCOL_TAG",
    "PURPOSE",
    "RICH_LOG_CHECKPOINTS",
    "STAGE",
    "TARGET_UPDATE_MODE",
    "TRAINING_RUN_COUNT",
    "assert_stage8_gate_guards",
    "episodes_per_seed_checkpoint",
    "n_scenario_blocks",
    "target_mode",
]
