"""Stage 8 arm2a frozen constants and guards.

Stage 8 arm2a is a **single-variable training-stability fix**: soft (Polyak)
target-network updates replacing the hard periodic copy used in arm0/arm1.

Motivation: arm0 and arm1 both showed large, non-monotonic seed-to-seed
swings in downstream_failure / success_rate across checkpoints (e.g. arm0
seed 65001: 56%->12.5%->37.5%->100% success across 25K/50K/75K/100K; arm1
seed 65004: 18.75%->6.25%->18.75%->68.75%), consistent with the classic
"hard target-network copy causes late-training instability" mechanism
already flagged in freeze doc SS10.3 (Double DQN's late-collapse seeds
increased relative to Vanilla). Arm2a changes exactly ONE variable relative
to arm0: `target_update_mode` "hard" -> "soft" with `target_soft_tau=0.005`
(applied every learner update instead of a full copy every 250 updates).
Nothing else -- not the reward, not the epsilon schedule (still arm0's
50,000), not the learning rate -- changes relative to arm0.

It makes NO competence-gate claim (no PASS/FAIL) -- same diagnostic-pilot
status as arm0/arm1.
"""

from __future__ import annotations

from thesis.agents.dqn_bootstrap import DQNTargetMode

PROTOCOL_TAG = "stage8-arm2a-protocol-v1"
STAGE = "stage8_arm2a"
PURPOSE = "training_stability_fix_soft_target_update"
ALGORITHM = "double_dqn"
CONDITION = "baseline"
BASE_REWARD_VERSION = "v2_active_time"

# Identical to Stage 7C-Q1 / arm0 / arm1 -- reward is not varied in arm2a.
ACTIVE_TIME_COST_PER_STEP = 0.0005
ACTIVE_TIME_COST_PER_SECOND = 0.0025
POLICY_INTERVAL_SECONDS = 0.20

PILOT_SEEDS: tuple[int, ...] = (65005, 65006)
STAGE8_RESERVED_SEED_BLOCK: tuple[int, ...] = tuple(range(65001, 65021))

# Forbidden seed blocks carried forward from every prior historical stage,
# plus arm0's (65001-65002) and arm1's (65003-65004) own seeds.
FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORBIDDEN_STAGE7A1_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
FORBIDDEN_STAGE7B_SEEDS: tuple[int, ...] = tuple(range(63001, 63021))
FORBIDDEN_STAGE7C_Q1_SEEDS: tuple[int, ...] = tuple(range(64001, 64021))
FORBIDDEN_STAGE8_ARM0_SEEDS: tuple[int, ...] = (65001, 65002)
FORBIDDEN_STAGE8_ARM1_SEEDS: tuple[int, ...] = (65003, 65004)

MAX_STEPS = 100_000
CHECKPOINT_STEPS: tuple[int, ...] = (0, 25_000, 50_000, 75_000, 100_000)

EARLY_EVAL_MAX_CHECKPOINT = 175_000
EARLY_SCENARIO_BLOCKS = 8
ASSIGNMENTS_PER_BLOCK = 2

# Unchanged from arm0 -- arm2a does not touch the exploration schedule.
EPSILON_DECAY_STEPS = 50_000

# The single variable arm2a changes relative to arm0.
TARGET_UPDATE_MODE = "soft"
TARGET_SOFT_TAU = 0.005

TRAINING_RUN_COUNT = 2

# No competence gate for arm2a -- diagnostic pilot only, deliberately no GATE_* constants.


def n_scenario_blocks(checkpoint_step: int) -> int:
    if int(checkpoint_step) <= EARLY_EVAL_MAX_CHECKPOINT:
        return EARLY_SCENARIO_BLOCKS
    raise RuntimeError(
        "Stage 8 arm2a never evaluates beyond the early 8-block plan "
        f"(got checkpoint_step={checkpoint_step})"
    )


def episodes_per_seed_checkpoint(checkpoint_step: int) -> int:
    return n_scenario_blocks(checkpoint_step) * ASSIGNMENTS_PER_BLOCK


def assert_stage8_arm2a_guards(
    *,
    algorithm: str,
    condition: str,
    reward_shaping_enabled: bool,
    shaping_coefficient: float,
    master_seed: int,
    max_steps: int,
    active_time_cost_per_step: float,
    target_update_mode: str,
    target_soft_tau: float,
    allow_mean_pbrs: bool = False,
    allow_min_pbrs: bool = False,
) -> None:
    """Hard-fail on any deviation from the frozen arm2a protocol."""
    if algorithm != ALGORITHM:
        raise RuntimeError(f"algorithm must be {ALGORITHM!r}, got {algorithm!r}")
    if condition != CONDITION:
        raise RuntimeError(f"condition must be {CONDITION!r}, got {condition!r}")
    if reward_shaping_enabled or allow_mean_pbrs or allow_min_pbrs:
        raise RuntimeError("Mean-PBRS / Min-PBRS / shaping forbidden in Stage 8 arm2a")
    if float(shaping_coefficient) != 0.0:
        raise RuntimeError("shaping coefficient must be 0")
    if abs(float(active_time_cost_per_step) - ACTIVE_TIME_COST_PER_STEP) > 1e-15:
        raise RuntimeError(
            f"active_time_cost_per_step must be {ACTIVE_TIME_COST_PER_STEP}, "
            f"got {active_time_cost_per_step}"
        )
    if target_update_mode != TARGET_UPDATE_MODE:
        raise RuntimeError(f"target_update_mode must be {TARGET_UPDATE_MODE!r}, got {target_update_mode!r}")
    if abs(float(target_soft_tau) - TARGET_SOFT_TAU) > 1e-15:
        raise RuntimeError(f"target_soft_tau must be {TARGET_SOFT_TAU}, got {target_soft_tau}")
    if master_seed not in PILOT_SEEDS:
        raise RuntimeError(f"master_seed {master_seed} not in frozen 65005-65006")
    if master_seed in FORBIDDEN_FORMAL_SEEDS:
        raise RuntimeError("formal seeds 61001-61010 forbidden")
    if master_seed in FORBIDDEN_STAGE7A1_SEEDS:
        raise RuntimeError("Stage 7A-1 seeds 62001-62020 forbidden")
    if master_seed in FORBIDDEN_STAGE7B_SEEDS:
        raise RuntimeError("Stage 7B seeds 63001-63020 forbidden")
    if master_seed in FORBIDDEN_STAGE7C_Q1_SEEDS:
        raise RuntimeError("Stage 7C-Q1 seeds 64001-64020 forbidden")
    if master_seed in FORBIDDEN_STAGE8_ARM0_SEEDS:
        raise RuntimeError("Stage 8 arm0 seeds 65001-65002 forbidden in arm2a")
    if master_seed in FORBIDDEN_STAGE8_ARM1_SEEDS:
        raise RuntimeError("Stage 8 arm1 seeds 65003-65004 forbidden in arm2a")
    if int(max_steps) != MAX_STEPS:
        raise RuntimeError(f"max_steps must be {MAX_STEPS}, got {max_steps}")


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
    "FORBIDDEN_STAGE8_ARM0_SEEDS",
    "FORBIDDEN_STAGE8_ARM1_SEEDS",
    "MAX_STEPS",
    "PILOT_SEEDS",
    "POLICY_INTERVAL_SECONDS",
    "PROTOCOL_TAG",
    "PURPOSE",
    "STAGE",
    "STAGE8_RESERVED_SEED_BLOCK",
    "TARGET_SOFT_TAU",
    "TARGET_UPDATE_MODE",
    "TRAINING_RUN_COUNT",
    "assert_stage8_arm2a_guards",
    "episodes_per_seed_checkpoint",
    "n_scenario_blocks",
    "target_mode",
]
