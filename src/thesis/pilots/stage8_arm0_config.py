"""Stage 8 arm0 frozen constants and guards.

Stage 8 arm0 is a **diagnostic reproduction** of the Stage 7C-Q1 baseline
configuration (Double DQN, Base Reward V2 active-time cost, condition=baseline)
at reduced scale, with per-step trajectory logging enabled from the start.

It makes NO competence-gate claim (no PASS/FAIL) and changes NO reward or
algorithm parameter relative to Stage 7C-Q1. Its only purpose is to confirm or
refute, via per-step evidence, the hypothesis that the `downstream_failure`
residual failure mode is caused by the post-peer-exit survivor getting stuck
in a prolonged brake/accelerate oscillation near a background vehicle.
"""

from __future__ import annotations

from thesis.agents.dqn_bootstrap import DQNTargetMode

PROTOCOL_TAG = "stage8-arm0-protocol-v1"
STAGE = "stage8_arm0"
PURPOSE = "diagnostic_reproduction_downstream_failure_mechanism"
ALGORITHM = "double_dqn"
CONDITION = "baseline"
BASE_REWARD_VERSION = "v2_active_time"

# Identical to Stage 7C-Q1 -- this is an exact reproduction, not a variant.
ACTIVE_TIME_COST_PER_STEP = 0.0005
ACTIVE_TIME_COST_PER_SECOND = 0.0025
POLICY_INTERVAL_SECONDS = 0.20

PILOT_SEEDS: tuple[int, ...] = (65001, 65002)
STAGE8_RESERVED_SEED_BLOCK: tuple[int, ...] = tuple(range(65001, 65021))

# Forbidden seed blocks carried forward from every prior historical stage.
FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORBIDDEN_STAGE7A1_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
FORBIDDEN_STAGE7B_SEEDS: tuple[int, ...] = tuple(range(63001, 63021))
FORBIDDEN_STAGE7C_Q1_SEEDS: tuple[int, ...] = tuple(range(64001, 64021))

MAX_STEPS = 100_000
CHECKPOINT_STEPS: tuple[int, ...] = (0, 25_000, 50_000, 75_000, 100_000)

# All arm0 checkpoints are <= EARLY_EVAL_MAX_CHECKPOINT, so evaluation always
# uses the 8-block / 16-episode "early" plan -- carried unchanged from
# Stage 7C-Q1 so the reused n_scenario_blocks()/episodes_per_seed_checkpoint()
# helpers behave identically.
EARLY_EVAL_MAX_CHECKPOINT = 175_000
EARLY_SCENARIO_BLOCKS = 8
ASSIGNMENTS_PER_BLOCK = 2
EPSILON_DECAY_STEPS = 50_000
TRAINING_RUN_COUNT = 2

# No competence gate for arm0 -- diagnostic only, deliberately no GATE_* constants.


def n_scenario_blocks(checkpoint_step: int) -> int:
    """Re-exported for callers that only import from this module.

    Numerically identical to thesis.pilots.stage7c_q1_config.n_scenario_blocks
    for every checkpoint arm0 actually uses (all <= 175_000).
    """
    if int(checkpoint_step) <= EARLY_EVAL_MAX_CHECKPOINT:
        return EARLY_SCENARIO_BLOCKS
    raise RuntimeError(
        "Stage 8 arm0 never evaluates beyond the early 8-block plan "
        f"(got checkpoint_step={checkpoint_step})"
    )


def episodes_per_seed_checkpoint(checkpoint_step: int) -> int:
    return n_scenario_blocks(checkpoint_step) * ASSIGNMENTS_PER_BLOCK


def assert_stage8_arm0_guards(
    *,
    algorithm: str,
    condition: str,
    reward_shaping_enabled: bool,
    shaping_coefficient: float,
    master_seed: int,
    max_steps: int,
    active_time_cost_per_step: float,
    allow_mean_pbrs: bool = False,
    allow_min_pbrs: bool = False,
) -> None:
    """Hard-fail on any deviation from the frozen arm0 diagnostic protocol."""
    if algorithm != ALGORITHM:
        raise RuntimeError(f"algorithm must be {ALGORITHM!r}, got {algorithm!r}")
    if condition != CONDITION:
        raise RuntimeError(f"condition must be {CONDITION!r}, got {condition!r}")
    if reward_shaping_enabled or allow_mean_pbrs or allow_min_pbrs:
        raise RuntimeError("Mean-PBRS / Min-PBRS / shaping forbidden in Stage 8 arm0")
    if float(shaping_coefficient) != 0.0:
        raise RuntimeError("shaping coefficient must be 0")
    if abs(float(active_time_cost_per_step) - ACTIVE_TIME_COST_PER_STEP) > 1e-15:
        raise RuntimeError(
            f"active_time_cost_per_step must be {ACTIVE_TIME_COST_PER_STEP}, "
            f"got {active_time_cost_per_step}"
        )
    if master_seed not in PILOT_SEEDS:
        raise RuntimeError(f"master_seed {master_seed} not in frozen 65001-65002")
    if master_seed in FORBIDDEN_FORMAL_SEEDS:
        raise RuntimeError("formal seeds 61001-61010 forbidden")
    if master_seed in FORBIDDEN_STAGE7A1_SEEDS:
        raise RuntimeError("Stage 7A-1 seeds 62001-62020 forbidden")
    if master_seed in FORBIDDEN_STAGE7B_SEEDS:
        raise RuntimeError("Stage 7B seeds 63001-63020 forbidden")
    if master_seed in FORBIDDEN_STAGE7C_Q1_SEEDS:
        raise RuntimeError("Stage 7C-Q1 seeds 64001-64020 forbidden")
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
    "MAX_STEPS",
    "PILOT_SEEDS",
    "POLICY_INTERVAL_SECONDS",
    "PROTOCOL_TAG",
    "PURPOSE",
    "STAGE",
    "STAGE8_RESERVED_SEED_BLOCK",
    "TRAINING_RUN_COUNT",
    "assert_stage8_arm0_guards",
    "episodes_per_seed_checkpoint",
    "n_scenario_blocks",
    "target_mode",
]
