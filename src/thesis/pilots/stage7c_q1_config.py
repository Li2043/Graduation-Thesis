"""Stage 7C-Q1 frozen constants and guards."""

from __future__ import annotations

from thesis.agents.dqn_bootstrap import DQNTargetMode

PROTOCOL_TAG = "stage7c-q1-protocol-v1"
STAGE = "stage7c_q1"
PURPOSE = "baseline_competence_qualification"
ALGORITHM = "double_dqn"
CONDITION = "baseline"
BASE_REWARD_VERSION = "v2_active_time"

ACTIVE_TIME_COST_PER_STEP = 0.0005
ACTIVE_TIME_COST_PER_SECOND = 0.0025
POLICY_INTERVAL_SECONDS = 0.20

PILOT_SEEDS: tuple[int, ...] = tuple(range(64001, 64021))
FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORBIDDEN_STAGE7A1_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
FORBIDDEN_STAGE7B_SEEDS: tuple[int, ...] = tuple(range(63001, 63021))

MAX_STEPS = 400_000
CHECKPOINT_STEPS: tuple[int, ...] = tuple(range(0, 400_001, 25_000))
GATE_CHECKPOINTS: tuple[int, ...] = (350_000, 375_000, 400_000)
LEARNING_CURVE_CHECKPOINTS: tuple[int, ...] = tuple(range(200_000, 400_001, 25_000))
EARLY_EVAL_MAX_CHECKPOINT = 175_000
LATE_EVAL_MIN_CHECKPOINT = 200_000
EARLY_SCENARIO_BLOCKS = 8
LATE_SCENARIO_BLOCKS = 32
ASSIGNMENTS_PER_BLOCK = 2
EPSILON_DECAY_STEPS = 50_000
TRAINING_RUN_COUNT = 20

# Competence gate (frozen)
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
    if int(checkpoint_step) <= EARLY_EVAL_MAX_CHECKPOINT:
        return EARLY_SCENARIO_BLOCKS
    return LATE_SCENARIO_BLOCKS


def episodes_per_seed_checkpoint(checkpoint_step: int) -> int:
    return n_scenario_blocks(checkpoint_step) * ASSIGNMENTS_PER_BLOCK


def assert_stage7c_q1_guards(
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
    if algorithm != ALGORITHM:
        raise RuntimeError(f"algorithm must be {ALGORITHM!r}, got {algorithm!r}")
    if condition != CONDITION:
        raise RuntimeError(f"condition must be {CONDITION!r}, got {condition!r}")
    if reward_shaping_enabled or allow_mean_pbrs or allow_min_pbrs:
        raise RuntimeError("Mean-PBRS / Min-PBRS / shaping forbidden in Stage 7C-Q1")
    if float(shaping_coefficient) != 0.0:
        raise RuntimeError("shaping coefficient must be 0")
    if abs(float(active_time_cost_per_step) - ACTIVE_TIME_COST_PER_STEP) > 1e-15:
        raise RuntimeError(
            f"active_time_cost_per_step must be {ACTIVE_TIME_COST_PER_STEP}, "
            f"got {active_time_cost_per_step}"
        )
    if master_seed not in PILOT_SEEDS:
        raise RuntimeError(f"master_seed {master_seed} not in frozen 64001-64020")
    if master_seed in FORBIDDEN_FORMAL_SEEDS:
        raise RuntimeError("formal seeds 61001-61010 forbidden")
    if master_seed in FORBIDDEN_STAGE7A1_SEEDS:
        raise RuntimeError("Stage 7A-1 seeds 62001-62020 forbidden")
    if master_seed in FORBIDDEN_STAGE7B_SEEDS:
        raise RuntimeError("Stage 7B seeds 63001-63020 forbidden")
    if int(max_steps) != MAX_STEPS:
        raise RuntimeError(f"max_steps must be {MAX_STEPS}, got {max_steps}")
    if int(max_steps) > MAX_STEPS:
        raise RuntimeError("training beyond 400K forbidden")


def target_mode() -> DQNTargetMode:
    return DQNTargetMode.DOUBLE


def late_collapse_7c(
    success_by_checkpoint: dict[int, float],
    *,
    prior_checkpoints: tuple[int, ...] = (350_000, 375_000),
    prior_min: float = 0.95,
    final_max: float = 0.75,
    final_checkpoint: int = 400_000,
) -> bool:
    """Stage 7C late-collapse: ≥0.95 at 350K or 375K, then later <0.75."""
    if float(success_by_checkpoint.get(final_checkpoint, 1.0)) >= final_max:
        # also check any post-prior checkpoint under final_max
        later = [c for c in success_by_checkpoint if c > max(prior_checkpoints)]
        if not later:
            return False
        if all(float(success_by_checkpoint[c]) >= final_max for c in later):
            return False
    peaked = any(
        float(success_by_checkpoint.get(p, 0.0)) >= prior_min for p in prior_checkpoints
    )
    if not peaked:
        return False
    # Any subsequent checkpoint below final_max
    for c, v in success_by_checkpoint.items():
        if int(c) > max(prior_checkpoints) and float(v) < final_max:
            return True
    return False


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
    "MAX_STEPS",
    "PILOT_SEEDS",
    "POLICY_INTERVAL_SECONDS",
    "PROTOCOL_TAG",
    "PURPOSE",
    "STAGE",
    "TRAINING_RUN_COUNT",
    "assert_stage7c_q1_guards",
    "episodes_per_seed_checkpoint",
    "late_collapse_7c",
    "n_scenario_blocks",
    "target_mode",
]
