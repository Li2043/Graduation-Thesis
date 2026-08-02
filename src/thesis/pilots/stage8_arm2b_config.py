"""Stage 8 arm2b frozen constants and guards.

Stage 8 arm2b is a **single-variable training-stability fix**: linear
learning-rate decay (0.0005 -> 0.0001 over the full 100,000-step budget)
instead of arm0/arm1's constant learning rate.

Motivation: same as arm2a (see stage8_arm2a_config.py docstring) -- arm0 and
arm1 both showed large, non-monotonic seed-to-seed swings across checkpoints.
Arm2b tests the complementary training-stability lever to arm2a's soft
target update: a smaller, decaying step size in the back half of training,
on the hypothesis that a fixed 0.0005 LR causes overly large gradient steps
late in training when the Q-function should be fine-tuning rather than
still taking large steps. Arm2b changes exactly ONE variable relative to
arm0: `learning_rate_end=0.0001`, `learning_rate_decay_environment_steps
=100_000` (linear decay spanning the whole run). Nothing else -- not the
reward, not the epsilon schedule (still arm0's 50,000), not the target
update mode (still arm0's hard copy every 250 updates) -- changes relative
to arm0.

It makes NO competence-gate claim (no PASS/FAIL) -- same diagnostic-pilot
status as arm0/arm1/arm2a.

**2026-08-02 extension**: the original 2-seed pilot (65007, 65008) showed a
strikingly clean result relative to every other Stage 8 arm -- both seeds
reached 100% success / 0% collision / 0 frozen_stall at the 100K checkpoint,
with zero residual downstream_failure at 75K or 100K (versus 49/44/37
downstream_failure episodes total in arm0/arm1/arm2a respectively). Given
arm2a's soft-target-update showed the opposite pattern (a late-stage
collapse right at the 100K checkpoint for both its seeds), this result is
promising but not yet trustworthy at n=2 -- confirmed with the user that
before committing to a formal qualification-gate design, 6 additional seeds
(65009-65014) should be run under the *exact same* frozen arm2b protocol (no
parameter change) to check whether the clean result holds or whether arm2b
has its own hidden failure mode not visible in the first 2 seeds. This is
scale-up of an already-frozen configuration, not a new arm -- no parameter
in this file changes as part of this extension.
"""

from __future__ import annotations

from thesis.agents.dqn_bootstrap import DQNTargetMode

PROTOCOL_TAG = "stage8-arm2b-protocol-v1"
STAGE = "stage8_arm2b"
PURPOSE = "training_stability_fix_lr_decay"
ALGORITHM = "double_dqn"
CONDITION = "baseline"
BASE_REWARD_VERSION = "v2_active_time"

# Identical to Stage 7C-Q1 / arm0 / arm1 / arm2a -- reward is not varied in arm2b.
ACTIVE_TIME_COST_PER_STEP = 0.0005
ACTIVE_TIME_COST_PER_SECOND = 0.0025
POLICY_INTERVAL_SECONDS = 0.20

# Original pilot: 65007, 65008 (2026-08-01). Extended 2026-08-02 with
# 65009-65014 (6 more seeds) to validate robustness -- same frozen protocol,
# no parameter change.
PILOT_SEEDS: tuple[int, ...] = (65007, 65008, 65009, 65010, 65011, 65012, 65013, 65014)
STAGE8_RESERVED_SEED_BLOCK: tuple[int, ...] = tuple(range(65001, 65021))

# Forbidden seed blocks carried forward from every prior historical stage,
# plus arm0's (65001-65002), arm1's (65003-65004), and arm2a's (65005-65006).
FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORBIDDEN_STAGE7A1_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
FORBIDDEN_STAGE7B_SEEDS: tuple[int, ...] = tuple(range(63001, 63021))
FORBIDDEN_STAGE7C_Q1_SEEDS: tuple[int, ...] = tuple(range(64001, 64021))
FORBIDDEN_STAGE8_ARM0_SEEDS: tuple[int, ...] = (65001, 65002)
FORBIDDEN_STAGE8_ARM1_SEEDS: tuple[int, ...] = (65003, 65004)
FORBIDDEN_STAGE8_ARM2A_SEEDS: tuple[int, ...] = (65005, 65006)

MAX_STEPS = 100_000
CHECKPOINT_STEPS: tuple[int, ...] = (0, 25_000, 50_000, 75_000, 100_000)

EARLY_EVAL_MAX_CHECKPOINT = 175_000
EARLY_SCENARIO_BLOCKS = 8
ASSIGNMENTS_PER_BLOCK = 2

# Unchanged from arm0 -- arm2b does not touch the exploration schedule or
# target-update mode.
EPSILON_DECAY_STEPS = 50_000

# The single variable arm2b changes relative to arm0.
LEARNING_RATE_START = 0.0005
LEARNING_RATE_END = 0.0001
LEARNING_RATE_DECAY_STEPS = 100_000

TRAINING_RUN_COUNT = 2

# No competence gate for arm2b -- diagnostic pilot only, deliberately no GATE_* constants.


def n_scenario_blocks(checkpoint_step: int) -> int:
    if int(checkpoint_step) <= EARLY_EVAL_MAX_CHECKPOINT:
        return EARLY_SCENARIO_BLOCKS
    raise RuntimeError(
        "Stage 8 arm2b never evaluates beyond the early 8-block plan "
        f"(got checkpoint_step={checkpoint_step})"
    )


def episodes_per_seed_checkpoint(checkpoint_step: int) -> int:
    return n_scenario_blocks(checkpoint_step) * ASSIGNMENTS_PER_BLOCK


def assert_stage8_arm2b_guards(
    *,
    algorithm: str,
    condition: str,
    reward_shaping_enabled: bool,
    shaping_coefficient: float,
    master_seed: int,
    max_steps: int,
    active_time_cost_per_step: float,
    learning_rate_end: float,
    learning_rate_decay_environment_steps: int,
    allow_mean_pbrs: bool = False,
    allow_min_pbrs: bool = False,
) -> None:
    """Hard-fail on any deviation from the frozen arm2b protocol."""
    if algorithm != ALGORITHM:
        raise RuntimeError(f"algorithm must be {ALGORITHM!r}, got {algorithm!r}")
    if condition != CONDITION:
        raise RuntimeError(f"condition must be {CONDITION!r}, got {condition!r}")
    if reward_shaping_enabled or allow_mean_pbrs or allow_min_pbrs:
        raise RuntimeError("Mean-PBRS / Min-PBRS / shaping forbidden in Stage 8 arm2b")
    if float(shaping_coefficient) != 0.0:
        raise RuntimeError("shaping coefficient must be 0")
    if abs(float(active_time_cost_per_step) - ACTIVE_TIME_COST_PER_STEP) > 1e-15:
        raise RuntimeError(
            f"active_time_cost_per_step must be {ACTIVE_TIME_COST_PER_STEP}, "
            f"got {active_time_cost_per_step}"
        )
    if abs(float(learning_rate_end) - LEARNING_RATE_END) > 1e-15:
        raise RuntimeError(f"learning_rate_end must be {LEARNING_RATE_END}, got {learning_rate_end}")
    if int(learning_rate_decay_environment_steps) != LEARNING_RATE_DECAY_STEPS:
        raise RuntimeError(
            f"learning_rate_decay_environment_steps must be {LEARNING_RATE_DECAY_STEPS}, "
            f"got {learning_rate_decay_environment_steps}"
        )
    if master_seed not in PILOT_SEEDS:
        raise RuntimeError(f"master_seed {master_seed} not in frozen 65007-65014")
    if master_seed in FORBIDDEN_FORMAL_SEEDS:
        raise RuntimeError("formal seeds 61001-61010 forbidden")
    if master_seed in FORBIDDEN_STAGE7A1_SEEDS:
        raise RuntimeError("Stage 7A-1 seeds 62001-62020 forbidden")
    if master_seed in FORBIDDEN_STAGE7B_SEEDS:
        raise RuntimeError("Stage 7B seeds 63001-63020 forbidden")
    if master_seed in FORBIDDEN_STAGE7C_Q1_SEEDS:
        raise RuntimeError("Stage 7C-Q1 seeds 64001-64020 forbidden")
    if master_seed in FORBIDDEN_STAGE8_ARM0_SEEDS:
        raise RuntimeError("Stage 8 arm0 seeds 65001-65002 forbidden in arm2b")
    if master_seed in FORBIDDEN_STAGE8_ARM1_SEEDS:
        raise RuntimeError("Stage 8 arm1 seeds 65003-65004 forbidden in arm2b")
    if master_seed in FORBIDDEN_STAGE8_ARM2A_SEEDS:
        raise RuntimeError("Stage 8 arm2a seeds 65005-65006 forbidden in arm2b")
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
    "FORBIDDEN_STAGE8_ARM2A_SEEDS",
    "LEARNING_RATE_DECAY_STEPS",
    "LEARNING_RATE_END",
    "LEARNING_RATE_START",
    "MAX_STEPS",
    "PILOT_SEEDS",
    "POLICY_INTERVAL_SECONDS",
    "PROTOCOL_TAG",
    "PURPOSE",
    "STAGE",
    "STAGE8_RESERVED_SEED_BLOCK",
    "TRAINING_RUN_COUNT",
    "assert_stage8_arm2b_guards",
    "episodes_per_seed_checkpoint",
    "n_scenario_blocks",
    "target_mode",
]
