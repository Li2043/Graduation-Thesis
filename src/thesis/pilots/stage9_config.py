"""Stage 9 formal confirmatory three-condition experiment: frozen constants
and guards.

Supersedes Stage 6A (E16, 100K baseline/mean_pbrs/min_pbrs main result) --
see `experiments/formal/stage9_confirmatory_three_condition/docs/
STAGE9_CONFIRMATORY_PROTOCOL.md` for the full frozen protocol and the
reasoning recorded there (E16's baseline never reliably passed a competence
gate, had 0/10 swap-estimable blocks for mean_pbrs, and used a single
100K training endpoint with no learning curve).

Three conditions, only one of which trains in this stage:
  - baseline: REUSED VERBATIM from the Stage 8 formal qualification gate
    (`stage8-gate-protocol-v1`, seeds 65021-65040, FAIL). Not retrained --
    see the protocol doc SS4 for the justification. This module still
    records baseline's identity for provenance/guard purposes, but
    `stage9_runner.run_training_job` refuses to train it.
  - mean_pbrs: new training, seeds 66001-66020, lambda_mean=0.2
  - min_pbrs: new training, seeds 67001-67020, lambda_min=0.2

Shared DQN configuration is copied VERBATIM from `stage8_gate_config.py`
(Double DQN, hard target sync every 250 updates, LR linear decay
0.0005->0.0001 over the full 400,000-step budget, epsilon decay over 50,000
steps floored at 0.10) -- only the reward differs across conditions, per
the protocol's single-variable-across-conditions design.

lambda_mean=lambda_min=0.2 is an EQUAL-COEFFICIENT comparison, reused
verbatim from `stage5c0_h1_r1_100k_protocol`'s frozen
`final_pbrs_parameters.yaml` (comparison_type=equal_coefficient,
rms_matched=False) -- these are also `FormalPBRSConfig`'s own defaults, so
no override is needed, but the values are asserted explicitly here for
auditability per the protocol's governance SS9 item 5 ("do not silently
substitute an RMS-matched value").
"""

from __future__ import annotations

from thesis.agents.dqn_bootstrap import DQNTargetMode

PROTOCOL_TAG = "stage9-confirmatory-v1"
STAGE = "stage9_confirmatory"
PURPOSE = "formal_three_condition_confirmatory_experiment_supersedes_e16"
ALGORITHM = "double_dqn"
BASE_REWARD_VERSION = "v2_active_time"

CONDITIONS: tuple[str, ...] = ("baseline", "mean_pbrs", "min_pbrs")
TRAINED_CONDITIONS: tuple[str, ...] = ("mean_pbrs", "min_pbrs")  # baseline is reused, not retrained
REUSED_BASELINE_PROTOCOL_TAG = "stage8-gate-protocol-v1"

ACTIVE_TIME_COST_PER_STEP = 0.0005
ACTIVE_TIME_COST_PER_SECOND = 0.0025
POLICY_INTERVAL_SECONDS = 0.20

# Equal-coefficient PBRS, reused verbatim from stage5c0_h1_r1_100k_protocol
# (also FormalPBRSConfig's own defaults -- pinned here explicitly).
LAMBDA_MEAN = 0.2
LAMBDA_MIN = 0.2
PBRS_GAMMA = 0.995
PBRS_COMPARISON_TYPE = "equal_coefficient"
PBRS_RMS_MATCHED = False
PBRS_MAGNITUDE_MATCHED = False

# Per-condition seed blocks.
BASELINE_SEEDS: tuple[int, ...] = tuple(range(65021, 65041))  # reused, not retrained here
MEAN_PBRS_SEEDS: tuple[int, ...] = tuple(range(66001, 66021))
MIN_PBRS_SEEDS: tuple[int, ...] = tuple(range(67001, 67021))

SEEDS_BY_CONDITION: dict[str, tuple[int, ...]] = {
    "baseline": BASELINE_SEEDS,
    "mean_pbrs": MEAN_PBRS_SEEDS,
    "min_pbrs": MIN_PBRS_SEEDS,
}

# Every seed block used anywhere in this project so far, plus this stage's
# own two new blocks -- the full forbidden set for smoke-test seed picking.
FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORBIDDEN_STAGE7A1_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
FORBIDDEN_STAGE7B_SEEDS: tuple[int, ...] = tuple(range(63001, 63021))
FORBIDDEN_STAGE7C_Q1_SEEDS: tuple[int, ...] = tuple(range(64001, 64021))
FORBIDDEN_STAGE8_PILOT_SEEDS: tuple[int, ...] = tuple(range(65001, 65021))
FORBIDDEN_STAGE8_GATE_SEEDS: tuple[int, ...] = BASELINE_SEEDS  # 65021-65040

ALL_FORBIDDEN_SEEDS: tuple[int, ...] = (
    FORBIDDEN_FORMAL_SEEDS
    + FORBIDDEN_STAGE7A1_SEEDS
    + FORBIDDEN_STAGE7B_SEEDS
    + FORBIDDEN_STAGE7C_Q1_SEEDS
    + FORBIDDEN_STAGE8_PILOT_SEEDS
    + FORBIDDEN_STAGE8_GATE_SEEDS
)

MAX_STEPS = 400_000
CHECKPOINT_STEPS: tuple[int, ...] = tuple(range(0, 400_001, 25_000))
LEARNING_CURVE_CHECKPOINTS: tuple[int, ...] = tuple(range(200_000, 400_001, 25_000))
EARLY_EVAL_MAX_CHECKPOINT = 175_000
LATE_EVAL_MIN_CHECKPOINT = 200_000
EARLY_SCENARIO_BLOCKS = 8
LATE_SCENARIO_BLOCKS = 32
ASSIGNMENTS_PER_BLOCK = 2

# Same 4 rich-log checkpoints as the Stage 8 gate -- these are also the ones
# the RQ1-RQ3 / non-inferiority analysis reads (see protocol doc SS7/SS8).
RICH_LOG_CHECKPOINTS: tuple[int, ...] = (0, 350_000, 375_000, 400_000)
GATE_CHECKPOINTS: tuple[int, ...] = (350_000, 375_000, 400_000)  # kept for schema/eval-driver reuse only; no PASS/FAIL gate this stage

EPSILON_DECAY_STEPS = 50_000
TARGET_UPDATE_MODE = "hard"
LEARNING_RATE_START = 0.0005
LEARNING_RATE_END = 0.0001
LEARNING_RATE_DECAY_STEPS = 400_000

TRAINING_RUN_COUNT = len(MEAN_PBRS_SEEDS) + len(MIN_PBRS_SEEDS)  # 40 new training runs; baseline reused

# Non-inferiority margins (RQ3), FROZEN 2026-08-03 -- see protocol doc SS7.2
# for the derivation. Not statistical outputs: substantive judgements, set
# once, not re-derived from the comparison data itself.
MARGIN_COLLISION_RATE_ABS = 0.03
MARGIN_MEAN_MOBILITY_RELATIVE = 0.10
# Originally frozen at 0.05 (2026-08-03); CORRECTED same day after
# `run_stage9_decision.py`'s self-test (baseline compared against a copy of
# itself) surfaced that 0.05 is not achievable at n=20 -- q's real per-seed
# SD (pooled across the 3 gate checkpoints, computed from the Stage 8 gate's
# own 20 baseline seeds) is 0.163, giving a one-sided CI half-width of
# ~0.101, roughly double the original margin. A margin narrower than the
# achievable precision would return "non_inferior: false" close to by
# construction, even for a true difference of exactly zero. Widened to 0.12
# (slack above the ~0.101 half-width) so the test is actually resolvable at
# this design's fixed n=20, not merely arithmetically well-defined.
MARGIN_RESOLUTION_Q_ABS = 0.12

# Baseline anchor for the mobility margin, computed 2026-08-03 from the
# Stage 8 gate's own rich trajectories (A/B learners only -- see protocol
# doc SS7.2 note on scope). Recorded here as a frozen reference value, not
# recomputed from Stage 9's own data.
BASELINE_MEAN_LEARNER_MOBILITY_ANCHOR = 0.8968
BASELINE_MEAN_LEARNER_MOBILITY_FLOOR = BASELINE_MEAN_LEARNER_MOBILITY_ANCHOR * (
    1.0 - MARGIN_MEAN_MOBILITY_RELATIVE
)

# Detection-floor MES for RQ1/RQ2 difference-style claims at n=20/condition
# (protocol doc SS7.1) -- documentation only, not enforced by a guard.
MES_SUCCESS_RATE = 0.20
# Corrected 2026-08-03 alongside MARGIN_RESOLUTION_Q_ABS: using the
# per-seed D_swap SD actually observed in Stage 8 gate baseline data
# (0.269, vs the coarser pooled-episode estimate of 0.311 used when this
# section was first drafted), n=20 resolves MES~=0.238, not 0.20 -- rounded
# up to 0.25.
MES_SWAP_ELIGIBILITY = 0.25


def n_scenario_blocks(checkpoint_step: int) -> int:
    if int(checkpoint_step) <= EARLY_EVAL_MAX_CHECKPOINT:
        return EARLY_SCENARIO_BLOCKS
    return LATE_SCENARIO_BLOCKS


def episodes_per_seed_checkpoint(checkpoint_step: int) -> int:
    return n_scenario_blocks(checkpoint_step) * ASSIGNMENTS_PER_BLOCK


def assert_stage9_guards(
    *,
    algorithm: str,
    condition: str,
    master_seed: int,
    max_steps: int,
    active_time_cost_per_step: float,
    target_update_mode: str,
    learning_rate_end: float,
    learning_rate_decay_environment_steps: int,
    lambda_mean: float,
    lambda_min: float,
) -> None:
    """Hard-fail on any deviation from the frozen Stage 9 protocol."""
    if algorithm != ALGORITHM:
        raise RuntimeError(f"algorithm must be {ALGORITHM!r}, got {algorithm!r}")
    if condition not in TRAINED_CONDITIONS:
        raise RuntimeError(
            f"condition must be one of {TRAINED_CONDITIONS!r} (baseline is reused, "
            f"not trained by this runner), got {condition!r}"
        )
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
    if abs(float(lambda_mean) - LAMBDA_MEAN) > 1e-15:
        raise RuntimeError(f"lambda_mean must be {LAMBDA_MEAN}, got {lambda_mean}")
    if abs(float(lambda_min) - LAMBDA_MIN) > 1e-15:
        raise RuntimeError(f"lambda_min must be {LAMBDA_MIN}, got {lambda_min}")
    allowed_seeds = SEEDS_BY_CONDITION[condition]
    if master_seed not in allowed_seeds:
        raise RuntimeError(
            f"master_seed {master_seed} not in {condition}'s frozen seed block {allowed_seeds}"
        )
    if master_seed in ALL_FORBIDDEN_SEEDS:
        raise RuntimeError(f"master_seed {master_seed} collides with a historical seed block")
    if int(max_steps) != MAX_STEPS:
        raise RuntimeError(f"max_steps must be {MAX_STEPS}, got {max_steps}")


def target_mode() -> DQNTargetMode:
    return DQNTargetMode.DOUBLE


__all__ = [
    "ACTIVE_TIME_COST_PER_SECOND",
    "ACTIVE_TIME_COST_PER_STEP",
    "ALGORITHM",
    "ALL_FORBIDDEN_SEEDS",
    "ASSIGNMENTS_PER_BLOCK",
    "BASELINE_MEAN_LEARNER_MOBILITY_ANCHOR",
    "BASELINE_MEAN_LEARNER_MOBILITY_FLOOR",
    "BASELINE_SEEDS",
    "BASE_REWARD_VERSION",
    "CHECKPOINT_STEPS",
    "CONDITIONS",
    "EARLY_EVAL_MAX_CHECKPOINT",
    "EARLY_SCENARIO_BLOCKS",
    "EPSILON_DECAY_STEPS",
    "FORBIDDEN_FORMAL_SEEDS",
    "FORBIDDEN_STAGE7A1_SEEDS",
    "FORBIDDEN_STAGE7B_SEEDS",
    "FORBIDDEN_STAGE7C_Q1_SEEDS",
    "FORBIDDEN_STAGE8_GATE_SEEDS",
    "FORBIDDEN_STAGE8_PILOT_SEEDS",
    "GATE_CHECKPOINTS",
    "LAMBDA_MEAN",
    "LAMBDA_MIN",
    "LATE_EVAL_MIN_CHECKPOINT",
    "LATE_SCENARIO_BLOCKS",
    "LEARNING_CURVE_CHECKPOINTS",
    "LEARNING_RATE_DECAY_STEPS",
    "LEARNING_RATE_END",
    "LEARNING_RATE_START",
    "MARGIN_COLLISION_RATE_ABS",
    "MARGIN_MEAN_MOBILITY_RELATIVE",
    "MARGIN_RESOLUTION_Q_ABS",
    "MAX_STEPS",
    "MEAN_PBRS_SEEDS",
    "MES_SUCCESS_RATE",
    "MES_SWAP_ELIGIBILITY",
    "MIN_PBRS_SEEDS",
    "PBRS_COMPARISON_TYPE",
    "PBRS_GAMMA",
    "PBRS_MAGNITUDE_MATCHED",
    "PBRS_RMS_MATCHED",
    "POLICY_INTERVAL_SECONDS",
    "PROTOCOL_TAG",
    "PURPOSE",
    "REUSED_BASELINE_PROTOCOL_TAG",
    "RICH_LOG_CHECKPOINTS",
    "SEEDS_BY_CONDITION",
    "STAGE",
    "TARGET_UPDATE_MODE",
    "TRAINED_CONDITIONS",
    "TRAINING_RUN_COUNT",
    "assert_stage9_guards",
    "episodes_per_seed_checkpoint",
    "n_scenario_blocks",
    "target_mode",
]
