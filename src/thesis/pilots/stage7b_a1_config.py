"""Stage 7B-A1 constants and guards."""

from __future__ import annotations

from typing import Any

from thesis.agents.dqn_bootstrap import DQNTargetMode

PILOT_SEEDS: tuple[int, ...] = tuple(range(63001, 63021))
FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORBIDDEN_STAGE7A1_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
CONDITIONS: tuple[str, ...] = ("vanilla_dqn", "double_dqn")
MAX_STEPS = 300_000
CHECKPOINT_STEPS: tuple[int, ...] = (
    0,
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
    150_000,
    200_000,
    250_000,
    300_000,
)
PRIMARY_STABILITY_CHECKPOINTS: tuple[int, ...] = (
    100_000,
    150_000,
    200_000,
    250_000,
    300_000,
)
EPSILON_DECAY_STEPS = 50_000
TRAINING_RUN_COUNT = 40
EXPECTED_EVAL_EPISODES = 6400


def condition_to_target_mode(condition: str) -> DQNTargetMode:
    if condition == "vanilla_dqn":
        return DQNTargetMode.VANILLA
    if condition == "double_dqn":
        return DQNTargetMode.DOUBLE
    raise ValueError(f"unknown Stage 7B-A1 condition {condition!r}")


def assert_stage7b_guards(
    *,
    condition: str,
    reward_shaping_enabled: bool,
    shaping_coefficient: float,
    master_seed: int,
    max_steps: int,
) -> None:
    if condition not in CONDITIONS:
        raise RuntimeError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    if reward_shaping_enabled:
        raise RuntimeError("reward shaping forbidden")
    if float(shaping_coefficient) != 0.0:
        raise RuntimeError("shaping coefficient must be 0")
    if master_seed not in PILOT_SEEDS:
        raise RuntimeError(f"master_seed {master_seed} not in frozen 63001-63020")
    if master_seed in FORBIDDEN_FORMAL_SEEDS:
        raise RuntimeError("formal seeds 61001-61010 forbidden")
    if master_seed in FORBIDDEN_STAGE7A1_SEEDS:
        raise RuntimeError("Stage 7A-1 seeds 62001-62020 forbidden")
    if int(max_steps) != MAX_STEPS:
        raise RuntimeError(f"max_steps must be {MAX_STEPS}, got {max_steps}")


def late_collapse(
    success_by_checkpoint: dict[int, float],
    *,
    prior_min: float = 0.75,
    final_max: float = 0.50,
    prior_checkpoints: tuple[int, ...] = (200_000, 250_000),
    final_checkpoint: int = 300_000,
) -> bool:
    final = float(success_by_checkpoint.get(final_checkpoint, 1.0))
    if final > final_max:
        return False
    return any(
        float(success_by_checkpoint.get(p, 0.0)) >= prior_min for p in prior_checkpoints
    )


__all__ = [
    "CHECKPOINT_STEPS",
    "CONDITIONS",
    "EPSILON_DECAY_STEPS",
    "EXPECTED_EVAL_EPISODES",
    "FORBIDDEN_FORMAL_SEEDS",
    "FORBIDDEN_STAGE7A1_SEEDS",
    "MAX_STEPS",
    "PILOT_SEEDS",
    "PRIMARY_STABILITY_CHECKPOINTS",
    "TRAINING_RUN_COUNT",
    "assert_stage7b_guards",
    "condition_to_target_mode",
    "late_collapse",
]
