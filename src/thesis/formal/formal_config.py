"""Frozen Formal 100K configuration (Stage 6A-0 / H1-R1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.protocol.h1_r1_100k_protocol import (
    EPSILON_DECAY_STEPS,
    FORMAL_CHECKPOINT_STEPS,
    FORMAL_CONDITIONS,
    FORMAL_EVALUATION_STEPS,
    FORMAL_MASTER_SEEDS,
    FORMAL_STEPS_PER_RUN,
    GAMMA,
    LAMBDA_MEAN,
    LAMBDA_MIN,
    PROTOCOL_VERSION,
)
from thesis.training.final_reward_conditions import RewardConditionName


@dataclass(frozen=True)
class FormalDQNConfig:
    observation_dimension: int = OBSERVATION_DIM
    action_count: int = 3
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 0.0005
    gamma: float = GAMMA
    replay_capacity_per_controller: int = 20_000
    batch_size: int = 64
    replay_warmup_per_controller: int = 512
    target_sync_interval_updates: int = 250
    device: str = "cpu"
    # Stage 8 arm2a: soft (Polyak) target update instead of periodic hard copy.
    # Default "hard" preserves exact prior behaviour (Stage 6/7/8-arm0/8-arm1) --
    # target_sync_interval_updates keeps being used whenever mode == "hard".
    target_update_mode: str = "hard"  # "hard" | "soft"
    target_soft_tau: float = 0.0  # only meaningful when target_update_mode == "soft"
    # Stage 8 arm2b: linear learning-rate decay. None means constant `learning_rate`
    # for the whole run -- the default, preserving prior behaviour exactly.
    learning_rate_end: float | None = None
    learning_rate_decay_environment_steps: int | None = None


@dataclass(frozen=True)
class FormalExplorationConfig:
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_environment_steps: int = EPSILON_DECAY_STEPS
    schedule: str = "linear"
    epsilon_after_decay: float = 0.10


@dataclass(frozen=True)
class FormalDurationConfig:
    environment_steps_per_run: int = FORMAL_STEPS_PER_RUN
    checkpoint_steps: tuple[int, ...] = FORMAL_CHECKPOINT_STEPS
    evaluation_steps: tuple[int, ...] = FORMAL_EVALUATION_STEPS
    early_stopping: bool = False
    max_policy_steps: int = 400


@dataclass(frozen=True)
class FormalPBRSConfig:
    lambda_mean: float = LAMBDA_MEAN
    lambda_min: float = LAMBDA_MIN
    gamma: float = GAMMA
    comparison_type: str = "equal_coefficient"
    magnitude_matched: bool = False
    rms_matched: bool = False


@dataclass(frozen=True)
class FormalConfig:
    protocol_version: str = PROTOCOL_VERSION
    conditions: tuple[str, ...] = FORMAL_CONDITIONS
    master_seeds: tuple[int, ...] = FORMAL_MASTER_SEEDS
    dqn: FormalDQNConfig = field(default_factory=FormalDQNConfig)
    exploration: FormalExplorationConfig = field(default_factory=FormalExplorationConfig)
    duration: FormalDurationConfig = field(default_factory=FormalDurationConfig)
    pbrs: FormalPBRSConfig = field(default_factory=FormalPBRSConfig)
    num_parallel_training_envs_per_run: int = 1
    vectorized_training: bool = False
    formal_training_started: bool = False
    training_protocol_final: bool = True
    pbrs_parameters_final: bool = True
    # When True, allow shortened step budgets for infrastructure unit tests only.
    allow_test_budget: bool = False

    def validate(self) -> None:
        if self.num_parallel_training_envs_per_run != 1:
            raise ValueError("exactly one training env per run")
        if self.vectorized_training:
            raise ValueError("vectorized training forbidden")
        if self.duration.early_stopping:
            raise ValueError("early stopping forbidden")
        if self.dqn.observation_dimension != OBSERVATION_DIM:
            raise ValueError("observation_dimension must be 27")
        if self.dqn.hidden_sizes != (64, 64):
            raise ValueError("hidden_sizes must be [64, 64]")
        if self.dqn.target_update_mode not in ("hard", "soft"):
            raise ValueError("target_update_mode must be 'hard' or 'soft'")
        if self.dqn.target_update_mode == "soft" and not (0.0 < self.dqn.target_soft_tau <= 1.0):
            raise ValueError("target_soft_tau must be in (0, 1] when target_update_mode == 'soft'")
        if self.dqn.target_update_mode == "hard" and self.dqn.target_soft_tau != 0.0:
            raise ValueError("target_soft_tau must be 0 when target_update_mode == 'hard'")
        if self.dqn.learning_rate_end is not None:
            if not (0.0 < self.dqn.learning_rate_end <= self.dqn.learning_rate):
                raise ValueError("learning_rate_end must be in (0, learning_rate]")
            if not self.dqn.learning_rate_decay_environment_steps or self.dqn.learning_rate_decay_environment_steps <= 0:
                raise ValueError("learning_rate_decay_environment_steps must be positive when learning_rate_end is set")
        n = len(self.conditions) * len(self.master_seeds)
        if n != 30:
            raise ValueError("expected 30 formal run slots")
        if self.allow_test_budget:
            if self.duration.environment_steps_per_run <= 0:
                raise ValueError("test budget steps must be positive")
            return
        if self.duration.environment_steps_per_run != FORMAL_STEPS_PER_RUN:
            raise ValueError("formal steps must be 100000")
        if self.exploration.epsilon_decay_environment_steps != EPSILON_DECAY_STEPS:
            raise ValueError("epsilon decay must be 50000")
        if n * self.duration.environment_steps_per_run != 3_000_000:
            raise ValueError("total planned steps must be 3000000")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def derive_formal_job_seeds(master_seed: int) -> dict[str, int]:
    s = int(master_seed)
    return {
        "master_seed": s,
        "environment_seed": s,
        "learner_A_seed": s + 100_000,
        "learner_B_seed": s + 200_000,
        "replay_A_seed": s + 300_000,
        "replay_B_seed": s + 400_000,
        "evaluation_seed": s + 500_000,
        "schedule_seed": s + 600_000,
    }


def epsilon_at_step(step: int, cfg: FormalExplorationConfig) -> float:
    t = max(0, int(step))
    decay = int(cfg.epsilon_decay_environment_steps)
    if t >= decay:
        return float(cfg.epsilon_after_decay)
    if decay <= 0:
        return float(cfg.epsilon_end)
    frac = t / float(decay)
    return float(cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start))


def lr_at_step(step: int, cfg: FormalDQNConfig) -> float:
    """Linear learning-rate decay, mirroring epsilon_at_step's shape.

    Returns the constant `cfg.learning_rate` whenever `learning_rate_end` is
    None (the default for every stage prior to Stage 8 arm2b) -- callers may
    invoke this unconditionally every step without changing behaviour for
    runs that do not opt into LR decay.
    """
    if cfg.learning_rate_end is None:
        return float(cfg.learning_rate)
    t = max(0, int(step))
    decay = int(cfg.learning_rate_decay_environment_steps)
    if t >= decay:
        return float(cfg.learning_rate_end)
    frac = t / float(decay)
    return float(cfg.learning_rate + frac * (cfg.learning_rate_end - cfg.learning_rate))


def assert_condition(name: str) -> RewardConditionName:
    if name not in FORMAL_CONDITIONS:
        raise ValueError(f"unknown formal condition {name!r}")
    return name  # type: ignore[return-value]


__all__ = [
    "FormalConfig",
    "FormalDQNConfig",
    "FormalDurationConfig",
    "FormalExplorationConfig",
    "FormalPBRSConfig",
    "assert_condition",
    "derive_formal_job_seeds",
    "lr_at_step",
    "epsilon_at_step",
]
