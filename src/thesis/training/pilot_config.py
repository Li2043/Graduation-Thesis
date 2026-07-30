"""Frozen Stage 5B-0 bounded engineering pilot configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.training.final_reward_conditions import FINAL_REWARD_CONDITIONS, RewardConditionName


PILOT_SEEDS: tuple[int, ...] = (51001, 51002)
PILOT_CONDITIONS: tuple[RewardConditionName, ...] = FINAL_REWARD_CONDITIONS


@dataclass(frozen=True)
class PilotPBRSScale:
    lambda_mean: float = 0.2
    lambda_min: float = 0.2
    pilot_only: bool = True
    integration_test_only: bool = False
    pbrs_parameters_final: bool = False

    def validate(self) -> None:
        if self.pbrs_parameters_final:
            raise ValueError("pilot must keep pbrs_parameters_final=false")
        if not self.pilot_only:
            raise ValueError("pilot PBRS scale must be pilot_only")
        if self.integration_test_only:
            raise ValueError("pilot PBRS scale must set integration_test_only=false")


@dataclass(frozen=True)
class PilotDQNConfig:
    algorithm: str = "vanilla_independent_dqn"
    observation_dimension: int = OBSERVATION_DIM
    action_count: int = 3
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "ReLU"
    loss: str = "mse"
    gamma: float = 0.995
    learning_rate: float = 0.0005
    replay_capacity_per_controller: int = 20_000
    batch_size: int = 64
    replay_warmup_per_controller: int = 512
    update_frequency: str = "every_environment_policy_step"
    updates_per_active_controller: int = 1
    target_sync_type: str = "hard"
    target_sync_interval_updates: int = 250
    device: str = "cpu"


@dataclass(frozen=True)
class PilotExplorationConfig:
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_environment_steps: int = 4000
    schedule: str = "linear"
    epsilon_after_decay: float = 0.10


@dataclass(frozen=True)
class PilotDurationConfig:
    environment_steps_per_run: int = 5000
    maximum_runs: int = 6
    checkpoint_steps: tuple[int, ...] = (1000, 2500, 5000)
    evaluation_steps: tuple[int, ...] = (0, 2500, 5000)


@dataclass(frozen=True)
class PilotConfig:
    """Exact frozen pilot configuration (hash before training)."""

    conditions: tuple[RewardConditionName, ...] = PILOT_CONDITIONS
    pilot_seeds: tuple[int, ...] = PILOT_SEEDS
    dqn: PilotDQNConfig = field(default_factory=PilotDQNConfig)
    exploration: PilotExplorationConfig = field(default_factory=PilotExplorationConfig)
    duration: PilotDurationConfig = field(default_factory=PilotDurationConfig)
    pbrs: PilotPBRSScale = field(default_factory=PilotPBRSScale)
    pilot_configuration_final_for_run: bool = True
    training_protocol_final: bool = False
    formal_training_started: bool = False
    environment_parameters_final: bool = True
    comfort_parameters_final: bool = True

    def validate(self) -> None:
        if self.conditions != PILOT_CONDITIONS:
            raise ValueError("pilot must use exactly baseline/mean_pbrs/min_pbrs")
        if self.pilot_seeds != PILOT_SEEDS:
            raise ValueError("pilot must use exactly seeds 51001 and 51002")
        if len(self.conditions) * len(self.pilot_seeds) != self.duration.maximum_runs:
            raise ValueError("maximum_runs must equal 3 conditions × 2 seeds")
        if self.dqn.observation_dimension != OBSERVATION_DIM:
            raise ValueError("observation_dimension must be 27")
        if self.training_protocol_final:
            raise ValueError("training_protocol_final must remain false")
        if self.formal_training_started:
            raise ValueError("formal_training_started must remain false")
        self.pbrs.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def derive_run_seeds(pilot_seed: int) -> dict[str, int]:
    """Condition-independent seed derivation from a pilot seed."""
    s = int(pilot_seed)
    return {
        "environment": s + 100,
        "learner_A": s + 200,
        "learner_B": s + 300,
        "replay_A": s + 400,
        "replay_B": s + 401,
        "evaluation": s + 500,
        "ic_schedule": s + 600,
        "torch": s + 700,
    }


def epsilon_at_step(step: int, cfg: PilotExplorationConfig) -> float:
    """Linear epsilon schedule over environment steps (0-indexed after increment)."""
    t = max(0, int(step))
    decay = int(cfg.epsilon_decay_environment_steps)
    if t >= decay:
        return float(cfg.epsilon_after_decay)
    if decay <= 0:
        return float(cfg.epsilon_end)
    frac = t / float(decay)
    return float(cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start))


__all__ = [
    "PILOT_CONDITIONS",
    "PILOT_SEEDS",
    "PilotConfig",
    "PilotDQNConfig",
    "PilotDurationConfig",
    "PilotExplorationConfig",
    "PilotPBRSScale",
    "derive_run_seeds",
    "epsilon_at_step",
]
