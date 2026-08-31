"""Final V3 reward-condition API (Stage 5A-0).

Authoritative conditions: baseline | mean_pbrs | min_pbrs.
PBRS lambda values here are integration-test-only and not final.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from thesis.rewards.pbrs_v2 import PBRSConfig

RewardConditionName = Literal["baseline", "mean_pbrs", "min_pbrs"]
FINAL_REWARD_CONDITIONS: tuple[RewardConditionName, ...] = (
    "baseline",
    "mean_pbrs",
    "min_pbrs",
)


@dataclass(frozen=True)
class IntegrationPBRSConfig:
    """Frozen test-only PBRS scale for Stage 5A-0 non-zero shaping regression."""

    learner_gamma: float = 0.995
    shaping_gamma: float = 0.995
    lambda_mean: float = 0.2
    lambda_min: float = 0.2
    integration_test_only: bool = True
    pbrs_parameters_final: bool = False

    def validate(self) -> None:
        if self.pbrs_parameters_final:
            raise ValueError("Stage 5A-0 must keep pbrs_parameters_final=false")
        if not self.integration_test_only:
            raise ValueError("Stage 5A-0 PBRS config must be integration_test_only")
        if abs(self.learner_gamma - self.shaping_gamma) > 1e-12:
            raise ValueError("shaping_gamma must equal learner_gamma")

    def to_pbrs_config(self) -> PBRSConfig:
        self.validate()
        cfg = PBRSConfig(
            learner_gamma=float(self.learner_gamma),
            shaping_gamma=float(self.shaping_gamma),
            lambda_mean=float(self.lambda_mean),
            lambda_min=float(self.lambda_min),
        )
        cfg.validate()
        return cfg

    def with_lambda_zero(self) -> "IntegrationPBRSConfig":
        return IntegrationPBRSConfig(
            learner_gamma=self.learner_gamma,
            shaping_gamma=self.shaping_gamma,
            lambda_mean=0.0,
            lambda_min=0.0,
            integration_test_only=True,
            pbrs_parameters_final=False,
        )


def select_condition_reward(
    *,
    condition: RewardConditionName,
    base_reward: float,
    scaled_mean_shaping: float,
    scaled_min_shaping: float,
) -> tuple[float, float]:
    """Return (learner_reward, shaping_component). Never rescales base."""
    if condition not in FINAL_REWARD_CONDITIONS:
        raise ValueError(f"unknown final reward condition {condition!r}")
    b = float(base_reward)
    if condition == "baseline":
        return b, 0.0
    if condition == "mean_pbrs":
        s = float(scaled_mean_shaping)
        return b + s, s
    s = float(scaled_min_shaping)
    return b + s, s


__all__ = [
    "FINAL_REWARD_CONDITIONS",
    "IntegrationPBRSConfig",
    "RewardConditionName",
    "select_condition_reward",
]
