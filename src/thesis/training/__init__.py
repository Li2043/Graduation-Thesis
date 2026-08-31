"""Final V3 training-facing integration package (Stage 5A-0).

Does not start pilot / sustained policy training.
"""

from thesis.training.final_lock_loader import (
    EXPECTED_COMFORT_LOCK_SHA256,
    EXPECTED_ENVIRONMENT_LOCK_SHA256,
    FinalLockBundle,
    load_final_locks,
)
from thesis.training.final_reward_conditions import (
    FINAL_REWARD_CONDITIONS,
    IntegrationPBRSConfig,
    RewardConditionName,
)

__all__ = [
    "EXPECTED_COMFORT_LOCK_SHA256",
    "EXPECTED_ENVIRONMENT_LOCK_SHA256",
    "FINAL_REWARD_CONDITIONS",
    "FinalLockBundle",
    "IntegrationPBRSConfig",
    "RewardConditionName",
    "load_final_locks",
]
