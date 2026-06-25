"""Rawlsian reward (core contribution).

Implements the Rawlsian maximin scalar reward defined in
``docs/V1_SYSTEM_SPEC.md``:

    R_rawls = min_i E_i,

where E_i is the per-agent experience computed by the V1 experience function.
This reward is a transformation of the multi-agent state into a single scalar;
it does not change the learning algorithm.
"""

from __future__ import annotations

from typing import Any, Optional

from v1.experience.experience import (
    ExperienceFunction,
    compute_all_experiences,
    least_advantaged_agent,
    rawlsian_objective,
)
from v1.rewards.base_reward import EnvState, RewardFunction


class RawlsianReward(RewardFunction):
    """Return R_rawls = min_i E_i for the post-step multi-agent state."""

    def __init__(self, experience_function: Optional[ExperienceFunction] = None) -> None:
        self.experience_function = (
            experience_function if experience_function is not None else ExperienceFunction()
        )
        self.last_least_advantaged: Any = None
        self.last_experiences: dict = {}

    def compute(
        self,
        state: Any,
        next_state: Any,
        env_state: EnvState,
        prev_env_state: Optional[EnvState],
    ) -> float:
        experiences = compute_all_experiences(
            env_state, prev_env_state, self.experience_function
        )
        self.last_experiences = dict(experiences)
        self.last_least_advantaged = least_advantaged_agent(experiences)
        return rawlsian_objective(experiences)
