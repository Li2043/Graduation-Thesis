"""Rawlsian reward (core contribution).

Implements the Rawlsian maximin scalar reward defined in
``docs/V1_SYSTEM_SPEC.md`` plus the shared merge-task term:

    R_rawls = min_i E_i + merge_task_bonus_or_penalty

where E_i is the per-agent experience computed by the V1 experience function.
The per-step ``compute`` returns the unchanged maximin scalar ``min_i E_i``; the
merge-task term is the shared ``terminal_adjustment`` inherited from
``RewardFunction`` (identical to the Egoistic condition) and is applied by the
training/evaluation loop at the terminal/merge step.

The experience function, ``rawlsian_objective``, and the least-advantaged-agent
logic are NOT modified: the task term is an external task constraint added to
the Rawlsian scalar reward, not part of E_i.
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
from v1.rewards.merge_task_reward import MergeTaskConfig


class RawlsianReward(RewardFunction):
    """Return R_rawls = min_i E_i for the post-step multi-agent state.

    The shared terminal merge-task term is added separately by the loop via the
    inherited ``terminal_adjustment``; ``compute`` itself returns only min_i E_i.
    """

    def __init__(
        self,
        experience_function: Optional[ExperienceFunction] = None,
        merge_task_config: Optional[MergeTaskConfig] = None,
    ) -> None:
        self.experience_function = (
            experience_function if experience_function is not None else ExperienceFunction()
        )
        self.last_least_advantaged: Any = None
        self.last_experiences: dict = {}
        # Shared terminal merge-task term (see RewardFunction.terminal_adjustment).
        self.merge_task_config = merge_task_config or MergeTaskConfig()

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
