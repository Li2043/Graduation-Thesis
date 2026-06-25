"""Rawlsian reward (core contribution).

Implements the Rawlsian maximin scalar reward defined in
``docs/V1_SYSTEM_SPEC.md`` plus the shared task/safety terms:

    R_rawls = rawlsian_objective_scale * min_i E_i
              + merge_task_bonus_or_penalty
              - terminal_collision_penalty_if_collision

where E_i is the per-agent experience computed by the V1 experience function.
The per-step ``compute`` returns ``rawlsian_objective_scale * min_i E_i``; the
merge-task term and the shared terminal collision penalty are the inherited
``terminal_adjustment`` / ``terminal_collision_adjustment`` from
``RewardFunction`` (identical to the Egoistic condition) and are applied by the
training/evaluation loop at the terminal/merge step.

``rawlsian_objective_scale`` is an explicit calibration parameter (default 1.0)
used to bring the maximin signal onto a comparable scale with the task/safety
constants; it is logged to the run config and results CSV. The experience
function, ``rawlsian_objective``, and the least-advantaged-agent logic are NOT
modified: scaling is applied outside ``rawlsian_objective`` and the task/safety
terms are external constraints added to the scalar reward, not part of E_i.
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
        objective_scale: float = 1.0,
    ) -> None:
        self.experience_function = (
            experience_function if experience_function is not None else ExperienceFunction()
        )
        self.last_least_advantaged: Any = None
        self.last_experiences: dict = {}
        # Shared terminal merge-task / collision terms (see RewardFunction).
        self.merge_task_config = merge_task_config or MergeTaskConfig()
        # Explicit calibration scale for the maximin objective (default 1.0).
        self.objective_scale = float(objective_scale)

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
        return self.objective_scale * rawlsian_objective(experiences)
