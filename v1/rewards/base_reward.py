"""V1 reward-function interface.

Defines the unified contract used to turn an environment transition into the
scalar learning signal. Both the Egoistic and Rawlsian conditions implement this
interface, so the difference between conditions is *only* which reward function
is injected into the (identical) training loop.

This module contains no reinforcement-learning logic and does not import any
training, policy, or legacy code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from v1.rewards.merge_task_reward import MergeTaskConfig, terminal_merge_adjustment

AgentId = Any
AgentState = Mapping[str, Any]
EnvState = Mapping[AgentId, AgentState]


class RewardFunction(ABC):
    """Abstract scalar reward function for a single environment transition."""

    @abstractmethod
    def compute(
        self,
        state: Any,
        next_state: Any,
        env_state: EnvState,
        prev_env_state: Optional[EnvState],
    ) -> float:
        """Return the per-step scalar learning reward for one transition.

        Parameters
        ----------
        state, next_state:
            The controlled agent's observation before and after the step.
        env_state, prev_env_state:
            The multi-agent environment states after and before the step,
            each mapping ``agent_id -> agent_state``.
        """
        raise NotImplementedError

    def terminal_adjustment(
        self,
        ego_state: AgentState,
        done: bool,
        truncated: bool,
        merged: Optional[bool] = None,
    ) -> float:
        """Shared terminal merge-task adjustment, identical across conditions.

        ``compute`` cannot see ``done``/``truncated``, so the training/evaluation
        loop calls this immediately after ``env.step`` to add the one-off
        merge-success bonus or non-merge failure penalty. Both the Egoistic and
        Rawlsian reward functions inherit this method unchanged, guaranteeing
        they solve the identical merge task. The experience function and the
        per-step ``compute`` are not affected.
        """
        config = getattr(self, "merge_task_config", None) or MergeTaskConfig()
        return terminal_merge_adjustment(ego_state, done, truncated, config, merged)
