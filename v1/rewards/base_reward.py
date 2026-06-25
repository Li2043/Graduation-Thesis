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
        """Return the scalar learning reward for one transition.

        Parameters
        ----------
        state, next_state:
            The controlled agent's observation before and after the step.
        env_state, prev_env_state:
            The multi-agent environment states after and before the step,
            each mapping ``agent_id -> agent_state``.
        """
        raise NotImplementedError
