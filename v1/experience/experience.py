"""V1 Experience Function.

Pure, stateless implementation of the per-agent experience defined in
``docs/V1_SYSTEM_SPEC.md``:

    E_i = w1 * mobility + w2 * safety - w3 * waiting_time

This module is evaluation-level only. It does not import or depend on any
reinforcement-learning, training, or reward code, and it does not reference any
legacy implementation. All functions are deterministic and stateless except for
their explicit inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

AgentId = Any
AgentState = Mapping[str, Any]

DEFAULT_TTC_HORIZON = 10.0


@dataclass(frozen=True)
class ExperienceWeights:
    """Non-negative weights for the experience components."""

    w_mobility: float = 1.0
    w_safety: float = 1.0
    w_waiting: float = 1.0


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float, returning ``default`` when not possible."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` to the closed interval [low, high]."""
    return float(max(low, min(high, value)))


class ExperienceFunction:
    """Compute per-agent experience E_i = w1*mobility + w2*safety - w3*waiting.

    The function is reusable for both Egoistic and Rawlsian policies: it only
    measures an agent's situation and never references an objective or reward.
    """

    def __init__(
        self,
        weights: Optional[ExperienceWeights] = None,
        ttc_horizon: float = DEFAULT_TTC_HORIZON,
    ) -> None:
        self.weights = weights if weights is not None else ExperienceWeights()
        ttc_horizon = float(ttc_horizon)
        if ttc_horizon <= 0.0:
            raise ValueError("ttc_horizon must be a positive number")
        self.ttc_horizon = ttc_horizon

    def mobility(self, agent_state: AgentState, prev_state: Optional[AgentState]) -> float:
        """Progress toward the goal: reduction in distance-to-goal vs prev_state.

        Positive when the agent moved closer to its goal. Returns 0.0 when there
        is no previous state to compare against.
        """
        if prev_state is None:
            return 0.0

        prev_distance = abs(
            _to_float(prev_state.get("goal_position")) - _to_float(prev_state.get("position"))
        )
        curr_distance = abs(
            _to_float(agent_state.get("goal_position")) - _to_float(agent_state.get("position"))
        )
        return float(prev_distance - curr_distance)

    def safety(self, agent_state: AgentState) -> float:
        """TTC-based safety score normalised to [0, 1].

        Higher time-to-collision denotes a safer situation. If TTC is missing
        (key absent, ``None``, or non-numeric), safety is 0.0 by definition.
        """
        if not isinstance(agent_state, Mapping) or "ttc" not in agent_state:
            return 0.0

        raw_ttc = agent_state.get("ttc")
        if raw_ttc is None:
            return 0.0
        try:
            ttc = float(raw_ttc)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(ttc):
            return 0.0

        return _clip(ttc / self.ttc_horizon, 0.0, 1.0)

    def waiting(self, agent_state: AgentState) -> float:
        """Accumulated waiting time read directly from the agent state."""
        return _to_float(agent_state.get("waiting_time"), 0.0)

    def compute(self, agent_state: AgentState, prev_state: Optional[AgentState]) -> float:
        """Return the scalar experience E_i for a single agent."""
        weights = self.weights
        mobility = self.mobility(agent_state, prev_state)
        safety = self.safety(agent_state)
        waiting = self.waiting(agent_state)
        return float(
            weights.w_mobility * mobility
            + weights.w_safety * safety
            - weights.w_waiting * waiting
        )


def compute_all_experiences(
    env_state: Mapping[AgentId, AgentState],
    prev_env_state: Optional[Mapping[AgentId, AgentState]],
    exp_fn: ExperienceFunction,
) -> dict[AgentId, float]:
    """Compute experience for every agent in the environment state.

    Returns a mapping ``{agent_id: E_i}``. When ``prev_env_state`` is None, or
    an agent has no matching previous entry, that agent's mobility term is 0.0.
    """
    experiences: dict[AgentId, float] = {}
    for agent_id, agent_state in env_state.items():
        prev_state = None
        if prev_env_state is not None:
            prev_state = prev_env_state.get(agent_id)
        experiences[agent_id] = exp_fn.compute(agent_state, prev_state)
    return experiences


def least_advantaged_agent(experiences: Mapping[AgentId, float]) -> AgentId:
    """Return the id of the worst-off agent: argmin_i E_i.

    Ties are broken deterministically by agent id so the result does not depend
    on dictionary insertion order.
    """
    if not experiences:
        raise ValueError("experiences must contain at least one agent")
    return min(experiences.items(), key=lambda item: (item[1], item[0]))[0]


def rawlsian_objective(experiences: Mapping[AgentId, float]) -> float:
    """Return the worst-off experience: min_i E_i."""
    if not experiences:
        raise ValueError("experiences must contain at least one agent")
    return float(min(experiences.values()))
