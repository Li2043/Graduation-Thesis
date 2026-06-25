"""Egoistic reward (control-group baseline).

Scientifically valid per-agent reward for the egoistic baseline, derived ONLY
from raw signals already present in ``env_state`` / ``prev_env_state`` for the
controlled agent. It does not depend on any environment-emitted reward (the V1
environment emits none), does not access environment internals, does not use the
experience function, and computes no global multi-agent objective.

Per-agent reward:

    R_i = progress_reward
          - collision_penalty
          - risk_penalty          (TTC-based)
          - waiting_penalty

All required signals come from the agent's own state:
    position, velocity, lane, ttc, waiting_time, and ``crashed`` (which the
    environment sets equal to its ``collision_flag``). ``goal_position`` is used
    only to project longitudinal movement toward the goal direction.

Deterministic, standard-library/NumPy only, no Torch dependency.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from v1.rewards.base_reward import EnvState, RewardFunction


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


class EgoisticReward(RewardFunction):
    """Per-agent egoistic reward derived from raw ``env_state`` signals."""

    def __init__(
        self,
        agent_id: Any,
        alpha: float = 0.1,
        beta: float = 0.1,
        epsilon: float = 1e-3,
        progress_weight: float = 1.0,
        collision_penalty: float = 1.0,
        waiting_normalizer: float = 100.0,
    ) -> None:
        self.agent_id = agent_id
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.progress_weight = float(progress_weight)
        self.collision_penalty_value = float(collision_penalty)
        self.waiting_normalizer = float(waiting_normalizer) if waiting_normalizer else 1.0

    def _progress_reward(
        self,
        current: Any,
        prev_env_state: Optional[EnvState],
    ) -> float:
        """Longitudinal movement projected toward the goal direction."""
        if prev_env_state is None or self.agent_id not in prev_env_state:
            return 0.0
        previous = prev_env_state[self.agent_id]
        goal = _to_float(current.get("goal_position"))
        scale = goal if abs(goal) > 1e-6 else 1.0
        prev_distance = abs(goal - _to_float(previous.get("position")))
        curr_distance = abs(goal - _to_float(current.get("position")))
        progress = (prev_distance - curr_distance) / scale
        return self.progress_weight * progress

    def _collision_penalty(self, current: Any) -> float:
        # ``crashed`` mirrors the environment's collision_flag for this agent.
        return self.collision_penalty_value if bool(current.get("crashed", False)) else 0.0

    def _risk_penalty(self, current: Any) -> float:
        ttc = _to_float(current.get("ttc"), default=math.inf)
        if math.isinf(ttc):
            return 0.0
        return self.alpha * max(0.0, 1.0 / (ttc + self.epsilon))

    def _waiting_penalty(self, current: Any) -> float:
        waiting = _to_float(current.get("waiting_time"), 0.0)
        return self.beta * (waiting / self.waiting_normalizer)

    def compute(
        self,
        state: Any,
        next_state: Any,
        env_state: EnvState,
        prev_env_state: Optional[EnvState],
    ) -> float:
        if not env_state or self.agent_id not in env_state:
            return 0.0
        current = env_state[self.agent_id]

        progress_reward = self._progress_reward(current, prev_env_state)
        collision_penalty = self._collision_penalty(current)
        risk_penalty = self._risk_penalty(current)
        waiting_penalty = self._waiting_penalty(current)

        return float(
            progress_reward - collision_penalty - risk_penalty - waiting_penalty
        )
