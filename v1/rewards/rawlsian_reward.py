"""Rawlsian reward (core contribution) — proposal-aligned delta-min shaping.

This implements the *change-based* Rawlsian shaping defined in the revised
research proposal. The per-step learning reward is the same individual driving
reward used by the Egoistic condition, plus a discrete shaping signal that
rewards/penalises improvements in the least-advantaged agent's experience:

    E_min_t       = min_i E_i,t
    delta_E_min_t = E_min_t - E_min_{t-1}

    r_rawls_t = +lambda_R   if delta_E_min_t >  epsilon_R
                -lambda_R   if delta_E_min_t < -epsilon_R
                 0.0        otherwise

    R_rawls = base_individual_reward + r_rawls_t

The shared merge-task adjustment and the shared terminal collision penalty are
applied on top of this by the training loop (identical to the Egoistic
condition), so the *only* research difference between conditions is the extra
delta-min shaping signal ``r_rawls_t``.

This replaces the earlier diagnostic implementation ``objective_scale * min_i E_i``
(raw maximin level). The experience function, ``rawlsian_objective``, and the
least-advantaged-agent logic are NOT modified: the shaping signal and the
task/safety terms are external to E_i.

NOTE on the previous-step reference. ``compute`` only receives the post-step
``env_state`` and the pre-step ``prev_env_state``; it does not receive the
state from two steps ago. We therefore approximate ``E_min_{t-1}`` by evaluating
the experience function on ``prev_env_state`` with ``None`` as its own previous
reference (so its mobility term is 0 for that evaluation). ``E_min_t`` is
evaluated on ``env_state`` with ``prev_env_state`` as reference, exactly as the
metrics layer does. This is a documented approximation; the shaping signal is a
diagnostic-grade discrete reward, not a precise difference of identically
referenced quantities.
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
from v1.rewards.egoistic_reward import EgoisticReward
from v1.rewards.merge_task_reward import MergeTaskConfig


class RawlsianReward(RewardFunction):
    """base individual driving reward + delta-min Rawlsian shaping signal.

    The shared terminal merge-task term and the shared terminal collision
    penalty are added separately by the loop via the inherited
    ``terminal_adjustment`` / ``terminal_collision_adjustment``.
    """

    def __init__(
        self,
        experience_function: Optional[ExperienceFunction] = None,
        merge_task_config: Optional[MergeTaskConfig] = None,
        rawlsian_lambda: float = 1.0,
        rawlsian_epsilon: float = 1e-6,
        agent_id: Any = None,
        base_reward: Optional[EgoisticReward] = None,
    ) -> None:
        self.experience_function = (
            experience_function if experience_function is not None else ExperienceFunction()
        )
        # Shared terminal merge-task / collision terms (see RewardFunction).
        self.merge_task_config = merge_task_config or MergeTaskConfig()
        # Discrete shaping parameters (proposal-aligned).
        self.rawlsian_lambda = float(rawlsian_lambda)
        self.rawlsian_epsilon = float(rawlsian_epsilon)
        # Base individual driving reward — identical to the Egoistic condition.
        self.base_reward = base_reward or EgoisticReward(
            agent_id=agent_id, merge_task_config=self.merge_task_config
        )

        # Read-only diagnostics, refreshed every compute() call.
        self.last_experiences: dict = {}
        self.last_least_advantaged: Any = None
        self.last_current_min_experience: Optional[float] = None
        self.last_previous_min_experience: Optional[float] = None
        self.last_delta_min_experience: Optional[float] = None
        self.last_rawlsian_signal: float = 0.0

    def compute_rawlsian_signal(
        self,
        env_state: EnvState,
        prev_env_state: Optional[EnvState],
    ) -> float:
        """Return the discrete delta-min shaping signal for this transition.

        Fails safe (returns 0.0) when there is no previous state or when
        experiences cannot be computed, so the policy never sees a spurious
        shaping reward at episode start or on degenerate states.
        """
        # No previous reference at episode start -> no shaping signal.
        if prev_env_state is None:
            self.last_current_min_experience = None
            self.last_previous_min_experience = None
            self.last_delta_min_experience = None
            self.last_rawlsian_signal = 0.0
            return 0.0

        current = compute_all_experiences(env_state, prev_env_state, self.experience_function)
        # E_min_{t-1} approximated on prev_env_state (None reference; see module docstring).
        previous = compute_all_experiences(prev_env_state, None, self.experience_function)

        # Fail safe on empty/unavailable experiences: no shaping signal.
        if not current or not previous:
            self.last_experiences = dict(current) if current else {}
            self.last_least_advantaged = (
                least_advantaged_agent(current) if current else None
            )
            self.last_current_min_experience = None
            self.last_previous_min_experience = None
            self.last_delta_min_experience = None
            self.last_rawlsian_signal = 0.0
            return 0.0

        self.last_experiences = dict(current)
        self.last_least_advantaged = least_advantaged_agent(current)
        current_min = float(rawlsian_objective(current))
        previous_min = float(rawlsian_objective(previous))
        delta_min = current_min - previous_min

        self.last_current_min_experience = current_min
        self.last_previous_min_experience = previous_min
        self.last_delta_min_experience = delta_min

        if delta_min > self.rawlsian_epsilon:
            signal = self.rawlsian_lambda
        elif delta_min < -self.rawlsian_epsilon:
            signal = -self.rawlsian_lambda
        else:
            signal = 0.0
        self.last_rawlsian_signal = float(signal)
        return float(signal)

    def compute(
        self,
        state: Any,
        next_state: Any,
        env_state: EnvState,
        prev_env_state: Optional[EnvState],
    ) -> float:
        base = self.base_reward.compute(state, next_state, env_state, prev_env_state)
        signal = self.compute_rawlsian_signal(env_state, prev_env_state)
        return float(base) + float(signal)
