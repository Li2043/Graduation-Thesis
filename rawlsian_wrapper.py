"""
Rawlsian Maximin reward wrapper for Gymnasium / highway-env.

Reuses shared metrics from metrics.py for experience and fairness logging.
"""

from __future__ import annotations

import gymnasium as gym

from metrics import collect_step_metrics, compute_experiences, min_experience


class RawlsianRewardWrapper(gym.Wrapper):
    """
    Adds a Rawlsian maximin shaping term on top of the environment reward.

    rawlsian_reward = +xi / -xi / 0 when min experience improves / worsens / unchanged
    total_reward = original_reward + rawlsian_reward
    """

    def __init__(
        self,
        env: gym.Env,
        xi: float = 0.2,
        speed_normalizer: float = 30.0,
    ) -> None:
        super().__init__(env)
        self.xi = xi
        self.speed_normalizer = speed_normalizer
        self._previous_min_experience: float = 0.0

    def _rawlsian_delta(self, current_min: float) -> float:
        if current_min > self._previous_min_experience:
            return self.xi
        if current_min < self._previous_min_experience:
            return -self.xi
        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        experiences = compute_experiences(self.env, self.speed_normalizer)
        self._previous_min_experience = min_experience(experiences)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        step_metrics = collect_step_metrics(self.env, self.speed_normalizer)
        current_min = step_metrics["min_experience"]
        rawlsian_reward = self._rawlsian_delta(current_min)
        self._previous_min_experience = current_min

        total_reward = float(original_reward) + rawlsian_reward

        info = dict(info) if info is not None else {}
        info.update(
            {
                "original_reward": float(original_reward),
                "rawlsian_reward": float(rawlsian_reward),
                "total_reward": float(total_reward),
                "min_experience": step_metrics["min_experience"],
                "mean_experience": step_metrics["mean_experience"],
                "gini_experience": step_metrics["gini_experience"],
                "collision_count": step_metrics["collision_count"],
                "n_vehicles": step_metrics["n_vehicles"],
            }
        )

        return obs, total_reward, terminated, truncated, info
