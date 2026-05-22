"""
Rawlsian Maximin reward wrapper for Gymnasium / highway-env.

Supports configurable vehicle scope (global, ego_neighbourhood, controlled).
"""

from __future__ import annotations

import gymnasium as gym

from metrics import collect_step_metrics, compute_experiences, min_experience


class RawlsianRewardWrapper(gym.Wrapper):
    """
    Adds a Rawlsian maximin shaping term on top of the environment reward.

    rawlsian_reward = +xi / -xi / 0 when scoped min experience improves / worsens / unchanged
    total_reward = original_reward + rawlsian_reward
    """

    def __init__(
        self,
        env: gym.Env,
        xi: float = 0.2,
        speed_normalizer: float = 30.0,
        scope: str = "global",
        radius: float = 50.0,
    ) -> None:
        super().__init__(env)
        self.xi = xi
        self.speed_normalizer = speed_normalizer
        self.scope = scope
        self.radius = radius
        self._previous_min_experience: float = 0.0

    def _rawlsian_delta(self, current_min: float) -> float:
        if current_min > self._previous_min_experience:
            return self.xi
        if current_min < self._previous_min_experience:
            return -self.xi
        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        experiences = compute_experiences(
            self.env,
            self.speed_normalizer,
            scope=self.scope,
            radius=self.radius,
        )
        self._previous_min_experience = min_experience(experiences)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        step_metrics = collect_step_metrics(
            self.env,
            self.speed_normalizer,
            scope=self.scope,
            radius=self.radius,
        )
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
                "rawlsian_scope": self.scope,
                "ego_neighbourhood_radius": float(self.radius),
                "metric_scope": step_metrics["metric_scope"],
                "scoped_vehicle_count": step_metrics["scoped_vehicle_count"],
                "min_experience": step_metrics["min_experience"],
                "mean_experience": step_metrics["mean_experience"],
                "gini_experience": step_metrics["gini_experience"],
                "collision_count": step_metrics["collision_count"],
                "n_vehicles": step_metrics["n_vehicles"],
                "least_advantaged_index": step_metrics["least_advantaged_index"],
                "least_advantaged_is_ego": step_metrics["least_advantaged_is_ego"],
                "least_advantaged_speed": step_metrics["least_advantaged_speed"],
                "least_advantaged_crashed": step_metrics["least_advantaged_crashed"],
                "least_advantaged_experience": step_metrics["least_advantaged_experience"],
                "least_advantaged_vehicle_count": step_metrics["least_advantaged_vehicle_count"],
            }
        )

        return obs, total_reward, terminated, truncated, info
