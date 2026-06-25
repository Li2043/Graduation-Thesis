"""
Rawlsian Maximin reward wrapper for Gymnasium / highway-env.

Supports configurable vehicle scope (global, ego_neighbourhood, controlled)
and experience mode (speed_collision, safety_mobility).
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
        mode: str = "speed_collision",
        target_speed: float = 30.0,
        low_speed_threshold: float = 2.0,
        w_mobility: float = 1.0,
        w_collision: float = 2.0,
        w_low_speed: float = 0.3,
        w_risk: float = 0.5,
        risk_distance_normalizer: float = 50.0,
    ) -> None:
        super().__init__(env)
        self.xi = xi
        self.speed_normalizer = speed_normalizer
        self.scope = scope
        self.radius = radius
        self.mode = mode
        self.target_speed = target_speed
        self.low_speed_threshold = low_speed_threshold
        self.w_mobility = w_mobility
        self.w_collision = w_collision
        self.w_low_speed = w_low_speed
        self.w_risk = w_risk
        self.risk_distance_normalizer = risk_distance_normalizer
        self._previous_min_experience: float = 0.0

    def _experience_kwargs(self) -> dict:
        return {
            "mode": self.mode,
            "target_speed": self.target_speed,
            "low_speed_threshold": self.low_speed_threshold,
            "w_mobility": self.w_mobility,
            "w_collision": self.w_collision,
            "w_low_speed": self.w_low_speed,
            "w_risk": self.w_risk,
            "risk_distance_normalizer": self.risk_distance_normalizer,
        }

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
            **self._experience_kwargs(),
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
            **self._experience_kwargs(),
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
                "experience_mode": step_metrics["experience_mode"],
                "metric_scope": step_metrics["metric_scope"],
                "scoped_vehicle_count": step_metrics["scoped_vehicle_count"],
                "min_experience": step_metrics["min_experience"],
                "mean_experience": step_metrics["mean_experience"],
                "gini_experience": step_metrics["gini_experience"],
                "collision_count": step_metrics["collision_count"],
                "n_vehicles": step_metrics["n_vehicles"],
                "mean_mobility_score": step_metrics["mean_mobility_score"],
                "mean_risk_penalty": step_metrics["mean_risk_penalty"],
                "mean_low_speed_penalty": step_metrics["mean_low_speed_penalty"],
                "mean_collision_penalty": step_metrics["mean_collision_penalty"],
                "least_advantaged_index": step_metrics["least_advantaged_index"],
                "least_advantaged_is_ego": step_metrics["least_advantaged_is_ego"],
                "least_advantaged_speed": step_metrics["least_advantaged_speed"],
                "least_advantaged_crashed": step_metrics["least_advantaged_crashed"],
                "least_advantaged_experience": step_metrics["least_advantaged_experience"],
                "least_advantaged_vehicle_count": step_metrics["least_advantaged_vehicle_count"],
                "least_advantaged_reason": step_metrics["least_advantaged_reason"],
                "least_advantaged_mobility_score": step_metrics[
                    "least_advantaged_mobility_score"
                ],
                "least_advantaged_collision_penalty": step_metrics[
                    "least_advantaged_collision_penalty"
                ],
                "least_advantaged_low_speed_penalty": step_metrics[
                    "least_advantaged_low_speed_penalty"
                ],
                "least_advantaged_risk_penalty": step_metrics["least_advantaged_risk_penalty"],
                "least_advantaged_nearest_distance": step_metrics[
                    "least_advantaged_nearest_distance"
                ],
                "least_advantaged_weighted_mobility": step_metrics[
                    "least_advantaged_weighted_mobility"
                ],
                "least_advantaged_weighted_collision_penalty": step_metrics[
                    "least_advantaged_weighted_collision_penalty"
                ],
                "least_advantaged_weighted_low_speed_penalty": step_metrics[
                    "least_advantaged_weighted_low_speed_penalty"
                ],
                "least_advantaged_weighted_risk_penalty": step_metrics[
                    "least_advantaged_weighted_risk_penalty"
                ],
            }
        )

        return obs, total_reward, terminated, truncated, info
