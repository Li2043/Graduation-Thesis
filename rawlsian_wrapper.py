"""
Rawlsian Maximin reward wrapper for Gymnasium / highway-env.

Shapes reward from changes in the minimum vehicle experience across steps.
Does not modify highway-env source code.
"""

from __future__ import annotations

from typing import Any #any type of vehicle

import gymnasium as gym


class RawlsianRewardWrapper(gym.Wrapper):
    """
    Adds a Rawlsian maximin shaping term on top of the environment reward.
    experience_i = normalized_speed_i - collision_penalty_i
    rawlsian_reward = +xi / -xi / 0 when min experience improves / worsens / unchanged
    rawlsian_reward set to 0.2
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

    def _get_vehicles(self) -> list[Any]:
        """Safely read vehicle list from unwrapped highway-env."""
        try:
            unwrapped = self.env.unwrapped
            road = getattr(unwrapped, "road", None) 
            # prevent crashing the code
            if road is None:
                return []
            vehicles = getattr(road, "vehicles", None)
            if vehicles is None:
                return []
            return list(vehicles)
        except (AttributeError, TypeError):
            return []

    # define how to calculate experience
    def _vehicle_experience(self, vehicle: Any) -> float:
        """experience = normalized_speed - collision_penalty."""
        speed = float(getattr(vehicle, "speed", 0.0))
        normalized_speed = speed / self.speed_normalizer
        crashed = bool(getattr(vehicle, "crashed", False))
        collision_penalty = 1.0 if crashed else 0.0
        return normalized_speed - collision_penalty

    # compute experiences for all vehicles
    def _compute_experiences(self) -> list[float]:
        vehicles = self._get_vehicles()
        if not vehicles:
            return [0.0]
        return [self._vehicle_experience(v) for v in vehicles]

    def _min_experience(self, experiences: list[float]) -> float:
        return float(min(experiences)) if experiences else 0.0

    # define how to calculate rawlsian reward
    def _rawlsian_delta(self, current_min: float) -> float:
        if current_min > self._previous_min_experience:
            return self.xi
        if current_min < self._previous_min_experience:
            return -self.xi
        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        experiences = self._compute_experiences()
        self._previous_min_experience = self._min_experience(experiences)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        experiences = self._compute_experiences()
        min_experience = self._min_experience(experiences)
        rawlsian_reward = self._rawlsian_delta(min_experience)
        self._previous_min_experience = min_experience

        total_reward = float(original_reward) + rawlsian_reward


        info = dict(info) if info is not None else {}
        info.update(
            {
                "original_reward": float(original_reward),
                "rawlsian_reward": float(rawlsian_reward),
                "total_reward": float(total_reward),
                "min_experience": min_experience,
                "vehicle_experiences": experiences,
                "n_vehicles": len(self._get_vehicles()),
            }
        )

        return obs, total_reward, terminated, truncated, info
