"""
Shared fairness and safety metrics for baseline and Rawlsian runs.

Centralizes vehicle experience, Gini coefficient, and collision counting
so baseline and RawlsianRewardWrapper use identical definitions.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

EPSILON = 1e-9


def get_vehicles(env: gym.Env) -> list[Any]:
    """Safely read vehicles from unwrapped highway-env (works through wrappers)."""
    try:
        unwrapped = env.unwrapped
        road = getattr(unwrapped, "road", None)
        if road is None:
            return []
        vehicles = getattr(road, "vehicles", None)
        if vehicles is None:
            return []
        return list(vehicles)
    except (AttributeError, TypeError):
        return []


def vehicle_experience(vehicle: Any, speed_normalizer: float = 30.0) -> float:
    """experience_i = normalized_speed_i - collision_penalty_i."""
    speed = float(getattr(vehicle, "speed", 0.0))
    normalized_speed = speed / speed_normalizer
    collision_penalty = 1.0 if bool(getattr(vehicle, "crashed", False)) else 0.0
    return normalized_speed - collision_penalty


def compute_experiences(env: gym.Env, speed_normalizer: float = 30.0) -> list[float]:
    vehicles = get_vehicles(env)
    if not vehicles:
        return [0.0]
    return [vehicle_experience(v, speed_normalizer) for v in vehicles]


def min_experience(experiences: list[float]) -> float:
    return float(min(experiences)) if experiences else 0.0


def mean_experience(experiences: list[float]) -> float:
    return float(sum(experiences) / len(experiences)) if experiences else 0.0


def gini(values: list[float]) -> float:
    """
    Gini coefficient of a list of values.

    Negative experiences are shifted to a non-negative range before computing.
    Returns 0.0 for empty lists or near-zero values.
    """
    if not values:
        return 0.0

    shifted = list(values)
    min_val = min(shifted)
    if min_val < 0:
        shifted = [v - min_val for v in shifted]

    shifted = [v + EPSILON for v in shifted]
    if all(abs(v) < EPSILON for v in shifted):
        return 0.0

    sorted_vals = sorted(shifted)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total <= EPSILON:
        return 0.0

    weighted_sum = sum((i + 1) * x for i, x in enumerate(sorted_vals))
    return float((2 * weighted_sum) / (n * total) - (n + 1) / n)


def collision_count(env: gym.Env) -> int:
    """Count vehicles with crashed=True on the current road."""
    return sum(1 for v in get_vehicles(env) if bool(getattr(v, "crashed", False)))


def collect_step_metrics(env: gym.Env, speed_normalizer: float = 30.0) -> dict:
    """Per-step fairness/safety metrics shared by baseline and Rawlsian logging."""
    experiences = compute_experiences(env, speed_normalizer)
    vehicles = get_vehicles(env)
    return {
        "min_experience": min_experience(experiences),
        "mean_experience": mean_experience(experiences),
        "gini_experience": gini(experiences),
        "collision_count": collision_count(env),
        "n_vehicles": len(vehicles),
    }
