"""
Shared fairness and safety metrics for baseline and Rawlsian runs.
Centralizes vehicle experience, Gini coefficient, and collision counting
so baseline and RawlsianRewardWrapper use identical definitions.
让baseline和rawlsian都使用相同的定义
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

# define a small number to avoid division by zero
EPSILON = 1e-9


def get_vehicles(env: gym.Env) -> list[Any]:
    """Safely read vehicles from unwrapped highway-env."""
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


def get_ego_vehicle(env: gym.Env) -> Any | None:
    """Safely read the controllable ego vehicle from unwrapped highway-env."""
    try:
        unwrapped = env.unwrapped
        ego = getattr(unwrapped, "vehicle", None)
        return ego
    except (AttributeError, TypeError):
        return None


def least_advantaged_vehicle_info(env: gym.Env, speed_normalizer: float = 30.0) -> dict:
    """
    Identify which vehicle has minimum experience and whether it is the ego vehicle.
    """
    empty = {
        "least_advantaged_index": -1,
        "least_advantaged_is_ego": False,
        "least_advantaged_speed": 0.0,
        "least_advantaged_crashed": False,
        "least_advantaged_experience": 0.0,
        "least_advantaged_vehicle_count": 0,
    }

    vehicles = get_vehicles(env)
    if not vehicles:
        return empty

    ego_vehicle = get_ego_vehicle(env)
    experiences = [vehicle_experience(v, speed_normalizer) for v in vehicles]
    least_index = min(range(len(experiences)), key=lambda i: experiences[i])
    least_vehicle = vehicles[least_index]

    return {
        "least_advantaged_index": int(least_index),
        "least_advantaged_is_ego": bool(least_vehicle is ego_vehicle),
        "least_advantaged_speed": float(getattr(least_vehicle, "speed", 0.0)),
        "least_advantaged_crashed": bool(getattr(least_vehicle, "crashed", False)),
        "least_advantaged_experience": float(experiences[least_index]),
        "least_advantaged_vehicle_count": int(len(vehicles)),
    }


def accumulate_least_advantaged_step(step_metrics: dict, counters: dict) -> None:
    """Update running least-advantaged counters from one step of collect_step_metrics."""
    if step_metrics.get("least_advantaged_is_ego"):
        counters["least_advantaged_ego_steps"] += 1
    else:
        counters["least_advantaged_non_ego_steps"] += 1
    counters["sum_least_advantaged_speed"] += float(
        step_metrics.get("least_advantaged_speed", 0.0)
    )
    if step_metrics.get("least_advantaged_crashed"):
        counters["least_advantaged_crash_steps"] += 1


def finalize_least_advantaged_episode(counters: dict, steps: int) -> dict:
    """Convert step counters into per-episode least-advantaged diagnostic fields."""
    denom = steps if steps > 0 else 1
    ego_steps = counters["least_advantaged_ego_steps"]
    non_ego_steps = counters["least_advantaged_non_ego_steps"]
    return {
        "least_advantaged_ego_steps": ego_steps,
        "least_advantaged_non_ego_steps": non_ego_steps,
        "least_advantaged_ego_ratio": ego_steps / denom,
        "mean_least_advantaged_speed": counters["sum_least_advantaged_speed"] / denom,
        "least_advantaged_crash_steps": counters["least_advantaged_crash_steps"],
    }


def new_least_advantaged_counters() -> dict:
    return {
        "least_advantaged_ego_steps": 0,
        "least_advantaged_non_ego_steps": 0,
        "sum_least_advantaged_speed": 0.0,
        "least_advantaged_crash_steps": 0,
    }



# experience_i = normalized_speed_i - collision_penalty_i.
# if collide, -1
def vehicle_experience(vehicle: Any, speed_normalizer: float = 30.0) -> float:
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
    la_info = least_advantaged_vehicle_info(env, speed_normalizer)
    return {
        "min_experience": min_experience(experiences),
        "mean_experience": mean_experience(experiences),
        "gini_experience": gini(experiences),
        "collision_count": collision_count(env),
        "n_vehicles": len(vehicles),
        **la_info,
    }
