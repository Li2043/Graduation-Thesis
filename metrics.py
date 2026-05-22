"""
Shared fairness and safety metrics for baseline and Rawlsian runs.

Supports global, ego_neighbourhood, and controlled vehicle scopes (v0.4).
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym

EPSILON = 1e-9
SUPPORTED_SCOPES = ("global", "ego_neighbourhood", "controlled")


def get_vehicles(env: gym.Env) -> list[Any]:
    """Safely read all vehicles from unwrapped highway-env."""
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
        return getattr(unwrapped, "vehicle", None)
    except (AttributeError, TypeError):
        return None


def get_controlled_vehicles(env: gym.Env) -> list[Any]:
    """Return controlled vehicles, or [ego] if only ego is available."""
    try:
        unwrapped = env.unwrapped
        controlled = getattr(unwrapped, "controlled_vehicles", None)
        if controlled:
            return list(controlled)
    except (AttributeError, TypeError):
        pass

    ego = get_ego_vehicle(env)
    if ego is not None:
        return [ego]
    return []


def vehicle_position(vehicle: Any) -> tuple[float, float] | None:
    """Read (x, y) from a highway-env vehicle position."""
    try:
        pos = getattr(vehicle, "position", None)
        if pos is None:
            return None
        return float(pos[0]), float(pos[1])
    except (AttributeError, TypeError, IndexError, ValueError):
        return None


def vehicle_distance(v1: Any, v2: Any) -> float | None:
    """Euclidean distance between two vehicles; None if positions unavailable."""
    p1 = vehicle_position(v1)
    p2 = vehicle_position(v2)
    if p1 is None or p2 is None:
        return None
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_ego_neighbourhood_vehicles(env: gym.Env, radius: float = 50.0) -> list[Any]:
    """Ego vehicle plus all vehicles within radius of ego."""
    all_vehicles = get_vehicles(env)
    ego = get_ego_vehicle(env)

    if ego is None:
        return all_vehicles

    neighbourhood: list[Any] = []
    seen: set[int] = set()

    for vehicle in all_vehicles:
        vid = id(vehicle)
        if vid in seen:
            continue
        if vehicle is ego:
            neighbourhood.append(vehicle)
            seen.add(vid)
            continue
        dist = vehicle_distance(vehicle, ego)
        if dist is not None and dist <= radius:
            neighbourhood.append(vehicle)
            seen.add(vid)

    if not neighbourhood and ego is not None:
        return [ego]
    return neighbourhood


def get_vehicles_by_scope(
    env: gym.Env,
    scope: str = "global",
    radius: float = 50.0,
) -> list[Any]:
    """Select vehicle set for scoped fairness metrics."""
    if scope == "global":
        return get_vehicles(env)
    if scope == "ego_neighbourhood":
        return get_ego_neighbourhood_vehicles(env, radius)
    if scope == "controlled":
        return get_controlled_vehicles(env)
    raise ValueError(
        f"Unsupported scope '{scope}'. Supported scopes: global, ego_neighbourhood, controlled"
    )


def vehicle_experience(vehicle: Any, speed_normalizer: float = 30.0) -> float:
    """experience_i = normalized_speed_i - collision_penalty_i."""
    speed = float(getattr(vehicle, "speed", 0.0))
    normalized_speed = speed / speed_normalizer
    collision_penalty = 1.0 if bool(getattr(vehicle, "crashed", False)) else 0.0
    return normalized_speed - collision_penalty


def compute_experiences(
    env: gym.Env,
    speed_normalizer: float = 30.0,
    scope: str = "global",
    radius: float = 50.0,
) -> list[float]:
    vehicles = get_vehicles_by_scope(env, scope, radius)
    if not vehicles:
        return [0.0]
    return [vehicle_experience(v, speed_normalizer) for v in vehicles]


def min_experience(experiences: list[float]) -> float:
    return float(min(experiences)) if experiences else 0.0


def mean_experience(experiences: list[float]) -> float:
    return float(sum(experiences) / len(experiences)) if experiences else 0.0


def gini(values: list[float]) -> float:
    """Gini coefficient; negative values shifted to non-negative range first."""
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
    """Global count of crashed vehicles on the road."""
    return sum(1 for v in get_vehicles(env) if bool(getattr(v, "crashed", False)))


def least_advantaged_vehicle_info(
    env: gym.Env,
    speed_normalizer: float = 30.0,
    scope: str = "global",
    radius: float = 50.0,
) -> dict:
    """Identify least advantaged vehicle within the scoped vehicle set."""
    empty = {
        "least_advantaged_index": -1,
        "least_advantaged_is_ego": False,
        "least_advantaged_speed": 0.0,
        "least_advantaged_crashed": False,
        "least_advantaged_experience": 0.0,
        "least_advantaged_vehicle_count": 0,
    }

    vehicles = get_vehicles_by_scope(env, scope, radius)
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


def collect_step_metrics(
    env: gym.Env,
    speed_normalizer: float = 30.0,
    scope: str = "global",
    radius: float = 50.0,
) -> dict:
    """Per-step fairness/safety metrics for a given vehicle scope."""
    experiences = compute_experiences(env, speed_normalizer, scope, radius)
    all_vehicles = get_vehicles(env)
    scoped_vehicles = get_vehicles_by_scope(env, scope, radius)
    la_info = least_advantaged_vehicle_info(env, speed_normalizer, scope, radius)

    return {
        "min_experience": min_experience(experiences),
        "mean_experience": mean_experience(experiences),
        "gini_experience": gini(experiences),
        "collision_count": collision_count(env),
        "n_vehicles": len(all_vehicles),
        "metric_scope": scope,
        "scoped_vehicle_count": len(scoped_vehicles),
        "ego_neighbourhood_radius": float(radius),
        **la_info,
    }
