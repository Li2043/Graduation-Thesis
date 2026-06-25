"""
Shared fairness and safety metrics for baseline and Rawlsian runs.

Supports global, ego_neighbourhood, and controlled vehicle scopes (v0.4).
Supports speed_collision and safety_mobility experience modes (v0.5).
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym

EPSILON = 1e-9
SUPPORTED_SCOPES = ("global", "ego_neighbourhood", "controlled")
SUPPORTED_EXPERIENCE_MODES = ("speed_collision", "safety_mobility")
REASON_STEP_FIELDS = (
    "reason_collision_steps",
    "reason_low_mobility_steps",
    "reason_low_speed_steps",
    "reason_risk_steps",
    "reason_none_steps",
    "reason_combined_steps",
)


def clip(value: float, low: float, high: float) -> float:
    """Return value clamped to [low, high]."""
    return float(max(low, min(high, value)))


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


def nearest_vehicle_distance(vehicle: Any, vehicles: list[Any]) -> float | None:
    """Nearest Euclidean distance from vehicle to any other vehicle in the list."""
    min_dist: float | None = None
    for other in vehicles:
        if other is vehicle:
            continue
        dist = vehicle_distance(vehicle, other)
        if dist is None:
            continue
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return min_dist


def risk_penalty_from_distance(distance: float | None, normalizer: float = 50.0) -> float:
    """Higher risk when nearest vehicle is closer; zero beyond normalizer."""
    if distance is None:
        return 0.0
    return float(1.0 - clip(distance / normalizer, 0.0, 1.0))


def vehicle_experience(
    vehicle: Any,
    speed_normalizer: float = 30.0,
    mode: str = "speed_collision",
    vehicles: list[Any] | None = None,
    target_speed: float = 30.0,
    low_speed_threshold: float = 2.0,
    w_mobility: float = 1.0,
    w_collision: float = 2.0,
    w_low_speed: float = 0.3,
    w_risk: float = 0.5,
    risk_distance_normalizer: float = 50.0,
) -> float:
    """Vehicle experience under speed_collision (v0.4) or safety_mobility (v0.5) mode."""
    if mode == "speed_collision":
        speed = float(getattr(vehicle, "speed", 0.0))
        normalized_speed = speed / speed_normalizer
        collision_penalty = 1.0 if bool(getattr(vehicle, "crashed", False)) else 0.0
        return normalized_speed - collision_penalty

    if mode == "safety_mobility":
        components = vehicle_experience_components(
            vehicle,
            vehicles=vehicles,
            target_speed=target_speed,
            low_speed_threshold=low_speed_threshold,
            w_mobility=w_mobility,
            w_collision=w_collision,
            w_low_speed=w_low_speed,
            w_risk=w_risk,
            risk_distance_normalizer=risk_distance_normalizer,
        )
        return float(components["experience"])

    raise ValueError(
        f"Unsupported experience mode '{mode}'. "
        "Supported modes: speed_collision, safety_mobility"
    )


def vehicle_experience_components(
    vehicle: Any,
    vehicles: list[Any] | None = None,
    target_speed: float = 30.0,
    low_speed_threshold: float = 2.0,
    w_mobility: float = 1.0,
    w_collision: float = 2.0,
    w_low_speed: float = 0.3,
    w_risk: float = 0.5,
    risk_distance_normalizer: float = 50.0,
) -> dict:
    """Return decomposed safety-mobility experience terms for one vehicle."""
    speed = float(getattr(vehicle, "speed", 0.0))
    crashed = bool(getattr(vehicle, "crashed", False))

    mobility_score = clip(speed / target_speed, 0.0, 1.0)
    collision_penalty = 1.0 if crashed else 0.0
    low_speed_penalty = 1.0 if speed < low_speed_threshold else 0.0

    nearest_dist = nearest_vehicle_distance(vehicle, vehicles or [])
    risk_penalty = risk_penalty_from_distance(nearest_dist, risk_distance_normalizer)

    weighted_mobility = w_mobility * mobility_score
    weighted_collision_penalty = w_collision * collision_penalty
    weighted_low_speed_penalty = w_low_speed * low_speed_penalty
    weighted_risk_penalty = w_risk * risk_penalty
    experience = (
        weighted_mobility
        - weighted_collision_penalty
        - weighted_low_speed_penalty
        - weighted_risk_penalty
    )

    return {
        "speed": speed,
        "mobility_score": mobility_score,
        "collision_penalty": collision_penalty,
        "low_speed_penalty": low_speed_penalty,
        "nearest_vehicle_distance": nearest_dist,
        "risk_penalty": risk_penalty,
        "weighted_mobility": weighted_mobility,
        "weighted_collision_penalty": weighted_collision_penalty,
        "weighted_low_speed_penalty": weighted_low_speed_penalty,
        "weighted_risk_penalty": weighted_risk_penalty,
        "experience": float(experience),
    }


def least_advantaged_reason(components: dict) -> str:
    """Explain why a vehicle is least advantaged from its experience components."""
    if components.get("collision_penalty", 0.0) > 0:
        return "collision"

    low_mobility_penalty = 1.0 - components.get("mobility_score", 0.0)
    penalties = {
        "low_mobility": low_mobility_penalty,
        "low_speed": components.get("weighted_low_speed_penalty", 0.0),
        "risk": components.get("weighted_risk_penalty", 0.0),
    }

    max_val = max(penalties.values())
    if max_val <= EPSILON and low_mobility_penalty <= EPSILON:
        return "none"

    top_reasons = [name for name, val in penalties.items() if abs(val - max_val) <= EPSILON]
    if len(top_reasons) > 1:
        return "combined"
    return top_reasons[0]


def compute_experiences(
    env: gym.Env,
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
) -> list[float]:
    vehicles = get_vehicles_by_scope(env, scope, radius)
    if not vehicles:
        return [0.0]
    return [
        vehicle_experience(
            vehicle,
            speed_normalizer=speed_normalizer,
            mode=mode,
            vehicles=vehicles,
            target_speed=target_speed,
            low_speed_threshold=low_speed_threshold,
            w_mobility=w_mobility,
            w_collision=w_collision,
            w_low_speed=w_low_speed,
            w_risk=w_risk,
            risk_distance_normalizer=risk_distance_normalizer,
        )
        for vehicle in vehicles
    ]


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


def _empty_la_info() -> dict:
    return {
        "least_advantaged_index": -1,
        "least_advantaged_is_ego": False,
        "least_advantaged_speed": 0.0,
        "least_advantaged_crashed": False,
        "least_advantaged_experience": 0.0,
        "least_advantaged_vehicle_count": 0,
        "least_advantaged_reason": "none",
        "least_advantaged_mobility_score": 0.0,
        "least_advantaged_collision_penalty": 0.0,
        "least_advantaged_low_speed_penalty": 0.0,
        "least_advantaged_risk_penalty": 0.0,
        "least_advantaged_nearest_distance": 0.0,
        "least_advantaged_weighted_mobility": 0.0,
        "least_advantaged_weighted_collision_penalty": 0.0,
        "least_advantaged_weighted_low_speed_penalty": 0.0,
        "least_advantaged_weighted_risk_penalty": 0.0,
    }


def _components_from_mode(
    vehicle: Any,
    vehicles: list[Any],
    mode: str,
    speed_normalizer: float,
    target_speed: float,
    low_speed_threshold: float,
    w_mobility: float,
    w_collision: float,
    w_low_speed: float,
    w_risk: float,
    risk_distance_normalizer: float,
) -> dict:
    if mode == "safety_mobility":
        return vehicle_experience_components(
            vehicle,
            vehicles=vehicles,
            target_speed=target_speed,
            low_speed_threshold=low_speed_threshold,
            w_mobility=w_mobility,
            w_collision=w_collision,
            w_low_speed=w_low_speed,
            w_risk=w_risk,
            risk_distance_normalizer=risk_distance_normalizer,
        )

    speed = float(getattr(vehicle, "speed", 0.0))
    crashed = bool(getattr(vehicle, "crashed", False))
    mobility_score = speed / speed_normalizer
    collision_penalty = 1.0 if crashed else 0.0
    experience = mobility_score - collision_penalty
    return {
        "speed": speed,
        "mobility_score": mobility_score,
        "collision_penalty": collision_penalty,
        "low_speed_penalty": 0.0,
        "nearest_vehicle_distance": nearest_vehicle_distance(vehicle, vehicles),
        "risk_penalty": 0.0,
        "weighted_mobility": mobility_score,
        "weighted_collision_penalty": collision_penalty,
        "weighted_low_speed_penalty": 0.0,
        "weighted_risk_penalty": 0.0,
        "experience": float(experience),
    }


def least_advantaged_vehicle_info(
    env: gym.Env,
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
) -> dict:
    """Identify least advantaged vehicle within the scoped vehicle set."""
    empty = _empty_la_info()
    vehicles = get_vehicles_by_scope(env, scope, radius)
    if not vehicles:
        return empty

    ego_vehicle = get_ego_vehicle(env)
    components_list = [
        _components_from_mode(
            vehicle,
            vehicles,
            mode,
            speed_normalizer,
            target_speed,
            low_speed_threshold,
            w_mobility,
            w_collision,
            w_low_speed,
            w_risk,
            risk_distance_normalizer,
        )
        for vehicle in vehicles
    ]
    experiences = [c["experience"] for c in components_list]
    least_index = min(range(len(experiences)), key=lambda i: experiences[i])
    least_vehicle = vehicles[least_index]
    components = components_list[least_index]
    nearest_dist = components.get("nearest_vehicle_distance")

    return {
        "least_advantaged_index": int(least_index),
        "least_advantaged_is_ego": bool(least_vehicle is ego_vehicle),
        "least_advantaged_speed": float(components["speed"]),
        "least_advantaged_crashed": bool(components["collision_penalty"] > 0),
        "least_advantaged_experience": float(experiences[least_index]),
        "least_advantaged_vehicle_count": int(len(vehicles)),
        "least_advantaged_reason": least_advantaged_reason(components),
        "least_advantaged_mobility_score": float(components["mobility_score"]),
        "least_advantaged_collision_penalty": float(components["collision_penalty"]),
        "least_advantaged_low_speed_penalty": float(components["low_speed_penalty"]),
        "least_advantaged_risk_penalty": float(components["risk_penalty"]),
        "least_advantaged_nearest_distance": float(
            nearest_dist if nearest_dist is not None else 0.0
        ),
        "least_advantaged_weighted_mobility": float(components["weighted_mobility"]),
        "least_advantaged_weighted_collision_penalty": float(
            components["weighted_collision_penalty"]
        ),
        "least_advantaged_weighted_low_speed_penalty": float(
            components["weighted_low_speed_penalty"]
        ),
        "least_advantaged_weighted_risk_penalty": float(
            components["weighted_risk_penalty"]
        ),
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


def new_reason_counters() -> dict:
    return {field: 0 for field in REASON_STEP_FIELDS}


def accumulate_reason_step(step_metrics: dict, counters: dict) -> None:
    """Increment per-episode reason count for the least advantaged vehicle."""
    reason = step_metrics.get("least_advantaged_reason", "none")
    field = f"reason_{reason}_steps"
    if field not in counters:
        field = "reason_combined_steps"
    counters[field] += 1


def finalize_reason_episode(counters: dict) -> dict:
    return dict(counters)


def collect_step_metrics(
    env: gym.Env,
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
) -> dict:
    """Per-step fairness/safety metrics for a given vehicle scope."""
    experiences = compute_experiences(
        env,
        speed_normalizer,
        scope,
        radius,
        mode,
        target_speed,
        low_speed_threshold,
        w_mobility,
        w_collision,
        w_low_speed,
        w_risk,
        risk_distance_normalizer,
    )
    all_vehicles = get_vehicles(env)
    scoped_vehicles = get_vehicles_by_scope(env, scope, radius)
    la_info = least_advantaged_vehicle_info(
        env,
        speed_normalizer,
        scope,
        radius,
        mode,
        target_speed,
        low_speed_threshold,
        w_mobility,
        w_collision,
        w_low_speed,
        w_risk,
        risk_distance_normalizer,
    )

    if scoped_vehicles:
        components_list = [
            _components_from_mode(
                vehicle,
                scoped_vehicles,
                mode,
                speed_normalizer,
                target_speed,
                low_speed_threshold,
                w_mobility,
                w_collision,
                w_low_speed,
                w_risk,
                risk_distance_normalizer,
            )
            for vehicle in scoped_vehicles
        ]
        mean_mobility_score = float(
            sum(c["mobility_score"] for c in components_list) / len(components_list)
        )
        mean_risk_penalty = float(
            sum(c["risk_penalty"] for c in components_list) / len(components_list)
        )
        mean_low_speed_penalty = float(
            sum(c["low_speed_penalty"] for c in components_list) / len(components_list)
        )
        mean_collision_penalty = float(
            sum(c["collision_penalty"] for c in components_list) / len(components_list)
        )
    else:
        mean_mobility_score = 0.0
        mean_risk_penalty = 0.0
        mean_low_speed_penalty = 0.0
        mean_collision_penalty = 0.0

    return {
        "min_experience": min_experience(experiences),
        "mean_experience": mean_experience(experiences),
        "gini_experience": gini(experiences),
        "collision_count": collision_count(env),
        "n_vehicles": len(all_vehicles),
        "metric_scope": scope,
        "scoped_vehicle_count": len(scoped_vehicles),
        "ego_neighbourhood_radius": float(radius),
        "experience_mode": mode,
        "mean_mobility_score": mean_mobility_score,
        "mean_risk_penalty": mean_risk_penalty,
        "mean_low_speed_penalty": mean_low_speed_penalty,
        "mean_collision_penalty": mean_collision_penalty,
        **la_info,
    }
