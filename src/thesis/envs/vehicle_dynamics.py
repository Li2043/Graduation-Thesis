"""Longitudinal vehicle dynamics helpers for Stage 4A candidate environment."""

from __future__ import annotations

import math
from typing import Literal


ActionName = Literal["ACCELERATE", "MAINTAIN", "DECELERATE"]


def desired_acceleration(action: int | str, *, accel: float = 2.0, maintain: float = 0.0, decel: float = -3.0) -> float:
    if isinstance(action, str):
        name = action.upper()
    else:
        mapping = {0: "MAINTAIN", 1: "ACCELERATE", 2: "DECELERATE"}
        name = mapping[int(action)]
    if name == "ACCELERATE":
        return float(accel)
    if name == "DECELERATE":
        return float(decel)
    if name == "MAINTAIN":
        return float(maintain)
    raise ValueError(f"unknown action {action!r}")


def integrate_longitudinal(
    *,
    route_position: float,
    speed: float,
    acceleration: float,
    dt: float,
    v_min: float = 0.0,
    v_max: float = 30.0,
) -> tuple[float, float, float]:
    """Semi-implicit Euler: clamp speed, advance route. No reverse."""
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if not math.isfinite(route_position) or not math.isfinite(speed) or not math.isfinite(acceleration):
        raise ValueError("non-finite kinematics")
    v1 = speed + acceleration * dt
    v1 = min(max(v1, v_min), v_max)
    # If clamp bites, realised accel differs
    a_real = (v1 - speed) / dt if dt > 0 else 0.0
    s1 = route_position + 0.5 * (speed + v1) * dt
    if s1 < route_position and v_min >= 0.0:
        # no reverse: freeze position if numerical noise
        s1 = route_position
        v1 = max(v1, 0.0)
    return float(s1), float(v1), float(a_real)


def bumper_gap_along_x(
    rear_x: float,
    front_x: float,
    *,
    vehicle_length: float = 5.0,
) -> float:
    """Bumper-to-bumper gap assuming centres at rear_x/front_x, same length."""
    return float(front_x - rear_x - vehicle_length)


def time_to_collision(
    *,
    gap: float,
    v_rear: float,
    v_front: float,
) -> float | None:
    """Valid same-lane TTC when rear closes on front with positive gap."""
    if not math.isfinite(gap) or not math.isfinite(v_rear) or not math.isfinite(v_front):
        return None
    if gap <= 0.0:
        return 0.0
    closing = v_rear - v_front
    if closing <= 1e-9:
        return None
    return float(gap / closing)
