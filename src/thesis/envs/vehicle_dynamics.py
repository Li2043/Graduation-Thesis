"""Longitudinal dynamics and oriented-rectangle collision (Stage 4A-0R)."""

from __future__ import annotations

import math
from typing import Literal, Sequence


ActionName = Literal["ACCELERATE", "MAINTAIN", "DECELERATE"]


def desired_acceleration(
    action: int | str, *, accel: float = 2.0, maintain: float = 0.0, decel: float = -3.0
) -> float:
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
    """Integrate with exact stopping inside a substep.

    When the vehicle stops inside the substep under negative commanded
    acceleration:
        t_stop = speed / abs(commanded_acceleration)
        distance is integrated only until t_stop;
        remaining-substep speed is zero.

    Realised acceleration is always:
        (v_new - v_old) / physics_dt
    over the full substep duration ``dt``.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if not math.isfinite(route_position) or not math.isfinite(speed) or not math.isfinite(acceleration):
        raise ValueError("non-finite kinematics")
    v0 = max(float(speed), 0.0)
    a_cmd = float(acceleration)
    s0 = float(route_position)

    # Exact stop inside substep
    if a_cmd < 0.0 and v0 > 0.0:
        t_stop = v0 / (-a_cmd)
        if t_stop < dt - 1e-15:
            s1 = s0 + v0 * t_stop + 0.5 * a_cmd * t_stop * t_stop
            v1 = 0.0
            a_real = (v1 - v0) / dt
            return float(s1), float(v1), float(a_real)

    v1 = v0 + a_cmd * dt
    v1 = min(max(v1, v_min), v_max)
    # Trapezoidal advance using realised endpoint speed
    s1 = s0 + 0.5 * (v0 + v1) * dt
    if s1 < s0 and v_min >= 0.0:
        s1 = s0
        v1 = max(v1, 0.0)
    a_real = (v1 - v0) / dt
    return float(s1), float(v1), float(a_real)


def bumper_gap_along_x(
    rear_x: float,
    front_x: float,
    *,
    vehicle_length: float = 5.0,
) -> float:
    return float(front_x - rear_x - vehicle_length)


def time_to_collision(
    *,
    gap: float,
    v_rear: float,
    v_front: float,
) -> float | None:
    if not math.isfinite(gap) or not math.isfinite(v_rear) or not math.isfinite(v_front):
        return None
    if gap <= 0.0:
        return 0.0
    closing = v_rear - v_front
    if closing <= 1e-9:
        return None
    return float(gap / closing)


def vehicle_rectangle_corners(
    x: float,
    y: float,
    heading: float,
    *,
    length: float = 5.0,
    width: float = 2.0,
) -> list[tuple[float, float]]:
    """Axis-aligned in vehicle frame, rotated by heading (centre at x,y)."""
    hx, hy = 0.5 * length, 0.5 * width
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    c, s = math.cos(heading), math.sin(heading)
    return [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]


def _axes_from_corners(corners: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    axes = []
    for i in range(len(corners)):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % len(corners)]
        ex, ey = x1 - x0, y1 - y0
        # outward normal
        nx, ny = -ey, ex
        nrm = math.hypot(nx, ny)
        if nrm < 1e-15:
            continue
        axes.append((nx / nrm, ny / nrm))
    return axes


def _project(corners: Sequence[tuple[float, float]], axis: tuple[float, float]) -> tuple[float, float]:
    ax, ay = axis
    dots = [p[0] * ax + p[1] * ay for p in corners]
    return min(dots), max(dots)


def rectangles_overlap_sat(
    corners_a: Sequence[tuple[float, float]],
    corners_b: Sequence[tuple[float, float]],
) -> bool:
    """Deterministic SAT overlap test for two convex quads."""
    for axis in _axes_from_corners(corners_a) + _axes_from_corners(corners_b):
        amin, amax = _project(corners_a, axis)
        bmin, bmax = _project(corners_b, axis)
        if amax < bmin - 1e-12 or bmax < amin - 1e-12:
            return False
    return True


def oriented_rectangles_collide(
    *,
    x1: float,
    y1: float,
    heading1: float,
    x2: float,
    y2: float,
    heading2: float,
    length: float = 5.0,
    width: float = 2.0,
) -> bool:
    c1 = vehicle_rectangle_corners(x1, y1, heading1, length=length, width=width)
    c2 = vehicle_rectangle_corners(x2, y2, heading2, length=length, width=width)
    return rectangles_overlap_sat(c1, c2)
