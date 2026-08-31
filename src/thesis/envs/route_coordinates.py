"""Route definitions and cumulative route-position helpers (Stage 2B-1).

Geometry here is an **integration-test configuration**, not frozen dissertation
road geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

TrafficRole = Literal["mainline", "ramp"]


def _finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value)!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


@dataclass(frozen=True)
class RouteDefinition:
    """Explicit route markers along cumulative arc length (metres)."""

    role: TrafficRole
    route_start: float
    merge_start: float
    merge_end: float
    route_exit: float
    total_route_length: float
    # World-x of the mainline origin used for display / collision in shared lane.
    mainline_world_origin_x: float = 0.0
    # For ramp: arc length of the ramp approach before joining the mainline.
    ramp_approach_length: float = 0.0
    # Mainline world-x where the ramp joins.
    join_world_x: float = 50.0

    def validate(self) -> None:
        for name in (
            "route_start",
            "merge_start",
            "merge_end",
            "route_exit",
            "total_route_length",
        ):
            _finite(name, getattr(self, name))
        if not (self.route_start < self.merge_start <= self.merge_end < self.route_exit):
            raise ValueError(
                f"invalid route ordering for role={self.role}: "
                f"start={self.route_start}, merge_start={self.merge_start}, "
                f"merge_end={self.merge_end}, exit={self.route_exit}"
            )
        if self.total_route_length < self.route_exit - self.route_start:
            raise ValueError(
                f"total_route_length {self.total_route_length} shorter than "
                f"exit-start span {self.route_exit - self.route_start}"
            )


def default_mainline_route() -> RouteDefinition:
    """Integration-test mainline route (not final dissertation geometry)."""
    return RouteDefinition(
        role="mainline",
        route_start=0.0,
        merge_start=50.0,
        merge_end=80.0,
        route_exit=200.0,
        total_route_length=200.0,
        mainline_world_origin_x=0.0,
        ramp_approach_length=0.0,
        join_world_x=50.0,
    )


def default_ramp_route() -> RouteDefinition:
    """Integration-test ramp route continuous through join onto mainline.

    Route arc:
      [0, L_ramp)     ramp approach
      [L_ramp, exit]  downstream mainline after join
    """
    l_ramp = 60.0
    join_x = 50.0
    main_exit_x = 200.0
    downstream = main_exit_x - join_x  # 150
    exit_route = l_ramp + downstream  # 210
    return RouteDefinition(
        role="ramp",
        route_start=0.0,
        merge_start=l_ramp - 10.0,  # 50
        merge_end=l_ramp + 30.0,  # 90
        route_exit=exit_route,
        total_route_length=exit_route,
        mainline_world_origin_x=0.0,
        ramp_approach_length=l_ramp,
        join_world_x=join_x,
    )


def normalised_route_progress(route_position: float, route: RouteDefinition) -> float:
    """rho = clip((pos - start) / (exit - start), 0, 1)."""
    pos = _finite("route_position", route_position)
    start = route.route_start
    exit_p = route.route_exit
    if exit_p <= start:
        raise ValueError("route_exit must be > route_start")
    rho = (pos - start) / (exit_p - start)
    if rho < 0.0:
        return 0.0
    if rho > 1.0:
        return 1.0
    return float(rho)


def world_xy_from_route(route_position: float, route: RouteDefinition) -> tuple[float, float]:
    """Map cumulative route position to a simple 2D world (x, y).

    Mainline: y=0. Ramp approach: y from -lane_offset to 0 over approach length.
    After join, ramp vehicles share y=0 with the mainline.
    """
    pos = _finite("route_position", route_position)
    lane_offset = 4.0
    if route.role == "mainline":
        return float(route.mainline_world_origin_x + pos), 0.0
    # ramp
    l_ramp = route.ramp_approach_length
    if pos <= l_ramp:
        frac = 0.0 if l_ramp <= 0 else pos / l_ramp
        x = route.join_world_x - (1.0 - frac) * l_ramp
        y = -lane_offset * (1.0 - frac)
        return float(x), float(y)
    # downstream mainline
    along = pos - l_ramp
    return float(route.join_world_x + along), 0.0


def route_position_from_world_x_mainline(world_x: float, route: RouteDefinition) -> float:
    """Inverse for mainline vehicles stored by world x (integration helper)."""
    return float(_finite("world_x", world_x) - route.mainline_world_origin_x)


def advance_route_position(
    route_position: float,
    speed: float,
    dt: float,
    *,
    allow_reverse: bool = True,
) -> float:
    """Integrate route arc length: s <- s + v * dt."""
    s = _finite("route_position", route_position)
    v = _finite("speed", speed)
    d = _finite("dt", dt)
    if d <= 0.0:
        raise ValueError(f"dt must be > 0, got {d}")
    if not allow_reverse and v < 0.0:
        v = 0.0
    return float(s + v * d)


def is_on_shared_mainline(route_position: float, route: RouteDefinition) -> bool:
    """True when the vehicle occupies the shared mainline lane for collisions."""
    if route.role == "mainline":
        return True
    return route_position >= route.ramp_approach_length
