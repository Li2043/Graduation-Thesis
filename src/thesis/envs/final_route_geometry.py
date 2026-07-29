"""Parameterised centreline geometry with true cumulative arc length (Stage 4A-0R).

Segments:
    mainline_approach, ramp_approach, merge_connector, merge_conflict,
    shared_mainline, exited

Ramp connector is a circular arc providing continuous world (x, y), heading,
and cumulative arc length at both boundaries. This does **not** reuse the
linear V2 world mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from thesis.envs.final_environment_config import GeometryCandidate

Role = Literal["mainline", "ramp"]
Segment = Literal[
    "mainline_approach",
    "ramp_approach",
    "merge_connector",
    "merge_conflict",
    "shared_mainline",
    "exited",
]


@dataclass(frozen=True)
class WorldPose:
    x: float
    y: float
    heading: float
    route_position: float
    segment: Segment


@dataclass(frozen=True)
class FinalRouteGeometry:
    """Arc-length parameterised dual-route merge geometry."""

    geometry: GeometryCandidate
    lateral_offset: float = 4.0  # also arc radius for the quarter-connector

    def __post_init__(self) -> None:
        if self.lateral_offset <= 0:
            raise ValueError("lateral_offset must be > 0")
        if self.connector_arc_length >= float(self.geometry.merge_start) - 1.0:
            raise ValueError("connector longer than ramp approach budget")

    @property
    def join_x(self) -> float:
        return float(self.geometry.merge_start)

    @property
    def merge_start(self) -> float:
        return float(self.geometry.merge_start)

    @property
    def merge_end(self) -> float:
        return float(self.geometry.merge_end)

    @property
    def downstream_exit(self) -> float:
        return float(self.geometry.downstream_exit)

    @property
    def R(self) -> float:
        return float(self.lateral_offset)

    @property
    def connector_theta0(self) -> float:
        return 0.5 * math.pi

    @property
    def connector_arc_length(self) -> float:
        return self.R * self.connector_theta0

    @property
    def ramp_straight_length(self) -> float:
        return float(self.geometry.merge_start) - self.connector_arc_length

    @property
    def ramp_join_route(self) -> float:
        """Route position on the ramp at the mainline join."""
        return float(self.geometry.merge_start)

    @property
    def ramp_exit_route(self) -> float:
        # Shared downstream length after join
        return self.ramp_join_route + (self.downstream_exit - self.join_x)

    def mainline_exit_route(self) -> float:
        return self.downstream_exit

    def exit_route(self, role: Role) -> float:
        return self.mainline_exit_route() if role == "mainline" else self.ramp_exit_route

    def merge_route_marker(self, role: Role) -> float:
        """Route position of merge entry for the role."""
        return self.merge_start if role == "mainline" else self.ramp_join_route

    def pose(self, role: Role, route_position: float) -> WorldPose:
        if role == "mainline":
            return self._pose_mainline(route_position)
        return self._pose_ramp(route_position)

    def segment(self, role: Role, route_position: float) -> Segment:
        return self.pose(role, route_position).segment

    def _pose_mainline(self, s: float) -> WorldPose:
        exit_s = self.downstream_exit
        if s >= exit_s - 1e-12:
            return WorldPose(exit_s, 0.0, 0.0, s, "exited")
        if s < self.merge_start:
            seg: Segment = "mainline_approach"
        elif s < self.merge_end:
            seg = "merge_conflict"
        else:
            seg = "shared_mainline"
        return WorldPose(float(s), 0.0, 0.0, float(s), seg)

    def _pose_ramp(self, s: float) -> WorldPose:
        s = float(s)
        L_st = self.ramp_straight_length
        L_conn = self.connector_arc_length
        join = self.ramp_join_route
        exit_s = self.ramp_exit_route
        R = self.R
        th0 = self.connector_theta0

        if s >= exit_s - 1e-12:
            # Exited along shared mainline x
            x = self.join_x + (s - join)
            return WorldPose(x, 0.0, 0.0, s, "exited")

        if s <= L_st + 1e-12:
            # Straight approach: northbound into connector start
            # Start: (join_x - R, -R - L_st), heading +π/2
            x = self.join_x - R
            y = -R - L_st + s
            return WorldPose(x, y, 0.5 * math.pi, s, "ramp_approach")

        if s < join - 1e-12:
            # Circular connector: θ decreases from th0 → 0
            u = s - L_st  # ∈ (0, L_conn)
            theta = th0 - u / R
            x = self.join_x - R * math.sin(theta)
            y = -R * (1.0 - math.cos(theta))
            heading = theta  # tangent when θ decreases along +s
            return WorldPose(x, y, heading, s, "merge_connector")

        # Past join: shared mainline / conflict in mainline coordinates
        mainline_s = self.join_x + (s - join)
        if mainline_s < self.merge_end:
            seg: Segment = "merge_conflict"
        else:
            seg = "shared_mainline"
        return WorldPose(float(mainline_s), 0.0, 0.0, s, seg)

    def recover_route_position(self, role: Role, x: float, y: float, *, tol: float = 1e-6) -> float:
        """Recover route position by 1-D search along the centreline (round-trip)."""
        exit_s = self.exit_route(role)
        lo, hi = 0.0, exit_s + 5.0
        best_s, best_d = 0.0, float("inf")
        # Coarse grid then refine
        for _ in range(2):
            n = 400
            for i in range(n + 1):
                s = lo + (hi - lo) * i / n
                p = self.pose(role, s)
                d = (p.x - x) ** 2 + (p.y - y) ** 2
                if d < best_d:
                    best_d = d
                    best_s = s
            lo = max(0.0, best_s - (hi - lo) / n * 2)
            hi = min(exit_s + 5.0, best_s + (hi - lo) / n * 2 + 1e-9)
        if best_d > tol**2 * 100:
            # still return best projection
            pass
        return float(best_s)

    def heading_continuity_samples(self, role: Role, n: int = 200) -> list[tuple[float, float, float]]:
        """Return (s, heading, dheading/ds estimate) samples up to exit."""
        exit_s = self.exit_route(role)
        out = []
        prev_h = None
        prev_s = None
        for i in range(n + 1):
            s = exit_s * i / n
            h = self.pose(role, s).heading
            dh = 0.0 if prev_h is None else (h - prev_h) / max(s - prev_s, 1e-12)  # type: ignore[operator]
            out.append((s, h, dh))
            prev_h, prev_s = h, s
        return out


def build_final_route_geometry(geometry: GeometryCandidate) -> FinalRouteGeometry:
    return FinalRouteGeometry(geometry=geometry)
