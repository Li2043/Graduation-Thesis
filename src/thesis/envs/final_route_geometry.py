"""Parameterised merge centreline with quintic convergence (Stage 4A-0R2).

Semantics:
    merge_start / merge_end define the ramp→mainline convergence interval.

Before merge_start:
    mainline y = 0, ramp y = -lateral_offset (parallel)
Between merge_start and merge_end:
    ramp follows a C^2 quintic lateral transition onto y = 0
After merge_end:
    both share the mainline centreline

Route position is true cumulative path arc length (connector uses a LUT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

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

# Pre-registered physical plausibility bound (geometry feasibility, not reward)
MAX_LATERAL_ACCEL_AT_20 = 3.0  # m/s^2
LUT_N = 8192


def quintic_q(u: float | np.ndarray) -> float | np.ndarray:
    """q(u) = 10u^3 - 15u^4 + 6u^5."""
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def quintic_qp(u: float | np.ndarray) -> float | np.ndarray:
    """dq/du."""
    return 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4


def quintic_qpp(u: float | np.ndarray) -> float | np.ndarray:
    """d²q/du²."""
    return 60.0 * u - 180.0 * u**2 + 120.0 * u**3


@dataclass(frozen=True)
class WorldPose:
    x: float
    y: float
    heading: float
    route_position: float
    segment: Segment
    curvature: float = 0.0


@dataclass
class FinalRouteGeometry:
    """Arc-length parameterised dual-route merge geometry (quintic connector)."""

    geometry: GeometryCandidate
    lateral_offset: float = 4.0
    _u_table: np.ndarray = field(init=False, repr=False)
    _s_table: np.ndarray = field(init=False, repr=False)
    _connector_arc: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lateral_offset <= 0:
            raise ValueError("lateral_offset must be > 0")
        if self.merge_end <= self.merge_start:
            raise ValueError("merge_end must be > merge_start")
        if self.downstream_exit <= self.merge_end:
            raise ValueError("downstream_exit must be > merge_end")
        self._build_connector_lut()

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
    def connector_world_length(self) -> float:
        return self.merge_end - self.merge_start

    @property
    def connector_arc_length(self) -> float:
        return float(self._connector_arc)

    @property
    def ramp_approach_length(self) -> float:
        """Parallel approach ends at merge_start (world-x == route-s)."""
        return self.merge_start

    @property
    def ramp_connector_end_route(self) -> float:
        return self.ramp_approach_length + self.connector_arc_length

    @property
    def ramp_exit_route(self) -> float:
        return self.ramp_connector_end_route + (self.downstream_exit - self.merge_end)

    def mainline_exit_route(self) -> float:
        return self.downstream_exit

    def exit_route(self, role: Role) -> float:
        return self.mainline_exit_route() if role == "mainline" else self.ramp_exit_route

    def merge_route_marker(self, role: Role) -> float:
        """Route position at the start of convergence (merge_start)."""
        return self.merge_start

    def _build_connector_lut(self) -> None:
        Lx = self.connector_world_length
        L = float(self.lateral_offset)
        u = np.linspace(0.0, 1.0, LUT_N + 1)
        qp = quintic_qp(u)
        dsdu = np.sqrt(Lx**2 + (L * qp) ** 2)
        du = u[1] - u[0]
        s = np.zeros_like(u)
        s[1:] = np.cumsum(0.5 * (dsdu[1:] + dsdu[:-1]) * du)
        self._u_table = u
        self._s_table = s
        self._connector_arc = float(s[-1])

    def u_from_connector_arc(self, s_conn: float) -> float:
        s_conn = float(min(max(s_conn, 0.0), self._connector_arc))
        return float(np.interp(s_conn, self._s_table, self._u_table))

    def connector_arc_from_u(self, u: float) -> float:
        u = float(min(max(u, 0.0), 1.0))
        return float(np.interp(u, self._u_table, self._s_table))

    def _ramp_y_heading_curvature(self, u: float) -> tuple[float, float, float]:
        L = float(self.lateral_offset)
        Lx = self.connector_world_length
        q = float(quintic_q(u))
        qp = float(quintic_qp(u))
        qpp = float(quintic_qpp(u))
        y = -L + L * q
        dy_dx = (L / Lx) * qp
        d2y_dx2 = (L / (Lx * Lx)) * qpp
        heading = math.atan(dy_dx)
        curv = d2y_dx2 / (1.0 + dy_dx * dy_dx) ** 1.5
        return y, heading, curv

    def pose(self, role: Role, route_position: float) -> WorldPose:
        if role == "mainline":
            return self._pose_mainline(route_position)
        return self._pose_ramp(route_position)

    def segment(self, role: Role, route_position: float) -> Segment:
        return self.pose(role, route_position).segment

    def _pose_mainline(self, s: float) -> WorldPose:
        s = float(s)
        exit_s = self.downstream_exit
        if s >= exit_s - 1e-12:
            return WorldPose(exit_s, 0.0, 0.0, s, "exited", 0.0)
        if s < self.merge_start:
            seg: Segment = "mainline_approach"
        elif s < self.merge_end:
            seg = "merge_conflict"
        else:
            seg = "shared_mainline"
        return WorldPose(s, 0.0, 0.0, s, seg, 0.0)

    def _pose_ramp(self, s: float) -> WorldPose:
        s = float(s)
        L_ap = self.ramp_approach_length
        L_arc = self.connector_arc_length
        s_conn_end = self.ramp_connector_end_route
        exit_s = self.ramp_exit_route
        L = float(self.lateral_offset)

        if s >= exit_s - 1e-12:
            x = self.merge_end + (s - s_conn_end)
            return WorldPose(float(x), 0.0, 0.0, s, "exited", 0.0)

        if s <= L_ap + 1e-12:
            # Parallel approach: world-x == route-s, y = -offset, heading 0
            return WorldPose(float(s), -L, 0.0, s, "ramp_approach", 0.0)

        if s < s_conn_end - 1e-12:
            s_conn = s - L_ap
            u = self.u_from_connector_arc(s_conn)
            x = self.merge_start + u * self.connector_world_length
            y, heading, curv = self._ramp_y_heading_curvature(u)
            return WorldPose(float(x), float(y), float(heading), s, "merge_connector", float(curv))

        # Shared mainline after full convergence
        x = self.merge_end + (s - s_conn_end)
        return WorldPose(float(x), 0.0, 0.0, s, "shared_mainline", 0.0)

    def recover_route_position(self, role: Role, x: float, y: float) -> float:
        """Piecewise analytic / bounded inversion; error target ≤ 0.01 m."""
        x = float(x)
        y = float(y)
        if role == "mainline":
            return float(min(max(x, 0.0), self.downstream_exit + 5.0))

        L = float(self.lateral_offset)
        if x <= self.merge_start + 1e-12:
            # Parallel approach
            return float(min(max(x, 0.0), self.merge_start))

        if x >= self.merge_end - 1e-12:
            return float(self.ramp_connector_end_route + (x - self.merge_end))

        # Connector: world-x determines u analytically; refine if y disagrees slightly
        u = (x - self.merge_start) / self.connector_world_length
        u = min(max(u, 0.0), 1.0)
        # Optional y-based correction via Newton on y(u) - y_obs
        y_u, _, _ = self._ramp_y_heading_curvature(u)
        if abs(y_u - y) > 1e-9:
            for _ in range(8):
                # dy/du = L * q'(u)
                qp = float(quintic_qp(u))
                dydu = L * qp
                if abs(dydu) < 1e-14:
                    break
                u_new = u - (y_u - y) / dydu
                u = min(max(u_new, 0.0), 1.0)
                y_u, _, _ = self._ramp_y_heading_curvature(u)
                if abs(y_u - y) < 1e-12:
                    break
        return float(self.ramp_approach_length + self.connector_arc_from_u(u))

    def curvature_samples(self, role: Role, n: int = 2000) -> list[tuple[float, float, float, float]]:
        """(s, heading, curvature, |a_lat| at 20 m/s) samples."""
        v = 20.0
        out = []
        exit_s = self.exit_route(role)
        for i in range(n + 1):
            s = exit_s * i / n
            p = self.pose(role, s)
            a_lat = (v * v) * abs(p.curvature)
            out.append((s, p.heading, p.curvature, a_lat))
        return out

    def diagnostics(self, *, v_ref: float = 20.0) -> dict[str, Any]:
        samples = self.curvature_samples("ramp", n=4000)
        headings = [abs(h) for _, h, _, _ in samples]
        curvs = [abs(c) for _, _, c, _ in samples]
        a_lats = [(v_ref * v_ref) * c for c in curvs]
        max_curv = max(curvs) if curvs else 0.0
        min_radius = float("inf") if max_curv < 1e-15 else 1.0 / max_curv
        # Boundary jumps
        eps = 1e-9
        p0a = self.pose("ramp", self.merge_start - eps)
        p0b = self.pose("ramp", self.merge_start + eps)
        p1a = self.pose("ramp", self.ramp_connector_end_route - eps)
        p1b = self.pose("ramp", self.ramp_connector_end_route + eps)
        # Also check at merge_end world via u=1-
        p_u1 = self.pose("ramp", self.ramp_connector_end_route)
        return {
            "geometry_id": self.geometry.geometry_id,
            "merge_start": self.merge_start,
            "merge_end": self.merge_end,
            "connector_world_x_length": self.connector_world_length,
            "connector_arc_length": self.connector_arc_length,
            "lateral_offset": float(self.lateral_offset),
            "maximum_abs_heading": max(headings) if headings else 0.0,
            "maximum_abs_curvature": max_curv,
            "minimum_curvature_radius": min_radius,
            "maximum_implied_lateral_acceleration_at_20": max(a_lats) if a_lats else 0.0,
            "lateral_accel_bound": MAX_LATERAL_ACCEL_AT_20,
            "physically_feasible": (max(a_lats) if a_lats else 0.0) <= MAX_LATERAL_ACCEL_AT_20 + 1e-12,
            "boundary_position_jump_merge_start": math.hypot(p0b.x - p0a.x, p0b.y - p0a.y),
            "boundary_heading_jump_merge_start": abs(p0b.heading - p0a.heading),
            "boundary_curvature_jump_merge_start": abs(p0b.curvature - p0a.curvature),
            "boundary_position_jump_merge_end": math.hypot(p1b.x - p1a.x, p1b.y - p1a.y),
            "boundary_heading_jump_merge_end": abs(p1b.heading - p1a.heading),
            "boundary_curvature_jump_merge_end": abs(p1b.curvature - p1a.curvature),
            "ramp_y_at_merge_end": p_u1.y,
        }

    def max_route_recovery_error(self, role: Role, n: int = 1000) -> float:
        err = 0.0
        exit_s = self.exit_route(role)
        for i in range(n):
            s = exit_s * (i + 0.5) / n
            p = self.pose(role, s)
            s_rec = self.recover_route_position(role, p.x, p.y)
            err = max(err, abs(s_rec - s))
        return float(err)


def build_final_route_geometry(geometry: GeometryCandidate, *, lateral_offset: float = 4.0) -> FinalRouteGeometry:
    return FinalRouteGeometry(geometry=geometry, lateral_offset=lateral_offset)
