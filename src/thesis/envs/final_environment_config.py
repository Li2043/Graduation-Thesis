"""Final-candidate environment configuration types (Stage 4A).

Comfort / eta / PBRS lambda / DQN settings are intentionally excluded from the
environment freeze surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TrafficRole = Literal["mainline", "ramp"]


@dataclass(frozen=True)
class GeometryCandidate:
    geometry_id: str
    mainline_route_start: float
    ramp_route_start: float
    merge_start: float
    merge_end: float
    downstream_exit: float
    priority_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IDMProfile:
    profile_id: str
    desired_speed: float
    minimum_gap: float
    desired_time_headway: float
    maximum_acceleration: float
    comfortable_deceleration: float
    acceleration_exponent: float
    priority_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentCandidate:
    candidate_id: str
    geometry: GeometryCandidate
    idm: IDMProfile
    priority_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "priority_rank": self.priority_rank,
            "geometry": self.geometry.to_dict(),
            "idm": self.idm.to_dict(),
        }


@dataclass(frozen=True)
class TimingConfig:
    physics_dt: float = 0.05
    policy_interval: float = 0.20
    physics_substeps_per_action: int = 4

    def validate(self) -> None:
        if self.physics_dt <= 0 or self.policy_interval <= 0:
            raise ValueError("timing intervals must be > 0")
        if self.physics_substeps_per_action <= 0:
            raise ValueError("substeps must be > 0")
        expected = self.physics_dt * self.physics_substeps_per_action
        if abs(expected - self.policy_interval) > 1e-12:
            raise ValueError(
                f"physics_dt*substeps ({expected}) != policy_interval ({self.policy_interval})"
            )


@dataclass(frozen=True)
class VehicleGeometry:
    length: float = 5.0
    width: float = 2.0


@dataclass(frozen=True)
class LearningDynamics:
    accel: float = 2.0
    maintain: float = 0.0
    decel: float = -3.0
    v_min: float = 0.0
    v_max: float = 30.0


@dataclass(frozen=True)
class TargetSpeeds:
    A: float = 20.0
    B: float = 20.0
    B_front: float = 20.0
    B_rear: float = 20.0

    def as_map(self) -> dict[str, float]:
        return {"A": self.A, "B": self.B, "B_front": self.B_front, "B_rear": self.B_rear}


@dataclass
class InitialConditionBlock:
    block_id: str
    block_set: str  # calibration | validation
    seed: int
    role_A: str
    role_B: str
    spawn_route_mainline: float
    spawn_route_ramp: float
    spawn_speed_mainline: float
    spawn_speed_ramp: float
    spawn_route_B_front: float
    spawn_route_B_rear: float
    spawn_speed_B_front: float
    spawn_speed_B_rear: float
    delta_arrival: float
    arrival_category: str
    background_time_headway: float
    target_speeds: TargetSpeeds = field(default_factory=TargetSpeeds)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def build_routes_for_geometry(geom: GeometryCandidate) -> dict[str, dict[str, float]]:
    """Return serialisable route markers for mainline and ramp."""
    join_x = float(geom.merge_start)
    l_ramp = float(geom.merge_start) - float(geom.ramp_route_start)
    downstream = float(geom.downstream_exit) - join_x
    ramp_exit = l_ramp + downstream
    return {
        "mainline": {
            "route_start": float(geom.mainline_route_start),
            "merge_start": float(geom.merge_start),
            "merge_end": float(geom.merge_end),
            "route_exit": float(geom.downstream_exit),
            "total_route_length": float(geom.downstream_exit),
            "ramp_approach_length": 0.0,
            "join_world_x": join_x,
        },
        "ramp": {
            "route_start": float(geom.ramp_route_start),
            "merge_start": l_ramp - 10.0 if l_ramp >= 10.0 else 0.5 * l_ramp,
            "merge_end": l_ramp + (float(geom.merge_end) - join_x),
            "route_exit": ramp_exit,
            "total_route_length": ramp_exit,
            "ramp_approach_length": l_ramp,
            "join_world_x": join_x,
        },
    }
