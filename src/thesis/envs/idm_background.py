"""Deterministic Intelligent Driver Model for Stage 4A background traffic."""

from __future__ import annotations

import math
from dataclasses import dataclass

from thesis.envs.final_environment_config import IDMProfile


@dataclass(frozen=True)
class IDMState:
    speed: float
    gap: float | None  # bumper gap to leader; None => free road
    leader_speed: float | None


def validate_idm_profile(profile: IDMProfile) -> None:
    for name in (
        "desired_speed",
        "minimum_gap",
        "desired_time_headway",
        "maximum_acceleration",
        "comfortable_deceleration",
        "acceleration_exponent",
    ):
        v = float(getattr(profile, name))
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"IDM {name} must be finite and > 0, got {v}")


def idm_acceleration(profile: IDMProfile, state: IDMState) -> float:
    """Standard IDM longitudinal acceleration (m/s^2)."""
    validate_idm_profile(profile)
    v = float(state.speed)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"speed must be finite and >= 0, got {v}")
    v0 = profile.desired_speed
    a = profile.maximum_acceleration
    b = profile.comfortable_deceleration
    delta = profile.acceleration_exponent
    s0 = profile.minimum_gap
    T = profile.desired_time_headway

    free = 1.0 - (v / v0) ** delta
    if state.gap is None or state.leader_speed is None:
        return float(a * free)

    s = float(state.gap)
    if s <= 0:
        # emergency-level braking response; finite
        return float(-min(8.0, a * 10.0))
    dv = v - float(state.leader_speed)
    s_star = s0 + max(0.0, v * T + (v * dv) / (2.0 * math.sqrt(a * b)))
    return float(a * (free - (s_star / max(s, 1e-6)) ** 2))
