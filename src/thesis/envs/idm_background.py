"""Deterministic bounded Intelligent Driver Model (Stage 4A-0R)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from thesis.envs.final_environment_config import IDMProfile


@dataclass(frozen=True)
class IDMState:
    speed: float
    gap: float | None  # bumper gap to leader; None => free road
    leader_speed: float | None


@dataclass(frozen=True)
class IDMCommand:
    commanded_acceleration: float
    unbounded_acceleration: float


def validate_idm_profile(profile: IDMProfile) -> None:
    for name in (
        "desired_speed",
        "minimum_gap",
        "desired_time_headway",
        "maximum_acceleration",
        "comfortable_deceleration",
        "acceleration_exponent",
        "maximum_emergency_deceleration",
    ):
        v = float(getattr(profile, name))
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"IDM {name} must be finite and > 0, got {v}")


def idm_acceleration(profile: IDMProfile, state: IDMState) -> float:
    """Bounded commanded IDM acceleration (m/s^2)."""
    return idm_command(profile, state).commanded_acceleration


def idm_command(profile: IDMProfile, state: IDMState) -> IDMCommand:
    """Compute unbounded IDM then clip to emergency/accel bounds."""
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
    a_min = -float(profile.maximum_emergency_deceleration)

    free = 1.0 - (v / v0) ** delta
    if state.gap is None or state.leader_speed is None:
        unbounded = float(a * free)
    else:
        s = float(state.gap)
        if not math.isfinite(s):
            raise ValueError(f"gap must be finite or None, got {s}")
        if s <= 0:
            # Explicit invalid/overlap: command emergency bound (caller may reject)
            unbounded = a_min
        else:
            dv = v - float(state.leader_speed)
            s_star = s0 + max(0.0, v * T + (v * dv) / (2.0 * math.sqrt(a * b)))
            unbounded = float(a * (free - (s_star / max(s, 1e-9)) ** 2))

    commanded = min(max(unbounded, a_min), float(a))
    if not math.isfinite(commanded):
        raise ValueError("non-finite IDM command")
    return IDMCommand(commanded_acceleration=commanded, unbounded_acceleration=unbounded)
