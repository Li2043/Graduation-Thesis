"""Stage-1 base reward v2 (per-transition egoistic components).

This module does **not** replace ``exp03_episode_rewards`` (absent / legacy).
It implements the frozen Stage-1 formula for learning controllers A and B:

    r_base[i,t]
        = 0.4 * delta_route_progress[i,t]
        + 0.6 * safe_exit_event[i,t+1]
        - 1.0 * stakeholder_collision_event[t+1]
        - eta_hard_brake * hard_braking_cost[i,t+1]

Route progress uses distance along the assigned route, not raw world-x.
Negative progress is preserved (not clipped to zero).

Braking uses SI sign convention: acceleration < 0 means braking.
See ``to_si_longitudinal_acceleration`` for simulator conversion notes.

``a_comfort``, ``a_hard``, and ``eta_hard_brake`` in the default config are
**test-only placeholders**, not final experimental calibrations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

LEARNING_CONTROLLERS: tuple[str, ...] = ("A", "B")
STAKEHOLDER_SET: tuple[str, ...] = ("A", "B", "B_front", "B_rear")

# Large |Δρ| on one transition is reported (not silently clipped).
DISCONTINUITY_REPORT_THRESHOLD = 0.25


def _require_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value)!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


@dataclass(frozen=True)
class BaseRewardConfig:
    """Weights and braking thresholds for Stage-1 base reward.

    Design-final weights: progress 0.4, exit 0.6, collision 1.0.
    Braking thresholds / eta are **test-only** unless overridden by a later stage.
    """

    progress_weight: float = 0.4
    exit_weight: float = 0.6
    collision_penalty: float = 1.0
    eta_hard_brake: float = 0.1  # TEST-ONLY placeholder
    a_comfort: float = 2.0  # TEST-ONLY placeholder (m/s^2, magnitude)
    a_hard: float = 6.0  # TEST-ONLY placeholder (m/s^2, magnitude)
    discontinuity_report_threshold: float = DISCONTINUITY_REPORT_THRESHOLD

    def validate(self) -> None:
        _require_finite("progress_weight", self.progress_weight)
        _require_finite("exit_weight", self.exit_weight)
        _require_finite("collision_penalty", self.collision_penalty)
        _require_finite("eta_hard_brake", self.eta_hard_brake)
        a_c = _require_finite("a_comfort", self.a_comfort)
        a_h = _require_finite("a_hard", self.a_hard)
        if not (a_c > 0.0):
            raise ValueError(f"a_comfort must be > 0, got {a_c}")
        if not (a_h > a_c):
            raise ValueError(f"a_hard must be > a_comfort, got a_hard={a_h}, a_comfort={a_c}")


@dataclass(frozen=True)
class AgentTransitionState:
    """Per-controller kinematics / route state for one transition s_t -> s_{t+1}."""

    route_position_t: float
    route_position_t1: float
    route_start: float
    route_exit: float
    acceleration: float
    """Longitudinal acceleration in SI convention (negative = braking)."""
    already_exited: bool = False
    """True if this agent already received a safe-exit bonus earlier in the episode."""


@dataclass(frozen=True)
class BaseRewardInputs:
    """Inputs for one joint transition.

    ``agents`` must include learning controllers A and B.
    ``stakeholder_collided`` must contain every member of the fixed stakeholder set V.
    Values are booleans: True means that stakeholder was involved in a collision
    during this transition.
    """

    agents: Mapping[str, AgentTransitionState]
    stakeholder_collided: Mapping[str, bool]
    terminated: bool = False
    truncated: bool = False


@dataclass
class BaseRewardBreakdown:
    progress_component: float
    exit_component: float
    collision_component: float
    hard_braking_component: float
    total_reward: float
    delta_route_progress: float
    safe_exit_event: float
    stakeholder_collision_event: float
    hard_braking_cost: float
    rho_t: float
    rho_t1: float
    warnings: list[str] = field(default_factory=list)


def to_si_longitudinal_acceleration(
    raw_acceleration: float,
    *,
    convention: str = "si_signed",
) -> float:
    """Convert simulator acceleration to SI signed longitudinal acceleration.

    Conventions
    -----------
    ``si_signed`` (default)
        Already SI: positive = accelerate, negative = brake. Pass-through.
    ``positive_braking_magnitude``
        Simulator stores braking as a non-negative magnitude. Converted to
        ``-magnitude`` (and positive throttle as ``+value`` if signed elsewhere
        is not used — callers should pass the braking magnitude only when
        braking).
    ``highway_env_action``
        highway-env continuous action acceleration uses the same signed SI
        convention (negative decelerates). Treated as ``si_signed``.

    Stage-1 unit tests always supply SI-signed values and call this helper
    with ``convention='si_signed'`` for documentation / auditability.
    """
    a = _require_finite("raw_acceleration", raw_acceleration)
    key = convention.strip().lower()
    if key in {"si_signed", "highway_env_action"}:
        return a
    if key == "positive_braking_magnitude":
        if a < 0.0:
            raise ValueError(
                "positive_braking_magnitude convention expects non-negative "
                f"braking magnitude, got {a}"
            )
        return -a
    raise ValueError(f"unknown acceleration convention {convention!r}")


def compute_normalised_route_progress(
    route_position: float,
    route_start: float,
    route_exit: float,
) -> float:
    """rho = clip((pos - start) / (exit - start), 0, 1)."""
    pos = _require_finite("route_position", route_position)
    start = _require_finite("route_start", route_start)
    exit_p = _require_finite("route_exit", route_exit)
    if exit_p <= start:
        raise ValueError(
            f"route_exit must be > route_start, got start={start}, exit={exit_p}"
        )
    rho = (pos - start) / (exit_p - start)
    if rho < 0.0:
        return 0.0
    if rho > 1.0:
        return 1.0
    return float(rho)


def compute_route_progress_delta(rho_t: float, rho_t1: float) -> float:
    """Δρ = ρ(s_{t+1}) - ρ(s_t). Negative values are preserved."""
    r0 = _require_finite("rho_t", rho_t)
    r1 = _require_finite("rho_t1", rho_t1)
    return float(r1 - r0)


def compute_hard_braking_cost(
    acceleration_si: float,
    a_comfort: float,
    a_hard: float,
) -> float:
    """H_i = clip(max(0, -a - a_comfort) / (a_hard - a_comfort), 0, 1)^2.

    ``acceleration_si`` must use SI sign (negative = braking).
    Requires a_hard > a_comfort > 0.
    """
    a = _require_finite("acceleration_si", acceleration_si)
    a_c = _require_finite("a_comfort", a_comfort)
    a_h = _require_finite("a_hard", a_hard)
    if not (a_c > 0.0):
        raise ValueError(f"a_comfort must be > 0, got {a_c}")
    if not (a_h > a_c):
        raise ValueError(f"a_hard must be > a_comfort, got a_hard={a_h}, a_comfort={a_c}")
    # a >= -a_comfort  =>  -a <= a_comfort  => excess = 0
    excess = max(0.0, -a - a_c)
    denom = a_h - a_c
    ratio = excess / denom
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    return float(ratio**2)


def _stakeholder_collision_event(
    stakeholder_collided: Mapping[str, bool],
) -> float:
    missing = [sid for sid in STAKEHOLDER_SET if sid not in stakeholder_collided]
    if missing:
        raise ValueError(
            "stakeholder_collided registry missing required stakeholders: "
            f"{missing}; required fixed set is {list(STAKEHOLDER_SET)}"
        )
    for sid in STAKEHOLDER_SET:
        if bool(stakeholder_collided[sid]):
            return 1.0
    return 0.0


def _safe_exit_event(
    *,
    route_position_t: float,
    route_position_t1: float,
    route_exit: float,
    already_exited: bool,
    stakeholder_collision_event: float,
) -> float:
    """First-time crossing of exit boundary without stakeholder collision."""
    if already_exited:
        return 0.0
    if stakeholder_collision_event >= 1.0:
        return 0.0
    pos_t = _require_finite("route_position_t", route_position_t)
    pos_t1 = _require_finite("route_position_t1", route_position_t1)
    exit_p = _require_finite("route_exit", route_exit)
    crossed = pos_t < exit_p <= pos_t1
    return 1.0 if crossed else 0.0


def compute_base_reward_for_agents(
    inputs: BaseRewardInputs,
    config: BaseRewardConfig | None = None,
) -> dict[str, BaseRewardBreakdown]:
    """Compute Stage-1 base reward decomposition for learning controllers A and B."""
    cfg = config or BaseRewardConfig()
    cfg.validate()

    for aid in LEARNING_CONTROLLERS:
        if aid not in inputs.agents:
            raise ValueError(f"inputs.agents missing learning controller {aid!r}")

    collision_event = _stakeholder_collision_event(inputs.stakeholder_collided)
    # Truncation alone must not invent exit/collision; only use provided flags.
    _ = bool(inputs.terminated)
    _ = bool(inputs.truncated)

    results: dict[str, BaseRewardBreakdown] = {}
    for aid in LEARNING_CONTROLLERS:
        st = inputs.agents[aid]
        rho_t = compute_normalised_route_progress(
            st.route_position_t, st.route_start, st.route_exit
        )
        rho_t1 = compute_normalised_route_progress(
            st.route_position_t1, st.route_start, st.route_exit
        )
        delta = compute_route_progress_delta(rho_t, rho_t1)
        warnings: list[str] = []
        if abs(delta) > cfg.discontinuity_report_threshold:
            warnings.append(
                f"route_progress_discontinuity: |Δρ|={abs(delta):.6f} "
                f"> threshold={cfg.discontinuity_report_threshold}"
            )

        exit_ev = _safe_exit_event(
            route_position_t=st.route_position_t,
            route_position_t1=st.route_position_t1,
            route_exit=st.route_exit,
            already_exited=bool(st.already_exited),
            stakeholder_collision_event=collision_event,
        )
        accel_si = to_si_longitudinal_acceleration(
            st.acceleration, convention="si_signed"
        )
        h_cost = compute_hard_braking_cost(accel_si, cfg.a_comfort, cfg.a_hard)

        progress_component = float(cfg.progress_weight * delta)
        exit_component = float(cfg.exit_weight * exit_ev)
        collision_component = float(-cfg.collision_penalty * collision_event)
        hard_braking_component = float(-cfg.eta_hard_brake * h_cost)
        total = (
            progress_component
            + exit_component
            + collision_component
            + hard_braking_component
        )
        if not math.isfinite(total):
            raise ValueError(f"non-finite total_reward for agent {aid}: {total}")

        results[aid] = BaseRewardBreakdown(
            progress_component=progress_component,
            exit_component=exit_component,
            collision_component=collision_component,
            hard_braking_component=hard_braking_component,
            total_reward=total,
            delta_route_progress=delta,
            safe_exit_event=exit_ev,
            stakeholder_collision_event=collision_event,
            hard_braking_cost=h_cost,
            rho_t=rho_t,
            rho_t1=rho_t1,
            warnings=warnings,
        )
    return results
