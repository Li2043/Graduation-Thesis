"""Study B matched-time-to-conflict scenario generator --
``new_research_plan.md``'s "速度异质性" / "Matched time-to-conflict" /
role x speed counterbalancing sections.

Deliberately environment-free (pure functions/dataclasses, no dependency on
``Stage10SymmetricMergeEnv``) so it can be property-tested at scale (the
Phase 0 gate wants >=10,000 generated scenarios checked) without paying for
a single environment step. ``heterogeneous_env.py`` is what actually wires
a ``ScenarioSpec`` into a live environment instance.

Coordinate convention: this project's env geometry has ``merge_start`` as
the route_position where a ramp vehicle's lateral offset begins converging
toward the mainline lane (``world_y_for`` -- constant ``-LANE_OFFSET``
before ``merge_start``, linearly interpolating to 0 by ``merge_end``,
constant 0 after). That is the first point at which a ramp/mainline pair
can become laterally close enough to be collision-relevant, so it is the
natural common reference coordinate for "time to conflict" -- matching a
ramp and mainline vehicle's arrival time AT ``merge_start`` is what
produces genuine temporal overlap instead of the accidental de-confliction
this design is meant to rule out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "FRONT_TTC_SECONDS",
    "REAR_TTC_SECONDS",
    "SHARED_JITTER_RANGE",
    "INDIVIDUAL_JITTER_RANGE",
    "MIN_SAME_LANE_GAP_M",
    "V_REF",
    "V_SLOW",
    "V_FAST",
    "VehicleSpawnSpec",
    "ScenarioSpec",
    "generate_scenario",
    "matched_ttc_deltas",
]

# Phase-0 pilot defaults (new_research_plan.md): frozen only after Phase 0
# environment validation confirms them geometrically sane for this env's
# actual merge_start/merge_end/route_exit values -- see FREEZE.md once
# written; do not silently change these constants after Phase 3 begins.
FRONT_TTC_SECONDS: float = 4.5
REAR_TTC_SECONDS: float = 6.5
SHARED_JITTER_RANGE: tuple[float, float] = (-0.25, 0.25)
INDIVIDUAL_JITTER_RANGE: tuple[float, float] = (-0.05, 0.05)
MIN_SAME_LANE_GAP_M: float = 15.0
V_REF: float = 20.0
V_SLOW: float = 18.0
V_FAST: float = 22.0


@dataclass(frozen=True)
class VehicleSpawnSpec:
    vehicle_id: str
    role: str  # "ramp" | "mainline"
    speed_class: str  # "fast" | "slow" | "homogeneous"
    ttc_slot: str  # "front" | "rear"
    target_speed: float
    spawn_speed: float  # == target_speed, per new_research_plan.md's v_{i,0} = v_i^target
    route_position: float  # absolute route coordinate at spawn
    nominal_ttc: float  # T_{i,0}^nominal actually used to derive route_position


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    episode_seed: int
    traffic_type: str  # "heterogeneous" | "homogeneous"
    vehicles: dict[str, VehicleSpawnSpec]

    def role_members(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for vid, spec in self.vehicles.items():
            out.setdefault(spec.role, []).append(vid)
        return out

    def to_json_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "episode_seed": self.episode_seed,
            "traffic_type": self.traffic_type,
            "agents": [
                {
                    "vehicle_id": v.vehicle_id,
                    "role": v.role,
                    "speed_class": v.speed_class,
                    "ttc_slot": v.ttc_slot,
                    "target_speed": v.target_speed,
                    "spawn_speed": v.spawn_speed,
                    "route_position": v.route_position,
                    "nominal_ttc": v.nominal_ttc,
                }
                for v in self.vehicles.values()
            ],
        }


def _same_lane_pairs_valid(vehicles: Sequence[VehicleSpawnSpec], *, min_gap: float) -> bool:
    """Same role == same lane before merge_start (world_y_for is constant
    per role in that range) -- checks every same-role pair's route_position
    separation, per new_research_plan.md's spawn-validity rule."""
    by_role: dict[str, list[float]] = {}
    for v in vehicles:
        by_role.setdefault(v.role, []).append(v.route_position)
    for positions in by_role.values():
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if abs(positions[i] - positions[j]) < min_gap:
                    return False
    return True


def generate_scenario(
    *,
    scenario_id: str,
    episode_seed: int,
    role_members: Mapping[str, Sequence[str]],
    traffic_type: str = "heterogeneous",
    v_ref: float = V_REF,
    v_slow: float = V_SLOW,
    v_fast: float = V_FAST,
    front_ttc: float = FRONT_TTC_SECONDS,
    rear_ttc: float = REAR_TTC_SECONDS,
    shared_jitter_range: tuple[float, float] = SHARED_JITTER_RANGE,
    individual_jitter_range: tuple[float, float] = INDIVIDUAL_JITTER_RANGE,
    merge_start: float = 200.0,
    min_same_lane_gap: float = MIN_SAME_LANE_GAP_M,
    max_resample_attempts: int = 50,
) -> ScenarioSpec:
    """Generate one matched-TTC N=4 scenario (2 ramp + 2 mainline, exactly
    2 members per role, per this project's frozen N=4 design).

    ``role_members``: e.g. ``{"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]}``
    -- deliberately NOT hardcoded to specific vehicle ids, since Study B's
    physical-identity-vs-role decoupling is handled upstream (whichever ids
    the live env assigned to each role this episode).

    ``traffic_type="homogeneous"``: every vehicle gets ``v_ref`` as its
    target/spawn speed instead of the fast/slow split -- used only for the
    H0 sensitivity bank, never for training.

    Raises ``RuntimeError`` if no valid (same-lane-gap-respecting) spawn is
    found within ``max_resample_attempts`` -- this should be extremely rare
    given the ~2s nominal gap between waves at 18-22 m/s (~36-44m expected
    separation, far above the 15m floor), so repeated failures likely
    indicate a misconfigured ``front_ttc``/``rear_ttc`` pair rather than bad
    luck; do not silently loop forever."""
    if traffic_type not in ("heterogeneous", "homogeneous"):
        raise ValueError(f"traffic_type must be 'heterogeneous' or 'homogeneous', got {traffic_type!r}")
    roles = list(role_members.keys())
    if len(roles) != 2:
        raise ValueError(f"expected exactly 2 roles, got {roles}")
    for role, members in role_members.items():
        if len(members) != 2:
            raise ValueError(f"role {role!r} must have exactly 2 members, got {members}")

    rng = np.random.default_rng(episode_seed)

    for _attempt in range(max_resample_attempts):
        shared_jitter = float(rng.uniform(*shared_jitter_range))
        vehicles: dict[str, VehicleSpawnSpec] = {}

        for role, members in role_members.items():
            members = list(members)
            # Two INDEPENDENT coin flips (speed-class assignment, ttc-slot
            # assignment) per role -- guarantees no artificial coupling
            # between "is fast" and "is front", satisfying the
            # counterbalancing requirement across many draws without
            # needing to track cross-scenario balance explicitly.
            speed_order = rng.permutation(members)
            slot_order = rng.permutation(members)

            for vid, speed_class in zip(speed_order, ["fast", "slow"]):
                vid = str(vid)
                target_speed = v_fast if speed_class == "fast" else v_slow
                if traffic_type == "homogeneous":
                    target_speed = v_ref
                    speed_class = "homogeneous"
                vehicles[vid] = {"role": role, "speed_class": speed_class, "target_speed": target_speed}  # type: ignore[dict-item]

            for vid, ttc_slot in zip(slot_order, ["front", "rear"]):
                vid = str(vid)
                nominal_ttc_base = front_ttc if ttc_slot == "front" else rear_ttc
                zeta = float(rng.uniform(*individual_jitter_range))
                nominal_ttc = nominal_ttc_base + shared_jitter + zeta
                info = vehicles[vid]
                target_speed = info["target_speed"]  # type: ignore[index]
                route_position = float(merge_start - target_speed * nominal_ttc)
                vehicles[vid] = VehicleSpawnSpec(
                    vehicle_id=vid,
                    role=info["role"],  # type: ignore[index]
                    speed_class=info["speed_class"],  # type: ignore[index]
                    ttc_slot=ttc_slot,
                    target_speed=float(target_speed),
                    spawn_speed=float(target_speed),
                    route_position=route_position,
                    nominal_ttc=nominal_ttc,
                )

        if _same_lane_pairs_valid(list(vehicles.values()), min_gap=min_same_lane_gap):
            return ScenarioSpec(
                scenario_id=scenario_id,
                episode_seed=episode_seed,
                traffic_type=traffic_type,
                vehicles=vehicles,
            )

    raise RuntimeError(
        f"could not generate a valid matched-TTC scenario for seed={episode_seed} "
        f"within {max_resample_attempts} attempts -- check front_ttc/rear_ttc separation"
    )


def matched_ttc_deltas(scenario: ScenarioSpec) -> dict[str, float]:
    """|delta TTC| between the two ramp/mainline pairs that share a
    ttc_slot ("front" pair, "rear" pair) -- the NOMINAL construction-time
    values, i.e. a self-consistency check on the generator's own arithmetic,
    NOT a substitute for Phase 0's real scripted-physics matched-TTC test
    (nominal spawn timing and actual simulated arrival can differ once
    acceleration dynamics are involved -- see
    ``scripts/phase0_validate_env.py``)."""
    by_slot: dict[str, list[float]] = {}
    for v in scenario.vehicles.values():
        by_slot.setdefault(v.ttc_slot, []).append(v.nominal_ttc)
    out: dict[str, float] = {}
    for slot, ttcs in by_slot.items():
        if len(ttcs) != 2:
            raise ValueError(f"expected exactly 2 vehicles in ttc_slot={slot!r}, got {ttcs}")
        out[slot] = abs(ttcs[0] - ttcs[1])
    return out
