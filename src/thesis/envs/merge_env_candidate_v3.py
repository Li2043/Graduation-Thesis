"""Stage 4A candidate merge environment with physics substeps.

Policy interval 0.20 s = 4 × physics_dt 0.05 s. Core reward only for ranking
(comfort / eta excluded from selection objective).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from thesis.envs.final_environment_config import (
    EnvironmentCandidate,
    InitialConditionBlock,
    LearningDynamics,
    TargetSpeeds,
    TimingConfig,
    VehicleGeometry,
    build_routes_for_geometry,
)
from thesis.envs.idm_background import IDMState, idm_acceleration
from thesis.envs.route_coordinates import (
    RouteDefinition,
    is_on_shared_mainline,
    normalised_route_progress,
)
from thesis.envs.vehicle_dynamics import (
    bumper_gap_along_x,
    desired_acceleration,
    integrate_longitudinal,
    time_to_collision,
)
from thesis.rewards.base_reward_v2 import LEARNING_CONTROLLERS, STAKEHOLDER_SET


class HighLevelAction(IntEnum):
    MAINTAIN = 0
    ACCELERATE = 1
    DECELERATE = 2


@dataclass
class VehicleState:
    identity: str
    role: str
    route_position: float
    speed: float
    acceleration: float = 0.0
    world_x: float = 0.0
    world_y: float = 0.0
    completed: bool = False
    target_speed: float = 20.0


def _route_from_dict(role: str, d: dict[str, float]) -> RouteDefinition:
    return RouteDefinition(
        role=role,  # type: ignore[arg-type]
        route_start=float(d["route_start"]),
        merge_start=float(d["merge_start"]),
        merge_end=float(d["merge_end"]),
        route_exit=float(d["route_exit"]),
        total_route_length=float(d["total_route_length"]),
        ramp_approach_length=float(d["ramp_approach_length"]),
        join_world_x=float(d["join_world_x"]),
    )


def world_xy(route_position: float, route: RouteDefinition) -> tuple[float, float]:
    """Map route position to world (x,y); continuous at ramp join."""
    if route.role == "mainline":
        return float(route.mainline_world_origin_x + route_position), 0.0
    l_ramp = route.ramp_approach_length
    if route_position <= l_ramp:
        frac = 0.0 if l_ramp <= 0 else route_position / l_ramp
        x = route.join_world_x - (1.0 - frac) * l_ramp
        y = -4.0 * (1.0 - frac)
        return float(x), float(y)
    return float(route.join_world_x + (route_position - l_ramp)), 0.0


@dataclass
class MergeEnvCandidateV3Config:
    candidate: EnvironmentCandidate
    block: InitialConditionBlock
    timing: TimingConfig = field(default_factory=TimingConfig)
    vehicle: VehicleGeometry = field(default_factory=VehicleGeometry)
    dynamics: LearningDynamics = field(default_factory=LearningDynamics)
    max_policy_steps: int = 400
    collision_bumper_gap: float = 0.0  # collide when gap <= 0
    discontinuity_threshold: float = 0.5

    def routes(self) -> dict[str, RouteDefinition]:
        raw = build_routes_for_geometry(self.candidate.geometry)
        return {k: _route_from_dict(k, v) for k, v in raw.items()}


class MergeEnvCandidateV3(gym.Env):
    """Gymnasium env: one step() = one 0.20 s policy transition (4 substeps)."""

    metadata = {"render_modes": []}

    def __init__(self, config: MergeEnvCandidateV3Config):
        super().__init__()
        self.config = config
        self.config.timing.validate()
        self._routes = config.routes()
        self.action_space = spaces.Dict(
            {
                "A": spaces.Discrete(3),
                "B": spaces.Discrete(3),
            }
        )
        self.observation_space = spaces.Dict(
            {
                aid: spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
                for aid in LEARNING_CONTROLLERS
            }
        )
        self._vehicles: dict[str, VehicleState] = {}
        self._policy_step = 0
        self._rng = np.random.default_rng(0)
        self._prev_route: dict[str, float] = {}
        self._exit_count = {"A": 0, "B": 0}
        self._exit_time: dict[str, int | None] = {"A": None, "B": None}

    def _route_for(self, role: str) -> RouteDefinition:
        return self._routes[role]

    def _sync(self, veh: VehicleState) -> None:
        x, y = world_xy(veh.route_position, self._route_for(veh.role))
        veh.world_x, veh.world_y = x, y

    def _obs(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for aid in LEARNING_CONTROLLERS:
            veh = self._vehicles[aid]
            route = self._route_for(veh.role)
            rho = normalised_route_progress(veh.route_position, route)
            out[aid] = np.array(
                [veh.route_position, veh.speed, rho, float(veh.completed)], dtype=np.float32
            )
        return out

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed = self.config.block.seed
        super().reset(seed=seed)
        self._rng = np.random.default_rng(int(seed))
        self._policy_step = 0
        self._exit_count = {"A": 0, "B": 0}
        self._exit_time = {"A": None, "B": None}
        b = self.config.block
        targets = b.target_speeds.as_map()
        # Physical role spawn → controller assignment
        role_spawn = {
            "mainline": (b.spawn_route_mainline, b.spawn_speed_mainline),
            "ramp": (b.spawn_route_ramp, b.spawn_speed_ramp),
        }
        self._vehicles = {}
        for aid, role in (("A", b.role_A), ("B", b.role_B)):
            s, v = role_spawn[role]
            veh = VehicleState(
                identity=aid,
                role=role,
                route_position=float(s),
                speed=float(v),
                target_speed=float(targets[aid]),
            )
            self._sync(veh)
            self._vehicles[aid] = veh
        for bid, route_pos, speed in (
            ("B_front", b.spawn_route_B_front, b.spawn_speed_B_front),
            ("B_rear", b.spawn_route_B_rear, b.spawn_speed_B_rear),
        ):
            veh = VehicleState(
                identity=bid,
                role="mainline",
                route_position=float(route_pos),
                speed=float(speed),
                target_speed=float(targets[bid]),
            )
            self._sync(veh)
            self._vehicles[bid] = veh
        self._prev_route = {k: v.route_position for k, v in self._vehicles.items()}
        return self._obs(), {"stakeholder_set": list(STAKEHOLDER_SET), "policy_step": 0}

    def _leader_for(self, sid: str) -> tuple[str | None, float | None]:
        """Nearest vehicle ahead on shared mainline by world_x."""
        veh = self._vehicles[sid]
        if not is_on_shared_mainline(veh.route_position, self._route_for(veh.role)):
            # ramp approach: no mainline leader for IDM gap (free-ish)
            if veh.role == "ramp":
                return None, None
        candidates = []
        for oid, other in self._vehicles.items():
            if oid == sid or other.completed:
                continue
            if not is_on_shared_mainline(other.route_position, self._route_for(other.role)):
                continue
            if other.world_x > veh.world_x + 1e-9:
                gap = bumper_gap_along_x(
                    veh.world_x, other.world_x, vehicle_length=self.config.vehicle.length
                )
                candidates.append((gap, oid))
        if not candidates:
            return None, None
        candidates.sort()
        gap, oid = candidates[0]
        return oid, gap

    def _idm_accel(self, sid: str) -> float:
        veh = self._vehicles[sid]
        lid, gap = self._leader_for(sid)
        if lid is None:
            st = IDMState(speed=veh.speed, gap=None, leader_speed=None)
        else:
            st = IDMState(
                speed=veh.speed,
                gap=gap,
                leader_speed=self._vehicles[lid].speed,
            )
        return idm_acceleration(self.config.candidate.idm, st)

    def _detect_collisions(self) -> list[tuple[str, str]]:
        ids = list(self._vehicles)
        pairs: list[tuple[str, str]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                va, vb = self._vehicles[a], self._vehicles[b]
                # Completed vehicles have left the interaction region.
                if va.completed or vb.completed:
                    continue
                # both must be near shared lane (y close)
                if abs(va.world_y - vb.world_y) > 1.5:
                    continue
                if va.world_x <= vb.world_x:
                    rear, front = va, vb
                    rid, fid = a, b
                else:
                    rear, front = vb, va
                    rid, fid = b, a
                # require both on shared path for learner-background / learner-learner
                if not (
                    is_on_shared_mainline(rear.route_position, self._route_for(rear.role))
                    and is_on_shared_mainline(front.route_position, self._route_for(front.role))
                ):
                    continue
                gap = bumper_gap_along_x(
                    rear.world_x, front.world_x, vehicle_length=self.config.vehicle.length
                )
                if gap <= self.config.collision_bumper_gap:
                    pairs.append(tuple(sorted((rid, fid))))  # type: ignore[arg-type]
        # unique
        return sorted(set(pairs))

    def _min_bumper_gap_learners(self) -> float | None:
        a, b = self._vehicles["A"], self._vehicles["B"]
        if a.completed or b.completed:
            return None
        if abs(a.world_y - b.world_y) > 1.5:
            return None
        if not (
            is_on_shared_mainline(a.route_position, self._route_for(a.role))
            and is_on_shared_mainline(b.route_position, self._route_for(b.role))
        ):
            return None
        if a.world_x <= b.world_x:
            return bumper_gap_along_x(a.world_x, b.world_x, vehicle_length=self.config.vehicle.length)
        return bumper_gap_along_x(b.world_x, a.world_x, vehicle_length=self.config.vehicle.length)

    def _ttc_learners(self) -> float | None:
        a, b = self._vehicles["A"], self._vehicles["B"]
        if a.completed or b.completed:
            return None
        gap = self._min_bumper_gap_learners()
        if gap is None:
            return None
        if a.world_x <= b.world_x:
            return time_to_collision(gap=gap, v_rear=a.speed, v_front=b.speed)
        return time_to_collision(gap=gap, v_rear=b.speed, v_front=a.speed)

    def step(self, action: dict[str, int]):
        timing = self.config.timing
        dyn = self.config.dynamics
        n_sub = timing.physics_substeps_per_action
        dt = timing.physics_dt

        snap_t = {k: VehicleState(**vars(v)) for k, v in self._vehicles.items()}
        sub_records: list[dict[str, Any]] = []
        collision_pairs: list[tuple[str, str]] = []
        max_abs_disc = 0.0

        for sub in range(n_sub):
            # Desired accelerations
            des_acc: dict[str, float] = {}
            for aid in LEARNING_CONTROLLERS:
                veh = self._vehicles[aid]
                if veh.completed:
                    des_acc[aid] = 0.0
                else:
                    des_acc[aid] = desired_acceleration(
                        int(action[aid]), accel=dyn.accel, maintain=dyn.maintain, decel=dyn.decel
                    )
            for bid in ("B_front", "B_rear"):
                des_acc[bid] = self._idm_accel(bid)

            # Integrate
            for sid, veh in self._vehicles.items():
                if veh.completed and sid in LEARNING_CONTROLLERS:
                    veh.acceleration = 0.0
                    continue
                s0 = veh.route_position
                s1, v1, a_real = integrate_longitudinal(
                    route_position=veh.route_position,
                    speed=veh.speed,
                    acceleration=des_acc[sid],
                    dt=dt,
                    v_min=dyn.v_min,
                    v_max=dyn.v_max,
                )
                max_abs_disc = max(max_abs_disc, abs(s1 - s0 - 0.5 * (veh.speed + v1) * dt))
                veh.route_position = s1
                veh.speed = v1
                veh.acceleration = a_real
                self._sync(veh)

            pairs = self._detect_collisions()
            if pairs and not collision_pairs:
                collision_pairs = pairs
            sub_records.append(
                {
                    "physics_substep": sub,
                    "vehicles": {
                        k: {
                            "route_position": v.route_position,
                            "speed": v.speed,
                            "acceleration": v.acceleration,
                            "world_x": v.world_x,
                            "world_y": v.world_y,
                        }
                        for k, v in self._vehicles.items()
                    },
                    "collision_pairs": [list(p) for p in pairs],
                }
            )
            if collision_pairs:
                break

        self._policy_step += 1
        collided = {sid: False for sid in STAKEHOLDER_SET}
        for a, b in collision_pairs:
            collided[a] = True
            collided[b] = True
        stakeholder_collision = 1.0 if collision_pairs else 0.0

        # Exits (blocked by collision this transition)
        exit_event = {"A": 0.0, "B": 0.0}
        for aid in LEARNING_CONTROLLERS:
            veh_t = snap_t[aid]
            veh = self._vehicles[aid]
            route = self._route_for(veh.role)
            crossed = veh_t.route_position < route.route_exit <= veh.route_position
            if crossed and stakeholder_collision < 1.0 and not veh_t.completed:
                exit_event[aid] = 1.0
                veh.completed = True
                self._exit_count[aid] += 1
                if self._exit_time[aid] is None:
                    self._exit_time[aid] = self._policy_step

        # Core reward components (no hard-braking term)
        rewards: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for aid in LEARNING_CONTROLLERS:
            route = self._route_for(self._vehicles[aid].role)
            rho_t = normalised_route_progress(snap_t[aid].route_position, route)
            rho_t1 = normalised_route_progress(self._vehicles[aid].route_position, route)
            progress = 0.4 * (rho_t1 - rho_t)
            exit_c = 0.6 * exit_event[aid]
            coll_c = -1.0 * stakeholder_collision
            core = progress + exit_c + coll_c
            rewards[aid] = float(core)
            components[aid] = {
                "progress_component": float(progress),
                "exit_component": float(exit_c),
                "collision_component": float(coll_c),
                "core_reward": float(core),
                "rho_t": float(rho_t),
                "rho_t1": float(rho_t1),
                "delta_rho": float(rho_t1 - rho_t),
            }

        both_done = all(self._vehicles[aid].completed for aid in LEARNING_CONTROLLERS)
        terminated = bool(stakeholder_collision >= 1.0 or both_done)
        truncated = bool(self._policy_step >= self.config.max_policy_steps and not terminated)
        if terminated and truncated:
            # invalid combo prevented
            truncated = False
        term_reason = (
            "collision"
            if stakeholder_collision >= 1.0
            else ("success" if both_done else ("truncation" if truncated else "ongoing"))
        )

        disc = float(0.995 ** (self._policy_step - 1))
        info = {
            "policy_step": self._policy_step,
            "physics_substeps": len(sub_records),
            "substep_records": sub_records,
            "term_reason": term_reason,
            "events": {
                "exit_event": exit_event,
                "stakeholder_collision_event": stakeholder_collision,
                "stakeholder_collided": collided,
                "collision_pairs": [list(p) for p in collision_pairs],
                "warnings": [],
            },
            "completion": {aid: self._vehicles[aid].completed for aid in LEARNING_CONTROLLERS},
            "exit_count": dict(self._exit_count),
            "exit_time": dict(self._exit_time),
            "min_bumper_gap": self._min_bumper_gap_learners(),
            "ttc": self._ttc_learners(),
            "components": components,
            "discount_factor": disc,
            "vehicles_t": {
                k: {
                    "role": snap_t[k].role,
                    "route_position": snap_t[k].route_position,
                    "speed": snap_t[k].speed,
                    "world_x": snap_t[k].world_x,
                    "world_y": snap_t[k].world_y,
                }
                for k in self._vehicles
            },
            "vehicles_t1": {
                k: {
                    "role": v.role,
                    "route_position": v.route_position,
                    "speed": v.speed,
                    "acceleration": v.acceleration,
                    "world_x": v.world_x,
                    "world_y": v.world_y,
                    "completed": v.completed,
                }
                for k, v in self._vehicles.items()
            },
            "fixture_only": False,
            "route_discontinuity": max_abs_disc > self.config.discontinuity_threshold,
        }
        return self._obs(), rewards, terminated, truncated, info
