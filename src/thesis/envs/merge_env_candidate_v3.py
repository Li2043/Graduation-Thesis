"""Stage 4A / 4A-0R candidate merge environment with physics substeps.

Policy interval 0.20 s = 4 × physics_dt 0.05 s.
Hardened in Stage 4A-0R: Markov observations, arc-length geometry, SAT
collision, exact stopping, substep exit removal, bounded IDM, background exit.
"""

from __future__ import annotations

import math
from copy import deepcopy
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
    TimingConfig,
    VehicleGeometry,
)
from thesis.envs.final_observation import OBSERVATION_DIM, build_learner_observation
from thesis.envs.final_route_geometry import FinalRouteGeometry, build_final_route_geometry
from thesis.envs.idm_background import IDMState, idm_command
from thesis.envs.vehicle_dynamics import (
    bumper_gap_along_x,
    desired_acceleration,
    integrate_longitudinal,
    oriented_rectangles_collide,
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
    commanded_acceleration: float = 0.0
    realised_acceleration: float = 0.0
    active_on_road: bool = True
    completed: bool = False
    physical_segment: str = "mainline_approach"
    heading: float = 0.0
    world_x: float = 0.0
    world_y: float = 0.0
    target_speed: float = 20.0


@dataclass
class MergeEnvCandidateV3Config:
    candidate: EnvironmentCandidate
    block: InitialConditionBlock
    timing: TimingConfig = field(default_factory=TimingConfig)
    vehicle: VehicleGeometry = field(default_factory=VehicleGeometry)
    dynamics: LearningDynamics = field(default_factory=LearningDynamics)
    max_policy_steps: int = 400
    discontinuity_threshold: float = 0.5

    def geometry_model(self) -> FinalRouteGeometry:
        return build_final_route_geometry(self.candidate.geometry)


class MergeEnvCandidateV3(gym.Env):
    """Gymnasium env: one step() = one 0.20 s policy transition (4 substeps)."""

    metadata = {"render_modes": []}

    def __init__(self, config: MergeEnvCandidateV3Config):
        super().__init__()
        self.config = config
        self.config.timing.validate()
        self._geom = config.geometry_model()
        self.action_space = spaces.Dict(
            {"A": spaces.Discrete(3), "B": spaces.Discrete(3)}
        )
        self.observation_space = spaces.Dict(
            {
                aid: spaces.Box(
                    low=-np.inf, high=np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32
                )
                for aid in LEARNING_CONTROLLERS
            }
        )
        self._vehicles: dict[str, VehicleState] = {}
        self._policy_step = 0
        self._rng = np.random.default_rng(0)
        self._exit_count = {sid: 0 for sid in STAKEHOLDER_SET}
        self._exit_time: dict[str, int | None] = {sid: None for sid in STAKEHOLDER_SET}
        self._exit_substep: dict[str, int | None] = {sid: None for sid in STAKEHOLDER_SET}

    def _sync(self, veh: VehicleState) -> None:
        pose = self._geom.pose(veh.role, veh.route_position)  # type: ignore[arg-type]
        veh.world_x = pose.x
        veh.world_y = pose.y
        veh.heading = pose.heading
        if veh.completed or not veh.active_on_road:
            veh.physical_segment = "exited"
        else:
            veh.physical_segment = pose.segment

    def _agent_features(self, veh: VehicleState) -> dict[str, Any]:
        exit_s = self._geom.exit_route(veh.role)  # type: ignore[arg-type]
        merge_s = self._geom.merge_route_marker(veh.role)  # type: ignore[arg-type]
        rho = min(max(veh.route_position / max(exit_s, 1e-9), 0.0), 1.0)
        return {
            "role": veh.role,
            "rho": rho,
            "speed": veh.speed,
            "target_speed": veh.target_speed,
            "dist_to_merge": merge_s - veh.route_position,
            "dist_to_exit": exit_s - veh.route_position,
            "active_on_road": veh.active_on_road,
            "completed": veh.completed,
            "world_x": veh.world_x,
            "world_y": veh.world_y,
        }

    def _nearest_longitudinal(
        self, sid: str, *, direction: str
    ) -> tuple[float | None, float | None]:
        """Nearest active vehicle front/rear by world_x when headings nearly aligned."""
        veh = self._vehicles[sid]
        if not veh.active_on_road:
            return None, None
        best_gap = None
        best_rel = None
        for oid, other in self._vehicles.items():
            if oid == sid or not other.active_on_road:
                continue
            # Only meaningful when roughly same lane corridor
            if abs(other.world_y - veh.world_y) > 2.5:
                continue
            if direction == "front" and other.world_x > veh.world_x + 1e-9:
                gap = bumper_gap_along_x(
                    veh.world_x, other.world_x, vehicle_length=self.config.vehicle.length
                )
                rel = other.speed - veh.speed
            elif direction == "rear" and other.world_x < veh.world_x - 1e-9:
                gap = bumper_gap_along_x(
                    other.world_x, veh.world_x, vehicle_length=self.config.vehicle.length
                )
                rel = veh.speed - other.speed
            else:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_rel = rel
        return best_gap, best_rel

    def _obs(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for aid in LEARNING_CONTROLLERS:
            peer_id = "B" if aid == "A" else "A"
            front_gap, front_rel = self._nearest_longitudinal(aid, direction="front")
            rear_gap, rear_rel = self._nearest_longitudinal(aid, direction="rear")
            out[aid] = build_learner_observation(
                own=self._agent_features(self._vehicles[aid]),
                peer=self._agent_features(self._vehicles[peer_id]),
                b_front=self._agent_features(self._vehicles["B_front"]),
                b_rear=self._agent_features(self._vehicles["B_rear"]),
                front_gap=front_gap,
                front_rel_speed=front_rel,
                rear_gap=rear_gap,
                rear_rel_speed=rear_rel,
            )
        return out

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed = self.config.block.seed
        super().reset(seed=seed)
        self._rng = np.random.default_rng(int(seed))
        self._policy_step = 0
        self._exit_count = {sid: 0 for sid in STAKEHOLDER_SET}
        self._exit_time = {sid: None for sid in STAKEHOLDER_SET}
        self._exit_substep = {sid: None for sid in STAKEHOLDER_SET}
        b = self.config.block
        targets = b.target_speeds.as_map()
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
        return self._obs(), {"stakeholder_set": list(STAKEHOLDER_SET), "policy_step": 0}

    def _leader_for(self, sid: str) -> tuple[str | None, float | None]:
        veh = self._vehicles[sid]
        if not veh.active_on_road:
            return None, None
        candidates = []
        for oid, other in self._vehicles.items():
            if oid == sid or not other.active_on_road:
                continue
            if abs(other.world_y - veh.world_y) > 2.5:
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
        if not veh.active_on_road:
            return 0.0
        lid, gap = self._leader_for(sid)
        if lid is None:
            st = IDMState(speed=veh.speed, gap=None, leader_speed=None)
        else:
            st = IDMState(
                speed=veh.speed,
                gap=gap,
                leader_speed=self._vehicles[lid].speed,
            )
        return idm_command(self.config.candidate.idm, st).commanded_acceleration

    def _collision_eligible(self, veh: VehicleState) -> bool:
        # Use active/completed flags only. Pose may already report segment="exited"
        # after integrating past the exit marker; collision still has precedence
        # until the vehicle is formally removed.
        return bool(veh.active_on_road and not veh.completed)

    def _detect_collisions(self) -> list[tuple[str, str]]:
        ids = [k for k, v in self._vehicles.items() if self._collision_eligible(v)]
        pairs: list[tuple[str, str]] = []
        L = self.config.vehicle.length
        W = self.config.vehicle.width
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                va, vb = self._vehicles[a], self._vehicles[b]
                if oriented_rectangles_collide(
                    x1=va.world_x,
                    y1=va.world_y,
                    heading1=va.heading,
                    x2=vb.world_x,
                    y2=vb.world_y,
                    heading2=vb.heading,
                    length=L,
                    width=W,
                ):
                    pairs.append(tuple(sorted((a, b))))  # type: ignore[arg-type]
        return sorted(set(pairs))

    def _mark_exit(self, sid: str, *, policy_step: int, substep: int) -> bool:
        """Mark safe exit once. Returns True if a new exit event was emitted."""
        veh = self._vehicles[sid]
        if veh.completed:
            return False
        veh.completed = True
        veh.active_on_road = False
        veh.physical_segment = "exited"
        veh.commanded_acceleration = 0.0
        veh.realised_acceleration = 0.0
        veh.speed = 0.0
        self._exit_count[sid] += 1
        if self._exit_time[sid] is None:
            self._exit_time[sid] = policy_step
        if self._exit_substep[sid] is None:
            self._exit_substep[sid] = substep
        return True

    def _check_exit_crossing(self, sid: str, s_before: float, s_after: float) -> bool:
        veh = self._vehicles[sid]
        exit_s = self._geom.exit_route(veh.role)  # type: ignore[arg-type]
        return s_before < exit_s <= s_after

    def _min_bumper_gap_learners(self) -> float | None:
        a, b = self._vehicles["A"], self._vehicles["B"]
        if not a.active_on_road or not b.active_on_road:
            return None
        if abs(a.world_y - b.world_y) > 2.5:
            return None
        if a.world_x <= b.world_x:
            return bumper_gap_along_x(a.world_x, b.world_x, vehicle_length=self.config.vehicle.length)
        return bumper_gap_along_x(b.world_x, a.world_x, vehicle_length=self.config.vehicle.length)

    def _ttc_learners(self) -> float | None:
        a, b = self._vehicles["A"], self._vehicles["B"]
        gap = self._min_bumper_gap_learners()
        if gap is None:
            return None
        if a.world_x <= b.world_x:
            return time_to_collision(gap=gap, v_rear=a.speed, v_front=b.speed)
        return time_to_collision(gap=gap, v_rear=b.speed, v_front=a.speed)

    def _snapshot(self) -> dict[str, VehicleState]:
        return {k: deepcopy(v) for k, v in self._vehicles.items()}

    def _veh_info(self, veh: VehicleState) -> dict[str, Any]:
        return {
            "role": veh.role,
            "route_position": veh.route_position,
            "speed": veh.speed,
            "commanded_acceleration": veh.commanded_acceleration,
            "realised_acceleration": veh.realised_acceleration,
            # Compatibility alias for Stage 4A certification traces
            "acceleration": veh.realised_acceleration,
            "world_x": veh.world_x,
            "world_y": veh.world_y,
            "heading": veh.heading,
            "completed": veh.completed,
            "active_on_road": veh.active_on_road,
            "physical_segment": veh.physical_segment,
        }

    def step(self, action: dict[str, int]):
        timing = self.config.timing
        dyn = self.config.dynamics
        n_sub = timing.physics_substeps_per_action
        dt = timing.physics_dt

        snap_t = self._snapshot()
        sub_records: list[dict[str, Any]] = []
        collision_pairs: list[tuple[str, str]] = []
        exit_event = {sid: 0.0 for sid in STAKEHOLDER_SET}
        max_abs_disc = 0.0
        nan_count = 0

        for sub in range(n_sub):
            des_acc: dict[str, float] = {}
            for aid in LEARNING_CONTROLLERS:
                veh = self._vehicles[aid]
                if not veh.active_on_road:
                    des_acc[aid] = 0.0
                else:
                    des_acc[aid] = desired_acceleration(
                        int(action[aid]), accel=dyn.accel, maintain=dyn.maintain, decel=dyn.decel
                    )
            for bid in ("B_front", "B_rear"):
                des_acc[bid] = self._idm_accel(bid) if self._vehicles[bid].active_on_road else 0.0

            s_before = {k: v.route_position for k, v in self._vehicles.items()}

            for sid, veh in self._vehicles.items():
                if not veh.active_on_road:
                    veh.commanded_acceleration = 0.0
                    veh.realised_acceleration = 0.0
                    continue
                veh.commanded_acceleration = float(des_acc[sid])
                s0 = veh.route_position
                s1, v1, a_real = integrate_longitudinal(
                    route_position=veh.route_position,
                    speed=veh.speed,
                    acceleration=veh.commanded_acceleration,
                    dt=dt,
                    v_min=dyn.v_min,
                    v_max=dyn.v_max,
                )
                if not all(math.isfinite(x) for x in (s1, v1, a_real)):
                    nan_count += 1
                # Exact-stop disc check uses commanded path when no early stop
                max_abs_disc = max(max_abs_disc, abs(s1 - s0))
                veh.route_position = s1
                veh.speed = v1
                veh.realised_acceleration = a_real
                self._sync(veh)

            # Collision has precedence in the same substep
            pairs = self._detect_collisions()
            if pairs and not collision_pairs:
                collision_pairs = list(pairs)

            if not collision_pairs:
                for sid in list(STAKEHOLDER_SET):
                    if not self._vehicles[sid].active_on_road:
                        continue
                    if self._check_exit_crossing(sid, s_before[sid], self._vehicles[sid].route_position):
                        if self._mark_exit(sid, policy_step=self._policy_step + 1, substep=sub):
                            exit_event[sid] = 1.0

            sub_records.append(
                {
                    "physics_substep": sub,
                    "vehicles": {k: self._veh_info(v) for k, v in self._vehicles.items()},
                    "collision_pairs": [list(p) for p in pairs],
                    "exit_event": dict(exit_event),
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

        # Core reward for learners (exit only if emitted this transition)
        rewards: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for aid in LEARNING_CONTROLLERS:
            exit_s = self._geom.exit_route(self._vehicles[aid].role)  # type: ignore[arg-type]
            rho_t = min(max(snap_t[aid].route_position / max(exit_s, 1e-9), 0.0), 1.0)
            rho_t1 = min(max(self._vehicles[aid].route_position / max(exit_s, 1e-9), 0.0), 1.0)
            # Completed vehicles keep rho at 1 for progress stability
            if self._vehicles[aid].completed and exit_event[aid] < 1.0:
                rho_t1 = 1.0
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
                "exit_event": {k: exit_event[k] for k in LEARNING_CONTROLLERS},
                "exit_event_all": dict(exit_event),
                "stakeholder_collision_event": stakeholder_collision,
                "stakeholder_collided": collided,
                "collision_pairs": [list(p) for p in collision_pairs],
                "warnings": [],
            },
            "completion": {sid: self._vehicles[sid].completed for sid in STAKEHOLDER_SET},
            "exit_count": dict(self._exit_count),
            "exit_time": dict(self._exit_time),
            "exit_substep": dict(self._exit_substep),
            "min_bumper_gap": self._min_bumper_gap_learners(),
            "ttc": self._ttc_learners(),
            "components": components,
            "discount_factor": disc,
            "vehicles_t": {k: self._veh_info(v) for k, v in snap_t.items()},
            "vehicles_t1": {k: self._veh_info(v) for k, v in self._vehicles.items()},
            "fixture_only": False,
            "route_discontinuity": max_abs_disc > 50.0,  # absurd jump only
            "nan_count": nan_count,
            "observation_dim": OBSERVATION_DIM,
        }
        return self._obs(), rewards, terminated, truncated, info
