"""Minimal merge environment for Stage 2B-1 reward / PBRS integration tests.

This is a **thesis-owned kinematic simulator** registered as a Gymnasium-style
API. ``highway-env`` is an installed dependency for reproducibility and future
wrapping, but Stage 2B-1 dynamics are explicit and deterministic here so that
collision / exit fixtures are controllable.

Acceleration convention (SI signed):
    positive = accelerate forward along the route
    negative = braking

Geometry and braking / lambda values used by smoke tests are **TEST-ONLY /
integration-test configuration**, not final dissertation calibrations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from thesis.envs.event_registry import (
    MergeVehicleRegistry,
    TransitionEvents,
    record_collision_pairs,
)
from thesis.envs.route_coordinates import (
    RouteDefinition,
    advance_route_position,
    default_mainline_route,
    default_ramp_route,
    is_on_shared_mainline,
    normalised_route_progress,
    world_xy_from_route,
)
from thesis.rewards.base_reward_v2 import (
    LEARNING_CONTROLLERS,
    STAKEHOLDER_SET,
    AgentTransitionState,
    BaseRewardConfig,
    BaseRewardInputs,
    compute_base_reward_for_agents,
    to_si_longitudinal_acceleration,
)
from thesis.rewards.pbrs_v2 import (
    PBRSConfig,
    PotentialState,
    StakeholderState,
    apply_pbrs_to_base_rewards,
    compute_potential_breakdown,
)


class HighLevelAction(IntEnum):
    MAINTAIN = 0
    ACCELERATE = 1
    DECELERATE = 2


ACTION_NAMES = {
    HighLevelAction.MAINTAIN: "MAINTAIN",
    HighLevelAction.ACCELERATE: "ACCELERATE",
    HighLevelAction.DECELERATE: "DECELERATE",
}


@dataclass
class IDMParams:
    """Deterministic IDM parameters for background vehicles (recorded)."""

    v0: float = 20.0
    T: float = 1.2
    a_max: float = 1.5
    b: float = 2.0
    delta: float = 4.0
    s0: float = 4.0


@dataclass
class MergeEnvConfig:
    """Stage 2B-1 integration-test environment configuration."""

    dt: float = 0.2
    max_steps: int = 200
    seed: int = 0
    # Role assignment: controller identity != traffic role
    role_A: str = "mainline"
    role_B: str = "ramp"
    # Kinematics
    accel_rate: float = 2.0
    decel_rate: float = 3.0
    maintain_accel: float = 0.0
    v_min: float = 0.0
    v_max: float = 30.0
    collision_distance: float = 4.0
    # Spawn (route positions / speeds) — integration fixtures
    spawn_route_A: float = 10.0
    spawn_route_B: float = 5.0
    spawn_route_B_front: float = 40.0
    spawn_route_B_rear: float = 0.0
    spawn_speed_A: float = 18.0
    spawn_speed_B: float = 16.0
    spawn_speed_B_front: float = 18.0
    spawn_speed_B_rear: float = 17.0
    target_speed: float = 20.0
    idm: IDMParams = field(default_factory=IDMParams)
    # TEST-ONLY reward / PBRS parameters
    base_reward: BaseRewardConfig = field(
        default_factory=lambda: BaseRewardConfig(
            eta_hard_brake=0.1,  # TEST-ONLY
            a_comfort=2.0,  # TEST-ONLY
            a_hard=6.0,  # TEST-ONLY
        )
    )
    pbrs: PBRSConfig = field(
        default_factory=lambda: PBRSConfig(
            learner_gamma=0.995,
            shaping_gamma=0.995,
            lambda_mean=0.5,  # TEST-ONLY
            lambda_min=0.5,  # TEST-ONLY
        )
    )
    discontinuity_report_threshold: float = 0.25
    # Optional fixture overrides for deterministic collision tests
    fixture_mode: str | None = None
    fixture_payload: dict[str, Any] = field(default_factory=dict)

    def routes(self) -> dict[str, RouteDefinition]:
        return {
            "mainline": default_mainline_route(),
            "ramp": default_ramp_route(),
        }

    def validate(self) -> None:
        if self.role_A == self.role_B:
            raise ValueError("A and B must have distinct traffic roles in Stage 2B-1")
        if {self.role_A, self.role_B} != {"mainline", "ramp"}:
            raise ValueError("roles must be exactly {mainline, ramp}")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        self.base_reward.validate()
        self.pbrs.validate()


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


def _idm_accel(
    v: float,
    v_lead: float | None,
    gap: float | None,
    params: IDMParams,
) -> float:
    """Classic IDM longitudinal acceleration (SI signed)."""
    free = 1.0 - (v / max(params.v0, 1e-6)) ** params.delta
    if v_lead is None or gap is None or gap <= 0:
        return float(params.a_max * free)
    dv = v - v_lead
    s_star = (
        params.s0
        + v * params.T
        + (v * dv) / (2.0 * math.sqrt(max(params.a_max * params.b, 1e-9)))
    )
    s_star = max(s_star, 0.0)
    interaction = (s_star / max(gap, 1e-3)) ** 2
    return float(params.a_max * (free - interaction))


class MergeEnvV2(gym.Env):
    """Minimal two-controller merge environment with fixed stakeholder set V."""

    metadata = {"render_modes": []}

    def __init__(self, config: MergeEnvConfig | None = None):
        super().__init__()
        self.config = config or MergeEnvConfig()
        self.config.validate()
        self.routes = self.config.routes()
        self.action_space = spaces.Dict(
            {
                "A": spaces.Discrete(len(HighLevelAction)),
                "B": spaces.Discrete(len(HighLevelAction)),
            }
        )
        # Compact kinematic obs per learner: [route_pos, speed, rho, completed]
        self.observation_space = spaces.Dict(
            {
                "A": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
                "B": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            }
        )
        self._np_random: np.random.Generator | None = None
        self._vehicles: dict[str, VehicleState] = {}
        self._registry = MergeVehicleRegistry(
            roles={"A": self.config.role_A, "B": self.config.role_B}
        )
        self._step_count = 0
        self._seed_used: int | None = None
        self._episode_active = False
        self.last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ utils
    def _route_for(self, role: str) -> RouteDefinition:
        return self.routes[role]

    def _role_of(self, identity: str) -> str:
        if identity in ("A", "B"):
            return self._registry.roles[identity]
        return "mainline"

    def legal_actions(self, role: str) -> list[str]:
        # No overtaking / lane-change actions in Stage 2B-1.
        return ["MAINTAIN", "ACCELERATE", "DECELERATE"]

    def _validate_action(self, identity: str, action: int) -> HighLevelAction:
        try:
            act = HighLevelAction(int(action))
        except ValueError as e:
            raise ValueError(f"illegal action {action!r} for {identity}") from e
        role = self._role_of(identity)
        if ACTION_NAMES[act] not in self.legal_actions(role):
            raise ValueError(f"action {ACTION_NAMES[act]} illegal for role {role}")
        return act

    def _action_to_accel(self, act: HighLevelAction) -> float:
        if act == HighLevelAction.ACCELERATE:
            return float(self.config.accel_rate)
        if act == HighLevelAction.DECELERATE:
            return float(-self.config.decel_rate)
        return float(self.config.maintain_accel)

    def _sync_world(self, veh: VehicleState) -> None:
        route = self._route_for(veh.role)
        x, y = world_xy_from_route(veh.route_position, route)
        veh.world_x, veh.world_y = x, y

    def _snapshot(self) -> dict[str, VehicleState]:
        return {
            sid: replace(veh) for sid, veh in self._vehicles.items()
        }

    def _obs_from_vehicles(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for aid in LEARNING_CONTROLLERS:
            veh = self._vehicles[aid]
            route = self._route_for(veh.role)
            rho = normalised_route_progress(veh.route_position, route)
            out[aid] = np.asarray(
                [
                    veh.route_position,
                    veh.speed,
                    rho,
                    1.0 if veh.completed else 0.0,
                ],
                dtype=np.float64,
            )
        return out

    # ----------------------------------------------------------------- reset
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if options and "config" in options:
            self.config = options["config"]
            self.config.validate()
            self.routes = self.config.routes()
            self._registry = MergeVehicleRegistry(
                roles={"A": self.config.role_A, "B": self.config.role_B}
            )
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.config.seed)
        self._seed_used = int(seed)
        self._np_random = np.random.default_rng(self._seed_used)
        self._step_count = 0
        self._episode_active = True
        self._registry.completed = {"A": False, "B": False}
        self._registry.validate()

        cfg = self.config
        # Small deterministic spawn jitter from seed (traceable, not silent reuse)
        jitter = float(self._np_random.uniform(-0.5, 0.5))

        def make(identity: str, role: str, s0: float, v0: float) -> VehicleState:
            veh = VehicleState(
                identity=identity,
                role=role,
                route_position=float(s0 + (jitter if identity in ("A", "B") else 0.0)),
                speed=float(v0),
                acceleration=0.0,
                completed=False,
            )
            self._sync_world(veh)
            return veh

        self._vehicles = {
            "A": make("A", cfg.role_A, cfg.spawn_route_A, cfg.spawn_speed_A),
            "B": make("B", cfg.role_B, cfg.spawn_route_B, cfg.spawn_speed_B),
            "B_front": make(
                "B_front", "mainline", cfg.spawn_route_B_front, cfg.spawn_speed_B_front
            ),
            "B_rear": make(
                "B_rear", "mainline", cfg.spawn_route_B_rear, cfg.spawn_speed_B_rear
            ),
        }
        # Fixture spawn overrides
        if cfg.fixture_mode and "spawn" in cfg.fixture_payload:
            for sid, vals in cfg.fixture_payload["spawn"].items():
                if sid in self._vehicles:
                    if "route_position" in vals:
                        self._vehicles[sid].route_position = float(vals["route_position"])
                    if "speed" in vals:
                        self._vehicles[sid].speed = float(vals["speed"])
                    self._sync_world(self._vehicles[sid])

        info = {
            "seed": self._seed_used,
            "registry": list(STAKEHOLDER_SET),
            "roles": dict(self._registry.roles),
            "fixture_mode": cfg.fixture_mode,
            "geometry_note": "Stage 2B-1 integration-test geometry (not final dissertation)",
        }
        self.last_info = info
        return self._obs_from_vehicles(), info

    # ------------------------------------------------------------------ step
    def step(self, action: Mapping[str, int]):
        if not self._episode_active:
            raise RuntimeError("step() called before reset() or after episode end")
        if set(action.keys()) != {"A", "B"}:
            raise ValueError("action must contain exactly keys A and B")

        act_A = self._validate_action("A", action["A"])
        act_B = self._validate_action("B", action["B"])

        # 3. snapshot s_t
        state_t = self._snapshot()

        # 4. dynamics
        self._apply_dynamics(act_A, act_B)

        # Optional collision fixture: force positions on this step
        self._apply_fixture_after_dynamics()

        # 5. snapshot s_{t+1}
        state_t1 = self._snapshot()
        self._step_count += 1

        # 6–9 events / flags
        events, terminated, truncated, term_reason = self._detect_events(state_t, state_t1)

        # 10–12 rewards / potentials / PBRS
        diagnostics = self._compute_rewards_and_pbrs(
            state_t, state_t1, events, terminated, truncated
        )

        # Apply completion flags only when safe exit awarded
        for aid in LEARNING_CONTROLLERS:
            if diagnostics["exit_event"][aid] >= 1.0:
                self._registry.mark_completed(aid)
                self._vehicles[aid].completed = True

        if terminated or truncated:
            self._episode_active = False

        reward = {
            aid: float(diagnostics["baseline_reward"][aid]) for aid in LEARNING_CONTROLLERS
        }
        info = {
            "seed": self._seed_used,
            "step": self._step_count,
            "term_reason": term_reason,
            "events": {
                "exit_event": dict(events.exit_event),
                "stakeholder_collided": dict(events.stakeholder_collided),
                "collision_pairs": list(events.collision_pairs),
                "stakeholder_collision_event": events.stakeholder_collision_event,
                "warnings": list(events.warnings),
            },
            "completion": dict(self._registry.completed),
            "vehicles_t": self._vehicle_dict(state_t),
            "vehicles_t1": self._vehicle_dict(state_t1),
            "diagnostics": diagnostics,
            "rewards": {
                "baseline": dict(diagnostics["baseline_reward"]),
                "mean_pbrs": dict(diagnostics["mean_pbrs_reward"]),
                "min_pbrs": dict(diagnostics["min_pbrs_reward"]),
            },
            "legal_actions": {
                "mainline": self.legal_actions("mainline"),
                "ramp": self.legal_actions("ramp"),
            },
            "action_masks": {
                aid: [True, True, True]  # MAINTAIN / ACCELERATE / DECELERATE
                for aid in LEARNING_CONTROLLERS
            },
            "controller_active": {
                aid: not bool(self._registry.completed.get(aid, False))
                for aid in LEARNING_CONTROLLERS
            },
        }
        self.last_info = info
        return self._obs_from_vehicles(), reward, bool(terminated), bool(truncated), info

    def _vehicle_dict(self, snap: Mapping[str, VehicleState]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for sid, veh in snap.items():
            route = self._route_for(veh.role)
            out[sid] = {
                "identity": veh.identity,
                "role": veh.role,
                "route_position": veh.route_position,
                "world_x": veh.world_x,
                "world_y": veh.world_y,
                "speed": veh.speed,
                "acceleration": veh.acceleration,
                "rho": normalised_route_progress(veh.route_position, route),
                "completed": veh.completed,
            }
        return out

    def _apply_dynamics(self, act_A: HighLevelAction, act_B: HighLevelAction) -> None:
        cfg = self.config
        dt = cfg.dt
        # Commanded accelerations for learners
        cmd = {
            "A": self._action_to_accel(act_A),
            "B": self._action_to_accel(act_B),
        }
        # Background IDM along mainline shared lane ordering by world_x
        bg_accel = self._background_idm_accels()

        for sid, veh in self._vehicles.items():
            if sid in ("A", "B"):
                # Option 1 (Stage 2B-2): completed learning controllers are inactive.
                # Ignore commanded accel; do not keep accelerating an exited vehicle.
                # Joint step API may still receive a placeholder action from the caller.
                if self._registry.completed.get(sid, False) or veh.completed:
                    a = 0.0
                else:
                    a = float(cmd[sid])
            else:
                a = float(bg_accel[sid])
            # Realised SI-signed acceleration
            a = to_si_longitudinal_acceleration(a, convention="si_signed")
            v_new = float(np.clip(veh.speed + a * dt, cfg.v_min, cfg.v_max))
            # Use average speed for position integration
            v_avg = 0.5 * (veh.speed + v_new)
            s_new = advance_route_position(veh.route_position, v_avg, dt)
            veh.acceleration = a
            veh.speed = v_new
            veh.route_position = s_new
            self._sync_world(veh)

    def _background_idm_accels(self) -> dict[str, float]:
        # Order mainline-occupying vehicles by world_x
        mainline_ids = [
            sid
            for sid, veh in self._vehicles.items()
            if is_on_shared_mainline(veh.route_position, self._route_for(veh.role))
        ]
        mainline_ids.sort(key=lambda s: self._vehicles[s].world_x)
        accels = {sid: 0.0 for sid in ("B_front", "B_rear")}
        for sid in ("B_front", "B_rear"):
            veh = self._vehicles[sid]
            # Find leader (next vehicle ahead on shared mainline)
            leader = None
            if sid in mainline_ids:
                idx = mainline_ids.index(sid)
                if idx + 1 < len(mainline_ids):
                    leader = self._vehicles[mainline_ids[idx + 1]]
            if leader is None:
                accels[sid] = _idm_accel(veh.speed, None, None, self.config.idm)
            else:
                gap = max(leader.world_x - veh.world_x - self.config.collision_distance, 0.1)
                accels[sid] = _idm_accel(veh.speed, leader.speed, gap, self.config.idm)
        return accels

    def _apply_fixture_after_dynamics(self) -> None:
        mode = self.config.fixture_mode
        if not mode:
            return
        payload = self.config.fixture_payload
        if mode.startswith("controlled_collision") and self._step_count + 1 >= int(
            payload.get("collide_at_step", 1)
        ):
            # Force two vehicles to the same shared mainline location
            target = payload.get("target_ids", ["A", "B"])
            x = float(payload.get("collision_world_x", 60.0))
            for sid in target:
                veh = self._vehicles[sid]
                # Place on shared mainline route coordinate corresponding to x
                if veh.role == "mainline":
                    veh.route_position = x
                else:
                    # ramp: shared when route_pos >= L_ramp; set join + (x - join)
                    route = self._route_for(veh.role)
                    veh.route_position = route.ramp_approach_length + max(
                        0.0, x - route.join_world_x
                    )
                veh.speed = float(payload.get("collision_speed", 10.0))
                self._sync_world(veh)
        if mode == "force_hard_brake" and "A" in self._vehicles:
            self._vehicles["A"].acceleration = float(payload.get("accel_A", -6.0))

    def _detect_events(
        self,
        state_t: Mapping[str, VehicleState],
        state_t1: Mapping[str, VehicleState],
    ) -> tuple[TransitionEvents, bool, bool, str]:
        events = TransitionEvents()
        # Collisions among stakeholders on shared mainline
        collided: set[str] = set()
        ids = list(STAKEHOLDER_SET)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                va, vb = state_t1[a], state_t1[b]
                if not (
                    is_on_shared_mainline(va.route_position, self._route_for(va.role))
                    and is_on_shared_mainline(vb.route_position, self._route_for(vb.role))
                ):
                    continue
                dist = abs(va.world_x - vb.world_x)
                if dist <= self.config.collision_distance and abs(va.world_y - vb.world_y) < 1.5:
                    collided.add(a)
                    collided.add(b)
        flags, pairs = record_collision_pairs(collided)
        events.stakeholder_collided = flags
        events.collision_pairs = pairs
        events.finalise()

        # Safe exits for A/B (blocked if collision this transition)
        for aid in LEARNING_CONTROLLERS:
            veh_t = state_t[aid]
            veh_t1 = state_t1[aid]
            route = self._route_for(veh_t.role)
            already = bool(self._registry.completed[aid] or veh_t.completed)
            crossed = veh_t.route_position < route.route_exit <= veh_t1.route_position
            if (
                crossed
                and not already
                and events.stakeholder_collision_event < 1.0
            ):
                events.exit_event[aid] = 1.0
            else:
                events.exit_event[aid] = 0.0

        # Route discontinuity warnings
        for sid, veh_t in state_t.items():
            veh_t1 = state_t1[sid]
            route = self._route_for(veh_t.role)
            rho_t = normalised_route_progress(veh_t.route_position, route)
            rho_t1 = normalised_route_progress(veh_t1.route_position, route)
            delta = rho_t1 - rho_t
            if abs(delta) > self.config.discontinuity_report_threshold:
                events.warnings.append(
                    f"route_discontinuity:{sid}:|Δρ|={abs(delta):.6f}"
                )

        collision = events.stakeholder_collision_event >= 1.0
        both_done = all(
            self._registry.completed[aid] or events.exit_event[aid] >= 1.0
            for aid in LEARNING_CONTROLLERS
        )
        # Success: both completed after this transition (exit this step counts)
        success = both_done and not collision
        # Update conceptual completion for success check already includes exit_event

        truncated = False
        terminated = False
        reason = "ongoing"
        if collision:
            terminated = True
            truncated = False
            reason = "collision"
        elif success:
            terminated = True
            truncated = False
            reason = "success"
        elif self._step_count >= self.config.max_steps:
            terminated = False
            truncated = True
            reason = "truncation"
        if terminated and truncated:
            raise ValueError("invalid simultaneous terminated and truncated")
        return events, terminated, truncated, reason

    def _stakeholder_states_for_potential(
        self,
        snap: Mapping[str, VehicleState],
        *,
        completed_override: Mapping[str, bool] | None = None,
    ) -> dict[str, StakeholderState]:
        completed = completed_override or self._registry.completed
        out: dict[str, StakeholderState] = {}
        for sid in STAKEHOLDER_SET:
            veh = snap[sid]
            is_completed = False
            if sid in ("A", "B") and completed.get(sid, False):
                is_completed = True
            out[sid] = StakeholderState(
                speed=veh.speed,
                target_speed=self.config.target_speed,
                completed=is_completed,
            )
        return out

    def _compute_rewards_and_pbrs(
        self,
        state_t: Mapping[str, VehicleState],
        state_t1: Mapping[str, VehicleState],
        events: TransitionEvents,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        # Completion for potentials at s_t uses registry before this transition's exits;
        # at s_{t+1}, include exits awarded this step unless collision.
        completed_t = dict(self._registry.completed)
        completed_t1 = dict(completed_t)
        if events.stakeholder_collision_event < 1.0:
            for aid in LEARNING_CONTROLLERS:
                if events.exit_event[aid] >= 1.0:
                    completed_t1[aid] = True

        agents = {}
        for aid in LEARNING_CONTROLLERS:
            veh_t = state_t[aid]
            veh_t1 = state_t1[aid]
            route = self._route_for(veh_t.role)
            agents[aid] = AgentTransitionState(
                route_position_t=veh_t.route_position,
                route_position_t1=veh_t1.route_position,
                route_start=route.route_start,
                route_exit=route.route_exit,
                acceleration=to_si_longitudinal_acceleration(
                    veh_t1.acceleration, convention="si_signed"
                ),
                already_exited=bool(completed_t[aid]),
            )
        base_inputs = BaseRewardInputs(
            agents=agents,
            stakeholder_collided=dict(events.stakeholder_collided),
            terminated=terminated,
            truncated=truncated,
        )
        base_out = compute_base_reward_for_agents(base_inputs, self.config.base_reward)

        pot_t = PotentialState(
            stakeholders=self._stakeholder_states_for_potential(
                state_t, completed_override=completed_t
            ),
            terminated=False,
            truncated=False,
        )
        pot_t1 = PotentialState(
            stakeholders=self._stakeholder_states_for_potential(
                state_t1, completed_override=completed_t1
            ),
            terminated=terminated,
            truncated=truncated,
            terminal_label=("collision" if terminated and events.stakeholder_collision_event else ("success" if terminated else None)),
        )

        mean_bd_t = compute_potential_breakdown(pot_t, "mean")
        mean_bd_t1 = compute_potential_breakdown(pot_t1, "mean")
        min_bd_t = compute_potential_breakdown(pot_t, "min")
        min_bd_t1 = compute_potential_breakdown(pot_t1, "min")

        base_totals = {aid: base_out[aid].total_reward for aid in LEARNING_CONTROLLERS}
        mean_shaped = apply_pbrs_to_base_rewards(
            base_totals, pot_t, pot_t1, "mean", self.config.pbrs
        )
        min_shaped = apply_pbrs_to_base_rewards(
            base_totals, pot_t, pot_t1, "min", self.config.pbrs
        )

        per_agent: dict[str, Any] = {}
        for aid in LEARNING_CONTROLLERS:
            b = base_out[aid]
            m = mean_shaped[aid]
            n = min_shaped[aid]
            per_agent[aid] = {
                "progress_component": b.progress_component,
                "exit_component": b.exit_component,
                "collision_component": b.collision_component,
                "hard_braking_component": b.hard_braking_component,
                "base_total": b.total_reward,
                "delta_rho": b.delta_route_progress,
                "rho_t": b.rho_t,
                "rho_t1": b.rho_t1,
                "hard_braking_cost": b.hard_braking_cost,
                "mean_F_t": m.shaping_signal,
                "min_F_t": n.shaping_signal,
                "scaled_mean_shaping": m.scaled_shaping_component,
                "scaled_min_shaping": n.scaled_shaping_component,
                "baseline_total": b.total_reward,
                "mean_pbrs_total": m.shaped_reward,
                "min_pbrs_total": n.shaped_reward,
            }

        return {
            "exit_event": dict(events.exit_event),
            "baseline_reward": base_totals,
            "mean_pbrs_reward": {
                aid: mean_shaped[aid].shaped_reward for aid in LEARNING_CONTROLLERS
            },
            "min_pbrs_reward": {
                aid: min_shaped[aid].shaped_reward for aid in LEARNING_CONTROLLERS
            },
            "stakeholder_experiences_t": mean_bd_t.stakeholder_experiences,
            "stakeholder_experiences_t1": mean_bd_t1.stakeholder_experiences,
            "raw_mean_potential_t": mean_bd_t.raw_potential,
            "actual_mean_potential_t": mean_bd_t.actual_potential,
            "raw_mean_potential_t1": mean_bd_t1.raw_potential,
            "actual_mean_potential_t1": mean_bd_t1.actual_potential,
            "raw_min_potential_t": min_bd_t.raw_potential,
            "actual_min_potential_t": min_bd_t.actual_potential,
            "raw_min_potential_t1": min_bd_t1.raw_potential,
            "actual_min_potential_t1": min_bd_t1.actual_potential,
            "mean_F_t": mean_shaped["A"].shaping_signal,
            "min_F_t": min_shaped["A"].shaping_signal,
            "per_agent": per_agent,
            "warnings": list(events.warnings),
        }
