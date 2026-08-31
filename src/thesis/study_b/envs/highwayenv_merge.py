"""M3 -- ``ThesisHighwayMergeEnv``: the HighwayEnv-backend physical
environment for Study B (Claude_Code_Autonomous_Experiment_Runbook_
HighwayEnv.md sec 7-16).

Subclasses ``highway_env.envs.merge_env.ConnectedLaneMergeGenericEnv``
(no fork -- see ``output/highwayenv_migration/M2_FORK_DECISION.md``).
Reuses its ``_make_road()`` unmodified (only the road-length CONFIG
values differ from HighwayEnv's defaults); everything else
(vehicle creation, action type, reward, termination) is overridden with
thesis-specific logic, per the runbook's explicit instruction not to
inherit any of HighwayEnv's own reward/termination/vehicle-population
behaviour.

This is a low-level, single-Gymnasium-instance env: ``reset()``/``step()``
follow the standard Gymnasium multi-agent-by-tuple convention (actions in,
one tuple per controlled vehicle; a scalar reward + one shared
terminated/truncated pair out, per Gymnasium's own API contract), NOT the
dict-per-vehicle-id convention the rest of this project's training code
expects. ``highwayenv_wrapper.StudyBHeterogeneousHighwayEnv`` is the
drop-in-compatible adapter that presents the same dict-keyed API as the
legacy ``thesis.study_b.heterogeneous_env.StudyBHeterogeneousEnv`` -- all
existing downstream code (``shared_local_dqn.py``,
``train_curriculum_stage.py``, evaluators) is meant to only ever import
that wrapper, never this class directly.

Per-vehicle reward/completion/collision detail (which a scalar Gymnasium
``reward``/``terminated`` can't carry) is threaded through via ``info``
(populated in ``_reward()``, read back out in ``_info()`` -- both are
sanctioned ``AbstractEnv`` override points, not a bypass of its API).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from highway_env.envs.merge_env import ConnectedLaneMergeGenericEnv
from highway_env.road.lane import StraightLane

from thesis.envs.stage10_symmetric_merge_env import A_COMFORT, A_HARD
from thesis.rewards.base_reward_v2 import compute_hard_braking_cost
from thesis.study_b.envs.highwayenv_action import (
    ThesisLongitudinalAction,
    ThesisMetaSpeedAction,
    ThesisMultiAgentAction,
)
from thesis.study_b.envs.highwayenv_vehicle import MetaSpeedControlledVehicle, ThesisControlledVehicle
from thesis.study_b.envs.scenario_adapter import mainline_lane_index, place_scenario
from thesis.study_b.scenario_generator import ScenarioSpec, generate_scenario

__all__ = ["ThesisHighwayMergeEnvConfig", "ThesisHighwayMergeEnv"]


@dataclass
class ThesisHighwayMergeEnvConfig:
    """All values recorded, with rationale, in
    ``output/highwayenv_migration/HIGHWAYENV_ENV_FREEZE.json`` once M5
    passes -- do not change silently afterward (runbook sec 31)."""

    policy_frequency: int = 5
    simulation_frequency: int = 15  # 3 physics substeps per policy step
    episode_max_steps: int = 200  # policy decisions -> 200/5 = 40s, matches T_max
    lanes_count: int = 2
    before_merge_length: float = 220.0  # margin over the ~150m max matched-TTC spawn distance
    converge_merge_length: float = 80.0
    parallel_merge_length: float = 80.0
    after_merge_length: float = 120.0  # must be >= 90 (HighwayEnv assert)
    route_exit_margin: float = 90.0  # exit threshold = before+converge+parallel+this
    accelerate_mps2: float = 2.0
    brake_mps2: float = 3.0
    hold_mps2: float = 0.0
    v_min_mps: float = 0.0
    v_max_mps: float = 30.0
    collision_distance_longitudinal_m: float = 4.0
    collision_lateral_threshold_m: float = 1.5
    progress_reward_weight: float = 0.4
    exit_reward_magnitude: float = 0.6
    collision_penalty_magnitude: float = 1.0  # Study B override, not the module default 5.0
    hard_braking_eta: float = 0.015
    time_cost_per_step: float = 0.0005
    # M6-R3 (runbook sec 36) action-representation comparison:
    # "direct_accel" (default, representation A -- ThesisLongitudinalAction/
    # ThesisControlledVehicle, three fixed acceleration bins) or
    # "meta_speed" (representation B -- ThesisMetaSpeedAction/
    # MetaSpeedControlledVehicle, HighwayEnv's own unmodified target-speed
    # cruise controller). See
    # output/highwayenv_migration/M6_R3_ACTION_REPRESENTATION_COMPARISON.md.
    action_representation: str = "direct_accel"


class ThesisHighwayMergeEnv(ConnectedLaneMergeGenericEnv):
    N_VEHICLES = 4

    def __init__(self, thesis_config: ThesisHighwayMergeEnvConfig | None = None, **kwargs):
        self.thesis_config = thesis_config or ThesisHighwayMergeEnvConfig()
        # AbstractEnv.__init__ (called via super().__init__ below) forces an
        # immediate throwaway reset() before any caller gets a chance to
        # stage a real scenario -- give it a deterministic placeholder so
        # that constructor-time reset doesn't crash; the wrapper's own
        # reset(seed=...) call (always made immediately after construction)
        # overwrites this with the real scenario right away.
        self._scenario_to_place: ScenarioSpec | None = generate_scenario(
            scenario_id="__construction_placeholder__",
            episode_seed=0,
            role_members={"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]},
            traffic_type="heterogeneous",
            merge_start=(thesis_config or ThesisHighwayMergeEnvConfig()).before_merge_length,
        )
        self._vehicle_ids: tuple[str, ...] = ()
        self._vehicle_by_id: dict[str, ThesisControlledVehicle] = {}
        self._role_by_id: dict[str, str] = {}
        self._target_speed_by_id: dict[str, float] = {}
        self._completed: dict[str, bool] = {}
        self._policy_step_count: int = 0
        self._last_step_info: dict[str, Any] = {}
        self._route_exit_x: float = 0.0
        self._merge_start_x: float = 0.0
        self._route_total_x: float = 0.0
        super().__init__(config=self._gym_config(), **kwargs)

    # ---------------------------------------------------------- config
    def _gym_config(self) -> dict:
        c = self.thesis_config
        return {
            "policy_frequency": c.policy_frequency,
            "simulation_frequency": c.simulation_frequency,
            "lanes_count": c.lanes_count,
            "vehicles_count": 0,  # no ambient IDM traffic -- only the 4 thesis vehicles
            "before_merge_length": c.before_merge_length,
            "converge_merge_length": c.converge_merge_length,
            "parallel_merge_length": c.parallel_merge_length,
            "after_merge_length": c.after_merge_length,
            "neighbour_vehicles_connected_lanes": True,
            "duration": 10_000,  # not used -- our own _is_truncated() governs episode length
            "offroad_terminal": False,
        }

    @classmethod
    def default_config(cls) -> dict:
        return super().default_config()

    # ------------------------------------------------------ spaces glue
    def define_spaces(self) -> None:
        # Called TWICE by AbstractEnv.reset(): once before controlled_vehicles
        # exists (harmless -- ThesisMultiAgentAction just builds an empty
        # tuple of sub-action-types), once after _reset() has populated
        # self.controlled_vehicles (real 4-way tuple action space).
        c = self.thesis_config
        if c.action_representation == "meta_speed":
            self.action_type = ThesisMultiAgentAction(self, sub_action_cls=ThesisMetaSpeedAction)
        else:
            self.action_type = ThesisMultiAgentAction(
                self, sub_action_cls=ThesisLongitudinalAction,
                accelerate_mps2=c.accelerate_mps2, brake_mps2=c.brake_mps2,
                hold_mps2=c.hold_mps2,
            )
        self.action_space = self.action_type.space()
        # No HighwayEnv-native ObservationType is used -- the wrapper builds
        # thesis local observations directly from vehicle state (sec 14.1).
        # This placeholder only exists so AbstractEnv.reset()/step()'s
        # unconditional self.observation_type.observe() call doesn't crash.
        self.observation_type = _PlaceholderObservation()
        self.observation_space = self.observation_type.space()

    # ------------------------------------------------------------ reset
    def _reset(self) -> None:
        c = self.thesis_config
        self._make_road()
        self._route_exit_x = c.before_merge_length + c.converge_merge_length + c.parallel_merge_length + c.route_exit_margin
        self._merge_start_x = c.before_merge_length
        self._route_total_x = c.before_merge_length + c.converge_merge_length + c.parallel_merge_length + c.after_merge_length
        # Vehicle speed bounds match the legacy simulator's v_min/v_max exactly.
        ThesisControlledVehicle.MIN_SPEED = c.v_min_mps
        ThesisControlledVehicle.MAX_SPEED = c.v_max_mps
        MetaSpeedControlledVehicle.MIN_SPEED = c.v_min_mps
        MetaSpeedControlledVehicle.MAX_SPEED = c.v_max_mps

        if self._scenario_to_place is None:
            raise RuntimeError(
                "ThesisHighwayMergeEnv._reset() called with no scenario staged -- "
                "set env._scenario_to_place before calling reset() (the "
                "StudyBHeterogeneousHighwayEnv wrapper does this automatically)."
            )
        scenario = self._scenario_to_place
        vehicle_cls = MetaSpeedControlledVehicle if c.action_representation == "meta_speed" else ThesisControlledVehicle
        vehicle_by_id = place_scenario(
            self.road, scenario,
            before_merge_length=c.before_merge_length,
            lanes_count=c.lanes_count,
            vehicle_cls=vehicle_cls,
        )
        self._vehicle_ids = tuple(vehicle_by_id.keys())
        self._vehicle_by_id = vehicle_by_id
        self._role_by_id = {vid: scenario.vehicles[vid].role for vid in self._vehicle_ids}
        self._target_speed_by_id = {vid: scenario.vehicles[vid].target_speed for vid in self._vehicle_ids}
        self._completed = {vid: False for vid in self._vehicle_ids}
        self.controlled_vehicles = [vehicle_by_id[vid] for vid in self._vehicle_ids]
        self._policy_step_count = 0
        self._last_step_info = {
            "per_vehicle_reward": {vid: 0.0 for vid in self._vehicle_ids},
            "active": {vid: True for vid in self._vehicle_ids},
            "completed_this_step": {vid: False for vid in self._vehicle_ids},
            "collision_event": False,
            "collision_pairs": [],
            "collision_this_step": {vid: False for vid in self._vehicle_ids},
        }

    # -------------------------------------------------------------- x/y
    def world_xy(self, vehicle: ThesisControlledVehicle) -> tuple[float, float]:
        return float(vehicle.position[0]), float(vehicle.position[1])

    def distance_to_merge(self, vehicle: ThesisControlledVehicle) -> float:
        x, _y = self.world_xy(vehicle)
        return float(self._merge_start_x - x)

    def route_progress_fraction(self, vehicle: ThesisControlledVehicle) -> float:
        x, _y = self.world_xy(vehicle)
        return float(np.clip(x / self._route_total_x, 0.0, 1.0))

    # ------------------------------------------------------------ reward
    def _reward(self, action) -> float:
        c = self.thesis_config
        self._policy_step_count += 1
        completed_before_step = dict(self._completed)

        collided_ids: set[str] = set()
        collision_pairs: list[tuple[str, str]] = []
        for i, a_id in enumerate(self._vehicle_ids):
            for b_id in self._vehicle_ids[i + 1:]:
                va, vb = self._vehicle_by_id[a_id], self._vehicle_by_id[b_id]
                xa, ya = self.world_xy(va)
                xb, yb = self.world_xy(vb)
                if (
                    abs(xa - xb) <= c.collision_distance_longitudinal_m
                    and abs(ya - yb) < c.collision_lateral_threshold_m
                ):
                    collided_ids.add(a_id)
                    collided_ids.add(b_id)
                    collision_pairs.append(tuple(sorted((a_id, b_id))))
        collision_event = len(collided_ids) > 0

        completed_this_step: dict[str, bool] = {vid: False for vid in self._vehicle_ids}
        if not collision_event:
            for vid in self._vehicle_ids:
                if self._completed[vid]:
                    continue
                x, _y = self.world_xy(self._vehicle_by_id[vid])
                if x >= self._route_exit_x:
                    completed_this_step[vid] = True

        reward_by_vid: dict[str, float] = {}
        for vid in self._vehicle_ids:
            vehicle = self._vehicle_by_id[vid]
            rho_t1 = self.route_progress_fraction(vehicle)
            # rho_t: approximate via reversing this step's displacement (dt small,
            # substep-integrated exactly by the physics -- this recomputes the
            # SAME progress fraction the previous step's rho_t1 already was,
            # since route_progress_fraction is a pure function of position).
            # Recorded from the position BEFORE this policy step via _pre_step_x,
            # snapshotted in step() below.
            x_t = self._pre_step_x.get(vid, vehicle.position[0])
            rho_t = float(np.clip(x_t / self._route_total_x, 0.0, 1.0))
            delta = rho_t1 - rho_t

            vehicle_collided = vid in collided_ids
            is_active_at_start = not completed_before_step[vid]
            # vehicle.action["acceleration"] (set by Vehicle.act(), always
            # populated regardless of action representation) rather than
            # the representation-A-only commanded_acceleration attribute,
            # so this reward code works unchanged under M6-R3's
            # meta_speed representation too.
            h_cost = (
                compute_hard_braking_cost(vehicle.action["acceleration"], A_COMFORT, A_HARD)
                if c.hard_braking_eta != 0.0
                else 0.0
            )
            r = (
                c.progress_reward_weight * delta
                + c.exit_reward_magnitude * (1.0 if completed_this_step[vid] else 0.0)
                - c.collision_penalty_magnitude * (1.0 if vehicle_collided else 0.0)
                - c.hard_braking_eta * h_cost
                - c.time_cost_per_step * (1.0 if is_active_at_start else 0.0)
            )
            reward_by_vid[vid] = float(r)

        for vid in self._vehicle_ids:
            if completed_this_step[vid]:
                self._completed[vid] = True
                # Freeze physical control from the NEXT step onward, matching
                # the legacy simulator's "if self._completed[vid]: a = 0.0"
                # convention -- see highwayenv_vehicle.py's docstring note
                # (audit finding, 2026-08-16).
                self._vehicle_by_id[vid].frozen = True

        self._last_step_info = {
            "per_vehicle_reward": reward_by_vid,
            "active": {vid: not self._completed[vid] for vid in self._vehicle_ids},
            "completed_this_step": completed_this_step,
            "collision_event": collision_event,
            "collision_pairs": collision_pairs,
            "collision_this_step": {vid: (vid in collided_ids) for vid in self._vehicle_ids},
        }
        return float(sum(reward_by_vid.values()))

    def _rewards(self, action) -> dict[str, float]:
        return dict(self._last_step_info.get("per_vehicle_reward", {}))

    # -------------------------------------------------------- terminal
    def _is_terminated(self) -> bool:
        if self._last_step_info.get("collision_event", False):
            return True
        return all(self._completed.values())

    def _is_truncated(self) -> bool:
        if self._is_terminated():
            return False
        return self._policy_step_count >= self.thesis_config.episode_max_steps

    def _info(self, obs, action=None) -> dict:
        info = dict(self._last_step_info)
        info["roles"] = dict(self._role_by_id)
        info["target_speeds"] = dict(self._target_speed_by_id)
        info["step_count"] = self._policy_step_count
        return info

    # --------------------------------------------------------- step hook
    def step(self, action):
        self._pre_step_x = {
            vid: float(self._vehicle_by_id[vid].position[0]) for vid in self._vehicle_ids
        }
        return super().step(action)


class _PlaceholderObservation:
    """See ``ThesisHighwayMergeEnv.define_spaces()`` docstring note --
    never consumed for anything except satisfying ``AbstractEnv``'s
    unconditional ``self.observation_type.observe()`` call."""

    def space(self):
        from gymnasium import spaces

        return spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

    def observe(self):
        return np.zeros((1,), dtype=np.float64)
