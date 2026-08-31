"""``StudyBHeterogeneousHighwayEnv`` -- the HighwayEnv-backend drop-in
replacement for ``thesis.study_b.heterogeneous_env.StudyBHeterogeneousEnv``.

Presents the EXACT same dict-per-vehicle-id API (``reset``/``step``
signatures, ``active_vehicle_ids``, ``episode_traces()``, ``dt()``,
``global_state()``, ``observation_dim``, ``n_actions``) so every existing
downstream script (``shared_local_dqn.py``, ``train_curriculum_stage.py``,
``evaluate_policy.py``, etc.) can switch backends by only changing which
class gets instantiated -- per runbook sec 7's recommended architecture
and this migration's overriding goal of not touching the DQN/training
code at all.

Reuses ``thesis.study_b.local_observation`` and
``thesis.study_b.scenario_generator`` UNMODIFIED (both are
environment-free pure functions/dataclasses -- see
``output/highwayenv_migration/CODE_PROVENANCE.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from thesis.pilots.stage11_welfare import target_speed_attainment
from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnv, ThesisHighwayMergeEnvConfig
from thesis.study_b.local_observation import (
    GLOBAL_STATE_DIM,
    LOCAL_OBS_DIM,
    VehicleSnapshot,
    build_global_state,
    build_local_observation,
)
from thesis.study_b.scenario_generator import ScenarioSpec, generate_scenario
from thesis.study_b.utility import EpisodeVehicleTrace, running_active_attainment

__all__ = ["StudyBHighwayWrapperConfig", "StudyBHeterogeneousHighwayEnv"]

_VEHICLE_ID_POOL = ("V0", "V1", "V2", "V3")


@dataclass
class StudyBHighwayWrapperConfig:
    env_config: ThesisHighwayMergeEnvConfig | None = None
    heterogeneous_probability: float = 0.5
    r_obs: float = 50.0
    v_scale: float = 10.0
    # LOCALITY AMENDMENT (frozen 2026-08-17, see LOCAL_SENSING_AMENDMENT.md):
    # a genuine finite neighbour-visibility range, distinct from ``r_obs``
    # (which is only a delta_d normalization/clip divisor, never an
    # exclusion radius). ``None`` preserves the exact pre-amendment
    # behaviour (unbounded nearest-3 selection) byte-for-byte.
    local_sensing_range_m: float | None = None
    # WSC (Welfare-State Communication) additive mode, default False =
    # Original 18D behaviour byte-identical. When True, _snapshot()
    # additionally computes each vehicle's own pre-decision M_i(t) from its
    # OWN episode trace (thesis.study_b.utility.running_active_attainment)
    # and _build_observations() passes include_welfare_state=True through
    # to build_local_observation, producing 22D observations instead of
    # 18D. Does not change R=50m, neighbour selection/sorting, presence
    # masking, or any Original field.
    include_welfare_state: bool = False


class StudyBHeterogeneousHighwayEnv:
    def __init__(self, config: StudyBHighwayWrapperConfig | None = None):
        self.config = config or StudyBHighwayWrapperConfig()
        self._env = ThesisHighwayMergeEnv(self.config.env_config)
        self._scenario: ScenarioSpec | None = None
        self._traces: dict[str, EpisodeVehicleTrace] = {}
        self._previous_action: dict[str, int] = {}
        self._merge_start_for_obs: float = self._env.thesis_config.before_merge_length

    # ------------------------------------------------------------ helpers
    def _role_members(self, *, seed: int) -> dict[str, list[str]]:
        # Salted, NOT the same seed passed to generate_scenario()'s own
        # internal RNG below -- reusing the identical seed for both
        # decisions was found (M4-E gate) to statistically entangle
        # physical-id<->role assignment with the generator's own
        # speed_class/ttc_slot permutation (both RNGs draw from the same
        # underlying bit-stream), producing a measured ~69/31 skew in
        # which physical id got "fast" vs "slow" instead of the required
        # ~50/50. The salt decorrelates the two draws.
        rng = np.random.default_rng(seed * 2_654_435_761 + 1)
        perm = rng.permutation(4)
        ids = [_VEHICLE_ID_POOL[i] for i in perm]
        return {"ramp": ids[:2], "mainline": ids[2:]}

    def _snapshot(self, vid: str) -> VehicleSnapshot:
        vehicle = self._env._vehicle_by_id[vid]  # noqa: SLF001
        spec = self._scenario.vehicles[vid]  # type: ignore[union-attr]
        x, _y = self._env.world_xy(vehicle)
        # WSC (default False -> welfare_state stays at VehicleSnapshot's own
        # 1.0 default, never computed, Original path untouched): M_i(t) is
        # read from THIS vehicle's OWN trace as accumulated by
        # _append_pre_step_trace_sample() so far this step -- i.e. only
        # already-realised, pre-this-step-physics samples (see that
        # method's call ordering in step() below), never the consequence
        # of an action not yet chosen. Reuses running_active_attainment
        # unchanged -- no second/parallel accumulator.
        welfare_state = (
            running_active_attainment(self._traces[vid])
            if self.config.include_welfare_state
            else 1.0
        )
        return VehicleSnapshot(
            vehicle_id=vid,
            role=spec.role,
            speed=float(vehicle.speed),
            route_position=x,
            acceleration=float(vehicle.action["acceleration"]),
            target_speed=spec.target_speed,
            active=not self._env._completed[vid],  # noqa: SLF001
            previous_action=self._previous_action.get(vid, 0),
            welfare_state=welfare_state,
        )

    def _build_observations(self) -> dict[str, np.ndarray]:
        snapshots = {vid: self._snapshot(vid) for vid in self._env._vehicle_ids}  # noqa: SLF001
        out: dict[str, np.ndarray] = {}
        for vid, ego in snapshots.items():
            others = [s for other_vid, s in snapshots.items() if other_vid != vid]
            out[vid] = build_local_observation(
                ego, others, merge_start=self._merge_start_for_obs,
                r_obs=self.config.r_obs, v_scale=self.config.v_scale,
                local_sensing_range_m=self.config.local_sensing_range_m,
                include_welfare_state=self.config.include_welfare_state,
            )
        return out

    def global_state(self) -> np.ndarray:
        snapshots = [self._snapshot(vid) for vid in self._env._vehicle_ids]  # noqa: SLF001
        return build_global_state(snapshots, merge_start=self._merge_start_for_obs)

    def _append_pre_step_trace_sample(self) -> None:
        for vid in self._env._vehicle_ids:  # noqa: SLF001
            if self._env._completed[vid]:  # noqa: SLF001
                continue
            vehicle = self._env._vehicle_by_id[vid]  # noqa: SLF001
            self._traces[vid].speeds.append(float(vehicle.speed))
            self._traces[vid].active_flags.append(True)
            self._traces[vid].accelerations.append(float(vehicle.action["acceleration"]))

    # -------------------------------------------------------------- API
    @property
    def observation_dim(self) -> int:
        return LOCAL_OBS_DIM

    @property
    def global_state_dim(self) -> int:
        return GLOBAL_STATE_DIM

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def active_vehicle_ids(self) -> tuple[str, ...]:
        return self._env._vehicle_ids  # noqa: SLF001

    def reset(
        self,
        *,
        seed: int,
        scenario: ScenarioSpec | None = None,
        traffic_type: str | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if scenario is not None:
            self._scenario = scenario
        else:
            role_members = self._role_members(seed=seed)
            if traffic_type is None:
                rng = np.random.default_rng(seed)
                traffic_type = "heterogeneous" if rng.random() < self.config.heterogeneous_probability else "homogeneous"
            self._scenario = generate_scenario(
                scenario_id=f"seed_{seed}",
                episode_seed=seed,
                role_members=role_members,
                traffic_type=traffic_type,
                merge_start=self._merge_start_for_obs,
            )

        self._env._scenario_to_place = self._scenario  # noqa: SLF001
        self._env.reset(seed=seed)

        self._previous_action = {vid: 0 for vid in self._env._vehicle_ids}  # noqa: SLF001
        self._traces = {
            vid: EpisodeVehicleTrace(vehicle_id=vid, target_speed=self._scenario.vehicles[vid].target_speed)
            for vid in self._env._vehicle_ids  # noqa: SLF001
        }

        obs = self._build_observations()
        info = {
            "roles": self._scenario.role_members(),
            "scenario_id": self._scenario.scenario_id,
            "traffic_type": self._scenario.traffic_type,
            "target_speeds": {vid: v.target_speed for vid, v in self._scenario.vehicles.items()},
        }
        return obs, info

    def step(
        self, actions: Mapping[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict[str, Any]]:
        pre_step_active = {vid: not self._env._completed[vid] for vid in self._env._vehicle_ids}  # noqa: SLF001
        self._append_pre_step_trace_sample()

        action_tuple = tuple(int(actions[vid]) for vid in self._env._vehicle_ids)  # noqa: SLF001
        _obs_unused, _reward_scalar, terminated, truncated, info = self._env.step(action_tuple)
        self._previous_action = {vid: int(a) for vid, a in actions.items()}

        collision_event = bool(info["collision_event"])
        if collision_event:
            colliding_ids = {vid for pair in info["collision_pairs"] for vid in pair}
            for vid in colliding_ids:
                if pre_step_active.get(vid, False):
                    self._traces[vid].collided = True

        obs = self._build_observations()
        attainments = {
            vid: target_speed_attainment(
                float(self._env._vehicle_by_id[vid].speed), self._scenario.vehicles[vid].target_speed  # noqa: SLF001
            )
            for vid in self._env._vehicle_ids  # noqa: SLF001
        }
        active_flags = {vid: not self._env._completed[vid] for vid in self._env._vehicle_ids}  # noqa: SLF001

        # term_reason/exit_event: kept in the exact shape the legacy
        # StudyBHeterogeneousEnv's info dict already used, so scripts
        # written against that API (e.g. train_curriculum_stage.py,
        # run_oracle_controller.py) need minimal changes to target this
        # backend instead.
        if collision_event:
            term_reason = "collision"
        elif terminated:
            term_reason = "success"
        elif truncated:
            term_reason = "truncation"
        else:
            term_reason = "ongoing"

        out_info = dict(info)
        out_info["attainments"] = attainments
        out_info["active"] = active_flags
        out_info["term_reason"] = term_reason
        out_info["exit_event"] = dict(info["completed_this_step"])
        return obs, dict(info["per_vehicle_reward"]), bool(terminated), bool(truncated), out_info

    def episode_traces(self) -> dict[str, EpisodeVehicleTrace]:
        return dict(self._traces)

    def dt(self) -> float:
        return 1.0 / float(self._env.thesis_config.policy_frequency)
