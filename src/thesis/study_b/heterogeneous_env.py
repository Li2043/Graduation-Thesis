"""Study B N=4 heterogeneous-mobility environment wrapper.

Wraps (does NOT modify) the validated ``Stage10SymmetricMergeEnv`` physics
engine, per new_research_plan.md's explicit instruction to reuse validated
road geometry/vehicle dynamics rather than rewrite the simulator. Three
things this wrapper adds on top of the raw env, none of which needed a
single line of that class changed:

1. **Matched-TTC heterogeneous spawn** (``scenario_generator.py``): right
   after calling the underlying env's own ``reset()`` (which handles
   physical-id <-> role permutation), this wrapper overwrites each active
   vehicle's ``route_position``/``speed`` to the matched-TTC spawn plan --
   safe because ``target_speed`` was confirmed (by direct grep during this
   package's design) to never be read anywhere inside the underlying env's
   own physics/reward code; it is a purely external bookkeeping value.
2. **Local partial observation** (``local_observation.py``) instead of the
   underlying env's own ``_obs_from_vehicles()`` (which exposes exact
   neighbour gap/speed -- closer to a sliced-centralized observation than
   genuine partial observability).
3. **Per-vehicle target-speed-aware utility/burden bookkeeping**
   (``episode_traces()``), following the EXACT same pre-step-snapshot
   convention ``stage11_dyad_merge_runner.py`` already uses for
   ``on_road_attainment`` (snapshot completion status and attainment
   BEFORE calling ``step()``, append only if the vehicle was still active
   at that point) -- not reinvented here, just mirrored, so Study A and
   Study B's utility numbers stay methodologically comparable.

The underlying env's own reward computation (progress/exit/collision/
hard-braking/time-cost) is reused byte-for-byte via ``step()``'s existing
keyword overrides -- this wrapper does not reimplement the base reward,
only adds the PBRS welfare-shaping term on top (in ``pbrs_reward.py``, kept
separate so the base-reward-vs-shaping distinction stays visible in code,
not just in the paper)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from thesis.envs.stage10_symmetric_merge_env import (
    EXIT_REWARD_MAGNITUDE,
    Stage10MergeEnvConfig,
    Stage10RouteGeometry,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage11_welfare import target_speed_attainment
from thesis.study_b.local_observation import (
    GLOBAL_STATE_DIM,
    LOCAL_OBS_DIM,
    VehicleSnapshot,
    build_global_state,
    build_local_observation,
)
from thesis.study_b.scenario_generator import ScenarioSpec, generate_scenario
from thesis.study_b.utility import EpisodeVehicleTrace

__all__ = [
    "STUDY_B_N_VEHICLES",
    "BASE_REWARD_KWARGS",
    "base_reward_kwargs",
    "StudyBEnvConfig",
    "StudyBHeterogeneousEnv",
]

STUDY_B_N_VEHICLES = 4

# Base reward parametrization: reuses Stage 11 v4/v5's already-validated
# values verbatim (stage11_dyad_merge_pilot_config.py:
# COLLISION_PENALTY_MAGNITUDE_V4=1.0, EXIT_REWARD_MAGNITUDE_V4=0.6 [same
# numeric value as the module default EXIT_REWARD_MAGNITUDE, imported
# below for self-documentation], HARD_BRAKING_ETA_V4=0.015,
# TIME_COST_PER_STEP_V4=0.0005, TTC_PENALTY_WEIGHT_V4=0.0/disabled).
#
# Kept EXACTLY as the original new_research_plan.md-era default (time cost
# INCLUDED) -- this is the reward the already-running PBRS/baseline
# solver-diagnostic qualification depends on. revised_new_research_plan.md
# / change.md's direct-welfare redesign DROPS the time cost for the formal
# study; that is controlled per-instance via
# ``StudyBEnvConfig.include_time_cost`` (see ``base_reward_kwargs`` below),
# NOT by changing this constant or this class's default -- so any script
# that doesn't explicitly ask for the new behaviour (e.g. re-running
# train_dqn_fallback.py later) keeps getting byte-identical old behaviour.
BASE_REWARD_KWARGS: dict[str, float] = {
    "collision_penalty_magnitude": 1.0,
    "ttc_penalty_weight": 0.0,
    "exit_reward_magnitude": EXIT_REWARD_MAGNITUDE,
    "hard_braking_eta": 0.015,
    "time_cost_per_step": 0.0005,
}


def base_reward_kwargs(*, include_time_cost: bool) -> dict[str, float]:
    """Fresh dict, same as ``BASE_REWARD_KWARGS`` except
    ``time_cost_per_step`` is zeroed when ``include_time_cost=False`` --
    revised_new_research_plan.md sec 12 / change.md #4: remove the
    ``-0.0005`` active-step cost for the direct-welfare formal study,
    while keeping the timeout penalty (``EXIT_REWARD_MAGNITUDE``/collision/
    hard-braking terms are computed by the underlying env unconditionally;
    the timeout penalty itself is baked into that env's base reward
    formula, not one of these kwargs, so it is unaffected either way)."""
    kwargs = dict(BASE_REWARD_KWARGS)
    if not include_time_cost:
        kwargs["time_cost_per_step"] = 0.0
    return kwargs


@dataclass
class StudyBEnvConfig:
    episode_max_steps: int = 200  # Phase-0 pilot default, see module docstring
    merge_start: float = 200.0
    merge_end: float = 300.0
    route_start: float = 0.0
    route_exit: float = 400.0
    r_obs: float = 50.0
    v_scale: float = 10.0
    heterogeneous_probability: float = 0.5  # new_research_plan.md's training-distribution split
    # Default True: preserves EXACT prior behaviour for any code that
    # doesn't know about this field (e.g. train_dqn_fallback.py). The new
    # direct-welfare training script explicitly passes False.
    include_time_cost: bool = True


class StudyBHeterogeneousEnv:
    """Multi-agent Dict-keyed API (``reset``/``step`` return
    ``dict[vehicle_id, ...]``), matching this project's existing
    ``Stage10SymmetricMergeEnv`` convention closely enough that
    ``mappo.py``/``shared_local_dqn.py`` stay simple, and so
    ``pettingzoo_wrapper.py``'s adapter is a thin translation layer."""

    def __init__(self, config: StudyBEnvConfig | None = None):
        self.config = config or StudyBEnvConfig()
        self._env = Stage10SymmetricMergeEnv(
            Stage10MergeEnvConfig(
                n_vehicles=STUDY_B_N_VEHICLES,
                max_steps=self.config.episode_max_steps,
                geometry=Stage10RouteGeometry(
                    route_start=self.config.route_start,
                    merge_start=self.config.merge_start,
                    merge_end=self.config.merge_end,
                    route_exit=self.config.route_exit,
                ),
                include_role_zone_features=False,
                include_priority_signal=False,
            )
        )
        self._scenario: ScenarioSpec | None = None
        self._traces: dict[str, EpisodeVehicleTrace] = {}
        self._previous_action: dict[str, int] = {}

    # ------------------------------------------------------------ helpers
    def _role_members(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for vid in self._env.active_vehicle_ids:
            out.setdefault(self._env.role_of(vid), []).append(vid)
        return out

    def _apply_scenario(self, scenario: ScenarioSpec) -> None:
        """Overrides route_position/speed AND role for every vehicle to
        exactly match ``scenario``. Role must also be forced (not just
        position/speed): the underlying env's own ``reset()`` runs its OWN
        independent role permutation from its internal RNG, which will
        generally NOT match a scenario generated under a different
        seed/reset call (e.g. replaying a frozen evaluation-bank scenario
        against a live env reset with a different seed). Leaving role
        un-overridden would silently desync a vehicle's assumed role (and
        therefore its lane offset via ``world_y_for``) from the physics
        engine's actual role for that vehicle id -- exactly the bug this
        wrapper's own reproducibility test caught."""
        for vid, spec in scenario.vehicles.items():
            self._env._roles[vid] = spec.role  # noqa: SLF001
            veh = self._env._vehicles[vid]  # noqa: SLF001
            veh.role = spec.role
            veh.route_position = spec.route_position
            veh.speed = spec.spawn_speed
            veh.acceleration = 0.0

    def _snapshot(self, vid: str) -> VehicleSnapshot:
        veh = self._env._vehicles[vid]  # noqa: SLF001
        spec = self._scenario.vehicles[vid]  # type: ignore[union-attr]
        return VehicleSnapshot(
            vehicle_id=vid,
            role=veh.role,
            speed=veh.speed,
            route_position=veh.route_position,
            acceleration=veh.acceleration,
            target_speed=spec.target_speed,
            active=not self._env.is_completed(vid),
            previous_action=self._previous_action.get(vid, 0),
        )

    def _build_observations(self) -> dict[str, np.ndarray]:
        snapshots = {vid: self._snapshot(vid) for vid in self._env.active_vehicle_ids}
        out: dict[str, np.ndarray] = {}
        for vid, ego in snapshots.items():
            others = [s for other_vid, s in snapshots.items() if other_vid != vid]
            out[vid] = build_local_observation(
                ego, others, merge_start=self.config.merge_start,
                r_obs=self.config.r_obs, v_scale=self.config.v_scale,
            )
        return out

    def global_state(self) -> np.ndarray:
        """Centralized-critic-only input (MAPPO). Not used by the DQN
        fallback, and NEVER exposed to any actor -- see
        ``local_observation.build_global_state``'s docstring."""
        snapshots = [self._snapshot(vid) for vid in self._env.active_vehicle_ids]
        return build_global_state(snapshots, merge_start=self.config.merge_start)

    def _append_pre_step_trace_sample(self) -> None:
        """Mirrors stage11_dyad_merge_runner.py's on_road_attainment
        convention exactly: snapshot BEFORE calling the underlying env's
        step(), append only for vehicles still active at that point."""
        for vid in self._env.active_vehicle_ids:
            if self._env.is_completed(vid):
                continue
            veh = self._env._vehicles[vid]  # noqa: SLF001
            self._traces[vid].speeds.append(veh.speed)
            self._traces[vid].active_flags.append(True)
            self._traces[vid].accelerations.append(veh.acceleration)

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
        return self._env.active_vehicle_ids

    def reset(
        self,
        *,
        seed: int,
        scenario: ScenarioSpec | None = None,
        traffic_type: str | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """``scenario``: pass a pre-generated, hashed ``ScenarioSpec`` (from
        a frozen scenario bank) for EVALUATION -- guarantees the exact same
        episode is replayed regardless of caller. Leave ``None`` for
        TRAINING, where a fresh scenario is generated from ``seed`` each
        call (``traffic_type`` sampled heterogeneous/homogeneous per
        ``config.heterogeneous_probability`` unless explicitly forced)."""
        self._env.reset(seed=seed)  # role permutation; own returned obs discarded, see module docstring
        role_members = self._role_members()

        if scenario is not None:
            self._scenario = scenario
        else:
            if traffic_type is None:
                rng = np.random.default_rng(seed)
                traffic_type = "heterogeneous" if rng.random() < self.config.heterogeneous_probability else "homogeneous"
            self._scenario = generate_scenario(
                scenario_id=f"seed_{seed}",
                episode_seed=seed,
                role_members=role_members,
                traffic_type=traffic_type,
                merge_start=self.config.merge_start,
            )
        self._apply_scenario(self._scenario)
        # Recompute AFTER applying the scenario: when scenario is not
        # None, this may differ from the pre-override role_members above
        # (see _apply_scenario's docstring) -- the info dict must reflect
        # what actually ended up in the live physics, not the discarded
        # pre-override permutation.
        role_members = self._role_members()

        self._previous_action = {vid: 0 for vid in self._env.active_vehicle_ids}
        self._traces = {
            vid: EpisodeVehicleTrace(vehicle_id=vid, target_speed=self._scenario.vehicles[vid].target_speed)
            for vid in self._env.active_vehicle_ids
        }

        obs = self._build_observations()
        info = {
            "roles": dict(role_members),
            "scenario_id": self._scenario.scenario_id,
            "traffic_type": self._scenario.traffic_type,
            "target_speeds": {vid: v.target_speed for vid, v in self._scenario.vehicles.items()},
        }
        return obs, info

    def step(
        self, actions: Mapping[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict[str, Any]]:
        pre_step_active = {vid: not self._env.is_completed(vid) for vid in self._env.active_vehicle_ids}
        self._append_pre_step_trace_sample()

        reward_kwargs = base_reward_kwargs(include_time_cost=self.config.include_time_cost)
        _obs_unused, base_reward, terminated, truncated, info = self._env.step(dict(actions), **reward_kwargs)
        self._previous_action = {vid: int(a) for vid, a in actions.items()}

        collision_event = bool(info["collision_event"])
        if collision_event:
            colliding_ids = {vid for pair in info["collision_pairs"] for vid in pair}
            for vid in colliding_ids:
                # A vehicle already completed BEFORE this step should not
                # have an already-banked success zeroed by a later physical
                # contact -- mirrors stage11_welfare.episode_collided_for_vehicle.
                if pre_step_active.get(vid, False):
                    self._traces[vid].collided = True

        obs = self._build_observations()
        attainments = {
            vid: target_speed_attainment(self._env._vehicles[vid].speed, self._scenario.vehicles[vid].target_speed)  # noqa: SLF001
            for vid in self._env.active_vehicle_ids
        }
        active_flags = {vid: not self._env.is_completed(vid) for vid in self._env.active_vehicle_ids}

        out_info = dict(info)
        out_info["attainments"] = attainments
        out_info["active"] = active_flags
        return obs, dict(base_reward), bool(terminated), bool(truncated), out_info

    def episode_traces(self) -> dict[str, EpisodeVehicleTrace]:
        return dict(self._traces)

    def dt(self) -> float:
        return float(self._env.config.dt)
