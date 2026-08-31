"""M4-H -- collision gate (runbook sec 25). Three hand-constructed cases
plus a segment-transition-does-not-falsely-collide check."""

from __future__ import annotations

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv
from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec

_C = ThesisHighwayMergeEnvConfig()


def _ttc(target_speed: float, desired_x: float) -> float:
    return (_C.before_merge_length - desired_x) / target_speed


def _scenario(*, v0_x: float, v1_x: float) -> ScenarioSpec:
    """Two ramp vehicles (same lane -> real collision candidates) at the
    given real spawn x, two mainline vehicles far away (never involved)."""
    specs = {
        "V0": VehicleSpawnSpec(
            vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
            target_speed=18.0, spawn_speed=18.0, route_position=v0_x, nominal_ttc=_ttc(18.0, v0_x),
        ),
        "V1": VehicleSpawnSpec(
            vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
            target_speed=18.0, spawn_speed=18.0, route_position=v1_x, nominal_ttc=_ttc(18.0, v1_x),
        ),
        "V2": VehicleSpawnSpec(
            vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
            target_speed=22.0, spawn_speed=22.0, route_position=50.0, nominal_ttc=_ttc(22.0, 50.0),
        ),
        "V3": VehicleSpawnSpec(
            vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
            target_speed=22.0, spawn_speed=22.0, route_position=10.0, nominal_ttc=_ttc(22.0, 10.0),
        ),
    }
    return ScenarioSpec(scenario_id="m4h_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)


def test_obvious_non_collision():
    # 30m apart, both HOLD -- nowhere near the 4.0m/1.5m thresholds.
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=0, scenario=_scenario(v0_x=100.0, v1_x=70.0))
    _obs, _rew, terminated, _trunc, info = env.step({vid: 0 for vid in env.active_vehicle_ids})
    assert info["collision_event"] is False
    assert terminated is False


def test_obvious_overlap_collision():
    # 1m apart, same lane (same y) -- immediate collision at reset-adjacent step.
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=0, scenario=_scenario(v0_x=101.0, v1_x=100.0))
    _obs, _rew, terminated, _trunc, info = env.step({vid: 0 for vid in env.active_vehicle_ids})
    assert info["collision_event"] is True
    assert ("V0", "V1") in info["collision_pairs"]
    assert terminated is True


def test_near_but_non_overlapping_state():
    # Exactly at the longitudinal threshold boundary (4.5m > 4.0m) -- must NOT collide.
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=0, scenario=_scenario(v0_x=104.5, v1_x=100.0))
    _obs, _rew, terminated, _trunc, info = env.step({vid: 0 for vid in env.active_vehicle_ids})
    assert info["collision_event"] is False
    assert terminated is False


def test_no_collision_triggered_merely_by_lane_segment_transition():
    """Two vehicles 10m apart straddling node "b" (different lane_index
    labels) must not be flagged as colliding purely because their
    lane_index differs -- collision is a pure real-world-position check,
    lane identity never enters it."""
    boundary = _C.before_merge_length + _C.converge_merge_length
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=0, scenario=_scenario(v0_x=boundary - 5.0, v1_x=boundary + 5.0))
    assert env._env._vehicle_by_id["V0"].lane_index[:2] != env._env._vehicle_by_id["V1"].lane_index[:2]  # noqa: SLF001
    _obs, _rew, terminated, _trunc, info = env.step({vid: 0 for vid in env.active_vehicle_ids})
    assert info["collision_event"] is False
    assert terminated is False
