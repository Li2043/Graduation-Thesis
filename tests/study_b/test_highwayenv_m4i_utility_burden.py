"""M4-I -- utility and burden gate (runbook sec 26).

``target_speed_attainment``/``episode_utilities``/``episode_burdens``
(``thesis.study_b.utility``, ``thesis.pilots.stage11_welfare``) are
UNCHANGED, backend-agnostic pure functions -- already exercised at the
unit level by this project's pre-existing test suite. What is new and
needs its own check is that ``StudyBHeterogeneousHighwayEnv`` actually
feeds them correctly through the real environment: attainment values
computed from live HighwayEnv vehicle speed, and the collision-zeroing
rule (U_i=0 after collision) firing correctly off the frozen-definition
collision check (M4-H)."""

from __future__ import annotations

import pytest

from thesis.pilots.stage11_welfare import target_speed_attainment
from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv
from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec
from thesis.study_b.utility import episode_burdens, episode_utilities

_C = ThesisHighwayMergeEnvConfig()


def _ttc(target_speed: float, desired_x: float) -> float:
    return (_C.before_merge_length - desired_x) / target_speed


def test_attainment_formula_matches_gate_spec_exactly():
    assert target_speed_attainment(18.0, 18.0) == pytest.approx(1.0)
    assert target_speed_attainment(22.0, 22.0) == pytest.approx(1.0)
    assert target_speed_attainment(15.0, 18.0) < 1.0
    assert target_speed_attainment(18.0, 22.0) < 1.0


def _far_apart_scenario() -> ScenarioSpec:
    """Every vehicle far from every other -- guaranteed no collision, so
    U_i is governed purely by attainment (S_i=1 as long as it doesn't
    time out)."""
    specs = {
        "V0": VehicleSpawnSpec(vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
                                target_speed=18.0, spawn_speed=18.0, route_position=150.0, nominal_ttc=_ttc(18.0, 150.0)),
        "V1": VehicleSpawnSpec(vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
                                target_speed=18.0, spawn_speed=18.0, route_position=50.0, nominal_ttc=_ttc(18.0, 50.0)),
        "V2": VehicleSpawnSpec(vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
                                target_speed=22.0, spawn_speed=22.0, route_position=100.0, nominal_ttc=_ttc(22.0, 100.0)),
        "V3": VehicleSpawnSpec(vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
                                target_speed=22.0, spawn_speed=22.0, route_position=10.0, nominal_ttc=_ttc(22.0, 10.0)),
    }
    return ScenarioSpec(scenario_id="m4i_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)


def test_utility_and_burden_through_real_env_hand_verified():
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=0, scenario=_far_apart_scenario())
    # 5 HOLD steps at spawn_speed == target_speed -> attainment == 1.0
    # every step for every vehicle -> U_i should come out to exactly 1.0
    # (no collision, and note truncation partway through does not itself
    # zero U_i under this project's S_i rule unless collided -- burden
    # should come out to exactly 0.0 for the same reason).
    for _ in range(5):
        env.step({vid: 0 for vid in env.active_vehicle_ids})
    traces = env.episode_traces()
    utilities = episode_utilities(traces)
    burdens = episode_burdens(traces, dt=env.dt())
    for vid in env.active_vehicle_ids:
        assert utilities[vid] == pytest.approx(1.0), (vid, utilities[vid])
        assert burdens[vid] == pytest.approx(0.0, abs=1e-9), (vid, burdens[vid])


def test_utility_zeroed_after_collision():
    from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig as C

    specs = {
        "V0": VehicleSpawnSpec(vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
                                target_speed=18.0, spawn_speed=18.0, route_position=101.0, nominal_ttc=_ttc(18.0, 101.0)),
        "V1": VehicleSpawnSpec(vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
                                target_speed=18.0, spawn_speed=18.0, route_position=100.0, nominal_ttc=_ttc(18.0, 100.0)),
        "V2": VehicleSpawnSpec(vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
                                target_speed=22.0, spawn_speed=22.0, route_position=50.0, nominal_ttc=_ttc(22.0, 50.0)),
        "V3": VehicleSpawnSpec(vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
                                target_speed=22.0, spawn_speed=22.0, route_position=10.0, nominal_ttc=_ttc(22.0, 10.0)),
    }
    scenario = ScenarioSpec(scenario_id="m4i_collision_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=0, scenario=scenario)
    _obs, _rew, terminated, _trunc, info = env.step({vid: 0 for vid in env.active_vehicle_ids})
    assert terminated and info["collision_event"]
    traces = env.episode_traces()
    utilities = episode_utilities(traces)
    assert utilities["V0"] == pytest.approx(0.0)
    assert utilities["V1"] == pytest.approx(0.0)
