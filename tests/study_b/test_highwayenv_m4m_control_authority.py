"""M4-M -- physical control authority gate (2026-08-16 amendment,
CONTROL_AUTHORITY_MISMATCH). Verifies HighwayEnv's own unmodified
``speed_control()`` proportional controller is still what determines the
REQUESTED acceleration under the ``meta_speed`` representation, but the
REALIZED (physics-affecting) acceleration is always clipped to the
frozen legacy longitudinal envelope ``[-3.0, +2.0]`` m/s^2, on every
physics substep, and that a completed (frozen) vehicle is entirely
unaffected by further policy control (including the clip's own inputs --
its target_speed simply stops changing).

Tests A-E per the amendment's own spec."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
from thesis.study_b.envs.highwayenv_vehicle import MetaSpeedControlledVehicle
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig

_CFG = ThesisHighwayMergeEnvConfig(action_representation="meta_speed")


def _env() -> StudyBHeterogeneousHighwayEnv:
    return StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=_CFG))


# --------------------------------------------------------------- Test A
def test_a_brake_requested_below_bound_realized_clipped_to_minus_3():
    env = _env()
    env.reset(seed=1)
    vid = env.active_vehicle_ids[0]
    v: MetaSpeedControlledVehicle = env._env._vehicle_by_id[vid]  # noqa: SLF001
    # Force a huge downward target_speed gap so speed_control() requests
    # far below -3.0.
    v.target_speed = -50.0
    env.step({vv: 0 for vv in env.active_vehicle_ids})  # HOLD -- don't nudge target_speed further
    assert v.last_requested_acceleration < -3.0
    assert v.last_realized_acceleration == pytest.approx(-3.0, abs=1e-9)
    assert v.action["acceleration"] == pytest.approx(-3.0, abs=1e-9)


# --------------------------------------------------------------- Test B
def test_b_accelerate_requested_above_bound_realized_clipped_to_plus_2():
    env = _env()
    env.reset(seed=1)
    vid = env.active_vehicle_ids[0]
    v: MetaSpeedControlledVehicle = env._env._vehicle_by_id[vid]  # noqa: SLF001
    v.target_speed = 100.0
    env.step({vv: 0 for vv in env.active_vehicle_ids})
    assert v.last_requested_acceleration > 2.0
    assert v.last_realized_acceleration == pytest.approx(2.0, abs=1e-9)
    assert v.action["acceleration"] == pytest.approx(2.0, abs=1e-9)


# --------------------------------------------------------------- Test C
@pytest.mark.parametrize("target_speed_delta", [50.0, -50.0, 0.0])
def test_c_hold_clips_correctly_across_positive_negative_near_zero_requests(target_speed_delta):
    env = _env()
    env.reset(seed=1)
    vid = env.active_vehicle_ids[0]
    v: MetaSpeedControlledVehicle = env._env._vehicle_by_id[vid]  # noqa: SLF001
    v.target_speed = v.speed + target_speed_delta
    env.step({vv: 0 for vv in env.active_vehicle_ids})  # HOLD
    assert -3.0 - 1e-9 <= v.last_realized_acceleration <= 2.0 + 1e-9
    assert v.action["acceleration"] == pytest.approx(v.last_realized_acceleration, abs=1e-9)
    if target_speed_delta == 0.0:
        # Near-zero request: controller output should itself be small and
        # well within bounds (not just coincidentally clipped).
        assert abs(v.last_requested_acceleration) < 2.0


# --------------------------------------------------------------- Test D
def test_d_clip_applies_to_every_physics_substep_not_just_the_first():
    """Force an enormous target_speed gap and confirm the speed change
    over one FULL 0.2s policy step (3 substeps at 15Hz sim / 5Hz policy)
    equals exactly the clipped-rate * dt -- if the clip only applied to
    the first substep and the other two used the raw unclipped
    acceleration, the total speed change would be far larger."""
    env = _env()
    env.reset(seed=1)
    vid = env.active_vehicle_ids[0]
    v: MetaSpeedControlledVehicle = env._env._vehicle_by_id[vid]  # noqa: SLF001
    v.target_speed = 500.0
    speed0 = v.speed
    env.step({vv: 0 for vv in env.active_vehicle_ids})
    speed1 = v.speed
    dt = env.dt()
    assert (speed1 - speed0) == pytest.approx(2.0 * dt, abs=1e-6)


# --------------------------------------------------------------- Test E
def test_e_completed_vehicle_frozen_target_speed_unaffected_by_new_actions():
    """A completed (frozen) vehicle must not receive a NEW clipped control
    action driven by the policy -- its target_speed stops changing
    entirely once frozen (validated by M4-L), and whatever residual
    acceleration speed_control() computes toward that now-fixed target is
    still subject to the same clip (physical realism), but is NOT
    influenced by any further ACCELERATE/HOLD/BRAKE the policy assigns
    it."""
    from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec

    def ttc(ts, x):
        return (_CFG.before_merge_length - x) / ts

    near_exit_x = _CFG.before_merge_length + _CFG.converge_merge_length + _CFG.parallel_merge_length + _CFG.route_exit_margin
    specs = {
        "V0": VehicleSpawnSpec(vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
                                target_speed=18.0, spawn_speed=18.0, route_position=near_exit_x, nominal_ttc=ttc(18.0, near_exit_x)),
        "V1": VehicleSpawnSpec(vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
                                target_speed=18.0, spawn_speed=18.0, route_position=50.0, nominal_ttc=ttc(18.0, 50.0)),
        "V2": VehicleSpawnSpec(vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
                                target_speed=22.0, spawn_speed=22.0, route_position=100.0, nominal_ttc=ttc(22.0, 100.0)),
        "V3": VehicleSpawnSpec(vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
                                target_speed=22.0, spawn_speed=22.0, route_position=10.0, nominal_ttc=ttc(22.0, 10.0)),
    }
    scenario = ScenarioSpec(scenario_id="m4m_frozen_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)
    env = _env()
    env.reset(seed=0, scenario=scenario)
    for _ in range(3):
        env.step({vid: 1 for vid in env.active_vehicle_ids})  # ACCELERATE everyone -> V0 completes
    v0: MetaSpeedControlledVehicle = env._env._vehicle_by_id["V0"]  # noqa: SLF001
    assert v0.frozen is True
    target_speed_at_completion = v0.target_speed

    # Hammer it with alternating ACCELERATE/BRAKE for several steps.
    for i in range(10):
        actions = {vid: 0 for vid in env.active_vehicle_ids}
        actions["V0"] = 1 if i % 2 == 0 else 2
        env.step(actions)
        assert v0.target_speed == pytest.approx(target_speed_at_completion, abs=1e-9)
        assert -3.0 - 1e-9 <= v0.last_realized_acceleration <= 2.0 + 1e-9
