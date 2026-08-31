"""M4-L -- completed-vehicle inactivity semantics gate (runbook amendment,
2026-08-16 audit finding).

A vehicle that has already safely completed (exited past
``route_exit_x``) must stop responding to further DQN actions entirely,
matching the legacy simulator's explicit
``if self._completed[vid]: a = 0.0`` convention. Before the fix, a
completed vehicle kept applying every subsequent action -- including
BRAKE -- which could decelerate it to a stop just past the exit line,
turning it into a stationary same-lane obstacle a still-active trailing
vehicle could then "collide" with, wrongly terminating the episode for
everyone. This is a plausible ADDITIONAL contributor to the same-lane
collision dominance found in M6 (not claimed to be the sole cause --
curve geometry and exploration dynamics may still also contribute).

Tests A/B/C (front vehicle completes, rear vehicle remains active,
completed vehicle then hammered with BRAKE / ACCELERATE / HOLD
respectively) verify the required invariant directly:

    state_completed(t+1 | ACCELERATE) == state_completed(t+1 | HOLD)
                                       == state_completed(t+1 | BRAKE)

i.e. the completed vehicle's physical trajectory must be
action-independent once frozen. Covers both accepted action
representations (direct_accel, meta_speed)."""

from __future__ import annotations

import pytest

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig
from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec

_C = ThesisHighwayMergeEnvConfig()


def _ttc(target_speed: float, desired_x: float) -> float:
    return (_C.before_merge_length - desired_x) / target_speed


def _scenario_with_front_rear_same_lane() -> ScenarioSpec:
    """V0 (front, near the exit) and V1 (rear, active, same ramp lane) --
    V0 completes quickly under ACCELERATE, V1 stays active throughout."""
    near_exit_x = _C.before_merge_length + _C.converge_merge_length + _C.parallel_merge_length + _C.route_exit_margin
    specs = {
        "V0": VehicleSpawnSpec(vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
                                target_speed=18.0, spawn_speed=18.0, route_position=near_exit_x,
                                nominal_ttc=_ttc(18.0, near_exit_x)),
        "V1": VehicleSpawnSpec(vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
                                target_speed=18.0, spawn_speed=18.0, route_position=50.0, nominal_ttc=_ttc(18.0, 50.0)),
        "V2": VehicleSpawnSpec(vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
                                target_speed=22.0, spawn_speed=22.0, route_position=100.0, nominal_ttc=_ttc(22.0, 100.0)),
        "V3": VehicleSpawnSpec(vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
                                target_speed=22.0, spawn_speed=22.0, route_position=10.0, nominal_ttc=_ttc(22.0, 10.0)),
    }
    return ScenarioSpec(scenario_id="m4l_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)


def _complete_v0_then_apply(env, post_completion_action: int, n_post_steps: int = 10):
    """Runs V0 to completion under ACCELERATE (identical across calls),
    then applies ``post_completion_action`` to V0 for ``n_post_steps``
    more steps (V1/V2/V3 always HOLD throughout, identical across calls).
    Returns V0's (speed, x) trajectory over those post-completion steps."""
    scenario = _scenario_with_front_rear_same_lane()
    obs, _info = env.reset(seed=0, scenario=scenario)
    for _ in range(3):
        env.step({vid: 1 for vid in env.active_vehicle_ids})  # ACCELERATE everyone -> V0 completes
    assert env._env._completed["V0"] is True  # noqa: SLF001

    trajectory = []
    for _ in range(n_post_steps):
        actions = {vid: 0 for vid in env.active_vehicle_ids}  # HOLD for V1/V2/V3
        actions["V0"] = post_completion_action
        env.step(actions)
        v0 = env._env._vehicle_by_id["V0"]  # noqa: SLF001
        trajectory.append((round(float(v0.speed), 9), round(float(v0.position[0]), 9)))
    return trajectory


@pytest.mark.parametrize("action_representation", ["direct_accel", "meta_speed"])
def test_completed_vehicle_trajectory_is_action_independent(action_representation):
    """Tests A (BRAKE), B (ACCELERATE), C (HOLD) combined: the completed
    vehicle's resulting trajectory must be IDENTICAL regardless of which
    of the three actions it keeps receiving after completion."""
    cfg = ThesisHighwayMergeEnvConfig(action_representation=action_representation)

    env_brake = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))
    traj_brake = _complete_v0_then_apply(env_brake, post_completion_action=2)  # Test A

    env_accel = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))
    traj_accel = _complete_v0_then_apply(env_accel, post_completion_action=1)  # Test B

    env_hold = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))
    traj_hold = _complete_v0_then_apply(env_hold, post_completion_action=0)  # Test C

    assert traj_brake == traj_accel == traj_hold, (
        f"completed vehicle's trajectory depended on the action it kept receiving "
        f"({action_representation}): BRAKE={traj_brake} ACCELERATE={traj_accel} HOLD={traj_hold}"
    )
    # Also confirm it isn't trivially "all zeros" / stuck at spawn (i.e. the
    # test is actually exercising real post-completion motion, not a no-op).
    assert traj_brake[0] != traj_brake[-1] or traj_brake[0][0] > 0


@pytest.mark.parametrize("action_representation", ["direct_accel", "meta_speed"])
def test_frozen_flag_set_immediately_on_completion(action_representation):
    cfg = ThesisHighwayMergeEnvConfig(action_representation=action_representation)
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))
    scenario = _scenario_with_front_rear_same_lane()
    env.reset(seed=0, scenario=scenario)
    assert env._env._vehicle_by_id["V0"].frozen is False  # noqa: SLF001
    for _ in range(3):
        env.step({vid: 1 for vid in env.active_vehicle_ids})
    assert env._env._completed["V0"] is True  # noqa: SLF001
    assert env._env._vehicle_by_id["V0"].frozen is True  # noqa: SLF001
    # Still-active vehicles must remain unfrozen.
    for vid in ("V1", "V2", "V3"):
        assert env._env._vehicle_by_id[vid].frozen is False  # noqa: SLF001


def test_no_spurious_collision_from_completed_vehicle_braking_into_trailing_active_vehicle():
    """Before the fix: a completed front vehicle that kept receiving BRAKE
    could be driven down to near-stationary just past the exit, and a
    trailing active vehicle in the SAME lane closing at normal speed
    could then register a collision purely because the completed vehicle
    was still responding to actions. After the fix, the completed
    vehicle's speed is frozen at its completion-time value (direct_accel)
    or converges to its already-set target_speed (meta_speed) regardless
    of what action index it's assigned, so it should not be driven to a
    near-stop by post-completion BRAKE commands."""
    cfg = ThesisHighwayMergeEnvConfig(action_representation="direct_accel")
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))
    scenario = _scenario_with_front_rear_same_lane()
    env.reset(seed=0, scenario=scenario)
    for _ in range(3):
        env.step({vid: 1 for vid in env.active_vehicle_ids})
    assert env._env._completed["V0"] is True  # noqa: SLF001
    speed_at_completion = env._env._vehicle_by_id["V0"].speed  # noqa: SLF001

    for _ in range(15):
        actions = {vid: 0 for vid in env.active_vehicle_ids}
        actions["V0"] = 2  # BRAKE, repeatedly, on the already-completed vehicle
        env.step(actions)

    assert env._env._vehicle_by_id["V0"].speed == pytest.approx(speed_at_completion, abs=1e-6)  # noqa: SLF001
