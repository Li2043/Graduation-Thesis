"""M4-A (basic API) and M4-C (action semantics) gates for the HighwayEnv
backend migration -- Claude_Code_Autonomous_Experiment_Runbook_
HighwayEnv.md sec 18/20.

Interactively verified before being written up here (see
``output/autonomous_highwayenv/AUTONOMOUS_EXPERIMENT_LOG.md``'s M3 entry):
HOLD/ACCELERATE/BRAKE gave exactly +0.0/+0.4/-0.6 m/s over one
0.2s policy step from an identical 20.0 m/s starting state -- i.e. exactly
+0.0/+2.0/-3.0 m/s^2, matching the legacy simulator's accel_rate/
decel_rate exactly (PRE_MIGRATION_CONFIG.json)."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv


def test_m4a_reset_gives_four_controlled_vehicles_with_local_obs():
    env = StudyBHeterogeneousHighwayEnv()
    obs, info = env.reset(seed=1)
    assert len(env.active_vehicle_ids) == 4
    assert set(obs.keys()) == set(env.active_vehicle_ids)
    for o in obs.values():
        assert o.shape == (env.observation_dim,)
        assert np.all(np.isfinite(o))
    roles = info["roles"]
    assert sorted(roles.keys()) == ["mainline", "ramp"]
    assert len(roles["ramp"]) == 2
    assert len(roles["mainline"]) == 2


def test_m4a_reset_is_deterministic_under_identical_seed():
    env_a = StudyBHeterogeneousHighwayEnv()
    obs_a, info_a = env_a.reset(seed=7)
    env_b = StudyBHeterogeneousHighwayEnv()
    obs_b, info_b = env_b.reset(seed=7)
    assert info_a["roles"] == info_b["roles"]
    assert info_a["target_speeds"] == info_b["target_speeds"]
    for vid in env_a.active_vehicle_ids:
        np.testing.assert_allclose(obs_a[vid], obs_b[vid])


def test_m4a_step_accepts_dict_actions_and_returns_gymnasium_semantics():
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=3)
    actions = {vid: 0 for vid in env.active_vehicle_ids}
    obs, reward, terminated, truncated, info = env.step(actions)
    assert set(reward.keys()) == set(env.active_vehicle_ids)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not (terminated and truncated)


def test_m4b_episode_step_count_matches_policy_frequency_and_horizon():
    # 200 policy decisions at 5 Hz -> 40s, per runbook sec 3.2/19. Uses
    # all-BRAKE: every vehicle brakes to v_min=0 and stays parked at
    # (or near) its spawn position, which is >=15m from same-lane peers by
    # construction and far from the merge-conflict zone for cross-lane
    # pairs -- guaranteed no collision, isolating the timing/horizon
    # question from collision dynamics. (A same-lane HOLD pair CAN
    # legitimately collide even under equal target speed, because a
    # vehicle traversing the SineLane merge curve diverts part of its
    # speed into lateral motion, so a straight-lane follower closes the
    # longitudinal gap during that stretch -- confirmed interactively;
    # that is real curve geometry, not a bug, and is exactly what M4-D's
    # dedicated matched-TTC-under-real-dynamics gate must characterize,
    # not something this basic timing test should depend on.)
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=11)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < 500:
        actions = {vid: 2 for vid in env.active_vehicle_ids}
        _obs, _reward, terminated, truncated, info = env.step(actions)
        steps += 1
    assert truncated and not terminated, (
        f"expected truncation with no collision; got terminated={terminated} "
        f"after {steps} steps (info={info!r} if defined)"
    )
    assert steps == env._env.thesis_config.episode_max_steps == 200
    assert abs(steps * env.dt() - 40.0) < 1e-9


@pytest.mark.parametrize(
    "action_index,expected_accel_mps2",
    [(0, 0.0), (1, 2.0), (2, -3.0)],
)
def test_m4c_action_semantics_match_legacy_accel_decel_exactly(action_index, expected_accel_mps2):
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=42)
    vid = env.active_vehicle_ids[0]
    v0 = float(env._env._vehicle_by_id[vid].speed)
    actions = {v: 0 for v in env.active_vehicle_ids}
    actions[vid] = action_index
    env.step(actions)
    v1 = float(env._env._vehicle_by_id[vid].speed)
    measured_accel = (v1 - v0) / env.dt()
    assert measured_accel == pytest.approx(expected_accel_mps2, abs=1e-9)


def test_m4c_index_agreement_network_replay_environment_logger():
    """The action index must mean the same thing everywhere it is used:
    network output index == replay-stored action == environment input ==
    what a logger would label. This env has exactly one place actions
    enter (StudyBHeterogeneousHighwayEnv.step()'s dict), so index
    agreement reduces to: the SAME integer produces the SAME physical
    acceleration regardless of which vehicle_id or step it's applied to."""
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=99)
    pre_speeds = {vid: float(env._env._vehicle_by_id[vid].speed) for vid in env.active_vehicle_ids}
    actions = {vid: 1 for vid in env.active_vehicle_ids}  # ACCELERATE, all agents
    env.step(actions)
    for vid in env.active_vehicle_ids:
        measured = (float(env._env._vehicle_by_id[vid].speed) - pre_speeds[vid]) / env.dt()
        assert measured == pytest.approx(2.0, abs=1e-9)


def test_ramp_vehicle_automatically_follows_connected_lanes_without_lateral_action():
    """No lateral action exists in this action space (sec 3.5/11) -- a ramp
    vehicle must still traverse j->k->b purely from ThesisControlledVehicle's
    automatic follow_road()/steering_control(), confirmed by watching its
    lane_index change over a short run of ACCELERATE actions."""
    env = StudyBHeterogeneousHighwayEnv()
    _obs, info = env.reset(seed=42)
    ramp_vid = info["roles"]["ramp"][0]
    seen_lane_indices = []
    for _ in range(60):
        actions = {vid: 1 for vid in env.active_vehicle_ids}
        env.step(actions)
        li = env._env._vehicle_by_id[ramp_vid].lane_index
        if not seen_lane_indices or seen_lane_indices[-1] != li:
            seen_lane_indices.append(li)
    assert seen_lane_indices[0] == ("j", "k", 0)
    assert ("k", "b", 0) in seen_lane_indices
