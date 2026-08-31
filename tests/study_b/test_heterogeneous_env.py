from __future__ import annotations

import numpy as np
import pytest

from thesis.study_b.heterogeneous_env import (
    BASE_REWARD_KWARGS,
    StudyBEnvConfig,
    StudyBHeterogeneousEnv,
    base_reward_kwargs,
)
from thesis.study_b.local_observation import LOCAL_OBS_DIM
from thesis.study_b.scenario_generator import generate_scenario


def _make_env(**kwargs) -> StudyBHeterogeneousEnv:
    return StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=200, **kwargs))


def test_reset_returns_four_agents_with_correct_obs_shape():
    env = _make_env()
    obs, info = env.reset(seed=1, traffic_type="heterogeneous")
    assert set(obs.keys()) == set(env.active_vehicle_ids)
    assert len(obs) == 4
    for arr in obs.values():
        assert arr.shape == (LOCAL_OBS_DIM,)


def test_reset_applies_matched_ttc_spawn_positions():
    env = _make_env()
    obs, info = env.reset(seed=2, traffic_type="heterogeneous")
    for vid in env.active_vehicle_ids:
        veh = env._env._vehicles[vid]  # noqa: SLF001 -- white-box test
        spec = env._scenario.vehicles[vid]  # noqa: SLF001
        assert veh.route_position == pytest.approx(spec.route_position)
        assert veh.speed == pytest.approx(spec.spawn_speed)


def test_reset_homogeneous_all_vehicles_target_v_ref():
    env = _make_env()
    obs, info = env.reset(seed=3, traffic_type="homogeneous")
    assert set(info["target_speeds"].values()) == {20.0}


def test_reset_with_fixed_scenario_is_reproducible():
    role_members = {"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]}
    scenario = generate_scenario(scenario_id="fixed", episode_seed=99, role_members=role_members)
    env_a = _make_env()
    env_b = _make_env()
    obs_a, info_a = env_a.reset(seed=1, scenario=scenario)
    obs_b, info_b = env_b.reset(seed=2, scenario=scenario)  # different env seed, SAME scenario
    for vid in obs_a:
        np.testing.assert_array_equal(obs_a[vid], obs_b[vid])
    assert info_a["scenario_id"] == info_b["scenario_id"] == "fixed"


def test_step_returns_base_reward_for_every_active_vehicle():
    env = _make_env()
    env.reset(seed=4, traffic_type="heterogeneous")
    actions = {vid: 0 for vid in env.active_vehicle_ids}  # MAINTAIN
    obs, reward, terminated, truncated, info = env.step(actions)
    assert set(reward.keys()) == set(env.active_vehicle_ids)
    assert all(isinstance(r, float) for r in reward.values())
    assert "attainments" in info and "active" in info


def test_episode_traces_length_matches_steps_taken_while_active():
    env = _make_env()
    env.reset(seed=5, traffic_type="heterogeneous")
    n_steps = 10
    for _ in range(n_steps):
        actions = {vid: 0 for vid in env.active_vehicle_ids}
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break
    traces = env.episode_traces()
    for vid, trace in traces.items():
        assert len(trace.speeds) == len(trace.active_flags)
        assert len(trace.speeds) <= n_steps
        assert all(trace.active_flags)  # only active samples are ever appended


def test_full_episode_runs_to_termination_without_crashing():
    env = _make_env()
    for seed in range(5):
        env.reset(seed=seed, traffic_type="heterogeneous")
        for _ in range(200):
            actions = {vid: 1 for vid in env.active_vehicle_ids}  # ACCELERATE (adversarial no-yield)
            _, _, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                assert info["term_reason"] in ("collision", "success", "truncation")
                break
        else:
            pytest.fail(f"seed {seed} never terminated within episode_max_steps")


def test_dt_matches_underlying_env():
    env = _make_env()
    assert env.dt() == pytest.approx(0.2)


def test_base_reward_kwargs_default_preserves_time_cost():
    kwargs = base_reward_kwargs(include_time_cost=True)
    assert kwargs["time_cost_per_step"] == pytest.approx(0.0005)
    # Every other term is untouched relative to the legacy constant.
    for key in ("collision_penalty_magnitude", "ttc_penalty_weight", "exit_reward_magnitude", "hard_braking_eta"):
        assert kwargs[key] == BASE_REWARD_KWARGS[key]


def test_base_reward_kwargs_can_zero_time_cost():
    kwargs = base_reward_kwargs(include_time_cost=False)
    assert kwargs["time_cost_per_step"] == 0.0
    for key in ("collision_penalty_magnitude", "ttc_penalty_weight", "exit_reward_magnitude", "hard_braking_eta"):
        assert kwargs[key] == BASE_REWARD_KWARGS[key]


def test_studybenvconfig_default_include_time_cost_is_true():
    # Byte-identical prior behaviour for any caller that doesn't know
    # about this new field (e.g. train_dqn_fallback.py).
    config = StudyBEnvConfig()
    assert config.include_time_cost is True


def test_env_with_time_cost_disabled_still_runs_end_to_end():
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=60, include_time_cost=False))
    env.reset(seed=1, traffic_type="heterogeneous")
    for _ in range(60):
        actions = {vid: 0 for vid in env.active_vehicle_ids}
        _, reward, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break
    assert isinstance(reward, dict)
