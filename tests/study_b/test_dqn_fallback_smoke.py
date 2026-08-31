from __future__ import annotations

import numpy as np
import pytest

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv
from thesis.study_b.pbrs_reward import PBRSRewardShaper, condition_by_name, experiences_from_step_info
from thesis.study_b.shared_local_dqn import (
    SharedLocalDQNAgent,
    build_study_b_dqn_config,
    epsilon_at_step_v12,
)


def test_build_study_b_dqn_config_uses_local_obs_dim():
    config = build_study_b_dqn_config(reward_condition="baseline")
    from thesis.study_b.local_observation import LOCAL_OBS_DIM

    assert config.obs_dim == LOCAL_OBS_DIM
    assert config.n_actions == 3
    assert config.reward_condition == "baseline"


def test_select_actions_returns_int_per_agent():
    config = build_study_b_dqn_config()
    agent = SharedLocalDQNAgent(config, seed=0)
    obs = {"V0": np.zeros(config.obs_dim), "V1": np.ones(config.obs_dim)}
    actions = agent.select_actions(obs, epsilon=0.5)
    assert set(actions.keys()) == set(obs.keys())
    for a in actions.values():
        assert a in (0, 1, 2)


def test_greedy_selection_is_deterministic():
    config = build_study_b_dqn_config()
    agent = SharedLocalDQNAgent(config, seed=0)
    obs = {"V0": np.array([0.1] * config.obs_dim)}
    a1 = agent.select_actions(obs, epsilon=0.0, greedy=True)
    a2 = agent.select_actions(obs, epsilon=0.0, greedy=True)
    assert a1 == a2


def test_truncated_but_still_active_vehicle_bootstraps_normally():
    """Regression test: a vehicle caught in a TRUNCATED (not terminated)
    episode without having individually exited must have
    controller_terminal=False (keeps its next_observation for normal
    bootstrapping) -- this exact combination previously raised via
    ReplayTransition.validate() when train_dqn_fallback.py incorrectly set
    controller_terminal=True for any episode-over condition including bare
    truncation."""
    config = build_study_b_dqn_config()
    agent = SharedLocalDQNAgent(config, seed=0)
    obs = np.zeros(config.obs_dim)
    transition = agent.build_transition(
        vehicle_id="V0", observation=obs, action=0, shaped_reward=0.0,
        next_observation=obs, terminated=False, truncated=True,
        controller_terminal=False, learner_completed=False, step=0,
    )
    assert transition.next_observation is not None


def test_truncated_and_controller_terminal_without_completion_is_invalid():
    """The combination train_dqn_fallback.py must never construct:
    truncated=True, controller_terminal=True, learner_completed=False,
    terminated=False -- ReplayTransition itself rejects it."""
    config = build_study_b_dqn_config()
    agent = SharedLocalDQNAgent(config, seed=0)
    obs = np.zeros(config.obs_dim)
    with pytest.raises(ValueError):
        agent.build_transition(
            vehicle_id="V0", observation=obs, action=0, shaped_reward=0.0,
            next_observation=obs, terminated=False, truncated=True,
            controller_terminal=True, learner_completed=False, step=0,
        )


def test_maybe_update_returns_none_before_warmup():
    config = build_study_b_dqn_config()
    agent = SharedLocalDQNAgent(config, seed=0)
    obs = np.zeros(config.obs_dim)
    for i in range(5):
        t = agent.build_transition(
            vehicle_id="V0", observation=obs, action=0, shaped_reward=0.0,
            next_observation=obs, terminated=False, truncated=False,
            controller_terminal=False, learner_completed=False, step=i,
        )
        agent.store_transition(t)
    assert agent.maybe_update(warmup=64) is None


def test_full_episode_produces_valid_transitions_and_eventually_updates():
    """Real integration run against StudyBHeterogeneousEnv: exercises the
    controller_terminal/learner_completed flag logic across several
    episodes and confirms ReplayTransition.validate() never rejects a row
    (would raise) and that an update eventually fires once warmup is
    crossed."""
    cond = condition_by_name("mean_pbrs")
    config = build_study_b_dqn_config(reward_condition=cond.name)
    agent = SharedLocalDQNAgent(config, seed=1)
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=100))

    n_updates = 0
    step_counter = 0
    for ep in range(10):
        seed = 5000 + ep
        obs, info = env.reset(seed=seed, traffic_type="heterogeneous")
        shaper = PBRSRewardShaper(cond)
        init_attain = {vid: 1.0 for vid in env.active_vehicle_ids}
        init_active = {vid: True for vid in env.active_vehicle_ids}
        shaper.reset(experiences=experiences_from_step_info(init_attain, init_active))
        prev_active = {vid: True for vid in env.active_vehicle_ids}

        for t in range(100):
            eps = epsilon_at_step_v12(step_counter, decay_steps=2000)
            actions = agent.select_actions(obs, epsilon=eps)
            prev_obs = obs
            obs, base_reward, terminated, truncated, step_info = env.step(actions)
            exps = experiences_from_step_info(step_info["attainments"], step_info["active"])
            shaping = shaper.step(experiences_next=exps, terminated=terminated, truncated=truncated)
            shaped = shaper.apply_per_vehicle(base_reward, shaping)

            for vid in env.active_vehicle_ids:
                if not prev_active[vid]:
                    continue
                exit_this_step = step_info["exit_event"][vid]
                # Truncation alone must NOT set controller_terminal for a
                # still-active vehicle -- see train_dqn_fallback.py's
                # identical comment; only a true terminal or this
                # vehicle's own exit ends its controller trajectory.
                controller_terminal = bool(terminated or exit_this_step)
                learner_completed = bool(exit_this_step and not step_info["collision_event"])
                transition = agent.build_transition(
                    vehicle_id=vid, observation=prev_obs[vid], action=actions[vid],
                    shaped_reward=shaped[vid], next_observation=obs[vid],
                    terminated=terminated, truncated=truncated,
                    controller_terminal=controller_terminal, learner_completed=learner_completed,
                    base_reward=base_reward[vid], shaping_component=shaping,
                    episode_id=f"seed_{seed}", step=t,
                )
                agent.store_transition(transition)  # would raise via .validate() if inconsistent

            prev_active = dict(step_info["active"])
            step_counter += 1
            if agent.maybe_update(warmup=32) is not None:
                n_updates += 1
            if terminated or truncated:
                break

    assert n_updates > 0
    assert len(agent.learner.replay) > 0
