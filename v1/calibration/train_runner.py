"""Quiet train+eval runner for Optuna calibration.

Uses the same training/evaluation logic as ``v1.training.train`` but skips
file I/O and console logging so calibration trials stay fast and isolated.
Does not modify env, policies, rewards, or DQN logic.
"""

from __future__ import annotations

from v1.env.highway_merge_env import HighwayMergeEnv
from v1.training.train import (
    RunConfig,
    _epsilon_for_episode,
    build_experience_function,
    build_policy,
    build_reward_function,
    evaluate,
    run_episode,
    seed_everything,
)


def run_condition(config: RunConfig) -> dict:
    """Train then evaluate one condition; return held-out eval metrics only."""
    seed_everything(config.seed)

    env = HighwayMergeEnv(max_steps=config.max_steps)
    metrics_exp_fn = build_experience_function(config)
    policy = build_policy(config, action_dim=env.action_dim, obs_dim=env.obs_dim)
    reward_fn = build_reward_function(config, env.ego_agent, metrics_exp_fn)

    for episode in range(config.episodes):
        epsilon = _epsilon_for_episode(config, episode)
        train_seed = config.seed * 100000 + episode
        run_episode(
            env,
            policy,
            reward_fn,
            metrics_exp_fn,
            seed=train_seed,
            epsilon=epsilon,
            max_steps=config.max_steps,
            learn=True,
        )

    return evaluate(env, policy, reward_fn, metrics_exp_fn, config)
