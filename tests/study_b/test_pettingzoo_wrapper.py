from __future__ import annotations

from pettingzoo.test import parallel_api_test

from thesis.study_b.heterogeneous_env import StudyBEnvConfig
from thesis.study_b.pettingzoo_wrapper import StudyBParallelEnv


def test_parallel_api_compliance():
    """new_research_plan.md Phase 0 checklist: 'PettingZoo test，若使用
    wrapper' -- runs the official pettingzoo.test.parallel_api_test."""
    env = StudyBParallelEnv(StudyBEnvConfig(episode_max_steps=150))
    parallel_api_test(env, num_cycles=500)


def test_reset_and_step_shapes():
    env = StudyBParallelEnv(StudyBEnvConfig(episode_max_steps=150))
    obs, infos = env.reset(seed=1)
    assert set(env.agents) == set(obs.keys()) == {"V0", "V1", "V2", "V3"}
    actions = {vid: 0 for vid in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert set(rewards.keys()) == set(env.agents)
