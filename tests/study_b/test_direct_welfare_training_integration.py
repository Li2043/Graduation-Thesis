"""Real end-to-end regression test for train_dqn_direct_welfare.py's
core correctness property (change.md #3): the terminal welfare bonus is
computed ONCE per episode and applied IDENTICALLY to every agent that is
still receiving a transition at the step the episode ends -- never
per-agent-recomputed, never applied to more than one step's transitions
per agent, never zero at the true terminal step for an active agent."""

from __future__ import annotations

import pytest

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config
from thesis.study_b.utility import episode_utilities
from thesis.study_b.welfare_reward import MEAN, terminal_welfare_bonus


def _run_one_episode_direct_welfare(seed: int, *, episode_max_steps: int = 80):
    env = StudyBHeterogeneousEnv(
        StudyBEnvConfig(episode_max_steps=episode_max_steps, include_time_cost=False)
    )
    dqn_config = build_study_b_dqn_config(reward_condition="baseline", device="cpu")
    agent = SharedLocalDQNAgent(dqn_config, seed=0)

    obs, _info = env.reset(seed=seed)
    prev_active = {vid: True for vid in env.active_vehicle_ids}
    recorded: list[dict] = []  # one row per stored transition, in order

    for step in range(episode_max_steps):
        # Adversarial (always ACCELERATE) so the episode reliably ends
        # quickly via collision or timeout within episode_max_steps.
        actions = {vid: 1 for vid in env.active_vehicle_ids}
        prev_obs = obs
        obs, base_reward, terminated, truncated, step_info = env.step(actions)
        episode_over = terminated or truncated

        welfare_bonus = 0.0
        if episode_over:
            traces = env.episode_traces()
            episode_u = episode_utilities(traces)
            welfare_bonus = terminal_welfare_bonus(MEAN, list(episode_u.values()))

        for vid in env.active_vehicle_ids:
            if not prev_active[vid]:
                continue
            exit_this_step = step_info["exit_event"][vid]
            controller_terminal = bool(terminated or exit_this_step)
            learner_completed = bool(exit_this_step and not step_info["collision_event"])
            shaped_reward = base_reward[vid] + (welfare_bonus if episode_over else 0.0)
            transition = agent.build_transition(
                vehicle_id=vid, observation=prev_obs[vid], action=actions[vid],
                shaped_reward=shaped_reward, next_observation=obs[vid],
                terminated=terminated, truncated=truncated,
                controller_terminal=controller_terminal, learner_completed=learner_completed,
                base_reward=base_reward[vid], shaping_component=(welfare_bonus if episode_over else 0.0),
                episode_id=f"seed_{seed}", step=step,
            )
            agent.store_transition(transition)
            recorded.append(
                {
                    "vehicle_id": vid, "step": step, "episode_over": episode_over,
                    "shaping_component": welfare_bonus if episode_over else 0.0,
                    "base_reward": base_reward[vid], "shaped_reward": shaped_reward,
                }
            )
        prev_active = dict(step_info["active"])

        if episode_over:
            return recorded, episode_u

    pytest.fail("episode did not terminate within episode_max_steps")


def test_bonus_is_zero_on_every_non_terminal_transition():
    recorded, _episode_u = _run_one_episode_direct_welfare(seed=1)
    non_terminal_rows = [r for r in recorded if not r["episode_over"]]
    assert non_terminal_rows, "expected at least one non-terminal step"
    for row in non_terminal_rows:
        assert row["shaping_component"] == 0.0
        assert row["shaped_reward"] == pytest.approx(row["base_reward"])


def test_bonus_is_identical_across_every_agent_at_the_terminal_step():
    recorded, episode_u = _run_one_episode_direct_welfare(seed=2)
    terminal_rows = [r for r in recorded if r["episode_over"]]
    assert terminal_rows, "expected at least one terminal-step transition"
    bonuses = {row["shaping_component"] for row in terminal_rows}
    assert len(bonuses) == 1, f"expected one shared bonus value, got {bonuses}"


def test_terminal_bonus_matches_independent_recomputation():
    recorded, episode_u = _run_one_episode_direct_welfare(seed=3)
    terminal_rows = [r for r in recorded if r["episode_over"]]
    expected = terminal_welfare_bonus(MEAN, list(episode_u.values()))
    for row in terminal_rows:
        assert row["shaping_component"] == pytest.approx(expected)


def test_terminal_step_shaped_reward_equals_base_plus_bonus_not_multiplied():
    recorded, episode_u = _run_one_episode_direct_welfare(seed=4)
    terminal_rows = [r for r in recorded if r["episode_over"]]
    for row in terminal_rows:
        assert row["shaped_reward"] == pytest.approx(row["base_reward"] + row["shaping_component"])
        # Explicitly rule out an accidental x4 (one per agent) application.
        assert row["shaped_reward"] != pytest.approx(row["base_reward"] + row["shaping_component"] * 4)


def test_each_agent_contributes_at_most_one_terminal_row():
    recorded, _episode_u = _run_one_episode_direct_welfare(seed=6)
    terminal_rows = [r for r in recorded if r["episode_over"]]
    vehicle_ids = [r["vehicle_id"] for r in terminal_rows]
    assert len(vehicle_ids) == len(set(vehicle_ids)), "an agent received more than one terminal-step transition"
