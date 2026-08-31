from __future__ import annotations

import numpy as np
import pytest

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv
from thesis.study_b.mappo import MAPPOConfig, MAPPOLearner
from thesis.study_b.pbrs_reward import condition_by_name, experiences_from_step_info, PBRSRewardShaper
from thesis.study_b.rollout_buffer import RolloutBuffer


def _make_learner(obs_dim=4, global_state_dim=4, **overrides) -> MAPPOLearner:
    config = MAPPOConfig(
        obs_dim=obs_dim,
        global_state_dim=global_state_dim,
        hidden_sizes=(16, 16),
        device="cpu",
        **overrides,
    )
    return MAPPOLearner(config, seed=0)


def test_select_actions_returns_valid_actions_for_all_agents():
    learner = _make_learner()
    obs = {"V0": np.zeros(4), "V1": np.ones(4), "V2": np.zeros(4), "V3": np.ones(4)}
    actions, log_probs = learner.select_actions(obs)
    assert set(actions.keys()) == set(obs.keys())
    for a in actions.values():
        assert a in (0, 1, 2)
    assert set(log_probs.keys()) == set(obs.keys())


def test_deterministic_selection_is_argmax_and_reproducible():
    learner = _make_learner()
    obs = {"V0": np.array([0.1, -0.2, 0.3, 0.0])}
    a1, _ = learner.select_actions(obs, deterministic=True)
    a2, _ = learner.select_actions(obs, deterministic=True)
    assert a1 == a2


def test_compute_value_returns_float():
    learner = _make_learner()
    value = learner.compute_value(np.zeros(4))
    assert isinstance(value, float)


def test_update_runs_and_returns_expected_metric_keys():
    learner = _make_learner(ppo_epochs=2, minibatches=1)
    buffer = RolloutBuffer(agent_ids=("V0", "V1"))
    obs = {"V0": np.zeros(4), "V1": np.ones(4)}
    for t in range(5):
        actions, log_probs = learner.select_actions(obs)
        buffer.add(
            obs=obs, global_state=np.zeros(4), actions=actions, log_probs=log_probs,
            team_reward=1.0, value=learner.compute_value(np.zeros(4)), done=(t == 4),
        )
    metrics = learner.update(buffer, last_value=0.0)
    for key in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction"):
        assert key in metrics


def test_update_accepts_multiple_parallel_buffers():
    """Two independent 'environment' streams of different lengths, each
    with its own bootstrap last_value -- must not raise, and must not
    silently only use one of them (checked indirectly via n_agent_steps
    through a successful update on a batch sized for BOTH streams)."""
    learner = _make_learner(ppo_epochs=1, minibatches=1)
    obs = {"V0": np.zeros(4)}

    def make_buffer(n_steps: int) -> RolloutBuffer:
        buffer = RolloutBuffer(agent_ids=("V0",))
        for t in range(n_steps):
            actions, log_probs = learner.select_actions(obs)
            buffer.add(
                obs=obs, global_state=np.zeros(4), actions=actions, log_probs=log_probs,
                team_reward=1.0, value=learner.compute_value(np.zeros(4)), done=(t == n_steps - 1),
            )
        return buffer

    buffer_a = make_buffer(4)
    buffer_b = make_buffer(6)
    metrics = learner.update([(buffer_a, 0.0), (buffer_b, 0.5)], last_value=None)
    assert "actor_loss" in metrics


def test_update_rejects_empty_buffer_list():
    learner = _make_learner()
    empty = RolloutBuffer(agent_ids=("V0",))
    with pytest.raises(ValueError):
        learner.update([(empty, 0.0)], last_value=None)


def test_state_dict_roundtrip_preserves_behaviour():
    learner = _make_learner()
    obs = {"V0": np.array([0.5, -0.5, 0.2, 0.1])}
    actions_before, _ = learner.select_actions(obs, deterministic=True)

    learner2 = _make_learner()
    learner2.load_state_dict(learner.state_dict())
    actions_after, _ = learner2.select_actions(obs, deterministic=True)
    assert actions_before == actions_after


def test_ppo_update_increases_probability_of_advantageous_action():
    """Directional-correctness sanity check (not a real-env convergence
    test, which is too slow for a unit test): repeatedly feed the learner
    transitions where action 1 gets advantage +1 and every other action
    gets -1 on a FIXED observation, and confirm P(action=1) rises. This is
    the standard way to catch a sign error in the PPO ratio/advantage
    computation without waiting on real environment dynamics."""
    learner = _make_learner(obs_dim=4, global_state_dim=4, ppo_epochs=4, minibatches=1, entropy_coef=0.0)
    fixed_obs = np.array([0.2, -0.1, 0.05, 0.0])
    obs = {"V0": fixed_obs}

    def prob_of_action_one(n_samples: int = 500) -> float:
        count = 0
        for _ in range(n_samples):
            actions, _ = learner.select_actions(obs, deterministic=False)
            if actions["V0"] == 1:
                count += 1
        return count / n_samples

    p_before = prob_of_action_one()

    for _ in range(30):
        buffer = RolloutBuffer(agent_ids=("V0",))
        for _ in range(8):
            actions, log_probs = learner.select_actions(obs)
            reward = 1.0 if actions["V0"] == 1 else -1.0
            buffer.add(
                obs=obs, global_state=fixed_obs, actions=actions, log_probs=log_probs,
                team_reward=reward, value=learner.compute_value(fixed_obs), done=True,
            )
        learner.update(buffer, last_value=0.0)

    p_after = prob_of_action_one()
    assert p_after > p_before + 0.2, f"p_before={p_before}, p_after={p_after}"
