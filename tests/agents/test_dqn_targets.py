"""DQN target and IndependentDQNLearner unit tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from thesis.agents.dqn_targets import compute_dqn_target, compute_dqn_targets_batch
from thesis.agents.independent_dqn_v2 import (
    DQNConfig,
    IndependentDQNLearner,
    build_independent_learners,
    select_learner_reward,
)
from thesis.agents.replay_buffer_v2 import ReplayBatch, ReplayTransition


def test_01_separate_learner_instances():
    cfg = DQNConfig()
    learners = build_independent_learners(cfg, seed_A=0, seed_B=1)
    a, b = learners["A"], learners["B"]
    assert a.online is not b.online
    assert a.target is not b.target
    assert a.optimiser is not b.optimiser
    assert a.replay is not b.replay
    assert a._rng is not b._rng
    # Change A params → B unchanged
    with torch.no_grad():
        for p in a.online.parameters():
            p.add_(1.0)
            break
    assert not np.allclose(a.parameter_vector(), b.parameter_vector())


def test_02_deterministic_initialisation():
    cfg = DQNConfig()
    a1 = IndependentDQNLearner("A", cfg, seed=7)
    a2 = IndependentDQNLearner("A", copy.deepcopy(cfg), seed=7)
    a3 = IndependentDQNLearner("A", copy.deepcopy(cfg), seed=8)
    assert np.allclose(a1.parameter_vector(), a2.parameter_vector())
    assert not np.allclose(a1.parameter_vector(), a3.parameter_vector())
    assert a1.seed == 7 and a3.seed == 8


def test_10_true_terminal_target():
    bd = compute_dqn_target(
        1.25,
        terminated=True,
        truncated=False,
        gamma=0.9,
        next_q_values=[10.0, 20.0, 30.0],
        next_action_mask=[True, True, True],
    )
    assert bd.target == pytest.approx(1.25)
    assert bd.bootstrap_multiplier == 0.0


def test_11_ordinary_bootstrap_target():
    bd = compute_dqn_target(
        1.0,
        terminated=False,
        truncated=False,
        gamma=0.9,
        next_q_values=[5.0, 1.0, 2.0],
        next_action_mask=[True, True, True],
    )
    assert bd.masked_next_q_max == pytest.approx(5.0)
    assert bd.target == pytest.approx(5.5)


def test_12_truncation_retains_bootstrap():
    bd = compute_dqn_target(
        1.0,
        terminated=False,
        truncated=True,
        gamma=0.9,
        next_q_values=[5.0, 0.0, 0.0],
        next_action_mask=[True, True, True],
    )
    assert bd.target == pytest.approx(5.5)
    assert bd.bootstrap_multiplier == 1.0


def test_13_illegal_next_action_excluded():
    bd = compute_dqn_target(
        0.0,
        terminated=False,
        truncated=False,
        gamma=0.9,
        next_q_values=[2.0, 100.0, 5.0],
        next_action_mask=[True, False, True],
    )
    assert bd.masked_next_q_max == pytest.approx(5.0)
    assert bd.masked_next_q_max != pytest.approx(100.0)


def test_14_batch_target_calculation():
    outs = compute_dqn_targets_batch(
        rewards=[1.0, 1.25, 1.0],
        terminated=[False, True, False],
        truncated=[False, False, True],
        gamma=0.9,
        next_q_values=np.array(
            [[5.0, 0.0, 0.0], [9.0, 9.0, 9.0], [5.0, 0.0, 0.0]], dtype=np.float64
        ),
        next_action_masks=np.array(
            [[True, True, True], [True, True, True], [True, True, True]]
        ),
    )
    assert outs[0].target == pytest.approx(5.5)
    assert outs[1].target == pytest.approx(1.25)
    assert outs[2].target == pytest.approx(5.5)


def test_15_16_17_reward_passage():
    assert select_learner_reward(
        condition="baseline", base_reward=0.4, scaled_mean_shaping=0.1, scaled_min_shaping=0.2
    ) == (0.4, 0.0)
    r, s = select_learner_reward(
        condition="mean_pbrs", base_reward=0.4, scaled_mean_shaping=0.1, scaled_min_shaping=0.2
    )
    assert r == pytest.approx(0.5)
    assert s == pytest.approx(0.1)
    r, s = select_learner_reward(
        condition="min_pbrs", base_reward=0.4, scaled_mean_shaping=0.1, scaled_min_shaping=0.2
    )
    assert r == pytest.approx(0.6) and s == pytest.approx(0.2)


def test_18_no_base_rescaling():
    base = 0.8
    lam = 0.5
    shaped = base + lam * 0.2
    # Learner path is additive, not (1-lambda)*base
    assert shaped != pytest.approx((1 - lam) * base)
    r, _ = select_learner_reward(
        condition="mean_pbrs",
        base_reward=base,
        scaled_mean_shaping=lam * 0.2,
        scaled_min_shaping=0.0,
    )
    assert r == pytest.approx(shaped)


def test_20_target_network_no_gradients():
    learner = IndependentDQNLearner("A", DQNConfig(), seed=0)
    obs = np.zeros(4)
    q = learner.q_values(obs, network="target")
    assert np.all(np.isfinite(q))
    for p in learner.target.parameters():
        assert p.grad is None
        assert p.requires_grad is False


def test_21_one_deterministic_optimiser_update():
    cfg = DQNConfig(learning_rate=1e-1, batch_size=4, gamma=0.9)
    learner = IndependentDQNLearner("A", cfg, seed=0)
    for i in range(8):
        learner.store_transition(
            ReplayTransition(
                observation=np.zeros(4),
                action=0,
                shaped_reward=1.0,
                next_observation=np.ones(4) * 0.1,
                terminated=False,
                truncated=False,
                action_mask=np.array([True, True, True]),
                next_action_mask=np.array([True, True, True]),
                base_reward=1.0,
                controller_id="A",
                step=i,
            )
        )
    q_before = float(learner.q_values(np.zeros(4))[0])
    target_vec_before = learner.parameter_vector(network="target").copy()
    batch = learner.replay.sample(4)
    stats = learner.update(batch)
    assert np.isfinite(stats["loss"])
    assert stats["online_param_changed"] is True
    assert stats["target_unchanged"] is True
    assert np.allclose(learner.parameter_vector(network="target"), target_vec_before)
    q_after = float(learner.q_values(np.zeros(4))[0])
    assert np.isfinite(q_before) and np.isfinite(q_after)


def test_22_target_network_synchronisation():
    learner = IndependentDQNLearner(
        "A", DQNConfig(learning_rate=1e-1, batch_size=4), seed=0
    )
    for i in range(8):
        learner.store_transition(
            ReplayTransition(
                observation=np.random.default_rng(i).normal(size=4),
                action=0,
                shaped_reward=0.5,
                next_observation=np.random.default_rng(i + 10).normal(size=4),
                terminated=False,
                truncated=False,
                action_mask=np.array([True, True, True]),
                next_action_mask=np.array([True, True, True]),
                controller_id="A",
                step=i,
            )
        )
    learner.update(learner.replay.sample(4))
    assert not np.allclose(
        learner.parameter_vector(network="online"),
        learner.parameter_vector(network="target"),
    )
    learner.hard_sync_target()
    assert np.allclose(
        learner.parameter_vector(network="online"),
        learner.parameter_vector(network="target"),
    )


def test_greedy_masks_illegal():
    learner = IndependentDQNLearner("A", DQNConfig(), seed=0)
    # Force online Q so illegal index 1 is largest
    with torch.no_grad():
        # Bias last linear layer
        last = list(learner.online.modules())[-1]
        if isinstance(last, torch.nn.Linear):
            last.bias.zero_()
            last.bias[1] = 50.0
            last.bias[0] = 0.0
            last.bias[2] = 1.0
    mask = np.array([True, False, True])
    a = learner.select_action(np.zeros(4), mask, greedy=True)
    assert a != 1
    assert a in (0, 2)
