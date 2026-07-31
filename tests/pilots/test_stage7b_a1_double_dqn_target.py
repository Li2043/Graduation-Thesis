"""Double DQN vs Vanilla target computation tests."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from thesis.agents.dqn_bootstrap import (
    DQNTargetMode,
    compute_bootstrap_values,
    mask_illegal_actions,
)
from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner, QNetwork
from thesis.agents.replay_buffer_v2 import ReplayBatch


class TinyNet(nn.Module):
    def __init__(self, q: torch.Tensor):
        super().__init__()
        self.register_buffer("q", q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.q.unsqueeze(0).expand(b, -1).contiguous()


def test_vanilla_matches_masked_max():
    online = TinyNet(torch.tensor([1.0, 5.0, 2.0]))
    target = TinyNet(torch.tensor([3.0, 1.0, 4.0]))
    obs = torch.zeros(2, 4)
    masks = torch.tensor([[True, True, False], [True, False, True]])
    vals = compute_bootstrap_values(
        online_network=online,
        target_network=target,
        next_observations=obs,
        next_action_masks=masks,
        mode=DQNTargetMode.VANILLA,
    )
    # row0: max(3,1)=3; row1: max(3,4)=4
    assert torch.allclose(vals, torch.tensor([3.0, 4.0]))


def test_double_uses_online_select_target_eval():
    # online prefers action 1; target ranks action 2 highest
    online = TinyNet(torch.tensor([0.0, 10.0, 1.0]))
    target = TinyNet(torch.tensor([1.0, 2.0, 9.0]))
    obs = torch.zeros(1, 4)
    masks = torch.tensor([[True, True, True]])
    vanilla = compute_bootstrap_values(
        online_network=online,
        target_network=target,
        next_observations=obs,
        next_action_masks=masks,
        mode=DQNTargetMode.VANILLA,
    )
    double = compute_bootstrap_values(
        online_network=online,
        target_network=target,
        next_observations=obs,
        next_action_masks=masks,
        mode=DQNTargetMode.DOUBLE,
    )
    assert float(vanilla.item()) == 9.0
    assert float(double.item()) == 2.0
    assert float(vanilla.item()) != float(double.item())


def test_double_no_gradient_on_online_or_target():
    online = QNetwork(4, 3, (8,))
    target = QNetwork(4, 3, (8,))
    for p in target.parameters():
        p.requires_grad_(False)
    obs = torch.randn(5, 4, requires_grad=True)
    masks = torch.ones(5, 3, dtype=torch.bool)
    with torch.no_grad():
        v = compute_bootstrap_values(
            online_network=online,
            target_network=target,
            next_observations=obs.detach(),
            next_action_masks=masks,
            mode=DQNTargetMode.DOUBLE,
        )
    assert v.requires_grad is False
    for p in online.parameters():
        assert p.grad is None
    for p in target.parameters():
        assert p.grad is None


def test_learner_vanilla_default_identical_path():
    cfg = DQNConfig(
        obs_dim=4,
        n_actions=3,
        hidden_sizes=(8, 8),
        batch_size=4,
        replay_capacity=64,
        target_mode=DQNTargetMode.VANILLA,
    )
    learner = IndependentDQNLearner("A", cfg, seed=1, replay_seed=2)
    # fill replay with bootstrap transitions
    from thesis.agents.replay_buffer_v2 import ReplayTransition

    for i in range(20):
        learner.store_transition(
            ReplayTransition(
                observation=np.zeros(4, dtype=np.float64),
                action=0,
                shaped_reward=1.0,
                next_observation=np.ones(4, dtype=np.float64),
                terminated=False,
                truncated=True,
                controller_terminal=False,
                learner_completed=False,
                action_mask=np.array([True, True, True]),
                next_action_mask=np.array([True, True, True]),
                base_reward=1.0,
                shaping_component=0.0,
                reward_condition="baseline",
                episode_id="e",
                step=i,
                controller_id="A",
                traffic_role="mainline",
            )
        )
    stats = learner.update()
    assert stats["target_mode"] == "vanilla_dqn"
    assert stats["n_bootstrap_rows"] > 0
