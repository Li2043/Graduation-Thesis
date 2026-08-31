"""Action masking tests for Double DQN bootstrap."""

from __future__ import annotations

import pytest
import torch

from thesis.agents.dqn_bootstrap import (
    DQNTargetMode,
    compute_bootstrap_values,
    mask_illegal_actions,
)


class ConstQ(torch.nn.Module):
    def __init__(self, row: list[float]):
        super().__init__()
        self.register_buffer("row", torch.tensor(row, dtype=torch.float32))

    def forward(self, x):
        return self.row.unsqueeze(0).expand(x.shape[0], -1)


def test_mask_affects_online_argmax_double():
    online = ConstQ([0.0, 100.0, 50.0])  # prefers 1 if legal
    target = ConstQ([7.0, 1.0, 3.0])
    obs = torch.zeros(1, 2)
    # action 1 illegal → online must pick 2
    masks = torch.tensor([[True, False, True]])
    v = compute_bootstrap_values(
        online_network=online,
        target_network=target,
        next_observations=obs,
        next_action_masks=masks,
        mode=DQNTargetMode.DOUBLE,
    )
    assert float(v.item()) == 3.0  # target Q of action 2


def test_illegal_cannot_be_gathered():
    online = ConstQ([0.0, 5.0, 1.0])
    target = ConstQ([0.0, 99.0, 1.0])
    obs = torch.zeros(1, 2)
    masks = torch.tensor([[True, False, True]])
    masked = mask_illegal_actions(
        q_values=target.row.unsqueeze(0), action_masks=masks
    )
    assert not torch.isfinite(masked[0, 1])
    v = compute_bootstrap_values(
        online_network=online,
        target_network=target,
        next_observations=obs,
        next_action_masks=masks,
        mode=DQNTargetMode.DOUBLE,
    )
    assert float(v.item()) != 99.0


def test_all_illegal_raises():
    with pytest.raises(ValueError, match="all-illegal"):
        mask_illegal_actions(
            q_values=torch.zeros(1, 3),
            action_masks=torch.zeros(1, 3, dtype=torch.bool),
        )
