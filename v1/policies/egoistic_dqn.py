"""Egoistic DQN policy (V1 control-group baseline).

A minimal, readable DQN where each agent optimises its own individual reward.
There is no fairness aggregation and no use of the experience function: the
stored reward is the agent's own scalar reward only. This is the control group
against which the Rawlsian policy is compared.
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v1.policies.base_policy import BasePolicy

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class QNetwork(nn.Module):
    """Simple multilayer perceptron mapping a state to per-action Q-values."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity experience replay buffer with a seeded sampler."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self._buffer: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done) -> None:
        self._buffer.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(bool(done)),
            )
        )

    def sample(self, batch_size: int) -> tuple:
        batch = self._rng.sample(self._buffer, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64).reshape(-1, 1)
        rewards = np.array([t.reward for t in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)


class EgoisticDQN(BasePolicy):
    """Independent DQN trained on each agent's own individual reward."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_interval: int = 1000,
        seed: int = 0,
        device: Optional[str] = None,
    ) -> None:
        self._set_seed(seed)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_capacity, seed=seed)

        self._action_rng = random.Random(seed)
        self._learn_steps = 0

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _to_tensor(self, array, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    def select_action(self, state: Any) -> int:
        with torch.no_grad():
            state_tensor = self._to_tensor(
                np.asarray(state, dtype=np.float32)
            ).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def act(self, state: Any, epsilon: float) -> int:
        if self._action_rng.random() < float(epsilon):
            return self._action_rng.randrange(self.action_dim)
        return self.select_action(state)

    def remember(self, state, action, reward, next_state, done) -> None:
        """Store an individual-reward transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, batch: Any) -> float:
        states, actions, rewards, next_states, dones = batch
        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.int64)
        rewards_t = self._to_tensor(rewards)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones)

        q_values = self.q_network(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(dim=1, keepdim=True).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def train_step(self) -> Optional[float]:
        """Sample a batch from the buffer and run one update, if possible."""
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        return self.update(batch)
