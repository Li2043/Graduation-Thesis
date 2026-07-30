"""Independent DQN learner (Stage 2B-2 integration tests).

TEST-ONLY MLP architecture [32, 32]. Not final experimental architecture.
Separate learner instances for A and B — no shared weights.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn

from thesis.agents.action_masking import (
    masked_argmax,
    masked_random_action,
    role_action_mask,
    validate_action_mask,
)
from thesis.agents.dqn_targets import TargetBreakdown, compute_dqn_target
from thesis.agents.replay_buffer_v2 import ReplayBuffer, ReplayBatch, ReplayTransition

RewardCondition = Literal["baseline", "mean_pbrs", "min_pbrs"]


@dataclass
class DQNConfig:
    """TEST-ONLY Independent DQN configuration for Stage 2B-2."""

    obs_dim: int = 4
    n_actions: int = 3
    hidden_sizes: tuple[int, ...] = (32, 32)  # TEST-ONLY
    learning_rate: float = 1e-3  # TEST-ONLY
    gamma: float = 0.995
    epsilon: float = 0.1  # TEST-ONLY exploration (not a tuned schedule)
    replay_capacity: int = 10_000
    batch_size: int = 32
    device: str = "cpu"
    reward_condition: RewardCondition = "baseline"

    def validate(self) -> None:
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be > 0")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be > 0")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must satisfy 0 <= gamma < 1, got {self.gamma}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.reward_condition not in {"baseline", "mean_pbrs", "min_pbrs"}:
            raise ValueError(f"unknown reward_condition {self.reward_condition!r}")


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_learner_reward(
    *,
    condition: RewardCondition,
    base_reward: float,
    scaled_mean_shaping: float,
    scaled_min_shaping: float,
) -> tuple[float, float]:
    """Return (learner_reward, shaping_component). Never rescales base."""
    b = float(base_reward)
    if not math.isfinite(b):
        raise ValueError(f"base_reward must be finite, got {b}")
    if condition == "baseline":
        return b, 0.0
    if condition == "mean_pbrs":
        s = float(scaled_mean_shaping)
        if not math.isfinite(s):
            raise ValueError(f"scaled_mean_shaping must be finite, got {s}")
        return b + s, s
    if condition == "min_pbrs":
        s = float(scaled_min_shaping)
        if not math.isfinite(s):
            raise ValueError(f"scaled_min_shaping must be finite, got {s}")
        return b + s, s
    raise ValueError(f"unknown reward_condition {condition!r}")


class IndependentDQNLearner:
    """One controller's DQN stack (online, target, optimiser, buffer, RNG)."""

    def __init__(
        self,
        controller_id: str,
        config: DQNConfig,
        *,
        seed: int,
    ):
        config.validate()
        if controller_id not in {"A", "B"}:
            raise ValueError(f"controller_id must be A or B, got {controller_id}")
        self.controller_id = controller_id
        self.config = config
        self.seed = int(seed)
        self.device = torch.device(config.device)

        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.online = QNetwork(
            config.obs_dim, config.n_actions, config.hidden_sizes
        ).to(self.device)
        self.target = QNetwork(
            config.obs_dim, config.n_actions, config.hidden_sizes
        ).to(self.device)
        self.hard_sync_target()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimiser = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.replay = ReplayBuffer(
            config.replay_capacity,
            obs_dim=config.obs_dim,
            n_actions=config.n_actions,
            seed=self.seed + 17,
        )
        self._update_count = 0

    def hard_sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def action_mask_for_role(self, role: str) -> np.ndarray:
        return role_action_mask(role, n_actions=self.config.n_actions)

    @torch.no_grad()
    def q_values(self, observation: np.ndarray, *, network: str = "online") -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (self.config.obs_dim,):
            raise ValueError(
                f"observation dim {obs.shape} != expected {(self.config.obs_dim,)}"
            )
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"non-finite observation: {obs}")
        net = self.online if network == "online" else self.target
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = net(t).squeeze(0).cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(q)):
            raise ValueError(f"non-finite Q-values: {q}")
        return q

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: Sequence[bool] | np.ndarray,
        *,
        epsilon: float | None = None,
        greedy: bool = False,
    ) -> int:
        eps = 0.0 if greedy else float(
            self.config.epsilon if epsilon is None else epsilon
        )
        mask = validate_action_mask(action_mask, self.config.n_actions)
        if greedy or self._rng.random() >= eps:
            q = self.q_values(observation, network="online")
            return masked_argmax(q, mask, n_actions=self.config.n_actions)
        return masked_random_action(mask, self.config.n_actions, self._rng)

    def store_transition(self, transition: ReplayTransition) -> None:
        if transition.controller_id and transition.controller_id != self.controller_id:
            raise ValueError(
                f"transition controller_id {transition.controller_id} != "
                f"learner {self.controller_id}"
            )
        transition.controller_id = self.controller_id
        self.replay.append(transition)

    @torch.no_grad()
    def compute_target_for_transition(
        self, transition: ReplayTransition
    ) -> TargetBreakdown:
        if bool(transition.controller_terminal):
            return compute_dqn_target(
                transition.shaped_reward,
                controller_terminal=True,
                truncated=transition.truncated,
                gamma=self.config.gamma,
                terminated=transition.terminated,
            )
        next_q = self.q_values(transition.next_observation, network="target")
        return compute_dqn_target(
            transition.shaped_reward,
            controller_terminal=False,
            truncated=transition.truncated,
            gamma=self.config.gamma,
            next_q_values=next_q,
            next_action_mask=transition.next_action_mask,
            terminated=transition.terminated,
        )

    def update(self, batch: ReplayBatch | None = None) -> dict[str, Any]:
        """One deterministic optimiser step on a batch (or sample from replay).

        Vanilla DQN: target network supplies masked max Q for bootstrap rows only.
        Controller-terminal rows set target = reward and never forward the target net.
        """
        if batch is None:
            batch = self.replay.sample(self.config.batch_size)

        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        n = len(batch.actions)
        targets_arr = np.asarray(batch.shaped_rewards, dtype=np.float64).copy()
        target_forward_calls = 0

        with torch.no_grad():
            boot_idx = [int(i) for i in batch.bootstrap_indices.tolist()]
            if boot_idx:
                next_obs_list = []
                next_masks = []
                for i in boot_idx:
                    nobs = batch.next_observations[i]
                    nmask = batch.next_action_masks[i]
                    if nobs is None or nmask is None:
                        raise ValueError(
                            f"bootstrap row {i} missing next_observation/next_action_mask"
                        )
                    next_obs_list.append(nobs)
                    next_masks.append(nmask)
                next_obs_t = torch.as_tensor(
                    np.stack(next_obs_list), dtype=torch.float32, device=self.device
                )
                next_q = self.target(next_obs_t).cpu().numpy()
                target_forward_calls = 1
                for j, i in enumerate(boot_idx):
                    bd = compute_dqn_target(
                        float(batch.shaped_rewards[i]),
                        controller_terminal=False,
                        truncated=bool(batch.truncated[i]),
                        gamma=self.config.gamma,
                        next_q_values=next_q[j],
                        next_action_mask=next_masks[j],
                        terminated=bool(batch.terminated[i]),
                    )
                    targets_arr[i] = bd.target
            # Terminal rows already hold reward in targets_arr
            for i in range(n):
                if bool(batch.controller_terminal[i]):
                    targets_arr[i] = float(batch.shaped_rewards[i])

            for p in self.target.parameters():
                if p.grad is not None:
                    raise RuntimeError("target network unexpectedly has gradients")
            target_t = torch.as_tensor(
                targets_arr, dtype=torch.float32, device=self.device
            )
            if not torch.isfinite(target_t).all():
                raise ValueError("non-finite targets in batch")

        q_all = self.online(obs)
        if not torch.isfinite(q_all).all():
            raise ValueError("non-finite online Q-values")
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(q_sa, target_t)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss: {loss.item()}")

        before_vec = torch.nn.utils.parameters_to_vector(
            self.online.parameters()
        ).detach().clone()
        target_before = {
            k: v.detach().clone() for k, v in self.target.state_dict().items()
        }

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        for p in self.online.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ValueError("non-finite gradients")
        self.optimiser.step()
        self._update_count += 1

        after_vec = torch.nn.utils.parameters_to_vector(self.online.parameters()).detach()
        online_changed = not torch.allclose(before_vec, after_vec, atol=0.0, rtol=0.0)
        target_unchanged = all(
            torch.equal(target_before[k], v)
            for k, v in self.target.state_dict().items()
        )
        for p in self.target.parameters():
            if p.grad is not None:
                raise RuntimeError("target network gained gradients during update")

        return {
            "loss": float(loss.item()),
            "online_param_changed": bool(online_changed),
            "target_unchanged": bool(target_unchanged),
            "mean_q_sa": float(q_sa.detach().mean().item()),
            "mean_target": float(target_t.mean().item()),
            "update_count": self._update_count,
            "target_network_forward_calls": int(target_forward_calls),
            "n_bootstrap_rows": int(len(boot_idx)),
            "n_terminal_rows": int(n - len(boot_idx)),
        }

    def parameter_vector(self, *, network: str = "online") -> np.ndarray:
        net = self.online if network == "online" else self.target
        return torch.nn.utils.parameters_to_vector(net.parameters()).detach().cpu().numpy()


def build_independent_learners(
    config: DQNConfig,
    *,
    seed_A: int,
    seed_B: int,
) -> dict[str, IndependentDQNLearner]:
    """Create persistent separate learners for A and B."""
    return {
        "A": IndependentDQNLearner("A", config, seed=seed_A),
        "B": IndependentDQNLearner("B", copy.deepcopy(config), seed=seed_B),
    }
