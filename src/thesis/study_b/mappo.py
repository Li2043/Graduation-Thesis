"""Study B parameter-shared MAPPO -- new_research_plan.md's "算法、开源路线与
实现参数" section.

Deliberately a small, self-contained PyTorch implementation rather than a
vendored copy of the official ``marlbenchmark/on-policy`` repository: that
repo's own config/vec-env/logging machinery is built around its own
benchmark environments, and adapting it to this project's custom Dict-keyed
env would itself be a substantial, higher-risk integration project (exactly
the kind of risk new_research_plan.md's 48-hour integration gate exists to
catch). This module instead implements the same ALGORITHM the official repo
and its NeurIPS 2022 paper describe (shared policy, centralized value
function, GAE, clipped PPO surrogate, value clipping, value normalization,
orthogonal init, entropy bonus) directly against ``StudyBHeterogeneousEnv``,
using the official repo's documented defaults as calibration reference
(see this file's hyperparameter defaults, matching new_research_plan.md's
own MAPPO table).

Team-reward semantics throughout (one scalar reward/value/advantage per
TIMESTEP, shared across all 4 agents) -- see ``pbrs_reward.PBRSRewardShaper.apply_team``
and ``rollout_buffer.py``.

Known deliberate simplification: an already-completed vehicle still gets an
action selected and its (obs, action, log_prob) still enters the PPO batch
every remaining step of the episode (the underlying env requires an action
for every vehicle id regardless of completion status, and its physical
effect is a no-op for that vehicle -- see ``stage10_symmetric_merge_env.py``'s
own ``step()``). This adds mild, likely-harmless noise to the actor's
gradient rather than corrupting it (the team reward/advantage those samples
are attached to still correctly reflects the OTHER, still-active vehicles'
outcomes) -- masking out post-completion agent-steps from the policy loss
is a reasonable follow-up refinement if later diagnostics show it matters,
not something Phase 0-3 need to block on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from thesis.study_b.rollout_buffer import RolloutBuffer

__all__ = ["MAPPOConfig", "RunningNormalizer", "ActorNetwork", "CriticNetwork", "MAPPOLearner"]


@dataclass
class MAPPOConfig:
    obs_dim: int
    global_state_dim: int
    n_actions: int = 3
    hidden_sizes: tuple[int, int] = (128, 128)
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.995  # MUST match pbrs_reward.GAMMA -- see that module's docstring
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.10
    ppo_epochs: int = 5
    minibatches: int = 1
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    value_normalization: bool = True
    device: str = "auto"  # "auto" | "cpu" | "cuda"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _orthogonal_init(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def _make_mlp(input_dim: int, hidden_sizes: tuple[int, int], output_dim: int, *, output_gain: float) -> nn.Sequential:
    h1, h2 = hidden_sizes
    l1, l2, l3 = nn.Linear(input_dim, h1), nn.Linear(h1, h2), nn.Linear(h2, output_dim)
    _orthogonal_init(l1, gain=math.sqrt(2))
    _orthogonal_init(l2, gain=math.sqrt(2))
    _orthogonal_init(l3, gain=output_gain)
    return nn.Sequential(l1, nn.Tanh(), l2, nn.Tanh(), l3)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: tuple[int, int]):
        super().__init__()
        # Small output gain (0.01, standard PPO practice) -- near-uniform
        # initial action distribution rather than an accidentally
        # near-deterministic one.
        self.net = _make_mlp(obs_dim, hidden_sizes, n_actions, output_gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNetwork(nn.Module):
    def __init__(self, global_state_dim: int, hidden_sizes: tuple[int, int]):
        super().__init__()
        self.net = _make_mlp(global_state_dim, hidden_sizes, 1, output_gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


class RunningNormalizer:
    """Welford's online mean/variance -- MAPPO's "value normalization"
    (new_research_plan.md's table, ``On``): the critic is trained to
    predict NORMALIZED returns, and ``denormalize`` converts a raw
    prediction back to reward-scale for bootstrapping/logging. Starts as
    the identity transform (mean=0, var=1) until enough data has been
    seen, so early-training bootstrapped values are not corrupted by a
    near-zero initial variance estimate."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        batch_count = len(values)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, values: np.ndarray | float) -> np.ndarray | float:
        return (values - self.mean) / math.sqrt(self.var + 1e-8)

    def denormalize(self, values: np.ndarray | float) -> np.ndarray | float:
        return values * math.sqrt(self.var + 1e-8) + self.mean


class MAPPOLearner:
    def __init__(self, config: MAPPOConfig, *, seed: int):
        self.config = config
        self.device = _resolve_device(config.device)
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.actor = ActorNetwork(config.obs_dim, config.n_actions, config.hidden_sizes).to(self.device)
        self.critic = CriticNetwork(config.global_state_dim, config.hidden_sizes).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr, eps=1e-5)
        self.value_normalizer = RunningNormalizer() if config.value_normalization else None

    # ------------------------------------------------------------ acting
    def select_actions(
        self, obs: dict[str, np.ndarray], *, deterministic: bool = False
    ) -> tuple[dict[str, int], dict[str, float]]:
        agent_ids = list(obs.keys())
        obs_tensor = torch.as_tensor(
            np.stack([obs[a] for a in agent_ids]), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            dist = Categorical(logits=logits)
            if deterministic:
                actions_t = torch.argmax(logits, dim=-1)
            else:
                actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t)
        actions = {a: int(actions_t[i].item()) for i, a in enumerate(agent_ids)}
        log_probs = {a: float(log_probs_t[i].item()) for i, a in enumerate(agent_ids)}
        return actions, log_probs

    def compute_value(self, global_state: np.ndarray) -> float:
        with torch.no_grad():
            state_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            value = self.critic(state_t).item()
        if self.value_normalizer is not None:
            value = float(self.value_normalizer.denormalize(value))
        return float(value)

    def _value_raw(self, global_state: np.ndarray) -> float:
        """Value used for GAE bootstrapping during rollout collection --
        SAME as ``compute_value`` (denormalized/reward-scale), kept as a
        separate name only for call-site clarity in training scripts."""
        return self.compute_value(global_state)

    # ------------------------------------------------------------ update
    def update(
        self, buffers: RolloutBuffer | Sequence[tuple[RolloutBuffer, float]], *, last_value: float | None = None
    ) -> dict[str, float]:
        """``buffers``: either a single ``RolloutBuffer`` (with ``last_value``
        given separately, for a single-environment rollout) or a sequence of
        ``(buffer, last_value)`` pairs, one per PARALLEL environment stream.

        GAE is a backward recursion over ONE temporal sequence -- it MUST be
        computed separately per environment stream and only concatenated
        afterward (into one flat per-agent-step batch for the actual PPO
        gradient step), never computed over transitions from different
        environments interleaved into a single sequence (that would treat
        env B's step 0 as if it temporally followed env A's last step,
        corrupting every advantage near a stream boundary)."""
        if isinstance(buffers, RolloutBuffer):
            if last_value is None:
                raise ValueError("last_value is required when passing a single buffer")
            buffer_last_value_pairs: Sequence[tuple[RolloutBuffer, float]] = [(buffers, last_value)]
        else:
            buffer_last_value_pairs = buffers

        all_advantages: list[np.ndarray] = []
        all_returns: list[np.ndarray] = []
        flat_obs, flat_actions, flat_old_log_probs = [], [], []
        flat_global_states, flat_advantages_per_state = [], []
        agent_ids: tuple[str, ...] | None = None

        for buffer, buf_last_value in buffer_last_value_pairs:
            T = len(buffer)
            if T == 0:
                continue
            if agent_ids is None:
                agent_ids = buffer.agent_ids
            advantages, returns = buffer.compute_gae(
                last_value=buf_last_value, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda
            )
            all_advantages.append(advantages)
            all_returns.append(returns)
            for t in range(T):
                for a in buffer.agent_ids:
                    flat_obs.append(buffer.obs[t][a])
                    flat_actions.append(buffer.actions[t][a])
                    flat_old_log_probs.append(buffer.log_probs[t][a])
                    flat_advantages_per_state.append(advantages[t])
                flat_global_states.append(buffer.global_states[t])

        if not flat_obs:
            raise ValueError("cannot update on empty rollout buffer(s)")
        assert agent_ids is not None

        advantages_concat = np.concatenate(all_advantages)
        returns_concat = np.concatenate(all_returns)
        # Normalize advantages GLOBALLY across all streams (standard PPO
        # practice) -- must happen AFTER concatenation, not per-stream,
        # or streams with genuinely different reward scales would be
        # miscalibrated relative to each other.
        adv_mean, adv_std = float(advantages_concat.mean()), float(advantages_concat.std())
        flat_advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in flat_advantages_per_state]

        if self.value_normalizer is not None:
            self.value_normalizer.update(returns_concat)
            value_targets_concat = self.value_normalizer.normalize(returns_concat)
        else:
            value_targets_concat = returns_concat
        flat_value_targets = list(value_targets_concat)

        obs_t = torch.as_tensor(np.stack(flat_obs), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(flat_actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(flat_old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(flat_advantages, dtype=torch.float32, device=self.device)
        global_states_t = torch.as_tensor(np.stack(flat_global_states), dtype=torch.float32, device=self.device)
        value_targets_t = torch.as_tensor(flat_value_targets, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            old_values_t = self.critic(global_states_t)

        n_agent_steps = obs_t.shape[0]
        n_states = global_states_t.shape[0]
        metrics: dict[str, float] = {}
        for _epoch in range(self.config.ppo_epochs):
            # minibatches=1 (new_research_plan.md's pilot default) -> one
            # full-batch gradient step per epoch; kept general for a
            # straightforward future increase if a larger rollout size
            # needs it.
            agent_step_indices = np.array_split(self._rng.permutation(n_agent_steps), self.config.minibatches)
            state_indices = np.array_split(self._rng.permutation(n_states), self.config.minibatches)
            for a_idx, s_idx in zip(agent_step_indices, state_indices):
                a_idx_t = torch.as_tensor(a_idx, dtype=torch.long, device=self.device)
                s_idx_t = torch.as_tensor(s_idx, dtype=torch.long, device=self.device)

                logits = self.actor(obs_t[a_idx_t])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[a_idx_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[a_idx_t])
                adv = advantages_t[a_idx_t]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - self.config.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                new_values = self.critic(global_states_t[s_idx_t])
                old_values_clip = old_values_t[s_idx_t]
                clipped_values = old_values_clip + torch.clamp(
                    new_values - old_values_clip, -self.config.clip_epsilon, self.config.clip_epsilon
                )
                targets = value_targets_t[s_idx_t]
                value_loss_unclipped = (new_values - targets) ** 2
                value_loss_clipped = (clipped_values - targets) ** 2
                value_loss = self.config.value_coef * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = float((old_log_probs_t[a_idx_t] - new_log_probs).mean().item())
                    clip_fraction = float(((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item())
                metrics = {
                    "actor_loss": float(policy_loss.item()),
                    "critic_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                }
        return metrics

    # -------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_normalizer": (
                {"mean": self.value_normalizer.mean, "var": self.value_normalizer.var, "count": self.value_normalizer.count}
                if self.value_normalizer is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if state.get("value_normalizer") is not None and self.value_normalizer is not None:
            self.value_normalizer.mean = state["value_normalizer"]["mean"]
            self.value_normalizer.var = state["value_normalizer"]["var"]
            self.value_normalizer.count = state["value_normalizer"]["count"]
