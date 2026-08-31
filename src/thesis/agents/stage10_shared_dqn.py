"""Stage 10 pilot (E28) -- shared-parameter DQN architecture (pilot v5).

Replaces rounds 1-4's six independent (role, zone)-keyed Q-networks with a
SINGLE shared Q-network used by every vehicle, regardless of role, zone, or
which curriculum stage (2/4/6 vehicles) is currently active. Every vehicle's
transitions flow into ONE replay buffer and train ONE set of parameters.

Rationale (2026-08-05/06, protocol doc S0.1, this session's literature/
GitHub research): rounds 1-4's independent-sub-policy architecture was
deliberately chosen over parameter sharing to preserve testability of a
"controller identity vs situational role" research question -- but every
actually-working multi-vehicle highway-merge RL implementation found this
session (including Chen et al. 2023's own repo, `DongChen06/MARL_CAVs`,
already cited in this thesis) uses genuine parameter sharing across all
agents. Round 4's 6-vehicle stage still got ~0-5% completion across 8 seeds
even after four rounds of incremental fixes -- the user decided to test
whether the independent-sub-policy choice itself is the dominant obstacle,
by switching to the canonical Gupta, Egorov & Kochenderfer (2017) approach:
one shared policy pi_theta(a | o, m) where m is a role/context-identifying
feature appended to the observation, rather than routing between separate
networks. Role and zone -- previously encoded ONLY implicitly via which
network got called -- are now explicit input features (see
stage10_symmetric_merge_env.py's OBS_DIM bump, OBS_DIM_WITH_ROLE_ZONE); this
file no longer needs any cross-network bootstrap machinery at all: a
curriculum-stage or zone-boundary "transition" is now an entirely ordinary
transition for the one network that always was and still is in charge.

Reuses the same QNetwork / DQNConfig / ReplayBuffer / dqn_bootstrap building
blocks as independent_dqn_v2.py and subpolicy_dqn.py -- no new
hyperparameters invented here. Kept as its own file rather than modifying
independent_dqn_v2.py (whose controller_id is hardcoded to {"A","B"} for
Stage 8/9's already-validated 2-learner path) or subpolicy_dqn.py (whose
whole point was avoiding this exact architecture) in place.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from thesis.agents.action_masking import (
    masked_argmax,
    masked_random_action,
    validate_action_mask,
)
from thesis.agents.dqn_bootstrap import DQNTargetMode, compute_bootstrap_values, compute_td_targets
from thesis.agents.independent_dqn_v2 import DQNConfig, QNetwork
from thesis.agents.replay_buffer_v2 import ReplayBatch, ReplayBuffer, ReplayTransition


class SharedDQNLearner:
    """One shared Double-DQN stack used by every vehicle regardless of role/zone.

    Structurally the plain single-network case of ``SubPolicyDQNLearner``
    with the cross-network bootstrap machinery removed entirely: there is
    only ever one network, so there is no "incoming sub-policy" to borrow
    from at a zone/stage boundary -- every transition bootstraps off this
    learner's own target network, exactly like a standard single-agent
    Double DQN update (this is now structurally simpler than
    ``subpolicy_dqn.py``, not more complex).
    """

    def __init__(self, config: DQNConfig, *, seed: int, replay_seed: int | None = None):
        config.validate()
        self.config = config
        self.seed = int(seed)
        self.replay_seed = int(self.seed + 17 if replay_seed is None else replay_seed)
        self.device = torch.device(config.device)

        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.online = QNetwork(config.obs_dim, config.n_actions, config.hidden_sizes).to(self.device)
        self.target = QNetwork(config.obs_dim, config.n_actions, config.hidden_sizes).to(self.device)
        self.hard_sync_target()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(
            config.replay_capacity,
            obs_dim=config.obs_dim,
            n_actions=config.n_actions,
            seed=self.replay_seed,
        )
        self._update_count = 0

    def hard_sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def set_learning_rate(self, lr: float) -> None:
        for group in self.optimiser.param_groups:
            group["lr"] = float(lr)

    @torch.no_grad()
    def q_values(self, observation: np.ndarray, *, network: str = "online") -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (self.config.obs_dim,):
            raise ValueError(f"observation dim {obs.shape} != expected {(self.config.obs_dim,)}")
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
        eps = 0.0 if greedy else float(self.config.epsilon if epsilon is None else epsilon)
        mask = validate_action_mask(action_mask, self.config.n_actions)
        if greedy or self._rng.random() >= eps:
            q = self.q_values(observation, network="online")
            return masked_argmax(q, mask, n_actions=self.config.n_actions)
        return masked_random_action(mask, self.config.n_actions, self._rng)

    def store_transition(self, transition: ReplayTransition) -> None:
        self.replay.append(transition)

    def update(
        self, *, batch: ReplayBatch | None = None, hysteresis_ratio: float | None = None
    ) -> dict[str, Any]:
        """One optimiser step. Every transition -- regardless of which
        vehicle, role, zone, or curriculum stage produced it -- is an
        entirely ordinary Double-DQN update against this learner's own
        target network: there is no cross-network case anymore, unlike
        ``subpolicy_dqn.SubPolicyDQNLearner.update``'s version.

        ``hysteresis_ratio`` (Stage 11 v9, default ``None`` = unchanged
        behaviour for every other caller): optional hysteretic Q-learning
        loss weighting (Matignon, Laurent & Le Fort-Piat 2007; deep-RL
        extension Omidshafiei et al. 2017) -- see
        ``stage11_dyad_merge_pilot_config.py``'s ``HYSTERESIS_RATIO_V9``
        docstring for the full rationale."""
        if batch is None:
            batch = self.replay.sample(self.config.batch_size)

        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        n = len(batch.actions)
        mode = DQNTargetMode(self.config.target_mode)
        gamma = float(self.config.gamma)

        with torch.no_grad():
            boot_idx = [int(i) for i in batch.bootstrap_indices.tolist()]
            next_values_arr: list[float] = []
            if boot_idx:
                next_obs_list = []
                next_masks = []
                for i in boot_idx:
                    nobs = batch.next_observations[i]
                    nmask = batch.next_action_masks[i]
                    if nobs is None or nmask is None:
                        raise ValueError(f"bootstrap row {i} missing next_observation/next_action_mask")
                    next_obs_list.append(nobs)
                    next_masks.append(np.asarray(nmask, dtype=bool))
                next_obs_t = torch.as_tensor(np.stack(next_obs_list), dtype=torch.float32, device=self.device)
                next_mask_t = torch.as_tensor(np.stack(next_masks), dtype=torch.bool, device=self.device)
                next_values = compute_bootstrap_values(
                    online_network=self.online,
                    target_network=self.target,
                    next_observations=next_obs_t,
                    next_action_masks=next_mask_t,
                    mode=mode,
                )
                if not torch.isfinite(next_values).all():
                    raise ValueError(f"non-finite bootstrap values: {next_values}")
                next_values_arr = [float(v) for v in next_values.tolist()]

            targets_arr = compute_td_targets(
                shaped_rewards=batch.shaped_rewards,
                controller_terminal=batch.controller_terminal,
                bootstrap_indices=boot_idx,
                next_values=next_values_arr,
                gamma=gamma,
            )

        if not np.all(np.isfinite(targets_arr)):
            raise ValueError("non-finite targets in batch")
        target_t = torch.as_tensor(targets_arr, dtype=torch.float32, device=self.device)

        q_all = self.online(obs)
        if not torch.isfinite(q_all).all():
            raise ValueError("non-finite online Q-values")
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            td_error = (target_t - q_sa).detach()
        if hysteresis_ratio is None:
            loss = nn.functional.mse_loss(q_sa, target_t)
        else:
            if not (0.0 < hysteresis_ratio <= 1.0):
                raise ValueError(f"hysteresis_ratio must be in (0, 1], got {hysteresis_ratio}")
            with torch.no_grad():
                weights = torch.where(
                    td_error >= 0,
                    torch.ones_like(td_error),
                    torch.full_like(td_error, float(hysteresis_ratio)),
                )
            loss = (weights * (q_sa - target_t) ** 2).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss: {loss.item()}")

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        for p in self.online.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ValueError("non-finite gradients")
        # Diagnostic_6_DQN_Pipeline_Verification_Protocol.md sec 15 (6K):
        # total L2 gradient norm across all online params, computed BEFORE
        # optimiser.step() consumes the grads -- lets callers log whether a
        # training instability episode coincides with a gradient spike.
        grad_norm = float(
            torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in self.online.parameters() if p.grad is not None))
        )
        self.optimiser.step()
        self._update_count += 1

        return {
            "loss": float(loss.item()),
            "update_count": self._update_count,
            "td_error_mean_abs": float(td_error.abs().mean().item()),
            "td_error_max_abs": float(td_error.abs().max().item()),
            "grad_norm": grad_norm,
            "n_bootstrap_rows": int(len(boot_idx)),
            "n_terminal_rows": int(n - len(boot_idx)),
        }


__all__ = ["SharedDQNLearner"]
