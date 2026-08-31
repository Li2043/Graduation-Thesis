"""Stage 10 pilot (E28) -- role x zone independent sub-DQN architecture.

Six independent Q-networks keyed by (role, zone) in {ramp, mainline} x
{pre, merging, post}, routed by a deterministic (non-learned) lookup on each
vehicle's own directly-observable role/zone state (protocol S2.3). This is
NOT parameter sharing: each sub-policy keeps its own online/target network,
optimiser and replay buffer -- deliberately avoiding the DQN + replay
non-stationarity risk Gupta, Egorov & Kochenderfer (2017) attribute to
sharing one network's weights across simultaneously-learning agents
(protocol S4).

Cross-network bootstrap (protocol S2.4, Sutton/Precup/Singh 1999 semi-MDP
option-boundary Bellman equation): when a stored transition's next state
falls in a different (role, zone) than the one that acted, the TD target
must bootstrap using the INCOMING sub-policy's own online/target networks
(both for Double-DQN action selection and evaluation), not the acting
("outgoing") sub-policy's networks -- the outgoing sub-policy never actually
controls that next state, so evaluating it there would be wrong.

Reuses the exact QNetwork architecture / DQNConfig / ReplayBuffer /
dqn_targets / dqn_bootstrap building blocks already validated in Stage 8/9's
``independent_dqn_v2`` -- no new hyperparameters invented here.
"""

from __future__ import annotations

import copy
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
from thesis.agents.dqn_bootstrap import DQNTargetMode, compute_bootstrap_values
from thesis.agents.independent_dqn_v2 import DQNConfig, QNetwork
from thesis.agents.replay_buffer_v2 import ReplayBatch, ReplayBuffer, ReplayTransition

ROLE_VALUES: tuple[str, ...] = ("ramp", "mainline")
ZONE_VALUES: tuple[str, ...] = ("pre", "merging", "post")


def subpolicy_key(role: str, zone: str) -> str:
    if role not in ROLE_VALUES:
        raise ValueError(f"unknown role {role!r}, expected one of {ROLE_VALUES}")
    if zone not in ZONE_VALUES:
        raise ValueError(f"unknown zone {zone!r}, expected one of {ZONE_VALUES}")
    return f"{role}_{zone}"


ALL_SUBPOLICY_KEYS: tuple[str, ...] = tuple(
    subpolicy_key(role, zone) for role in ROLE_VALUES for zone in ZONE_VALUES
)


class SubPolicyDQNLearner:
    """One (role, zone) sub-policy's independent DQN stack.

    Structurally identical to ``IndependentDQNLearner`` (same QNetwork,
    ReplayBuffer, Adam optimiser, hard/soft target sync) but generalised to
    an arbitrary ``policy_id`` from ``ALL_SUBPOLICY_KEYS`` instead of a
    hardcoded {"A", "B"}, and with cross-network bootstrap support in
    ``update()``. Kept as a separate module rather than modifying
    ``independent_dqn_v2.py`` in place, to avoid touching Stage 8/9's
    already-validated 2-learner code path.
    """

    def __init__(
        self,
        policy_id: str,
        config: DQNConfig,
        *,
        seed: int,
        replay_seed: int | None = None,
    ):
        config.validate()
        if policy_id not in ALL_SUBPOLICY_KEYS:
            raise ValueError(
                f"policy_id must be one of {ALL_SUBPOLICY_KEYS}, got {policy_id!r}"
            )
        self.policy_id = policy_id
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
        """Pilot v2 only -- mirrors IndependentDQNLearner.set_learning_rate
        (Stage 8 arm2b pattern). Safe to call every step even under a
        constant LR (pilot v1): just re-assigns the same value."""
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
        """``transition.traffic_role`` carries the bootstrap-policy-id for this
        row (protocol S2.4) -- repurposed free-form field, not validated by
        ``ReplayTransition.validate()``. Empty string means "same sub-policy",
        i.e. no zone crossing on this transition."""
        transition.controller_id = self.policy_id
        self.replay.append(transition)

    def _resolve_bootstrap_learner(
        self, transition: ReplayTransition, others: dict[str, "SubPolicyDQNLearner"]
    ) -> "SubPolicyDQNLearner":
        bootstrap_id = transition.traffic_role or self.policy_id
        if bootstrap_id == self.policy_id:
            return self
        learner = others.get(bootstrap_id)
        if learner is None:
            raise ValueError(
                f"transition requests bootstrap from unknown sub-policy {bootstrap_id!r}"
            )
        return learner

    def compute_targets(
        self,
        batch: ReplayBatch,
        others: dict[str, "SubPolicyDQNLearner"],
    ) -> tuple[np.ndarray, int]:
        """Pure target computation (no gradient step) -- factored out so the
        cross-network bootstrap routing (protocol S2.4) can be unit-tested
        directly against a hand-built batch (protocol S8 item 2), instead of
        only inferred indirectly from post-``update()`` parameter drift.

        Returns ``(targets_arr, n_cross_network_rows)``.
        """
        n = len(batch.actions)
        targets_arr = np.asarray(batch.shaped_rewards, dtype=np.float64).copy()
        mode = DQNTargetMode(self.config.target_mode)
        gamma = float(self.config.gamma)
        cross_network_rows = 0

        with torch.no_grad():
            boot_idx = [int(i) for i in batch.bootstrap_indices.tolist()]
            groups: dict[str, list[int]] = {}
            for i in boot_idx:
                learner = self._resolve_bootstrap_learner(batch.transitions[i], others)
                groups.setdefault(learner.policy_id, []).append(i)
                if learner.policy_id != self.policy_id:
                    cross_network_rows += 1

            for policy_id, idxs in groups.items():
                learner = self if policy_id == self.policy_id else others[policy_id]
                next_obs_list = []
                next_masks = []
                for i in idxs:
                    nobs = batch.next_observations[i]
                    nmask = batch.next_action_masks[i]
                    if nobs is None or nmask is None:
                        raise ValueError(
                            f"bootstrap row {i} missing next_observation/next_action_mask"
                        )
                    next_obs_list.append(nobs)
                    next_masks.append(np.asarray(nmask, dtype=bool))
                next_obs_t = torch.as_tensor(np.stack(next_obs_list), dtype=torch.float32, device=self.device)
                next_mask_t = torch.as_tensor(np.stack(next_masks), dtype=torch.bool, device=self.device)
                next_values = compute_bootstrap_values(
                    online_network=learner.online,
                    target_network=learner.target,
                    next_observations=next_obs_t,
                    next_action_masks=next_mask_t,
                    mode=mode,
                )
                for j, i in enumerate(idxs):
                    if bool(batch.controller_terminal[i]):
                        raise ValueError(f"bootstrap index {i} marked controller_terminal")
                    nv = float(next_values[j].item())
                    if not math.isfinite(nv):
                        raise ValueError(f"non-finite bootstrap value at row {i}: {nv}")
                    targets_arr[i] = float(batch.shaped_rewards[i]) + gamma * nv

            for i in range(n):
                if bool(batch.controller_terminal[i]):
                    targets_arr[i] = float(batch.shaped_rewards[i])

        if not np.all(np.isfinite(targets_arr)):
            raise ValueError("non-finite targets in batch")
        return targets_arr, cross_network_rows

    def update(
        self,
        others: dict[str, "SubPolicyDQNLearner"],
        *,
        batch: ReplayBatch | None = None,
    ) -> dict[str, Any]:
        """One optimiser step on ``self.online`` only. Bootstrap rows are grouped
        by which sub-policy actually controls the next state (protocol S2.4) --
        the common case (no zone crossing) uses ``self``; boundary-crossing rows
        borrow the incoming sub-policy's frozen online/target networks (no_grad)
        to evaluate the Double-DQN bootstrap value, but only ``self.online`` ever
        receives gradients."""
        if batch is None:
            batch = self.replay.sample(self.config.batch_size)

        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)

        targets_arr, cross_network_rows = self.compute_targets(batch, others)
        boot_idx = [int(i) for i in batch.bootstrap_indices.tolist()]
        target_t = torch.as_tensor(targets_arr, dtype=torch.float32, device=self.device)

        q_all = self.online(obs)
        if not torch.isfinite(q_all).all():
            raise ValueError("non-finite online Q-values")
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(q_sa, target_t)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss: {loss.item()}")

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        for p in self.online.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ValueError("non-finite gradients")
        self.optimiser.step()
        self._update_count += 1

        return {
            "policy_id": self.policy_id,
            "loss": float(loss.item()),
            "update_count": self._update_count,
            "n_bootstrap_rows": int(len(boot_idx)),
            "n_terminal_rows": int(len(batch.actions) - len(boot_idx)),
            "n_cross_network_rows": int(cross_network_rows),
        }


def make_bootstrap_transition(
    *,
    observation: np.ndarray,
    action: int,
    shaped_reward: float,
    next_observation: np.ndarray | None,
    terminated: bool,
    truncated: bool,
    action_mask: np.ndarray,
    next_action_mask: np.ndarray | None,
    controller_terminal: bool,
    bootstrap_policy_id: str,
    learner_completed: bool = False,
) -> ReplayTransition:
    """Build a ReplayTransition carrying the Stage 10 bootstrap-routing tag
    in ``traffic_role`` (protocol S2.4). ``bootstrap_policy_id`` should be
    ``""`` when the acting sub-policy also owns the next state (no zone
    crossing), or the incoming sub-policy's key when it does."""
    return ReplayTransition(
        observation=observation,
        action=action,
        shaped_reward=shaped_reward,
        next_observation=next_observation,
        terminated=terminated,
        truncated=truncated,
        action_mask=action_mask,
        next_action_mask=next_action_mask,
        controller_terminal=controller_terminal,
        learner_completed=learner_completed,
        traffic_role=bootstrap_policy_id,
    )


class SubPolicyManager:
    """Owns all 6 independent sub-policy learners and the deterministic router."""

    def __init__(self, config: DQNConfig, *, seed: int):
        rng = np.random.default_rng(seed)
        self.learners: dict[str, SubPolicyDQNLearner] = {}
        for key in ALL_SUBPOLICY_KEYS:
            learner_seed = int(rng.integers(0, 2**31 - 1))
            self.learners[key] = SubPolicyDQNLearner(
                key, copy.deepcopy(config), seed=learner_seed
            )

    def route(self, role: str, zone: str) -> str:
        return subpolicy_key(role, zone)

    def learner_for(self, role: str, zone: str) -> SubPolicyDQNLearner:
        return self.learners[self.route(role, zone)]

    def select_action(
        self,
        role: str,
        zone: str,
        observation: np.ndarray,
        action_mask: Sequence[bool] | np.ndarray,
        *,
        epsilon: float | None = None,
        greedy: bool = False,
    ) -> int:
        return self.learner_for(role, zone).select_action(
            observation, action_mask, epsilon=epsilon, greedy=greedy
        )

    def store_transition(self, role: str, zone: str, transition: ReplayTransition) -> None:
        self.learner_for(role, zone).store_transition(transition)

    def update_all(self, *, min_buffer: int | None = None) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for key, learner in self.learners.items():
            threshold = min_buffer if min_buffer is not None else learner.config.batch_size
            if len(learner.replay) < threshold:
                continue
            others = {k: l for k, l in self.learners.items() if k != key}
            results[key] = learner.update(others)
        return results

    def hard_sync_all_targets(self) -> None:
        for learner in self.learners.values():
            learner.hard_sync_target()

    def set_learning_rate_for_all(self, lr: float) -> None:
        for learner in self.learners.values():
            learner.set_learning_rate(lr)


__all__ = [
    "ROLE_VALUES",
    "ZONE_VALUES",
    "subpolicy_key",
    "ALL_SUBPOLICY_KEYS",
    "SubPolicyDQNLearner",
    "make_bootstrap_transition",
    "SubPolicyManager",
]
