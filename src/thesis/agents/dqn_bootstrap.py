"""Bellman bootstrap value computation: Vanilla vs Double DQN."""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
import torch


class DQNTargetMode(str, Enum):
    VANILLA = "vanilla_dqn"
    DOUBLE = "double_dqn"


def mask_illegal_actions(
    *,
    q_values: torch.Tensor,
    action_masks: torch.Tensor,
) -> torch.Tensor:
    """Set illegal actions to -inf. ``action_masks`` is bool [B, A]."""
    if q_values.shape != action_masks.shape:
        raise ValueError(
            f"q_values shape {tuple(q_values.shape)} != masks {tuple(action_masks.shape)}"
        )
    if action_masks.dtype != torch.bool:
        raise ValueError("action_masks must be bool")
    legal_counts = action_masks.sum(dim=1)
    if torch.any(legal_counts <= 0):
        bad = torch.nonzero(legal_counts <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"all-illegal action mask at batch rows {bad}")
    masked = q_values.masked_fill(~action_masks, -torch.inf)
    return masked


def compute_bootstrap_values(
    *,
    online_network: torch.nn.Module,
    target_network: torch.nn.Module,
    next_observations: torch.Tensor,
    next_action_masks: torch.Tensor,
    mode: DQNTargetMode | str,
) -> torch.Tensor:
    """Return next-state values for bootstrap rows only (no reward / gamma).

    Must be called under ``torch.no_grad()`` by the learner. Neither network
    receives gradients from this path.
    """
    mode = DQNTargetMode(mode)
    if next_observations.ndim != 2:
        raise ValueError("next_observations must be [B, obs_dim]")
    if next_action_masks.ndim != 2:
        raise ValueError("next_action_masks must be [B, n_actions]")
    if next_observations.shape[0] != next_action_masks.shape[0]:
        raise ValueError("batch size mismatch between obs and masks")

    for p in target_network.parameters():
        if p.requires_grad:
            # Soft check: target params should be frozen; still detach outputs
            pass

    if mode is DQNTargetMode.VANILLA:
        target_q = target_network(next_observations)
        masked_target_q = mask_illegal_actions(
            q_values=target_q, action_masks=next_action_masks
        )
        next_values = masked_target_q.max(dim=1).values
        return next_values

    if mode is DQNTargetMode.DOUBLE:
        online_q = online_network(next_observations)
        masked_online_q = mask_illegal_actions(
            q_values=online_q, action_masks=next_action_masks
        )
        selected_actions = masked_online_q.argmax(dim=1, keepdim=True)
        target_q = target_network(next_observations)
        masked_target_q = mask_illegal_actions(
            q_values=target_q, action_masks=next_action_masks
        )
        next_values = masked_target_q.gather(dim=1, index=selected_actions).squeeze(1)
        return next_values

    raise ValueError(f"unknown DQNTargetMode {mode!r}")


def compute_td_targets(
    *,
    shaped_rewards: Sequence[float],
    controller_terminal: Sequence[bool],
    bootstrap_indices: Sequence[int],
    next_values: Sequence[float],
    gamma: float,
) -> np.ndarray:
    """Combines per-row reward + (gamma * bootstrap value) into final 1-step
    TD targets, with terminal rows forced to the bare reward (no
    bootstrap) -- Diagnostic_6_DQN_Pipeline_Verification_Protocol.md secs
    6B/6C's exact required property. Extracted out of
    ``SharedDQNLearner.update()``'s previously-inlined target-computation
    loop specifically so it can be hand-verified in isolation against
    known reward/gamma/next-Q values, without needing to mock or train a
    real network.

    ``bootstrap_indices``: row indices needing a bootstrap term (i.e. NOT
    controller_terminal), in the SAME order as ``next_values`` (row
    ``bootstrap_indices[j]`` pairs with ``next_values[j]``) -- matches
    ``ReplayBatch.bootstrap_indices``'s contract.

    Raises if any ``bootstrap_indices`` row is marked ``controller_terminal``
    (a caller bug, not a data issue -- terminal rows must never be asked
    to bootstrap in the first place)."""
    targets = np.asarray(shaped_rewards, dtype=np.float64).copy()
    controller_terminal = np.asarray(controller_terminal, dtype=bool)
    for j, i in enumerate(bootstrap_indices):
        i = int(i)
        if bool(controller_terminal[i]):
            raise ValueError(f"bootstrap index {i} marked controller_terminal")
        targets[i] = float(shaped_rewards[i]) + float(gamma) * float(next_values[j])
    for i in range(len(targets)):
        if bool(controller_terminal[i]):
            targets[i] = float(shaped_rewards[i])
    return targets


__all__ = [
    "DQNTargetMode",
    "compute_bootstrap_values",
    "compute_td_targets",
    "mask_illegal_actions",
]
