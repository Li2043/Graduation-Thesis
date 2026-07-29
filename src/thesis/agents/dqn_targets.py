"""DQN Bellman targets with masked next-action max (Stage 2B-2).

    y = r + gamma * (1 - terminated) * max_{a' legal} Q_target(o', a')

External truncation does **not** suppress bootstrap.
Simultaneous terminated=True and truncated=True is rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from thesis.agents.action_masking import masked_max_q, validate_action_mask


@dataclass
class TargetBreakdown:
    reward: float
    terminated: bool
    truncated: bool
    gamma: float
    next_q_values: np.ndarray
    next_action_mask: np.ndarray
    masked_next_q_max: float
    bootstrap_multiplier: float
    target: float

    def validate_finite(self) -> None:
        if not math.isfinite(self.target):
            raise ValueError(f"non-finite target: {self.target}")
        if not math.isfinite(self.masked_next_q_max):
            raise ValueError(f"non-finite masked_next_q_max: {self.masked_next_q_max}")


def compute_dqn_target(
    reward: float,
    *,
    terminated: bool,
    truncated: bool,
    gamma: float,
    next_q_values: Sequence[float] | np.ndarray,
    next_action_mask: Sequence[bool] | np.ndarray,
) -> TargetBreakdown:
    """Compute a single DQN target with legal-action masking."""
    if bool(terminated) and bool(truncated):
        raise ValueError(
            "invalid flags: terminated=True and truncated=True simultaneously"
        )
    r = float(reward)
    if not math.isfinite(r):
        raise ValueError(f"reward must be finite, got {r}")
    g = float(gamma)
    if not math.isfinite(g) or not (0.0 <= g < 1.0):
        raise ValueError(f"gamma must satisfy 0 <= gamma < 1, got {g}")

    q = np.asarray(next_q_values, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError(f"non-finite next_q_values: {q}")
    mask = validate_action_mask(next_action_mask, int(q.shape[0]))
    q_max = masked_max_q(q, mask)
    # Bootstrap suppressed ONLY by true termination — not truncation.
    boot = 0.0 if bool(terminated) else 1.0
    target = r + g * boot * q_max
    if not math.isfinite(target):
        raise ValueError(f"non-finite target: {target}")
    bd = TargetBreakdown(
        reward=r,
        terminated=bool(terminated),
        truncated=bool(truncated),
        gamma=g,
        next_q_values=q,
        next_action_mask=mask,
        masked_next_q_max=q_max,
        bootstrap_multiplier=boot,
        target=float(target),
    )
    bd.validate_finite()
    return bd


def compute_dqn_targets_batch(
    rewards: Sequence[float] | np.ndarray,
    terminated: Sequence[bool] | np.ndarray,
    truncated: Sequence[bool] | np.ndarray,
    gamma: float,
    next_q_values: np.ndarray,
    next_action_masks: np.ndarray,
) -> list[TargetBreakdown]:
    """Batch target calculation; each row validated independently."""
    r = np.asarray(rewards, dtype=np.float64)
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    nq = np.asarray(next_q_values, dtype=np.float64)
    nm = np.asarray(next_action_masks)
    n = r.shape[0]
    if not (term.shape[0] == trunc.shape[0] == nq.shape[0] == nm.shape[0] == n):
        raise ValueError("batch dimension mismatch in compute_dqn_targets_batch")
    out: list[TargetBreakdown] = []
    for i in range(n):
        out.append(
            compute_dqn_target(
                float(r[i]),
                terminated=bool(term[i]),
                truncated=bool(trunc[i]),
                gamma=gamma,
                next_q_values=nq[i],
                next_action_mask=nm[i],
            )
        )
    return out
