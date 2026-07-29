"""Action masking for Independent DQN (selection + Bellman target).

Masks are role/state based, never controller-identity based.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_action_mask(mask: Sequence[bool] | np.ndarray, n_actions: int) -> np.ndarray:
    """Return a boolean mask of length ``n_actions``.

    Raises
    ------
    ValueError
        On wrong length, non-boolean-compatible values, or all-False mask.
    """
    arr = np.asarray(mask)
    if arr.ndim != 1:
        raise ValueError(f"action mask must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != n_actions:
        raise ValueError(
            f"action mask length {arr.shape[0]} != action-space size {n_actions}"
        )
    # Force boolean; reject non 0/1 numeric ambiguity only after bool cast
    bool_mask = arr.astype(bool)
    if not np.any(bool_mask):
        raise ValueError("action mask is all-False; at least one legal action required")
    return bool_mask


def legal_action_indices(mask: Sequence[bool] | np.ndarray, n_actions: int) -> np.ndarray:
    m = validate_action_mask(mask, n_actions)
    return np.flatnonzero(m)


def masked_argmax(
    q_values: Sequence[float] | np.ndarray,
    mask: Sequence[bool] | np.ndarray,
    *,
    n_actions: int | None = None,
) -> int:
    """Greedy action = argmax over legal actions only.

    Ties among legal actions: smallest index wins (deterministic).
    """
    q = np.asarray(q_values, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"q_values must be 1-D, got shape {q.shape}")
    n = int(n_actions if n_actions is not None else q.shape[0])
    if q.shape[0] != n:
        raise ValueError(f"q_values length {q.shape[0]} != n_actions {n}")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"non-finite Q-values: {q}")
    legal = legal_action_indices(mask, n)
    legal_q = q[legal]
    # np.argmax returns first max → smallest index among ties
    return int(legal[int(np.argmax(legal_q))])


def masked_random_action(
    mask: Sequence[bool] | np.ndarray,
    n_actions: int,
    rng: np.random.Generator,
) -> int:
    """Sample uniformly from legal actions only (illegal prob = 0)."""
    legal = legal_action_indices(mask, n_actions)
    idx = int(rng.integers(0, len(legal)))
    return int(legal[idx])


def masked_max_q(
    q_values: Sequence[float] | np.ndarray,
    mask: Sequence[bool] | np.ndarray,
    *,
    n_actions: int | None = None,
) -> float:
    """max_{a' legal} Q(s', a'). Never uses illegal entries."""
    q = np.asarray(q_values, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"q_values must be 1-D, got shape {q.shape}")
    n = int(n_actions if n_actions is not None else q.shape[0])
    if q.shape[0] != n:
        raise ValueError(f"q_values length {q.shape[0]} != n_actions {n}")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"non-finite Q-values: {q}")
    legal = legal_action_indices(mask, n)
    return float(np.max(q[legal]))


def role_action_mask(role: str, n_actions: int = 3) -> np.ndarray:
    """Legal actions for Stage 2B traffic roles (no MERGE action).

    Both ``mainline`` and ``ramp`` share {MAINTAIN, ACCELERATE, DECELERATE}.
    Mask follows role, not controller identity.
    """
    if role not in {"mainline", "ramp"}:
        raise ValueError(f"unknown traffic role {role!r}")
    if n_actions != 3:
        raise ValueError(
            f"Stage 2B action space size is 3 (MAINTAIN/ACCELERATE/DECELERATE); "
            f"got n_actions={n_actions}"
        )
    return np.array([True, True, True], dtype=bool)
