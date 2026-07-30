"""Action masking for Independent DQN (selection + Bellman target).

Masks are role/state based, never controller-identity based.

Stage 2B-2R: strict mask validation — no silent float coercion via ``astype(bool)``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_action_mask(mask: Sequence | np.ndarray, n_actions: int) -> np.ndarray:
    """Return a canonical Boolean mask of length ``n_actions``.

    Accepted input dtypes (before conversion):
    - Boolean dtype;
    - Integer dtype whose values are exactly ``{0, 1}`` only.

    Rejected (no silent coercion):
    - floating dtypes (including ``0.0`` / ``1.0``);
    - values other than exact 0/1 for integers;
    - negatives, NaN, infinity;
    - strings / object arrays;
    - scalars, wrong length, multidimensional;
    - all-illegal (no legal action).
    """
    if np.isscalar(mask) or isinstance(mask, (str, bytes)):
        raise ValueError(f"action mask must be a 1-D sequence, got scalar/type {type(mask)!r}")

    arr = np.asarray(mask)
    if arr.dtype == object or arr.dtype.kind in {"U", "S", "O"}:
        raise ValueError(f"action mask dtype {arr.dtype} is not allowed (object/string)")
    if arr.ndim != 1:
        raise ValueError(f"action mask must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != n_actions:
        raise ValueError(
            f"action mask length {arr.shape[0]} != action-space size {n_actions}"
        )
    if arr.size == 0:
        raise ValueError("action mask is empty; at least one legal action required")

    # Strict dtype gate — never call astype(bool) before this check.
    if np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            f"float action masks are rejected (got dtype={arr.dtype}); "
            "use bool or integer 0/1"
        )
    if np.issubdtype(arr.dtype, np.bool_):
        values = arr
    elif np.issubdtype(arr.dtype, np.integer):
        # Reject any value outside {0, 1} before bool conversion.
        as_int = arr.astype(np.int64, copy=False)
        if np.any(as_int < 0):
            raise ValueError(f"action mask contains negative values: {as_int}")
        if not np.all((as_int == 0) | (as_int == 1)):
            raise ValueError(
                f"integer action mask must contain only exact 0/1, got {as_int}"
            )
        values = as_int
    else:
        raise ValueError(f"unsupported action mask dtype {arr.dtype}")

    if not np.all(np.isfinite(values.astype(np.float64))):
        raise ValueError(f"action mask contains non-finite values: {values}")

    bool_mask = np.asarray(values, dtype=bool)
    if not np.any(bool_mask):
        raise ValueError("action mask is all-False; at least one legal action required")
    return bool_mask


def legal_action_indices(mask: Sequence | np.ndarray, n_actions: int) -> np.ndarray:
    m = validate_action_mask(mask, n_actions)
    return np.flatnonzero(m)


def masked_argmax(
    q_values: Sequence[float] | np.ndarray,
    mask: Sequence | np.ndarray,
    *,
    n_actions: int | None = None,
) -> int:
    """Greedy action = argmax over legal actions only.

    Ties among legal actions: smallest index wins (deterministic).
    Illegal Q-values never affect the result.
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
    return int(legal[int(np.argmax(legal_q))])


def masked_random_action(
    mask: Sequence | np.ndarray,
    n_actions: int,
    rng: np.random.Generator,
) -> int:
    """Sample uniformly from legal actions only (illegal prob = 0)."""
    legal = legal_action_indices(mask, n_actions)
    idx = int(rng.integers(0, len(legal)))
    return int(legal[idx])


def masked_max_q(
    q_values: Sequence[float] | np.ndarray,
    mask: Sequence | np.ndarray,
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
