"""DQN Bellman targets with masked next-action max (Stage 2B-2 / 2B-2R).

Vanilla DQN (retained): target network provides masked max over legal next actions.

Terminal semantics (Stage 2B-2R)
--------------------------------
``controller_terminal``
    True when this learner must **not** bootstrap:
    safe exit on this transition, global collision terminal, or global success
    terminal. Max-step truncation alone is **not** controller-terminal unless
    the learner also completed.

``terminated``
    Environment-level true terminal (collision or joint success).

``truncated``
    Max-step (or other) truncation; bootstraps unless ``controller_terminal``.

``learner_completed``
    Diagnostic: this learner has exited (possibly on this transition).

Rule:
    if controller_terminal:  target = reward
    else:                    target = reward + gamma * masked_next_q

Terminal rows must not evaluate the target network and must not validate
next-state masks / observations.
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
    controller_terminal: bool
    terminated: bool
    truncated: bool
    gamma: float
    next_q_values: np.ndarray | None
    next_action_mask: np.ndarray | None
    masked_next_q_max: float
    bootstrap_multiplier: float
    target: float

    def validate_finite(self) -> None:
        if not math.isfinite(self.target):
            raise ValueError(f"non-finite target: {self.target}")
        if self.bootstrap_multiplier != 0.0 and not math.isfinite(self.masked_next_q_max):
            raise ValueError(f"non-finite masked_next_q_max: {self.masked_next_q_max}")


def compute_dqn_target(
    reward: float,
    *,
    controller_terminal: bool,
    truncated: bool,
    gamma: float,
    next_q_values: Sequence[float] | np.ndarray | None = None,
    next_action_mask: Sequence | np.ndarray | None = None,
    terminated: bool = False,
) -> TargetBreakdown:
    """Compute a single DQN target with legal-action masking.

    For ``controller_terminal=True``:
    - ``next_q_values`` / ``next_action_mask`` may be ``None`` or malformed;
    - they are **ignored** (not validated, not used);
    - ``target == reward`` exactly.
    """
    if bool(terminated) and bool(truncated):
        raise ValueError(
            "invalid flags: terminated=True and truncated=True simultaneously"
        )
    if bool(terminated) and not bool(controller_terminal):
        raise ValueError("terminated=True requires controller_terminal=True")
    r = float(reward)
    if not math.isfinite(r):
        raise ValueError(f"reward must be finite, got {r}")
    g = float(gamma)
    if not math.isfinite(g) or not (0.0 <= g < 1.0):
        raise ValueError(f"gamma must satisfy 0 <= gamma < 1, got {g}")

    if bool(controller_terminal):
        bd = TargetBreakdown(
            reward=r,
            controller_terminal=True,
            terminated=bool(terminated),
            truncated=bool(truncated),
            gamma=g,
            next_q_values=None,
            next_action_mask=None,
            masked_next_q_max=0.0,
            bootstrap_multiplier=0.0,
            target=float(r),
        )
        bd.validate_finite()
        return bd

    # Bootstrap row — next state mandatory
    if next_q_values is None:
        raise ValueError("non-terminal row requires next_q_values")
    if next_action_mask is None:
        raise ValueError("non-terminal row requires next_action_mask")

    q = np.asarray(next_q_values, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError(f"non-finite next_q_values: {q}")
    mask = validate_action_mask(next_action_mask, int(q.shape[0]))
    q_max = masked_max_q(q, mask)
    target = r + g * q_max
    if not math.isfinite(target):
        raise ValueError(f"non-finite target: {target}")
    bd = TargetBreakdown(
        reward=r,
        controller_terminal=False,
        terminated=bool(terminated),
        truncated=bool(truncated),
        gamma=g,
        next_q_values=q,
        next_action_mask=mask,
        masked_next_q_max=q_max,
        bootstrap_multiplier=1.0,
        target=float(target),
    )
    bd.validate_finite()
    return bd


def compute_dqn_targets_batch(
    rewards: Sequence[float] | np.ndarray,
    controller_terminal: Sequence[bool] | np.ndarray,
    truncated: Sequence[bool] | np.ndarray,
    gamma: float,
    next_q_values: np.ndarray | None,
    next_action_masks: np.ndarray | None,
    *,
    terminated: Sequence[bool] | np.ndarray | None = None,
    bootstrap_indices: Sequence[int] | None = None,
) -> list[TargetBreakdown]:
    """Batch targets.

    ``next_q_values`` / ``next_action_masks`` must align with bootstrap rows only
    when ``bootstrap_indices`` is provided; otherwise full-batch arrays are used
    and terminal rows ignore their next-state slices without validating them.
    """
    r = np.asarray(rewards, dtype=np.float64)
    cterm = np.asarray(controller_terminal, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    n = r.shape[0]
    if terminated is not None:
        term_arr = np.asarray(terminated, dtype=bool)
    else:
        term_arr = np.zeros(n, dtype=bool)

    if trunc.shape[0] != n or cterm.shape[0] != n or term_arr.shape[0] != n:
        raise ValueError("batch dimension mismatch in compute_dqn_targets_batch")

    bad = np.flatnonzero(term_arr & ~cterm)
    if bad.size:
        raise ValueError(
            "terminated=True requires controller_terminal=True; "
            f"offending row indices: {bad.tolist()}"
        )

    out: list[TargetBreakdown] = []
    if bootstrap_indices is not None:
        try:
            boot = [int(i) for i in bootstrap_indices]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"bootstrap_indices must be a sequence of integers: {exc}"
            ) from exc
        expected = [int(i) for i in np.flatnonzero(~cterm).tolist()]
        if len(boot) != len(set(boot)):
            raise ValueError(
                f"bootstrap_indices must be unique, got {boot}"
            )
        for idx in boot:
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"bootstrap_indices out of range: {idx} not in [0, {n})"
                )
        if set(boot) != set(expected):
            raise ValueError(
                "bootstrap_indices must equal all and only non-terminal indices; "
                f"got {sorted(boot)}, expected {expected}"
            )
        if next_q_values is None or next_action_masks is None:
            if boot:
                raise ValueError("bootstrap rows require next_q_values and masks")
        nq = None if next_q_values is None else np.asarray(next_q_values, dtype=np.float64)
        nm = None if next_action_masks is None else np.asarray(next_action_masks)
        boot_pos = {idx: j for j, idx in enumerate(boot)}
        for i in range(n):
            if bool(cterm[i]):
                out.append(
                    compute_dqn_target(
                        float(r[i]),
                        controller_terminal=True,
                        truncated=bool(trunc[i]),
                        gamma=gamma,
                        terminated=bool(term_arr[i]),
                    )
                )
            else:
                j = boot_pos.get(i)
                if j is None:
                    raise ValueError(
                        f"bootstrap_indices mapping missing non-terminal index {i}"
                    )
                out.append(
                    compute_dqn_target(
                        float(r[i]),
                        controller_terminal=False,
                        truncated=bool(trunc[i]),
                        gamma=gamma,
                        next_q_values=nq[j],
                        next_action_mask=nm[j],
                        terminated=bool(term_arr[i]),
                    )
                )
        return out

    # Legacy full-batch path: terminal rows still skip next validation
    nq = np.asarray(next_q_values) if next_q_values is not None else None
    nm = np.asarray(next_action_masks) if next_action_masks is not None else None
    for i in range(n):
        if bool(cterm[i]):
            out.append(
                compute_dqn_target(
                    float(r[i]),
                    controller_terminal=True,
                    truncated=bool(trunc[i]),
                    gamma=gamma,
                    terminated=bool(term_arr[i]),
                )
            )
        else:
            if nq is None or nm is None:
                raise ValueError("non-terminal batch rows require next_q_values/masks")
            out.append(
                compute_dqn_target(
                    float(r[i]),
                    controller_terminal=False,
                    truncated=bool(trunc[i]),
                    gamma=gamma,
                    next_q_values=nq[i],
                    next_action_mask=nm[i],
                    terminated=bool(term_arr[i]),
                )
            )
    return out
