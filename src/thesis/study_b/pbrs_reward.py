"""Study B PBRS welfare-shaping reward -- new_research_plan.md's "三种训练
reward condition" section.

Base reward (progress/exit/collision/hard-braking/time-cost) is computed
entirely by the underlying environment (see ``heterogeneous_env.BASE_REWARD_KWARGS``)
and is NEVER touched here. This module adds ONLY the PBRS shaping term
``F_c(s_t, s_{t+1}) = gamma * Phi_c(s_{t+1}) - Phi_c(s_t)`` on top, for the
three formal conditions (baseline: lambda=0, i.e. this module is a no-op;
mean_pbrs: Phi = mean welfare; min_pbrs: Phi = min welfare).

Critical, easy-to-get-wrong point carried over from Study A's own design
(``thesis.pilots.stage11_welfare.stakeholder_experience``): the welfare
POTENTIAL Phi is computed from each vehicle's ABSORBING experience E_i
(pinned at 1.0 once that vehicle has completed), over ALL vehicles that
participated in the episode -- NOT the plain per-step attainment restricted
to still-active vehicles that ``utility.py``'s episode-level U_i/C_i use.
Using the wrong one here would silently make Phi drop every time a vehicle
finishes (an already-completed vehicle would look like it stopped
contributing to welfare, when it should instead be pinned at its best
possible value) -- getting this reversed would corrupt every mean_pbrs/
min_pbrs run without raising any error."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from thesis.pilots.stage11_welfare import (
    actual_potential,
    mean_welfare,
    min_welfare,
    pbrs_shaping_signal,
    stakeholder_experience,
)

__all__ = [
    "PBRS_LAMBDA",
    "GAMMA",
    "PBRSCondition",
    "BASELINE",
    "MEAN_PBRS",
    "MIN_PBRS",
    "condition_by_name",
    "experiences_from_step_info",
    "compute_potential",
    "PBRSRewardShaper",
]

# new_research_plan.md: reuses Study A's already-validated PBRS_LAMBDA_V12
# (0.2) and DQN discount GAMMA (0.995) verbatim -- both independently
# confirmed against stage11_dyad_merge_pilot_config.py during this
# package's design (PBRS_LAMBDA_V12=0.2 at line 547, GAMMA=0.995 at line
# 652). The doc is explicit: whatever algorithm/discount Study B actually
# trains with (MAPPO or the DQN fallback) MUST use this SAME gamma for its
# own returns, not just for this shaping term -- an inconsistent discount
# between the shaping potential and the learner's own optimized return
# breaks PBRS's theoretical motivation entirely.
PBRS_LAMBDA: float = 0.2
GAMMA: float = 0.995


@dataclass(frozen=True)
class PBRSCondition:
    name: str
    lam: float
    welfare_fn: Callable[[Sequence[float]], float] | None  # None <=> baseline, no potential needed


BASELINE = PBRSCondition("baseline", 0.0, None)
MEAN_PBRS = PBRSCondition("mean_pbrs", PBRS_LAMBDA, mean_welfare)
MIN_PBRS = PBRSCondition("min_pbrs", PBRS_LAMBDA, min_welfare)

_BY_NAME = {c.name: c for c in (BASELINE, MEAN_PBRS, MIN_PBRS)}


def condition_by_name(name: str) -> PBRSCondition:
    if name not in _BY_NAME:
        raise ValueError(f"unknown PBRS condition {name!r}, expected one of {sorted(_BY_NAME)}")
    return _BY_NAME[name]


def experiences_from_step_info(
    attainments: Mapping[str, float], active: Mapping[str, bool]
) -> list[float]:
    """Builds the E_i list ``compute_potential`` needs, from
    ``heterogeneous_env.py`` step/reset info's ``attainments``/``active``
    dicts (both keyed by the SAME vehicle ids, covering every vehicle that
    participated in the episode -- an already-completed vehicle stays in
    both dicts with ``active=False`` and its last real attainment value,
    which ``stakeholder_experience`` then overrides to 1.0)."""
    vehicle_ids = attainments.keys()
    return [
        stakeholder_experience(attainments[vid], completed=not active[vid]) for vid in vehicle_ids
    ]


def compute_potential(
    condition: PBRSCondition,
    experiences: Sequence[float],
    *,
    terminated: bool,
    truncated: bool,
) -> float:
    """Phi_c(s_t) -- 0.0 for baseline (no shaping) or an empty vehicle set
    (should not occur in practice, N=4 always), otherwise the condition's
    welfare aggregate over ``experiences``, zeroed only at a TRUE terminal
    state (never at truncation) via ``actual_potential``."""
    if condition.welfare_fn is None or not experiences:
        raw = 0.0
    else:
        raw = condition.welfare_fn(experiences)
    return actual_potential(raw, terminated=terminated, truncated=truncated)


class PBRSRewardShaper:
    """Stateful per-episode driver: call ``reset()`` once at episode start
    with the initial experiences, then ``step()`` once per environment
    step with the POST-step experiences -- returns the scalar shaping
    increment to add to that step's reward (0.0 for every step under
    ``BASELINE``, so callers can use this uniformly across all three
    conditions without branching)."""

    def __init__(self, condition: PBRSCondition, *, gamma: float = GAMMA):
        self.condition = condition
        self.gamma = gamma
        self._phi_prev: float | None = None

    def reset(self, *, experiences: Sequence[float]) -> None:
        self._phi_prev = compute_potential(self.condition, experiences, terminated=False, truncated=False)

    def step(
        self, *, experiences_next: Sequence[float], terminated: bool, truncated: bool
    ) -> float:
        if self._phi_prev is None:
            raise RuntimeError("PBRSRewardShaper.step() called before reset()")
        if self.condition.lam == 0.0:
            # Still advance phi_prev for bookkeeping consistency, but skip
            # the (zero-valued) arithmetic -- baseline runs pay no cost for
            # this shaper's presence.
            self._phi_prev = compute_potential(
                self.condition, experiences_next, terminated=terminated, truncated=truncated
            )
            return 0.0
        phi_next = compute_potential(
            self.condition, experiences_next, terminated=terminated, truncated=truncated
        )
        shaping = float(self.condition.lam * pbrs_shaping_signal(self._phi_prev, phi_next, gamma=self.gamma))
        self._phi_prev = phi_next
        return shaping

    def apply_per_vehicle(self, base_reward: Mapping[str, float], shaping: float) -> dict[str, float]:
        """DQN-fallback semantics (matches Study A's own established
        convention): every vehicle keeps its OWN base reward, with the SAME
        state-level shaping increment added to each."""
        return {vid: float(r) + shaping for vid, r in base_reward.items()}

    def apply_team(self, base_reward: Mapping[str, float], shaping: float) -> float:
        """MAPPO semantics (new_research_plan.md's explicit
        ``r_t^{team,c} = mean_i(r_i^base) + lambda F_c`` formula): a single
        scalar reward broadcast to every agent."""
        return float(np.mean(list(base_reward.values()))) + shaping
