"""Stage-2A PBRS potentials and shaping (pure functions; no environment / DQN).

Fixed stakeholder set
---------------------
    V = {A, B, B_front, B_rear}   (order preserved; denominator always 4)
    N = {A, B}                    (learning controllers)

Experience
----------
Active stakeholder:
    e_i(s) = clip(v_i(s) / v_target_i, 0, 1)
Completed (safely exited, episode continues):
    E_i(s) = 1
Completed stakeholders stay in V; mean denominator never shrinks.

Potentials (non-terminal)
-------------------------
    raw_phi_mean(s) = (1/4) * sum_i E_i(s)
    raw_phi_min(s)  = min_i E_i(s)

Actual potential
----------------
    phi_c(s) = 0           if terminated (true terminal; success or collision)
             = raw_phi_c(s) otherwise

External truncation (terminated=False, truncated=True) does **not** zero the
potential. Simultaneous terminated=True and truncated=True is rejected.

PBRS signal
-----------
    F_c,t = gamma * phi_c(s_{t+1}) - phi_c(s_t)
    r_shaped[i,t] = r_base[i,t] + lambda_c * F_c,t

Base reward is never rescaled. ``lambda_*`` values in the default config are
**TEST-ONLY placeholders**, not final experimental calibrations.

Shaping gamma must equal learner gamma within a strict tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

STAKEHOLDER_ORDER: tuple[str, ...] = ("A", "B", "B_front", "B_rear")
LEARNING_CONTROLLERS: tuple[str, ...] = ("A", "B")
PotentialCondition = Literal["mean", "min"]

GAMMA_MATCH_TOLERANCE = 1e-12


def _require_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value)!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def _validate_termination_flags(*, terminated: bool, truncated: bool) -> None:
    if bool(terminated) and bool(truncated):
        raise ValueError(
            "invalid flags: terminated=True and truncated=True simultaneously; "
            "true termination and external truncation must be represented separately"
        )


def _require_exact_stakeholder_set(mapping: Mapping[str, object], *, label: str) -> None:
    keys = set(mapping.keys())
    expected = set(STAKEHOLDER_ORDER)
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    if missing or unexpected:
        raise ValueError(
            f"{label} must contain exactly stakeholders {list(STAKEHOLDER_ORDER)}; "
            f"missing={missing}, unexpected={unexpected}"
        )


@dataclass(frozen=True)
class StakeholderState:
    """One stakeholder's state for potential evaluation.

    Provide either (speed, target_speed) for active experience, or set
    ``completed=True`` (E_i = 1). Optionally supply ``explicit_experience``
    to bypass speed-based calculation (still subject to finite checks and
    [0, 1] clipping for active stakeholders).
    """

    speed: float | None = None
    target_speed: float | None = None
    completed: bool = False
    explicit_experience: float | None = None


@dataclass(frozen=True)
class PotentialState:
    """Full four-stakeholder state plus termination flags."""

    stakeholders: Mapping[str, StakeholderState]
    terminated: bool = False
    truncated: bool = False
    terminal_label: str | None = None
    """Optional diagnostic label, e.g. 'success' or 'collision'."""


@dataclass(frozen=True)
class PBRSConfig:
    """PBRS configuration.

    ``lambda_mean`` / ``lambda_min`` are TEST-ONLY placeholders in Stage 2A.
    """

    learner_gamma: float = 0.995
    shaping_gamma: float = 0.995
    lambda_mean: float = 0.5  # TEST-ONLY
    lambda_min: float = 0.5  # TEST-ONLY
    gamma_match_tolerance: float = GAMMA_MATCH_TOLERANCE

    def validate(self) -> None:
        g_l = _require_finite("learner_gamma", self.learner_gamma)
        g_s = _require_finite("shaping_gamma", self.shaping_gamma)
        if not (0.0 <= g_l < 1.0):
            raise ValueError(f"learner_gamma must satisfy 0 <= gamma < 1, got {g_l}")
        if not (0.0 <= g_s < 1.0):
            raise ValueError(f"shaping_gamma must satisfy 0 <= gamma < 1, got {g_s}")
        if abs(g_l - g_s) > self.gamma_match_tolerance:
            raise ValueError(
                f"shaping_gamma ({g_s}) must equal learner_gamma ({g_l}) "
                f"within tolerance {self.gamma_match_tolerance}"
            )
        for name, lam in (("lambda_mean", self.lambda_mean), ("lambda_min", self.lambda_min)):
            v = _require_finite(name, lam)
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0, got {v}")

    def lambda_for(self, condition: PotentialCondition) -> float:
        if condition == "mean":
            return float(self.lambda_mean)
        if condition == "min":
            return float(self.lambda_min)
        raise ValueError(f"unknown condition {condition!r}")


@dataclass
class PotentialBreakdown:
    stakeholder_experiences: dict[str, float]
    worst_off_ids: tuple[str, ...]
    raw_potential: float
    actual_potential: float
    condition: PotentialCondition
    terminated: bool
    truncated: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class PBRSSignalBreakdown:
    condition: PotentialCondition
    phi_t: float
    phi_t1: float
    raw_phi_t: float
    raw_phi_t1: float
    gamma: float
    shaping_signal: float
    stakeholder_experiences_t: dict[str, float]
    stakeholder_experiences_t1: dict[str, float]
    worst_off_ids_t: tuple[str, ...]
    worst_off_ids_t1: tuple[str, ...]


@dataclass
class ShapedRewardBreakdown:
    controller: str
    condition: PotentialCondition
    base_reward: float
    shaping_signal: float
    scaled_shaping_component: float
    shaped_reward: float
    lambda_c: float
    gamma: float
    phi_t: float
    phi_t1: float
    raw_potential_t: float
    raw_potential_t1: float
    actual_potential_t: float
    actual_potential_t1: float
    stakeholder_experiences_t: dict[str, float]
    stakeholder_experiences_t1: dict[str, float]
    worst_off_ids_t: tuple[str, ...]
    worst_off_ids_t1: tuple[str, ...]


def compute_active_experience(speed: float, target_speed: float) -> float:
    """e_i = clip(v_i / v_target_i, 0, 1)."""
    v = _require_finite("speed", speed)
    vt = _require_finite("target_speed", target_speed)
    if not (vt > 0.0):
        raise ValueError(f"target_speed must be strictly positive, got {vt}")
    ratio = v / vt
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return float(ratio)


def compute_stakeholder_experiences(
    stakeholders: Mapping[str, StakeholderState],
) -> dict[str, float]:
    """Return E_i for exactly the fixed stakeholder set V, in STAKEHOLDER_ORDER keys."""
    _require_exact_stakeholder_set(stakeholders, label="stakeholders")
    experiences: dict[str, float] = {}
    for sid in STAKEHOLDER_ORDER:
        st = stakeholders[sid]
        if st.completed:
            experiences[sid] = 1.0
            continue
        if st.explicit_experience is not None:
            e = _require_finite(f"{sid}.explicit_experience", st.explicit_experience)
            if e < 0.0:
                e = 0.0
            elif e > 1.0:
                e = 1.0
            experiences[sid] = float(e)
            continue
        if st.speed is None or st.target_speed is None:
            raise ValueError(
                f"stakeholder {sid!r} is active but missing speed/target_speed "
                "(and no explicit_experience)"
            )
        experiences[sid] = compute_active_experience(st.speed, st.target_speed)
    return experiences


def _experiences_from_mapping(
    experiences: Mapping[str, float],
) -> dict[str, float]:
    """Validate and normalise an explicit E mapping (tests / diagnostics)."""
    _require_exact_stakeholder_set(experiences, label="experiences")
    out: dict[str, float] = {}
    for sid in STAKEHOLDER_ORDER:
        e = _require_finite(f"experience[{sid}]", experiences[sid])
        if e < 0.0 or e > 1.0:
            raise ValueError(f"experience[{sid}] must be in [0, 1], got {e}")
        out[sid] = float(e)
    return out


def _worst_off_ids(experiences: Mapping[str, float]) -> tuple[str, ...]:
    """All stakeholder IDs achieving the minimum experience (ties preserved)."""
    vals = [experiences[sid] for sid in STAKEHOLDER_ORDER]
    m = min(vals)
    return tuple(sid for sid in STAKEHOLDER_ORDER if experiences[sid] == m)


def compute_raw_mean_potential(experiences: Mapping[str, float]) -> float:
    exp = _experiences_from_mapping(experiences)
    return float(sum(exp[sid] for sid in STAKEHOLDER_ORDER) / 4.0)


def compute_raw_min_potential(experiences: Mapping[str, float]) -> float:
    exp = _experiences_from_mapping(experiences)
    return float(min(exp[sid] for sid in STAKEHOLDER_ORDER))


def compute_actual_potential(
    raw_potential: float,
    *,
    terminated: bool,
    truncated: bool,
) -> float:
    """Zero only for true terminal states; truncation preserves raw potential."""
    _validate_termination_flags(terminated=terminated, truncated=truncated)
    raw = _require_finite("raw_potential", raw_potential)
    if terminated:
        return 0.0
    return float(raw)


def compute_potential_breakdown(
    state: PotentialState,
    condition: PotentialCondition,
    *,
    experiences: Mapping[str, float] | None = None,
) -> PotentialBreakdown:
    """Compute raw/actual potential and worst-off IDs for one state."""
    _validate_termination_flags(
        terminated=state.terminated, truncated=state.truncated
    )
    if experiences is None:
        exp = compute_stakeholder_experiences(state.stakeholders)
    else:
        exp = _experiences_from_mapping(experiences)

    if condition == "mean":
        raw = compute_raw_mean_potential(exp)
    elif condition == "min":
        raw = compute_raw_min_potential(exp)
    else:
        raise ValueError(f"unknown condition {condition!r}")

    actual = compute_actual_potential(
        raw, terminated=state.terminated, truncated=state.truncated
    )
    return PotentialBreakdown(
        stakeholder_experiences=dict(exp),
        worst_off_ids=_worst_off_ids(exp),
        raw_potential=raw,
        actual_potential=actual,
        condition=condition,
        terminated=bool(state.terminated),
        truncated=bool(state.truncated),
    )


def compute_pbrs_signal(
    state_t: PotentialState,
    state_t1: PotentialState,
    condition: PotentialCondition,
    config: PBRSConfig | None = None,
    *,
    experiences_t: Mapping[str, float] | None = None,
    experiences_t1: Mapping[str, float] | None = None,
) -> PBRSSignalBreakdown:
    """F = gamma * phi(s_{t+1}) - phi(s_t) using actual potentials."""
    cfg = config or PBRSConfig()
    cfg.validate()
    gamma = float(cfg.shaping_gamma)

    bd_t = compute_potential_breakdown(
        state_t, condition, experiences=experiences_t
    )
    bd_t1 = compute_potential_breakdown(
        state_t1, condition, experiences=experiences_t1
    )
    phi_t = bd_t.actual_potential
    phi_t1 = bd_t1.actual_potential
    f_sig = gamma * phi_t1 - phi_t
    if not math.isfinite(f_sig):
        raise ValueError(f"non-finite shaping signal: {f_sig}")
    return PBRSSignalBreakdown(
        condition=condition,
        phi_t=phi_t,
        phi_t1=phi_t1,
        raw_phi_t=bd_t.raw_potential,
        raw_phi_t1=bd_t1.raw_potential,
        gamma=gamma,
        shaping_signal=float(f_sig),
        stakeholder_experiences_t=dict(bd_t.stakeholder_experiences),
        stakeholder_experiences_t1=dict(bd_t1.stakeholder_experiences),
        worst_off_ids_t=bd_t.worst_off_ids,
        worst_off_ids_t1=bd_t1.worst_off_ids,
    )


def apply_pbrs_to_base_rewards(
    base_rewards: Mapping[str, float],
    state_t: PotentialState,
    state_t1: PotentialState,
    condition: PotentialCondition,
    config: PBRSConfig | None = None,
    *,
    experiences_t: Mapping[str, float] | None = None,
    experiences_t1: Mapping[str, float] | None = None,
) -> dict[str, ShapedRewardBreakdown]:
    """r_shaped[i] = r_base[i] + lambda_c * F_c  (common F for all controllers)."""
    cfg = config or PBRSConfig()
    cfg.validate()
    for aid in LEARNING_CONTROLLERS:
        if aid not in base_rewards:
            raise ValueError(f"base_rewards missing learning controller {aid!r}")
    for aid, br in base_rewards.items():
        _require_finite(f"base_rewards[{aid}]", br)

    signal = compute_pbrs_signal(
        state_t,
        state_t1,
        condition,
        cfg,
        experiences_t=experiences_t,
        experiences_t1=experiences_t1,
    )
    lam = cfg.lambda_for(condition)
    scaled = float(lam * signal.shaping_signal)

    out: dict[str, ShapedRewardBreakdown] = {}
    for aid in LEARNING_CONTROLLERS:
        base = float(base_rewards[aid])
        shaped = base + scaled
        if not math.isfinite(shaped):
            raise ValueError(f"non-finite shaped_reward for {aid}: {shaped}")
        out[aid] = ShapedRewardBreakdown(
            controller=aid,
            condition=condition,
            base_reward=base,
            shaping_signal=signal.shaping_signal,
            scaled_shaping_component=scaled,
            shaped_reward=shaped,
            lambda_c=lam,
            gamma=signal.gamma,
            phi_t=signal.phi_t,
            phi_t1=signal.phi_t1,
            raw_potential_t=signal.raw_phi_t,
            raw_potential_t1=signal.raw_phi_t1,
            actual_potential_t=signal.phi_t,
            actual_potential_t1=signal.phi_t1,
            stakeholder_experiences_t=dict(signal.stakeholder_experiences_t),
            stakeholder_experiences_t1=dict(signal.stakeholder_experiences_t1),
            worst_off_ids_t=signal.worst_off_ids_t,
            worst_off_ids_t1=signal.worst_off_ids_t1,
        )
    return out


def potential_state_from_experiences(
    experiences: Mapping[str, float],
    *,
    terminated: bool = False,
    truncated: bool = False,
    terminal_label: str | None = None,
) -> PotentialState:
    """Build a PotentialState from explicit E_i values (synthetic tests)."""
    exp = _experiences_from_mapping(experiences)
    stakeholders = {
        sid: StakeholderState(explicit_experience=exp[sid], completed=False)
        for sid in STAKEHOLDER_ORDER
    }
    return PotentialState(
        stakeholders=stakeholders,
        terminated=terminated,
        truncated=truncated,
        terminal_label=terminal_label,
    )


def telescoping_sum(
    potentials: Sequence[float],
    gamma: float,
) -> tuple[float, list[float]]:
    """Compute sum_t gamma^t * F_t for F_t = gamma*phi_{t+1} - phi_t.

    Returns (sum, list of F_t). Does not enforce terminal/truncation semantics;
    callers supply the actual potential sequence already.
    """
    g = _require_finite("gamma", gamma)
    if not (0.0 <= g < 1.0):
        raise ValueError(f"gamma must satisfy 0 <= gamma < 1, got {g}")
    phis = [_require_finite(f"phi[{i}]", p) for i, p in enumerate(potentials)]
    if len(phis) < 2:
        raise ValueError("potentials must contain at least two values")
    f_list: list[float] = []
    total = 0.0
    disc = 1.0
    for t in range(len(phis) - 1):
        f_t = g * phis[t + 1] - phis[t]
        f_list.append(f_t)
        total += disc * f_t
        disc *= g
    return float(total), f_list
