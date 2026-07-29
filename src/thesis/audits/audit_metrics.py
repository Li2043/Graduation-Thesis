"""Audit metrics for Stage 3A base-outcome incentive audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def require_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value)!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def discounted_return(rewards: Sequence[float], gamma: float) -> float:
    g = require_finite("gamma", gamma)
    if not (0.0 <= g < 1.0):
        raise ValueError(f"gamma must satisfy 0 <= gamma < 1, got {g}")
    total = 0.0
    disc = 1.0
    for r in rewards:
        total += disc * require_finite("reward", float(r))
        disc *= g
    return float(total)


def undiscounted_return(rewards: Sequence[float]) -> float:
    return float(sum(require_finite("reward", float(r)) for r in rewards))


def component_returns(
    component_series: Mapping[str, Sequence[float]],
    gamma: float,
) -> dict[str, dict[str, float]]:
    """Return discounted and undiscounted aggregates per component name."""
    out: dict[str, dict[str, float]] = {}
    for name, series in component_series.items():
        out[name] = {
            "discounted": discounted_return(series, gamma),
            "undiscounted": undiscounted_return(series),
        }
    return out


def normalised_order_gap(g_mainline_first: float, g_ramp_first: float) -> dict[str, float]:
    a = require_finite("G_mainline_first", g_mainline_first)
    b = require_finite("G_ramp_first", g_ramp_first)
    gap = abs(a - b)
    denom = max(abs((a + b) / 2.0), 1e-12)
    return {
        "order_gap": float(gap),
        "normalised_order_gap": float(gap / denom),
        "G_team_mainline_first": a,
        "G_team_ramp_first": b,
    }


def closed_cycle_progress_sum(deltas: Sequence[float]) -> float:
    return float(sum(require_finite("delta_rho", float(d)) for d in deltas))


def discounted_cycle_progress_return(deltas: Sequence[float], gamma: float) -> float:
    """Discounted return of progress components 0.4 * delta_rho."""
    return discounted_return([0.4 * float(d) for d in deltas], gamma)


def oscillation_ratio(
    discounted_oscillation_team_return: float,
    nominal_safe_team_return: float,
) -> float:
    num = abs(require_finite("osc", discounted_oscillation_team_return))
    den = abs(require_finite("nominal", nominal_safe_team_return))
    return float(num / max(den, 1e-12))


def braking_penalty_share(
    g_hard_braking: float,
    g_progress: float,
    g_exit: float,
) -> float:
    num = abs(require_finite("G_hard_braking", g_hard_braking))
    den = require_finite("G_progress", g_progress) + require_finite("G_exit", g_exit)
    return float(num / max(den, 1e-12))


@dataclass(frozen=True)
class IncentiveOrderingResult:
    block_id: str
    ok: bool
    violations: tuple[str, ...]
    G_team_safe_mainline_first: float | None
    G_team_safe_ramp_first: float | None
    G_team_stall_after_partial: float | None
    G_team_early_collision: float | None
    G_team_late_collision: float | None


def check_incentive_ordering(
    block_id: str,
    *,
    g_safe_mainline: float | None,
    g_safe_ramp: float | None,
    g_slow_mainline: float | None,
    g_slow_ramp: float | None,
    g_stall_partial: float | None,
    g_early_coll: float | None,
    g_late_coll: float | None,
) -> IncentiveOrderingResult:
    """Block-level ordering; failures are listed, never averaged away."""
    violations: list[str] = []
    required = {
        "safe_mainline_first": g_safe_mainline,
        "safe_ramp_first": g_safe_ramp,
        "stall_after_partial_progress": g_stall_partial,
        "early_collision": g_early_coll,
        "late_collision": g_late_coll,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        violations.append(f"missing_scenarios:{','.join(missing)}")
        return IncentiveOrderingResult(
            block_id=block_id,
            ok=False,
            violations=tuple(violations),
            G_team_safe_mainline_first=g_safe_mainline,
            G_team_safe_ramp_first=g_safe_ramp,
            G_team_stall_after_partial=g_stall_partial,
            G_team_early_collision=g_early_coll,
            G_team_late_collision=g_late_coll,
        )

    assert g_safe_mainline is not None and g_safe_ramp is not None
    assert g_stall_partial is not None
    assert g_early_coll is not None and g_late_coll is not None

    min_safe = min(g_safe_mainline, g_safe_ramp)
    max_coll = max(g_early_coll, g_late_coll)

    if min_safe <= g_stall_partial:
        violations.append(
            f"safe_not_above_stall: min_safe={min_safe:.6f} stall={g_stall_partial:.6f}"
        )
    if g_stall_partial <= max_coll:
        violations.append(
            f"stall_not_above_collision: stall={g_stall_partial:.6f} max_coll={max_coll:.6f}"
        )

    # Nominal vs slow (if available)
    for label, g_nom, g_slow in (
        ("mainline", g_safe_mainline, g_slow_mainline),
        ("ramp", g_safe_ramp, g_slow_ramp),
    ):
        if g_slow is not None and g_nom <= g_slow:
            violations.append(
                f"nominal_not_above_slow_{label}: nom={g_nom:.6f} slow={g_slow:.6f}"
            )

    return IncentiveOrderingResult(
        block_id=block_id,
        ok=len(violations) == 0,
        violations=tuple(violations),
        G_team_safe_mainline_first=g_safe_mainline,
        G_team_safe_ramp_first=g_safe_ramp,
        G_team_stall_after_partial=g_stall_partial,
        G_team_early_collision=g_early_coll,
        G_team_late_collision=g_late_coll,
    )


def median(values: Sequence[float]) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("median of empty sequence")
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])
