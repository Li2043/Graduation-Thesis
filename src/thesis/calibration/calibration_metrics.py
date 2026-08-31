"""Metrics and feasibility helpers for Stage 3B comfort calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from thesis.audits.audit_metrics import (
    braking_penalty_share,
    check_incentive_ordering,
    discounted_return,
    median,
    normalised_order_gap,
    require_finite,
)
from thesis.rewards.base_reward_v2 import compute_hard_braking_cost


TIE_TOL = 1e-12


def percentile(values: Sequence[float], p: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def h_from_accel(acceleration: float, a_comfort: float, a_hard: float) -> float:
    return compute_hard_braking_cost(acceleration, a_comfort, a_hard)


@dataclass(frozen=True)
class HDistribution:
    n: int
    nonzero_rate: float
    mean_h: float
    median_h: float
    p95_h: float
    saturation_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "nonzero_rate": self.nonzero_rate,
            "mean_h": self.mean_h,
            "median_h": self.median_h,
            "p95_h": self.p95_h,
            "saturation_rate": self.saturation_rate,
        }


def summarise_h(h_values: Sequence[float]) -> HDistribution:
    xs = [require_finite("H", float(h)) for h in h_values]
    n = len(xs)
    if n == 0:
        return HDistribution(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    nonzero = sum(1 for h in xs if h > 0.0)
    sat = sum(1 for h in xs if abs(h - 1.0) <= 1e-12)
    return HDistribution(
        n=n,
        nonzero_rate=nonzero / n,
        mean_h=float(sum(xs) / n),
        median_h=median(xs),
        p95_h=percentile(xs, 95),
        saturation_rate=sat / n,
    )


def valid_threshold_pair(a_comfort: float, a_hard: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not (a_hard > a_comfort):
        reasons.append("a_hard_not_greater_than_a_comfort")
    if (a_hard - a_comfort) < 1.5 - TIE_TOL:
        reasons.append("a_hard_minus_a_comfort_lt_1.5")
    return len(reasons) == 0, reasons


def threshold_pair_feasible(
    *,
    nominal: HDistribution,
    hard_window: HDistribution,
    separation_score: float,
    n_blocks_hard_detection_lt_0_70: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if nominal.nonzero_rate > 0.10 + TIE_TOL:
        reasons.append(f"nominal_nonzero_rate={nominal.nonzero_rate:.6f}>0.10")
    if nominal.saturation_rate > 0.01 + TIE_TOL:
        reasons.append(f"nominal_saturation_rate={nominal.saturation_rate:.6f}>0.01")
    if hard_window.nonzero_rate < 0.80 - TIE_TOL:
        reasons.append(f"hard_window_nonzero_rate={hard_window.nonzero_rate:.6f}<0.80")
    if hard_window.mean_h < 0.20 - TIE_TOL:
        reasons.append(f"hard_window_mean_h={hard_window.mean_h:.6f}<0.20")
    if separation_score < 0.15 - TIE_TOL:
        reasons.append(f"separation_score={separation_score:.6f}<0.15")
    if n_blocks_hard_detection_lt_0_70 > 1:
        reasons.append(
            f"block_hard_detection_failures={n_blocks_hard_detection_lt_0_70}>1"
        )
    return len(reasons) == 0, reasons


def compare_threshold_lex(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> int:
    """Compare (separation, -nominal_mean_h, a_comfort, a_hard). Higher is better."""
    for x, y in zip(a, b):
        if x > y + TIE_TOL:
            return 1
        if x < y - TIE_TOL:
            return -1
    return 0


def assert_h_monotonicities() -> None:
    """Mathematical / numerical monotonicity checks for H."""
    # braking magnitude ↑ ⇒ H non-decreasing
    a_c, a_h = 2.0, 6.0
    prev = -1.0
    for mag in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        h = h_from_accel(-mag, a_c, a_h)
        if h + TIE_TOL < prev:
            raise AssertionError(f"H not non-decreasing in braking magnitude: {mag}")
        prev = h
    # a = -a_comfort ⇒ H=0; a <= -a_hard ⇒ H=1
    if h_from_accel(-a_c, a_c, a_h) != 0.0:
        raise AssertionError("H must be 0 at comfort boundary")
    if abs(h_from_accel(-a_h, a_c, a_h) - 1.0) > TIE_TOL:
        raise AssertionError("H must be 1 at hard threshold")
    # intermediate quadratic
    a = -4.0
    expected = ((4.0 - a_c) / (a_h - a_c)) ** 2
    if abs(h_from_accel(a, a_c, a_h) - expected) > TIE_TOL:
        raise AssertionError("intermediate H mismatch")
    # increasing a_comfort cannot increase H
    accel = -4.0
    h_lo = h_from_accel(accel, 1.5, 6.0)
    h_hi = h_from_accel(accel, 2.5, 6.0)
    if h_hi > h_lo + TIE_TOL:
        raise AssertionError("increasing a_comfort increased H")
    # increasing a_hard cannot increase H before saturation
    h_ah_lo = h_from_accel(accel, 2.0, 5.0)
    h_ah_hi = h_from_accel(accel, 2.0, 7.0)
    if h_ah_hi > h_ah_lo + TIE_TOL:
        raise AssertionError("increasing a_hard increased H")


def assert_eta_penalty_monotone(h: float, etas: Sequence[float]) -> None:
    prev = -1.0
    for eta in etas:
        pen = abs(-float(eta) * float(h))
        if pen + TIE_TOL < prev:
            raise AssertionError("braking penalty not non-decreasing in eta")
        prev = pen


__all__ = [
    "HDistribution",
    "TIE_TOL",
    "assert_eta_penalty_monotone",
    "assert_h_monotonicities",
    "braking_penalty_share",
    "check_incentive_ordering",
    "compare_threshold_lex",
    "discounted_return",
    "h_from_accel",
    "median",
    "normalised_order_gap",
    "percentile",
    "summarise_h",
    "threshold_pair_feasible",
    "valid_threshold_pair",
]
