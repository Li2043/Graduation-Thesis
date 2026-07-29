"""Metrics for Stage 4A choice-state certification (core reward only)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from thesis.audits.audit_metrics import discounted_return, median, normalised_order_gap, require_finite


GAMMA = 0.995
TIE = 1e-12


@dataclass
class CellOutcome:
    cell: str
    success: bool
    collision: bool
    truncated: bool
    exit_order: str
    G_team_core: float
    G_A_core: float
    G_B_core: float
    episode_length: int
    exit_time_mainline: int | None
    exit_time_ramp: int | None
    min_bumper_gap: float | None
    min_ttc: float | None
    min_accel: float
    max_accel: float
    route_discontinuity: int
    nan_count: int
    invalid_flags: int
    repeated_exit: int
    fixture_count: int
    bg_min_speed: dict[str, float]
    bg_max_brake: dict[str, float]
    bg_min_gap_to_learners: float | None
    selected_macros: dict[str, str] = field(default_factory=dict)
    physically_safe: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: getattr(self, k) for k in self.__dataclass_fields__},
        }


def classify_exit_order(exit_time_mainline: int | None, exit_time_ramp: int | None) -> str:
    if exit_time_mainline is None or exit_time_ramp is None:
        return "partial" if (exit_time_mainline is not None or exit_time_ramp is not None) else "none"
    if exit_time_mainline == exit_time_ramp:
        return "simultaneous"
    return "mainline_first" if exit_time_mainline < exit_time_ramp else "ramp_first"


def cell_physically_safe(o: CellOutcome) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if o.collision:
        reasons.append("collision")
    if o.min_bumper_gap is not None and o.min_bumper_gap < 2.0 - TIE:
        reasons.append(f"min_gap={o.min_bumper_gap:.3f}<2.0")
    if o.min_ttc is not None and o.min_ttc < 1.0 - TIE:
        reasons.append(f"ttc={o.min_ttc:.3f}<1.0")
    if o.min_accel < -6.0 - TIE:
        reasons.append(f"accel_low={o.min_accel}")
    if o.max_accel > 3.0 + TIE:
        reasons.append(f"accel_high={o.max_accel}")
    if o.route_discontinuity:
        reasons.append("route_discontinuity")
    if o.invalid_flags:
        reasons.append("invalid_flags")
    if o.nan_count:
        reasons.append("nan")
    if o.fixture_count:
        reasons.append("fixture")
    if not o.success:
        reasons.append("not_success")
    return len(reasons) == 0, reasons


def go_go_problematic(go_go: CellOutcome, ml_first: CellOutcome, rp_first: CellOutcome) -> bool:
    """GO/GO must not dominate both asymmetric conventions."""
    if go_go.collision or go_go.truncated or not go_go.success:
        return True
    if go_go.min_bumper_gap is not None and go_go.min_bumper_gap < 2.0:
        return True
    if go_go.min_ttc is not None and go_go.min_ttc < 1.0:
        return True
    # substantially greater emergency deceleration
    if go_go.min_accel + 1.0 < min(ml_first.min_accel, rp_first.min_accel):
        return True
    # not both safer and more efficient
    better_than_both = (
        go_go.success
        and go_go.G_team_core > ml_first.G_team_core
        and go_go.G_team_core > rp_first.G_team_core
        and go_go.episode_length <= min(ml_first.episode_length, rp_first.episode_length)
    )
    return not better_than_both


def yield_yield_inefficient(
    yy: CellOutcome, ml_first: CellOutcome, rp_first: CellOutcome, policy_dt: float = 0.2
) -> bool:
    if yy.collision:
        return False
    if not yy.success:
        return True  # unresolved/delayed
    faster = min(ml_first.episode_length, rp_first.episode_length)
    if yy.episode_length * policy_dt + 1e-12 < faster * policy_dt + 1.0:
        # must be at least 1s later
        if yy.G_team_core < min(ml_first.G_team_core, rp_first.G_team_core) - TIE:
            return True
        return False
    return yy.G_team_core < min(ml_first.G_team_core, rp_first.G_team_core) - TIE


def no_unilateral_guarantee(matrix: dict[str, CellOutcome]) -> bool:
    """Changing the other role's macro must change order/safety/completion/return class."""
    # For mainline fixed GO: ramp GO vs YIELD
    a, b = matrix["GO_GO"], matrix["GO_YIELD"]
    if not _differs(a, b):
        return False
    # mainline fixed YIELD
    a, b = matrix["YIELD_GO"], matrix["YIELD_YIELD"]
    if not _differs(a, b):
        return False
    # ramp fixed GO
    a, b = matrix["GO_GO"], matrix["YIELD_GO"]
    if not _differs(a, b):
        return False
    a, b = matrix["GO_YIELD"], matrix["YIELD_YIELD"]
    if not _differs(a, b):
        return False
    return True


def _differs(a: CellOutcome, b: CellOutcome) -> bool:
    if a.exit_order != b.exit_order:
        return True
    if a.success != b.success or a.collision != b.collision:
        return True
    # return classification: sign / coarse bucket
    if (a.G_team_core > 0) != (b.G_team_core > 0):
        return True
    if abs(a.G_team_core - b.G_team_core) > 0.05:
        return True
    return False


def background_meaningful(c1: CellOutcome, c2: CellOutcome) -> bool:
    for key in ("B_front", "B_rear"):
        if abs(c1.bg_min_speed.get(key, 0) - c2.bg_min_speed.get(key, 0)) >= 0.5:
            return True
        if abs(c1.bg_max_brake.get(key, 0) - c2.bg_max_brake.get(key, 0)) >= 1.0:
            return True
        # max-speed diagnostic keys optionally attached during certification
        kmax = f"{key}_max"
        if abs(c1.bg_min_speed.get(kmax, 0) - c2.bg_min_speed.get(kmax, 0)) >= 0.5:
            return True
    g1, g2 = c1.bg_min_gap_to_learners, c2.bg_min_gap_to_learners
    if g1 is not None and g2 is not None and abs(g1 - g2) >= 2.0:
        return True
    return False


def _go_go_unsafe(gg: CellOutcome) -> bool:
    if gg.collision or (not gg.success) or gg.truncated:
        return True
    if gg.min_bumper_gap is not None and gg.min_bumper_gap < 2.0 - TIE:
        return True
    if gg.min_ttc is not None and gg.min_ttc < 1.0 - TIE:
        return True
    return False


def core_ordering_ok(ml: CellOutcome, rp: CellOutcome, yy: CellOutcome, gg: CellOutcome) -> bool:
    if not (ml.G_team_core > yy.G_team_core + TIE and rp.G_team_core > yy.G_team_core + TIE):
        return False
    if _go_go_unsafe(gg):
        return yy.G_team_core > gg.G_team_core + TIE
    return (
        ml.G_team_core > yy.G_team_core + TIE
        and rp.G_team_core > yy.G_team_core + TIE
        and yy.G_team_core > gg.G_team_core + TIE
    )


def aggregate_order_gaps(gaps: Sequence[float]) -> dict[str, float]:
    xs = [require_finite("gap", g) for g in gaps]
    return {
        "median_normalised_order_gap": median(xs) if xs else float("nan"),
        "maximum_normalised_order_gap": max(xs) if xs else float("nan"),
    }


__all__ = [
    "CellOutcome",
    "GAMMA",
    "aggregate_order_gaps",
    "background_meaningful",
    "cell_physically_safe",
    "classify_exit_order",
    "core_ordering_ok",
    "discounted_return",
    "go_go_problematic",
    "no_unilateral_guarantee",
    "normalised_order_gap",
    "yield_yield_inefficient",
]
