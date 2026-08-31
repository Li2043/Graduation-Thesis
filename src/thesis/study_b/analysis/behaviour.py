"""Behavioural mechanism analysis -- new_research_plan.md's "Behavioural
analysis" / "P(worse-off | class)" sections. Operates directly on
``evaluate_policy.py``'s per-scenario CSV rows (``welfare.read_eval_csv``)."""

from __future__ import annotations

from collections import Counter
from typing import Any

__all__ = ["worse_off_frequency_by_class", "mean_hard_brake_rate"]

_VEHICLE_IDS = ("V0", "V1", "V2", "V3")


def worse_off_frequency_by_class(rows: list[dict[str, str]]) -> dict[str, float]:
    """P(class is the argmin-utility vehicle | class is present in the
    episode) for each of the 4 (role, speed_class) combinations. Every
    scenario has exactly one member of each combination (Ramp-Fast,
    Ramp-Slow, Mainline-Fast, Mainline-Slow), so the denominator is simply
    the row count -- this only changes if a future bank ever varies N or
    the fast/slow split per role."""
    counts: Counter[str] = Counter()
    n = len(rows)
    for row in rows:
        key = f"{row['min_U_role']}_{row['min_U_speed_class']}"
        counts[key] += 1
    all_classes = {f"{role}_{cls}" for role in ("ramp", "mainline") for cls in ("fast", "slow")}
    return {cls: counts.get(cls, 0) / n for cls in all_classes}


def mean_hard_brake_rate(rows: list[dict[str, str]]) -> dict[str, float]:
    """Mean per-episode hard-brake COUNT, overall and split by class --
    raw counts (not yet a per-active-step rate; that needs episode length,
    which this CSV doesn't carry per-vehicle -- acceptable for a Phase 3
    behavioural-mechanism read, not claimed as a precise per-step rate)."""
    n = len(rows)
    overall = sum(float(r["hard_brake_total"]) for r in rows) / n
    by_class: dict[str, list[float]] = {}
    for row in rows:
        for vid in _VEHICLE_IDS:
            key = f"{row[f'role_{vid}']}_{row[f'speed_class_{vid}']}"
            by_class.setdefault(key, []).append(float(row[f"hard_brake_{vid}"]))
    return {"overall": overall, **{k: sum(v) / len(v) for k, v in by_class.items()}}
