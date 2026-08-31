"""Seed-level aggregation of ``evaluate_policy.py``'s per-scenario CSV
output -- new_research_plan.md's "Seed 是 inferential unit" principle: a
seed's 256 (or however many) evaluation episodes collapse to ONE row here
before any cross-condition comparison happens in ``bootstrap.py``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

__all__ = ["read_eval_csv", "seed_level_summary"]


def read_eval_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_level_summary(path: Path) -> dict[str, Any]:
    rows = read_eval_csv(path)
    if not rows:
        raise ValueError(f"{path} has no rows")
    n = len(rows)

    def col_mean(name: str) -> float:
        return sum(float(r[name]) for r in rows) / n

    gini_values = [float(r["gini"]) for r in rows if r["gini"] not in ("", "None", "NA")]

    return {
        "path": str(path),
        "n_scenarios": n,
        "completion_rate": col_mean("completion"),
        "collision_rate": col_mean("collision"),
        "timeout_rate": col_mean("timeout"),
        "mean_U": col_mean("mean_U"),
        "min_U": col_mean("min_U"),
        "ggi": col_mean("ggi"),
        "gini": (sum(gini_values) / len(gini_values)) if gini_values else None,
        "all_zero_utility_rate": 1.0 - len(gini_values) / n,
        "C_max": col_mean("C_max"),
        "C_mean": col_mean("C_mean"),
        "hard_brake_total": col_mean("hard_brake_total"),
    }
