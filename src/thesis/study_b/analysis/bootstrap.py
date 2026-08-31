"""Seed-blocked bootstrap CIs + Holm correction -- new_research_plan.md's
"Bootstrap" / "Multiple comparisons" sections. Operates on already-
seed-aggregated values (one float per seed, from ``welfare.seed_level_summary``)
-- never on raw per-scenario rows, per the "seed is the inferential unit"
principle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["BootstrapResult", "paired_bootstrap_contrast", "holm_correction"]


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float  # two-sided, from the bootstrap distribution
    p_improvement: float  # P(condition - baseline > 0)
    n_seeds: int


def paired_bootstrap_contrast(
    condition_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    n_replicates: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """``condition_values[i]``/``baseline_values[i]`` must come from the
    SAME seed block (new_research_plan.md's matched-seed-blocking device --
    seed S01 under both conditions is index 0 in both lists, etc.). Small
    n (4-6 seeds, per the Phase 3 minimum/target) means these intervals
    will legitimately be wide -- report them as such, do not treat a wide
    CI as a bug."""
    if len(condition_values) != len(baseline_values):
        raise ValueError("condition_values and baseline_values must be the same length (matched seed blocks)")
    n = len(condition_values)
    if n < 2:
        raise ValueError("need at least 2 seeds for a bootstrap CI")

    diffs = np.asarray(condition_values, dtype=np.float64) - np.asarray(baseline_values, dtype=np.float64)
    point_estimate = float(diffs.mean())

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_replicates, dtype=np.float64)
    for i in range(n_replicates):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()

    ci_lower, ci_upper = (float(x) for x in np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    p_improvement = float((boot_means > 0).mean())
    p_value = float(min(1.0, 2 * min(p_improvement, 1 - p_improvement)))

    return BootstrapResult(
        point_estimate=point_estimate, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=p_value, p_improvement=p_improvement, n_seeds=n,
    )


def holm_correction(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, dict]:
    """Standard Holm step-down procedure. Once one hypothesis (in ascending
    p-value order) fails to clear its threshold, every hypothesis with a
    LARGER p-value is also not rejected, regardless of its own individual
    threshold -- this is what makes it a valid family-wise-error-rate
    correction rather than m independent tests at alpha/m."""
    m = len(p_values)
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    results: dict[str, dict] = {}
    stopped = False
    for rank, (name, p) in enumerate(ordered, start=1):
        threshold = alpha / (m - rank + 1)
        reject = (not stopped) and (p <= threshold)
        if not reject:
            stopped = True
        results[name] = {"p_raw": p, "rank": rank, "threshold": threshold, "reject_null": reject}
    return results
