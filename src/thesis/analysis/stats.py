"""Paired seed-level statistics for Stage 6B (bootstrap, Wilcoxon, Holm, dz)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:  # pragma: no cover
    scipy_wilcoxon = None


def paired_differences(
    values_a: Mapping[int, float | None],
    values_b: Mapping[int, float | None],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """values_* map seed -> float|None. Uses complete pairs only."""
    complete: list[tuple[int, float]] = []
    missing = 0
    for s in seeds:
        va = values_a.get(s)
        vb = values_b.get(s)
        if va is None or vb is None or not math.isfinite(float(va)) or not math.isfinite(float(vb)):
            missing += 1
            continue
        complete.append((int(s), float(va) - float(vb)))
    diffs = np.asarray([d for _, d in complete], dtype=np.float64)
    return {
        "paired_seeds": [s for s, _ in complete],
        "differences": diffs,
        "n_complete": int(len(complete)),
        "n_missing": int(missing),
        "mean_diff": float(diffs.mean()) if len(diffs) else float("nan"),
        "median_diff": float(np.median(diffs)) if len(diffs) else float("nan"),
    }


def paired_bootstrap_ci(
    diffs: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 91_001,
    alpha: float = 0.05,
) -> dict[str, Any]:
    diffs = np.asarray(diffs, dtype=np.float64)
    if len(diffs) == 0:
        return {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "defined": False,
            "n_boot": n_boot,
            "seed": seed,
        }
    rng = np.random.default_rng(int(seed))
    n = len(diffs)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "ci_low": lo,
        "ci_high": hi,
        "defined": True,
        "n_boot": n_boot,
        "seed": seed,
        "method": "percentile",
    }


def paired_cohen_dz(diffs: np.ndarray) -> dict[str, Any]:
    diffs = np.asarray(diffs, dtype=np.float64)
    if len(diffs) < 2:
        return {"dz": float("nan"), "defined": False, "reason": "n<2"}
    sd = float(diffs.std(ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return {"dz": float("nan"), "defined": False, "reason": "zero_or_nonfinite_sd"}
    return {"dz": float(diffs.mean() / sd), "defined": True, "reason": ""}


def paired_wilcoxon(diffs: np.ndarray) -> dict[str, Any]:
    diffs = np.asarray(diffs, dtype=np.float64)
    if scipy_wilcoxon is None:
        return {
            "stat": float("nan"),
            "pvalue": float("nan"),
            "defined": False,
            "reason": "scipy_missing",
        }
    # Drop exact zeros for wilcox zero_method; scipy handles via zero_method
    if len(diffs) == 0:
        return {"stat": float("nan"), "pvalue": float("nan"), "defined": False, "reason": "empty"}
    if np.allclose(diffs, 0.0):
        return {
            "stat": float("nan"),
            "pvalue": float("nan"),
            "defined": False,
            "reason": "all_zero_differences",
        }
    nonzero = diffs[diffs != 0.0]
    if len(nonzero) == 0:
        return {
            "stat": float("nan"),
            "pvalue": float("nan"),
            "defined": False,
            "reason": "no_nonzero_differences",
        }
    try:
        stat, p = scipy_wilcoxon(
            diffs,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        return {
            "stat": float(stat),
            "pvalue": float(p),
            "defined": True,
            "reason": "",
        }
    except ValueError as exc:
        return {
            "stat": float("nan"),
            "pvalue": float("nan"),
            "defined": False,
            "reason": str(exc),
        }


def holm_adjust(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjustment; None inputs stay None."""
    indexed = [(i, p) for i, p in enumerate(pvalues)]
    out: list[float | None] = [None] * len(pvalues)
    valid = [(i, float(p)) for i, p in indexed if p is not None and math.isfinite(float(p))]
    valid.sort(key=lambda t: t[1])
    m = len(valid)
    prev = 0.0
    for rank, (i, p) in enumerate(valid):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev)
        prev = adj
        out[i] = float(adj)
    return out


__all__ = [
    "holm_adjust",
    "paired_bootstrap_ci",
    "paired_cohen_dz",
    "paired_differences",
    "paired_wilcoxon",
]
