"""Shared statistics primitives for the DWS final re-evaluation (Sections 5-18
of CLAUDE_CODE_FINAL_DWS_REEVALUATION_PROMPT.md). Pure functions, no I/O.

Paired bootstrap convention used throughout: seed identity is the matched
resampling unit (Section 4's own inferential-unit rule). Each bootstrap
resample draws 12 seed INDICES with replacement (numpy.random.default_rng(0)),
and for each resample takes the mean of the (already seed-level, already
paired) per-seed effect values at those resampled indices. 10,000 resamples,
95% percentile CI. Two-sided bootstrap p-value for a null of zero:
p = 2 * min(P(resampled_mean <= 0), P(resampled_mean >= 0)), capped at 1.0
(the standard "flip the tail that crosses zero" percentile-bootstrap p-value
construction for a paired-difference null).
"""
from __future__ import annotations

import numpy as np

N_BOOT = 10_000
RNG_SEED = 0
CI_LEVEL = 0.95


def paired_bootstrap(seed_effects: list[float], *, n_boot: int = N_BOOT, rng_seed: int = RNG_SEED) -> dict:
    """seed_effects: one already-computed per-seed paired difference per seed
    (length = n_seeds, order arbitrary but must be consistent with any other
    array you resample jointly with this one)."""
    arr = np.asarray(seed_effects, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100])
    p_le = float(np.mean(boot_means <= 0))
    p_ge = float(np.mean(boot_means >= 0))
    p_value = min(1.0, 2 * min(p_le, p_ge))
    return {
        "mean_effect": float(arr.mean()),
        "median_effect": float(np.median(arr)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "raw_p": p_value,
        "n_positive": int(np.sum(arr > 0)),
        "n_negative": int(np.sum(arr < 0)),
        "n_zero": int(np.sum(arr == 0)),
        "n_seeds": n,
        "seed_effects": arr.tolist(),
    }


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    SAME order as the input (not sorted)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * p_values[i]
        running_max = max(running_max, adj)
        adjusted[i] = min(1.0, running_max)
    return adjusted


def leave_one_out(seed_ids: list, seed_effects: list[float]) -> dict:
    """Full n estimate, min/max leave-one-out estimate, and which omitted
    seed produces each extreme."""
    arr = np.asarray(seed_effects, dtype=float)
    full = float(arr.mean())
    loo_means = []
    for i in range(len(arr)):
        mask = np.ones(len(arr), dtype=bool)
        mask[i] = False
        loo_means.append(float(arr[mask].mean()))
    min_i = int(np.argmin(loo_means))
    max_i = int(np.argmax(loo_means))
    return {
        "full_n_effect": full,
        "loo_min": loo_means[min_i], "loo_min_omitted_seed": seed_ids[min_i],
        "loo_max": loo_means[max_i], "loo_max_omitted_seed": seed_ids[max_i],
        "direction_changes": bool((full > 0) != (loo_means[min_i] > 0) or (full > 0) != (loo_means[max_i] > 0)),
    }
