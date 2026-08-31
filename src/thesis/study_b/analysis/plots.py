"""Learning-curve and bootstrap-CI ("forest plot") figures --
new_research_plan.md's "Learning-curve chart template" section, adapted to
Study B's manifest/bootstrap-result shapes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless -- this runs on a training/analysis machine, not interactively
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from thesis.study_b.analysis.bootstrap import BootstrapResult  # noqa: E402

__all__ = ["plot_learning_curves", "plot_bootstrap_forest"]


def plot_learning_curves(
    manifests_by_condition: Mapping[str, Sequence[dict]], *, metric: str, output_path: Path
) -> None:
    """``manifests_by_condition``: e.g. ``{"baseline": [manifest_seed1, manifest_seed2, ...], ...}``,
    each manifest dict as written by ``train_mappo.py``/``train_dqn_fallback.py``
    (has a ``checkpoints`` list of ``{"step":..., "window": {metric: ...}}``).
    Plots per-condition mean +/- 95% normal-approximation band across seeds
    at each shared checkpoint step (bootstrap bands are more correct for
    the FINAL thesis figures per new_research_plan.md -- this quick
    version is for monitoring, not the confirmatory figure)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, manifests in manifests_by_condition.items():
        steps = sorted({rec["step"] for m in manifests for rec in m["checkpoints"]})
        means, los, his = [], [], []
        for step in steps:
            values = []
            for m in manifests:
                for rec in m["checkpoints"]:
                    if rec["step"] == step:
                        v = rec["window"].get(metric)
                        if v is not None:
                            values.append(v)
            if not values:
                means.append(np.nan)
                los.append(np.nan)
                his.append(np.nan)
                continue
            arr = np.asarray(values, dtype=np.float64)
            mean = float(arr.mean())
            se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            means.append(mean)
            los.append(mean - 1.96 * se)
            his.append(mean + 1.96 * se)
        ax.plot(steps, means, label=condition)
        ax.fill_between(steps, los, his, alpha=0.2)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_bootstrap_forest(results: Mapping[str, BootstrapResult], *, output_path: Path) -> None:
    """One horizontal error bar per contrast (e.g. 'mean_pbrs - baseline
    (U_mean)', 'min_pbrs - baseline (U_min)') -- point estimate + 95% CI,
    with a vertical zero-effect reference line."""
    names = list(results.keys())
    points = [results[n].point_estimate for n in names]
    los = [results[n].point_estimate - results[n].ci_lower for n in names]
    his = [results[n].ci_upper - results[n].point_estimate for n in names]

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1.5))
    y_pos = np.arange(len(names))
    ax.errorbar(points, y_pos, xerr=[los, his], fmt="o", capsize=4)
    ax.axvline(0.0, linestyle="--", color="grey")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Effect (condition - baseline), 95% bootstrap CI")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
