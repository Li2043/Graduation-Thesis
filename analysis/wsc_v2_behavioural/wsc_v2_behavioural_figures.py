"""Figures for the WSC v2 behavioural mechanism analysis. Reads only the CSV
outputs of wsc_v2_behavioural_aggregate.py; does not recompute statistics.
Style matches the project's existing thesis figure convention.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                      "figure.dpi": 150, "savefig.dpi": 150})

OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS_12 = [900101, 900102, 900103, 900104, 910101, 910102,
            920101, 920102, 920103, 920104, 920105, 920106]
HIGH_LEVERAGE_SEEDS = {910102, 920102}
CONDITIONS = ["baseline", "mean", "ggi", "maximin"]
WELFARE_CONDITIONS = ["mean", "ggi", "maximin"]
COND_LABEL = {"baseline": "Baseline", "mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}
COND_COLOR = {"baseline": "tab:gray", "mean": "tab:blue", "ggi": "tab:orange", "maximin": "tab:green"}
PRIMARY_K = 25


def load_seed_summary():
    with open(OUT_DIR / "wsc_behavioural_seed_summary.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_interactions():
    with open(OUT_DIR / "wsc_behavioural_interactions.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fig_metric_paired_by_condition(rows, metric, title, fname, ylabel=None):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    by_key = {(r["seed"], r["condition"], r["regime"]): r for r in rows}
    for ax, cond in zip(axes, CONDITIONS):
        for seed in SEEDS_12:
            o = _f(by_key.get((str(seed), cond, "original"), {}), metric)
            w = _f(by_key.get((str(seed), cond, "wsc"), {}), metric)
            if not (np.isfinite(o) and np.isfinite(w)):
                continue
            color = "tab:red" if seed in HIGH_LEVERAGE_SEEDS else "gray"
            ax.plot([0, 1], [o, w], color=color, alpha=0.6, linewidth=1.2, marker="o", markersize=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Original", "+WSC"])
        ax.set_title(COND_LABEL[cond])
        ax.set_xlim(-0.3, 1.3)
    axes[0].set_ylabel(ylabel or metric)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / (fname + '.png')}")


def fig_interaction_forest(interaction_rows):
    metrics = sorted({r["metric"] for r in interaction_rows})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        y = 0
        yticks, ylabels = [], []
        for cond in WELFARE_CONDITIONS:
            for analysis, marker, alpha, dy in (("primary_n12", "o", 1.0, 0.15), ("sensitivity_n11_excl_910102", "s", 0.5, -0.15)):
                sub = [r for r in interaction_rows if r["metric"] == metric and r["condition"] == cond and r["analysis"] == analysis]
                if not sub or sub[0].get("mean", "") == "":
                    continue
                r = sub[0]
                mean, lo, hi = _f(r, "mean"), _f(r, "CI95_low"), _f(r, "CI95_high")
                if not np.isfinite(mean):
                    continue
                ax.errorbar(mean, y + dy, xerr=[[mean - lo], [hi - mean]], fmt=marker,
                            color=COND_COLOR[cond], alpha=alpha, capsize=4,
                            label=("n=12 (primary)" if (y == 0 and analysis == "primary_n12") else
                                   ("n=11 (sensitivity)" if (y == 0 and analysis != "primary_n12") else None)))
            yticks.append(y); ylabels.append(COND_LABEL[cond]); y += 1
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
        ax.set_title(metric)
        ax.invert_yaxis()
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle("Figure 5: Behavioural interaction point estimates + 95% CI", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_behavioural_interaction_forest.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_behavioural_interaction_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig5_behavioural_interaction_forest.png'}")


def main():
    rows = load_seed_summary()
    inter = load_interactions()

    fig_metric_paired_by_condition(rows, "RY", "Figure 1: Welfare-responsive yielding contrast (RY), Original vs WSC",
                                    "fig1_RY_paired", ylabel="RY = P(yield|worse-off) / P(yield|not worse-off)")
    fig_metric_paired_by_condition(rows, "P_priority_worse", "Figure 2: P(worse-off vehicle receives merge priority)",
                                    "fig2_priority_paired", ylabel="P(priority | currently worse-off)")
    fig_metric_paired_by_condition(rows, "BC", "Figure 3: Cooperative burden/sacrifice contrast (BC)",
                                    "fig3_burden_paired", ylabel="BC = P(costly action|neighbour worse-off) / P(costly action|not)")
    fig_metric_paired_by_condition(rows, f"GapClosure_k{PRIMARY_K}", f"Figure 4: Worst-off gap closure (k={PRIMARY_K} steps)",
                                    "fig4_gapclosure_paired", ylabel=f"GapClosure_k{PRIMARY_K}")
    fig_interaction_forest(inter)


if __name__ == "__main__":
    main()
