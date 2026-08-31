"""Figures for the formal WSC v2 evaluation. Reads only the CSV/JSON outputs
of wsc_v2_formal_analysis.py (does not recompute statistics). Style matches
the project's existing thesis figure convention
(pooled12/pooled_figures.py's rcParams block).
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
OUTLIER_SEED = 910102
WELFARE_CONDITIONS = ["mean", "ggi", "maximin"]
COND_LABEL = {"mean": "Mean", "ggi": "GGI", "maximin": "Maximin", "baseline": "Baseline"}
COND_COLOR = {"mean": "tab:blue", "ggi": "tab:orange", "maximin": "tab:green", "baseline": "tab:gray"}


def load_seed_level():
    with open(OUT_DIR / "wsc_v2_formal_seed_level.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_bootstrap():
    with open(OUT_DIR / "wsc_v2_formal_bootstrap_results.json", encoding="utf-8") as f:
        return json.load(f)


def _seed_marker(seed):
    return "D" if seed == OUTLIER_SEED else "o"


def fig_interaction_by_seed(rows, key, ylabel, title, fname):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, cond in zip(axes, WELFARE_CONDITIONS):
        vals = {int(r["seed"]): float(r[key]) for r in rows if r["condition"] == cond}
        xs = np.arange(len(SEEDS_12))
        ys = [vals[s] for s in SEEDS_12]
        colors = ["tab:red" if s == OUTLIER_SEED else COND_COLOR[cond] for s in SEEDS_12]
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.scatter(xs, ys, c=colors, s=55, zorder=3, edgecolors="black", linewidths=0.5)
        for x, y, s in zip(xs, ys, SEEDS_12):
            if s == OUTLIER_SEED:
                ax.annotate(str(s), (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
        mean_val = float(np.mean(ys))
        ax.axhline(mean_val, color=COND_COLOR[cond], linewidth=1.5, alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in SEEDS_12], rotation=90, fontsize=7)
        ax.set_title(f"{COND_LABEL[cond]} (mean={mean_val:+.3f})")
        ax.set_xlabel("seed")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / (fname + '.png')}")


def fig_paired_task_metric(rows, metric, title, fname):
    conditions = ["baseline", "mean", "ggi", "maximin"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, cond in zip(axes, conditions):
        orig_vals = {int(r["seed"]): float(r[f"orig_{metric}"]) for r in rows if r["condition"] == cond}
        wsc_vals = {int(r["seed"]): float(r[f"wsc_{metric}"]) for r in rows if r["condition"] == cond}
        for s in SEEDS_12:
            color = "tab:red" if s == OUTLIER_SEED else "gray"
            ax.plot([0, 1], [orig_vals[s], wsc_vals[s]], color=color, alpha=0.6, linewidth=1.2,
                    marker=_seed_marker(s), markersize=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Original", "+WSC"])
        ax.set_title(COND_LABEL[cond])
        ax.set_xlim(-0.3, 1.3)
    axes[0].set_ylabel(metric.capitalize())
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / (fname + '.png')}")


def fig_ci_forest(boot):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, outcome in zip(axes, ("U_min", "Gini")):
        primary = boot["primary_n12"]["outcomes"][outcome]
        sens = boot["sensitivity_n11"]["outcomes"][outcome]
        y_positions = []
        labels = []
        y = 0
        for cond in WELFARE_CONDITIONS:
            p = primary[cond]
            s = sens[cond]
            ax.errorbar(p["mean"], y + 0.15, xerr=[[p["mean"] - p["CI95_low"]], [p["CI95_high"] - p["mean"]]],
                        fmt="o", color=COND_COLOR[cond], capsize=4, label="n=12 (primary)" if y == 0 else None)
            ax.errorbar(s["mean"], y - 0.15, xerr=[[s["mean"] - s["CI95_low"]], [s["CI95_high"] - s["mean"]]],
                        fmt="s", color=COND_COLOR[cond], alpha=0.5, capsize=4, label="n=11 (sensitivity)" if y == 0 else None)
            labels.append(COND_LABEL[cond])
            y_positions.append(y)
            y += 1
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"I_{outcome} (95% CI)")
        ax.set_title(f"{outcome} interaction: point estimate + 95% CI")
        ax.invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_interaction_forest.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_interaction_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig5_interaction_forest.png'}")


def main():
    rows = load_seed_level()
    boot = load_bootstrap()

    welfare_rows = [r for r in rows if r["condition"] in WELFARE_CONDITIONS]
    fig_interaction_by_seed(welfare_rows, "I_Umin", "I_Umin", "Figure 1: Seed-level U_min interaction (WSC vs Original), by condition",
                             "fig1_umin_interaction_by_seed")
    fig_interaction_by_seed(welfare_rows, "I_Gini", "I_Gini", "Figure 2: Seed-level Utility Gini interaction (WSC vs Original), by condition",
                             "fig2_gini_interaction_by_seed")
    fig_paired_task_metric(rows, "completion", "Figure 3: Original vs WSC paired Completion by condition", "fig3_completion_paired")
    fig_paired_task_metric(rows, "collision", "Figure 4: Original vs WSC paired Collision by condition", "fig4_collision_paired")
    fig_ci_forest(boot)


if __name__ == "__main__":
    main()
