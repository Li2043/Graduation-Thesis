#!/usr/bin/env python3
"""Replication figures per new_protocol.md §40–§42.

Style matches analysis_scripts/make_rq_figures.py (formal U_min-by-seed plot).
Writes PNG+PDF under outputs/seed_replication_v1/analysis/figures/.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BUNDLE_ROOT, OUTPUTS  # noqa: E402
from replication_common import SEEDS  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 150,
})

FIG_DIR = OUTPUTS / "seed_replication_v1" / "analysis" / "figures"
WELFARE_CSV = OUTPUTS / "seed_replication_v1" / "welfare_eval" / "replication_welfare_evaluation_merged.csv"
UMIN_CSV = BUNDLE_ROOT / "new_seed_umin_contrasts.csv"
TAIL_CSV = BUNDLE_ROOT / "new_seed_tail_burden.csv"
BEHAV_CSV = BUNDLE_ROOT / "new_seed_behavioral_diagnostic.csv"

COND_COLORS = {"mean": "#1f77b4", "ggi": "#2ca02c", "maximin": "#d62728"}
COND_LABELS = {"mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}
CONDS = ("mean", "ggi", "maximin")
SEED_LIST = list(SEEDS)


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / name}.png|.pdf")


def fig40_umin_by_seed() -> None:
    rows = [r for r in load(UMIN_CSV) if r["seed"] != "MEAN"]
    by_seed = {int(r["seed"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(SEED_LIST))
    keys = {"mean": "U_min_Mean", "ggi": "U_min_GGI", "maximin": "U_min_Maximin"}
    for cond in CONDS:
        vals = [float(by_seed[s][keys[cond]]) for s in SEED_LIST]
        ax.plot(x, vals, marker="o", markersize=8, linewidth=1.5,
                color=COND_COLORS[cond], label=COND_LABELS[cond], alpha=0.9)
    for s_i in x:
        ax.axvline(s_i, color="gray", alpha=0.08, linewidth=6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEED_LIST], rotation=20)
    ax.set_xlabel("Training seed")
    ax.set_ylabel(r"Seed-mean worst-off utility $U_{\min}$ (H1)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Worst-off utility by new training seed and welfare condition\n"
                 "(matched-seed replication, checkpoint-Q ensemble, H1)")
    ax.legend(fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    save(fig, "fig_replication_umin_by_seed")


def fig41_burden_tail() -> None:
    episode = load(WELFARE_CSV)
    tail = [r for r in load(TAIL_CSV) if r["bank"] == "H1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # Left: vehicle-level C_i distribution on H1 (all episodes × 4 vehicles)
    data, labels = [], []
    for cond in CONDS:
        cs = []
        for r in episode:
            if r["condition"] != cond or r["bank"] != "H1":
                continue
            for v in ("V0", "V1", "V2", "V3"):
                cs.append(float(r[f"C_{v}"]))
        data.append(cs)
        labels.append(COND_LABELS[cond])
    bp = axes[0].boxplot(
        data, tick_labels=labels, showfliers=True, whis=(5, 95),
        medianprops={"color": "black", "linewidth": 1.4},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.25},
        patch_artist=True,
    )
    for patch, cond in zip(bp["boxes"], CONDS):
        patch.set_facecolor(COND_COLORS[cond])
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")
    c_max = max(max(cs) if cs else 0 for cs in data)
    axes[0].set_ylim(0, 2.5)
    axes[0].text(
        0.02, 0.98,
        f"y clipped at 2.5; Maximin max $C_i$={c_max:.1f}\n(see $C_{{95}}$ panel for seed tails)",
        transform=axes[0].transAxes, va="top", ha="left", fontsize=7.5, color="dimgray",
    )
    axes[0].set_ylabel("Vehicle-level mobility burden $C_i$ (H1)")
    axes[0].set_title("Burden distribution\n(boxes 5–95th pct; dots = outliers)")
    axes[0].grid(True, axis="y", alpha=0.25, linewidth=0.6)

    # Right: seed-level C95
    x = np.arange(len(CONDS))
    for i, seed in enumerate(SEED_LIST):
        vals = []
        for cond in CONDS:
            rec = next(r for r in tail if int(r["seed"]) == seed and r["condition"] == cond)
            vals.append(float(rec["C95_unconditional"]))
        axes[1].plot(x, vals, marker="o", markersize=6, linewidth=1.2, alpha=0.85,
                     label=str(seed))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([COND_LABELS[c] for c in CONDS])
    axes[1].set_xlabel("Welfare condition")
    axes[1].set_ylabel(r"Seed-level $C_{95}$ (H1, vehicle-level)")
    axes[1].set_title("Upper-tail burden $C_{95}$\n(one polyline per training seed)")
    axes[1].legend(fontsize=7.5, title="Seed", ncol=2)
    axes[1].grid(True, axis="y", alpha=0.25, linewidth=0.6)

    fig.suptitle("Mobility-burden tail by welfare condition (new six seeds, H1)", y=1.02, fontsize=12)
    save(fig, "fig_replication_burden_tail")


def fig42_behavioral() -> None:
    rows = load(BEHAV_CSV)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    x = np.arange(len(CONDS))
    width = 0.12
    for i, seed in enumerate(SEED_LIST):
        vals = []
        for cond in CONDS:
            rec = next(r for r in rows if int(r["seed"]) == seed and r["condition"] == cond)
            vals.append(100.0 * float(rec["high_burden_goes_before_conflict_rate"]))
        offset = (i - (len(SEED_LIST) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.92, edgecolor="black", linewidth=0.5,
               label=str(seed), alpha=0.9)
    # condition means as black markers
    means = []
    for cond in CONDS:
        vs = [100.0 * float(r["high_burden_goes_before_conflict_rate"])
              for r in rows if r["condition"] == cond]
        means.append(float(np.mean(vs)))
    ax.plot(x, means, color="black", marker="D", markersize=7, linewidth=1.2,
            linestyle="none", label="condition mean", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDS])
    ax.set_xlabel("Welfare condition")
    ax.set_ylabel("P(high-burden vehicle merges before conflict) (%)")
    ax.set_ylim(0, 70)
    ax.set_title("High-burden vehicle later merge priority\n"
                 "(exploratory behavioural diagnostic, H1; bars = seeds)")
    ax.legend(fontsize=7.5, ncol=4)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    save(fig, "fig_replication_behavioral")


def main() -> int:
    fig40_umin_by_seed()
    fig41_burden_tail()
    fig42_behavioral()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
