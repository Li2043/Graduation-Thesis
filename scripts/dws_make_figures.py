"""DWS final re-evaluation -- Section 23 figures D1-D5. Reads only the
already-produced CSVs under outputs/dws_final_reevaluation_v1/. Read-only,
no new computation beyond simple aggregation for plotting.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEEDS = ["900101", "900102", "900103", "900104", "910101", "910102",
         "920101", "920102", "920103", "920104", "920105", "920106"]

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / {name}.pdf")


def read_csv(name):
    return list(csv.DictReader(open(OUT / name, encoding="utf-8")))


# ---------------------------------------------------------------- D1
def fig_d1():
    rows = read_csv("dws_primary_fairness_summary.csv")
    seed_rows = read_csv("dws_primary_fairness_seed_effects.csv")
    labels = ["Cell2-Cell1\nU_min (Original)", "Cell4-Cell3\nU_min (WSC)",
              "Cell2-Cell1\nGini (Original)", "Cell4-Cell3\nGini (WSC)"]
    keys = [("U_min", "Original DWS effect"), ("U_min", "WSC DWS effect"),
            ("Gini", "Original DWS effect"), ("Gini", "WSC DWS effect")]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (outcome, contrast) in enumerate(keys):
        r = next(r for r in rows if r["outcome"] == outcome and r["contrast"] == contrast)
        mean, lo, hi = float(r["mean_effect"]), float(r["ci_lower"]), float(r["ci_upper"])
        sig = float(r["holm_p"]) < 0.05
        color = "#c0392b" if sig else "#7f8c8d"
        ax.errorbar([mean], [i], xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=4, zorder=3)
        seed_vals = [float(s["effect"]) for s in seed_rows if s["outcome"] == outcome and s["contrast"] == contrast]
        jitter = np.linspace(-0.18, 0.18, len(seed_vals))
        ax.scatter(seed_vals, [i + j for j in jitter], color=color, alpha=0.25, s=10, zorder=1)
        ax.text(hi + 0.02, i, f"Holm p={float(r['holm_p']):.3f}", va="center", fontsize=7, color=color)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Effect (positive = favourable for U_min, unfavourable for Gini)")
    ax.set_title("D1 — Primary DWS fairness effects (12-seed paired bootstrap, 95% CI)")
    save(fig, "D1_primary_fairness_forest")


# ---------------------------------------------------------------- D2
def fig_d2():
    sl = read_csv("dws_final_seed_level_metrics.csv")
    cells = ["cell1", "cell2", "cell3", "cell4"]
    cell_labels = {"cell1": "Maximin", "cell2": "Maximin\n+DWS", "cell3": "Maximin\n+WSC", "cell4": "Maximin\n+WSC+DWS"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, title in ((axes[0], "u_min", "U_min"), (axes[1], "gini", "Utility Gini")):
        data = [[float(r[metric]) for r in sl if r["cell"] == c] for c in cells]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for i, c in enumerate(cells):
            x = np.random.default_rng(0).normal(i + 1, 0.04, size=len(data[i]))
            ax.scatter(x, data[i], s=12, alpha=0.6, color="#2c3e50", zorder=3)
        ax.set_xticks(range(1, len(cells) + 1)); ax.set_xticklabels([cell_labels[c] for c in cells])
        ax.set_title(title)
    fig.suptitle("D2 — Four-cell seed-level fairness outcomes (12 seeds each)")
    save(fig, "D2_four_cell_outcomes")


# ---------------------------------------------------------------- D3
def fig_d3():
    rows = read_csv("dws_behavioural_mechanisms_summary.csv")
    mechs = ["welfare_responsive_yielding (RY)", "merge_priority_allocation",
             "cooperative_burden_transfer", "worst_off_recovery_k25"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ypos = 0
    yticks, ylabels = [], []
    for mech in mechs:
        for info in ("Original", "WSC"):
            r = next(r for r in rows if r["mechanism"] == mech and r["information_regime"] == info)
            yticks.append(ypos); ylabels.append(f"{mech.split(' (')[0]}\n({info})")
            if r["mean_effect"] in (None, "", "None"):
                ax.text(0, ypos, "too sparse (n<3 finite seeds)", va="center", fontsize=7, color="#95a5a6")
            else:
                mean, lo, hi = float(r["mean_effect"]), float(r["ci_lower"]), float(r["ci_upper"])
                holm = r["holm_p"]
                sig = holm not in (None, "", "None") and float(holm) < 0.05
                color = "#c0392b" if sig else "#7f8c8d"
                ax.errorbar([mean], [ypos], xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=4)
            ypos += 1
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Effect (Cell2-Cell1 / Cell4-Cell3)")
    ax.set_title("D3 — DWS behavioural mechanisms (none Holm-significant)")
    save(fig, "D3_behavioural_mechanisms")


# ---------------------------------------------------------------- D4
def fig_d4():
    rows = read_csv("dws_signal_timing_profile.csv")
    cells = ["cell1", "cell2", "cell3", "cell4"]
    cell_labels = {"cell1": "Maximin", "cell2": "Maximin+DWS", "cell3": "Maximin+WSC", "cell4": "Maximin+WSC+DWS"}
    colors = {"cell1": "#3498db", "cell2": "#e74c3c", "cell3": "#2ecc71", "cell4": "#9b59b6"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cell in cells:
        for seed in SEEDS:
            sub = sorted([r for r in rows if r["cell"] == cell and r["seed"] == seed], key=lambda r: int(r["progress_bin"]))
            xs = [int(r["progress_bin"]) for r in sub]
            ys = [float(r["mean_min_running_M"]) if r["mean_min_running_M"] not in (None, "", "None") else np.nan for r in sub]
            ax.plot(xs, ys, color=colors[cell], alpha=0.15, lw=0.8)
        # mean across seeds, per bin
        means = []
        for b in range(10):
            vals = [float(r["mean_min_running_M"]) for r in rows if r["cell"] == cell and int(r["progress_bin"]) == b
                    and r["mean_min_running_M"] not in (None, "", "None")]
            means.append(sum(vals) / len(vals) if vals else np.nan)
        ax.plot(range(10), means, color=colors[cell], lw=2.5, label=cell_labels[cell])
    ax.set_xlabel("Normalized episode progress (bin 0-9)")
    ax.set_ylabel("Minimum running welfare M_i(t) across active vehicles")
    ax.set_title("D4 — Running (minimum) welfare over episode progress\n(faint lines = individual seeds, bold = seed mean; heterogeneity preserved)")
    ax.legend(fontsize=7)
    save(fig, "D4_running_welfare_dynamics")


# ---------------------------------------------------------------- D5
def fig_d5():
    prim = read_csv("dws_primary_fairness_seed_effects.csv")
    mech = read_csv("dws_behavioural_mechanisms_seed_level.csv")
    sl = read_csv("dws_final_seed_level_metrics.csv")

    def get_sl(cell, seed, key):
        return float(next(r for r in sl if r["cell"] == cell and r["seed"] == seed)[key])

    def ry_diff(s):
        a = next(r["RY"] for r in mech if r["cell"] == "cell2" and r["seed"] == s)
        b = next(r["RY"] for r in mech if r["cell"] == "cell1" and r["seed"] == s)
        if a in ("", "None", None) or b in ("", "None", None):
            return np.nan
        return float(a) - float(b)

    metrics_orig = {
        "U_min (Original)": {s: next(float(r["effect"]) for r in prim if r["outcome"] == "U_min" and r["contrast"] == "Original DWS effect" and r["seed"] == s) for s in SEEDS},
        "Completion (Original)": {s: get_sl("cell2", s, "completion") - get_sl("cell1", s, "completion") for s in SEEDS},
        "RY (Original)": {s: ry_diff(s) for s in SEEDS},
    }

    row_names = list(metrics_orig.keys())
    mat = np.array([[metrics_orig[r][s] for s in SEEDS] for r in row_names], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(SEEDS))); ax.set_xticklabels(SEEDS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_names))); ax.set_yticklabels(row_names, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Cell2-Cell1 effect")
    ax.set_title("D5 — Seed heterogeneity: Original-regime DWS effect by seed\n(fairness, task, and behavioural indicators)")
    save(fig, "D5_seed_heterogeneity_heatmap")


def main():
    print("[dws_make_figures] generating D1-D5 ...")
    fig_d1(); fig_d2(); fig_d3(); fig_d4(); fig_d5()
    print("done")


if __name__ == "__main__":
    main()
