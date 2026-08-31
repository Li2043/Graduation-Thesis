"""Figures for the whole-thesis synthesis (isolated output dir)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "outputs" / "whole_thesis_evidence_synthesis_v1"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load(name: str) -> list[dict]:
    return list(csv.DictReader(open(OUT / name, encoding="utf-8")))


def fig_aliasing() -> None:
    rows = [r for r in _load("reward_observation_aliasing_summary.csv") if r["k"] == "25"]
    labels = ["Maximin\n(Orig)", "Maximin+DWS\n(Orig)", "Maximin+WSC", "Maximin+WSC+DWS"]
    cells = ["cell1", "cell2", "cell3", "cell4"]
    traffic, welfare = [], []
    for c in cells:
        t = next(r for r in rows if r["cell"] == c and r["state"] == "traffic_4d")
        w = next(r for r in rows if r["cell"] == c and r["state"] == "welfare_aug_8d")
        traffic.append(float(t["mean_sign_disagreement"]))
        welfare.append(float(w["mean_sign_disagreement"]))
    x = np.arange(len(labels))
    wbar = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - wbar / 2, traffic, wbar, label="Traffic proxy (x only)", color="#4C72B0")
    ax.bar(x + wbar / 2, welfare, wbar, label="Welfare-augmented (x + M)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("k-NN sign-disagreement rate (k=25)")
    ax.set_title("Reconstructed-state DWS-sign aliasing (proxy; not 18D/22D obs)")
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "aliasing_original_vs_wsc_proxy.png", dpi=160)
    plt.close(fig)


def fig_ordering() -> None:
    rows = _load("condition_absolute_means.csv")
    want = ["Baseline", "Mean", "GGI", "Maximin"]
    umin = [float(next(r for r in rows if r["condition_label"] == c)["u_min"]) for c in want]
    gini = [float(next(r for r in rows if r["condition_label"] == c)["gini"]) for c in want]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].plot(want, umin, "o-", color="#4C72B0")
    axes[0].set_ylabel(r"$U_{\min}$")
    axes[0].set_title("Original: worst-off utility")
    axes[1].plot(want, gini, "o-", color="#C44E52")
    axes[1].set_ylabel("Utility Gini")
    axes[1].set_title("Original: Utility Gini")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.suptitle("No monotonic fairness gain as inequality aversion increases", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "objective_strength_vs_fairness_original.png", dpi=160)
    plt.close(fig)


def fig_dws_fairness() -> None:
    rows = _load("condition_absolute_means.csv")
    labs = ["Maximin", "Maximin+DWS", "Maximin+WSC", "Maximin+WSC+DWS"]
    umin = [float(next(r for r in rows if r["condition_label"] == c)["u_min"]) for c in labs]
    gini = [float(next(r for r in rows if r["condition_label"] == c)["gini"]) for c in labs]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    axes[0].bar(labs, umin, color=colors)
    axes[0].set_ylabel(r"$U_{\min}$")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labs, gini, color=colors)
    axes[1].set_ylabel("Utility Gini")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Four-cell Maximin means (12 seeds, H1)")
    fig.tight_layout()
    fig.savefig(FIG / "four_cell_maximin_absolute.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_aliasing()
    fig_ordering()
    fig_dws_fairness()
    print("figures written to", FIG)
