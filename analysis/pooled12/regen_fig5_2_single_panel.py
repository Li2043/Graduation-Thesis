import os
"""Regenerate fig5_2 as a single panel (C64 qualification gate, 4 seeds only).
Drops the old right-hand "Mean qualification" panel per user request: the
Mean-only qualification isn't parallel to the 4-seed C64 gate (only Mean was
tested, not GGI/Maximin), so it doesn't belong side-by-side with it.
Overwrites the actually-published filename (fig5_2_qualification_seed_bars),
not the stale fig5_2_qualification_boxplots name the original script used.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path(os.environ.get("THESIS_FIGURES_DIR", str(Path(__file__).resolve().parent / "outputs" / "figures")))
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 150})

c64_seeds = ["900101", "900102", "900103", "900104"]
c64_metrics = {
    "Completion": [0.984, 1.000, 0.453, 1.000],
    "Collision": [0.016, 0.000, 0.547, 0.000],
    "Timeout": [0.000, 0.000, 0.000, 0.000],
}
METRIC_COLORS = {"Completion": "#1f77b4", "Collision": "#d62728", "Timeout": "#7f7f7f"}

fig, ax = plt.subplots(figsize=(8, 5.5))
n_seeds = len(c64_seeds)
n_metrics = len(c64_metrics)
group_width = 0.72
bar_w = group_width / n_metrics
x = np.arange(n_seeds)
for j, (metric, vals) in enumerate(c64_metrics.items()):
    offset = (j - (n_metrics - 1) / 2) * bar_w
    ax.bar(x + offset, vals, width=bar_w * 0.92, color=METRIC_COLORS[metric],
           edgecolor="black", linewidth=0.8, label=metric, zorder=3)
    for xi, v in zip(x + offset, vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
ax.axhline(0.90, color="green", linestyle=":", linewidth=1.4, alpha=0.85, zorder=2)
ax.text(3.5, 0.925, "completion criterion (0.90)", fontsize=8.5, color="green", ha="right")
ax.set_xticks(x)
ax.set_xticklabels(c64_seeds)
ax.set_xlabel("Training seed")
ax.set_ylabel("Rate")
ax.set_ylim(0, 1.18)
ax.set_title("C64$_{R50}$ final qualification gate\n(4 seeds, $\\epsilon=0$, 64 held-out scenarios)")
ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9.5, framealpha=0.95, borderaxespad=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_2_qualification_seed_bars.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_2_qualification_seed_bars.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig5_2_qualification_seed_bars.png/.pdf (single panel)")
