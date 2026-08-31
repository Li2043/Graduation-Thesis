"""Pooled n=12 versions of fig5_5/5_6/5_7 (make_rq_figures.py), fig5_10
(make_behavioral_figure.py), and fig5_8/5_9 (baseline_dependence.py), plus a
new fig5_11 reporting the standalone new-6 replication check. All figures are
real matplotlib output computed from the pooled 12-seed episode data written
by merge_and_audit.py -- none are schematic/hand-drawn.

Writes to the live thesis figures directory (overwrites the n=6 fig5_5/5_6/
5_7/5_8/5_9/5_10 files with their n=12 pooled equivalents; adds fig5_11 new).
"""
import os
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(os.environ.get("THESIS_FIGURES_DIR", str(Path(__file__).resolve().parent / "outputs" / "figures")))
OUT_DIR = Path(__file__).resolve().parent / "outputs"
WELFARE_CSV = OUT_DIR / "pooled12_welfare_evaluation_merged.csv"
BEHAV_CSV = OUT_DIR / "pooled12_behavioral_evaluation_merged.csv"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                      "figure.dpi": 150, "savefig.dpi": 150})

SEEDS_ORIG = [900101, 900102, 900103, 900104, 910101, 910102]
SEEDS_NEW = [920101, 920102, 920103, 920104, 920105, 920106]
SEEDS12 = SEEDS_ORIG + SEEDS_NEW
CLASS_ORDER = ["ramp-fast", "ramp-slow", "mainline-fast", "mainline-slow"]
CLASS_LABELS = ["Ramp-Fast", "Ramp-Slow", "Mainline-Fast", "Mainline-Slow"]
VIDS = ["V0", "V1", "V2", "V3"]
COND_COLORS = {"mean": "#1f77b4", "ggi": "#2ca02c", "maximin": "#d62728"}
COND_LABELS = {"mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}

with open(WELFARE_CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
with open(BEHAV_CSV, encoding="utf-8") as f:
    brows = list(csv.DictReader(f))


def seed_mean(rowset, field, seed):
    v = [float(r[field]) for r in rowset if int(r["seed"]) == seed]
    return float(np.mean(v))


def worst_off_fractional(rowset):
    cnt = defaultdict(float)
    n = 0
    for r in rowset:
        us = {v: float(r[f"U_{v}"]) for v in VIDS}
        m = min(us.values())
        tied = [v for v, u in us.items() if abs(u - m) < 1e-9]
        if len(tied) == 4:
            continue
        n += 1
        w = 1.0 / len(tied)
        for v in tied:
            cls = f"{r['role_' + v]}-{r['speed_class_' + v]}"
            cnt[cls] += w
    return cnt, n


# =====================================================================
# fig5_5 -- burden by class x condition (unconditional, n=12)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(CLASS_ORDER))
width = 0.26
for i, cond in enumerate(("mean", "ggi", "maximin")):
    rowset = [r for r in rows if r["condition"] == cond and r["bank"] == "H1"]
    c_by_class = defaultdict(list)
    for r in rowset:
        for vid in VIDS:
            cls = f"{r['role_' + vid]}-{r['speed_class_' + vid]}"
            c_by_class[cls].append(float(r[f"C_{vid}"]))
    vals = [np.mean(c_by_class[c]) for c in CLASS_ORDER]
    offset = (i - 1) * width
    ax.bar(x + offset, vals, width=width * 0.92, color=COND_COLORS[cond], edgecolor="black", linewidth=0.7, label=COND_LABELS[cond])
    for xi, v in zip(x + offset, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Mean coordination burden $C_i$")
ax.set_title("Coordination burden by role-speed class and welfare condition\n(H1, 12 seeds pooled, formal runs)")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_5_formal_burden_by_class.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_5_formal_burden_by_class.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# fig5_6 -- tie-corrected worst-off frequency by class x condition (n=12)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))
for i, cond in enumerate(("mean", "ggi", "maximin")):
    rowset = [r for r in rows if r["condition"] == cond and r["bank"] == "H1"]
    cnt, n = worst_off_fractional(rowset)
    vals = [100 * cnt[c] / n for c in CLASS_ORDER]
    offset = (i - 1) * width
    ax.bar(x + offset, vals, width=width * 0.92, color=COND_COLORS[cond], edgecolor="black", linewidth=0.7, label=f"{COND_LABELS[cond]} (n={n})")
    for xi, v in zip(x + offset, vals):
        ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.axhline(25, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
ax.text(3.55, 26, "25% (uniform baseline)", fontsize=7.5, color="gray", ha="right")
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Share of non-degenerate episodes worst-off (%)")
ax.set_title("Worst-off vehicle class, tie-corrected (12 seeds pooled)\n(episodes with a genuine minimum only; excludes 4-way perfect ties)")
ax.legend(fontsize=8.5); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_6_formal_worst_off_corrected.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_6_formal_worst_off_corrected.pdf", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# fig5_7 -- U_min by seed x condition, 12-seed x-axis (widened + grouped)
# =====================================================================
fig, ax = plt.subplots(figsize=(12.5, 5.8))
x_seed = np.arange(len(SEEDS12))
for cond in ("mean", "ggi", "maximin"):
    rowset = [r for r in rows if r["condition"] == cond and r["bank"] == "H1"]
    vals = [seed_mean(rowset, "min_U", s) for s in SEEDS12]
    ax.plot(x_seed, vals, marker="o", markersize=8, linewidth=1.5, color=COND_COLORS[cond], label=COND_LABELS[cond], alpha=0.9)
for s_i in x_seed:
    ax.axvline(s_i, color="gray", alpha=0.08, linewidth=6, zorder=0)
ax.axvline(5.5, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
ax.text(5.5, 1.07, "original 6  |  new-6 replication", ha="center", va="bottom", fontsize=8.5, color="black")
ax.set_xticks(x_seed); ax.set_xticklabels([str(s) for s in SEEDS12], rotation=25)
ax.set_xlabel("Formal seed (original 6 | independent replication 6)")
ax.set_ylabel("Seed-mean worst-off utility $U_{\\min}$ (H1)")
ax.set_ylim(-0.02, 1.12)
ax.set_title("Worst-off utility by formal seed and welfare condition, n=12 pooled (matched-seed)")
ax.legend(fontsize=9.5); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_7_formal_umin_by_seed.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_7_formal_umin_by_seed.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote fig5_5, fig5_6, fig5_7 (n=12 pooled)")

# =====================================================================
# fig5_10 -- behavioral corroboration, n=12
# =====================================================================
def bsubset(condition, bank):
    return [r for r in brows if r["condition"] == condition and r["bank"] == bank]


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
x = np.arange(len(CLASS_ORDER))

ax = axes[0]
for i, cond in enumerate(("mean", "ggi", "maximin")):
    seedvals_by_class = defaultdict(list)
    for s in SEEDS12:
        srows = [r for r in bsubset(cond, "H1") if int(r["seed"]) == s]
        hb = defaultdict(list)
        for r in srows:
            for v in VIDS:
                cls = f"{r[f'role_{v}']}-{r[f'speed_class_{v}']}"
                hb[cls].append(int(r[f"hard_brake_count_{v}"]))
        for c in CLASS_ORDER:
            seedvals_by_class[c].append(np.mean(hb[c]))
    vals = [np.mean(seedvals_by_class[c]) for c in CLASS_ORDER]
    offset = (i - 1) * width
    ax.bar(x + offset, vals, width=width * 0.92, color=COND_COLORS[cond], edgecolor="black", linewidth=0.7, label=COND_LABELS[cond])
    for xi, v in zip(x + offset, vals):
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Mean hard-brake events per vehicle-episode\n(accel $\\leq -3.0\\ \\mathrm{m/s^2}$)")
ax.set_title("Hard-braking rate by class (12 seeds pooled)")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

ax = axes[1]
for i, cond in enumerate(("mean", "ggi", "maximin")):
    seedvals_by_class = defaultdict(list)
    for s in SEEDS12:
        srows = [r for r in bsubset(cond, "H1") if int(r["seed"]) == s]
        first_count = defaultdict(int); n_valid = 0
        for r in srows:
            mo = r["merge_order"]
            if mo in ("DNF", ""):
                continue
            first_vid = mo.split(">")[0]
            cls = f"{r[f'role_{first_vid}']}-{r[f'speed_class_{first_vid}']}"
            first_count[cls] += 1; n_valid += 1
        for c in CLASS_ORDER:
            seedvals_by_class[c].append(100 * first_count[c] / n_valid if n_valid else np.nan)
    vals = [np.nanmean(seedvals_by_class[c]) for c in CLASS_ORDER]
    offset = (i - 1) * width
    ax.bar(x + offset, vals, width=width * 0.92, color=COND_COLORS[cond], edgecolor="black", linewidth=0.7, label=COND_LABELS[cond])
    for xi, v in zip(x + offset, vals):
        ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.axhline(25, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
ax.text(3.55, 26, "25% (uniform)", fontsize=7.5, color="gray", ha="right")
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Share of episodes where this class crosses first (%)")
ax.set_title("Merge-order priority by class (12 seeds pooled)")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

fig.suptitle("Behavioural corroboration: hard-braking and merge order by role-speed class\n(H1, seed-mean across 12 seeds)", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_10_behavioral_corroboration.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_10_behavioral_corroboration.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig5_10 (n=12 pooled)")

# =====================================================================
# fig5_8 / fig5_9 -- baseline dependence, n=12 (parse both log naming schemes)
# =====================================================================
LOG_RE = re.compile(
    r"step=\s*(\d+)\s+completion=([\d.]+)\s+collision=([\d.]+)\s+timeout=([\d.]+)\s+"
    r"mean_Q\(policy\)=([-\d.]+)\s+mean_Q\(oracle_ref\)=([-\d.]+)"
)


def log_path(cond, seed):
    if seed in SEEDS_ORIG:
        return Path(os.environ.get("FINAL_NEW_BUNDLE", "")) / "logs" / f"formal_{cond}_{seed}.log"
    return Path(os.environ.get("SEED_REPL_BUNDLE", "")) / "logs" / f"replication_welfare_{cond}_{seed}.log"


def parse_log(cond, seed):
    path = log_path(cond, seed)
    points = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_RE.search(line)
        if m:
            step, comp, coll, to, qp, qo = m.groups()
            points.append({"step": int(step), "completion": float(comp), "collision": float(coll),
                            "timeout": float(to), "q_policy": float(qp), "q_oracle": float(qo)})
    return points


def drop_resume_artifact(points):
    if points and points[0]["completion"] == 0.0 and points[0]["collision"] == 0.0 and points[0]["timeout"] == 0.0:
        return points[1:]
    return points


CONDITIONS = ["mean", "ggi", "maximin"]
all_curves = {}
for cond in CONDITIONS:
    for seed in SEEDS12:
        pts = drop_resume_artifact(parse_log(cond, seed))
        all_curves[(cond, seed)] = pts
        print(f"{cond:8s} {seed}: {len(pts)} points")

curve_csv = OUT_DIR / "pooled12_training_window_curves.csv"
with open(curve_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condition", "seed", "step", "completion", "collision", "timeout", "q_policy", "q_oracle"])
    for (cond, seed), pts in all_curves.items():
        for p in pts:
            w.writerow([cond, seed, p["step"], p["completion"], p["collision"], p["timeout"], p["q_policy"], p["q_oracle"]])
print(f"wrote {curve_csv}")

COLORS12 = {
    900101: "#1f77b4", 900102: "#ff7f0e", 900103: "#2ca02c", 900104: "#d62728",
    910101: "#9467bd", 910102: "#8c564b",
    920101: "#e377c2", 920102: "#7f7f7f", 920103: "#bcbd22", 920104: "#17becf",
    920105: "#aec7e8", 920106: "#ffbb78",
}
HIGHLIGHT = (900104, 910102)  # seeds already discussed in the original text

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for ax, cond, title in zip(axes, ("ggi", "maximin"), ("GGI", "Maximin")):
    for seed in SEEDS12:
        pts = all_curves[(cond, seed)]
        if not pts:
            continue
        steps = [p["step"] for p in pts]
        comp = [p["completion"] for p in pts]
        lw = 2.4 if seed in HIGHLIGHT else 1.1
        alpha = 1.0 if seed in HIGHLIGHT else 0.45
        ls = "-" if seed in HIGHLIGHT else "--"
        ax.plot(steps, comp, color=COLORS12[seed], linewidth=lw, alpha=alpha, linestyle=ls, marker="o", markersize=2.6, label=f"seed {seed}")
    ax.set_title(f"{title} fine-tuning (1.2M\u21922.0M steps)")
    ax.set_xlabel("Absolute training step")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_ylim(-0.05, 1.05)
axes[0].set_ylabel("Training-window completion rate")
axes[1].legend(fontsize=7, loc="lower left", bbox_to_anchor=(1.01, 0.0), ncol=1)
fig.suptitle("Training-window completion during welfare fine-tuning, n=12 pooled\n(solid+bold = 900104, 910102, discussed in text; dashed = other 10 seeds)", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_8_welfare_finetuning_curves.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_8_welfare_finetuning_curves.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig5_8 (n=12 pooled)")

with open(WELFARE_CSV, encoding="utf-8") as f:
    wrows = list(csv.DictReader(f))


def seed_mean_field(cond, bank, field, seed):
    vals = [float(r[field]) for r in wrows if r["condition"] == cond and r["bank"] == bank and int(r["seed"]) == seed]
    return float(np.mean(vals))


corr_rows = []
base_comp, d_ggi_comp, d_max_comp = [], [], []
base_umin, d_ggi_umin, d_max_umin = [], [], []
for s in SEEDS12:
    bc = seed_mean_field("mean", "H1", "completion", s)
    gc = seed_mean_field("ggi", "H1", "completion", s) - bc
    mc = seed_mean_field("maximin", "H1", "completion", s) - bc
    bu = seed_mean_field("mean", "H1", "min_U", s)
    gu = seed_mean_field("ggi", "H1", "min_U", s) - bu
    mu = seed_mean_field("maximin", "H1", "min_U", s) - bu
    corr_rows.append({"seed": s, "baseline_completion": bc, "delta_completion_ggi": gc, "delta_completion_maximin": mc,
                       "baseline_umin": bu, "delta_umin_ggi": gu, "delta_umin_maximin": mu})
    base_comp.append(bc); d_ggi_comp.append(gc); d_max_comp.append(mc)
    base_umin.append(bu); d_ggi_umin.append(gu); d_max_umin.append(mu)

r_ggi_comp = float(np.corrcoef(base_comp, d_ggi_comp)[0, 1])
r_max_comp = float(np.corrcoef(base_comp, d_max_comp)[0, 1])
r_ggi_umin = float(np.corrcoef(base_umin, d_ggi_umin)[0, 1])
r_max_umin = float(np.corrcoef(base_umin, d_max_umin)[0, 1])
print(f"Pearson r (n=12), baseline completion vs completion-delta: GGI={r_ggi_comp:.4f}  Maximin={r_max_comp:.4f}")
print(f"Pearson r (n=12), baseline U_min vs U_min-delta:            GGI={r_ggi_umin:.4f}  Maximin={r_max_umin:.4f}")

corr_csv = OUT_DIR / "pooled12_baseline_quality_vs_welfare_delta.csv"
with open(corr_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
    w.writeheader(); w.writerows(corr_rows)
print(f"wrote {corr_csv}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, xfield, y1field, y2field, r_ggi, r_max, ylabel in [
    (axes[0], "baseline_completion", "delta_completion_ggi", "delta_completion_maximin", r_ggi_comp, r_max_comp, "$\\Delta$ completion (condition $-$ Mean)"),
    (axes[1], "baseline_umin", "delta_umin_ggi", "delta_umin_maximin", r_ggi_umin, r_max_umin, "$\\Delta U_{\\min}$ (condition $-$ Mean)"),
]:
    xs = [r[xfield] for r in corr_rows]
    ys_ggi = [r[y1field] for r in corr_rows]
    ys_max = [r[y2field] for r in corr_rows]
    ax.axhline(0, color="gray", linewidth=1, alpha=0.6)
    is_new = [r["seed"] in SEEDS_NEW for r in corr_rows]
    ax.scatter([x for x, n in zip(xs, is_new) if not n], [y for y, n in zip(ys_ggi, is_new) if not n],
               color="#2ca02c", s=70, label=f"GGI, original 6 (r={r_ggi:.2f}, n=12)", zorder=3)
    ax.scatter([x for x, n in zip(xs, is_new) if n], [y for y, n in zip(ys_ggi, is_new) if n],
               color="#2ca02c", s=70, marker="D", edgecolor="black", linewidth=0.8, label="GGI, new-6 replication", zorder=3)
    ax.scatter([x for x, n in zip(xs, is_new) if not n], [y for y, n in zip(ys_max, is_new) if not n],
               color="#d62728", marker="^", s=70, label=f"Maximin, original 6 (r={r_max:.2f}, n=12)", zorder=3)
    ax.scatter([x for x, n in zip(xs, is_new) if n], [y for y, n in zip(ys_max, is_new) if n],
               color="#d62728", marker="^", s=90, edgecolor="black", linewidth=0.8, label="Maximin, new-6 replication", zorder=3)
    for x, y, r in zip(xs, ys_max, corr_rows):
        ax.annotate(str(r["seed"]), (x, y), fontsize=6.5, xytext=(4, 4), textcoords="offset points", color="#d62728")
    ax.set_xlabel("Mean-policy baseline (H1)"); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7.5)
axes[0].set_title("Baseline completion vs. completion change")
axes[1].set_title("Baseline $U_{\\min}$ vs. $U_{\\min}$ change")
fig.suptitle("Welfare fine-tuning delta vs. Mean-policy baseline quality (n=12 pooled; diamonds/triangle-outline = new-6 replication)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_9_baseline_dependence.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_9_baseline_dependence.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig5_9 (n=12 pooled)")

# =====================================================================
# fig5_11 -- NEW: standalone new-6 replication check (U_min, matched-seed)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))
x_seed = np.arange(len(SEEDS_NEW))
for cond in ("mean", "ggi", "maximin"):
    rowset = [r for r in rows if r["condition"] == cond and r["bank"] == "H1"]
    vals = [seed_mean(rowset, "min_U", s) for s in SEEDS_NEW]
    ax.plot(x_seed, vals, marker="o", markersize=9, linewidth=1.8, color=COND_COLORS[cond], label=COND_LABELS[cond])
for s_i in x_seed:
    ax.axvline(s_i, color="gray", alpha=0.08, linewidth=6, zorder=0)
ax.set_xticks(x_seed); ax.set_xticklabels([str(s) for s in SEEDS_NEW], rotation=20)
ax.set_xlabel("Independent replication seed (920101-920106)")
ax.set_ylabel("Seed-mean worst-off utility $U_{\\min}$ (H1)")
ax.set_ylim(-0.02, 1.08)
ax.set_title("Independent replication check: worst-off utility by seed\n(new-6 cohort only, standalone -- not pooled)")
ax.legend(fontsize=9.5); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_11_replication_umin_standalone.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_11_replication_umin_standalone.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig5_11 (new-6 standalone)")
