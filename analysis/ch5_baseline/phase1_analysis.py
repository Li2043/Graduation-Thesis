"""Phase 1 analysis for 05_results_latest.md: everything computable from the
existing Baseline (taskonly_evaluation_merged.csv) and Mean/GGI/Maximin
(pooled12_welfare_evaluation_merged.csv) episode-level CSVs, no new
evaluation infrastructure needed. Read-only against those two files;
writes only under ch5_baseline/outputs/ and to the thesis figures dir.
"""
from __future__ import annotations
import os
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_CSV = Path(__file__).resolve().parent.parent / "data" / "taskonly_evaluation_merged.csv"
POOLED_CSV = Path(__file__).resolve().parent.parent / "pooled12" / "outputs" / "pooled12_welfare_evaluation_merged.csv"
OUT = Path(__file__).resolve().parent / "outputs"
FIG_DIR = Path(os.environ.get("THESIS_FIGURES_DIR", str(Path(__file__).resolve().parent / "outputs" / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                      "figure.dpi": 150, "savefig.dpi": 150})

SEEDS12 = [900101, 900102, 900103, 900104, 910101, 910102,
           920101, 920102, 920103, 920104, 920105, 920106]
SEEDS_ORIG, SEEDS_NEW = SEEDS12[:6], SEEDS12[6:]
CLASS_ORDER = ["ramp-fast", "ramp-slow", "mainline-fast", "mainline-slow"]
CLASS_LABELS = ["Ramp-Fast", "Ramp-Slow", "Mainline-Fast", "Mainline-Slow"]
VIDS = ["V0", "V1", "V2", "V3"]
COND_COLORS = {"baseline": "#7f7f7f", "mean": "#1f77b4", "ggi": "#2ca02c", "maximin": "#d62728"}
COND_LABELS = {"baseline": "Baseline", "mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}
CONDS4 = ["baseline", "mean", "ggi", "maximin"]

base = pd.read_csv(BASE_CSV)
pooled = pd.read_csv(POOLED_CSV)
base["condition"] = "baseline"
df = pd.concat([base, pooled], ignore_index=True)
df.to_csv(OUT / "all4_conditions_merged.csv", index=False)
print(f"loaded: baseline={len(base)} pooled={len(pooled)} total={len(df)}")

REPORT = []
def log(msg=""):
    print(msg); REPORT.append(str(msg))

def gini(values):
    n = len(values); total = sum(values)
    if total == 0: return None
    num = sum(abs(a - b) for a in values for b in values)
    return float(num / (2.0 * n * total))

def bootstrap_ci_paired(diffs, n_boot=10000, seed0=0):
    rng = np.random.default_rng(seed0); n = len(diffs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n); boot[i] = np.mean(diffs[idx])
    return float(np.mean(diffs)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def bootstrap_ci_single(values, n_boot=10000, seed0=0):
    rng = np.random.default_rng(seed0); n = len(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n); boot[i] = np.mean(values[idx])
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def seed_mean(sub, field, seed):
    v = sub[sub.seed == seed][field]
    return float(v.mean()) if len(v) else float("nan")

# ===========================================================================
# 5.4.2 -- Table 5.2 (4-condition competence) + Figure 5.3
# ===========================================================================
log("\n" + "="*78); log("5.4.2 -- Table 5.2: four-condition task competence, H1"); log("="*78)
table52 = []
appendix_rows = {s: {"seed": s} for s in SEEDS12}
for cond in CONDS4:
    sub = df[(df.condition == cond) & (df.bank == "H1")]
    comp = np.array([seed_mean(sub, "completion", s) for s in SEEDS12])
    coll = np.array([seed_mean(sub, "collision", s) for s in SEEDS12])
    to = np.array([seed_mean(sub, "timeout", s) for s in SEEDS12])
    npass = sum(1 for c, k, t in zip(comp, coll, to) if c >= 0.90 and k <= 0.05 and t <= 0.05)
    table52.append({"condition": COND_LABELS[cond], "mean_completion": round(comp.mean(),4),
                     "mean_collision": round(coll.mean(),4), "mean_timeout": round(to.mean(),4),
                     "seeds_meeting_competence": f"{npass}/12"})
    log(f"{cond:8s}: completion={comp.mean():.3f} collision={coll.mean():.3f} timeout={to.mean():.3f} pass={npass}/12")
    for s, c, k, t in zip(SEEDS12, comp, coll, to):
        appendix_rows[s][f"{cond}_completion"] = round(c,4)
        appendix_rows[s][f"{cond}_collision"] = round(k,4)
        appendix_rows[s][f"{cond}_timeout"] = round(t,4)
pd.DataFrame(table52).to_csv(OUT / "table5_2.csv", index=False)
pd.DataFrame(list(appendix_rows.values())).to_csv(OUT / "table5_2_appendix_per_seed.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 5.8))
x = np.arange(len(SEEDS12))
for cond in CONDS4:
    sub = df[(df.condition == cond) & (df.bank == "H1")]
    vals = [seed_mean(sub, "completion", s) for s in SEEDS12]
    ax.plot(x, vals, marker="o", markersize=8, linewidth=1.3, color=COND_COLORS[cond], label=COND_LABELS[cond], alpha=0.9)
ax.axhline(0.90, color="green", linestyle=":", linewidth=1.3, alpha=0.8)
ax.text(11.3, 0.905, "0.90", fontsize=8, color="green", ha="right")
ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEEDS12], rotation=25)
ax.set_xlabel("Formal seed"); ax.set_ylabel("Completion rate (H1)")
ax.set_ylim(-0.02, 1.08)
ax.set_title("Task completion by formal seed and condition (Baseline, Mean, GGI, Maximin)")
ax.legend(fontsize=9.5); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_3_formal_completion_by_seed.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_3_formal_completion_by_seed.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_3_formal_completion_by_seed")

# ===========================================================================
# 5.5.1 -- Table 5.3: Baseline-only competence
# ===========================================================================
log("\n" + "="*78); log("5.5.1 -- Table 5.3: Baseline H1 competence"); log("="*78)
b_h1 = df[(df.condition == "baseline") & (df.bank == "H1")]
table53 = []
for metric in ("completion", "collision", "timeout"):
    vals = np.array([seed_mean(b_h1, metric, s) for s in SEEDS12])
    m, lo, hi = bootstrap_ci_single(vals)
    table53.append({"metric": metric, "mean": round(m,4), "median": round(float(np.median(vals)),4),
                     "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"{metric:12s}: mean={m:.3f} median={np.median(vals):.3f} CI=[{lo:.3f},{hi:.3f}]")
pd.DataFrame(table53).to_csv(OUT / "table5_3.csv", index=False)
weak_seeds = [(s, seed_mean(b_h1,"completion",s)) for s in SEEDS12 if seed_mean(b_h1,"completion",s) < 0.90]
log(f"weak Baseline seeds (<0.90 completion): {weak_seeds}")

# ===========================================================================
# 5.5.2 -- Table 5.4: within-episode inequality, Baseline H1 + Figure 5.4a/b
# ===========================================================================
log("\n" + "="*78); log("5.5.2 -- Table 5.4: within-episode inequality, Baseline H1"); log("="*78)

def episode_metrics_row(row):
    u = [row[f"U_{v}"] for v in VIDS]
    return {"U_mean": sum(u)/4.0, "U_min": min(u), "utility_gini": gini(u), "utility_range": max(u)-min(u)}

b_h1 = b_h1.copy()
em = b_h1.apply(episode_metrics_row, axis=1, result_type="expand")
b_h1 = pd.concat([b_h1, em], axis=1)

table54 = []
seedlevel54 = {}
for metric in ("U_mean", "U_min", "utility_gini", "utility_range"):
    vals = np.array([b_h1[(b_h1.seed==s)][metric].mean() for s in SEEDS12])
    seedlevel54[metric] = vals
    m, lo, hi = bootstrap_ci_single(vals)
    table54.append({"metric": metric, "mean": round(m,4), "median": round(float(np.median(vals)),4),
                     "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"{metric:14s}: mean={m:.3f} median={np.median(vals):.3f} CI=[{lo:.3f},{hi:.3f}]")
pd.DataFrame(table54).to_csv(OUT / "table5_4.csv", index=False)

for metric, fname, ylabel in [("utility_gini","fig5_4a_baseline_gini_by_seed","Utility Gini (Baseline, H1)"),
                                ("U_min","fig5_4b_baseline_umin_by_seed","Worst-off utility $U_{\\min}$ (Baseline, H1)")]:
    vals = seedlevel54[metric]
    m, lo, hi = bootstrap_ci_single(vals)
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(SEEDS12))
    ax.scatter(x, vals, color="#7f7f7f", s=60, zorder=3, edgecolor="black", linewidth=0.6)
    ax.axhline(m, color="#d62728", linewidth=1.5, label=f"12-seed mean = {m:.3f}")
    ax.axhspan(lo, hi, color="#d62728", alpha=0.12, label=f"95% CI [{lo:.3f},{hi:.3f}]")
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEEDS12], rotation=25)
    ax.set_xlabel("Formal seed"); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {fname}")

# ===========================================================================
# 5.5.3 -- Table 5.5: systematic class disadvantage, Baseline H1 + Fig 5.5/5.6
# ===========================================================================
log("\n" + "="*78); log("5.5.3 -- Table 5.5: class disadvantage, Baseline H1"); log("="*78)

class_util_by_seed = defaultdict(dict)
for s in SEEDS12:
    ssub = b_h1[b_h1.seed == s]
    u_by_class = defaultdict(list)
    for _, r in ssub.iterrows():
        for v in VIDS:
            cls = f"{r[f'role_{v}']}-{r[f'speed_class_{v}']}"
            u_by_class[cls].append(r[f"U_{v}"])
    for c in CLASS_ORDER:
        class_util_by_seed[s][c] = float(np.mean(u_by_class[c]))

panelA = []
for c, lab in zip(CLASS_ORDER, CLASS_LABELS):
    vals = np.array([class_util_by_seed[s][c] for s in SEEDS12])
    m, lo, hi = bootstrap_ci_single(vals)
    panelA.append({"class_or_contrast": lab, "estimate": round(m,4), "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"{lab:16s}: U={m:.3f} CI=[{lo:.3f},{hi:.3f}]")

fast_vals = np.array([(class_util_by_seed[s]["ramp-fast"]+class_util_by_seed[s]["mainline-fast"])/2 for s in SEEDS12])
slow_vals = np.array([(class_util_by_seed[s]["ramp-slow"]+class_util_by_seed[s]["mainline-slow"])/2 for s in SEEDS12])
ramp_vals = np.array([(class_util_by_seed[s]["ramp-fast"]+class_util_by_seed[s]["ramp-slow"])/2 for s in SEEDS12])
main_vals = np.array([(class_util_by_seed[s]["mainline-fast"]+class_util_by_seed[s]["mainline-slow"])/2 for s in SEEDS12])
for name, diffs in [("Fast-Slow", fast_vals-slow_vals), ("Ramp-Mainline", ramp_vals-main_vals)]:
    m, lo, hi = bootstrap_ci_paired(diffs)
    panelA.append({"class_or_contrast": name, "estimate": round(m,4), "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"{name:16s}: delta={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
pd.DataFrame(panelA).to_csv(OUT / "table5_5_panelA.csv", index=False)

# worst-off identity, fractional tie credit, 4-way ties excluded
worst_off_by_seed = {}
n_nondegenerate_total = 0
for s in SEEDS12:
    ssub = b_h1[b_h1.seed == s]
    frac = defaultdict(float); n_nd = 0
    for _, r in ssub.iterrows():
        us = {v: r[f"U_{v}"] for v in VIDS}
        mn = min(us.values())
        tied = [v for v, u in us.items() if abs(u-mn) < 1e-6]
        if len(tied) == 4: continue
        n_nd += 1
        w = 1.0/len(tied)
        for v in tied:
            cls = f"{r[f'role_{v}']}-{r[f'speed_class_{v}']}"
            frac[cls] += w
    worst_off_by_seed[s] = {c: (frac[c]/n_nd if n_nd else float("nan")) for c in CLASS_ORDER}
    worst_off_by_seed[s]["_n_nd"] = n_nd
    n_nondegenerate_total += n_nd

panelB = []
for c, lab in zip(CLASS_ORDER, CLASS_LABELS):
    vals = np.array([worst_off_by_seed[s][c] for s in SEEDS12])
    m, lo, hi = bootstrap_ci_single(vals)
    panelB.append({"class": lab, "mean_seed_share": round(m,4), "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"worst-off {lab:16s}: share={m:.3f} CI=[{lo:.3f},{hi:.3f}]")
fast_share = np.array([worst_off_by_seed[s]["ramp-fast"]+worst_off_by_seed[s]["mainline-fast"] for s in SEEDS12])
slow_share = np.array([worst_off_by_seed[s]["ramp-slow"]+worst_off_by_seed[s]["mainline-slow"] for s in SEEDS12])
mf, flo, fhi = bootstrap_ci_single(fast_share); ms, slo, shi = bootstrap_ci_single(slow_share)
log(f"Fast combined worst-off share: {mf:.3f} CI=[{flo:.3f},{fhi:.3f}] (25% uniform reference x2 classes = 50%)")
log(f"Slow combined worst-off share: {ms:.3f} CI=[{slo:.3f},{shi:.3f}]")
log(f"total non-degenerate episodes across 12 seeds: {n_nondegenerate_total}")
pd.DataFrame(panelB).to_csv(OUT / "table5_5_panelB.csv", index=False)

fig, ax = plt.subplots(figsize=(8.5,5.5))
x = np.arange(len(CLASS_ORDER)); width=0.6
means = [np.mean([class_util_by_seed[s][c] for s in SEEDS12]) for c in CLASS_ORDER]
for i, c in enumerate(CLASS_ORDER):
    vals = [class_util_by_seed[s][c] for s in SEEDS12]
    ax.scatter([i]*len(vals), vals, color="#1f77b4", alpha=0.6, s=45, zorder=3)
ax.scatter(range(4), means, color="#d62728", marker="D", s=90, zorder=4, label="12-seed mean")
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Individual utility $U_i$ (Baseline, H1)")
ax.set_title("Individual utility by role-speed class, all seeds (Baseline)")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_5_baseline_utility_by_class.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_5_baseline_utility_by_class.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_5_baseline_utility_by_class")

fig, ax = plt.subplots(figsize=(8,5.3))
vals = [100*np.mean([worst_off_by_seed[s][c] for s in SEEDS12]) for c in CLASS_ORDER]
bars = ax.bar(x, vals, width=width, color="#7f7f7f", edgecolor="black", linewidth=0.8)
for xi, v in zip(x, vals):
    ax.text(xi, v+0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.axhline(25, color="green", linestyle=":", linewidth=1.3)
ax.text(3.5, 26, "25% (uniform reference)", fontsize=8, color="green", ha="right")
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Share of non-degenerate episodes worst-off (%)")
ax.set_title(f"Worst-off vehicle class, Baseline H1, tie-corrected (n={n_nondegenerate_total} non-degenerate episodes)")
ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_6_baseline_worst_off_share.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_6_baseline_worst_off_share.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_6_baseline_worst_off_share")

print("\n\nPHASE1_PART1_DONE")
with open(OUT / "phase1_part1_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT))
