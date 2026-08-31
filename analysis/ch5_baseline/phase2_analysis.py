"""Phase 2 analysis: 5.5.5 (Baseline braking/coordination), 5.6.3 (mobility/
braking costs, 4 conditions), 5.6.4 (behavioural response, 4 conditions),
5.7.3 (zero-burden diagnostic recompute). Merges the 6 behavioral-window
shards with the existing episode-level welfare CSV (for C_i / burden
cross-reference in 5.7.3)."""
from __future__ import annotations
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BW_DIR = Path(os.environ.get("FINAL_NEW_BUNDLE", "")) / "outputs" / "behavioral_window"
WELFARE_CSV = Path(__file__).resolve().parent / "outputs" / "all4_conditions_merged.csv"
OUT = Path(__file__).resolve().parent / "outputs"
FIG_DIR = Path(os.environ.get("THESIS_FIGURES_DIR", str(Path(__file__).resolve().parent / "outputs" / "figures")))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                      "figure.dpi": 150, "savefig.dpi": 150})

SEEDS12 = [900101, 900102, 900103, 900104, 910101, 910102,
           920101, 920102, 920103, 920104, 920105, 920106]
CLASS_ORDER = ["ramp-fast", "ramp-slow", "mainline-fast", "mainline-slow"]
CLASS_LABELS = ["Ramp-Fast", "Ramp-Slow", "Mainline-Fast", "Mainline-Slow"]
VIDS = ["V0", "V1", "V2", "V3"]
COND_COLORS = {"baseline": "#7f7f7f", "mean": "#1f77b4", "ggi": "#2ca02c", "maximin": "#d62728"}
COND_LABELS = {"baseline": "Baseline", "mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}
CONDS4 = ["baseline", "mean", "ggi", "maximin"]

bw = pd.concat([pd.read_csv(BW_DIR / f"behavioral_window_shard{i}.csv") for i in range(6)], ignore_index=True)
bw.to_csv(OUT / "behavioral_window_merged.csv", index=False)
print(f"loaded behavioral window data: {len(bw)} rows (expect 12288)")
welfare = pd.read_csv(WELFARE_CSV)

REPORT = []
def log(msg=""):
    print(msg); REPORT.append(str(msg))

def bootstrap_ci_paired(diffs, n_boot=10000, seed0=0):
    rng = np.random.default_rng(seed0); n = len(diffs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0,n,n); boot[i] = np.mean(diffs[idx])
    return float(np.mean(diffs)), float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))

def bootstrap_ci_single(values, n_boot=10000, seed0=0):
    rng = np.random.default_rng(seed0); n = len(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0,n,n); boot[i] = np.mean(values[idx])
    return float(np.mean(values)), float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))

def class_of(row, v):
    return f"{row[f'role_{v}']}-{row[f'speed_class_{v}']}"

# ===========================================================================
# 5.5.5 -- Table 5.7: Baseline-only braking/coordination + Fig 5.8a/5.8b
# ===========================================================================
log("="*78); log("5.5.5 -- Table 5.7: Baseline braking/coordination behaviour"); log("="*78)
b = bw[bw.condition=="baseline"].copy()

def seed_class_agg(df_cond, field, agg="mean"):
    """returns {seed: {class: value}}"""
    out = defaultdict(dict)
    for s in SEEDS12:
        ssub = df_cond[df_cond.seed==s]
        vals_by_class = defaultdict(list)
        for _, r in ssub.iterrows():
            for v in VIDS:
                val = r[f"{field}_{v}"]
                if pd.notna(val):
                    vals_by_class[class_of(r,v)].append(val)
        for c in CLASS_ORDER:
            out[s][c] = float(np.mean(vals_by_class[c])) if vals_by_class[c] else float("nan")
    return out

brake_rate = seed_class_agg(b, "brake_rate_window")
c_brake = seed_class_agg(b, "c_brake_window")
hb_win = seed_class_agg(b, "hard_brake_events_window")

table57 = []
for name, data in [("Pre-merge BRAKE-action rate", brake_rate), ("Braking burden C_brake", c_brake), ("Hard-brake events (window)", hb_win)]:
    for c, lab in zip(CLASS_ORDER, CLASS_LABELS):
        vals = np.array([data[s][c] for s in SEEDS12])
        vals = vals[~np.isnan(vals)]
        m, lo, hi = bootstrap_ci_single(vals)
        table57.append({"measure": name, "class": lab, "estimate": round(m,4), "ci_low": round(lo,4), "ci_high": round(hi,4)})
        log(f"{name:28s} {lab:16s}: {m:.4f} CI=[{lo:.4f},{hi:.4f}]")
    fast = np.array([(data[s]["ramp-fast"]+data[s]["mainline-fast"])/2 for s in SEEDS12])
    slow = np.array([(data[s]["ramp-slow"]+data[s]["mainline-slow"])/2 for s in SEEDS12])
    m,lo,hi = bootstrap_ci_paired(fast-slow)
    table57.append({"measure": name, "class": "Fast-Slow", "estimate": round(m,4), "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"{name:28s} {'Fast-Slow':16s}: delta={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")

# first-to-merge share (from merge_order string, fractional handled at step-tie already; here just first entry)
def first_to_merge_share(df_cond):
    per_seed = {}
    for s in SEEDS12:
        ssub = df_cond[df_cond.seed==s]
        share = defaultdict(float); n_valid=0
        for _, r in ssub.iterrows():
            mo = r["merge_order"]
            if mo == "DNF" or not isinstance(mo,str): continue
            first_v = mo.split(">")[0]
            share[class_of(r, first_v)] += 1
            n_valid += 1
        per_seed[s] = {c: (share[c]/n_valid if n_valid else float("nan")) for c in CLASS_ORDER}
        per_seed[s]["_n"] = n_valid
    return per_seed

ftm = first_to_merge_share(b)
for c, lab in zip(CLASS_ORDER, CLASS_LABELS):
    vals = np.array([ftm[s][c] for s in SEEDS12])
    m, lo, hi = bootstrap_ci_single(vals)
    table57.append({"measure":"First-to-merge share","class":lab,"estimate":round(m,4),"ci_low":round(lo,4),"ci_high":round(hi,4)})
    log(f"{'First-to-merge share':28s} {lab:16s}: {m:.4f} CI=[{lo:.4f},{hi:.4f}]")
pd.DataFrame(table57).to_csv(OUT/"table5_7.csv", index=False)

fig, ax = plt.subplots(figsize=(8.5,5.5))
x = np.arange(4)
means = [np.mean([hb_win[s][c] for s in SEEDS12]) for c in CLASS_ORDER]
for i,c in enumerate(CLASS_ORDER):
    vals=[hb_win[s][c] for s in SEEDS12]
    ax.scatter([i]*len(vals), vals, color="#7f7f7f", alpha=0.6, s=45, zorder=3)
ax.scatter(range(4), means, color="#d62728", marker="D", s=90, zorder=4, label="12-seed mean")
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Hard-brake events per vehicle-episode (window, contiguous)")
ax.set_title("Hard-brake events by class, Baseline H1 (interaction window)")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_8a_baseline_hardbrake_by_class.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_8a_baseline_hardbrake_by_class.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_8a_baseline_hardbrake_by_class")

fig, ax = plt.subplots(figsize=(8,5.3))
vals = [100*np.mean([ftm[s][c] for s in SEEDS12]) for c in CLASS_ORDER]
ax.bar(x, vals, width=0.6, color="#7f7f7f", edgecolor="black", linewidth=0.8)
for xi,v in zip(x,vals): ax.text(xi, v+0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.axhline(25, color="green", linestyle=":", linewidth=1.3)
ax.text(3.5,26,"25% (uniform reference)",fontsize=8,color="green",ha="right")
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Share of episodes crossing the merge zone first (%)")
ax.set_title("First-to-merge class share, Baseline H1")
ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_8b_baseline_first_to_merge.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_8b_baseline_first_to_merge.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_8b_baseline_first_to_merge")

print("\nPHASE2_PART1_DONE")
with open(OUT/"phase2_part1_log.txt","w",encoding="utf-8") as f:
    f.write("\n".join(REPORT))
