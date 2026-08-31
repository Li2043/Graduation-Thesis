"""Phase 1 analysis part 2: 5.5.4 burden, 5.6.1/5.6.2/5.6.6 RQ2 vs Baseline,
5.7.1 robustness, 5.7.2 outcome decomposition, 5.7.4 Mean-policy-quality
correlation. Same data sources/conventions as part 1."""
from __future__ import annotations
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

df = pd.read_csv(OUT / "all4_conditions_merged.csv")

REPORT = []
def log(msg=""):
    print(msg); REPORT.append(str(msg))

def gini(values):
    n = len(values); total = sum(values)
    if total == 0: return None
    num = sum(abs(a-b) for a in values for b in values)
    return float(num/(2.0*n*total))

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

def seed_mean(sub, field, seed):
    v = sub[sub.seed==seed][field]
    return float(v.mean()) if len(v) else float("nan")

# ===========================================================================
# 5.5.4 -- Table 5.6: below-target mobility burden, Baseline H1 + Figure 5.7
# ===========================================================================
log("="*78); log("5.5.4 -- Table 5.6: burden, Baseline H1"); log("="*78)
b_h1 = df[(df.condition=="baseline")&(df.bank=="H1")].copy()
def burden_row(r):
    c = [r[f"C_{v}"] for v in VIDS]
    return pd.Series({"burden_range": max(c)-min(c),
                       "all_zero": 1 if sum(c)<1e-9 else 0})
b_h1 = pd.concat([b_h1, b_h1.apply(burden_row, axis=1)], axis=1)

table56A = []
for metric in ("C_mean","C_max","burden_range"):
    vals = np.array([seed_mean(b_h1, metric, s) for s in SEEDS12])
    m, lo, hi = bootstrap_ci_single(vals)
    table56A.append({"metric": metric, "mean": round(m,4), "median": round(float(np.median(vals)),4),
                      "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"{metric:14s}: mean={m:.4f} median={np.median(vals):.4f} CI=[{lo:.4f},{hi:.4f}]")
zero_share = np.array([seed_mean(b_h1,"all_zero",s) for s in SEEDS12])
m,lo,hi = bootstrap_ci_single(zero_share)
table56A.append({"metric":"zero_burden_episode_share","mean":round(m,4),"median":round(float(np.median(zero_share)),4),
                  "ci_low":round(lo,4),"ci_high":round(hi,4)})
log(f"zero_burden_share: mean={m:.4f} CI=[{lo:.4f},{hi:.4f}]")
pd.DataFrame(table56A).to_csv(OUT/"table5_6_panelA.csv", index=False)

class_c_by_seed = defaultdict(dict)
for s in SEEDS12:
    ssub = b_h1[b_h1.seed==s]
    c_by_class = defaultdict(list)
    for _, r in ssub.iterrows():
        for v in VIDS:
            cls = f"{r[f'role_{v}']}-{r[f'speed_class_{v}']}"
            c_by_class[cls].append(r[f"C_{v}"])
    for c in CLASS_ORDER:
        class_c_by_seed[s][c] = float(np.mean(c_by_class[c]))

table56B = []
for c, lab in zip(CLASS_ORDER, CLASS_LABELS):
    vals = np.array([class_c_by_seed[s][c] for s in SEEDS12])
    m, lo, hi = bootstrap_ci_single(vals)
    table56B.append({"class": lab, "mean_burden": round(m,4), "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"burden {lab:16s}: {m:.4f} CI=[{lo:.4f},{hi:.4f}]")
fastb = np.array([(class_c_by_seed[s]["ramp-fast"]+class_c_by_seed[s]["mainline-fast"])/2 for s in SEEDS12])
slowb = np.array([(class_c_by_seed[s]["ramp-slow"]+class_c_by_seed[s]["mainline-slow"])/2 for s in SEEDS12])
rampb = np.array([(class_c_by_seed[s]["ramp-fast"]+class_c_by_seed[s]["ramp-slow"])/2 for s in SEEDS12])
mainb = np.array([(class_c_by_seed[s]["mainline-fast"]+class_c_by_seed[s]["mainline-slow"])/2 for s in SEEDS12])
for name, diffs in [("Fast-Slow burden", fastb-slowb), ("Ramp-Mainline burden", rampb-mainb)]:
    m, lo, hi = bootstrap_ci_paired(diffs)
    table56B.append({"class": name, "mean_burden": round(m,4), "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"{name:20s}: delta={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
pd.DataFrame(table56B).to_csv(OUT/"table5_6_panelB.csv", index=False)

fig, ax = plt.subplots(figsize=(8.5,5.5))
x = np.arange(4)
for i,c in enumerate(CLASS_ORDER):
    vals=[class_c_by_seed[s][c] for s in SEEDS12]
    ax.scatter([i]*len(vals), vals, color="#7f7f7f", alpha=0.6, s=45, zorder=3)
means=[np.mean([class_c_by_seed[s][c] for s in SEEDS12]) for c in CLASS_ORDER]
ax.scatter(range(4), means, color="#d62728", marker="D", s=90, zorder=4, label="12-seed mean")
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Below-target mobility burden $C_i$ (Baseline, H1)")
ax.set_title("Mobility burden by role-speed class, all seeds (Baseline)")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_7_baseline_burden_by_class.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_7_baseline_burden_by_class.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_7_baseline_burden_by_class")

# Appendix: Burden Gini, only when total burden > 0
n_defined = 0
gini_vals = []
for s in SEEDS12:
    ssub = b_h1[b_h1.seed==s]
    gs = [gini([r[f"C_{v}"] for v in VIDS]) for _, r in ssub.iterrows()]
    gs = [g for g in gs if g is not None]
    if gs:
        n_defined += 1
        gini_vals.append((s, float(np.mean(gs)), len(gs), len(ssub)))
log(f"\nBurden Gini defined for {n_defined}/12 seeds (nonzero-burden episodes only):")
for s,g,n,tot in gini_vals:
    log(f"  seed {s}: mean burden Gini={g:.4f} (n={n}/{tot} nonzero-burden episodes)")
pd.DataFrame(gini_vals, columns=["seed","mean_burden_gini","n_nonzero_episodes","n_total_episodes"]).to_csv(OUT/"appendix_burden_gini.csv", index=False)

# ===========================================================================
# 5.6.1 -- Table 5.8: primary RQ2 contrast, U_min vs Baseline + Figure 5.9
# ===========================================================================
log("\n"+"="*78); log("5.6.1 -- Table 5.8: U_min contrasts vs Baseline"); log("="*78)
base_h1 = df[(df.condition=="baseline")&(df.bank=="H1")]
base_umin = np.array([seed_mean(base_h1,"min_U",s) for s in SEEDS12])
table58 = []
for cond in ("mean","ggi","maximin"):
    sub = df[(df.condition==cond)&(df.bank=="H1")]
    c_umin = np.array([seed_mean(sub,"min_U",s) for s in SEEDS12])
    diffs = c_umin - base_umin
    m, lo, hi = bootstrap_ci_paired(diffs)
    npos = int((diffs>0).sum())
    table58.append({"contrast": f"{COND_LABELS[cond]} - Baseline", "mean_delta": round(m,4),
                     "median_delta": round(float(np.median(diffs)),4), "positive_seeds": f"{npos}/12",
                     "ci_low": round(lo,4), "ci_high": round(hi,4)})
    log(f"{cond:8s} - Baseline, U_min: delta={m:+.4f} median={np.median(diffs):+.4f} pos={npos}/12 CI=[{lo:+.4f},{hi:+.4f}]")
pd.DataFrame(table58).to_csv(OUT/"table5_8.csv", index=False)

fig, ax = plt.subplots(figsize=(12,5.8))
x = np.arange(len(SEEDS12))
for cond in CONDS4:
    sub = df[(df.condition==cond)&(df.bank=="H1")]
    vals = [seed_mean(sub,"min_U",s) for s in SEEDS12]
    ax.plot(x, vals, marker="o", markersize=8, linewidth=1.3, color=COND_COLORS[cond], label=COND_LABELS[cond], alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEEDS12], rotation=25)
ax.set_xlabel("Formal seed"); ax.set_ylabel("Seed-mean worst-off utility $U_{\\min}$ (H1)")
ax.set_ylim(-0.02,1.08)
ax.set_title("Worst-off utility by formal seed and condition (matched seeds)")
ax.legend(fontsize=9.5); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_9_rq2_umin_by_seed.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_9_rq2_umin_by_seed.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_9_rq2_umin_by_seed")

# ===========================================================================
# 5.6.2 -- Table 5.9: Gini/U_mean/range vs Baseline
# ===========================================================================
log("\n"+"="*78); log("5.6.2 -- Table 5.9: Gini/U_mean/range, 4 conditions"); log("="*78)
def seed_gini(sub, s):
    ssub = sub[sub.seed==s]
    gs = [gini([r[f"U_{v}"] for v in VIDS]) for _, r in ssub.iterrows()]
    gs = [g for g in gs if g is not None]
    return float(np.mean(gs)) if gs else float("nan")

table59 = {}
seedgini = {}
for cond in CONDS4:
    sub = df[(df.condition==cond)&(df.bank=="H1")]
    umin = np.array([seed_mean(sub,"min_U",s) for s in SEEDS12])
    umean = np.array([seed_mean(sub,"mean_U",s) for s in SEEDS12])
    grange = np.array([seed_mean(sub,"mean_U",s) for s in SEEDS12])  # placeholder overwritten below
    gvals = np.array([seed_gini(sub, s) for s in SEEDS12])
    seedgini[cond] = gvals
    # utility range per seed
    def urange_row(r):
        u=[r[f"U_{v}"] for v in VIDS]; return max(u)-min(u)
    rng_vals = np.array([sub[sub.seed==s].apply(urange_row,axis=1).mean() for s in SEEDS12])
    table59[cond] = {"U_min": umin.mean(), "Utility_Gini": gvals.mean(), "U_mean": umean.mean(), "utility_range": rng_vals.mean()}
    log(f"{cond:8s}: U_min={umin.mean():.4f} Gini={gvals.mean():.4f} U_mean={umean.mean():.4f} range={rng_vals.mean():.4f}")
pd.DataFrame(table59).T.round(4).to_csv(OUT/"table5_9.csv")

log("\nGini contrasts vs Baseline:")
base_gini = seedgini["baseline"]
gini_contrasts=[]
for cond in ("mean","ggi","maximin"):
    diffs = seedgini[cond]-base_gini
    m,lo,hi = bootstrap_ci_paired(diffs)
    gini_contrasts.append({"contrast":f"{COND_LABELS[cond]} - Baseline","delta_gini":round(m,4),"ci_low":round(lo,4),"ci_high":round(hi,4)})
    log(f"{cond:8s} - Baseline, Gini: delta={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
pd.DataFrame(gini_contrasts).to_csv(OUT/"table5_9_gini_contrasts.csv", index=False)

# ===========================================================================
# 5.6.6 -- Table 5.11: safety/task-performance margins vs Baseline
# ===========================================================================
log("\n"+"="*78); log("5.6.6 -- Table 5.11: non-inferiority margins vs Baseline"); log("="*78)
base_comp = np.array([seed_mean(base_h1,"completion",s) for s in SEEDS12])
base_coll = np.array([seed_mean(base_h1,"collision",s) for s in SEEDS12])
base_to = np.array([seed_mean(base_h1,"timeout",s) for s in SEEDS12])
table511 = []
for cond in ("mean","ggi","maximin"):
    sub = df[(df.condition==cond)&(df.bank=="H1")]
    c_comp = np.array([seed_mean(sub,"completion",s) for s in SEEDS12])
    c_coll = np.array([seed_mean(sub,"collision",s) for s in SEEDS12])
    c_to = np.array([seed_mean(sub,"timeout",s) for s in SEEDS12])
    ec,lc,hc = bootstrap_ci_paired(c_comp-base_comp)
    ek,lk,hk = bootstrap_ci_paired(c_coll-base_coll)
    et,lt,ht = bootstrap_ci_paired(c_to-base_to)
    comp_ni = lc > -0.05; coll_ni = hk < 0.03
    table511.append({"contrast":f"{COND_LABELS[cond]} - Baseline","completion_delta":round(ec,4),
                      "completion_ci":f"[{lc:+.4f},{hc:+.4f}]","completion_NI":"Yes" if comp_ni else "No",
                      "collision_delta":round(ek,4),"collision_ci":f"[{lk:+.4f},{hk:+.4f}]",
                      "collision_NI":"Yes" if coll_ni else "No","timeout_delta":round(et,4)})
    log(f"{cond:8s} completion delta={ec:+.4f} CI=[{lc:+.4f},{hc:+.4f}] NI={comp_ni}")
    log(f"{cond:8s} collision  delta={ek:+.4f} CI=[{lk:+.4f},{hk:+.4f}] NI={coll_ni}")
    log(f"{cond:8s} timeout    delta={et:+.4f} CI=[{lt:+.4f},{ht:+.4f}]")
pd.DataFrame(table511).to_csv(OUT/"table5_11.csv", index=False)

print("\n\nPHASE1_PART2_DONE")
with open(OUT/"phase1_part2_log.txt","w",encoding="utf-8") as f:
    f.write("\n".join(REPORT))
