"""Phase 2 part 2: 5.6.3 (mobility/braking costs, 4 conditions) + Fig 5.10,
5.6.4 (behavioural response, 4 conditions) + Fig 5.11/5.12, 5.7.3 (zero-burden
diagnostic recompute) + Table 5.13."""
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

bw = pd.read_csv(OUT / "behavioral_window_merged.csv")
welfare = pd.read_csv(OUT / "all4_conditions_merged.csv")
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

def seed_mean_bw(df_cond, field, seed):
    ssub = df_cond[df_cond.seed==seed]
    vals=[]
    for v in VIDS:
        vals.extend(ssub[f"{field}_{v}"].dropna().tolist())
    return float(np.mean(vals)) if vals else float("nan")

def seed_mean_w(df_cond, field, seed):
    v = df_cond[df_cond.seed==seed][field]
    return float(v.mean()) if len(v) else float("nan")

# ===========================================================================
# 5.6.3 -- Table 5.10: mobility/braking costs, 4 conditions + Fig 5.10
# ===========================================================================
log("="*78); log("5.6.3 -- Table 5.10: mobility/braking costs, 4 conditions"); log("="*78)
table510=[]
c_brake_seedmean = {}
for cond in CONDS4:
    wsub = welfare[(welfare.condition==cond)&(welfare.bank=="H1")]
    bsub = bw[bw.condition==cond]
    c_mean = np.mean([seed_mean_w(wsub,"C_mean",s) for s in SEEDS12])
    c_max = np.mean([seed_mean_w(wsub,"C_max",s) for s in SEEDS12])
    def rng_row(r):
        c=[r[f"C_{v}"] for v in VIDS]; return max(c)-min(c)
    brange = np.mean([wsub[wsub.seed==s].apply(rng_row,axis=1).mean() for s in SEEDS12])
    cbrake_vals = np.array([seed_mean_bw(bsub,"c_brake_window",s) for s in SEEDS12])
    c_brake_seedmean[cond] = cbrake_vals
    cbrake = cbrake_vals.mean()
    brate = np.mean([seed_mean_bw(bsub,"brake_rate_window",s) for s in SEEDS12])
    hb = np.mean([seed_mean_bw(bsub,"hard_brake_events_window",s) for s in SEEDS12])
    table510.append({"condition":COND_LABELS[cond],"C_mean":round(c_mean,4),"C_max":round(c_max,4),
                      "burden_range":round(brange,4),"C_brake":round(cbrake,4),
                      "brake_action_rate":round(brate,4),"hard_brake_events":round(hb,4)})
    log(f"{cond:8s}: C_mean={c_mean:.4f} C_max={c_max:.4f} range={brange:.4f} C_brake={cbrake:.4f} brake_rate={brate:.4f} hb={hb:.4f}")
pd.DataFrame(table510).to_csv(OUT/"table5_10.csv", index=False)

log("\nC_brake paired contrasts vs Baseline:")
for cond in ("mean","ggi","maximin"):
    diffs = c_brake_seedmean[cond]-c_brake_seedmean["baseline"]
    m,lo,hi = bootstrap_ci_paired(diffs)
    log(f"  {cond:8s} - Baseline, C_brake: delta={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")

fig, axes = plt.subplots(1,2, figsize=(12,5.3))
ax=axes[0]
x=np.arange(4)
vals=[np.mean([seed_mean_w(welfare[(welfare.condition==c)&(welfare.bank=='H1')],'C_mean',s) for s in SEEDS12]) for c in CONDS4]
ax.bar(x, vals, color=[COND_COLORS[c] for c in CONDS4], edgecolor="black", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels([COND_LABELS[c] for c in CONDS4])
ax.set_ylabel("Below-target mobility burden $C_{\\mathrm{mean}}$ (unconditional)")
ax.set_title("Original mobility burden"); ax.grid(True,axis="y",alpha=0.25,linewidth=0.6)
ax=axes[1]
vals2=[c_brake_seedmean[c].mean() for c in CONDS4]
ax.bar(x, vals2, color=[COND_COLORS[c] for c in CONDS4], edgecolor="black", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels([COND_LABELS[c] for c in CONDS4])
ax.set_ylabel("Braking burden $C_{\\mathrm{brake}}$ (window, 5Hz-sampled)")
ax.set_title("Physical braking burden"); ax.grid(True,axis="y",alpha=0.25,linewidth=0.6)
fig.suptitle("Mobility burden vs. physical braking burden, all 4 conditions (H1)", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_10_burden_vs_braking.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_10_burden_vs_braking.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_10_burden_vs_braking")

# ===========================================================================
# 5.6.4 -- worst-off share (4 cond) + Fig 5.11, hard-brake/merge-order + Fig 5.12
# ===========================================================================
log("\n"+"="*78); log("5.6.4 -- behavioural response to welfare shaping, 4 conditions"); log("="*78)

def worst_off_by_seed_class(wsub):
    out={}
    for s in SEEDS12:
        ssub = wsub[wsub.seed==s]
        frac=defaultdict(float); n_nd=0
        for _,r in ssub.iterrows():
            us={v:r[f"U_{v}"] for v in VIDS}
            mn=min(us.values())
            tied=[v for v,u in us.items() if abs(u-mn)<1e-6]
            if len(tied)==4: continue
            n_nd+=1; w=1.0/len(tied)
            for v in tied: frac[class_of(r,v)]+=w
        out[s]={c:(frac[c]/n_nd if n_nd else float("nan")) for c in CLASS_ORDER}
        out[s]["_n"]=n_nd
    return out

worst_off_4cond = {}
fig, ax = plt.subplots(figsize=(9,5.5))
x=np.arange(4); width=0.2
for i,cond in enumerate(CONDS4):
    wsub = welfare[(welfare.condition==cond)&(welfare.bank=="H1")]
    wo = worst_off_by_seed_class(wsub)
    worst_off_4cond[cond]=wo
    n_excluded = sum(1 for s in SEEDS12 if wo[s]["_n"] == 0)
    vals=[100*np.nanmean([wo[s][c] for s in SEEDS12]) for c in CLASS_ORDER]
    offset=(i-1.5)*width
    ax.bar(x+offset, vals, width=width*0.9, color=COND_COLORS[cond], edgecolor="black", linewidth=0.6, label=COND_LABELS[cond])
    log(f"{cond:8s} worst-off share: " + " ".join(f"{c}={v:.1f}%" for c,v in zip(CLASS_LABELS,vals))
        + (f"  [{n_excluded} seed(s) excluded: 0 non-degenerate episodes]" if n_excluded else ""))
ax.axhline(25,color="green",linestyle=":",linewidth=1.3)
ax.text(3.6,26,"25%",fontsize=8,color="green",ha="right")
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Share of non-degenerate episodes worst-off (%)")
ax.set_title("Worst-off vehicle class by condition (tie-corrected)")
ax.legend(fontsize=9); ax.grid(True,axis="y",alpha=0.25,linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_11_worst_off_by_condition.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_11_worst_off_by_condition.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_11_worst_off_by_condition")

fig, axes = plt.subplots(1,2, figsize=(13,5.3))
ax=axes[0]
for i,cond in enumerate(CONDS4):
    bsub = bw[bw.condition==cond]
    hb_by_class = defaultdict(list)
    for s in SEEDS12:
        ssub = bsub[bsub.seed==s]
        cvals=defaultdict(list)
        for _,r in ssub.iterrows():
            for v in VIDS:
                val=r[f"hard_brake_events_window_{v}"]
                if pd.notna(val): cvals[class_of(r,v)].append(val)
        for c in CLASS_ORDER: hb_by_class[c].append(np.mean(cvals[c]) if cvals[c] else np.nan)
    vals=[np.nanmean(hb_by_class[c]) for c in CLASS_ORDER]
    offset=(i-1.5)*width
    ax.bar(x+offset, vals, width=width*0.9, color=COND_COLORS[cond], edgecolor="black", linewidth=0.6, label=COND_LABELS[cond])
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("Hard-brake events (window, contiguous)")
ax.set_title("Hard-brake events by class and condition")
ax.legend(fontsize=8.5); ax.grid(True,axis="y",alpha=0.25,linewidth=0.6)

ax=axes[1]
for i,cond in enumerate(CONDS4):
    bsub = bw[bw.condition==cond]
    share_by_class=defaultdict(list)
    for s in SEEDS12:
        ssub=bsub[bsub.seed==s]
        cnt=defaultdict(int); n_valid=0
        for _,r in ssub.iterrows():
            mo=r["merge_order"]
            if mo=="DNF" or not isinstance(mo,str): continue
            first_v=mo.split(">")[0]
            cnt[class_of(r,first_v)]+=1; n_valid+=1
        for c in CLASS_ORDER: share_by_class[c].append(100*cnt[c]/n_valid if n_valid else np.nan)
    vals=[np.nanmean(share_by_class[c]) for c in CLASS_ORDER]
    offset=(i-1.5)*width
    ax.bar(x+offset, vals, width=width*0.9, color=COND_COLORS[cond], edgecolor="black", linewidth=0.6, label=COND_LABELS[cond])
ax.axhline(25,color="green",linestyle=":",linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=15, ha="right")
ax.set_ylabel("First-to-merge share (%)")
ax.set_title("Merge-order priority by class and condition")
ax.legend(fontsize=8.5); ax.grid(True,axis="y",alpha=0.25,linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_12_hardbrake_mergeorder_by_condition.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_12_hardbrake_mergeorder_by_condition.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_12_hardbrake_mergeorder_by_condition")

# ===========================================================================
# 5.7.3 -- Table 5.13: zero-burden diagnostic, frozen contiguous definition
# ===========================================================================
log("\n"+"="*78); log("5.7.3 -- Table 5.13: zero-burden diagnostic (contiguous, frozen)"); log("="*78)
# join welfare (C_i, completion) with bw (hard_brake_events_episode) on condition+seed+scenario_id
# (NOT run_id -- the two source pipelines used different run_id prefixes for
# Baseline, "taskonly_*" vs "baseline_*", so joining on run_id silently drops
# every Baseline row; condition+seed+scenario_id is unambiguous and correct
# for all four conditions).
key = ["condition","seed","scenario_id"]
merged = welfare[welfare.bank=="H1"].merge(
    bw[["condition","seed","scenario_id"]+[f"hard_brake_events_episode_{v}" for v in VIDS]+[f"brake_rate_window_{v}" for v in VIDS]+[f"c_brake_window_{v}" for v in VIDS]],
    on=key, how="inner")
log(f"merged rows: {len(merged)} (expect 12288)")

merged["all_zero_burden"] = merged[[f"C_{v}" for v in VIDS]].abs().sum(axis=1) < 1e-9

def hb_total(row):
    return sum(row[f"hard_brake_events_episode_{v}"] for v in VIDS)
def cbrake_total(row):
    return sum(row[f"c_brake_window_{v}"] for v in VIDS)/4.0

merged["hb_total"] = merged.apply(hb_total, axis=1)
merged["cbrake_mean"] = merged.apply(cbrake_total, axis=1)

for scope_name, scope_df in [("all episodes, Baseline", merged[merged.condition=="baseline"]),
                               ("successful episodes only, Baseline", merged[(merged.condition=="baseline")&(merged.term_reason=="success")])]:
    zero = scope_df[scope_df.all_zero_burden]
    nonzero = scope_df[~scope_df.all_zero_burden]
    log(f"\n-- {scope_name} -- n_zero={len(zero)} n_nonzero={len(nonzero)}")
    rows=[]
    for name, series_zero, series_nonzero in [
        (">=1 hard-brake event (share)", (zero.hb_total>0).mean(), (nonzero.hb_total>0).mean()),
        ("median hard-brake count", zero.hb_total.median(), nonzero.hb_total.median()),
        ("mean hard-brake count", zero.hb_total.mean(), nonzero.hb_total.mean()),
        ("95th pct hard-brake count", zero.hb_total.quantile(0.95), nonzero.hb_total.quantile(0.95)),
        ("mean braking burden C_brake", zero.cbrake_mean.mean(), nonzero.cbrake_mean.mean()),
    ]:
        rows.append({"measure":name,"zero_mobility_burden":round(float(series_zero),4),"nonzero_mobility_burden":round(float(series_nonzero),4)})
        log(f"  {name:32s}: zero-burden={series_zero:.4f}  nonzero-burden={series_nonzero:.4f}")
    pd.DataFrame(rows).to_csv(OUT/f"table5_13_{'success' if 'success' in scope_name else 'all'}.csv", index=False)

print("\nPHASE2_PART2_DONE")
with open(OUT/"phase2_part2_log.txt","w",encoding="utf-8") as f:
    f.write("\n".join(REPORT))
