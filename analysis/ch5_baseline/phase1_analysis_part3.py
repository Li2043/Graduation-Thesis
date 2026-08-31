"""Phase 1 part 3: 5.7.1 robustness table, 5.7.2 outcome decomposition +
Figure 5.14, 5.7.4 Mean-policy-quality correlation + Figure 5.16,
5.3.2 training-window trajectories (Baseline + Mean/GGI/Maximin logs) +
Figure 5.2."""
from __future__ import annotations
import os
import re
from pathlib import Path
from collections import defaultdict

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
SEEDS_ORIG, SEEDS_NEW = SEEDS12[:6], SEEDS12[6:]
COND_COLORS = {"baseline": "#7f7f7f", "mean": "#1f77b4", "ggi": "#2ca02c", "maximin": "#d62728"}
COND_LABELS = {"baseline": "Baseline", "mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}
CONDS4 = ["baseline", "mean", "ggi", "maximin"]

df = pd.read_csv(OUT / "all4_conditions_merged.csv")
REPORT = []
def log(msg=""):
    print(msg); REPORT.append(str(msg))

def bootstrap_ci_paired(diffs, n_boot=10000, seed0=0):
    rng = np.random.default_rng(seed0); n = len(diffs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0,n,n); boot[i] = np.mean(diffs[idx])
    return float(np.mean(diffs)), float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))

def seed_mean(sub, field, seed):
    v = sub[sub.seed==seed][field]
    return float(v.mean()) if len(v) else float("nan")

def gini(values):
    n=len(values); total=sum(values)
    if total==0: return None
    return float(sum(abs(a-b) for a in values for b in values)/(2.0*n*total))

def seed_gini(sub, s):
    ssub = sub[sub.seed==s]
    gs=[gini([r[f"U_{v}"] for v in ("V0","V1","V2","V3")]) for _,r in ssub.iterrows()]
    gs=[g for g in gs if g is not None]
    return float(np.mean(gs)) if gs else float("nan")

# ===========================================================================
# 5.7.1 -- robustness table
# ===========================================================================
log("="*78); log("5.7.1 -- robustness table (per-seed deltas)"); log("="*78)
base_h1 = df[(df.condition=="baseline")&(df.bank=="H1")]
base_umin = {s: seed_mean(base_h1,"min_U",s) for s in SEEDS12}
base_gini = {s: seed_gini(base_h1, s) for s in SEEDS12}
rows=[]
for s in SEEDS12:
    row={"seed": s}
    for cond in ("mean","ggi","maximin"):
        sub = df[(df.condition==cond)&(df.bank=="H1")]
        row[f"delta_umin_{cond}"] = round(seed_mean(sub,"min_U",s)-base_umin[s],4)
        row[f"delta_gini_{cond}"] = round(seed_gini(sub,s)-base_gini[s],4)
    rows.append(row)
    log(f"seed {s}: " + " ".join(f"{k}={v:+.3f}" for k,v in row.items() if k!="seed"))
pd.DataFrame(rows).to_csv(OUT/"table_robustness.csv", index=False)

log("\nBatch sensitivity (original 6 vs new 6), U_min delta vs Baseline:")
for cond in ("mean","ggi","maximin"):
    sub = df[(df.condition==cond)&(df.bank=="H1")]
    d_orig = np.array([seed_mean(sub,"min_U",s)-base_umin[s] for s in SEEDS_ORIG])
    d_new = np.array([seed_mean(sub,"min_U",s)-base_umin[s] for s in SEEDS_NEW])
    log(f"  {cond:8s}: original-6 mean delta={d_orig.mean():+.4f}  new-6 mean delta={d_new.mean():+.4f}")

# ===========================================================================
# 5.7.2 -- Table 5.12 outcome decomposition + Figure 5.14
# ===========================================================================
log("\n"+"="*78); log("5.7.2 -- Table 5.12: outcome decomposition of C_mean"); log("="*78)
table512=[]
decomp_for_fig = {}
for cond in CONDS4:
    sub = df[(df.condition==cond)&(df.bank=="H1")]
    overall = sub.C_mean.mean()
    contribs={}
    for outcome, col in (("success","completion"),("collision","collision"),("timeout","timeout")):
        p = sub[col].mean()
        cond_mean = sub[sub[col]==1].C_mean.mean() if (sub[col]==1).any() else 0.0
        contribs[outcome] = p*cond_mean
    check = sum(contribs.values())
    table512.append({"condition": COND_LABELS[cond], "overall_C_mean": round(overall,4),
                      "success_contribution": round(contribs["success"],4),
                      "collision_contribution": round(contribs["collision"],4),
                      "timeout_contribution": round(contribs["timeout"],4),
                      "check_sum": round(check,4)})
    decomp_for_fig[cond] = contribs
    log(f"{cond:8s}: overall={overall:.4f}  success_contrib={contribs['success']:.4f}  collision_contrib={contribs['collision']:.4f}  timeout_contrib={contribs['timeout']:.4f}  check_sum={check:.4f}")
pd.DataFrame(table512).to_csv(OUT/"table5_12.csv", index=False)

fig, ax = plt.subplots(figsize=(8,5.5))
x = np.arange(4)
bottoms = np.zeros(4)
for outcome, color in (("success","#2ca02c"),("collision","#d62728"),("timeout","#7f7f7f")):
    vals = np.array([decomp_for_fig[c][outcome] for c in CONDS4])
    ax.bar(x, vals, bottom=bottoms, color=color, label=outcome.capitalize(), edgecolor="black", linewidth=0.6)
    bottoms += vals
ax.set_xticks(x); ax.set_xticklabels([COND_LABELS[c] for c in CONDS4])
ax.set_ylabel("Contribution to unconditional mean burden $C_{\\mathrm{mean}}$")
ax.set_title("Outcome decomposition of unconditional mobility burden (H1)")
ax.legend(fontsize=9.5); ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_14_outcome_decomposition.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_14_outcome_decomposition.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_14_outcome_decomposition")

# ===========================================================================
# 5.7.4 -- Mean-policy quality vs welfare-objective response (rename) + Fig 5.16
# ===========================================================================
log("\n"+"="*78); log("5.7.4 -- Mean-policy quality vs welfare-objective response"); log("="*78)
mean_h1 = df[(df.condition=="mean")&(df.bank=="H1")]
base_comp = np.array([seed_mean(mean_h1,"completion",s) for s in SEEDS12])
base_umin = np.array([seed_mean(mean_h1,"min_U",s) for s in SEEDS12])
rows=[]
for cond in ("ggi","maximin"):
    sub = df[(df.condition==cond)&(df.bank=="H1")]
    d_comp = np.array([seed_mean(sub,"completion",s) for s in SEEDS12]) - base_comp
    d_umin = np.array([seed_mean(sub,"min_U",s) for s in SEEDS12]) - base_umin
    r_comp = float(np.corrcoef(base_comp, d_comp)[0,1])
    r_umin = float(np.corrcoef(base_umin, d_umin)[0,1])
    log(f"{cond:8s}: r(Mean completion, delta completion)={r_comp:.4f}   r(Mean U_min, delta U_min)={r_umin:.4f}")
    rows.append({"condition":cond,"r_completion":round(r_comp,4),"r_umin":round(r_umin,4)})
pd.DataFrame(rows).to_csv(OUT/"table_mean_policy_quality_corr.csv", index=False)

fig, axes = plt.subplots(1,2, figsize=(11,5))
for ax, base_x, ylabel, title in [
    (axes[0], base_comp, "$\\Delta$ completion (condition-Mean)", "Baseline: Mean completion vs. completion change"),
    (axes[1], base_umin, "$\\Delta U_{\\min}$ (condition-Mean)", "Mean $U_{\\min}$ vs. $U_{\\min}$ change"),
]:
    for cond, color, marker in [("ggi","#2ca02c","o"),("maximin","#d62728","^")]:
        sub = df[(df.condition==cond)&(df.bank=="H1")]
        if ylabel.startswith("$\\Delta$ completion"):
            y = np.array([seed_mean(sub,"completion",s) for s in SEEDS12]) - base_comp
        else:
            y = np.array([seed_mean(sub,"min_U",s) for s in SEEDS12]) - base_umin
        r = float(np.corrcoef(base_x, y)[0,1])
        ax.scatter(base_x, y, color=color, marker=marker, s=70, label=f"{COND_LABELS[cond]} (r={r:.2f})", zorder=3)
    ax.axhline(0, color="gray", linewidth=1, alpha=0.6)
    ax.set_xlabel("Mean-policy quality (H1)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.6); ax.legend(fontsize=9)
fig.suptitle("Mean-policy quality vs. welfare-objective response (n=12, exploratory)", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_16_mean_policy_quality_response.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_16_mean_policy_quality_response.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_16_mean_policy_quality_response")

# ===========================================================================
# 5.3.2 -- training-window trajectories, all 48 branches + Figure 5.2
# ===========================================================================
log("\n"+"="*78); log("5.3.2 -- training-window trajectories, 48 branches"); log("="*78)
LOG_RE = re.compile(r"step=\s*(\d+)\s+completion=([\d.]+)\s+collision=([\d.]+)\s+timeout=([\d.]+)")

def log_path(cond, seed):
    if cond == "baseline":
        return Path(os.environ.get("FINAL_NEW_BUNDLE", "")) / "logs" / f"taskonly_{seed}.log"
    if seed in SEEDS_ORIG:
        return Path(os.environ.get("FINAL_NEW_BUNDLE", "")) / "logs" / f"formal_{cond}_{seed}.log"
    return Path(os.environ.get("SEED_REPL_BUNDLE", "")) / "logs" / f"replication_welfare_{cond}_{seed}.log"

def parse_log(cond, seed):
    p = log_path(cond, seed)
    if not p.exists():
        return []
    pts=[]
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_RE.search(line)
        if m:
            step,comp,coll,to = m.groups()
            pts.append({"step":int(step),"completion":float(comp),"collision":float(coll),"timeout":float(to)})
    if pts and pts[0]["completion"]==0.0 and pts[0]["collision"]==0.0 and pts[0]["timeout"]==0.0:
        pts = pts[1:]
    return pts

instability=[]
for cond in CONDS4:
    for seed in SEEDS12:
        pts = parse_log(cond, seed)
        if not pts:
            log(f"  WARNING: no log points for {cond}/{seed}"); continue
        last3 = pts[-3:]
        to_rise = last3[-1]["timeout"] - (pts[0]["timeout"] if len(pts)>3 else 0)
        comp_drop = max(p["completion"] for p in pts[:-3]) - min(p["completion"] for p in last3) if len(pts)>3 else 0
        if last3[-1]["timeout"] >= 0.10 or comp_drop >= 0.30:
            instability.append((cond, seed, last3[-1]["completion"], last3[-1]["collision"], last3[-1]["timeout"]))
log("\nSeeds/conditions with a clear late-training instability signature (final timeout>=0.10 or completion drop>=0.30 from peak):")
for row in instability:
    log(f"  {row}")

fig, axes = plt.subplots(1,4, figsize=(18,4.6), sharey=True)
HIGHLIGHT = {900104:"maximin", 910102:"mean"}
for ax, cond in zip(axes, CONDS4):
    for seed in SEEDS12:
        pts = parse_log(cond, seed)
        if not pts: continue
        steps=[p["step"] for p in pts]; comp=[p["completion"] for p in pts]
        hl = HIGHLIGHT.get(seed)==cond
        ax.plot(steps, comp, color=COND_COLORS[cond] if not hl else "black",
                linewidth=2.2 if hl else 0.9, alpha=1.0 if hl else 0.35,
                linestyle="-" if hl else "--")
    ax.set_title(COND_LABELS[cond]); ax.set_xlabel("Absolute step")
    ax.grid(True, alpha=0.25, linewidth=0.6); ax.set_ylim(-0.05,1.05)
axes[0].set_ylabel("Training-window completion rate")
fig.suptitle("Training-window completion during formal continuation, 12 seeds x 4 conditions\n(bold black = seeds with a flagged late-training instability signature)", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig5_2_formal_training_trajectories.png", bbox_inches="tight")
fig.savefig(FIG_DIR/"fig5_2_formal_training_trajectories.pdf", bbox_inches="tight")
plt.close(fig)
log("wrote fig5_2_formal_training_trajectories")

print("\n\nPHASE1_PART3_DONE")
with open(OUT/"phase1_part3_log.txt","w",encoding="utf-8") as f:
    f.write("\n".join(REPORT))
