"""Conflict-exposure diagnostic: statistical analysis (sections G/H/I/J/K).

Reads conflict_exposure_episode_level.csv (12 seeds x 256 H1 Baseline
episodes, produced by conflict_exposure_diagnostic_eval.py). All episodes
retained throughout (collision/timeout/success). Seed is the inferential
replication unit -- every proportion/mean/contrast is computed per-seed
first, then summarized across the 12 seeds with a 10,000-resample
seed-level bootstrap (percentile 95% CI). No p-values, no "significant".
"""
from __future__ import annotations
import os

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR = Path(__file__).resolve().parent / "outputs" / "conflict_exposure"
FIG_DIR = Path(os.environ.get("THESIS_FIGURES_DIR", str(Path(__file__).resolve().parent / "outputs" / "figures")))
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS12 = [900101, 900102, 900103, 900104, 910101, 910102,
           920101, 920102, 920103, 920104, 920105, 920106]

REPORT: list[str] = []
def log(msg=""):
    print(msg); REPORT.append(str(msg))


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed0: int = 0):
    rng = np.random.default_rng(seed0)
    n = len(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = np.mean(values[idx])
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------
# Load + merge shards
# ---------------------------------------------------------------------
shards = [pd.read_csv(IN_DIR / f"conflict_exposure_shard{i}.csv") for i in range(12)]
df = pd.concat(shards, ignore_index=True)
df.to_csv(IN_DIR / "conflict_exposure_episode_level.csv", index=False)
log(f"Loaded {len(df)} episodes across {df.seed.nunique()} seeds.")
assert set(df.seed.unique()) == set(SEEDS12)
for s in SEEDS12:
    n = len(df[df.seed == s])
    assert n == 256, f"seed {s}: expected 256, got {n}"
log("Validated: 12 seeds x 256 H1 episodes each, all outcomes retained (success/collision/timeout).")
log(f"Outcome mix (all 3072 episodes): {df.outcome.value_counts().to_dict()}")

seed_rows = []

# ======================================================================
# G1: how much conflict exposure exists
# ======================================================================
log("\n" + "=" * 78); log("G1 -- conflict exposure prevalence"); log("=" * 78)

overlap_prop = np.array([df[df.seed == s].cross_road_overlap_any.mean() for s in SEEDS12])
m, lo, hi = bootstrap_ci(overlap_prop)
log(f"P(any cross-road overlap): seed-level mean={m:.4f}  median={np.median(overlap_prop):.4f}  "
    f"range=[{overlap_prop.min():.4f},{overlap_prop.max():.4f}]  95% CI=[{lo:.4f},{hi:.4f}]")

dur_mean = np.array([df[df.seed == s].cross_road_overlap_duration_s.mean() for s in SEEDS12])
m, lo, hi = bootstrap_ci(dur_mean)
log(f"Overlap duration (s), per-seed mean: cross-seed mean={m:.3f}  median={np.median(dur_mean):.3f}  "
    f"95% CI=[{lo:.3f},{hi:.3f}]")
log(f"Pooled overlap-duration IQR (descriptive, all overlap episodes): "
    f"{df[df.cross_road_overlap_any==1].cross_road_overlap_duration_s.quantile([.25,.5,.75]).to_dict()}")

max_simul_dist = df.max_simultaneous_merge_vehicles.value_counts(normalize=True).sort_index()
log(f"Pooled distribution of max_simultaneous_merge_vehicles (descriptive): {max_simul_dist.to_dict()}")
per_seed_mean_maxsimul = np.array([df[df.seed == s].max_simultaneous_merge_vehicles.mean() for s in SEEDS12])
m, lo, hi = bootstrap_ci(per_seed_mean_maxsimul)
log(f"max_simultaneous_merge_vehicles, per-seed mean: cross-seed mean={m:.3f}  95% CI=[{lo:.3f},{hi:.3f}]")

for gapcol, label in [("min_crossroad_crossing_gap_x380", "x=380 (merge-zone end)"),
                       ("min_crossroad_crossing_gap_x300", "x=300 (parallel-merge start)")]:
    valid = df[df[gapcol].notna()]
    n_missing = len(df) - len(valid)
    log(f"\nMinimum Ramp-Mainline crossing-time gap at {label}: "
        f"{n_missing}/{len(df)} episodes have no valid pair (e.g. collision before either boundary).")
    per_seed_median = np.array([valid[valid.seed == s][gapcol].median() for s in SEEDS12 if len(valid[valid.seed==s])>0])
    log(f"  per-seed median gap: cross-seed median-of-medians={np.median(per_seed_median):.4f}s  "
        f"range=[{per_seed_median.min():.4f},{per_seed_median.max():.4f}]")
    for thresh in (0.5, 1.0, 1.5):
        per_seed_prop = np.array([ (valid[valid.seed==s][gapcol] <= thresh).mean() if len(valid[valid.seed==s])>0 else np.nan for s in SEEDS12])
        per_seed_prop = per_seed_prop[~np.isnan(per_seed_prop)]
        m, lo, hi = bootstrap_ci(per_seed_prop)
        log(f"  proportion with gap <= {thresh}s: cross-seed mean={m:.4f}  95% CI=[{lo:.4f},{hi:.4f}]  (descriptive sensitivity threshold)")

xgap_mean = np.array([df[df.seed == s].min_crossroad_x_gap.mean() for s in SEEDS12])
m, lo, hi = bootstrap_ci(xgap_mean)
log(f"\nMinimum cross-road longitudinal separation (m), per-seed mean: cross-seed mean={m:.3f}  95% CI=[{lo:.3f},{hi:.3f}]")
log(f"Pooled distribution (descriptive): {df.min_crossroad_x_gap.quantile([.05,.25,.5,.75,.95]).to_dict()}")

for s in SEEDS12:
    sub = df[df.seed == s]
    seed_rows.append({
        "seed": s,
        "n_episodes": len(sub),
        "n_overlap": int(sub.cross_road_overlap_any.sum()),
        "n_no_overlap": int((sub.cross_road_overlap_any == 0).sum()),
        "overlap_proportion": round(sub.cross_road_overlap_any.mean(), 4),
        "overlap_duration_s_mean": round(sub.cross_road_overlap_duration_s.mean(), 4),
        "max_simultaneous_merge_vehicles_mean": round(sub.max_simultaneous_merge_vehicles.mean(), 4),
        "min_crossroad_x_gap_mean": round(sub.min_crossroad_x_gap.mean(), 4),
        "min_gap_x380_median": round(sub.min_crossroad_crossing_gap_x380.median(), 4) if sub.min_crossroad_crossing_gap_x380.notna().any() else None,
    })

log(f"\nSeeds with zero no-overlap episodes: "
    f"{[r['seed'] for r in seed_rows if r['n_no_overlap']==0]}")
log(f"Seeds with zero overlap episodes: "
    f"{[r['seed'] for r in seed_rows if r['n_overlap']==0]}")

# ======================================================================
# helper: within-seed group contrast (overlap vs no-overlap), skip empty groups
# ======================================================================
def within_seed_contrast(data: pd.DataFrame, col: str, agg: str = "mean"):
    """Returns (seeds_used, overlap_vals, nooverlap_vals, contrast_vals) --
    seeds where either group is empty are excluded and reported separately."""
    seeds_used, ov_vals, no_vals, contrasts = [], [], [], []
    excluded = []
    for s in SEEDS12:
        sub = data[data.seed == s]
        ov = sub[sub.cross_road_overlap_any == 1]
        no = sub[sub.cross_road_overlap_any == 0]
        if len(ov) == 0 or len(no) == 0:
            excluded.append((s, len(ov), len(no)))
            continue
        if agg == "mean":
            ov_v, no_v = ov[col].mean(), no[col].mean()
        else:
            raise ValueError(agg)
        seeds_used.append(s); ov_vals.append(ov_v); no_vals.append(no_v); contrasts.append(ov_v - no_v)
    return seeds_used, np.array(ov_vals), np.array(no_vals), np.array(contrasts), excluded


def report_contrast(data: pd.DataFrame, col: str, label: str):
    seeds_used, ov, no, contrasts, excluded = within_seed_contrast(data, col)
    if excluded:
        log(f"  [{label}] excluded {len(excluded)} seed(s) with an empty group "
            f"(seed, n_overlap, n_no_overlap): {excluded}")
    if len(contrasts) < 2:
        log(f"  [{label}] fewer than 2 seeds with both groups present -- cannot bootstrap. "
            f"Raw per-seed values: overlap={dict(zip(seeds_used, ov))}, no_overlap={dict(zip(seeds_used, no))}")
        return seeds_used, ov, no, contrasts
    m, lo, hi = bootstrap_ci(contrasts)
    direction = "uncertain / CI crosses zero" if lo < 0 < hi else ("higher during overlap" if m > 0 else "lower during overlap")
    log(f"  [{label}] n_seeds_used={len(contrasts)}  overlap_mean={ov.mean():.4f}  no_overlap_mean={no.mean():.4f}  "
        f"contrast(overlap-no_overlap)={m:+.4f}  95% CI=[{lo:+.4f},{hi:+.4f}]  -> {direction}")
    return seeds_used, ov, no, contrasts


# ======================================================================
# G2: does overlap produce behavioural coordination
# ======================================================================
log("\n" + "=" * 78); log("G2 -- behavioural response, overlap vs no-overlap"); log("=" * 78)
for col, label in [
    ("any_BRAKE_action", "P(any BRAKE)"),
    ("any_hard_brake_event", "P(any hard brake)"),
    ("any_below_target_burden", "P(any below-target burden)"),
    ("C_brake", "mean acceleration-based braking burden"),
    ("C_mean", "mean below-target burden"),
]:
    report_contrast(df, col, label)

# ======================================================================
# G3: is inequality concentrated in conflict-exposed episodes
# ======================================================================
log("\n" + "=" * 78); log("G3 -- fairness outcomes, overlap vs no-overlap"); log("=" * 78)
_, _, _, gini_contrasts = report_contrast(df, "Utility_Gini", "Delta Gini_conflict (overlap - no_overlap)")
_, _, _, umin_contrasts = report_contrast(df, "U_min", "Delta Umin_conflict (overlap - no_overlap)")
_, _, _, range_contrasts = report_contrast(df, "utility_range", "Delta Range_conflict (overlap - no_overlap)")

# ======================================================================
# G4: conflict intensity vs inequality (per-seed Spearman, continuous)
# ======================================================================
log("\n" + "=" * 78); log("G4 -- conflict intensity vs inequality (per-seed Spearman)"); log("=" * 78)
X_VARS = [("cross_road_overlap_duration_s", "overlap duration (s)"),
          ("min_crossroad_crossing_gap_x380", "min crossing-time gap @x380 (s)"),
          ("min_crossroad_x_gap", "min cross-road separation (m)")]
Y_VARS = [("Utility_Gini", "Utility Gini"), ("U_min", "U_min"), ("utility_range", "utility range")]

g4_rows = []
for xcol, xlabel in X_VARS:
    for ycol, ylabel in Y_VARS:
        rhos = []
        for s in SEEDS12:
            sub = df[(df.seed == s) & df[xcol].notna() & df[ycol].notna()]
            if len(sub) < 5:
                continue
            rho = spearman_rho(sub[xcol].to_numpy(), sub[ycol].to_numpy())
            if not np.isnan(rho):
                rhos.append(rho)
        rhos = np.array(rhos)
        log(f"[{xlabel} vs {ylabel}] n_seeds={len(rhos)}  median rho={np.median(rhos):+.3f}  "
            f"range=[{rhos.min():+.3f},{rhos.max():+.3f}]")
        g4_rows.append({"x": xlabel, "y": ylabel, "n_seeds": len(rhos),
                         "median_rho": round(float(np.median(rhos)), 4),
                         "min_rho": round(float(rhos.min()), 4), "max_rho": round(float(rhos.max()), 4)})

# ======================================================================
# H: success-only sensitivity
# ======================================================================
log("\n" + "=" * 78); log("H -- SECONDARY: success-only sensitivity"); log("=" * 78)
succ = df[df.completion == 1]
log(f"Success-only subsample: {len(succ)} / {len(df)} episodes ({len(succ)/len(df):.1%}).")
for col, label in [("Utility_Gini", "Delta Gini (success-only)"), ("U_min", "Delta Umin (success-only)"),
                   ("utility_range", "Delta Range (success-only)"), ("any_BRAKE_action", "P(any BRAKE) (success-only)")]:
    report_contrast(succ, col, label)

# ======================================================================
# I: exposure / behaviour / welfare-sacrifice decomposition
# ======================================================================
log("\n" + "=" * 78); log("I -- exposure/behaviour/welfare-sacrifice decomposition"); log("=" * 78)

def categorize(row):
    if row.cross_road_overlap_any == 0:
        return "no_overlap"
    if row.any_BRAKE_action == 0:
        return "overlap_no_brake"
    if row.any_below_target_burden == 0:
        return "overlap_brake_zero_burden"
    return "overlap_brake_positive_burden"

df["exposure_category"] = df.apply(categorize, axis=1)
CATS = ["no_overlap", "overlap_no_brake", "overlap_brake_zero_burden", "overlap_brake_positive_burden"]
cat_props = {}
for cat in CATS:
    per_seed = np.array([(df[df.seed == s].exposure_category == cat).mean() for s in SEEDS12])
    m, lo, hi = bootstrap_ci(per_seed)
    cat_props[cat] = (m, lo, hi)
    log(f"  {cat:32s}: cross-seed mean proportion={m:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")
pooled_check = df.exposure_category.value_counts(normalize=True)
log(f"  pooled (descriptive) proportions: {pooled_check.to_dict()}")

# seed summary CSV -----------------------------------------------------
for i, s in enumerate(SEEDS12):
    seed_rows[i]["prop_any_BRAKE_given_overlap"] = round(
        df[(df.seed == s) & (df.cross_road_overlap_any == 1)].any_BRAKE_action.mean(), 4) if (df.seed==s).sum() else None
    seed_rows[i]["prop_any_BRAKE_given_no_overlap"] = (
        round(df[(df.seed == s) & (df.cross_road_overlap_any == 0)].any_BRAKE_action.mean(), 4)
        if len(df[(df.seed == s) & (df.cross_road_overlap_any == 0)]) > 0 else None)
    seed_rows[i]["gini_overlap_mean"] = round(df[(df.seed == s) & (df.cross_road_overlap_any == 1)].Utility_Gini.mean(), 4)
    seed_rows[i]["gini_no_overlap_mean"] = (
        round(df[(df.seed == s) & (df.cross_road_overlap_any == 0)].Utility_Gini.mean(), 4)
        if len(df[(df.seed == s) & (df.cross_road_overlap_any == 0)]) > 0 else None)
    seed_rows[i]["umin_overlap_mean"] = round(df[(df.seed == s) & (df.cross_road_overlap_any == 1)].U_min.mean(), 4)
    seed_rows[i]["umin_no_overlap_mean"] = (
        round(df[(df.seed == s) & (df.cross_road_overlap_any == 0)].U_min.mean(), 4)
        if len(df[(df.seed == s) & (df.cross_road_overlap_any == 0)]) > 0 else None)
    for cat in CATS:
        seed_rows[i][f"prop_{cat}"] = round((df[df.seed == s].exposure_category == cat).mean(), 4)

seed_df = pd.DataFrame(seed_rows)
seed_df.to_csv(IN_DIR / "conflict_exposure_seed_summary.csv", index=False)
pd.DataFrame(g4_rows).to_csv(IN_DIR / "conflict_exposure_g4_correlations.csv", index=False)
log(f"\nwrote {IN_DIR / 'conflict_exposure_seed_summary.csv'}")
log(f"wrote {IN_DIR / 'conflict_exposure_g4_correlations.csv'}")

# ======================================================================
# Figures
# ======================================================================
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                      "figure.dpi": 150, "savefig.dpi": 150})

# Figure A: seed-level overlap proportion
fig, ax = plt.subplots(figsize=(7, 4.5))
order = np.argsort(overlap_prop)
ax.bar(range(12), overlap_prop[order], color="#1f77b4")
ax.set_xticks(range(12)); ax.set_xticklabels([SEEDS12[i] for i in order], rotation=45, fontsize=8)
ax.set_ylabel("Proportion of episodes with cross-road overlap"); ax.set_ylim(0, 1.02)
ax.set_title("Figure A. Seed-level cross-road merge-zone overlap proportion (Baseline, H1)")
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "conflict_exposure_A_overlap_proportion.png", bbox_inches="tight"); plt.close(fig)

# Figure B: distribution of min crossing-time gap
fig, ax = plt.subplots(figsize=(7, 4.5))
vals = df.min_crossroad_crossing_gap_x380.dropna()
ax.hist(vals, bins=40, color="#2ca02c", edgecolor="black", linewidth=0.4)
ax.set_xlabel("Minimum Ramp-Mainline crossing-time gap at x=380 (s)")
ax.set_ylabel("Episodes"); ax.set_title("Figure B. Distribution of minimum crossing-time gap (all 3072 episodes)")
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "conflict_exposure_B_crossing_gap_dist.png", bbox_inches="tight"); plt.close(fig)

# Figure C/D: seed-level Gini / Umin, overlap vs no-overlap
for col, letter, ylabel in [("Utility_Gini", "C", "Utility Gini"), ("U_min", "D", "$U_{\\min}$")]:
    seeds_used, ov, no, _, _ = within_seed_contrast(df, col)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(seeds_used))
    ax.plot(x, no, "o-", color="#7f7f7f", label="No overlap")
    ax.plot(x, ov, "o-", color="#d62728", label="Overlap")
    ax.set_xticks(x); ax.set_xticklabels(seeds_used, rotation=45, fontsize=8)
    ax.set_ylabel(ylabel); ax.legend()
    ax.set_title(f"Figure {letter}. Seed-level {ylabel}: overlap vs. no-overlap (Baseline, H1)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"conflict_exposure_{letter}_{col}_overlap_vs_no.png", bbox_inches="tight"); plt.close(fig)

# Figure E: conflict intensity vs Gini (pooled scatter, seed-colored, descriptive)
fig, ax = plt.subplots(figsize=(7, 5))
cmap = plt.get_cmap("tab20")
for i, s in enumerate(SEEDS12):
    sub = df[df.seed == s]
    ax.scatter(sub.cross_road_overlap_duration_s, sub.Utility_Gini, s=8, alpha=0.5, color=cmap(i), label=str(s))
ax.set_xlabel("Cross-road overlap duration (s)"); ax.set_ylabel("Utility Gini")
ax.set_title("Figure E. Conflict intensity (overlap duration) vs. Utility Gini, all episodes")
ax.legend(fontsize=6, ncol=2, loc="upper right")
ax.grid(True, alpha=0.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "conflict_exposure_E_intensity_vs_gini.png", bbox_inches="tight"); plt.close(fig)

# Figure F: exposure decomposition
fig, ax = plt.subplots(figsize=(7.5, 4.5))
means = [cat_props[c][0] for c in CATS]
los = [cat_props[c][1] for c in CATS]
his = [cat_props[c][2] for c in CATS]
x = np.arange(4)
ax.bar(x, means, yerr=[np.array(means)-np.array(los), np.array(his)-np.array(means)],
       color=["#7f7f7f", "#ff7f0e", "#1f77b4", "#d62728"], capsize=4, edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(["no overlap", "overlap,\nno brake", "overlap+brake,\nzero burden",
                                       "overlap+brake,\npositive burden"], fontsize=8.5)
ax.set_ylabel("Proportion of episodes (cross-seed mean)")
ax.set_title("Figure F. Exposure / behavioural-response / welfare-sacrifice decomposition")
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "conflict_exposure_F_decomposition.png", bbox_inches="tight"); plt.close(fig)

log("\nwrote figures A-F to " + str(FIG_DIR))

with open(IN_DIR / "conflict_exposure_report_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT) + "\n")
log(f"\nwrote {IN_DIR / 'conflict_exposure_report_log.txt'}")
