"""Task A3/A4 (followup prompt): valid pre-branch moderation analysis.

Correlates C64 (pre-branch, 1.2M-step) policy quality against the
condition-minus-BASELINE welfare effect at 2.0M (matched by seed), for
Mean/GGI/Maximin separately. Deliberately does NOT use (condition - Mean)
or (GGI - Mean) as the x/y pairing -- Mean's own value never appears on
both sides of any correlation computed here.

Inputs (read-only):
  - c64_prebranch_h1_merged.csv  (this task's own new C64-on-H1 evaluation,
    12 seeds x 256 H1 scenarios, produced by evaluate_c64_prebranch_h1.py)
  - all4_conditions_merged.csv  (existing Phase-1 artifact: Baseline/Mean/
    GGI/Maximin at 2.0M, H1, 12 seeds, already merged and used elsewhere
    in this chapter)

Outputs (this task's own new artifacts, under ch5_baseline/outputs/):
  - table_c64_prebranch_12seed.csv       (A: 12-seed C64 pre-branch metrics)
  - table_condition_minus_baseline_12seed.csv (B: 12-seed Delta_c vs Baseline)
  - table_prebranch_moderation_corr.csv  (C: r/rho/bootstrap CI per relationship)
  - fig5_15_prebranch_moderation.png/.pdf (D: scatter, seed-labelled)
"""
from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])

OUT = Path(__file__).resolve().parent / "outputs"
FIG_DIR = Path(os.environ.get("THESIS_FIGURES_DIR", str(Path(__file__).resolve().parent / "outputs" / "figures")))

SEEDS12 = [900101, 900102, 900103, 900104, 910101, 910102,
           920101, 920102, 920103, 920104, 920105, 920106]

REPORT = []
def log(msg=""):
    print(msg); REPORT.append(str(msg))


def seed_mean(sub: pd.DataFrame, field: str, seed: int) -> float:
    v = sub[sub.seed == seed][field]
    return float(v.mean()) if len(v) else float("nan")


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int = 10_000, seed0: int = 0):
    rng = np.random.default_rng(seed0)
    n = len(x)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            boot[i] = np.nan
        else:
            boot[i] = np.corrcoef(xi, yi)[0, 1]
    boot = boot[~np.isnan(boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(boot)


# ---------------------------------------------------------------------
# A2/validation: load and validate the new C64-on-H1 evaluation
# ---------------------------------------------------------------------
c64 = pd.read_csv(OUT / "c64_prebranch_h1_merged.csv")
log(f"C64-on-H1 rows loaded: {len(c64)}")
assert set(c64.seed.unique()) == set(SEEDS12), f"seed mismatch: {sorted(c64.seed.unique())}"
for s in SEEDS12:
    n = len(c64[c64.seed == s])
    assert n == 256, f"seed {s}: expected 256 H1 episodes, got {n}"
assert (c64.bank == "H1").all()
log("Validated: all 12 seeds present, all 256 H1 scenarios present per seed, bank=H1 throughout.")

c64_table = []
for s in SEEDS12:
    sub = c64[c64.seed == s]
    row = {
        "seed": s,
        "c64_completion": seed_mean(c64, "completion", s),
        "c64_collision": seed_mean(c64, "collision", s),
        "c64_timeout": seed_mean(c64, "timeout", s),
        "c64_umin": seed_mean(c64, "min_U", s),
        "c64_gini": seed_mean(c64, "gini", s),
        "c64_mean_U": seed_mean(c64, "mean_U", s),
    }
    c64_table.append(row)
c64_df = pd.DataFrame(c64_table)
c64_df.to_csv(OUT / "table_c64_prebranch_12seed.csv", index=False)
log("\n12-seed C64 pre-branch table:")
log(c64_df.to_string(index=False))

# ---------------------------------------------------------------------
# A3: matched 2.0M Baseline/Mean/GGI/Maximin (existing Phase-1 artifact)
# ---------------------------------------------------------------------
df4_all = pd.read_csv(OUT / "all4_conditions_merged.csv")
df4 = df4_all[df4_all.bank == "H1"].reset_index(drop=True)
for s in SEEDS12:
    for cond in ("baseline", "mean", "ggi", "maximin"):
        n = len(df4[(df4.seed == s) & (df4.condition == cond)])
        assert n == 256, f"{cond} seed {s}: expected 256 H1 episodes at 2.0M, got {n}"
log("\nValidated: all 12 seeds x 4 conditions x 256 H1 episodes present at 2.0M (bank=H1 only).")

delta_table = []
for s in SEEDS12:
    b_umin = seed_mean(df4[df4.condition == "baseline"], "min_U", s)
    b_gini = seed_mean(df4[df4.condition == "baseline"], "gini", s)
    b_comp = seed_mean(df4[df4.condition == "baseline"], "completion", s)
    row = {"seed": s, "baseline_umin": b_umin, "baseline_gini": b_gini, "baseline_completion": b_comp}
    for cond in ("mean", "ggi", "maximin"):
        row[f"delta_umin_{cond}"] = seed_mean(df4[df4.condition == cond], "min_U", s) - b_umin
        row[f"delta_gini_{cond}"] = seed_mean(df4[df4.condition == cond], "gini", s) - b_gini
        row[f"delta_completion_{cond}"] = seed_mean(df4[df4.condition == cond], "completion", s) - b_comp
    delta_table.append(row)
delta_df = pd.DataFrame(delta_table)
delta_df.to_csv(OUT / "table_condition_minus_baseline_12seed.csv", index=False)
log("\n12-seed condition-minus-Baseline effects table:")
log(delta_df.to_string(index=False))

# ---------------------------------------------------------------------
# A4: pre-branch quality vs later welfare response
# ---------------------------------------------------------------------
merged = c64_df.merge(delta_df, on="seed")

RELATIONSHIPS = [
    ("primary", "c64_umin", "delta_umin_{c}", "C64 U_min vs Delta_Umin"),
    ("primary", "c64_completion", "delta_umin_{c}", "C64 completion vs Delta_Umin"),
    ("secondary", "c64_gini", "delta_gini_{c}", "C64 Gini vs Delta_Gini"),
    ("secondary", "c64_completion", "delta_completion_{c}", "C64 completion vs Delta_completion"),
]

corr_rows = []
for kind, xcol, ytpl, label in RELATIONSHIPS:
    for cond in ("mean", "ggi", "maximin"):
        ycol = ytpl.format(c=cond)
        x = merged[xcol].to_numpy(dtype=float)
        y = merged[ycol].to_numpy(dtype=float)
        r = float(np.corrcoef(x, y)[0, 1])
        rho = spearman_rho(x, y)
        lo, hi, n_valid = bootstrap_corr(x, y)
        corr_rows.append({
            "kind": kind, "relationship": label, "condition": cond,
            "x": xcol, "y": ycol, "pearson_r": round(r, 4), "spearman_rho": round(rho, 4),
            "boot_ci_lo": round(lo, 4), "boot_ci_hi": round(hi, 4), "n_boot_valid": n_valid, "n_seeds": len(x),
        })
        log(f"[{kind:9s}] {label:32s} {cond:8s}: r={r:+.3f} rho={rho:+.3f} "
            f"boot95%CI=[{lo:+.3f},{hi:+.3f}]")

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUT / "table_prebranch_moderation_corr.csv", index=False)

# ---------------------------------------------------------------------
# Scatter figure: the two PRIMARY relationships, seed-labelled, all 3 conditions
# ---------------------------------------------------------------------
COND_COLORS = {"mean": "#1f77b4", "ggi": "#2ca02c", "maximin": "#d62728"}
COND_LABELS = {"mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, xcol, ytpl, title in [
    (axes[0], "c64_umin", "delta_umin_{c}", "C64 $U_{\\min}$ vs. $\\Delta U_{\\min}$ (condition $-$ Baseline)"),
    (axes[1], "c64_completion", "delta_umin_{c}", "C64 completion vs. $\\Delta U_{\\min}$ (condition $-$ Baseline)"),
]:
    x = merged[xcol].to_numpy(dtype=float)
    for cond in ("mean", "ggi", "maximin"):
        y = merged[ytpl.format(c=cond)].to_numpy(dtype=float)
        r = float(np.corrcoef(x, y)[0, 1])
        ax.scatter(x, y, color=COND_COLORS[cond], s=60, label=f"{COND_LABELS[cond]} (r={r:+.2f})", zorder=3)
    for xi, yi, s in zip(x, merged["delta_umin_mean"].to_numpy(dtype=float), merged["seed"]):
        ax.annotate(str(s), (xi, yi), fontsize=6.5, xytext=(3, 3), textcoords="offset points", color="#555")
    ax.axhline(0, color="gray", linewidth=1, alpha=0.6)
    ax.set_xlabel(xcol.replace("c64_", "C64 "))
    ax.set_ylabel("$\\Delta U_{\\min}$ (condition $-$ Baseline)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8.5)
fig.suptitle("Pre-branch (C64) policy quality vs. welfare-minus-Baseline response (n=12, exploratory)", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_15_prebranch_moderation.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_15_prebranch_moderation.pdf", bbox_inches="tight")
plt.close(fig)
log("\nwrote fig5_15_prebranch_moderation.png/.pdf")

with open(OUT / "prebranch_moderation_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT) + "\n")
log(f"\nwrote report -> {OUT / 'prebranch_moderation_report.txt'}")
