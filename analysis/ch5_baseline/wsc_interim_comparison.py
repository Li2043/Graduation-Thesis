"""INTERIM, NON-FORMAL comparison: Original vs. WSC, matched on whichever
seeds have completed so far. Descriptive only -- no bootstrap CIs (n this
small would make a percentile bootstrap itself misleading), no p-values,
no claims. Designed to be re-run unchanged once all 12 seeds are done,
at which point it becomes the real analysis with the project's usual
10,000-resample bootstrap added.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

WSC_DIR = Path(__file__).resolve().parent / "outputs" / "wsc_interim"
ORIG_BASELINE_CSV = Path(__file__).resolve().parent.parent / "data" / "taskonly_evaluation_merged.csv"
ORIG_WELFARE_CSV = Path(__file__).resolve().parent.parent / "pooled12" / "outputs" / "pooled12_welfare_evaluation_merged.csv"

SEEDS = [900101, 900102, 900103, 900104, 910101]
CONDS = ["baseline", "mean", "ggi", "maximin"]

print("=" * 78)
print("INTERIM COMPARISON -- Original vs. WSC, n=%d seeds ONLY (not the full 12)." % len(SEEDS))
print("NOT a formal result. Descriptive means only, no inferential claims.")
print("Seeds:", SEEDS)
print("=" * 78)

# ---- load WSC interim results ----
wsc_frames = [pd.read_csv(p) for p in glob.glob(str(WSC_DIR / "wsc_interim_evaluation_seed*.csv"))]
wsc = pd.concat(wsc_frames, ignore_index=True)
wsc = wsc[wsc.seed.isin(SEEDS)]
for s in SEEDS:
    for c in CONDS:
        n = len(wsc[(wsc.seed == s) & (wsc.condition == c)])
        if n != 256:
            print(f"  WARNING: {c}_wsc_{s} has {n} rows, expected 256")

# ---- load Original results (same seeds) ----
orig_base = pd.read_csv(ORIG_BASELINE_CSV)
orig_base = orig_base[(orig_base.bank == "H1") & (orig_base.seed.isin(SEEDS))].copy()
orig_base["condition"] = "baseline"
orig_welfare = pd.read_csv(ORIG_WELFARE_CSV)
orig_welfare = orig_welfare[(orig_welfare.bank == "H1") & (orig_welfare.seed.isin(SEEDS))].copy()
orig = pd.concat([orig_base, orig_welfare], ignore_index=True)


def seed_mean(df, cond, field, seed):
    v = df[(df.condition == cond) & (df.seed == seed)][field]
    return float(v.mean()) if len(v) else float("nan")


rows = []
for s in SEEDS:
    row = {"seed": s}
    for c in CONDS:
        row[f"orig_{c}_umin"] = seed_mean(orig, c, "min_U", s)
        row[f"orig_{c}_gini"] = seed_mean(orig, c, "gini", s)
        row[f"orig_{c}_completion"] = seed_mean(orig, c, "completion", s)
        row[f"wsc_{c}_umin"] = seed_mean(wsc, c, "min_U", s)
        row[f"wsc_{c}_gini"] = seed_mean(wsc, c, "gini", s)
        row[f"wsc_{c}_completion"] = seed_mean(wsc, c, "completion", s)
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(WSC_DIR / "wsc_interim_seed_level_comparison.csv", index=False)

print("\n--- Seed-level U_min: Original vs WSC, by condition ---")
for c in CONDS:
    print(f"\n{c.upper()}:")
    for _, r in df.iterrows():
        print(f"  seed {int(r['seed'])}: orig_Umin={r[f'orig_{c}_umin']:.4f}  wsc_Umin={r[f'wsc_{c}_umin']:.4f}  "
              f"delta(wsc-orig)={r[f'wsc_{c}_umin']-r[f'orig_{c}_umin']:+.4f}")
    o = df[f"orig_{c}_umin"].mean(); w = df[f"wsc_{c}_umin"].mean()
    print(f"  MEAN across these {len(df)} seeds: orig={o:.4f}  wsc={w:.4f}  delta={w-o:+.4f}")

print("\n--- Seed-level Utility Gini: Original vs WSC, by condition ---")
for c in CONDS:
    print(f"\n{c.upper()}:")
    for _, r in df.iterrows():
        print(f"  seed {int(r['seed'])}: orig_Gini={r[f'orig_{c}_gini']:.4f}  wsc_Gini={r[f'wsc_{c}_gini']:.4f}  "
              f"delta(wsc-orig)={r[f'wsc_{c}_gini']-r[f'orig_{c}_gini']:+.4f}")
    o = df[f"orig_{c}_gini"].mean(); w = df[f"wsc_{c}_gini"].mean()
    print(f"  MEAN across these {len(df)} seeds: orig={o:.4f}  wsc={w:.4f}  delta={w-o:+.4f}")

print("\n--- Completion: Original vs WSC, by condition ---")
for c in CONDS:
    o = df[f"orig_{c}_completion"].mean(); w = df[f"wsc_{c}_completion"].mean()
    print(f"  {c}: orig={o:.4f}  wsc={w:.4f}  delta={w-o:+.4f}")

print("\n--- Within-WSC: does WSC's own Mean/GGI/Maximin beat WSC's own Baseline? (n=%d) ---" % len(df))
for c in ("mean", "ggi", "maximin"):
    d_umin = df[f"wsc_{c}_umin"] - df["wsc_baseline_umin"]
    d_gini = df[f"wsc_{c}_gini"] - df["wsc_baseline_gini"]
    print(f"  {c}_wsc - baseline_wsc: mean_delta_Umin={d_umin.mean():+.4f} (per-seed: {[round(x,3) for x in d_umin]})  "
          f"mean_delta_Gini={d_gini.mean():+.4f} (per-seed: {[round(x,3) for x in d_gini]})")

print(f"\nwrote {WSC_DIR / 'wsc_interim_seed_level_comparison.csv'}")
print("\nREMINDER: n=%d seeds, no bootstrap/inferential claim attached. This is NOT the formal result." % len(df))
