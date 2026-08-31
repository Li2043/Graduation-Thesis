"""Ad hoc diagnostic (not a thesis-text edit): did each welfare condition's
OWN objective actually improve on the TRAINING scenarios during the
1.2M->2.0M continuation window, under exploration (epsilon>0), regardless
of whether that transferred to the deterministic held-out H1 evaluation?

Reads only the already-written per-seed training manifests (written by
train_dqn_direct_welfare.py's own checkpoint bookkeeping); no new training,
no modification of any frozen file.
"""
from __future__ import annotations
import os

import json
from pathlib import Path

import numpy as np

FINAL_NEW = Path(os.environ.get("FINAL_NEW_BUNDLE", ""))  # raw logs not distributed with this repo; set env var
SEED_REPL = Path(os.environ.get("SEED_REPL_BUNDLE", ""))  # raw logs not distributed with this repo; set env var

SEEDS_ORIG = [900101, 900102, 900103, 900104, 910101, 910102]
SEEDS_NEW = [920101, 920102, 920103, 920104, 920105, 920106]
SEEDS12 = SEEDS_ORIG + SEEDS_NEW
CONDS = ["mean", "ggi", "maximin"]
COND_DIR = {"mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}
# the metric each condition's OWN objective directly optimizes
OWN_METRIC = {"mean": "mean_U_mean", "ggi": "ggi_mean", "maximin": "min_U_mean"}


def manifest_path(cond: str, seed: int) -> Path:
    if seed in SEEDS_ORIG:
        return FINAL_NEW / "checkpoints" / "formal_runs" / f"{cond}_{seed}" / f"seed_{seed}_Formal_{cond}_manifest.json"
    return (SEED_REPL / "checkpoints" / "seed_replication_v1" / "welfare" / str(seed) / COND_DIR[cond]
            / f"seed_{seed}_Formal_{cond}_manifest.json")


def load_series(cond: str, seed: int):
    p = manifest_path(cond, seed)
    d = json.loads(p.read_text(encoding="utf-8"))
    pts = [(c["step"], c["window"]) for c in d["checkpoints"] if c["window"]["episodes"] > 0]
    return pts


rows = []
for cond in CONDS:
    metric = OWN_METRIC[cond]
    print(f"\n===== {cond.upper()}  (own objective = {metric}) =====")
    firsts, lasts, slopes = [], [], []
    for seed in SEEDS12:
        pts = load_series(cond, seed)
        steps = np.array([s for s, _ in pts], dtype=float)
        vals = np.array([w[metric] for _, w in pts], dtype=float)
        first_val, last_val = vals[0], vals[-1]
        # simple linear trend (slope per 100k steps) across the whole window
        slope = float(np.polyfit(steps, vals, 1)[0]) * 100_000
        comp_first, comp_last = pts[0][1]["completion_rate"], pts[-1][1]["completion_rate"]
        print(f"  seed {seed}: n_ckpt={len(pts):2d}  first(step={int(steps[0])})={first_val:.4f} -> "
              f"last(step={int(steps[-1])})={last_val:.4f}   delta={last_val-first_val:+.4f}  "
              f"slope/100k={slope:+.4f}   completion first->last: {comp_first:.3f}->{comp_last:.3f}")
        firsts.append(first_val); lasts.append(last_val); slopes.append(slope)
        rows.append({"condition": cond, "seed": seed, "metric": metric,
                     "first_step": int(steps[0]), "first_val": round(first_val, 4),
                     "last_step": int(steps[-1]), "last_val": round(last_val, 4),
                     "delta": round(last_val - first_val, 4), "slope_per_100k": round(slope, 4)})
    firsts, lasts, slopes = map(np.array, (firsts, lasts, slopes))
    deltas = lasts - firsts
    print(f"  -- across {len(SEEDS12)} seeds: mean delta={deltas.mean():+.4f}  "
          f"n_improved={int((deltas>0).sum())}/{len(SEEDS12)}  mean slope/100k={slopes.mean():+.4f}  "
          f"n_positive_slope={int((slopes>0).sum())}/{len(SEEDS12)}")

OUT = Path(__file__).resolve().parent / "outputs"
import csv
with open(OUT / "training_window_own_objective_trajectory.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {OUT / 'training_window_own_objective_trajectory.csv'}")
