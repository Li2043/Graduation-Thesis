#!/usr/bin/env python3
"""Who is the worst-off stakeholder, and how often? Chapter 3 SS3.10.2 lists
"worst-off stakeholder identity" as a secondary outcome -- this computes it
directly from the already-written B_front/B_rear-enabled trajectory data
(results/stage9_worst_off/v1), no new evaluation needed.

Tests a specific alternative account for the RQ3-improvement null result:
if the argmin controller is disproportionately a background (uncontrolled)
vehicle, the learner's action space may have little to no causal leverage
over the quantity min_pbrs's shaping signal is trying to move, independent
of whether real inequality exists between stakeholders (it does -- see
Table 4.4's Ubar/U^min gap).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage9_analysis import compute_stakeholder_mobility  # noqa: E402

GATE_CHECKPOINTS = (350_000, 375_000, 400_000)
SOURCES = {
    "baseline": REPO / "results" / "stage9_worst_off" / "v1" / "baseline",
    "mean_pbrs": REPO / "results" / "stage9_worst_off" / "v1" / "mean_pbrs",
    "min_pbrs": REPO / "results" / "stage9_worst_off" / "v1" / "min_pbrs",
}
EPISODE_KEYS = ["master_seed", "checkpoint_step", "validation_block_id", "assignment", "scenario_block", "episode_index"]


def main() -> int:
    for cond, root in SOURCES.items():
        ep = pd.read_csv(root / "raw" / "evaluation_episodes.csv")
        traj_dir = root / "raw" / "trajectories"
        detail = compute_stakeholder_mobility(ep, traj_dir, checkpoint_steps=GATE_CHECKPOINTS)
        detail = detail.dropna(subset=["U_i"])

        counts = detail.groupby(EPISODE_KEYS)["controller"].nunique()
        complete = counts[counts == 4].index
        detail = detail.set_index(EPISODE_KEYS).loc[complete].reset_index()

        idxmin = detail.loc[detail.groupby(EPISODE_KEYS)["U_i"].idxmin()]
        n = len(idxmin)
        vc = idxmin["controller"].value_counts()
        frac = (vc / n * 100).round(1)

        learner_frac = vc.get("A", 0) + vc.get("B", 0)
        bg_frac = vc.get("B_front", 0) + vc.get("B_rear", 0)

        print(f"=== {cond} (n_episodes={n}) ===")
        for ctrl in ("A", "B", "B_front", "B_rear"):
            print(f"  worst-off = {ctrl}: {vc.get(ctrl, 0)} ({frac.get(ctrl, 0.0)}%)")
        print(f"  learner (A+B) total: {learner_frac} ({learner_frac/n*100:.1f}%)")
        print(f"  background (B_front+B_rear) total: {bg_frac} ({bg_frac/n*100:.1f}%)")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
