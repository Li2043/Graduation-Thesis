#!/usr/bin/env python3
"""Report Chapter 3's five formally-defined primary endpoints (SS3.10.1) for
all three Stage 9 conditions, pooled over the three gate checkpoints
(350K/375K/400K):

  1. evaluation success rate       (unrestricted, all evaluation episodes)
  2. stakeholder-collision rate    (unrestricted, all evaluation episodes)
  3. mean stakeholder utility      U-bar, Eq. 3.14/3.48/3.50 (4 stakeholders)
  4. minimum stakeholder utility   U^min, Eq. 3.15/3.49/3.51 (4 stakeholders)
  5. convention consistency        kappa, Eq. 3.41-3.45

(1)-(2) come from seed_checkpoint_summary.csv (already computed by the
evaluation drivers). (3)-(4) come from results/stage9_worst_off/v1 (the
B_front/B_rear-logging-enabled re-evaluation). (5) comes from
evaluation_episodes.csv's `passing_order` column, unrestricted -- Chapter
3's own definition, not the certified-state-restricted D_swap/p_MF this
chapter's RQ1/RQ2 tables otherwise use.

No training, no new evaluation -- pure aggregation over already-written CSVs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage9_analysis import (  # noqa: E402
    compute_convention_consistency_per_seed,
    compute_mean_stakeholder_utility,
    compute_worst_off_mobility,
)

GATE_CHECKPOINTS = (350_000, 375_000, 400_000)

SOURCES = {
    "baseline": REPO / "results" / "stage8_gate" / "v1",
    "mean_pbrs": REPO / "results" / "stage9_confirmatory" / "v1" / "mean_pbrs",
    "min_pbrs": REPO / "results" / "stage9_confirmatory" / "v1" / "min_pbrs",
}
WORST_OFF_SOURCES = {
    "baseline": REPO / "results" / "stage9_worst_off" / "v1" / "baseline",
    "mean_pbrs": REPO / "results" / "stage9_worst_off" / "v1" / "mean_pbrs",
    "min_pbrs": REPO / "results" / "stage9_worst_off" / "v1" / "min_pbrs",
}


def main() -> int:
    report: dict[str, dict] = {}
    for cond, root in SOURCES.items():
        sc = pd.read_csv(root / "raw" / "seed_checkpoint_summary.csv")
        sc_gate = sc[sc["checkpoint_step"].isin(GATE_CHECKPOINTS)]
        per_seed_sc = sc_gate.groupby("master_seed")[["success_rate", "collision_rate"]].mean()

        ep = pd.read_csv(root / "raw" / "evaluation_episodes.csv")
        kappa_df = compute_convention_consistency_per_seed(ep, checkpoint_steps=GATE_CHECKPOINTS)

        wo_root = WORST_OFF_SOURCES[cond]
        wo_ep = pd.read_csv(wo_root / "raw" / "evaluation_episodes.csv")
        traj_dir = wo_root / "raw" / "trajectories"
        u_bar = compute_mean_stakeholder_utility(wo_ep, traj_dir, checkpoint_steps=GATE_CHECKPOINTS)
        u_min = compute_worst_off_mobility(wo_ep, traj_dir, checkpoint_steps=GATE_CHECKPOINTS)

        kappa_vals = kappa_df["kappa"].dropna().to_numpy(dtype=float)

        report[cond] = {
            "n_seeds": int(per_seed_sc.shape[0]),
            "success_rate": {
                "mean": float(per_seed_sc["success_rate"].mean()),
                "sd": float(per_seed_sc["success_rate"].std(ddof=1)),
            },
            "collision_rate": {
                "mean": float(per_seed_sc["collision_rate"].mean()),
                "sd": float(per_seed_sc["collision_rate"].std(ddof=1)),
            },
            "U_bar": {"mean": u_bar["mean_U_bar"], "sd": u_bar["sd_U_bar"], "n_episodes": u_bar["n_episodes"]},
            "U_min": {"mean": u_min["mean_U_worst"], "sd": u_min["sd_U_worst"], "n_episodes": u_min["n_episodes"]},
            "kappa": {
                "mean": float(np.mean(kappa_vals)) if len(kappa_vals) else None,
                "sd": float(np.std(kappa_vals, ddof=1)) if len(kappa_vals) > 1 else None,
                "n_seeds_defined": int(len(kappa_vals)),
                "n_seeds_total": int(len(kappa_df)),
            },
            "per_seed_U_bar": u_bar["per_seed_mean_U_bar"],
            "per_seed_U_min": u_min["per_seed_mean_U_worst"],
            "per_seed_kappa": {int(r.master_seed): (None if pd.isna(r.kappa) else float(r.kappa)) for r in kappa_df.itertuples()},
        }

    print(json.dumps(report, indent=2))

    out = REPO / "analysis" / "stage9_worst_off" / "v1" / "CHAPTER3_PRIMARY_ENDPOINTS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
