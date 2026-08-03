#!/usr/bin/env python3
"""RQ3 "improvement" decision report: does PBRS make the worst-off
stakeholder (min over A, B, B_front, B_rear) better off than baseline, not
merely no-worse (that's `run_stage9_decision.py`'s non-inferiority test,
already run). This was previously untested -- see protocol doc SS7.2 scope
note and STAGE8_PLAN_DRAFT.md's follow-up.

Reads `results/stage9_worst_off/v1/<condition>/raw/` (produced by
`run_stage9_worst_off_evaluation.py`, which re-evaluated the SAME
already-trained checkpoints as the main Stage 9 gate/decision, only with
B_front/B_rear per-step logging enabled -- no retraining, no new policies).

Pooled over the same 3 gate checkpoints (350K/375K/400K) as
`run_stage9_decision.py`'s RQ3 non-inferiority tests, for direct
comparability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage9_analysis import (  # noqa: E402
    compute_worst_off_mobility,
    one_sided_superiority_test,
)

GATE_CHECKPOINTS = (350_000, 375_000, 400_000)
OUT_DIR = (Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "analysis" / "stage9_worst_off" / "v1").resolve()


def _load_condition(results_root: Path, condition: str) -> dict[str, Any]:
    base = results_root / condition
    ep_path = base / "raw" / "evaluation_episodes.csv"
    traj_dir = base / "raw" / "trajectories"
    if not ep_path.is_file():
        return {"status": "MISSING", "path": str(ep_path)}
    ep = pd.read_csv(ep_path)
    mobility = compute_worst_off_mobility(ep, traj_dir, checkpoint_steps=GATE_CHECKPOINTS)
    return {"status": "OK", "mobility": mobility}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=REPO / "results" / "stage9_worst_off" / "v1")
    args = p.parse_args(argv)

    conditions = {c: _load_condition(args.results_root, c) for c in ("baseline", "mean_pbrs", "min_pbrs")}
    missing = [c for c, d in conditions.items() if d["status"] != "OK" or d["mobility"].get("status") != "OK"]

    report: dict[str, Any] = {
        "protocol_tag": "stage9-worst-off-extension-v1",
        "gate_checkpoints": list(GATE_CHECKPOINTS),
        "conditions_status": {
            c: (d["mobility"].get("status") if d["status"] == "OK" else d["status"]) for c, d in conditions.items()
        },
    }

    if missing:
        report["status"] = "INCOMPLETE"
        report["missing_conditions"] = missing
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "STAGE9_WORST_OFF_DECISION.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(report, indent=2, default=str))
        return 3

    report["status"] = "COMPLETE"
    report["descriptive_by_condition"] = {
        c: {
            "mean_U_worst": d["mobility"]["mean_U_worst"],
            "sd_U_worst": d["mobility"]["sd_U_worst"],
            "n_episodes": d["mobility"]["n_episodes"],
            "n_incomplete_episodes": d["mobility"]["n_incomplete_episodes"],
            "controllers_present": d["mobility"]["controllers_present"],
        }
        for c, d in conditions.items()
    }

    per_seed = {c: np.array(list(d["mobility"]["per_seed_mean_U_worst"].values())) for c, d in conditions.items()}

    report["rq3_improvement"] = {
        "mean_pbrs_vs_baseline": one_sided_superiority_test(per_seed["mean_pbrs"], per_seed["baseline"]),
        "min_pbrs_vs_baseline": one_sided_superiority_test(per_seed["min_pbrs"], per_seed["baseline"]),
        "min_pbrs_vs_mean_pbrs": one_sided_superiority_test(per_seed["min_pbrs"], per_seed["mean_pbrs"]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "STAGE9_WORST_OFF_DECISION.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT_DIR / 'STAGE9_WORST_OFF_DECISION.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
