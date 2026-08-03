#!/usr/bin/env python3
"""Stage 9 RQ1-RQ3 decision report.

Not a PASS/FAIL/INVALID gate (that is the Stage 8 gate's role, already
resolved: FAIL). This produces a descriptive + inferential report:

  RQ1 -- baseline's convention pattern (q, p_MF, D_swap) within certified
         choice states.
  RQ2 -- do mean_pbrs / min_pbrs change q / D_swap relative to baseline,
         and does min_pbrs change them relative to mean_pbrs (the necessary
         aggregation-rule-isolated control, per Chapter 2 SS2.5)? Reported
         via `two_sample_diff_test` (protocol doc SS7.1): significant AND
         >= the pre-registered MES (0.20) is required to call a difference
         "detected" at this design's resolution.
  RQ3 -- does min_pbrs avoid material loss vs baseline on collision rate,
         mean learner mobility, and q? `non_inferiority_test` against the
         frozen margins (protocol doc SS7.2): collision <=0.03 absolute,
         mobility <=10% relative (anchor U~=0.8968), q <=0.05 absolute.

Requires baseline (reused, `results/stage8_gate/v1`) and BOTH mean_pbrs and
min_pbrs evaluation output (`--results-root/<condition>/raw/`, produced by
`run_stage9_evaluation.py`) to be present -- reports INVALID_INCOMPLETE
per condition if missing rather than silently skipping a comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage9_analysis import (  # noqa: E402
    compute_learner_mobility,
    compute_rq1_metrics,
    compute_rq1_metrics_per_seed,
    non_inferiority_test,
    two_sample_diff_test,
)
from thesis.pilots.stage9_config import (  # noqa: E402
    MARGIN_COLLISION_RATE_ABS,
    MARGIN_MEAN_MOBILITY_RELATIVE,
    MARGIN_RESOLUTION_Q_ABS,
    MES_SUCCESS_RATE,
    MES_SWAP_ELIGIBILITY,
)

GATE_CHECKPOINTS = (350_000, 375_000, 400_000)
OUT_DIR = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "analysis" / "stage9_confirmatory" / "v1"
OUT_DIR = OUT_DIR.resolve()


def _load_condition(results_root: Path, condition: str, *, reused_from: Path | None = None) -> dict[str, Any]:
    base = reused_from if reused_from is not None else (results_root / condition)
    ep_path = base / "raw" / "evaluation_episodes.csv"
    traj_dir = base / "raw" / "trajectories"
    if not ep_path.is_file():
        return {"status": "MISSING", "path": str(ep_path)}
    ep = pd.read_csv(ep_path)
    rq1 = compute_rq1_metrics(ep, checkpoint_steps=GATE_CHECKPOINTS)
    per_seed = compute_rq1_metrics_per_seed(ep, checkpoint_steps=GATE_CHECKPOINTS)
    mobility = compute_learner_mobility(ep, traj_dir, checkpoint_steps=GATE_CHECKPOINTS)
    n_seeds = int(ep["master_seed"].nunique())
    n_checkpoints = int(ep.groupby("master_seed")["checkpoint_step"].nunique().min()) if n_seeds else 0
    return {
        "status": "OK",
        "n_seeds": n_seeds,
        "min_checkpoints_present_per_seed": n_checkpoints,
        "rq1": rq1,
        "per_seed": per_seed,
        "mobility": mobility,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-root",
        type=Path,
        default=REPO / "results" / "stage9_confirmatory" / "v1",
        help="root containing <condition>/raw/... for mean_pbrs and min_pbrs",
    )
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO / "results" / "stage8_gate" / "v1",
        help="reused Stage 8 gate results (baseline, not retrained)",
    )
    args = p.parse_args(argv)

    conditions: dict[str, dict[str, Any]] = {
        "baseline": _load_condition(args.results_root, "baseline", reused_from=Path(args.baseline_root)),
        "mean_pbrs": _load_condition(args.results_root, "mean_pbrs"),
        "min_pbrs": _load_condition(args.results_root, "min_pbrs"),
    }

    missing = [c for c, d in conditions.items() if d["status"] != "OK"]

    report: dict[str, Any] = {
        "protocol_tag": "stage9-confirmatory-v1",
        "gate_checkpoints": list(GATE_CHECKPOINTS),
        "conditions_status": {c: d["status"] for c, d in conditions.items()},
    }

    if missing:
        report["status"] = "INCOMPLETE"
        report["missing_conditions"] = missing
        report["note"] = (
            "One or more conditions have no evaluation data yet -- RQ1 is "
            "reported for whichever conditions ARE present; RQ2/RQ3 "
            "comparisons requiring a missing condition are skipped, not "
            "fabricated."
        )
    else:
        report["status"] = "COMPLETE"

    # RQ1: per-condition descriptive report, for every condition that has data.
    report["rq1_by_condition"] = {
        c: {
            "q_safe_resolution_rate": d["rq1"].get("q_safe_resolution_rate"),
            "p_MF_mainline_first_given_resolved": d["rq1"].get("p_MF_mainline_first_given_resolved"),
            "D_swap_complementary_order_rate": d["rq1"].get("D_swap_complementary_order_rate"),
            "n_certified_episodes": d["rq1"].get("n_certified_episodes"),
            "mean_learner_mobility_U": d["mobility"].get("mean_U"),
        }
        for c, d in conditions.items()
        if d["status"] == "OK"
    }

    # RQ2: baseline vs mean_pbrs, mean_pbrs vs min_pbrs (aggregation-rule-isolated control).
    report["rq2"] = {}
    pairs = [("mean_pbrs", "baseline"), ("min_pbrs", "mean_pbrs")]
    for treat, ctrl in pairs:
        if conditions[treat]["status"] != "OK" or conditions[ctrl]["status"] != "OK":
            report["rq2"][f"{treat}_vs_{ctrl}"] = {"status": "SKIPPED_MISSING_DATA"}
            continue
        pt = conditions[treat]["per_seed"]
        pc = conditions[ctrl]["per_seed"]
        report["rq2"][f"{treat}_vs_{ctrl}"] = {
            "q": two_sample_diff_test(pt["q"].to_numpy(), pc["q"].to_numpy(), mes=MES_SUCCESS_RATE),
            "D_swap": two_sample_diff_test(
                pt["D_swap"].dropna().to_numpy(), pc["D_swap"].dropna().to_numpy(), mes=MES_SWAP_ELIGIBILITY
            ),
        }

    # RQ3: min_pbrs vs baseline, non-inferiority on collision / mobility / q.
    report["rq3_min_pbrs_vs_baseline"] = {"status": "SKIPPED_MISSING_DATA"}
    if conditions["min_pbrs"]["status"] == "OK" and conditions["baseline"]["status"] == "OK":
        t = conditions["min_pbrs"]["per_seed"]
        c = conditions["baseline"]["per_seed"]
        t_mob = conditions["min_pbrs"]["mobility"].get("per_seed_mean_U", {})
        c_mob = conditions["baseline"]["mobility"].get("per_seed_mean_U", {})
        import numpy as np

        report["rq3_min_pbrs_vs_baseline"] = {
            "status": "OK",
            "collision_rate": non_inferiority_test(
                t["collision_rate"].to_numpy(),
                c["collision_rate"].to_numpy(),
                margin=MARGIN_COLLISION_RATE_ABS,
                higher_is_better=False,
            ),
            "mean_learner_mobility": non_inferiority_test(
                np.array(list(t_mob.values())),
                np.array(list(c_mob.values())),
                margin=MARGIN_MEAN_MOBILITY_RELATIVE * float(np.mean(list(c_mob.values()))) if c_mob else 0.0,
                higher_is_better=True,
            ),
            "q": non_inferiority_test(
                t["q"].to_numpy(), c["q"].to_numpy(), margin=MARGIN_RESOLUTION_Q_ABS, higher_is_better=True
            ),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "STAGE9_CONFIRMATORY_DECISION.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT_DIR / 'STAGE9_CONFIRMATORY_DECISION.json'}")
    return 0 if report["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
