#!/usr/bin/env python3
"""RQ1 report: baseline's convention pattern (q, p_MF, D_swap) within
certified choice states, plus its learner-mobility anchor. Runs entirely
against the ALREADY-EXISTING Stage 8 gate data (results/stage8_gate/v1) --
does not require mean_pbrs/min_pbrs training to be finished, since RQ1 is a
single-condition question (Chapter 2 SS2.10: "under the egoistic baseline").

Restricted to the 3 gate checkpoints (350K/375K/400K), matching every other
Stage 8/9 gate-scale metric reported so far.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage9_analysis import (  # noqa: E402
    compute_learner_mobility,
    compute_rq1_metrics,
)

BASELINE_RESULTS = REPO / "results" / "stage8_gate" / "v1"
GATE_CHECKPOINTS = (350_000, 375_000, 400_000)
OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "statistics"


def main() -> int:
    ep_path = BASELINE_RESULTS / "raw" / "evaluation_episodes.csv"
    if not ep_path.is_file():
        print(f"ABORT: {ep_path} missing", file=sys.stderr)
        return 1
    ep = pd.read_csv(ep_path)

    rq1 = compute_rq1_metrics(ep, checkpoint_steps=GATE_CHECKPOINTS)
    mobility = compute_learner_mobility(
        ep,
        BASELINE_RESULTS / "raw" / "trajectories",
        checkpoint_steps=GATE_CHECKPOINTS,
    )

    report = {
        "condition": "baseline",
        "source": "stage8-gate-protocol-v1 (reused verbatim, not retrained)",
        "checkpoints": list(GATE_CHECKPOINTS),
        "rq1": rq1,
        "learner_mobility": mobility,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "stage9_rq1_baseline_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
