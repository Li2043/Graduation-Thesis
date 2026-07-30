#!/usr/bin/env python3
"""Stage 7A-1 resume equivalence gate (seed 62001)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interruption-step", type=int, default=25_000)
    parser.add_argument("--comparison-step", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=62_001)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from thesis.pilots.stage7a1_checkpoint import sha256_file
    from thesis.pilots.stage7a1_resume import run_resume_equivalence

    protocol = PILOT_ROOT / "configs" / "stage7a1_baseline_budget_protocol.yaml"
    work = PILOT_ROOT / "output" / "diagnostics" / "resume_equivalence"
    report = run_resume_equivalence(
        work_dir=work,
        protocol_hash=sha256_file(protocol),
        master_seed=int(args.seed),
        interruption_step=int(args.interruption_step),
        comparison_step=int(args.comparison_step),
    )
    # also publish to manifests/
    dest = PILOT_ROOT / "manifests" / "resume_equivalence_report.json"
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
