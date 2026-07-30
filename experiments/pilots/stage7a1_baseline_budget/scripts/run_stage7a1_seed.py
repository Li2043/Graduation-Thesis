#!/usr/bin/env python3
"""Run one Stage 7A-1 Baseline seed to 300K."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]


def _configure_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    _configure_threads()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.pilots.stage7a1_runner import run_seed

    protocol = PILOT_ROOT / "configs" / "stage7a1_baseline_budget_protocol.yaml"
    result = run_seed(
        master_seed=int(args.seed),
        pilot_root=PILOT_ROOT,
        protocol_path=protocol,
        max_steps=int(args.max_steps),
        resume=not args.no_resume,
        force_rerun=bool(args.force_rerun),
        skip_eval=bool(args.skip_eval),
    )
    print(json_dumps(result))
    return 0 if result.get("success") or result.get("skipped_completed") else 1


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
