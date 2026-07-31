#!/usr/bin/env python3
"""Experiment-machine entry for Stage 7C-Q1.

Example:
  python -m scripts.stage7c_q1.run_experiment \\
    --config configs/stage7c_q1.yaml \\
    --output-root /ABSOLUTE/PATH/OUTSIDE/REPO \\
    --checkpoint-root /ABSOLUTE/PATH/OUTSIDE/REPO_CKPTS \\
    --workers 8 \\
    --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO
    / "experiments"
    / "pilots"
    / "stage7c_q1_baseline_competence"
    / "scripts"
    / "launch_stage7c_q1_parallel.py"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 7C-Q1 parallel launcher")
    p.add_argument(
        "--config",
        "--protocol",
        dest="protocol",
        type=Path,
        default=REPO / "configs" / "stage7c_q1.yaml",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Must be outside the git repo. Defaults to <output-root>_checkpoints.",
    )
    p.add_argument("--workers", "--max-workers", dest="max_workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--checkpoint-write-slots", type=int, default=2)
    p.add_argument("--seeds", default="64001-64020")
    p.add_argument("--max-steps", type=int, default=400_000)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)

    ckpt = args.checkpoint_root
    if ckpt is None:
        ckpt = Path(str(args.output_root) + "_checkpoints")

    # Refuse in-repo roots
    for label, path in (("output-root", args.output_root), ("checkpoint-root", ckpt)):
        try:
            Path(path).resolve().relative_to(REPO.resolve())
            print(f"ABORT: {label} must be outside the git repository", flush=True)
            return 2
        except ValueError:
            pass

    if args.output_root.resolve() == Path(ckpt).resolve():
        print("ABORT: output-root and checkpoint-root must be distinct", flush=True)
        return 2

    forwarded = [
        "--protocol",
        str(args.protocol),
        "--seeds",
        str(args.seeds),
        "--max-workers",
        str(args.max_workers),
        "--threads-per-worker",
        str(args.threads_per_worker),
        "--checkpoint-write-slots",
        str(args.checkpoint_write_slots),
        "--output-root",
        str(args.output_root),
        "--checkpoint-root",
        str(ckpt),
        "--max-steps",
        str(args.max_steps),
    ]
    if args.no_resume:
        forwarded.append("--no-resume")

    spec = importlib.util.spec_from_file_location("stage7c_q1_launch", LAUNCHER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
