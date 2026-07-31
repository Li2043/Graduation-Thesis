#!/usr/bin/env python3
"""Single Stage 7B-A1 training job."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]


def _configure_threads(n: int = 1) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    import torch

    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 7B-A1 single training job")
    p.add_argument("--protocol", type=Path, default=PILOT_ROOT / "configs" / "stage7b_a1_protocol.yaml")
    p.add_argument("--condition", required=True, choices=["vanilla_dqn", "double_dqn"])
    p.add_argument("--master-seed", type=int, required=True)
    p.add_argument("--output-root", type=Path, default=PILOT_ROOT / "output")
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--strict", action="store_true", default=True)
    p.add_argument("--no-strict", action="store_true")
    p.add_argument("--max-steps", type=int, default=300_000)
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)

    _configure_threads(int(args.threads))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.pilots.stage7b_a1_runner import run_training_job

    result = run_training_job(
        condition=args.condition,
        master_seed=int(args.master_seed),
        protocol_path=Path(args.protocol),
        output_root=Path(args.output_root),
        checkpoint_root=Path(args.checkpoint_root),
        max_steps=int(args.max_steps),
        resume=not args.no_resume,
        device=args.device,
        strict=not args.no_strict,
        allow_smoke=bool(args.allow_smoke),
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") or result.get("skipped_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
