#!/usr/bin/env python3
"""Single Stage 9 training job entrypoint (mean_pbrs or min_pbrs only —
baseline is reused from the Stage 8 gate, never trained here)."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage9_runner import run_training_job  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["mean_pbrs", "min_pbrs"])
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--master-seed", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=400_000)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--strict", action="store_true", default=True)
    p.add_argument("--no-strict", action="store_true")
    args = p.parse_args(argv)

    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = str(args.threads)
    try:
        import torch

        torch.set_num_threads(int(args.threads))
    except Exception:
        pass

    resume = False if args.no_resume else bool(args.resume)
    strict = False if args.no_strict else bool(args.strict)
    log_dir = Path(args.output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.condition}_seed_{args.master_seed}.log"
    try:
        result = run_training_job(
            condition=str(args.condition),
            master_seed=int(args.master_seed),
            protocol_path=Path(args.protocol),
            output_root=Path(args.output_root),
            checkpoint_root=Path(args.checkpoint_root),
            max_steps=int(args.max_steps),
            resume=resume,
            strict=strict,
            allow_smoke=bool(args.smoke),
        )
        log_path.write_text(str(result) + "\n", encoding="utf-8")
        print(result)
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        log_path.write_text(tb, encoding="utf-8")
        print(tb, file=sys.stderr)
        print(f"FAILED condition={args.condition} seed={args.master_seed}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
