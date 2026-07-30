#!/usr/bin/env python3
"""Launch Stage 7A-1 seeds (optionally parallel OS processes)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]
SEED_RUNNER = SCRIPT.parent / "run_stage7a1_seed.py"


def _run_one(seed: int, max_steps: int) -> tuple[int, int]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    py = sys.executable
    proc = subprocess.run(
        [
            py,
            str(SEED_RUNNER),
            "--seed",
            str(seed),
            "--max-steps",
            str(max_steps),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    log = PILOT_ROOT / "logs" / f"seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    return seed, proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="62001-62020")
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    if "-" in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = {
            ex.submit(_run_one, s, int(args.max_steps)): s for s in seeds
        }
        for fut in as_completed(futs):
            seed, rc = fut.result()
            results[seed] = rc
            print(f"seed {seed} rc={rc}", flush=True)

    failed = [s for s, rc in results.items() if rc != 0]
    summary = PILOT_ROOT / "logs" / "launch_summary.json"
    import json

    summary.write_text(
        json.dumps(
            {
                "results": results,
                "failed": failed,
                "elapsed_sec": time.time() - t0,
                "workers": args.workers,
                "max_steps": args.max_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done failed={failed} elapsed={time.time()-t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
