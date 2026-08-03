#!/usr/bin/env python3
"""Parallel launcher for Stage 9 (mean_pbrs + min_pbrs, 20 seeds each, 40
total training jobs, 400,000 steps each). Baseline is reused from the
Stage 8 gate and is never launched by this script.

Default --max-workers/--threads-per-worker match the Stage 8 gate launcher's
defaults (tuned for a 32-core machine) -- override for other hardware. See
that script's docstring for the process-vs-thread-parallelism reasoning
(unchanged here: tiny 64x64 MLP, process-level parallelism across
independent seeds is what actually uses extra cores).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]
JOB_SCRIPT = SCRIPT.parent / "run_stage9_training.py"

sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis.pilots.stage9_config import SEEDS_BY_CONDITION, TRAINED_CONDITIONS  # noqa: E402


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol",
        type=Path,
        default=PILOT_ROOT / "configs" / "stage9_confirmatory_protocol.yaml",
    )
    p.add_argument(
        "--conditions",
        default=",".join(TRAINED_CONDITIONS),
        help="comma-separated subset of mean_pbrs,min_pbrs (default: both)",
    )
    p.add_argument(
        "--seeds",
        default=None,
        help="comma/range spec, applied identically to every selected condition; "
        "default: each condition's own full 20-seed block",
    )
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--threads-per-worker", type=int, default=2)
    p.add_argument("--checkpoint-write-slots", type=int, default=4)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=400_000)
    args = p.parse_args(argv)

    try:
        Path(args.checkpoint_root).resolve().relative_to(REPO_ROOT.resolve())
        print("ABORT: --checkpoint-root must be outside the git repository", flush=True)
        return 2
    except ValueError:
        pass

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in TRAINED_CONDITIONS:
            print(f"ABORT: condition {c!r} not in {TRAINED_CONDITIONS!r}", flush=True)
            return 2

    jobs: list[tuple[str, int]] = []
    for c in conditions:
        seeds = _parse_seeds(args.seeds) if args.seeds else list(SEEDS_BY_CONDITION[c])
        for s in seeds:
            if s not in SEEDS_BY_CONDITION[c]:
                print(f"ABORT: seed {s} not in {c}'s frozen seed block {SEEDS_BY_CONDITION[c]}", flush=True)
                return 2
            jobs.append((c, s))

    write_sem = threading.Semaphore(int(args.checkpoint_write_slots))
    lock_path = PILOT_ROOT / "logs" / "launch_stage9.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        os.close(fd)
    except FileExistsError:
        print(f"ABORT: launcher already running ({lock_path})", flush=True)
        return 2

    status: dict[str, dict] = {}
    failed: list[str] = []

    def run_one(condition: str, seed: int) -> tuple[str, int, int]:
        with write_sem:
            out = Path(args.output_root)
            ckpt = Path(args.checkpoint_root)
            out.mkdir(parents=True, exist_ok=True)
            ckpt.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src")
            env["OMP_NUM_THREADS"] = str(args.threads_per_worker)
            env["MKL_NUM_THREADS"] = str(args.threads_per_worker)
            env["OPENBLAS_NUM_THREADS"] = str(args.threads_per_worker)
            env["NUMEXPR_NUM_THREADS"] = str(args.threads_per_worker)
            cmd = [
                sys.executable,
                str(JOB_SCRIPT),
                "--condition",
                condition,
                "--protocol",
                str(args.protocol),
                "--master-seed",
                str(seed),
                "--output-root",
                str(args.output_root),
                "--checkpoint-root",
                str(args.checkpoint_root),
                "--max-steps",
                str(args.max_steps),
                "--threads",
                str(args.threads_per_worker),
            ]
            if args.no_resume:
                cmd.append("--no-resume")
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
            return condition, seed, int(proc.returncode)

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
            futs = [ex.submit(run_one, c, s) for c, s in jobs]
            for fut in as_completed(futs):
                condition, seed, code = fut.result()
                key = f"{condition}_{seed}"
                status[key] = {"condition": condition, "seed": seed, "returncode": code}
                if code != 0:
                    failed.append(key)
                print(f"{condition} seed {seed}: {'OK' if code == 0 else 'FAILED'}", flush=True)
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass

    summary = {
        "planned": len(jobs),
        "conditions": conditions,
        "failed": sorted(failed),
        "elapsed_sec": time.time() - t0,
        "status": status,
        "output_root": str(Path(args.output_root).resolve()),
        "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
    }
    (Path(args.output_root) / "logs").mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "logs" / "launch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
