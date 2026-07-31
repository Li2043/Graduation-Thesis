#!/usr/bin/env python3
"""Parallel launcher for Stage 7C-Q1 (20 Double-DQN baseline seeds)."""

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
JOB_SCRIPT = SCRIPT.parent / "run_stage7c_q1_training.py"


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
        default=PILOT_ROOT / "configs" / "stage7c_q1_protocol.yaml",
    )
    p.add_argument("--seeds", default="64001-64020")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--checkpoint-write-slots", type=int, default=2)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=400_000)
    args = p.parse_args(argv)

    # Refuse in-repo checkpoint root
    try:
        Path(args.checkpoint_root).resolve().relative_to(REPO_ROOT.resolve())
        print("ABORT: --checkpoint-root must be outside the git repository", flush=True)
        return 2
    except ValueError:
        pass

    seeds = _parse_seeds(args.seeds)
    if len(seeds) > 20:
        print(f"ABORT: {len(seeds)} seeds exceeds 20", flush=True)
        return 2

    write_sem = threading.Semaphore(int(args.checkpoint_write_slots))
    lock_path = PILOT_ROOT / "logs" / "launch_stage7c_q1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        os.close(fd)
    except FileExistsError:
        print(f"ABORT: launcher already running ({lock_path})", flush=True)
        return 2

    status: dict[str, dict] = {}
    failed: list[int] = []

    def run_one(seed: int) -> tuple[int, int]:
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
            return seed, int(proc.returncode)

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
            futs = [ex.submit(run_one, s) for s in seeds]
            for fut in as_completed(futs):
                seed, code = fut.result()
                status[str(seed)] = {"returncode": code}
                if code != 0:
                    failed.append(seed)
                    # Do not swallow failures
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass

    summary = {
        "planned": len(seeds),
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
