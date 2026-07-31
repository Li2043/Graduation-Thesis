#!/usr/bin/env python3
"""Parallel launcher for Stage 7B-A1 (condition × seed jobs)."""

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
JOB_SCRIPT = SCRIPT.parent / "run_stage7b_a1_training.py"


def _parse_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in _parse_list(spec)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, default=PILOT_ROOT / "configs" / "stage7b_a1_protocol.yaml")
    p.add_argument("--conditions", default="vanilla_dqn,double_dqn")
    p.add_argument("--seeds", default="63001-63020")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--checkpoint-write-slots", type=int, default=2)
    p.add_argument("--output-root", type=Path, default=PILOT_ROOT / "output")
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=300_000)
    args = p.parse_args(argv)

    conditions = _parse_list(args.conditions)
    seeds = _parse_seeds(args.seeds)
    jobs = [(c, s) for c in conditions for s in seeds]
    if len(jobs) > 40:
        print(f"ABORT: {len(jobs)} jobs exceeds 40", flush=True)
        return 2

    write_sem = threading.Semaphore(int(args.checkpoint_write_slots))
    # Expose slots to child via env (children currently save internally; semaphore
    # serialises job starts around heavy checkpoint phases only loosely).
    os.environ["STAGE7B_CKPT_WRITE_SLOTS"] = str(args.checkpoint_write_slots)

    lock_path = PILOT_ROOT / "logs" / "launch_stage7b_a1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
    except FileExistsError:
        print(f"ABORT: launcher already running ({lock_path})", flush=True)
        return 2

    status: dict[str, dict] = {}

    def run_one(cond: str, seed: int) -> tuple[str, int]:
        job_id = f"{cond}__{seed}"
        # Isolated writable dirs per job
        out = Path(args.output_root) / "runs" / cond / f"seed_{seed}"
        ckpt = Path(args.checkpoint_root) / cond / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        ckpt.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        env["OMP_NUM_THREADS"] = str(args.threads_per_worker)
        cmd = [
            sys.executable,
            str(JOB_SCRIPT),
            "--protocol",
            str(args.protocol),
            "--condition",
            cond,
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
        status[job_id] = {"status": "running", "condition": cond, "seed": seed}
        # Acquire write slot around process lifetime start; release when done
        write_sem.acquire()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
        finally:
            write_sem.release()
        log = PILOT_ROOT / "logs" / f"{job_id}.log"
        log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        status[job_id] = {
            "status": "complete" if proc.returncode == 0 else "failed",
            "condition": cond,
            "seed": seed,
            "returncode": proc.returncode,
            "output_dir": str(out.as_posix()),
            "checkpoint_dir": str(ckpt.as_posix()),
        }
        return job_id, proc.returncode

    t0 = time.time()
    results = {}
    try:
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
            futs = {ex.submit(run_one, c, s): (c, s) for c, s in jobs}
            for fut in as_completed(futs):
                job_id, rc = fut.result()
                results[job_id] = rc
                print(f"{job_id} rc={rc}", flush=True)
    finally:
        try:
            os.close(fd)
        except Exception:
            pass
        lock_path.unlink(missing_ok=True)

    table_path = PILOT_ROOT / "logs" / "job_status_table.json"
    table_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    failed = [k for k, rc in results.items() if rc != 0]
    summary = {
        "n_jobs": len(jobs),
        "failed": failed,
        "elapsed_sec": time.time() - t0,
        "max_workers": args.max_workers,
        "checkpoint_write_slots": args.checkpoint_write_slots,
        "vectorized": False,
    }
    (PILOT_ROOT / "logs" / "launch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
