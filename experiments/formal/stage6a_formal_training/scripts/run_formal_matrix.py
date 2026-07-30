#!/usr/bin/env python3
"""Multi-job formal orchestrator (Stage 6A-0).

Process-level parallelism across independent formal jobs. Default workers=12.
Does not replace seeds. Does not start retained formal training unless invoked
with the authoritative matrix and without --dry-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
JOB_RUNNER = SCRIPT_PATH.parent / "run_formal_job.py"


def _set_thread_env() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    """Run one formal job in a spawned process."""
    _set_thread_env()
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    cmd = [
        sys.executable,
        str(task["job_runner"]),
        "--condition",
        str(task["condition"]),
        "--master-seed",
        str(task["master_seed"]),
        "--protocol-lock",
        str(task["protocol_lock"]),
        "--output-root",
        str(task["output_root"]),
        "--device",
        "cpu",
    ]
    if task.get("resume"):
        cmd.append("--resume")
    if task.get("skip_runner_release_check"):
        cmd.append("--skip-runner-release-check")
    if task.get("max_steps") is not None:
        cmd.extend(["--max-steps", str(task["max_steps"])])

    job_id = task["formal_job_id"]
    log_dir = Path(task["output_root"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{job_id}.stdout.log"
    stderr_path = log_dir / f"{job_id}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.run(cmd, cwd=str(task["repo_root"]), stdout=out, stderr=err)
    status_file = Path(task["output_root"]) / "jobs" / job_id / "status.json"
    status = "FAILED_WITH_REASON"
    reason = f"exit_code={proc.returncode}"
    if status_file.is_file():
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
            status = str(payload.get("status", status))
            reason = str(payload.get("reason", reason))
        except Exception as exc:  # noqa: BLE001
            reason = f"status_parse_error:{exc}"
    elif proc.returncode == 0:
        status = "COMPLETE"
        reason = ""
    return {
        "formal_job_id": job_id,
        "condition": task["condition"],
        "master_seed": task["master_seed"],
        "status": status,
        "reason": reason,
        "returncode": proc.returncode,
    }


def _load_matrix(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-matrix", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish-root", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan jobs only; do not spawn training workers",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Infrastructure test override; never use for retained formal runs",
    )
    parser.add_argument("--skip-runner-release-check", action="store_true")
    args = parser.parse_args(argv)

    if int(args.threads_per_worker) != 1:
        raise SystemExit("threads-per-worker must be 1")
    if int(args.workers) < 1:
        raise SystemExit("workers must be >= 1")

    _set_thread_env()
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from thesis.formal.status_registry import (
        FormalStatusRegistry,
        TERMINAL_COMPLETE,
        TERMINAL_FAILED,
        TERMINAL_INTERRUPTED,
    )

    rows = _load_matrix(Path(args.run_matrix))
    if len(rows) != 30 and args.max_steps is None:
        # Allow smaller matrices only when explicitly testing
        if not args.dry_run and args.max_steps is None:
            raise SystemExit(f"formal matrix must have 30 rows, got {len(rows)}")

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    registry = FormalStatusRegistry(out_root / "status_registry.json")
    expected_ids = [r["formal_job_id"] for r in rows]

    plan = {
        "n_jobs": len(rows),
        "workers": int(args.workers),
        "threads_per_worker": 1,
        "dry_run": bool(args.dry_run),
        "replace_failed_seeds": False,
        "vectorized_training": False,
        "num_parallel_training_envs_per_run": 1,
        "formal_training_started": False if args.dry_run else True,
        "jobs": expected_ids,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "orchestrator_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        print(json.dumps({"dry_run": True, "n_jobs": len(rows)}, indent=2))
        return 0

    tasks: list[dict[str, Any]] = []
    for row in rows:
        jid = row["formal_job_id"]
        if registry.should_skip(jid):
            continue
        tasks.append(
            {
                "formal_job_id": jid,
                "condition": row["condition"],
                "master_seed": int(row["master_seed"]),
                "protocol_lock": str(Path(args.protocol_lock).resolve()),
                "output_root": str(out_root),
                "resume": bool(args.resume) or registry.should_resume(jid),
                "job_runner": str(JOB_RUNNER),
                "repo_root": str(REPO_ROOT),
                "max_steps": args.max_steps,
                "skip_runner_release_check": bool(args.skip_runner_release_check),
            }
        )

    # Mark pending tasks running in registry before spawn
    for t in tasks:
        registry.upsert(
            t["formal_job_id"],
            {
                "status": "RUNNING",
                "condition": t["condition"],
                "master_seed": t["master_seed"],
            },
        )

    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    # Survive individual worker failures: collect exceptions as FAILED
    with ctx.Pool(processes=int(args.workers)) as pool:
        async_results = [pool.apply_async(_worker, (t,)) for t in tasks]
        for ar, t in zip(async_results, tasks):
            try:
                res = ar.get()
            except Exception as exc:  # noqa: BLE001
                res = {
                    "formal_job_id": t["formal_job_id"],
                    "condition": t["condition"],
                    "master_seed": t["master_seed"],
                    "status": TERMINAL_FAILED,
                    "reason": f"worker_process_failure:{exc}",
                    "returncode": -1,
                }
            results.append(res)
            registry.upsert(
                res["formal_job_id"],
                {
                    "status": res["status"],
                    "reason": res.get("reason", ""),
                    "condition": res["condition"],
                    "master_seed": res["master_seed"],
                    # Never replace seeds
                    "seed_replaced": False,
                },
            )

    summary = registry.account_all(expected_ids)
    summary["results"] = results
    summary["replace_failed_seeds"] = False
    (out_root / "orchestrator_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    # All slots must eventually have COMPLETE or FAILED_WITH_REASON for experiment
    # completion; interrupted is resumable and yields non-zero here.
    if summary["interrupted"] > 0:
        return 2
    if summary["failed"] > 0:
        return 1
    if summary["complete"] == summary["expected"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
