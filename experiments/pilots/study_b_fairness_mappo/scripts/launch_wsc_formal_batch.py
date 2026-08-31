#!/usr/bin/env python3
"""Orchestration-only launcher for the 48 formal WSC runs. Contains NO
scientific formulas, reward logic, or observation logic -- it only reads
the frozen 48-row matrix CSV, constructs exact command lines for the
frozen train_curriculum_stage_highwayenv_wsc.py, enforces unique output
directories, enforces the chosen concurrency, redirects stdout/stderr to
per-run logs, and records PIDs/start times/exit codes to a CSV registry.

Mirrors F:\\正式训练\\scripts\\launch_formal.py's own orchestration
pattern (process-level parallelism only, OMP_NUM_THREADS=1 per worker,
never touches batch size/update frequency to use more cores).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# DENSE-REWARD COPY (2026-08-26): the original absolute paths here (PY_EXE,
# SCENARIO_BANK pointing at the stale "F:\正式训练" and MATRIX_CSV/LOGS_DIR/
# REGISTRY_CSV/COMMANDS_TXT pointing at the OLD formal bundle
# "F:\正式训练_seed_replication_v1") were replaced with paths relative to
# this copy's own BUNDLE_ROOT. This is a write-path script (registry CSVs,
# per-run logs) -- left unmodified it would have written into (or, being
# stale, failed to find) the OLD formal environment, which must never be
# touched. This script is still not part of the active dense-reward smoke
# pipeline (11_START_WELFARE.bat calls launch_replication_welfare.py, not
# this file) and is not executed as part of this environment setup.
BUNDLE_ROOT = Path(__file__).resolve().parents[5]
PY_EXE = str(BUNDLE_ROOT / ".venv" / "Scripts" / "python.exe")
WSC_SCRIPT = Path(__file__).resolve().parent / "train_curriculum_stage_highwayenv_wsc.py"
SCENARIO_BANK = BUNDLE_ROOT / "scenario_banks" / "Q.json"
MATRIX_CSV = BUNDLE_ROOT / "validation_artifacts" / "wsc" / "wsc_future_4x2_matrix_dryrun.csv"
LOGS_DIR = BUNDLE_ROOT / "checkpoints" / "wsc_formal_runs" / "_logs"
REGISTRY_CSV = BUNDLE_ROOT / "validation_artifacts" / "wsc_formal_launch" / "wsc_formal_run_registry.csv"
COMMANDS_TXT = BUNDLE_ROOT / "validation_artifacts" / "wsc_formal_launch" / "wsc_formal_launch_commands.txt"

REGISTRY_FIELDS = [
    "run_id", "seed", "condition", "status", "pid", "start_time", "last_check_time",
    "source_checkpoint", "output_dir", "stdout_log", "stderr_log",
    "latest_checkpoint_step", "latest_checkpoint_time", "planned_end_step", "exit_code",
    "restart_count", "notes",
]


def scenario_ids() -> list[str]:
    data = json.loads(SCENARIO_BANK.read_text(encoding="utf-8"))
    return [s["scenario_id"] for s in data]


def load_matrix() -> list[dict]:
    with open(MATRIX_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_command(row: dict) -> list[str]:
    out_root = Path(row["output_dir"])
    ckpt_root = Path(row["output_dir"])  # checkpoint_dir is out_root/seed_{seed}_{stage}; script builds that itself
    return [
        PY_EXE, str(WSC_SCRIPT),
        "--scenario-bank", str(SCENARIO_BANK),
        "--scenario-ids", *scenario_ids(),
        "--stage-name", row["stage_name"],
        "--master-seed", row["seed"],
        "--output-root", str(out_root),
        "--checkpoint-root", str(ckpt_root),
        "--start-step", row["start_step"],
        "--max-additional-steps", row["additional_steps"],
        "--episode-max-steps", "200",
        "--checkpoint-every", row["checkpoint_every"],
        "--device", "cpu",
        "--replay-warmup", "512",
        "--eps-decay-steps-absolute", "640000",
        "--lr-decay-steps-absolute", "800000",
        "--welfare-lambda", row["welfare_lambda"],
        "--condition", row["condition"] if row["condition"] != "baseline" else "mean",
        "--resume-from", row["source_checkpoint"],
        "--action-representation", "meta_speed",
        "--local-sensing-range-m", "50.0",
    ]


def write_command_archive(rows: list[dict]) -> None:
    with open(COMMANDS_TXT, "w", encoding="utf-8") as f:
        for row in rows:
            cmd = build_command(row)
            stdout_log = LOGS_DIR / f"{row['run_id']}.stdout.log"
            stderr_log = LOGS_DIR / f"{row['run_id']}.stderr.log"
            f.write(f"run_id={row['run_id']}\n")
            f.write(f"seed={row['seed']}\n")
            f.write(f"condition={row['condition']}\n")
            f.write("command=" + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
            f.write(f"output_directory={row['output_dir']}\n")
            f.write(f"stdout_log={stdout_log}\n")
            f.write(f"stderr_log={stderr_log}\n\n")
    print(f"[launch_wsc_formal_batch] wrote {len(rows)} command records -> {COMMANDS_TXT}")


def init_registry(rows: list[dict]) -> None:
    REGISTRY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({
                "run_id": row["run_id"], "seed": row["seed"], "condition": row["condition"],
                "status": "PENDING", "pid": "", "start_time": "", "last_check_time": "",
                "source_checkpoint": row["source_checkpoint"], "output_dir": row["output_dir"],
                "stdout_log": str(LOGS_DIR / f"{row['run_id']}.stdout.log"),
                "stderr_log": str(LOGS_DIR / f"{row['run_id']}.stderr.log"),
                "latest_checkpoint_step": "", "latest_checkpoint_time": "",
                "planned_end_step": row["planned_end_step"], "exit_code": "", "restart_count": 0, "notes": "",
            })
    print(f"[launch_wsc_formal_batch] initialized registry ({len(rows)} rows) -> {REGISTRY_CSV}")


def update_registry_row(run_id: str, **updates) -> None:
    rows = []
    with open(REGISTRY_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["run_id"] == run_id:
            r.update({k: str(v) for k, v in updates.items()})
    with open(REGISTRY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_matrix()
    assert len(rows) == 48, f"expected 48 rows, got {len(rows)}"
    run_ids = [r["run_id"] for r in rows]
    assert len(run_ids) == len(set(run_ids)), "duplicate run_id in matrix"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    write_command_archive(rows)
    init_registry(rows)

    if args.dry_run:
        print(f"[launch_wsc_formal_batch] DRY RUN -- would launch {len(rows)} runs at concurrency={args.max_concurrent}")
        for r in rows:
            print(f"  would launch: {r['run_id']} (seed={r['seed']} condition={r['condition']})")
        return 0

    concurrency = args.max_concurrent
    print(f"[launch_wsc_formal_batch] launching {len(rows)} runs at concurrency={concurrency}")

    queue = list(rows)
    running: list[tuple[str, subprocess.Popen, object, object]] = []

    def _launch_one(row: dict):
        run_id = row["run_id"]
        out_dir = Path(row["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = LOGS_DIR / f"{run_id}.stdout.log"
        stderr_log = LOGS_DIR / f"{run_id}.stderr.log"
        cmd = build_command(row)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        out_f = open(stdout_log, "w", encoding="utf-8")
        err_f = open(stderr_log, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env,
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        update_registry_row(run_id, status="RUNNING", pid=proc.pid, start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                             last_check_time=time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[launch_wsc_formal_batch] started {run_id} pid={proc.pid}")
        return proc, out_f, err_f

    while queue or running:
        while queue and len(running) < concurrency:
            row = queue.pop(0)
            proc, out_f, err_f = _launch_one(row)
            running.append((row["run_id"], proc, out_f, err_f))
        time.sleep(5)
        still = []
        for run_id, proc, out_f, err_f in running:
            ret = proc.poll()
            if ret is None:
                still.append((run_id, proc, out_f, err_f))
            else:
                out_f.close(); err_f.close()
                status = "COMPLETED" if ret == 0 else "FAILED_TECHNICAL"
                update_registry_row(run_id, status=status, exit_code=ret,
                                     last_check_time=time.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"[launch_wsc_formal_batch] {run_id} finished rc={ret} ({status})")
        running = still

    print("[launch_wsc_formal_batch] all 48 runs have exited (see registry for status).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
