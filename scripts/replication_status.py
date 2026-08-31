#!/usr/bin/env python3
"""Read-only status table for the independent-seed replication curriculum."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replication_common import REPL_RUN_STATE, SEEDS, seed_progress  # noqa: E402


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return False


def main() -> int:
    print(f"{'seed':>8}  {'status':<14}  {'stage':<16}  {'step':>10}  {'pid':>8}  checkpoint")
    print("-" * 100)
    n_done = 0
    for seed in SEEDS:
        prog = seed_progress(seed)
        state_path = REPL_RUN_STATE / f"curriculum_{seed}.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        pid = state.get("pid")
        if prog["curriculum_complete"]:
            status = "completed"
            n_done += 1
        elif state.get("technical_failure"):
            status = "tech_failed"
        elif _pid_alive(pid):
            status = "running"
        elif state.get("started"):
            status = "interrupted"
        elif prog["latest_step"] > 0:
            status = "interrupted"
        else:
            status = "never_started"
        stage = prog["current_stage"] or "C64_R50"
        ckpt = state.get("latest_checkpoint") or ""
        if isinstance(ckpt, str) and len(ckpt) > 60:
            ckpt = "..." + ckpt[-57:]
        print(f"{seed:>8}  {status:<14}  {stage:<16}  {prog['latest_step']:>10}  {str(pid or '-'):>8}  {ckpt}")
    print("-" * 100)
    print(f"complete {n_done}/6   (re-run scripts/launch_replication_curriculum.py to resume interrupted seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
