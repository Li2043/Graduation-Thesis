#!/usr/bin/env python3
"""Engineering helper: repeatedly call resume_curriculum.py until both
910101/910102 have C64_R50 ckpt_step_1200000.pt, or NEEDS_USER_DECISION.md
appears. Does not change scientific protocol."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "curriculum_loop.log"
SEEDS = (910101, 910102)
MAX_ITERS = 40


def c64_done() -> bool:
    for s in SEEDS:
        d = ROOT / "checkpoints" / "curriculum_910101_910102" / str(s) / "C64_R50" / f"seed_{s}_C64_R50"
        if not (d / "ckpt_step_1200000.pt").exists():
            return False
    return True


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    resume = ROOT / "scripts" / "resume_curriculum.py"
    with LOG.open("a", encoding="utf-8") as log:
        for i in range(1, MAX_ITERS + 1):
            if (ROOT / "NEEDS_USER_DECISION.md").exists():
                log.write("STOPPED_FOR_DECISION\n")
                log.flush()
                print("STOPPED_FOR_DECISION", flush=True)
                return 2
            if c64_done():
                log.write("ALL_CURRICULUM_COMPLETE\n")
                log.flush()
                print("ALL_CURRICULUM_COMPLETE", flush=True)
                return 0
            msg = f"===== CURRICULUM LOOP ITER {i} =====\n"
            log.write(msg)
            log.flush()
            print(msg, end="", flush=True)
            proc = subprocess.run(
                [str(py), str(resume)],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.write(f"[loop] resume_curriculum exit={proc.returncode}\n")
            log.flush()
            print(f"[loop] resume_curriculum exit={proc.returncode}", flush=True)
            if (ROOT / "NEEDS_USER_DECISION.md").exists():
                log.write("STOPPED_FOR_DECISION\n")
                log.flush()
                print("STOPPED_FOR_DECISION", flush=True)
                return 2
            if c64_done():
                log.write("ALL_CURRICULUM_COMPLETE\n")
                log.flush()
                print("ALL_CURRICULUM_COMPLETE", flush=True)
                return 0
            time.sleep(2)
    log.write("MAX_ITERS_REACHED\n")
    print("MAX_ITERS_REACHED", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
