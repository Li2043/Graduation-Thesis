#!/usr/bin/env python3
"""Engineering driver: after any in-flight resume_curriculum finishes,
loop curriculum to C64_R50 for 910101/910102, then freeze + launch formal.
Does not change scientific hyperparameters or protocol rules."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "pipeline_continue.log"
SEEDS = (910101, 910102)
MAX_CURRICULUM_ITERS = 40


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def py() -> str:
    return str(ROOT / ".venv" / "Scripts" / "python.exe")


def resume_or_train_running() -> bool:
    try:
        import psutil
    except ImportError:
        return False
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if "resume_curriculum.py" in cmd or "train_curriculum_stage_highwayenv.py" in cmd:
            # ignore this pipeline process itself
            if "_pipeline_continue.py" in cmd:
                continue
            return True
    return False


def c64_done() -> bool:
    for s in SEEDS:
        d = (ROOT / "checkpoints" / "curriculum_910101_910102" / str(s)
             / "C64_R50" / f"seed_{s}_C64_R50" / "ckpt_step_1200000.pt")
        if not d.exists():
            return False
    return True


def run_checked(args: list[str], label: str) -> int:
    log(f"RUN {label}: {' '.join(args)}")
    proc = subprocess.run(args, cwd=str(ROOT))
    log(f"EXIT {label}: {proc.returncode}")
    return proc.returncode


def wait_for_inflight() -> None:
    log("Waiting for any in-flight curriculum train/resume to finish...")
    while resume_or_train_running():
        time.sleep(30)
    log("No in-flight curriculum processes; continuing.")


def loop_curriculum() -> int:
    for i in range(1, MAX_CURRICULUM_ITERS + 1):
        if (ROOT / "NEEDS_USER_DECISION.md").exists():
            log("STOPPED_FOR_DECISION")
            return 2
        if c64_done():
            log("ALL_CURRICULUM_COMPLETE")
            return 0
        log(f"===== CURRICULUM LOOP ITER {i} =====")
        rc = run_checked([py(), str(ROOT / "scripts" / "resume_curriculum.py")], f"resume_iter_{i}")
        if (ROOT / "NEEDS_USER_DECISION.md").exists():
            log("STOPPED_FOR_DECISION")
            return 2
        if c64_done():
            log("ALL_CURRICULUM_COMPLETE")
            return 0
        if rc != 0:
            log(f"resume_curriculum returned {rc}; will retry after short pause")
            time.sleep(5)
    log("MAX_CURRICULUM_ITERS_REACHED")
    return 1


def main() -> int:
    wait_for_inflight()
    rc = loop_curriculum()
    if rc != 0:
        return rc

    log("Running 03 freeze formal...")
    rc = run_checked([py(), str(ROOT / "scripts" / "freeze_formal_manifest.py")], "freeze_formal")
    if rc != 0 or (ROOT / "NEEDS_USER_DECISION.md").exists():
        log("Freeze stopped or needs decision")
        return 2 if (ROOT / "NEEDS_USER_DECISION.md").exists() else rc

    log("Running 04 launch formal (resource-aware)...")
    rc = run_checked([py(), str(ROOT / "scripts" / "launch_formal.py")], "launch_formal")
    log(f"Pipeline launch phase finished rc={rc}")
    log("Formal runs are long-lived; use 05_STATUS.bat / scripts/monitor_formal.py to watch.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
