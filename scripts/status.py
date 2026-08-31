#!/usr/bin/env python3
"""Report the status of every formal run (and the 910101/910102
curriculum build) by inspecting run manifests + checkpoints + process
liveness. Categorizes each as: completed / running / interrupted /
never_started / technically_failed.

Poor scientific performance is NEVER technically_failed -- only a
crashed process, missing/corrupt checkpoint, or exception counts."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CHECKPOINTS_ROOT, RUN_STATE_DIR, find_latest_checkpoint, load_frozen_config, write_json_atomic,
)

HEARTBEAT_STALE_SECONDS = 900  # 15 min with no heartbeat + no completion marker = interrupted


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        # Fallback without psutil: os.kill with signal 0 (Windows-compatible via ctypes would be
        # better, but this best-effort check is fine since heartbeat staleness is the real signal).
        import os
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
        except AttributeError:
            return True  # unknown platform behaviour -- don't claim it's dead


def classify_run(manifest: dict) -> str:
    if manifest.get("completed"):
        return "completed"
    if manifest.get("technical_failure"):
        return "technically_failed"
    pid = manifest.get("pid")
    last_heartbeat = manifest.get("last_update_unix", 0)
    stale = (time.time() - last_heartbeat) > HEARTBEAT_STALE_SECONDS
    if _pid_alive(pid) and not stale:
        return "running"
    if manifest.get("started"):
        return "interrupted"
    return "never_started"


def main() -> int:
    cfg = load_frozen_config()
    rows = []

    if RUN_STATE_DIR.exists():
        for f in sorted(RUN_STATE_DIR.glob("*.json")):
            manifest = json.loads(f.read_text(encoding="utf-8"))
            status = classify_run(manifest)
            rows.append({"run_id": f.stem, "status": status,
                         "seed": manifest.get("seed"), "condition": manifest.get("condition"),
                         "current_step": manifest.get("current_step"),
                         "latest_checkpoint": manifest.get("latest_checkpoint")})

    # curriculum build state for 910101/910102, even if resume_curriculum.py hasn't written a run manifest
    for seed in (910101, 910102):
        seed_root = CHECKPOINTS_ROOT / "curriculum_910101_910102" / str(seed)
        stage_progress = {}
        for stage in ("M6_R50_audited", "C4_R50", "C4_R50ext", "C16_R50", "C64_R50"):
            d = seed_root / stage / f"seed_{seed}_{stage}"
            latest = find_latest_checkpoint(d)
            if latest:
                stage_progress[stage] = latest[0]
        rows.append({"run_id": f"curriculum_{seed}", "status": "see stage_progress",
                     "seed": seed, "condition": "task_only_curriculum", "stage_progress": stage_progress})

    print(json.dumps(rows, indent=2))
    write_json_atomic(RUN_STATE_DIR.parent / "status_report.json",
                       {"generated_unix": time.time(), "formal_fairness_started": cfg.get("formal_fairness_started"),
                        "rows": rows})

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nSummary:", counts)
    print("\nOnly 'interrupted' runs may be auto-resumed (see resume_interrupted.py). "
          "'technically_failed' requires a human look. Poor scientific performance in a "
          "'completed' run is a RESULT, not a failure -- never auto-rerun it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
