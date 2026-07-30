#!/usr/bin/env python3
"""Stage 5C-0-H1-R1 — write authoritative 100K protocol amendment locks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[4]


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(sha: str) -> str:
    return f"{_utc().strftime('%Y%m%dT%H%M%SZ')}_{(sha or 'nogit')[:8]}"


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _ensure(run_id: str) -> dict[str, Path]:
    dirs = {
        "reports": EXP_ROOT / "reports" / run_id,
        "logs": EXP_ROOT / "logs" / run_id,
        "artifacts": EXP_ROOT / "artifacts" / run_id,
    }
    for p in dirs.values():
        if p.exists():
            raise RuntimeError(f"refusing overwrite: {p}")
        p.mkdir(parents=True, exist_ok=False)
    return dirs


def _parse_pytest(text: str) -> dict:
    passed = failed = errors = skipped = 0
    for label, attr in (
        ("passed", "passed"),
        ("failed", "failed"),
        ("error", "errors"),
        ("skipped", "skipped"),
    ):
        mm = re.findall(rf"(\d+)\s+{label}", text)
        if mm:
            val = int(mm[-1])
            if attr == "passed":
                passed = val
            elif attr == "failed":
                failed = val
            elif attr == "errors":
                errors = val
            else:
                skipped = val
    status = "PASS" if failed == 0 and errors == 0 and passed > 0 else "FAIL"
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.protocol.h1_r1_100k_protocol import write_h1_r1_artifact_bundle

    git_commit = _git(["rev-parse", "HEAD"]) or "unknown"
    git_dirty = bool(_git(["status", "--porcelain"]))
    run_id = _run_id(git_commit)
    dirs = _ensure(run_id)
    log_path = dirs["logs"] / "runner.log"

    def log(msg: str) -> None:
        line = f"[{_utc().isoformat()}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"run_id={run_id}")
    log("no sustained 100K formal training in H1-R1")

    overall = "PASS"
    try:
        bundle = write_h1_r1_artifact_bundle(
            dirs["artifacts"], git_commit=git_commit, repo_root=REPO_ROOT
        )
        log(f"pbrs_sha={bundle['pbrs_lock_sha256']}")
        log(f"protocol_sha={bundle['training_protocol_sha256']}")
    except Exception as exc:  # noqa: BLE001
        log(f"BLOCKED/FAIL: {exc}")
        overall = "BLOCKED"
        bundle = {"pbrs_lock_sha256": "", "training_protocol_sha256": "", "n_rows": 0}

    test_info = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "status": "SKIPPED"}
    if not args.skip_pytest and overall != "BLOCKED":
        targets = [
            "tests/protocol/test_h1_r1_100k_protocol.py",
            "tests/formal/test_formal_config.py",
            "tests/formal/test_formal_run_matrix_100k.py",
            "tests/formal/test_replay_seed_injection.py",
            "tests/formal/test_formal_schedule.py",
            "tests/formal/test_formal_evaluation_seeds.py",
            "tests/formal/test_formal_checkpoint_resume.py",
            "tests/formal/test_orchestrator_unit.py",
            "tests/formal/test_publish_policy.py",
            "tests/formal/test_notify_payload.py",
            "tests/integration/test_stage5c0_h1_r1_and_6a0.py",
            "tests/protocol/test_final_pbrs_lock.py",
            "tests/protocol/test_final_training_protocol.py",
        ]
        cmd = [sys.executable, "-m", "pytest", "-q", *targets]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env
        )
        (dirs["logs"] / "pytest.log").write_text(
            proc.stdout + "\n" + proc.stderr, encoding="utf-8"
        )
        test_info = _parse_pytest(proc.stdout + "\n" + proc.stderr)
        log(f"pytest={test_info}")
        if test_info["status"] != "PASS":
            overall = "FAIL"

    if git_dirty and overall == "PASS":
        # Artifact write dirties tree; acceptable during runner — final release
        # script requires clean tree before push.
        log("note: git dirty after artifact write (expected before commit)")

    summary = {
        "run_id": run_id,
        "overall": overall,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "tests": test_info,
        "protocol_version": "5C-0-H1-R1-100K",
        "pbrs_lock_sha256": bundle.get("pbrs_lock_sha256"),
        "training_protocol_sha256": bundle.get("training_protocol_sha256"),
        "n_formal_run_slots": bundle.get("n_rows", 0),
        "formal_environment_steps_per_run": 100000,
        "total_planned_environment_steps": 3000000,
        "epsilon_decay_environment_steps": 50000,
        "formal_training_started": False,
        "sustained_training_invoked_in_this_stage": False,
        "exact_command": f"{sys.executable} {SCRIPT_PATH}",
    }
    (dirs["reports"] / "stage5c0_h1_r1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (dirs["reports"] / "stage5c0_h1_r1_report.md").write_text(
        f"# Stage 5C-0-H1-R1\n\nOverall: **{overall}**\n\n"
        f"protocol_sha=`{bundle.get('training_protocol_sha256')}`\n",
        encoding="utf-8",
    )
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "overall": overall}, indent=2), encoding="utf-8"
    )
    log(f"overall={overall}")
    return 0 if overall == "PASS" else (2 if overall == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
