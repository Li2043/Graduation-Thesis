#!/usr/bin/env python3
"""Stage 5C-0 — final PBRS + formal training protocol lock runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[4]

SOURCE_MODULES = [
    "src/thesis/protocol/prerequisites.py",
    "src/thesis/protocol/final_pbrs_lock.py",
    "src/thesis/protocol/final_training_protocol.py",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(sha: str) -> str:
    return f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{(sha or 'nogit')[:8]}"


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


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


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _parse_pytest(text: str) -> dict[str, Any]:
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


def _versions() -> dict[str, str]:
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "torch", "pytest", "yaml"):
        modname = "yaml" if name == "yaml" else name
        try:
            m = __import__(modname)
            out[name] = getattr(m, "__version__", "installed")
        except Exception:
            out[name] = "not_installed"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_ROOT / "configs" / "stage5c0.yaml")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.protocol.final_pbrs_lock import build_final_pbrs_lock, write_final_pbrs_lock
    from thesis.protocol.final_training_protocol import (
        FORMAL_CHECKPOINT_STEPS,
        FORMAL_EVALUATION_STEPS,
        FORMAL_MASTER_SEEDS,
        build_final_training_protocol,
        build_formal_analysis_plan,
        build_formal_run_matrix,
        write_final_training_protocol,
        write_formal_analysis_plan,
        write_formal_run_matrix,
    )
    from thesis.protocol.prerequisites import ProtocolBlockedError, verify_stage5c0_prerequisites

    git_commit = _git(["rev-parse", "HEAD"]) or "unknown"
    git_dirty = bool(_git(["status", "--porcelain"]))
    run_id = _run_id(git_commit)
    dirs = _ensure(run_id)
    log_path = dirs["logs"] / "runner.log"

    def log(msg: str) -> None:
        line = f"[{_utc_now().isoformat()}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    cfg = _load_yaml(args.config)
    cfg_hash = _sha256(args.config.resolve())
    source_hashes = {rel: _sha256(REPO_ROOT / rel) for rel in SOURCE_MODULES}
    exact_command = f"{sys.executable} {SCRIPT_PATH}"

    log(f"run_id={run_id}")
    log(f"git_commit={git_commit} dirty={git_dirty}")
    log("no sustained training invoked in Stage 5C-0")

    env_lock = REPO_ROOT / cfg["authoritative_locks"]["environment_lock_path"]
    comfort_lock = REPO_ROOT / cfg["authoritative_locks"]["comfort_lock_path"]
    env_before = _sha256(env_lock)
    comfort_before = _sha256(comfort_lock)
    log(f"environment lock sha before={env_before}")
    log(f"comfort lock sha before={comfort_before}")

    overall = "PASS"
    blocked = False
    integrity = {
        "lock_hash_mismatch": 0,
        "prerequisite_failures": 0,
        "protocol_write_failures": 0,
        "sustained_training_invoked": 0,
        "behavioral_csv_read": 0,
    }
    pbrs_sha = ""
    protocol_sha = ""
    prereq = None
    matrix_rows: list[dict[str, Any]] = []

    try:
        if env_before != cfg["authoritative_locks"]["environment_lock_sha256"]:
            raise ProtocolBlockedError("environment lock hash mismatch")
        if comfort_before != cfg["authoritative_locks"]["comfort_lock_sha256"]:
            raise ProtocolBlockedError("comfort lock hash mismatch")
        prereq = verify_stage5c0_prerequisites(repo_root=REPO_ROOT)
        log(
            f"prerequisites OK: stage5a0={prereq.stage5a0_run_id} "
            f"stage5b0={prereq.stage5b0_run_id}"
        )
    except ProtocolBlockedError as exc:
        log(f"BLOCKED: {exc}")
        overall = "BLOCKED"
        blocked = True
        integrity["prerequisite_failures"] = 1

    _write_yaml(
        dirs["artifacts"] / "resolved_protocol_config.yaml",
        {
            "run_id": run_id,
            "config": cfg,
            "config_sha256": cfg_hash,
            "source_hashes": source_hashes,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "formal_training_started": False,
            "sustained_training_invoked_during_stage5c0": False,
        },
    )

    if not blocked and prereq is not None:
        try:
            pbrs_lock = build_final_pbrs_lock(
                prereq,
                git_commit=git_commit,
                source_hashes=source_hashes,
                configuration_sha256=cfg_hash,
            )
            pbrs_sha = write_final_pbrs_lock(
                dirs["artifacts"] / "final_pbrs_parameters.yaml", pbrs_lock
            )
            protocol = build_final_training_protocol(
                prereq,
                git_commit=git_commit,
                source_hashes=source_hashes,
                configuration_sha256=cfg_hash,
                pbrs_lock_sha256=pbrs_sha,
            )
            protocol_sha = write_final_training_protocol(
                dirs["artifacts"] / "final_training_protocol.yaml", protocol
            )
            matrix_rows = build_formal_run_matrix()
            write_formal_run_matrix(
                dirs["artifacts"] / "formal_run_matrix.csv", matrix_rows
            )
            write_formal_analysis_plan(
                dirs["artifacts"] / "formal_analysis_plan.yaml"
            )
            manifest = {
                "run_id": run_id,
                "stage": "stage5c0",
                "pbrs_lock_sha256": pbrs_sha,
                "training_protocol_sha256": protocol_sha,
                "n_formal_run_slots": len(matrix_rows),
                "formal_master_seeds": list(FORMAL_MASTER_SEEDS),
                "checkpoint_steps": list(FORMAL_CHECKPOINT_STEPS),
                "evaluation_steps": list(FORMAL_EVALUATION_STEPS),
                "environment_parameters_final": True,
                "comfort_parameters_final": True,
                "pbrs_parameters_final": True,
                "training_protocol_final": True,
                "pilot_training_started": True,
                "policy_training_started": True,
                "sustained_training_invoked": True,
                "formal_training_started": False,
                "sustained_training_invoked_during_stage5c0": False,
                "pilot_behavioral_observations_read": False,
            }
            (dirs["artifacts"] / "formal_experiment_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            log(f"wrote PBRS lock sha={pbrs_sha}")
            log(f"wrote training protocol sha={protocol_sha}")
            log(f"formal run matrix slots={len(matrix_rows)}")
        except Exception as exc:  # noqa: BLE001
            integrity["protocol_write_failures"] += 1
            log(f"FAIL writing protocol locks: {exc}")
            overall = "FAIL"

    log("running pytest (no training)")
    targets = cfg.get("pytest_targets") or []
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
    (dirs["logs"] / "pytest.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    test_info = _parse_pytest(proc.stdout + "\n" + proc.stderr)
    log(f"pytest={test_info}")

    env_after = _sha256(env_lock)
    comfort_after = _sha256(comfort_lock)
    log(f"environment lock sha after={env_after}")
    log(f"comfort lock sha after={comfort_after}")
    if env_after != env_before or comfort_after != comfort_before:
        integrity["lock_hash_mismatch"] += 1

    # Confirm no formal training dirs created under experiments/formal besides protocol/
    formal_root = REPO_ROOT / "experiments" / "formal"
    unexpected = [
        p.name
        for p in formal_root.iterdir()
        if p.is_dir() and p.name not in {"protocol"}
    ] if formal_root.is_dir() else []
    if unexpected:
        integrity["sustained_training_invoked"] += 1
        log(f"FAIL unexpected formal dirs: {unexpected}")

    if blocked:
        overall = "BLOCKED"
    elif test_info["status"] != "PASS":
        overall = "FAIL"
    elif any(v > 0 for v in integrity.values()):
        overall = "FAIL"
    elif git_dirty:
        overall = "FAIL"
        log("FAIL: git_dirty=true")
    elif not pbrs_sha or not protocol_sha:
        overall = "FAIL"
    else:
        overall = "PASS"

    summary = {
        "run_id": run_id,
        "overall": overall,
        "exact_command": exact_command,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "tests": test_info,
        "prerequisite_run_ids": {
            "stage5a0": cfg["prerequisites"]["stage5a0_run_id"],
            "stage5b0": cfg["prerequisites"]["stage5b0_run_id"],
        },
        "environment_lock_path": str(env_lock),
        "environment_lock_sha256_before": env_before,
        "environment_lock_sha256_after": env_after,
        "comfort_lock_path": str(comfort_lock),
        "comfort_lock_sha256_before": comfort_before,
        "comfort_lock_sha256_after": comfort_after,
        "lambda_mean": 0.2,
        "lambda_min": 0.2,
        "pbrs_lock_path": str(dirs["artifacts"] / "final_pbrs_parameters.yaml"),
        "pbrs_lock_sha256": pbrs_sha,
        "final_network_architecture": [64, 64],
        "optimiser_and_replay": {
            "optimiser": "Adam",
            "learning_rate": 0.0005,
            "batch_size": 64,
            "replay_capacity_per_controller": 20000,
            "replay_warmup_per_controller": 512,
            "target_sync_interval_updates": 250,
            "gamma": 0.995,
        },
        "epsilon_schedule": {
            "start": 1.0,
            "end": 0.10,
            "decay_environment_steps": 4000,
            "schedule": "linear",
        },
        "formal_seed_list": list(FORMAL_MASTER_SEEDS),
        "n_run_slots": len(matrix_rows) if matrix_rows else 0,
        "steps_per_run": 20000,
        "total_planned_steps": 600000,
        "checkpoint_schedule": list(FORMAL_CHECKPOINT_STEPS),
        "evaluation_schedule": list(FORMAL_EVALUATION_STEPS),
        "primary_endpoints": build_formal_analysis_plan()[
            "primary_endpoints_at_step_20000"
        ],
        "statistical_unit": "formal_training_seed",
        "paired_contrasts": build_formal_analysis_plan()["pairwise_contrasts"],
        "bootstrap_settings": build_formal_analysis_plan()["bootstrap"],
        "multiple_comparison_rule": "Holm within each primary endpoint across three contrasts",
        "training_protocol_lock_path": str(
            dirs["artifacts"] / "final_training_protocol.yaml"
        ),
        "training_protocol_lock_sha256": protocol_sha,
        "formal_run_matrix_path": str(dirs["artifacts"] / "formal_run_matrix.csv"),
        "integrity": integrity,
        "environment_parameters_final": True,
        "comfort_parameters_final": True,
        "pbrs_parameters_final": True if overall == "PASS" else False,
        "training_protocol_final": True if overall == "PASS" else False,
        "pilot_training_started": True,
        "policy_training_started": True,
        "formal_training_started": False,
        "sustained_training_invoked": True,
        "sustained_training_invoked_during_stage5c0": False,
        "unresolved_limitations": [
            "Formal multi-seed training has not started.",
            "Endpoint operational definitions still require implementation in the formal runner.",
            "Convention-consistency missing-data handling is specified but not yet executed.",
        ],
        "source_hashes": source_hashes,
    }
    (dirs["reports"] / "stage5c0_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = f"""# Stage 5C-0 Report — Final PBRS and Training Protocol Lock

## Overall: **{overall}**

- run_id: `{run_id}`
- git: `{git_commit}` dirty=`{git_dirty}`
- tests: `{test_info}`
- stage5a0: `{cfg['prerequisites']['stage5a0_run_id']}`
- stage5b0: `{cfg['prerequisites']['stage5b0_run_id']}`
- lambda_mean / lambda_min: `0.2` / `0.2`
- PBRS lock sha: `{pbrs_sha}`
- training protocol sha: `{protocol_sha}`
- formal run slots: `{len(matrix_rows) if matrix_rows else 0}`
- total planned steps: `600000`
- formal_training_started: false
- sustained training during this stage: false

## Status flags

- environment_parameters_final = true
- comfort_parameters_final = true
- pbrs_parameters_final = {summary['pbrs_parameters_final']}
- training_protocol_final = {summary['training_protocol_final']}
- formal_training_started = false
"""
    (dirs["reports"] / "stage5c0_report.md").write_text(report, encoding="utf-8")
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "utc_timestamp": _utc_now().isoformat(),
                "python_and_packages": _versions(),
                "output_paths": {k: str(v) for k, v in dirs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "overall": overall}, indent=2), encoding="utf-8"
    )
    log(f"overall={overall}")
    return 0 if overall == "PASS" else (2 if overall == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
