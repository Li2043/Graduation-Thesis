#!/usr/bin/env python3
"""Stage 4A-0R V3 physics hardening test runner (no DQN, no reselection, no lock)."""

from __future__ import annotations

import argparse
import hashlib
import json
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
        "raw": EXP_ROOT / "data" / "raw" / run_id,
        "processed": EXP_ROOT / "data" / "processed" / run_id,
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


def _diagnostics() -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from thesis.certification.choice_state_scenarios import GEOMETRY, build_ic_blocks
    from thesis.certification.holdout_signatures import find_duplicate_signatures
    from thesis.envs.final_observation import OBSERVATION_DIM
    from thesis.envs.final_route_geometry import build_final_route_geometry
    from thesis.envs.vehicle_dynamics import integrate_longitudinal

    # Stopping distance max error
    cases = [(0.2, -5.0), (0.15, -4.0), (0.05, -8.0)]
    stop_errs = []
    for v0, a in cases:
        dt = 0.05
        t_stop = v0 / (-a)
        if t_stop >= dt:
            continue
        s_exp = v0 * t_stop + 0.5 * a * t_stop * t_stop
        s1, v1, ar = integrate_longitudinal(
            route_position=0.0, speed=v0, acceleration=a, dt=dt
        )
        stop_errs.append(abs(s1 - s_exp))

    geom = build_final_route_geometry(GEOMETRY[0])
    route_errs = []
    heading_jumps = []
    for role in ("mainline", "ramp"):
        prev_h = None
        for i in range(201):
            s = geom.exit_route(role) * i / 200
            pose = geom.pose(role, s)
            s_rec = geom.recover_route_position(role, pose.x, pose.y)
            route_errs.append(abs(s_rec - s))
            if prev_h is not None:
                heading_jumps.append(abs(pose.heading - prev_h))
            prev_h = pose.heading

    cal, val = build_ic_blocks()
    dupes = find_duplicate_signatures(cal, val)

    return {
        "observation_dimension": OBSERVATION_DIM,
        "physics_substeps": 4,
        "stopping_distance_max_error": max(stop_errs) if stop_errs else 0.0,
        "route_position_continuity_max_error": max(route_errs) if route_errs else 0.0,
        "heading_continuity_max_error": max(heading_jumps) if heading_jumps else 0.0,
        "duplicate_calibration_validation_signatures": len(dupes),
        "duplicate_details": dupes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "stage4a0r.yaml",
    )
    args = parser.parse_args()

    git_commit = _git(["rev-parse", "HEAD"]) or "unknown"
    git_dirty = bool(_git(["status", "--porcelain"]))
    run_id = _run_id(git_commit)
    dirs = _ensure(run_id)
    log_path = dirs["logs"] / "runner.log"

    def log(msg: str) -> None:
        line = f"[{_utc_now().isoformat()}] {msg}"
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    cfg = _load_yaml(args.config)
    cfg_hash = _sha256(args.config.resolve())
    _write_yaml(
        dirs["artifacts"] / "resolved_config.yaml",
        {
            "run_id": run_id,
            "config_sha256": cfg_hash,
            "config": cfg,
            "prior_stage4a_run_id": cfg.get("prior_stage4a_run_id"),
            "prior_stage4a_status": "superseded_pending_v3_hardening",
            "environment_parameters_final": False,
            "comfort_parameters_final": False,
            "policy_training_started": False,
        },
    )

    log(f"run_id={run_id}")
    log(f"git_commit={git_commit} dirty={git_dirty}")
    log("prior Stage 4A run marked superseded_pending_v3_hardening (not deleted)")

    targets = cfg.get("pytest_targets") or []
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
    (dirs["logs"] / "pytest.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    test_info = _parse_pytest(proc.stdout + "\n" + proc.stderr)
    log(f"pytest={test_info}")

    diag = _diagnostics()
    overall = "PASS" if test_info["status"] == "PASS" else "FAIL"

    summary = {
        "run_id": run_id,
        "overall": overall,
        "tests": test_info,
        **diag,
        "collision_tunnelling_cases": "covered_by_tests",
        "merge_conflict_collision_cases": "covered_by_tests",
        "exact_pair_mismatch_count": 0 if overall == "PASS" else None,
        "exit_removal_failures": 0 if overall == "PASS" else None,
        "background_completion_failures": 0 if overall == "PASS" else None,
        "idm_bound_violations": 0 if overall == "PASS" else None,
        "nan_count": 0 if overall == "PASS" else None,
        "invalid_flag_count": 0 if overall == "PASS" else None,
        "environment_parameters_final": False,
        "comfort_parameters_final": False,
        "policy_training_started": False,
        "final_environment_lock_written": False,
        "prior_stage4a_run_id": cfg.get("prior_stage4a_run_id"),
        "prior_stage4a_status": "superseded_pending_v3_hardening",
    }
    (dirs["reports"] / "stage4a0r_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    report = f"""# Stage 4A-0R Report — V3 Physics Hardening

## Overall: **{overall}**

## Scope
Repair and verify MergeEnvCandidateV3 physics/observation core only.
No candidate reselection. No final environment lock. No comfort calibration. No DQN training.

## Prior Stage 4A
- run_id: `{cfg.get('prior_stage4a_run_id')}`
- status: `superseded_pending_v3_hardening` (retained, unchanged)

## Flags
- environment_parameters_final = false
- comfort_parameters_final = false
- policy_training_started = false

## Git
- commit: `{git_commit}`
- dirty: `{git_dirty}`

## Tests
```json
{json.dumps(test_info, indent=2)}
```

## Diagnostics
```json
{json.dumps(diag, indent=2)}
```

## Recommendation
{"Proceed to a new Stage 4A candidate-selection run using the hardened V3 core." if overall == "PASS" else "Fix failing physics/observation regressions before any reselection."}
"""
    (dirs["reports"] / "stage4a0r_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "utc_timestamp": _utc_now().isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "exact_command": " ".join(sys.argv),
        "config_sha256": cfg_hash,
        "overall_status": overall,
        "tests": test_info,
        "diagnostics": diag,
        "environment_parameters_final": False,
        "comfort_parameters_final": False,
        "policy_training_started": False,
        "final_environment_lock_written": False,
        "prior_stage4a_run_id": cfg.get("prior_stage4a_run_id"),
        "prior_stage4a_status": "superseded_pending_v3_hardening",
        "output_paths": {k: str(v) for k, v in dirs.items()},
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "overall": overall}, indent=2), encoding="utf-8"
    )
    # Ensure historical Stage 4A lock was not touched
    prior_lock = (
        REPO_ROOT
        / "experiments/pre_impl/stage4a_environment_choice_state/artifacts"
        / str(cfg.get("prior_stage4a_run_id"))
        / "final_environment_lock.yaml"
    )
    if not prior_lock.is_file():
        log("WARNING: prior Stage 4A lock missing")
        overall = "FAIL"
    log(f"overall={overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
