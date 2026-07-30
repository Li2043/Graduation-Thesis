#!/usr/bin/env python3
"""Stage 4A-0R2 merge geometry correction runner (no DQN, no reselection, no lock)."""

from __future__ import annotations

import argparse
import csv
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


def _geometry_rows() -> list[dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from thesis.certification.choice_state_scenarios import GEOMETRY
    from thesis.envs.final_route_geometry import build_final_route_geometry

    rows = []
    for g in GEOMETRY:
        geom = build_final_route_geometry(g)
        d = geom.diagnostics()
        d["route_recovery_max_error_mainline"] = geom.max_route_recovery_error("mainline", n=1000)
        d["route_recovery_max_error_ramp"] = geom.max_route_recovery_error("ramp", n=1000)
        rows.append(d)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "stage4a0r2.yaml",
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
            "environment_parameters_final": False,
            "comfort_parameters_final": False,
            "policy_training_started": False,
            "prior_stage4a_status": "superseded_pending_v3_hardening",
        },
    )

    log(f"run_id={run_id}")
    log(f"git_commit={git_commit} dirty={git_dirty}")

    targets = cfg.get("pytest_targets") or []
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
    (dirs["logs"] / "pytest.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    test_info = _parse_pytest(proc.stdout + "\n" + proc.stderr)
    log(f"pytest={test_info}")

    rows = _geometry_rows()
    csv_path = dirs["processed"] / "geometry_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    overall = "PASS" if test_info["status"] == "PASS" else "FAIL"
    if any(not r["physically_feasible"] for r in rows):
        overall = "FAIL"
    if any(
        r["route_recovery_max_error_ramp"] > 0.01 or r["route_recovery_max_error_mainline"] > 0.01
        for r in rows
    ):
        overall = "FAIL"

    summary = {
        "run_id": run_id,
        "overall": overall,
        "tests": test_info,
        "geometries": rows,
        "environment_parameters_final": False,
        "comfort_parameters_final": False,
        "policy_training_started": False,
        "final_environment_lock_written": False,
        "prior_stage4a0r_run_id": cfg.get("prior_stage4a0r_run_id"),
        "prior_stage4a_run_id": cfg.get("prior_stage4a_run_id"),
        "prior_stage4a_status": "superseded_pending_v3_hardening",
        "connector_model": "quintic_lateral_transition",
    }
    (dirs["reports"] / "stage4a0r2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Stage 4A-0R2 Report — Merge Centreline Correction",
        "",
        f"## Overall: **{overall}**",
        "",
        "## Scope",
        "Quintic ramp→mainline convergence between merge_start and merge_end.",
        "No candidate reselection. No final lock. No comfort calibration. No DQN.",
        "",
        f"## Git: `{git_commit}` dirty=`{git_dirty}`",
        "",
        "## Tests",
        f"```json\n{json.dumps(test_info, indent=2)}\n```",
        "",
        "## Geometry diagnostics (G1/G2/G3)",
    ]
    for r in rows:
        lines.append(
            f"- **{r['geometry_id']}**: world_x={r['connector_world_x_length']:.3f} m, "
            f"arc={r['connector_arc_length']:.6f} m, "
            f"max|heading|={r['maximum_abs_heading']:.6f} rad, "
            f"max|κ|={r['maximum_abs_curvature']:.8f} 1/m, "
            f"min R={r['minimum_curvature_radius']:.3f} m, "
            f"a_lat@20={r['maximum_implied_lateral_acceleration_at_20']:.6f} m/s², "
            f"recover_err_ramp={r['route_recovery_max_error_ramp']:.6e} m, "
            f"Δpos_start={r['boundary_position_jump_merge_start']:.3e}, "
            f"Δψ_start={r['boundary_heading_jump_merge_start']:.3e}, "
            f"Δκ_start={r['boundary_curvature_jump_merge_start']:.3e}, "
            f"Δpos_end={r['boundary_position_jump_merge_end']:.3e}, "
            f"Δψ_end={r['boundary_heading_jump_merge_end']:.3e}, "
            f"Δκ_end={r['boundary_curvature_jump_merge_end']:.3e}"
        )
    lines += [
        "",
        "## Flags",
        "- environment_parameters_final = false",
        "- comfort_parameters_final = false",
        "- policy_training_started = false",
        "",
        "## Recommendation",
        (
            "Geometry semantics corrected; proceed later to Stage 4A-R1 candidate selection."
            if overall == "PASS"
            else "Fix geometry failures before any reselection."
        ),
    ]
    (dirs["reports"] / "stage4a0r2_report.md").write_text("\n".join(lines), encoding="utf-8")

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
        "environment_parameters_final": False,
        "comfort_parameters_final": False,
        "policy_training_started": False,
        "final_environment_lock_written": False,
        "prior_stage4a0r_run_id": cfg.get("prior_stage4a0r_run_id"),
        "prior_stage4a_status": "superseded_pending_v3_hardening",
        "output_paths": {k: str(v) for k, v in dirs.items()},
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "overall": overall}, indent=2), encoding="utf-8"
    )
    log(f"overall={overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
