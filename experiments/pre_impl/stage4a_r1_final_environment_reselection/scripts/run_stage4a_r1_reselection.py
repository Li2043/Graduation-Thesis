#!/usr/bin/env python3
"""Stage 4A-R1 — final environment reselection on hardened V3 + quintic geometry.

No DQN. No comfort calibration. No PBRS λ calibration.
Does not overwrite historical Stage 4A / 4A-0R / 4A-0R2 artifacts.
"""

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

SOURCE_MODULES = [
    "src/thesis/envs/merge_env_candidate_v3.py",
    "src/thesis/envs/final_route_geometry.py",
    "src/thesis/envs/final_observation.py",
    "src/thesis/envs/vehicle_dynamics.py",
    "src/thesis/envs/idm_background.py",
    "src/thesis/envs/final_environment_config.py",
    "src/thesis/certification/choice_state_scenarios.py",
    "src/thesis/certification/choice_state_certification.py",
    "src/thesis/certification/choice_state_metrics.py",
    "src/thesis/certification/environment_candidate_selection.py",
    "src/thesis/certification/holdout_signatures.py",
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


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


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
    out: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "gymnasium", "pytest", "torch", "yaml"):
        modname = "yaml" if name == "yaml" else name
        try:
            m = __import__(modname)
            out[name] = getattr(m, "__version__", "installed")
        except Exception:
            out[name] = "not_installed"
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {
                k: (json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v)
                for k, v in r.items()
            }
            w.writerow(flat)


def _geometry_rows() -> list[dict[str, Any]]:
    from thesis.certification.choice_state_scenarios import GEOMETRY
    from thesis.envs.final_route_geometry import build_final_route_geometry

    rows = []
    for g in GEOMETRY:
        geom = build_final_route_geometry(g)
        d = geom.diagnostics()
        d["route_recovery_max_error_mainline"] = geom.max_route_recovery_error("mainline", n=1000)
        d["route_recovery_max_error_ramp"] = geom.max_route_recovery_error("ramp", n=1000)
        d["connector_model"] = "quintic_lateral_transition"
        rows.append(d)
    return rows


def _resolved_blocks_payload() -> dict[str, Any]:
    from thesis.certification.choice_state_scenarios import (
        GEOMETRY,
        build_ic_blocks,
        materialize_block_for_geometry,
    )

    cal, val = build_ic_blocks()
    by_geom: dict[str, Any] = {}
    for g in GEOMETRY:
        by_geom[g.geometry_id] = {
            "calibration": [materialize_block_for_geometry(b, g).to_dict() for b in cal],
            "validation": [materialize_block_for_geometry(b, g).to_dict() for b in val],
        }
    return {
        "calibration_template": [b.to_dict() for b in cal],
        "validation_template": [b.to_dict() for b in val],
        "materialised_by_geometry": by_geom,
        "validation_replacements": {
            "validation_001": {
                "mainline_speed": 20.0,
                "ramp_speed": 20.0,
                "delta_arrival": -0.4,
                "background_headway": 1.8,
            },
            "validation_006": {
                "mainline_speed": 16.0,
                "ramp_speed": 18.0,
                "delta_arrival": 0.0,
                "background_headway": 1.2,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "stage4a_r1.yaml",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

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
    source_hashes = {rel: _sha256(REPO_ROOT / rel) for rel in SOURCE_MODULES}

    log(f"run_id={run_id}")
    log(f"git_commit={git_commit} dirty={git_dirty}")
    log("writing fully resolved block definitions before candidate execution")
    resolved_blocks = _resolved_blocks_payload()
    _write_yaml(EXP_ROOT / "configs" / "resolved_ic_blocks.yaml", resolved_blocks)
    _write_yaml(dirs["artifacts"] / "resolved_ic_blocks.yaml", resolved_blocks)

    from thesis.certification.choice_state_scenarios import GEOMETRY, build_ic_blocks
    from thesis.certification.holdout_signatures import (
        assert_no_duplicate_holdout_for_geometries,
        audit_holdout_signatures_for_geometries,
    )

    cal_blocks, val_blocks = build_ic_blocks()
    assert_no_duplicate_holdout_for_geometries(cal_blocks, val_blocks, GEOMETRY)
    holdout_audit = audit_holdout_signatures_for_geometries(cal_blocks, val_blocks, GEOMETRY)
    dup_count = sum(int(r["n_duplicate_signatures"]) for r in holdout_audit)
    log(f"holdout duplicate signature count across G1–G3 = {dup_count}")

    geom_rows = _geometry_rows()
    for r in geom_rows:
        if not r["physically_feasible"]:
            raise RuntimeError(f"geometry not feasible: {r['geometry_id']}")
        if r["route_recovery_max_error_ramp"] > 0.01 + 1e-12:
            raise RuntimeError(f"route recovery error too large: {r['geometry_id']}")
        if r["maximum_implied_lateral_acceleration_at_20"] > 3.0 + 1e-12:
            raise RuntimeError(f"a_lat@20 too large: {r['geometry_id']}")

    _write_yaml(
        dirs["artifacts"] / "resolved_config.yaml",
        {
            "run_id": run_id,
            "stage": "stage4a_r1_final_environment_reselection",
            "config_sha256": cfg_hash,
            "config": cfg,
            "source_hashes": source_hashes,
            "prior_stage4a0r_run_id": cfg.get("prior_stage4a0r_run_id"),
            "prior_stage4a0r2_run_id": cfg.get("prior_stage4a0r2_run_id"),
            "superseded_stage4a_run_id": cfg.get("superseded_stage4a_run_id"),
            "superseded_lock_sha256": cfg.get("superseded_lock_sha256"),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "holdout_audit": holdout_audit,
            "n_calibration_blocks": len(cal_blocks),
            "n_validation_blocks": len(val_blocks),
            "comfort_parameters_final": False,
            "policy_training_started": False,
        },
    )

    test_info = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "status": "SKIPPED"}
    if not args.skip_tests:
        log("running pytest")
        targets = cfg.get("pytest_targets") or []
        cmd = [sys.executable, "-m", "pytest", "-q", *targets]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
        (dirs["logs"] / "pytest.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        test_info = _parse_pytest(proc.stdout + "\n" + proc.stderr)
        log(f"pytest status={test_info}")

    from thesis.certification.choice_state_scenarios import (
        GO_PROFILES,
        YIELD_PROFILES,
        build_environment_candidates,
    )
    from thesis.certification.environment_candidate_selection import (
        build_final_environment_lock,
        select_environment_candidate,
        write_processed_tables,
    )

    log("selecting environment candidate (calibration only; holdout once; no reselection)")
    result = select_environment_candidate()
    overall = result["overall"]
    if test_info["status"] == "FAIL":
        overall = "FAIL"
        result["overall"] = overall
        result["environment_parameters_final"] = False

    for tr in result["traces"]:
        tr["run_id"] = run_id

    candidates = build_environment_candidates()

    _jsonl(dirs["raw"] / "environment_candidates.jsonl", [c.to_dict() for c in candidates])
    _jsonl(
        dirs["raw"] / "initial_condition_blocks.jsonl",
        [b.to_dict() for b in cal_blocks + val_blocks],
    )
    _jsonl(dirs["raw"] / "transition_trace.jsonl", result["traces"])
    _jsonl(dirs["raw"] / "choice_matrices.jsonl", result["matrices"])
    _jsonl(dirs["raw"] / "background_response.jsonl", result["background_rows"])
    _jsonl(dirs["raw"] / "candidate_failures.jsonl", result["failures"])
    _jsonl(
        dirs["raw"] / "validation_results.jsonl",
        [result["validation"]] if result.get("validation") else [],
    )

    write_processed_tables(result, dirs["processed"])
    _write_csv(dirs["processed"] / "geometry_diagnostics.csv", geom_rows)
    _write_csv(
        dirs["processed"] / "holdout_signature_audit.csv",
        [
            {
                "geometry_id": r["geometry_id"],
                "n_calibration": r["n_calibration"],
                "n_validation": r["n_validation"],
                "n_duplicate_signatures": r["n_duplicate_signatures"],
                "pass": r["pass"],
                "duplicates": r["duplicates"],
            }
            for r in holdout_audit
        ],
    )

    lock_hash = None
    lock_path = None
    if overall == "PASS" and result.get("selected_candidate") and result.get("environment_parameters_final"):
        lock = build_final_environment_lock(
            selected=result["selected_candidate"],
            calibration_blocks=result["calibration_blocks"],
            validation_blocks=result["validation_blocks"],
            git_commit=git_commit,
            config_hashes={"stage4a_r1.yaml": cfg_hash},
            source_hashes=source_hashes,
            holdout_audit=holdout_audit,
            superseded_stage4a_run_id=str(cfg.get("superseded_stage4a_run_id")),
            superseded_lock_sha256=str(cfg.get("superseded_lock_sha256")),
        )
        lock_path = dirs["artifacts"] / "final_environment_lock.yaml"
        _write_yaml(lock_path, lock)
        lock_hash = _sha256(lock_path)
        (dirs["artifacts"] / "final_environment_lock.sha256").write_text(
            f"{lock_hash}  final_environment_lock.yaml\n", encoding="utf-8"
        )
        log(f"wrote final lock hash={lock_hash}")
    else:
        result["environment_parameters_final"] = False
        log("no final environment lock (stage not full PASS)")

    sel = result.get("selected_candidate") or {}
    val = result.get("validation") or {}
    cal_row = next(
        (r for r in result["candidates"] if r.get("selected")),
        result["candidates"][0] if result["candidates"] else {},
    )

    summary = {
        "run_id": run_id,
        "overall": overall,
        "selected_candidate_id": sel.get("candidate_id"),
        "candidate_count": len(candidates),
        "n_calibration_blocks": len(cal_blocks),
        "n_validation_blocks": len(val_blocks),
        "duplicate_signature_count": dup_count,
        "feasible_candidate_count": len(result.get("feasible_candidate_ids", [])),
        "calibration_certified": cal_row.get("calibration_certified"),
        "validation_certified": val.get("n_certified"),
        "order_gap": {
            "calibration_median": cal_row.get("median_normalised_order_gap"),
            "calibration_max": cal_row.get("maximum_normalised_order_gap"),
            "validation_median": val.get("median_normalised_order_gap"),
            "validation_max": val.get("maximum_normalised_order_gap"),
        },
        "background_relevance": {
            "calibration": cal_row.get("background_relevance_rate_calibration"),
            "validation": val.get("background_relevance_rate"),
        },
        "spontaneous_background_collision_count": cal_row.get(
            "spontaneous_background_collision_count"
        ),
        "label_swap_max_error": val.get("label_swap_max_error"),
        "geometry_diagnostics": geom_rows,
        "integrity_counts": {
            k: cal_row.get(k)
            for k in (
                "route_discontinuity_count",
                "repeated_exit_count",
                "invalid_flag_count",
                "nan_inf_count",
                "fixture_count",
            )
        },
        "environment_parameters_final": bool(result.get("environment_parameters_final")),
        "comfort_parameters_final": False,
        "policy_training_started": False,
        "tests": test_info,
        "final_lock_sha256": lock_hash,
        "prior_stage4a0r_run_id": cfg.get("prior_stage4a0r_run_id"),
        "prior_stage4a0r2_run_id": cfg.get("prior_stage4a0r2_run_id"),
        "superseded_stage4a_run_id": cfg.get("superseded_stage4a_run_id"),
        "superseded_lock_sha256": cfg.get("superseded_lock_sha256"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    (dirs["reports"] / "stage4a_r1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    report_lines = [
        "# Stage 4A-R1 Report — Final Environment Reselection",
        "",
        f"## 1. Overall: **{overall}**",
        "",
        "## 2. Scope",
        "Reselection on hardened V3 physics (4A-0R) and quintic merge geometry (4A-0R2).",
        "No DQN. No comfort calibration. No PBRS λ calibration.",
        "",
        f"## 3. Git: `{git_commit}` dirty=`{git_dirty}`",
        "",
        f"## 4. Prior runs",
        f"- Stage 4A-0R: `{cfg.get('prior_stage4a0r_run_id')}`",
        f"- Stage 4A-0R2: `{cfg.get('prior_stage4a0r2_run_id')}`",
        f"- Superseded Stage 4A: `{cfg.get('superseded_stage4a_run_id')}`",
        f"- Superseded lock SHA-256: `{cfg.get('superseded_lock_sha256')}`",
        "",
        "## 5. Tests",
        f"```json\n{json.dumps(test_info, indent=2)}\n```",
        "",
        f"## 6. Candidates: {len(candidates)}; feasible: {len(result.get('feasible_candidate_ids', []))}",
        f"## 7. Blocks: calibration={len(cal_blocks)}, validation={len(val_blocks)}, "
        f"duplicate_signatures={dup_count}",
        f"## 8. Selected: `{sel.get('candidate_id')}`",
        f"## 9. Calibration certified: {cal_row.get('calibration_certified')}/{cal_row.get('calibration_n')}",
        f"## 10. Validation certified: {val.get('n_certified')}/{val.get('n_blocks')} pass={val.get('pass')}",
        f"## 11. Order gaps: cal med/max="
        f"{cal_row.get('median_normalised_order_gap')}/{cal_row.get('maximum_normalised_order_gap')}; "
        f"val med/max={val.get('median_normalised_order_gap')}/{val.get('maximum_normalised_order_gap')}",
        f"## 12. Background relevance: cal={cal_row.get('background_relevance_rate_calibration')} "
        f"val={val.get('background_relevance_rate')}",
        f"## 13. Spontaneous collisions: {cal_row.get('spontaneous_background_collision_count')}",
        f"## 14. Label-swap max error: {val.get('label_swap_max_error')}",
        "",
        "## 15. Geometry diagnostics",
    ]
    for r in geom_rows:
        report_lines.append(
            f"- **{r['geometry_id']}**: world_x={r['connector_world_x_length']:.3f}, "
            f"arc={r['connector_arc_length']:.6f}, "
            f"max|heading|={r['maximum_abs_heading']:.6f}, "
            f"max|κ|={r['maximum_abs_curvature']:.8f}, "
            f"a_lat@20={r['maximum_implied_lateral_acceleration_at_20']:.6f}, "
            f"recover_err_ramp={r['route_recovery_max_error_ramp']:.3e}"
        )
    report_lines += [
        "",
        f"## 16. Integrity: {json.dumps(summary['integrity_counts'])}",
        f"## 17. Final lock: `{lock_path}` sha256=`{lock_hash}`",
        f"## 18. Flags: environment_parameters_final={result.get('environment_parameters_final')}, "
        "comfort_parameters_final=false, policy_training_started=false",
        "",
    ]
    (dirs["reports"] / "stage4a_r1_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    manifest = {
        "run_id": run_id,
        "utc_timestamp": _utc_now().isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_and_packages": _versions(),
        "exact_command": " ".join([sys.executable, str(SCRIPT_PATH)] + sys.argv[1:]),
        "configuration_file_hashes": {"stage4a_r1.yaml": cfg_hash},
        "source_hashes": source_hashes,
        "prior_stage4a0r_run_id": cfg.get("prior_stage4a0r_run_id"),
        "prior_stage4a0r2_run_id": cfg.get("prior_stage4a0r2_run_id"),
        "superseded_stage4a_run_id": cfg.get("superseded_stage4a_run_id"),
        "superseded_lock_sha256": cfg.get("superseded_lock_sha256"),
        "environment_candidate_ids": [c.candidate_id for c in candidates],
        "feasible_candidate_ids": result.get("feasible_candidate_ids"),
        "selected_candidate_id": sel.get("candidate_id"),
        "n_calibration_blocks": len(cal_blocks),
        "n_validation_blocks": len(val_blocks),
        "duplicate_signature_count": dup_count,
        "macro_definitions": {
            "GO": [p.profile_id for p in GO_PROFILES],
            "YIELD": [p.profile_id for p in YIELD_PROFILES],
        },
        "physics_dt": 0.05,
        "policy_interval": 0.20,
        "physics_substeps_per_action": 4,
        "observation_dimension": 27,
        "route_geometry_version": "v3_quintic_arc_length_4a0r2",
        "output_paths": {k: str(v) for k, v in dirs.items()},
        "final_lock_path": str(lock_path) if lock_path else None,
        "final_lock_sha256": lock_hash,
        "overall_status": overall,
        "environment_parameters_final": bool(result.get("environment_parameters_final")),
        "comfort_parameters_final": False,
        "policy_training_started": False,
        "tests": test_info,
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    # This experiment's latest_run pointer only (do not alter 4A-0R / 4A-0R2).
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "overall": overall,
                "stage": "stage4a_r1_final_environment_reselection",
                "selected_candidate_id": sel.get("candidate_id"),
                "environment_parameters_final": bool(result.get("environment_parameters_final")),
                "prior_stage4a0r_run_id": cfg.get("prior_stage4a0r_run_id"),
                "prior_stage4a0r2_run_id": cfg.get("prior_stage4a0r2_run_id"),
                "superseded_stage4a_run_id": cfg.get("superseded_stage4a_run_id"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"overall={overall}")
    return 0 if overall == "PASS" and test_info["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
