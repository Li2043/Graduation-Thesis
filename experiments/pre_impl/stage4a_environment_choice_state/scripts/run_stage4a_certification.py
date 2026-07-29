#!/usr/bin/env python3
"""Stage 4A environment candidate selection + choice-state certification (no DQN)."""

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


def _write_report(
    path: Path,
    *,
    overall: str,
    result: dict[str, Any],
    git_commit: str,
    git_dirty: bool,
    lock_hash: str | None,
) -> None:
    sel = result.get("selected_candidate") or {}
    val = result.get("validation") or {}
    lines = [
        "# Stage 4A Report — Environment Candidate Selection & Choice-State Certification",
        "",
        f"## 1. Overall: **{overall}**",
        "",
        "## 2. Scope",
        "Environment calibration and genuine choice-state certification.",
        "",
        "## 3. Policy training",
        "Policy training did **not** occur (`policy_training_started=false`).",
        "",
        "## 4. Comfort parameters",
        "Comfort parameters remain unresolved (`comfort_parameters_final=false`).",
        "Stage 3B failed; comfort does not influence Stage 4A ranking.",
        "",
        f"## 5. Git",
        f"- commit: `{git_commit}`",
        f"- dirty: `{git_dirty}`",
        "",
        "## 6. Timing",
        "- physics_dt = 0.05 s",
        "- policy_interval = 0.20 s",
        "- physics_substeps_per_action = 4",
        "",
        "## 7–8. Candidates",
        "Geometries G1–G3; IDM I1–I3; priority G1-I1 … G3-I3.",
        "",
        "## 9. Blocks",
        f"- calibration: {len(result.get('calibration_blocks', []))}",
        f"- validation: {len(result.get('validation_blocks', []))}",
        "",
        "## 10. Feasibility table",
    ]
    for row in result.get("candidates", []):
        lines.append(
            f"- `{row['candidate_id']}` rank={row['priority_rank']} "
            f"cal={row.get('calibration_certified')}/{row.get('calibration_n')} "
            f"feasible={row.get('calibration_feasible')} "
            f"reasons={row.get('rejection_reasons')}"
        )
    lines += [
        "",
        "## 11. Rejection reasons",
        json.dumps(result.get("failures", []), indent=2, default=str),
        "",
        f"## 12. Selected candidate: `{sel.get('candidate_id')}`",
        "",
        f"## 13. Calibration certification: see candidate table",
        f"## 14. Holdout certification: {val.get('n_certified')}/{val.get('n_blocks')} pass={val.get('pass')}",
        "",
        "## 15–18. Choice matrices / conventions / GOGO / YY",
        "See `data/processed/*/choice_matrix_summary.csv` and raw `choice_matrices.jsonl`.",
        "",
        f"## 19. Order-gap (validation): median={val.get('median_normalised_order_gap')} "
        f"max={val.get('maximum_normalised_order_gap')}",
        "",
        "## 20–22. Unilateral / background",
        f"- validation background relevance: {val.get('background_relevance_rate')}",
        "",
        f"## 23. Label-swap max error (validation): {val.get('label_swap_max_error')}",
        "",
        "## 24. Integrity",
        "See candidate_summary integrity columns.",
        "",
        "## 25. Acceleration diagnostics",
        "See `acceleration_trace_summary.csv` (comfort H not used for ranking).",
        "",
        f"## 26. Final lock hash: `{lock_hash}`",
        "",
        "## 27. Recommendation",
        (
            "Proceed to subsequent stages with the locked environment; resolve comfort in Stage 3B-R1."
            if overall == "PASS"
            else "Do not freeze environment parameters; do not begin policy training."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "environment_candidates.yaml",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))

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

    log(f"run_id={run_id}")
    log(f"git_commit={git_commit} dirty={git_dirty}")

    cfg_paths = {
        "environment_candidates": (EXP_ROOT / "configs" / "environment_candidates.yaml").resolve(),
        "calibration_blocks": (EXP_ROOT / "configs" / "calibration_blocks.yaml").resolve(),
        "validation_blocks": (EXP_ROOT / "configs" / "validation_blocks.yaml").resolve(),
        "choice_macros": (EXP_ROOT / "configs" / "choice_macros.yaml").resolve(),
    }
    config_hashes = {k: _sha256(p) for k, p in cfg_paths.items()}
    _write_yaml(
        dirs["artifacts"] / "resolved_config.yaml",
        {
            "run_id": run_id,
            "config_paths": {k: str(v) for k, v in cfg_paths.items()},
            "config_sha256": config_hashes,
            "loaded_primary": _load_yaml(args.config),
        },
    )

    test_info = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "status": "SKIPPED"}
    if not args.skip_tests:
        log("running pytest")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/envs/test_merge_env_candidate_v3.py",
            "tests/envs/test_idm_background.py",
            "tests/certification/test_choice_state_metrics.py",
            "tests/certification/test_environment_candidate_selection.py",
            "tests/integration/test_stage4a_choice_state_pipeline.py",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        (dirs["logs"] / "pytest.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        test_info = _parse_pytest(proc.stdout + "\n" + proc.stderr)
        log(f"pytest status={test_info}")

    from thesis.certification.environment_candidate_selection import (
        build_final_environment_lock,
        select_environment_candidate,
        write_processed_tables,
    )
    from thesis.certification.choice_state_scenarios import (
        GO_PROFILES,
        YIELD_PROFILES,
        build_environment_candidates,
        build_ic_blocks,
    )

    log("selecting environment candidate (calibration only; holdout once)")
    result = select_environment_candidate()
    overall = result["overall"]
    if test_info["status"] == "FAIL":
        overall = "FAIL"
        result["overall"] = overall
        result["environment_parameters_final"] = False

    # Stamp run_id on traces
    for tr in result["traces"]:
        tr["run_id"] = run_id

    candidates = build_environment_candidates()
    cal_blocks, val_blocks = build_ic_blocks()

    _jsonl(
        dirs["raw"] / "environment_candidates.jsonl",
        [c.to_dict() for c in candidates],
    )
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

    lock_hash = None
    lock_path = None
    if overall == "PASS" and result.get("selected_candidate"):
        lock = build_final_environment_lock(
            selected=result["selected_candidate"],
            calibration_blocks=result["calibration_blocks"],
            validation_blocks=result["validation_blocks"],
            git_commit=git_commit,
            config_hashes=config_hashes,
        )
        lock_path = dirs["artifacts"] / "final_environment_lock.yaml"
        _write_yaml(lock_path, lock)
        lock_hash = _sha256(lock_path)
        (dirs["artifacts"] / "final_environment_lock.sha256").write_text(
            f"{lock_hash}  final_environment_lock.yaml\n", encoding="utf-8"
        )
        log(f"wrote final lock hash={lock_hash}")
    else:
        log("no final environment lock (stage not PASS)")

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
        "calibration_certified": cal_row.get("calibration_certified"),
        "validation_certified": val.get("n_certified"),
        "feasible_candidate_count": len(result.get("feasible_candidate_ids", [])),
        "environment_parameters_final": result.get("environment_parameters_final"),
        "comfort_parameters_final": False,
        "policy_training_started": False,
        "tests": test_info,
        "final_lock_sha256": lock_hash,
    }
    (dirs["reports"] / "stage4a_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    _write_report(
        dirs["reports"] / "stage4a_report.md",
        overall=overall,
        result=result,
        git_commit=git_commit,
        git_dirty=git_dirty,
        lock_hash=lock_hash,
    )

    manifest = {
        "run_id": run_id,
        "utc_timestamp": _utc_now().isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_and_packages": _versions(),
        "exact_command": " ".join(sys.argv),
        "configuration_file_hashes": config_hashes,
        "environment_candidate_ids": [c.candidate_id for c in candidates],
        "candidate_priority_order": [c.candidate_id for c in candidates],
        "physics_dt": 0.05,
        "policy_interval": 0.20,
        "n_calibration_blocks": len(cal_blocks),
        "n_validation_blocks": len(val_blocks),
        "macro_definitions": {
            "GO": [p.profile_id for p in GO_PROFILES],
            "YIELD": [p.profile_id for p in YIELD_PROFILES],
        },
        "calibration_certification_counts": {
            r["candidate_id"]: r.get("calibration_certified") for r in result["candidates"]
        },
        "validation_certification_counts": {
            sel.get("candidate_id"): val.get("n_certified")
        },
        "selected_candidate_id": sel.get("candidate_id"),
        "selected_geometry": (sel.get("geometry") or {}).get("geometry_id"),
        "selected_idm_profile": (sel.get("idm") or {}).get("profile_id"),
        "order_gap_metrics": {
            "calibration_median": cal_row.get("median_normalised_order_gap"),
            "calibration_max": cal_row.get("maximum_normalised_order_gap"),
            "validation_median": val.get("median_normalised_order_gap"),
            "validation_max": val.get("maximum_normalised_order_gap"),
        },
        "background_relevance_rates": {
            "calibration": cal_row.get("background_relevance_rate_calibration"),
            "validation": val.get("background_relevance_rate"),
        },
        "background_spontaneous_collision_count": cal_row.get(
            "spontaneous_background_collision_count"
        ),
        "label_swap_maximum_error": max(
            float(c.get("label_swap_max_error") or 0) for c in result["certifications"]
        )
        if result["certifications"]
        else 0.0,
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
        "output_paths": {k: str(v) for k, v in dirs.items()},
        "final_lock_path": str(lock_path) if lock_path else None,
        "final_lock_sha256": lock_hash,
        "overall_status": overall,
        "policy_training_started": False,
        "comfort_parameters_final": False,
        "environment_parameters_final": bool(result.get("environment_parameters_final")),
        "tests": test_info,
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "overall": overall}, indent=2), encoding="utf-8"
    )
    log(f"overall={overall}")
    return 0 if overall == "PASS" and test_info["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
