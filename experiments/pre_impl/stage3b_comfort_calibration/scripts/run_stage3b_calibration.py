#!/usr/bin/env python3
"""Stage 3B comfort calibration runner (no DQN training)."""

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


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    out: dict[str, str] = {}
    for name in ("numpy", "gymnasium", "pytest", "torch", "yaml"):
        modname = "yaml" if name == "yaml" else name
        try:
            m = __import__(modname)
            out[name] = getattr(m, "__version__", "installed")
        except Exception:
            out[name] = "not_installed"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "comfort_calibration.yaml",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()

    git_commit = _git(["rev-parse", "HEAD"]) or "unknown"
    git_dirty = bool(_git(["status", "--porcelain"]))
    run_id = _run_id(git_commit)
    dirs = _ensure(run_id)
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(f"[{_utc_now().isoformat()}] {msg}")
        print(msg)

    log(f"run_id={run_id}")
    log(f"git_dirty={git_dirty}")
    if git_dirty:
        log("WARNING: git working tree is DIRTY — dissertation retention wants git_dirty=false")

    cfg = _load_yaml(config_path)
    resolved = dirs["artifacts"] / "resolved_config.yaml"
    _write_yaml(resolved, cfg)
    config_hash = _sha256(resolved)

    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    env = os.environ.copy()
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    paths = [REPO_ROOT / p for p in cfg.get("pytest", {}).get("test_paths", [])]
    extra = list(cfg.get("pytest", {}).get("extra_args", ["-q", "--tb=short"]))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *[str(p) for p in paths], *extra],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    (dirs["logs"] / "pytest.log").write_text(text, encoding="utf-8")
    counts = _parse_pytest(text)
    log(f"pytest rc={proc.returncode} counts={counts}")

    from thesis.calibration.comfort_calibration import (
        confirmatory_scripted_rerun,
        reconstruct_scenario_returns,
        run_comfort_calibration,
    )
    from thesis.calibration.trace_loader import load_and_validate_stage3a_source

    src_cfg = cfg["source_stage3a"]
    manifest, transitions, outcomes, order_pairs = load_and_validate_stage3a_source(
        repo_root=REPO_ROOT,
        stage3a_run_id=str(src_cfg["run_id"]),
        expected_git_commit=str(src_cfg["git_commit"]),
        dt=float(cfg["dt"]),
        gamma=float(cfg["gamma"]),
    )
    # Verify source hashes unchanged after load (byte identity)
    for name, digest in manifest.file_hashes.items():
        if name == "stage3a_summary.json":
            p = (
                REPO_ROOT
                / "experiments/pre_impl/stage3a_scripted_base_outcome_audit/reports"
                / manifest.stage3a_run_id
                / "stage3a_summary.json"
            )
        else:
            p = manifest.raw_dir / name
        if _sha256(p) != digest:
            raise RuntimeError(f"Stage 3A source hash changed during run: {name}")

    log("running offline comfort calibration (no DQN training)")
    result = run_comfort_calibration(
        manifest=manifest,
        transitions=transitions,
        outcomes=outcomes,
        a_comfort_candidates=[float(x) for x in cfg["a_comfort_candidates"]],
        a_hard_candidates=[float(x) for x in cfg["a_hard_candidates"]],
        eta_candidates=[float(x) for x in cfg["eta_candidates"]],
        gamma=float(cfg["gamma"]),
    )
    selection = result["selection"]
    overall = str(selection.overall)
    if counts["status"] != "PASS":
        overall = "FAIL"
        selection.notes.append("pytest_failed")

    # Raw outputs
    (dirs["raw"] / "source_trace_manifest.json").write_text(
        json.dumps(
            {
                "stage3a_run_id": manifest.stage3a_run_id,
                "stage3a_git_commit": manifest.stage3a_git_commit,
                "file_hashes": manifest.file_hashes,
                "summary_overall": manifest.summary_overall,
                "git_dirty": manifest.summary_git_dirty,
                "policy_training_started": manifest.policy_training_started,
                "dt": manifest.dt,
                "gamma": manifest.gamma,
                "n_transitions": manifest.n_transitions,
                "n_outcomes": manifest.n_outcomes,
                "matched_order_pairs_count": len(order_pairs),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _jsonl(dirs["raw"] / "included_transitions.jsonl", result["included"])
    _jsonl(dirs["raw"] / "excluded_transitions.jsonl", result["excluded"])
    _jsonl(
        dirs["raw"] / "hard_braking_windows.jsonl",
        [w.to_dict() for w in result["windows"]] + result["window_rows"],
    )
    _jsonl(
        dirs["raw"] / "threshold_candidates.jsonl",
        [t.to_dict() for t in result["threshold_metrics"]],
    )
    _jsonl(
        dirs["raw"] / "eta_candidates.jsonl",
        [e.to_dict() for e in result["eta_metrics"]],
    )

    # Processed
    _csv(dirs["processed"] / "acceleration_distribution.csv", result["accel_distribution"])
    _csv(
        dirs["processed"] / "threshold_candidate_summary.csv",
        [
            {
                "a_comfort": t.a_comfort,
                "a_hard": t.a_hard,
                "feasible": t.feasible,
                "separation_score": t.separation_score,
                "nominal_nonzero_rate": t.nominal.get("nonzero_rate"),
                "nominal_mean_h": t.nominal.get("mean_h"),
                "nominal_saturation_rate": t.nominal.get("saturation_rate"),
                "hard_nonzero_rate": t.hard_window.get("nonzero_rate"),
                "hard_mean_h": t.hard_window.get("mean_h"),
                "rejection_reasons": ";".join(t.rejection_reasons),
                "selection_rank": t.selection_rank,
            }
            for t in result["threshold_metrics"]
        ],
    )
    _csv(
        dirs["processed"] / "eta_candidate_summary.csv",
        [
            {
                "a_comfort": e.a_comfort,
                "a_hard": e.a_hard,
                "eta_hard_brake": e.eta_hard_brake,
                "feasible": e.feasible,
                "median_nominal_share": e.median_nominal_share,
                "max_nominal_share": e.max_nominal_share,
                "median_paired_share_diff": e.median_paired_share_diff,
                "median_order_gap": e.median_order_gap,
                "max_order_gap": e.max_order_gap,
                "n_ordering_violations": e.n_ordering_violations,
                "rejection_reasons": ";".join(e.rejection_reasons),
                "selection_rank": e.selection_rank,
            }
            for e in result["eta_metrics"]
        ],
    )
    block_thr_rows: list[dict[str, Any]] = []
    for t in result["threshold_metrics"]:
        for b in t.per_block:
            block_thr_rows.append(
                {
                    "a_comfort": t.a_comfort,
                    "a_hard": t.a_hard,
                    "feasible": t.feasible,
                    **b,
                }
            )
    _csv(dirs["processed"] / "block_threshold_summary.csv", block_thr_rows)
    block_eta_rows: list[dict[str, Any]] = []
    for e in result["eta_metrics"]:
        for b in e.per_block:
            block_eta_rows.append(
                {
                    "eta_hard_brake": e.eta_hard_brake,
                    "feasible": e.feasible,
                    **b,
                }
            )
    _csv(dirs["processed"] / "block_eta_summary.csv", block_eta_rows)

    sel_thr = result["selected_threshold"]
    sel_eta = result["selected_eta"]
    _csv(
        dirs["processed"] / "selected_parameter_summary.csv",
        [
            {
                "a_comfort": selection.a_comfort,
                "a_hard": selection.a_hard,
                "eta_hard_brake": selection.eta_hard_brake,
                "threshold_feasible_count": selection.threshold_feasible_count,
                "eta_feasible_count": selection.eta_feasible_count,
                "overall": selection.overall,
                "notes": ";".join(selection.notes),
                "parameters_final": False,
            }
        ],
    )

    # Confirmatory rerun only when parameters selected
    confirm_max_err = None
    confirm_ok = False
    confirmatory_transitions: list[dict[str, Any]] = []
    confirm_rows: list[dict[str, Any]] = []
    if (
        selection.a_comfort is not None
        and selection.a_hard is not None
        and selection.eta_hard_brake is not None
    ):
        offline = reconstruct_scenario_returns(
            transitions,
            a_comfort=float(selection.a_comfort),
            a_hard=float(selection.a_hard),
            eta=float(selection.eta_hard_brake),
            gamma=float(cfg["gamma"]),
        )
        log("running confirmatory scripted audit with selected parameters")
        conf = confirmatory_scripted_rerun(
            a_comfort=float(selection.a_comfort),
            a_hard=float(selection.a_hard),
            eta=float(selection.eta_hard_brake),
            gamma=float(cfg["gamma"]),
            offline_recon=offline,
        )
        confirm_max_err = conf["max_abs_return_difference"]
        confirm_ok = bool(conf["ok"])
        confirmatory_transitions = conf["confirmatory_transitions"]
        confirm_rows = conf["comparisons"]
        if not confirm_ok:
            overall = "FAIL"
            selection.notes.append("confirmatory_rerun_mismatch")
    else:
        log("skipping confirmatory rerun — no fully selected parameter set")
        selection.notes.append("confirmatory_skipped_no_selected_eta")

    _jsonl(dirs["raw"] / "confirmatory_transitions.jsonl", confirmatory_transitions)
    _csv(dirs["processed"] / "confirmatory_return_comparison.csv", confirm_rows)

    # dt sensitivity: no valid alternate-dt reruns available without fabricating accel
    dt_status = "BLOCKED"
    dt_rows = [
        {
            "dt": d,
            "status": "PRIMARY" if abs(float(d) - 0.2) < 1e-15 else "BLOCKED",
            "note": (
                "primary calibration dt"
                if abs(float(d) - 0.2) < 1e-15
                else "valid alternate-dt scripted reruns unavailable; do not naive-interpolate accelerations"
            ),
        }
        for d in cfg.get("dt_sensitivity", {}).get("candidates", [0.1, 0.2, 0.4])
    ]
    _csv(dirs["processed"] / "dt_sensitivity_summary.csv", dt_rows)

    # If overall PASS but dt blocked, remain PASS provisionally per spec
    if overall == "PASS" and dt_status == "BLOCKED":
        selection.notes.append(
            "dt_sensitivity_BLOCKED_final_freeze_requires_Stage4_revalidation"
        )

    filter_stats = result["filter_stats"]
    selected_eta_metrics = sel_eta
    hard_window_metrics = None if sel_thr is None else sel_thr.hard_window
    nominal_share_range = None
    if selected_eta_metrics is not None:
        nominal_share_range = [
            selected_eta_metrics.median_nominal_share,
            selected_eta_metrics.max_nominal_share,
        ]

    summary = {
        "overall": overall,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "unit_tests": counts,
        "source_stage3a_run_id": manifest.stage3a_run_id,
        "source_file_hashes": manifest.file_hashes,
        "included_transition_count": filter_stats.included,
        "excluded_transition_counts": filter_stats.excluded_by_reason,
        "n_threshold_candidates": len(result["threshold_metrics"]),
        "n_threshold_feasible": selection.threshold_feasible_count,
        "selected_a_comfort": selection.a_comfort,
        "selected_a_hard": selection.a_hard,
        "n_eta_candidates": len(result["eta_metrics"]),
        "n_eta_feasible": selection.eta_feasible_count,
        "selected_eta_hard_brake": selection.eta_hard_brake,
        "nominal_safe_braking_share_median_max": nominal_share_range,
        "hard_window_metrics": hard_window_metrics,
        "n_incentive_ordering_violations": (
            None if selected_eta_metrics is None else selected_eta_metrics.n_ordering_violations
        ),
        "median_order_gap": None if selected_eta_metrics is None else selected_eta_metrics.median_order_gap,
        "maximum_order_gap": None if selected_eta_metrics is None else selected_eta_metrics.max_order_gap,
        "confirmatory_max_abs_return_difference": confirm_max_err,
        "confirmatory_ok": confirm_ok,
        "dt_sensitivity_status": dt_status,
        "parameters_final": False,
        "policy_training_started": False,
        "selection_notes": selection.notes,
        "recommendation": (
            "PROCEED TO Stage 4 with provisional comfort parameters; revalidate at final dt"
            if overall == "PASS"
            else "DO NOT freeze comfort parameters — no jointly feasible (threshold, eta) under preregistered grid/rules"
        ),
        "scope": "offline scripted calibration; no DQN training",
    }
    (dirs["reports"] / "stage3b_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Report markdown
    thr_lines = "\n".join(
        f"| {t.a_comfort} | {t.a_hard} | {t.feasible} | {t.separation_score:.6f} | "
        f"{';'.join(t.rejection_reasons) if t.rejection_reasons else ''} |"
        for t in result["threshold_metrics"]
    )
    eta_lines = "\n".join(
        f"| {e.eta_hard_brake} | {e.feasible} | {e.median_nominal_share:.6f} | "
        f"{e.median_paired_share_diff:.6f} | {';'.join(e.rejection_reasons) if e.rejection_reasons else ''} |"
        for e in result["eta_metrics"]
    )
    report = f"""# Stage 3B Report — Comfort / Hard-Braking Calibration

## 1. Overall: **{overall}**

## 2. Scope

**Offline scripted calibration only. No policy training. No DQN parameter updates.**

- `git_dirty = {str(git_dirty).lower()}`
- Git commit: `{git_commit}`
- Config SHA-256: `{config_hash}`
- Primary `dt = {cfg['dt']}` s (eta is a per-transition coefficient at this dt)
- Frozen weights: progress=0.4, exit=0.6, collision=1.0 (unchanged)

## 3. Source Stage 3A

- run_id: `{manifest.stage3a_run_id}`
- expected git commit: `{manifest.stage3a_git_commit}`
- overall: `{manifest.summary_overall}`
- hashes:
{json.dumps(manifest.file_hashes, indent=2)}

## 4. Tests

| Metric | Value |
|--------|-------|
| Passed | {counts['passed']} |
| Failed | {counts['failed']} |
| Status | {counts['status']} |

## 5. Transitions

| Kind | Count |
|------|-------|
| Included (active calibration) | {filter_stats.included} |
| Excluded | {sum(filter_stats.excluded_by_reason.values())} |

Excluded by reason: `{json.dumps(filter_stats.excluded_by_reason)}`

## 6. Acceleration distributions

See `data/processed/{run_id}/acceleration_distribution.csv`.

## 7–8. Threshold candidates

| a_comfort | a_hard | feasible | separation | rejection |
|-----------|--------|----------|------------|-----------|
{thr_lines}

Selected threshold: `a_comfort={selection.a_comfort}`, `a_hard={selection.a_hard}`

## 9–11. Eta candidates

| eta | feasible | median nominal share | paired diff | rejection |
|-----|----------|----------------------|-------------|-----------|
{eta_lines}

Selected eta (smallest feasible): `{selection.eta_hard_brake}`

## 12–15. Selected metrics

- Nominal-safe braking share (median, max): `{nominal_share_range}`
- Hard-window metrics: `{hard_window_metrics}`
- Ordering violations: `{summary['n_incentive_ordering_violations']}`
- Order gaps (median, max): `{summary['median_order_gap']}`, `{summary['maximum_order_gap']}`

## 16. Confirmatory rerun

- max abs return difference: `{confirm_max_err}`
- ok: `{confirm_ok}`

## 17. dt sensitivity

**{dt_status}** — valid scripted reruns at dt∈{{0.1,0.4}} were not available without fabricating accelerations via naive interpolation.

Final parameter freeze requires Stage 4 revalidation at the final simulation frequency.

## 18. Provisional status

**parameters_final = false**

These values are calibrated for this simulator and `dt=0.2` only. They are not
universally valid comfort standards.

## 19. Recommendation

{summary['recommendation']}

## Notes

{selection.notes}
"""
    (dirs["reports"] / "stage3b_report.md").write_text(report, encoding="utf-8")

    manifest_out = {
        "run_id": run_id,
        "utc_timestamp": _utc_now().isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "package_versions": _versions(),
        "operating_system": platform.platform(),
        "command": " ".join(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--config",
                str(config_path),
            ]
        ),
        "configuration_sha256": config_hash,
        "stage3a_source_run_id": manifest.stage3a_run_id,
        "source_file_hashes": manifest.file_hashes,
        "dt": float(cfg["dt"]),
        "gamma": float(cfg["gamma"]),
        "candidate_grids": {
            "a_comfort_candidates": cfg["a_comfort_candidates"],
            "a_hard_candidates": cfg["a_hard_candidates"],
            "eta_candidates": cfg["eta_candidates"],
        },
        "n_threshold_candidates": len(result["threshold_metrics"]),
        "n_eta_candidates": len(result["eta_metrics"]),
        "selected_a_comfort": selection.a_comfort,
        "selected_a_hard": selection.a_hard,
        "selected_eta_hard_brake": selection.eta_hard_brake,
        "selection_rules": cfg.get("selection"),
        "acceptance_thresholds": {
            "threshold_pair_constraints": cfg.get("threshold_pair_constraints"),
            "eta_constraints": cfg.get("eta_constraints"),
        },
        "nominal_safe_penalty_metrics": nominal_share_range,
        "hard_window_metrics": hard_window_metrics,
        "ordering_violation_count": summary["n_incentive_ordering_violations"],
        "order_gap_metrics": {
            "median": summary["median_order_gap"],
            "maximum": summary["maximum_order_gap"],
        },
        "confirmatory_maximum_return_error": confirm_max_err,
        "dt_sensitivity_status": dt_status,
        "output_paths": {k: str(v) for k, v in dirs.items()},
        "overall_status": overall,
        "policy_training_started": False,
        "parameters_final": False,
        "unit_tests": counts,
        "included_transition_count": filter_stats.included,
        "excluded_transition_counts": filter_stats.excluded_by_reason,
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )
    (dirs["logs"] / "runner.log").write_text("\n".join(logs) + "\n", encoding="utf-8")

    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "overall": overall,
                "git_dirty": git_dirty,
                "reports": str(dirs["reports"] / "stage3b_report.md"),
                "manifest": str(dirs["artifacts"] / "manifest.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log(f"overall={overall}")
    log(
        f"selected a_comfort={selection.a_comfort} a_hard={selection.a_hard} "
        f"eta={selection.eta_hard_brake}"
    )
    return 0 if overall in {"PASS", "BLOCKED"} and counts["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
