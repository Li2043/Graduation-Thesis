#!/usr/bin/env python3
"""Stage 3B-R1 — joint comfort calibration on locked Stage 4A-R1 environment."""

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
    "src/thesis/calibration/final_environment_trace_loader.py",
    "src/thesis/calibration/policy_acceleration.py",
    "src/thesis/calibration/joint_comfort_calibration.py",
    "src/thesis/calibration/comfort_validation.py",
    "src/thesis/calibration/comfort_lock.py",
    "src/thesis/rewards/base_reward_v2.py",
    "src/thesis/envs/merge_env_candidate_v3.py",
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
    for name in ("numpy", "gymnasium", "pytest", "yaml"):
        modname = "yaml" if name == "yaml" else name
        try:
            m = __import__(modname)
            out[name] = getattr(m, "__version__", "installed")
        except Exception:
            out[name] = "not_installed"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_ROOT / "configs" / "stage3b_r1.yaml")
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
    log("development Stage 3B = FAIL (historical; unchanged)")

    _write_yaml(
        dirs["artifacts"] / "resolved_config.yaml",
        {
            "run_id": run_id,
            "config_sha256": cfg_hash,
            "config": cfg,
            "source_hashes": source_hashes,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "development_calibration": "FAIL",
            "environment_parameters_final": True,
            "comfort_parameters_final": False,
            "policy_training_started": False,
        },
    )

    from thesis.calibration.comfort_lock import build_comfort_lock, write_comfort_lock
    from thesis.calibration.comfort_validation import (
        confirmatory_online_rerun,
        validate_selected_tuple,
    )
    from thesis.calibration.final_environment_trace_loader import (
        EnvironmentLockError,
        load_final_environment_lock,
    )
    from thesis.calibration.joint_comfort_calibration import (
        A_COMFORT_GRID,
        A_HARD_GRID,
        ETA_GRID,
        assert_hard_window_coverage,
        generate_immutable_traces,
        run_joint_calibration,
    )

    overall = "FAIL"
    comfort_final = False
    status_note = ""
    try:
        lock_path = REPO_ROOT / cfg["source_environment"]["lock_path"]
        loaded = load_final_environment_lock(lock_path)
        if loaded.lock_sha256 != cfg["source_environment"]["expected_lock_sha256"]:
            raise EnvironmentLockError(
                f"lock sha mismatch vs config: {loaded.lock_sha256} != "
                f"{cfg['source_environment']['expected_lock_sha256']}"
            )
        log(f"loaded environment lock sha={loaded.lock_sha256}")
        log(f"candidate={loaded.candidate.candidate_id}")
    except EnvironmentLockError as exc:
        log(f"BLOCKED: {exc}")
        overall = "BLOCKED"
        summary = {
            "run_id": run_id,
            "overall": overall,
            "error": str(exc),
            "environment_parameters_final": True,
            "comfort_parameters_final": False,
            "policy_training_started": False,
        }
        (dirs["reports"] / "stage3b_r1_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (EXP_ROOT / "latest_run.json").write_text(
            json.dumps({"run_id": run_id, "overall": overall}, indent=2), encoding="utf-8"
        )
        return 2

    (dirs["raw"] / "source_environment_lock.json").write_text(
        json.dumps(loaded.summary(), indent=2, default=str), encoding="utf-8"
    )
    _jsonl(
        dirs["raw"] / "initial_condition_blocks.jsonl",
        [b.to_dict() for b in loaded.calibration_blocks + loaded.validation_blocks],
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
        log(f"pytest={test_info}")
        if test_info["status"] == "FAIL":
            overall = "FAIL"

    log("generating immutable final-environment traces")
    try:
        bundle = generate_immutable_traces(
            candidate=loaded.candidate,
            calibration_blocks=loaded.calibration_blocks,
            validation_blocks=loaded.validation_blocks,
            lock_hash=loaded.lock_sha256,
            run_id=run_id,
        )
        assert_hard_window_coverage(bundle)
    except RuntimeError as exc:
        if str(exc).startswith("BLOCKED"):
            log(str(exc))
            overall = "BLOCKED"
            status_note = str(exc)
            bundle = None  # type: ignore[assignment]
        else:
            raise

    selected = None
    validation = None
    confirmatory = None
    cal_result = None
    comfort_hash = None
    comfort_path = None

    if bundle is not None and overall != "BLOCKED":
        _jsonl(dirs["raw"] / "transition_trace.jsonl", bundle.transitions)
        _jsonl(dirs["raw"] / "substep_acceleration_trace.jsonl", bundle.substep_rows)
        _jsonl(dirs["raw"] / "episode_classification.jsonl", bundle.episode_rows)
        _jsonl(dirs["raw"] / "hard_braking_windows.jsonl", bundle.hard_windows)

        log("offline joint tuple evaluation (calibration only)")
        cal_result = run_joint_calibration(bundle)
        selected = cal_result["selected"]
        _write_csv(
            dirs["processed"] / "threshold_candidates.csv",
            cal_result["threshold_pairs"],
        )
        _write_csv(
            dirs["processed"] / "tuple_candidates.csv",
            [
                {k: v for k, v in r.items() if k != "per_block"}
                for r in cal_result["tuple_results"]
            ],
        )
        _write_csv(
            dirs["processed"] / "calibration_feasibility.csv",
            [
                {
                    "a_comfort": r["a_comfort"],
                    "a_hard": r["a_hard"],
                    "eta_H": r["eta_H"],
                    "feasible": r["feasible"],
                    "rejection_reasons": r["rejection_reasons"],
                }
                for r in cal_result["tuple_results"]
            ],
        )

        if selected is None:
            log("no feasible complete tuple — FAIL")
            overall = "FAIL"
            comfort_final = False
        else:
            log(
                f"selected tuple a_comfort={selected['a_comfort']} "
                f"a_hard={selected['a_hard']} eta_H={selected['eta_H']}"
            )
            _write_csv(
                dirs["processed"] / "calibration_block_metrics.csv",
                selected.get("per_block") or [],
            )
            _write_csv(
                dirs["processed"] / "braking_share_summary.csv",
                [
                    {
                        "block_set": "calibration",
                        "median_nominal_share": selected["median_nominal_share"],
                        "max_nominal_share": selected["max_nominal_share"],
                        "median_paired_share_diff": selected["median_paired_share_diff"],
                    }
                ],
            )
            _write_csv(
                dirs["processed"] / "hard_window_pairing.csv",
                [
                    {
                        "median_paired_share_diff": selected["median_paired_share_diff"],
                        "n_paired": selected["n_paired_diffs"],
                        "hard_nonzero_rate": selected["hard_nonzero_rate"],
                        "mean_H_hard": selected["mean_H_hard"],
                        "mean_H_nominal": selected["mean_H_nominal"],
                        "H_separation": selected["H_separation"],
                    }
                ],
            )
            _write_csv(
                dirs["processed"] / "order_bias_summary.csv",
                [
                    {
                        "median_order_gap": selected["median_order_gap"],
                        "max_order_gap": selected["max_order_gap"],
                        "ordering_violations": selected["ordering_violations"],
                    }
                ],
            )

            validation = validate_selected_tuple(
                bundle,
                a_comfort=float(selected["a_comfort"]),
                a_hard=float(selected["a_hard"]),
                eta_h=float(selected["eta_H"]),
            )
            _write_csv(
                dirs["processed"] / "validation_metrics.csv",
                [{k: v for k, v in validation.items() if k != "per_block"}],
            )
            log(f"validation pass={validation['pass']} reasons={validation['rejection_reasons']}")

            if validation["pass"] and test_info["status"] != "FAIL":
                log("confirmatory online rerun")
                confirmatory = confirmatory_online_rerun(
                    candidate=loaded.candidate,
                    calibration_blocks=loaded.calibration_blocks,
                    validation_blocks=loaded.validation_blocks,
                    offline_transitions=bundle.transitions,
                    a_comfort=float(selected["a_comfort"]),
                    a_hard=float(selected["a_hard"]),
                    eta_h=float(selected["eta_H"]),
                    lock_hash=loaded.lock_sha256,
                    run_id=run_id,
                )
                _jsonl(
                    dirs["raw"] / "confirmatory_transition_trace.jsonl",
                    confirmatory["online_transitions"],
                )
                _write_csv(
                    dirs["processed"] / "confirmatory_equivalence.csv",
                    [
                        {
                            "max_per_transition_reward_error": confirmatory[
                                "max_per_transition_reward_error"
                            ],
                            "max_discounted_return_error": confirmatory[
                                "max_discounted_return_error"
                            ],
                            "pass": confirmatory["pass"],
                        }
                    ],
                )
                log(
                    f"confirmatory pass={confirmatory['pass']} "
                    f"rew_err={confirmatory['max_per_transition_reward_error']} "
                    f"ret_err={confirmatory['max_discounted_return_error']}"
                )
                if confirmatory["pass"]:
                    overall = "PASS"
                    comfort_final = True
                    lock = build_comfort_lock(
                        a_comfort=float(selected["a_comfort"]),
                        a_hard=float(selected["a_hard"]),
                        eta_h=float(selected["eta_H"]),
                        candidate_grids={
                            "a_comfort": list(A_COMFORT_GRID),
                            "a_hard": list(A_HARD_GRID),
                            "eta_H": list(ETA_GRID),
                        },
                        calibration_metrics={
                            k: selected[k]
                            for k in (
                                "median_nominal_share",
                                "max_nominal_share",
                                "hard_nonzero_rate",
                                "mean_H_hard",
                                "mean_H_nominal",
                                "H_separation",
                                "median_paired_share_diff",
                                "median_order_gap",
                                "max_order_gap",
                                "ordering_violations",
                            )
                        },
                        validation_metrics={
                            k: validation[k]
                            for k in (
                                "median_nominal_share",
                                "max_nominal_share",
                                "hard_nonzero_rate",
                                "mean_H_hard",
                                "median_paired_share_diff",
                                "median_order_gap",
                                "max_order_gap",
                                "pass",
                            )
                        },
                        environment_lock_path=str(loaded.lock_path),
                        environment_lock_sha256=loaded.lock_sha256,
                        git_commit=git_commit,
                        source_hashes=source_hashes,
                        config_hash=cfg_hash,
                        confirmatory_errors={
                            "max_per_transition_reward_error": confirmatory[
                                "max_per_transition_reward_error"
                            ],
                            "max_discounted_return_error": confirmatory[
                                "max_discounted_return_error"
                            ],
                        },
                    )
                    comfort_path = dirs["artifacts"] / "final_comfort_parameters.yaml"
                    comfort_hash = write_comfort_lock(comfort_path, lock)
                    log(f"wrote comfort lock sha={comfort_hash}")
                else:
                    overall = "FAIL"
            else:
                overall = "FAIL"
                comfort_final = False

    summary = {
        "run_id": run_id,
        "overall": overall,
        "status_note": status_note,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "tests": test_info,
        "source_environment_lock_path": str(loaded.lock_path),
        "source_environment_lock_sha256": loaded.lock_sha256,
        "n_calibration_blocks": len(loaded.calibration_blocks),
        "n_validation_blocks": len(loaded.validation_blocks),
        "counts": None if bundle is None else bundle.counts,
        "integrity": None if bundle is None else bundle.integrity,
        "n_threshold_pairs": None if cal_result is None else cal_result["n_threshold_pairs"],
        "n_complete_tuples": None if cal_result is None else cal_result["n_complete_tuples"],
        "n_feasible_tuples": None if cal_result is None else cal_result["n_feasible"],
        "selected": None
        if selected is None
        else {
            "a_comfort": selected["a_comfort"],
            "a_hard": selected["a_hard"],
            "eta_H": selected["eta_H"],
        },
        "calibration_metrics": None
        if selected is None
        else {
            "median_nominal_share": selected["median_nominal_share"],
            "max_nominal_share": selected["max_nominal_share"],
            "hard_nonzero_rate": selected["hard_nonzero_rate"],
            "mean_H_hard": selected["mean_H_hard"],
            "median_paired_share_diff": selected["median_paired_share_diff"],
            "median_order_gap": selected["median_order_gap"],
            "max_order_gap": selected["max_order_gap"],
            "ordering_violations": selected["ordering_violations"],
        },
        "validation_metrics": None
        if validation is None
        else {
            "pass": validation["pass"],
            "median_nominal_share": validation["median_nominal_share"],
            "max_nominal_share": validation["max_nominal_share"],
            "rejection_reasons": validation["rejection_reasons"],
        },
        "confirmatory": None
        if confirmatory is None
        else {
            "pass": confirmatory["pass"],
            "max_per_transition_reward_error": confirmatory["max_per_transition_reward_error"],
            "max_discounted_return_error": confirmatory["max_discounted_return_error"],
        },
        "comfort_lock_path": str(comfort_path) if comfort_path else None,
        "comfort_lock_sha256": comfort_hash,
        "environment_parameters_final": True,
        "comfort_parameters_final": comfort_final,
        "policy_training_started": False,
        "development_calibration": "FAIL",
    }
    (dirs["reports"] / "stage3b_r1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    report = [
        "# Stage 3B-R1 Report — Final-Environment Joint Comfort Calibration",
        "",
        f"## Overall: **{overall}**",
        "",
        "Development Stage 3B = FAIL (historical). This is a new joint grid on G1-I1.",
        "",
        f"## Git: `{git_commit}` dirty=`{git_dirty}`",
        f"## Environment lock: `{loaded.lock_path}` sha=`{loaded.lock_sha256}`",
        f"## Selected: `{summary['selected']}`",
        f"## Comfort final: `{comfort_final}`",
        f"## Tests: `{test_info}`",
        "",
    ]
    (dirs["reports"] / "stage3b_r1_report.md").write_text("\n".join(report), encoding="utf-8")

    manifest = {
        **summary,
        "utc_timestamp": _utc_now().isoformat(),
        "exact_command": " ".join([sys.executable, str(SCRIPT_PATH), *sys.argv[1:]]),
        "python_and_packages": _versions(),
        "configuration_sha256": cfg_hash,
        "source_hashes": source_hashes,
        "output_paths": {k: str(v) for k, v in dirs.items()},
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "overall": overall,
                "selected": summary["selected"],
                "comfort_parameters_final": comfort_final,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"overall={overall}")
    if overall == "PASS":
        return 0
    if overall == "BLOCKED":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
