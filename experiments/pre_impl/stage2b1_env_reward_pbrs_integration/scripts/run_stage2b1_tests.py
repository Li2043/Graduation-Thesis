#!/usr/bin/env python3
"""Stage 2B-1 environment + reward/PBRS integration runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[4]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(git_sha: str) -> str:
    return f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{(git_sha or 'nogit')[:8]}"


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_unique(run_id: str) -> dict[str, Path]:
    dirs = {
        k: EXP_ROOT / k / run_id
        for k in ("data/raw", "data/processed", "reports", "logs", "artifacts")
    }
    # fix keys
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _pkg_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, modname in (
        ("numpy", "numpy"),
        ("gymnasium", "gymnasium"),
        ("highway-env", "highway_env"),
        ("pytest", "pytest"),
        ("PyYAML", "yaml"),
    ):
        try:
            mod = __import__(modname)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            out[name] = "not_installed"
    return out


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


def _run_pytest(paths: list[Path], extra: list[str], log_path: Path, env: dict[str, str]):
    cmd = [sys.executable, "-m", "pytest", *[str(p) for p in paths], *extra]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(text, encoding="utf-8")
    return proc.returncode, text


def _collect_scenario_traces(run_id: str, seed: int) -> dict[str, Any]:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from thesis.envs.scripted_scenarios import build_scenarios, run_scenario

    transitions: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    exits: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []

    nan_count = 0
    disc_count = 0
    repeated_exit_count = 0
    invalid_flag_count = 0

    def finite_ok(v: Any) -> bool:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return math.isfinite(float(v))
        return True

    for sid, spec in build_scenarios().items():
        env, records = run_scenario(spec)
        exit_counts = {"A": 0, "B": 0}
        term_reason = "ongoing"
        for rec in records:
            info = rec["info"]
            if rec["terminated"] and rec["truncated"]:
                invalid_flag_count += 1
            for w in info["events"].get("warnings", []):
                if "route_discontinuity" in w:
                    disc_count += 1
            for aid in ("A", "B"):
                if info["events"]["exit_event"][aid] >= 1.0:
                    exit_counts[aid] += 1
                    exits.append(
                        {
                            "run_id": run_id,
                            "scenario_id": sid,
                            "step": info["step"],
                            "controller": aid,
                            "exit_event": 1.0,
                        }
                    )
            if info["events"]["stakeholder_collision_event"] >= 1.0:
                collisions.append(
                    {
                        "run_id": run_id,
                        "scenario_id": sid,
                        "step": info["step"],
                        "stakeholder_collided": info["events"]["stakeholder_collided"],
                        "collision_pairs": info["events"]["collision_pairs"],
                    }
                )
            d = info["diagnostics"]
            # NaN scan
            for veh in info["vehicles_t1"].values():
                for k, val in veh.items():
                    if not finite_ok(val):
                        nan_count += 1
            for aid, p in d["per_agent"].items():
                for k, val in p.items():
                    if not finite_ok(val):
                        nan_count += 1
                reward_rows.append(
                    {
                        "run_id": run_id,
                        "scenario_id": sid,
                        "step": info["step"],
                        "controller": aid,
                        **{k: p[k] for k in p},
                        "raw_mean_t1": d["raw_mean_potential_t1"],
                        "actual_mean_t1": d["actual_mean_potential_t1"],
                        "raw_min_t1": d["raw_min_potential_t1"],
                        "actual_min_t1": d["actual_min_potential_t1"],
                    }
                )
            transitions.append(
                {
                    "run_id": run_id,
                    "scenario_id": sid,
                    "seed": info.get("seed", seed),
                    "step": info["step"],
                    "vehicle_identities": list(info["vehicles_t1"].keys()),
                    "traffic_roles": {
                        k: v["role"] for k, v in info["vehicles_t1"].items()
                    },
                    "world_positions": {
                        k: {"x": v["world_x"], "y": v["world_y"]}
                        for k, v in info["vehicles_t1"].items()
                    },
                    "route_positions": {
                        k: v["route_position"] for k, v in info["vehicles_t1"].items()
                    },
                    "rho_values": {k: v["rho"] for k, v in info["vehicles_t1"].items()},
                    "delta_rho": {
                        aid: d["per_agent"][aid]["delta_rho"] for aid in ("A", "B")
                    },
                    "realised_acceleration": {
                        k: v["acceleration"] for k, v in info["vehicles_t1"].items()
                    },
                    "completion_flags": info["completion"],
                    "exit_events": info["events"]["exit_event"],
                    "collision_registry": info["events"]["stakeholder_collided"],
                    "collision_pairs": info["events"]["collision_pairs"],
                    "terminated": rec["terminated"],
                    "truncated": rec["truncated"],
                    "base_reward_decomposition": {
                        aid: {
                            "progress": d["per_agent"][aid]["progress_component"],
                            "exit": d["per_agent"][aid]["exit_component"],
                            "collision": d["per_agent"][aid]["collision_component"],
                            "hard_braking": d["per_agent"][aid]["hard_braking_component"],
                            "total": d["per_agent"][aid]["base_total"],
                        }
                        for aid in ("A", "B")
                    },
                    "mean_potential": {
                        "experiences_t": d["stakeholder_experiences_t"],
                        "experiences_t1": d["stakeholder_experiences_t1"],
                        "raw_t": d["raw_mean_potential_t"],
                        "actual_t": d["actual_mean_potential_t"],
                        "raw_t1": d["raw_mean_potential_t1"],
                        "actual_t1": d["actual_mean_potential_t1"],
                        "F_t": d["mean_F_t"],
                    },
                    "min_potential": {
                        "raw_t": d["raw_min_potential_t"],
                        "actual_t": d["actual_min_potential_t"],
                        "raw_t1": d["raw_min_potential_t1"],
                        "actual_t1": d["actual_min_potential_t1"],
                        "F_t": d["min_F_t"],
                    },
                    "rewards": {
                        "baseline": d["baseline_reward"],
                        "mean_pbrs": d["mean_pbrs_reward"],
                        "min_pbrs": d["min_pbrs_reward"],
                    },
                    "term_reason": info.get("term_reason"),
                }
            )
            term_reason = info.get("term_reason", term_reason)
        for aid, c in exit_counts.items():
            if c > 1:
                repeated_exit_count += c - 1
        scenario_rows.append(
            {
                "scenario_id": sid,
                "n_transitions": len(records),
                "exit_count_A": exit_counts["A"],
                "exit_count_B": exit_counts["B"],
                "final_terminated": records[-1]["terminated"] if records else None,
                "final_truncated": records[-1]["truncated"] if records else None,
                "term_reason": term_reason,
            }
        )

    return {
        "transitions": transitions,
        "collisions": collisions,
        "exits": exits,
        "scenario_rows": scenario_rows,
        "reward_rows": reward_rows,
        "nan_count": nan_count,
        "route_discontinuity_count": disc_count,
        "repeated_exit_count": repeated_exit_count,
        "invalid_flag_count": invalid_flag_count,
        "scenario_count": len(build_scenarios()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "env_integration_test.yaml",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()

    git_commit = _git(["rev-parse", "HEAD"]) or "unknown"
    git_dirty = bool(_git(["status", "--porcelain"]))
    run_id = _run_id(git_commit)
    dirs = _ensure_unique(run_id)
    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{_utc_now().isoformat()}] {msg}"
        logs.append(line)
        print(msg)

    log(f"run_id={run_id}")
    log(f"python={sys.version}")
    log(f"git_dirty={git_dirty}")
    if git_dirty:
        log("WARNING: git working tree is DIRTY")

    cfg = _load_yaml(config_path)
    resolved = dirs["artifacts"] / "resolved_config.yaml"
    _write_yaml(resolved, cfg)
    config_hash = _sha256_file(resolved)
    seed = int(cfg.get("seed", 0))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    test_paths = [REPO_ROOT / p for p in cfg.get("pytest", {}).get("test_paths", [])]
    extra = list(cfg.get("pytest", {}).get("extra_args", ["-q", "--tb=short"]))
    log(f"running pytest on {test_paths}")
    rc, text = _run_pytest(test_paths, extra, dirs["logs"] / "pytest.log", env)
    counts = _parse_pytest(text)
    log(f"pytest rc={rc} counts={counts}")

    data = _collect_scenario_traces(run_id, seed)
    _write_jsonl(dirs["raw"] / "transition_trace.jsonl", data["transitions"])
    _write_jsonl(dirs["raw"] / "collision_events.jsonl", data["collisions"])
    _write_jsonl(dirs["raw"] / "exit_events.jsonl", data["exits"])
    _write_csv(dirs["processed"] / "scenario_summary.csv", data["scenario_rows"])
    _write_csv(dirs["processed"] / "reward_potential_summary.csv", data["reward_rows"])

    unit_ok = rc == 0 and counts["status"] == "PASS"
    metrics_ok = (
        data["nan_count"] == 0
        and data["repeated_exit_count"] == 0
        and data["invalid_flag_count"] == 0
    )
    # Prior stage integrity
    s1 = (
        REPO_ROOT
        / "experiments/pre_impl/stage1_base_reward_unit_tests/artifacts/20260729T163419Z_e55e4170/manifest.json"
    )
    s2a = list(
        (REPO_ROOT / "experiments/pre_impl/stage2a_pbrs_math_tests/artifacts").glob(
            "*/manifest.json"
        )
    )
    prior_ok = s1.exists() and bool(s2a)
    overall = "PASS" if unit_ok and metrics_ok and prior_ok else "FAIL"
    recommendation = (
        "PROCEED TO STAGE 2B-2 (DQN/replay wiring) — DQN still unverified"
        if overall == "PASS"
        else "DO NOT PROCEED"
    )

    summary = {
        "overall": overall,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "package_versions": _pkg_versions(),
        "unit_integration_tests": counts,
        "pytest_returncode": rc,
        "scenario_count": data["scenario_count"],
        "nan_count": data["nan_count"],
        "route_discontinuity_count": data["route_discontinuity_count"],
        "repeated_exit_count": data["repeated_exit_count"],
        "invalid_flag_count": data["invalid_flag_count"],
        "prior_stage_outputs_intact": prior_ok,
        "recommendation": recommendation,
        "dqn_verified": False,
        "replay_buffer_verified": False,
        "test_only_braking": cfg.get("reward_test_only"),
        "test_only_lambdas": cfg.get("pbrs_test_only"),
    }
    (dirs["reports"] / "stage2b1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = f"""# Stage 2B-1 Report — Env / Reward / PBRS Integration

## 1. Overall: **{overall}**

{" > **WARNING: git_dirty = true**. Dissertation-retained runs should be clean." if git_dirty else ""}

- Git commit: `{git_commit}`
- Git dirty: `{git_dirty}`
- Python: `{sys.version.split()[0]}`
- Config SHA-256: `{config_hash}`
- Packages: `{json.dumps(_pkg_versions())}`

## 2. Tests

| Metric | Value |
|--------|-------|
| Passed | {counts['passed']} |
| Failed | {counts['failed']} |
| Errors | {counts['errors']} |
| Status | {counts['status']} |

## 3. Scenario metrics

| Metric | Value |
|--------|-------|
| Scenarios | {data['scenario_count']} |
| NaN count | {data['nan_count']} |
| Route discontinuity count | {data['route_discontinuity_count']} |
| Repeated exit count | {data['repeated_exit_count']} |
| Invalid term/trunc flags | {data['invalid_flag_count']} |

## 4. Unresolved limitations

- Dynamics are thesis-owned kinematics (not a live highway-env wrap); highway-env is installed for reproducibility.
- Geometry is integration-test configuration, not final dissertation geometry.
- DQN and replay-buffer integration remain **UNVERIFIED**.
- Lambda / comfort parameters remain **TEST-ONLY**.

## 5. Recommendation

**{recommendation}**
"""
    (dirs["reports"] / "stage2b1_report.md").write_text(report, encoding="utf-8")

    outputs = {
        "transition_trace_jsonl": str(dirs["raw"] / "transition_trace.jsonl"),
        "collision_events_jsonl": str(dirs["raw"] / "collision_events.jsonl"),
        "exit_events_jsonl": str(dirs["raw"] / "exit_events.jsonl"),
        "scenario_summary_csv": str(dirs["processed"] / "scenario_summary.csv"),
        "reward_potential_summary_csv": str(
            dirs["processed"] / "reward_potential_summary.csv"
        ),
        "stage2b1_report_md": str(dirs["reports"] / "stage2b1_report.md"),
        "stage2b1_summary_json": str(dirs["reports"] / "stage2b1_summary.json"),
        "pytest_log": str(dirs["logs"] / "pytest.log"),
        "runner_log": str(dirs["logs"] / "runner.log"),
        "resolved_config_yaml": str(resolved),
        "manifest_json": str(dirs["artifacts"] / "manifest.json"),
    }
    manifest = {
        "run_id": run_id,
        "utc_timestamp": _utc_now().isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "operating_system": f"{platform.system()} {platform.release()}",
        "package_versions": _pkg_versions(),
        "random_seeds": [seed],
        "command": " ".join([sys.executable, str(SCRIPT_PATH), "--config", str(config_path)]),
        "resolved_configuration_path": str(resolved),
        "configuration_sha256": config_hash,
        "road_geometry": cfg.get("env", {}).get("geometry_note"),
        "vehicle_parameters": {
            k: cfg.get("env", {}).get(k)
            for k in (
                "dt",
                "accel_rate",
                "decel_rate",
                "collision_distance",
                "target_speed",
                "role_A",
                "role_B",
            )
        },
        "idm_parameters": cfg.get("env", {}).get("idm"),
        "action_definitions": cfg.get("env", {}).get("actions"),
        "maximum_simulation_steps": cfg.get("env", {}).get("max_steps"),
        "test_only_braking_values": cfg.get("reward_test_only"),
        "test_only_lambda_values": cfg.get("pbrs_test_only"),
        "test_counts": counts,
        "scenario_counts": data["scenario_count"],
        "nan_count": data["nan_count"],
        "route_discontinuity_count": data["route_discontinuity_count"],
        "repeated_exit_count": data["repeated_exit_count"],
        "invalid_flag_count": data["invalid_flag_count"],
        "outputs": outputs,
        "pass_fail_status": overall,
        "dqn_verified": False,
        "replay_buffer_verified": False,
        "note": "Stage 1 / 2A outputs not modified. No training in this stage.",
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (dirs["logs"] / "runner.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "overall": overall,
                "git_dirty": git_dirty,
                "reports": outputs["stage2b1_report_md"],
                "manifest": outputs["manifest_json"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"overall={overall}")
    log(f"nan_count={data['nan_count']} disc={data['route_discontinuity_count']} "
        f"repeat_exit={data['repeated_exit_count']} invalid_flags={data['invalid_flag_count']}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
