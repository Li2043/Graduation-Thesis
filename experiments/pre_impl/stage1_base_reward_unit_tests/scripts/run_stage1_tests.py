#!/usr/bin/env python3
"""Stage-1 base reward unit-test experiment runner.

Creates a unique run directory, executes pytest, attempts an environment
integration smoke test (or records BLOCKED), and writes raw/processed/report
artifacts. Never overwrites an existing run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]  # stage1_base_reward_unit_tests/
REPO_ROOT = SCRIPT_PATH.parents[4]  # final_new/


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(git_sha: str) -> str:
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    short = (git_sha or "nogit")[:8]
    return f"{stamp}_{short}"


def _git(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(
            ["git", *cmd],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_unique_run_dirs(run_id: str) -> dict[str, Path]:
    dirs = {
        "raw": EXP_ROOT / "data" / "raw" / run_id,
        "processed": EXP_ROOT / "data" / "processed" / run_id,
        "reports": EXP_ROOT / "reports" / run_id,
        "logs": EXP_ROOT / "logs" / run_id,
        "artifacts": EXP_ROOT / "artifacts" / run_id,
    }
    for p in dirs.values():
        if p.exists():
            raise RuntimeError(f"run directory already exists (refusing overwrite): {p}")
        p.mkdir(parents=True, exist_ok=False)
    return dirs


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required. Install with: pip install -r requirements-stage1.txt"
        ) from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _pkg_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("pytest", "yaml", "numpy", "gymnasium", "highway_env"):
        try:
            if name == "yaml":
                import yaml as mod

                versions["PyYAML"] = getattr(mod, "__version__", "unknown")
            else:
                mod = __import__(name)
                versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name if name != "yaml" else "PyYAML"] = "not_installed"
    return versions


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _run_pytest(
    test_path: Path,
    extra_args: list[str],
    log_path: Path,
    env: dict[str, str],
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_path),
        *extra_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(text, encoding="utf-8")
    return proc.returncode, text


def _parse_pytest_counts(pytest_log: str) -> dict[str, Any]:
    """Best-effort parse of pytest summary line."""
    import re

    passed = failed = errors = skipped = 0
    m = re.search(
        r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error|(\d+)\s+skipped",
        pytest_log,
    )
    # Collect all
    for label, var in (
        ("passed", "passed"),
        ("failed", "failed"),
        ("error", "errors"),
        ("skipped", "skipped"),
    ):
        mm = re.findall(rf"(\d+)\s+{label}", pytest_log)
        if mm:
            val = int(mm[-1])
            if label == "passed":
                passed = val
            elif label == "failed":
                failed = val
            elif label == "error":
                errors = val
            else:
                skipped = val
    status = "PASS" if failed == 0 and errors == 0 and passed > 0 else "FAIL"
    if passed == 0 and failed == 0 and errors == 0:
        status = "FAIL"
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "status": status,
    }


def _build_unit_case_records() -> list[dict[str, Any]]:
    """Execute representative reward cases and record decompositions for JSONL/CSV."""
    # Ensure src is importable
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from thesis.rewards.base_reward_v2 import (
        AgentTransitionState,
        BaseRewardConfig,
        BaseRewardInputs,
        compute_base_reward_for_agents,
    )

    cfg = BaseRewardConfig(
        progress_weight=0.4,
        exit_weight=0.6,
        collision_penalty=1.0,
        eta_hard_brake=0.1,
        a_comfort=2.0,
        a_hard=6.0,
    )

    def agent(**kw: Any) -> AgentTransitionState:
        defaults = dict(
            route_position_t=0.0,
            route_position_t1=0.0,
            route_start=0.0,
            route_exit=100.0,
            acceleration=0.0,
            already_exited=False,
        )
        defaults.update(kw)
        return AgentTransitionState(**defaults)

    def no_coll() -> dict[str, bool]:
        return {"A": False, "B": False, "B_front": False, "B_rear": False}

    cases: list[tuple[str, BaseRewardInputs]] = [
        (
            "T01_stationary",
            BaseRewardInputs(
                agents={"A": agent(), "B": agent()},
                stakeholder_collided=no_coll(),
            ),
        ),
        (
            "T02_positive_progress",
            BaseRewardInputs(
                agents={
                    "A": agent(route_position_t=0.0, route_position_t1=10.0),
                    "B": agent(route_position_t=0.0, route_position_t1=10.0),
                },
                stakeholder_collided=no_coll(),
            ),
        ),
        (
            "T04_safe_exit_A",
            BaseRewardInputs(
                agents={
                    "A": agent(route_position_t=99.0, route_position_t1=101.0),
                    "B": agent(route_position_t=50.0, route_position_t1=51.0),
                },
                stakeholder_collided=no_coll(),
            ),
        ),
        (
            "T08_collision_blocks_exit",
            BaseRewardInputs(
                agents={
                    "A": agent(route_position_t=99.0, route_position_t1=101.0),
                    "B": agent(),
                },
                stakeholder_collided={
                    "A": False,
                    "B": False,
                    "B_front": True,
                    "B_rear": False,
                },
            ),
        ),
        (
            "T10_intermediate_braking",
            BaseRewardInputs(
                agents={
                    "A": agent(acceleration=-4.0),
                    "B": agent(),
                },
                stakeholder_collided=no_coll(),
            ),
        ),
        (
            "T14_negative_progress",
            BaseRewardInputs(
                agents={
                    "A": agent(route_position_t=50.0, route_position_t1=45.0),
                    "B": agent(),
                },
                stakeholder_collided=no_coll(),
            ),
        ),
        (
            "T16_decomposition_combo",
            BaseRewardInputs(
                agents={
                    "A": agent(
                        route_position_t=0.0,
                        route_position_t1=10.0,
                        acceleration=-4.0,
                    ),
                    "B": agent(
                        route_position_t=0.0,
                        route_position_t1=20.0,
                        acceleration=-8.0,
                    ),
                },
                stakeholder_collided={
                    "A": True,
                    "B": False,
                    "B_front": False,
                    "B_rear": False,
                },
            ),
        ),
    ]

    records: list[dict[str, Any]] = []
    for case_id, inputs in cases:
        out = compute_base_reward_for_agents(inputs, cfg)
        for aid, br in out.items():
            records.append(
                {
                    "case_id": case_id,
                    "controller": aid,
                    "progress_component": br.progress_component,
                    "exit_component": br.exit_component,
                    "collision_component": br.collision_component,
                    "hard_braking_component": br.hard_braking_component,
                    "total_reward": br.total_reward,
                    "delta_route_progress": br.delta_route_progress,
                    "safe_exit_event": br.safe_exit_event,
                    "stakeholder_collision_event": br.stakeholder_collision_event,
                    "hard_braking_cost": br.hard_braking_cost,
                    "rho_t": br.rho_t,
                    "rho_t1": br.rho_t1,
                    "warnings": list(br.warnings),
                    "decomposition_ok": abs(
                        br.total_reward
                        - (
                            br.progress_component
                            + br.exit_component
                            + br.collision_component
                            + br.hard_braking_component
                        )
                    )
                    < 1e-12,
                }
            )
    return records


def _run_integration_smoke(
    seed: int,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Attempt env smoke test. Returns (status, blocker_reason, transitions)."""
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    # Exact blocker for this cleared final_new repository.
    candidates = [
        "thesis.envs.long_repeated_merge",
        "thesis.envs",
        "thesis.environment",
    ]
    missing: list[str] = []
    for mod in candidates:
        try:
            __import__(mod)
        except ModuleNotFoundError as e:
            missing.append(f"{mod} ({e})")
        except Exception as e:
            missing.append(f"{mod} (import error: {e})")

    # Also check for common env class names under src/thesis
    env_dir = REPO_ROOT / "src" / "thesis" / "envs"
    if not env_dir.exists():
        reason = (
            "BLOCKED: environment integration smoke cannot run because "
            f"`{env_dir.as_posix()}` does not exist in this repository. "
            "Tried imports: "
            + "; ".join(missing)
            + ". No LongRepeatedMergeEnv (or equivalent) is available; "
            "refusing to fabricate an adapter. Pure unit tests remain authoritative."
        )
        return "BLOCKED", reason, []

    reason = (
        "BLOCKED: src/thesis/envs exists but required merge environment module "
        "could not be imported: " + "; ".join(missing)
    )
    return "BLOCKED", reason, []


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            flat = {
                k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                for k, v in row.items()
            }
            w.writerow(flat)


def _write_report(
    path: Path,
    *,
    overall: str,
    git_commit: str,
    config_hash: str,
    unit_counts: dict[str, Any],
    integration_status: str,
    integration_reason: str,
    case_records: list[dict[str, Any]],
    recommendation: str,
    unresolved: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Stage 1 Report — Base Reward Unit Tests")
    lines.append("")
    lines.append(f"## 1. Overall: **{overall}**")
    lines.append("")
    lines.append(f"- Git commit: `{git_commit}`")
    lines.append(f"- Configuration SHA-256: `{config_hash}`")
    lines.append("")
    lines.append("## 2. Unit-test summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Passed | {unit_counts.get('passed', 0)} |")
    lines.append(f"| Failed | {unit_counts.get('failed', 0)} |")
    lines.append(f"| Errors | {unit_counts.get('errors', 0)} |")
    lines.append(f"| Skipped | {unit_counts.get('skipped', 0)} |")
    lines.append(f"| Status | {unit_counts.get('status', 'UNKNOWN')} |")
    lines.append("")
    lines.append("Required coverage: Tests 1–18 in `tests/rewards/test_base_reward_v2.py`.")
    lines.append("")
    lines.append("## 3. Integration smoke-test result")
    lines.append("")
    lines.append(f"**Status:** `{integration_status}`")
    lines.append("")
    lines.append(integration_reason)
    lines.append("")
    if integration_status == "BLOCKED":
        lines.append(
            "> A blocked smoke test does **not** validate environment integration. "
            "Only pure unit tests have been validated."
        )
        lines.append("")
    lines.append("## 4. Reward decomposition examples")
    lines.append("")
    lines.append(
        "| case_id | controller | progress | exit | collision | hard_brake | total |"
    )
    lines.append(
        "|---------|------------|----------|------|------------|------------|-------|"
    )
    for r in case_records:
        lines.append(
            f"| {r['case_id']} | {r['controller']} | "
            f"{r['progress_component']:.6f} | {r['exit_component']:.6f} | "
            f"{r['collision_component']:.6f} | {r['hard_braking_component']:.6f} | "
            f"{r['total_reward']:.6f} |"
        )
    lines.append("")
    lines.append("## 5. Route-coordinate discontinuity")
    lines.append("")
    disc = [r for r in case_records if r.get("warnings")]
    if not disc:
        lines.append("None detected in recorded unit-case examples.")
    else:
        for r in disc:
            lines.append(f"- {r['case_id']} / {r['controller']}: {r['warnings']}")
    lines.append("")
    lines.append("## 6. Repeated exit bonus")
    lines.append("")
    lines.append(
        "Unit Test 5 asserts exit bonus cannot repeat when `already_exited=True`. "
        "No integration trajectory was available to scan for repeated exits."
    )
    lines.append("")
    lines.append("## 7. NaN / invalid-state events")
    lines.append("")
    lines.append(
        "Unit Test 17 asserts NaN/infinity raise clear errors. "
        "No NaN accepted by the reward module under those cases."
    )
    lines.append("")
    lines.append("## 8. Unresolved implementation issues")
    lines.append("")
    if unresolved:
        for u in unresolved:
            lines.append(f"- {u}")
    else:
        lines.append("- None for pure unit-test scope.")
    lines.append("")
    lines.append("## 9. Recommendation")
    lines.append("")
    lines.append(f"**{recommendation}**")
    lines.append("")
    lines.append(
        "Note: braking `a_comfort`, `a_hard`, and `eta_hard_brake` remain "
        "**TEST-ONLY placeholders** and must not be treated as final experiment values."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-1 base reward unit tests runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "reward_unit_tests.yaml",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()

    git_commit = _git(["rev-parse", "HEAD"]) or "unknown"
    git_dirty = bool(_git(["status", "--porcelain"]))
    run_id = _run_id(git_commit)
    dirs = _ensure_unique_run_dirs(run_id)

    runner_log = dirs["logs"] / "runner.log"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{_utc_now().isoformat()}] {msg}"
        log_lines.append(line)
        print(msg)

    log(f"run_id={run_id}")
    log(f"repo_root={REPO_ROOT}")
    log(f"config={config_path}")

    cfg = _load_yaml(config_path)
    resolved_path = dirs["artifacts"] / "resolved_config.yaml"
    _write_yaml(resolved_path, cfg)
    config_hash = _sha256_file(resolved_path)

    command = " ".join(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)]
    )

    # --- pytest ---
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    test_rel = cfg.get("pytest", {}).get(
        "test_path", "tests/rewards/test_base_reward_v2.py"
    )
    test_path = REPO_ROOT / test_rel
    extra = list(cfg.get("pytest", {}).get("extra_args", ["-q", "--tb=short"]))
    log(f"running pytest: {test_path}")
    pytest_rc, pytest_text = _run_pytest(
        test_path, extra, dirs["logs"] / "pytest.log", env
    )
    unit_counts = _parse_pytest_counts(pytest_text)
    log(f"pytest returncode={pytest_rc} counts={unit_counts}")

    # --- case records ---
    case_records = _build_unit_case_records()
    _append_jsonl(dirs["raw"] / "unit_test_cases.jsonl", case_records)

    summary_rows = []
    for r in case_records:
        summary_rows.append(
            {
                "case_id": r["case_id"],
                "controller": r["controller"],
                "total_reward": r["total_reward"],
                "decomposition_ok": r["decomposition_ok"],
            }
        )
    _write_csv(dirs["processed"] / "unit_test_summary.csv", summary_rows)
    _write_csv(dirs["processed"] / "reward_component_summary.csv", case_records)

    # --- integration ---
    seed = int(cfg.get("seed", 0))
    integ_status, integ_reason, transitions = _run_integration_smoke(seed)
    log(f"integration_status={integ_status}")
    log(integ_reason)
    integ_path = dirs["raw"] / "integration_transitions.jsonl"
    if transitions:
        _append_jsonl(integ_path, transitions)
    else:
        integ_path.write_text("", encoding="utf-8")

    unit_ok = pytest_rc == 0 and unit_counts.get("status") == "PASS"
    overall = "PASS" if unit_ok else "FAIL"
    unresolved: list[str] = []
    if integ_status == "BLOCKED":
        unresolved.append(
            "Environment integration smoke is BLOCKED; Stage 1 pure unit tests "
            "do not validate LongRepeatedMergeEnv / simulator wiring."
        )
    if unit_ok and integ_status in {"PASS", "BLOCKED"}:
        # Spec: blocked smoke does not fail Stage 1 if unit tests pass.
        recommendation = "PROCEED TO STAGE 2"
        if integ_status == "BLOCKED":
            recommendation = (
                "PROCEED TO STAGE 2 (with caveat: environment smoke BLOCKED; "
                "wire simulator before claiming env-integrated reward validation)"
            )
    else:
        recommendation = "DO NOT PROCEED"

    # --- summary json ---
    summary = {
        "overall": overall,
        "run_id": run_id,
        "git_commit": git_commit,
        "config_sha256": config_hash,
        "unit_tests": unit_counts,
        "pytest_returncode": pytest_rc,
        "integration_smoke": {
            "status": integ_status,
            "reason": integ_reason,
            "n_transitions": len(transitions),
        },
        "recommendation": recommendation,
        "test_only_braking_params": {
            "a_comfort": 2.0,
            "a_hard": 6.0,
            "eta_hard_brake": 0.1,
            "note": "TEST-ONLY placeholders; not final experimental values",
        },
    }
    (dirs["reports"] / "stage1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _write_report(
        dirs["reports"] / "stage1_report.md",
        overall=overall,
        git_commit=git_commit,
        config_hash=config_hash,
        unit_counts=unit_counts,
        integration_status=integ_status,
        integration_reason=integ_reason,
        case_records=case_records,
        recommendation=recommendation,
        unresolved=unresolved,
    )

    # --- manifest ---
    outputs = {
        "unit_test_cases_jsonl": str(dirs["raw"] / "unit_test_cases.jsonl"),
        "integration_transitions_jsonl": str(
            dirs["raw"] / "integration_transitions.jsonl"
        ),
        "unit_test_summary_csv": str(dirs["processed"] / "unit_test_summary.csv"),
        "reward_component_summary_csv": str(
            dirs["processed"] / "reward_component_summary.csv"
        ),
        "stage1_report_md": str(dirs["reports"] / "stage1_report.md"),
        "stage1_summary_json": str(dirs["reports"] / "stage1_summary.json"),
        "pytest_log": str(dirs["logs"] / "pytest.log"),
        "runner_log": str(dirs["logs"] / "runner.log"),
        "resolved_config_yaml": str(resolved_path),
        "manifest_json": str(dirs["artifacts"] / "manifest.json"),
    }
    manifest = {
        "run_id": run_id,
        "utc_timestamp": _utc_now().isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "operating_system": f"{platform.system()} {platform.release()} ({platform.version()})",
        "package_versions": _pkg_versions(),
        "random_seed": seed,
        "command": command,
        "resolved_configuration_path": str(resolved_path),
        "configuration_sha256": config_hash,
        "test_counts": unit_counts,
        "pass_fail_status": overall,
        "integration_smoke_test_status": integ_status,
        "integration_smoke_blocker": integ_reason if integ_status == "BLOCKED" else None,
        "outputs": outputs,
        "note": "Large environment dumps are intentionally omitted from this manifest.",
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    runner_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # latest_run pointer (does not replace historical run dirs)
    latest = {
        "run_id": run_id,
        "utc_timestamp": manifest["utc_timestamp"],
        "overall": overall,
        "reports": str(dirs["reports"] / "stage1_report.md"),
        "manifest": str(dirs["artifacts"] / "manifest.json"),
    }
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps(latest, indent=2), encoding="utf-8"
    )

    log(f"overall={overall} recommendation={recommendation}")
    return 0 if unit_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
