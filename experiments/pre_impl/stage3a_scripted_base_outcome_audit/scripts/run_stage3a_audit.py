#!/usr/bin/env python3
"""Stage 3A scripted base-outcome audit runner (no DQN training)."""

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
import traceback
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
    for name, mod in (
        ("numpy", "numpy"),
        ("gymnasium", "gymnasium"),
        ("pytest", "pytest"),
        ("torch", "torch"),
    ):
        try:
            m = __import__(mod)
            out[name] = getattr(m, "__version__", "installed")
        except Exception:
            out[name] = "not_installed"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "base_outcome_audit.yaml",
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
    # Embed matched blocks into resolved config
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from thesis.audits.audit_scenarios import build_matched_blocks
    from thesis.audits.base_outcome_audit import run_full_audit

    cfg["matched_blocks"] = [b.to_dict() for b in build_matched_blocks()]
    resolved = dirs["artifacts"] / "resolved_config.yaml"
    _write_yaml(resolved, cfg)
    config_hash = _sha256(resolved)
    gamma = float(cfg.get("gamma", 0.995))

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

    log("running full scripted audit (no DQN training)")
    audit = run_full_audit(run_id=run_id, gamma=gamma)
    outcomes = audit["outcomes"]
    metrics = audit["metrics"]

    # Raw outputs
    transitions = []
    for o in outcomes:
        transitions.extend(o.transitions)
    _jsonl(dirs["raw"] / "transition_trace.jsonl", transitions)

    scenario_outcomes = []
    for o in outcomes:
        scenario_outcomes.append(
            {
                "block_id": o.block_id,
                "scenario_id": o.scenario_id,
                "fixture_only": o.fixture_only,
                "primary_ranking": o.primary_ranking,
                "terminated": o.terminated,
                "truncated": o.truncated,
                "term_reason": o.term_reason,
                "episode_length": o.episode_length,
                "exit_count_A": o.exit_count_A,
                "exit_count_B": o.exit_count_B,
                "exit_time_A": o.exit_time_A,
                "exit_time_B": o.exit_time_B,
                "exit_order": o.exit_order,
                "collision": o.collision,
                "collision_pairs": o.collision_pairs,
                "G_A": o.G_A,
                "G_B": o.G_B,
                "G_team": o.G_team,
                "G_A_undiscounted": o.G_A_undiscounted,
                "G_B_undiscounted": o.G_B_undiscounted,
                "G_progress": o.G_progress,
                "G_exit": o.G_exit,
                "G_collision": o.G_collision,
                "G_hard_braking": o.G_hard_braking,
                "mean_speed": o.mean_speed,
                "min_speed": o.min_speed,
                "max_brake_magnitude": o.max_brake_magnitude,
                "cumulative_H": o.cumulative_H,
                "hard_brake_events": o.hard_brake_events,
                "min_gap": o.min_gap,
                "blocked_reason": o.blocked_reason,
                "fixture_injection_used": o.fixture_only,
            }
        )
    _jsonl(dirs["raw"] / "scenario_outcomes.jsonl", scenario_outcomes)
    _jsonl(dirs["raw"] / "matched_order_pairs.jsonl", audit["order_rows"])
    _jsonl(dirs["raw"] / "oscillation_cycles.jsonl", audit["osc_rows"])
    _jsonl(
        dirs["raw"] / "collision_audit.jsonl",
        [r for r in scenario_outcomes if r["collision"] or "collision" in r["scenario_id"]],
    )

    _csv(dirs["processed"] / "scenario_summary.csv", scenario_outcomes)
    _csv(dirs["processed"] / "incentive_ordering.csv", audit["incentive_rows"])
    _csv(dirs["processed"] / "order_bias_summary.csv", audit["order_rows"])
    _csv(dirs["processed"] / "comfort_summary.csv", audit["comfort_rows"])
    _csv(dirs["processed"] / "oscillation_summary.csv", audit["osc_rows"])
    _csv(dirs["processed"] / "identity_invariance_summary.csv", audit["identity_rows"])

    # Prior stages intact
    prior_ok = all(
        any(
            (
                REPO_ROOT
                / "experiments/pre_impl"
                / name
                / "artifacts"
            ).glob("*/manifest.json")
        )
        for name in (
            "stage1_base_reward_unit_tests",
            "stage2a_pbrs_math_tests",
            "stage2b1_env_reward_pbrs_integration",
            "stage2b2_dqn_replay_bootstrap",
        )
    )

    unit_ok = proc.returncode == 0 and counts["status"] == "PASS"
    overall = audit["overall"]
    if not unit_ok or not prior_ok:
        overall = "FAIL"

    redesign = bool(metrics.get("base_reward_requires_redesign"))
    recommendation = {
        "PASS": "PROCEED TO comfort calibration / Stage 3B — base reward structurally acceptable",
        "FAIL": "DO NOT PROCEED — resolve incentive / integrity failures first",
        "BLOCKED": "BLOCKED — physical collision ranking unavailable; do not treat fixture collisions as behavioural proof",
    }[overall if overall in {"PASS", "FAIL", "BLOCKED"} else "FAIL"]
    if redesign:
        recommendation += (
            "; progress reward may require redesign (oscillation discounted return)"
        )

    summary = {
        "overall": overall,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "unit_tests": counts,
        "metrics": metrics,
        "recommendation": recommendation,
        "base_reward_requires_redesign": redesign,
        "policy_training_started": False,
        "scope": "scripted audit; no DQN training",
    }
    (dirs["reports"] / "stage3a_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    failed_blocks = audit["ordering_violations"]
    report = f"""# Stage 3A Report — Scripted Base-Outcome Incentive Audit

## 1. Overall: **{overall}**

## 2. Scope

**Scripted audit only. No policy training. No DQN parameter updates.**

PBRS diagnostics are not used for PASS/FAIL.

{" > **WARNING: git_dirty = true**" if git_dirty else "- `git_dirty = false`"}

- Git commit: `{git_commit}`
- Config SHA-256: `{config_hash}`
- Gamma: `{gamma}`
- Frozen weights: progress=0.4, exit=0.6, collision=1.0
- TEST-ONLY comfort: a_comfort=2.0, a_hard=6.0, eta=0.1

## 3. Tests

| Metric | Value |
|--------|-------|
| Passed | {counts['passed']} |
| Failed | {counts['failed']} |
| Status | {counts['status']} |

## 4. Matched blocks

Completed blocks: **{metrics['n_blocks']}**

## 5. Safe / stall / collision ranking

Incentive-ordering violations: **{metrics['n_incentive_ordering_violations']}**

Failed blocks:
{json.dumps(failed_blocks, indent=2)}

## 6. Order bias (mainline-first vs ramp-first)

| Metric | Value |
|--------|-------|
| Median normalised gap | {metrics['median_normalised_order_gap']} |
| Maximum normalised gap | {metrics['maximum_normalised_order_gap']} |
| Both orders in all blocks | {metrics['both_safe_orders_achieved_in_all_blocks']} |

## 7. Controller-label invariance

Max label-swap error: `{metrics['label_swap_max_error']}`

## 8. Oscillation exploit

Max oscillation ratio: `{metrics['max_oscillation_ratio']}`

## 9. Hard-braking shares (nominal safe)

Range: `{metrics['nominal_safe_braking_share_min']}` … `{metrics['nominal_safe_braking_share_max']}`

## 10. Discontinuities

| Kind | Count |
|------|-------|
| Physical | {metrics['physical_route_discontinuity']} |
| Fixture-injected | {metrics['fixture_injected_discontinuity']} |

## 11. Integrity

| Metric | Count |
|--------|-------|
| Repeated exits | {metrics['repeated_exit_count']} |
| Invalid flags | {metrics['invalid_flag_count']} |
| NaN | {metrics['nan_count']} |
| Decomp mismatch | {metrics['decomp_mismatch_count']} |
| Collision-exit conflict | {metrics['collision_exit_conflict_count']} |
| Stakeholder mismatch | {metrics['stakeholder_mismatch_count']} |

## 12. Base reward redesign?

**{metrics['base_reward_requires_redesign']}**

## 13. Recommendation

**{recommendation}**

Policy training has **not** started.
"""
    (dirs["reports"] / "stage3a_report.md").write_text(report, encoding="utf-8")

    outputs = {
        "transition_trace_jsonl": str(dirs["raw"] / "transition_trace.jsonl"),
        "scenario_outcomes_jsonl": str(dirs["raw"] / "scenario_outcomes.jsonl"),
        "matched_order_pairs_jsonl": str(dirs["raw"] / "matched_order_pairs.jsonl"),
        "oscillation_cycles_jsonl": str(dirs["raw"] / "oscillation_cycles.jsonl"),
        "collision_audit_jsonl": str(dirs["raw"] / "collision_audit.jsonl"),
        "scenario_summary_csv": str(dirs["processed"] / "scenario_summary.csv"),
        "incentive_ordering_csv": str(dirs["processed"] / "incentive_ordering.csv"),
        "order_bias_summary_csv": str(dirs["processed"] / "order_bias_summary.csv"),
        "comfort_summary_csv": str(dirs["processed"] / "comfort_summary.csv"),
        "oscillation_summary_csv": str(dirs["processed"] / "oscillation_summary.csv"),
        "identity_invariance_summary_csv": str(
            dirs["processed"] / "identity_invariance_summary.csv"
        ),
        "stage3a_report_md": str(dirs["reports"] / "stage3a_report.md"),
        "stage3a_summary_json": str(dirs["reports"] / "stage3a_summary.json"),
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
        "package_versions": _versions(),
        "operating_system": f"{platform.system()} {platform.release()}",
        "command": " ".join(
            [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)]
        ),
        "configuration_sha256": config_hash,
        "gamma": gamma,
        "frozen_reward_weights": cfg.get("frozen_reward_weights"),
        "test_only_comfort_parameters": cfg.get("comfort_test_only"),
        "matched_initial_condition_blocks": cfg.get("matched_blocks"),
        "scenario_names": sorted({o.scenario_id for o in outcomes}),
        "scenario_counts": len(outcomes),
        "safe_collision_truncation_counts": {
            "safe": metrics["safe_scenario_count"],
            "collision": metrics["collision_scenario_count"],
            "truncation": metrics["truncation_scenario_count"],
        },
        "order_gap_metrics": {
            "median_normalised_order_gap": metrics["median_normalised_order_gap"],
            "maximum_normalised_order_gap": metrics["maximum_normalised_order_gap"],
        },
        "oscillation_metrics": {
            "max_oscillation_ratio": metrics["max_oscillation_ratio"],
            "ratios": metrics["oscillation_ratios"],
        },
        "hard_braking_metrics": {
            "nominal_share_min": metrics["nominal_safe_braking_share_min"],
            "nominal_share_max": metrics["nominal_safe_braking_share_max"],
        },
        "integrity_counts": {
            "physical_route_discontinuity": metrics["physical_route_discontinuity"],
            "fixture_injected_discontinuity": metrics["fixture_injected_discontinuity"],
            "repeated_exit_count": metrics["repeated_exit_count"],
            "invalid_flag_count": metrics["invalid_flag_count"],
            "nan_count": metrics["nan_count"],
            "decomp_mismatch_count": metrics["decomp_mismatch_count"],
        },
        "test_counts": counts,
        "outputs": outputs,
        "pass_fail_status": overall,
        "policy_training_started": False,
        "base_reward_requires_redesign": redesign,
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
                "reports": outputs["stage3a_report_md"],
                "manifest": outputs["manifest_json"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"overall={overall}")
    log(f"metrics={json.dumps(metrics, default=str)[:500]}")
    return 0 if overall == "PASS" and unit_ok else (0 if overall == "BLOCKED" else 1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
