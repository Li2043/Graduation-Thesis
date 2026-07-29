#!/usr/bin/env python3
"""Stage-2A PBRS mathematical correctness experiment runner.

No environment, DQN, or policy training. Never overwrites existing runs.
Does not modify Stage 1 historical outputs.
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
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    short = (git_sha or "nogit")[:8]
    return f"{stamp}_{short}"


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
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
            "PyYAML required: pip install -r requirements-stage1.txt"
        ) from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _pkg_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("pytest", "yaml"):
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Union of keys so heterogeneous case/shaped rows share one schema.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            flat = {
                k: (json.dumps(v) if isinstance(v, (list, dict, tuple)) else v)
                for k, v in row.items()
            }
            w.writerow(flat)


def _run_pytest(
    test_path: Path,
    extra_args: list[str],
    log_path: Path,
    env: dict[str, str],
) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", str(test_path), *extra_args]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True
    )
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(text, encoding="utf-8")
    return proc.returncode, text


def _parse_pytest_counts(pytest_log: str) -> dict[str, Any]:
    passed = failed = errors = skipped = 0
    for label in ("passed", "failed", "error", "skipped"):
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
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "status": status,
    }


def _ensure_src_path() -> None:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _build_potential_cases(gamma: float, lam_mean: float, lam_min: float) -> list[dict[str, Any]]:
    _ensure_src_path()
    from thesis.rewards.pbrs_v2 import (
        PBRSConfig,
        apply_pbrs_to_base_rewards,
        compute_potential_breakdown,
        potential_state_from_experiences,
    )

    def E(a, b, f, r):
        return {"A": a, "B": b, "B_front": f, "B_rear": r}

    cfg = PBRSConfig(
        learner_gamma=gamma,
        shaping_gamma=gamma,
        lambda_mean=lam_mean,
        lambda_min=lam_min,
    )
    cases: list[tuple[str, dict[str, float], bool, bool]] = [
        ("active_target", E(1.0, 1.0, 1.0, 1.0), False, False),
        ("mean_example", E(0.2, 0.4, 0.6, 0.8), False, False),
        ("worst_off", E(0.2, 0.8, 0.8, 0.8), False, False),
        ("terminal_success", E(0.9, 0.9, 0.9, 0.9), True, False),
        ("truncation", E(0.5, 0.6, 0.7, 0.8), False, True),
    ]
    records: list[dict[str, Any]] = []
    for case_id, exp, term, trunc in cases:
        for cond in ("mean", "min"):
            bd = compute_potential_breakdown(
                potential_state_from_experiences(
                    exp, terminated=term, truncated=trunc
                ),
                cond,
                experiences=exp,
            )
            records.append(
                {
                    "case_id": case_id,
                    "condition": cond,
                    "experiences": bd.stakeholder_experiences,
                    "worst_off_ids": list(bd.worst_off_ids),
                    "raw_potential": bd.raw_potential,
                    "actual_potential": bd.actual_potential,
                    "terminated": term,
                    "truncated": trunc,
                }
            )
        # shaping decomposition example for non-terminal pair
        if not term:
            e_t1 = {k: min(1.0, v + 0.1) for k, v in exp.items()}
            out = apply_pbrs_to_base_rewards(
                {"A": 0.04, "B": -0.02},
                potential_state_from_experiences(exp),
                potential_state_from_experiences(e_t1, truncated=trunc),
                "mean",
                cfg,
                experiences_t=exp,
                experiences_t1=e_t1,
            )
            for aid, br in out.items():
                records.append(
                    {
                        "case_id": f"{case_id}_shaped",
                        "condition": "mean",
                        "controller": aid,
                        "base_reward": br.base_reward,
                        "scaled_shaping_component": br.scaled_shaping_component,
                        "shaped_reward": br.shaped_reward,
                        "shaping_signal": br.shaping_signal,
                        "phi_t": br.phi_t,
                        "phi_t1": br.phi_t1,
                        "decomposition_ok": abs(
                            br.shaped_reward
                            - (br.base_reward + br.scaled_shaping_component)
                        )
                        < 1e-12,
                    }
                )
    return records


def _trajectory_records(
    *,
    trajectory_id: str,
    experience_seq: list[dict[str, float]],
    condition: str,
    gamma: float,
    terminated_last: bool,
    truncated_last: bool,
    tol: float,
) -> tuple[list[dict[str, Any]], float]:
    """Build per-transition telescoping records; return (records, abs_error vs identity)."""
    _ensure_src_path()
    from thesis.rewards.pbrs_v2 import (
        compute_potential_breakdown,
        potential_state_from_experiences,
    )

    n = len(experience_seq)
    assert n >= 2
    phis_raw: list[float] = []
    phis_act: list[float] = []
    worst: list[list[str]] = []
    for i, exp in enumerate(experience_seq):
        is_last = i == n - 1
        term = bool(terminated_last and is_last)
        trunc = bool(truncated_last and is_last)
        bd = compute_potential_breakdown(
            potential_state_from_experiences(exp, terminated=term, truncated=trunc),
            condition,  # type: ignore[arg-type]
            experiences=exp,
        )
        phis_raw.append(bd.raw_potential)
        phis_act.append(bd.actual_potential)
        worst.append(list(bd.worst_off_ids))

    records: list[dict[str, Any]] = []
    cum = 0.0
    disc = 1.0
    for t in range(n - 1):
        phi_t = phis_act[t]
        phi_t1 = phis_act[t + 1]
        f_t = gamma * phi_t1 - phi_t
        disc_f = disc * f_t
        cum += disc_f
        is_last_trans = t == n - 2
        term = bool(terminated_last and is_last_trans)
        trunc = bool(truncated_last and is_last_trans)
        records.append(
            {
                "trajectory_id": trajectory_id,
                "step": t,
                "condition": condition,
                "stakeholder_experiences": experience_seq[t],
                "stakeholder_experiences_t1": experience_seq[t + 1],
                "worst_off_stakeholder_ids": worst[t],
                "worst_off_stakeholder_ids_t1": worst[t + 1],
                "raw_phi_t": phis_raw[t],
                "actual_phi_t": phi_t,
                "raw_phi_t1": phis_raw[t + 1],
                "actual_phi_t1": phi_t1,
                "gamma": gamma,
                "F_t": f_t,
                "discounted_F_t": disc_f,
                "cumulative_discounted_shaping": cum,
                "terminated": term,
                "truncated": trunc,
            }
        )
        disc *= gamma

    k = n - 1
    if terminated_last:
        expected = -phis_act[0]
    else:
        expected = -phis_act[0] + (gamma**k) * phis_act[k]
    abs_err = abs(cum - expected)
    for rec in records:
        rec["expected_telescoping_value"] = expected
        rec["absolute_error"] = abs_err
        rec["telescoping_ok"] = abs_err <= tol
    return records, abs_err


def _generate_all_trajectories(gamma: float, tol: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def E(a, b, f, r):
        return {"A": a, "B": b, "B_front": f, "B_rear": r}

    specs: list[dict[str, Any]] = [
        {
            "trajectory_id": "smooth_improvement",
            "seq": [E(0.2, 0.3, 0.4, 0.5), E(0.3, 0.4, 0.5, 0.6), E(0.4, 0.5, 0.6, 0.7), E(0.5, 0.6, 0.7, 0.8)],
            "terminated_last": False,
            "truncated_last": False,
        },
        {
            "trajectory_id": "mean_improve_min_constant",
            "seq": [E(0.2, 0.8, 0.8, 0.8), E(0.2, 0.9, 0.9, 0.9), E(0.2, 1.0, 1.0, 1.0)],
            "terminated_last": False,
            "truncated_last": False,
        },
        {
            "trajectory_id": "worst_off_improvement",
            "seq": [E(0.2, 0.8, 0.8, 0.8), E(0.25, 0.8, 0.8, 0.8), E(0.3, 0.8, 0.8, 0.8)],
            "terminated_last": False,
            "truncated_last": False,
        },
        {
            "trajectory_id": "worst_off_identity_switch",
            "seq": [E(0.30, 0.31, 0.80, 0.80), E(0.32, 0.29, 0.80, 0.80), E(0.33, 0.28, 0.80, 0.80)],
            "terminated_last": False,
            "truncated_last": False,
        },
        {
            "trajectory_id": "true_terminal_success",
            "seq": [E(0.5, 0.6, 0.7, 0.8), E(0.7, 0.75, 0.8, 0.85), E(0.9, 0.9, 0.9, 0.9)],
            "terminated_last": True,
            "truncated_last": False,
        },
        {
            "trajectory_id": "true_terminal_collision",
            "seq": [E(0.4, 0.5, 0.6, 0.7), E(0.45, 0.55, 0.65, 0.75), E(0.2, 0.2, 0.2, 0.2)],
            "terminated_last": True,
            "truncated_last": False,
        },
        {
            "trajectory_id": "external_truncation_nonzero",
            "seq": [E(0.3, 0.4, 0.5, 0.6), E(0.35, 0.45, 0.55, 0.65), E(0.4, 0.5, 0.6, 0.7)],
            "terminated_last": False,
            "truncated_last": True,
        },
    ]

    all_recs: list[dict[str, Any]] = []
    terminal_errs: list[float] = []
    trunc_errs: list[float] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in specs:
        for cond in ("mean", "min"):
            tid = f"{spec['trajectory_id']}__{cond}"
            recs, err = _trajectory_records(
                trajectory_id=tid,
                experience_seq=spec["seq"],
                condition=cond,
                gamma=gamma,
                terminated_last=spec["terminated_last"],
                truncated_last=spec["truncated_last"],
                tol=tol,
            )
            all_recs.extend(recs)
            if spec["terminated_last"]:
                terminal_errs.append(err)
            if spec["truncated_last"]:
                trunc_errs.append(err)
            # Also check non-terminal open segments against truncated identity
            if not spec["terminated_last"] and not spec["truncated_last"]:
                trunc_errs.append(err)
            summary_rows.append(
                {
                    "trajectory_id": tid,
                    "condition": cond,
                    "n_transitions": len(recs),
                    "terminated_last": spec["terminated_last"],
                    "truncated_last": spec["truncated_last"],
                    "absolute_error": err,
                    "telescoping_ok": err <= tol,
                    "final_cumulative": recs[-1]["cumulative_discounted_shaping"] if recs else None,
                    "expected_telescoping_value": recs[-1]["expected_telescoping_value"] if recs else None,
                }
            )

    metrics = {
        "terminal_telescoping_max_abs_error": max(terminal_errs) if terminal_errs else None,
        "truncation_telescoping_max_abs_error": max(trunc_errs) if trunc_errs else None,
        "all_telescoping_ok": all(r["telescoping_ok"] for r in summary_rows),
        "summary_rows": summary_rows,
    }
    return all_recs, metrics


def _write_report(
    path: Path,
    *,
    overall: str,
    git_commit: str,
    git_dirty: bool,
    config_hash: str,
    unit_counts: dict[str, Any],
    gamma: float,
    lam_mean: float,
    lam_min: float,
    tele_metrics: dict[str, Any],
    recommendation: str,
) -> None:
    lines: list[str] = []
    lines.append("# Stage 2A Report — PBRS Mathematical Correctness")
    lines.append("")
    lines.append(f"## 1. Overall: **{overall}**")
    lines.append("")
    if git_dirty:
        lines.append("> **WARNING: git working tree is DIRTY (`git_dirty = true`).**")
        lines.append(
            "> For dissertation-grade acceptance, the final retained Stage 2A run "
            "should have `git_dirty = false`."
        )
        lines.append("")
    lines.append(f"- Git commit: `{git_commit}`")
    lines.append(f"- Git dirty: `{git_dirty}`")
    lines.append(f"- Configuration SHA-256: `{config_hash}`")
    lines.append(f"- Learner / shaping gamma: `{gamma}`")
    lines.append(
        f"- Test-only lambda_mean / lambda_min: `{lam_mean}` / `{lam_min}` "
        "(NOT final experimental values)"
    )
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
    lines.append("## 3. Telescoping identities")
    lines.append("")
    lines.append(
        f"- Terminal max |error|: `{tele_metrics.get('terminal_telescoping_max_abs_error')}`"
    )
    lines.append(
        f"- Truncation/open-segment max |error|: "
        f"`{tele_metrics.get('truncation_telescoping_max_abs_error')}`"
    )
    lines.append(f"- All telescoping OK: `{tele_metrics.get('all_telescoping_ok')}`")
    lines.append("")
    lines.append("| trajectory_id | abs_error | ok |")
    lines.append("|---------------|-----------|----|")
    for row in tele_metrics.get("summary_rows", []):
        lines.append(
            f"| {row['trajectory_id']} | {row['absolute_error']:.3e} | {row['telescoping_ok']} |"
        )
    lines.append("")
    lines.append("## 4. Scope limits")
    lines.append("")
    lines.append(
        "- Environment integration: **UNVERIFIED** (not in Stage 2A scope)."
    )
    lines.append("- DQN / policy training: **UNVERIFIED** (not in Stage 2A scope).")
    lines.append("- Lambda values: **TEST-ONLY**; do not treat as calibrated.")
    lines.append("")
    lines.append("## 5. Recommendation")
    lines.append("")
    lines.append(f"**{recommendation}**")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-2A PBRS math tests runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "pbrs_math_tests.yaml",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()

    git_commit = _git(["rev-parse", "HEAD"]) or "unknown"
    git_dirty = bool(_git(["status", "--porcelain"]))
    run_id = _run_id(git_commit)
    dirs = _ensure_unique_run_dirs(run_id)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{_utc_now().isoformat()}] {msg}"
        log_lines.append(line)
        print(msg)

    log(f"run_id={run_id}")
    log(f"git_dirty={git_dirty}")
    if git_dirty:
        log("WARNING: git working tree is DIRTY")

    cfg = _load_yaml(config_path)
    resolved_path = dirs["artifacts"] / "resolved_config.yaml"
    _write_yaml(resolved_path, cfg)
    config_hash = _sha256_file(resolved_path)

    pbrs_cfg = cfg.get("pbrs", {})
    gamma = float(pbrs_cfg.get("learner_gamma", 0.995))
    shaping_gamma = float(pbrs_cfg.get("shaping_gamma", gamma))
    lam_mean = float(pbrs_cfg.get("lambda_mean", 0.5))
    lam_min = float(pbrs_cfg.get("lambda_min", 0.5))
    tol = float(cfg.get("telescoping", {}).get("tolerance", 1e-12))
    seed = int(cfg.get("seed", 0))

    command = " ".join(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)]
    )

    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    test_rel = cfg.get("pytest", {}).get("test_path", "tests/rewards/test_pbrs_v2.py")
    test_path = REPO_ROOT / test_rel
    extra = list(cfg.get("pytest", {}).get("extra_args", ["-q", "--tb=short"]))

    log(f"running pytest: {test_path}")
    pytest_rc, pytest_text = _run_pytest(
        test_path, extra, dirs["logs"] / "pytest.log", env
    )
    unit_counts = _parse_pytest_counts(pytest_text)
    log(f"pytest returncode={pytest_rc} counts={unit_counts}")

    potential_cases = _build_potential_cases(gamma, lam_mean, lam_min)
    _append_jsonl(dirs["raw"] / "potential_cases.jsonl", potential_cases)
    summary_keys = {
        "case_id",
        "condition",
        "controller",
        "raw_potential",
        "actual_potential",
        "base_reward",
        "shaped_reward",
        "decomposition_ok",
        "terminated",
        "truncated",
        "worst_off_ids",
    }
    _write_csv(
        dirs["processed"] / "potential_summary.csv",
        [{k: v for k, v in r.items() if k in summary_keys} for r in potential_cases],
    )

    tele_recs, tele_metrics = _generate_all_trajectories(gamma, tol)
    _append_jsonl(dirs["raw"] / "telescoping_transitions.jsonl", tele_recs)
    _write_csv(dirs["processed"] / "telescoping_summary.csv", tele_metrics["summary_rows"])

    log(
        f"terminal_telescoping_max_abs_error="
        f"{tele_metrics['terminal_telescoping_max_abs_error']}"
    )
    log(
        f"truncation_telescoping_max_abs_error="
        f"{tele_metrics['truncation_telescoping_max_abs_error']}"
    )

    unit_ok = pytest_rc == 0 and unit_counts.get("status") == "PASS"
    tele_ok = bool(tele_metrics.get("all_telescoping_ok"))
    gamma_ok = abs(gamma - shaping_gamma) <= float(
        pbrs_cfg.get("gamma_match_tolerance", 1e-12)
    )
    overall = "PASS" if (unit_ok and tele_ok and gamma_ok) else "FAIL"
    if overall == "PASS":
        recommendation = "PROCEED TO STAGE 2B (environment / learner wiring only; lambdas still uncalibrated)"
    else:
        recommendation = "DO NOT PROCEED"

    summary = {
        "overall": overall,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_sha256": config_hash,
        "learner_gamma": gamma,
        "shaping_gamma": shaping_gamma,
        "test_only_lambdas": {
            "lambda_mean": lam_mean,
            "lambda_min": lam_min,
            "note": "TEST-ONLY placeholders; not final experimental values",
        },
        "unit_tests": unit_counts,
        "pytest_returncode": pytest_rc,
        "terminal_telescoping_max_abs_error": tele_metrics[
            "terminal_telescoping_max_abs_error"
        ],
        "truncation_telescoping_max_abs_error": tele_metrics[
            "truncation_telescoping_max_abs_error"
        ],
        "all_telescoping_ok": tele_ok,
        "recommendation": recommendation,
        "scope": {
            "environment_verified": False,
            "dqn_verified": False,
        },
    }
    (dirs["reports"] / "stage2a_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(
        dirs["reports"] / "stage2a_report.md",
        overall=overall,
        git_commit=git_commit,
        git_dirty=git_dirty,
        config_hash=config_hash,
        unit_counts=unit_counts,
        gamma=gamma,
        lam_mean=lam_mean,
        lam_min=lam_min,
        tele_metrics=tele_metrics,
        recommendation=recommendation,
    )

    outputs = {
        "potential_cases_jsonl": str(dirs["raw"] / "potential_cases.jsonl"),
        "telescoping_transitions_jsonl": str(
            dirs["raw"] / "telescoping_transitions.jsonl"
        ),
        "potential_summary_csv": str(dirs["processed"] / "potential_summary.csv"),
        "telescoping_summary_csv": str(dirs["processed"] / "telescoping_summary.csv"),
        "stage2a_report_md": str(dirs["reports"] / "stage2a_report.md"),
        "stage2a_summary_json": str(dirs["reports"] / "stage2a_summary.json"),
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
        "learner_gamma": gamma,
        "shaping_gamma": shaping_gamma,
        "test_only_lambda_values": {
            "lambda_mean": lam_mean,
            "lambda_min": lam_min,
            "note": "TEST-ONLY; not final",
        },
        "test_counts": unit_counts,
        "terminal_telescoping_maximum_absolute_error": tele_metrics[
            "terminal_telescoping_max_abs_error"
        ],
        "truncation_telescoping_maximum_absolute_error": tele_metrics[
            "truncation_telescoping_max_abs_error"
        ],
        "outputs": outputs,
        "pass_fail_status": overall,
        "environment_verified": False,
        "dqn_verified": False,
        "note": "Stage 1 outputs were not modified. No env/DQN in this stage.",
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (dirs["logs"] / "runner.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    latest = {
        "run_id": run_id,
        "utc_timestamp": manifest["utc_timestamp"],
        "overall": overall,
        "git_dirty": git_dirty,
        "reports": str(dirs["reports"] / "stage2a_report.md"),
        "manifest": str(dirs["artifacts"] / "manifest.json"),
    }
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps(latest, indent=2), encoding="utf-8"
    )

    # Confirm Stage 1 run still present
    s1 = (
        REPO_ROOT
        / "experiments"
        / "pre_impl"
        / "stage1_base_reward_unit_tests"
        / "artifacts"
        / "20260729T163419Z_e55e4170"
        / "manifest.json"
    )
    log(f"stage1_manifest_intact={s1.exists()}")
    log(f"overall={overall} recommendation={recommendation}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
