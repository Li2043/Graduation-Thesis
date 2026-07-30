#!/usr/bin/env python3
"""Stage 5A-0 — final V3 end-to-end integration regression runner."""

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
    "src/thesis/training/final_lock_loader.py",
    "src/thesis/training/final_reward_conditions.py",
    "src/thesis/training/final_v3_pipeline.py",
    "src/thesis/training/final_experiment_runtime.py",
    "src/thesis/agents/action_masking.py",
    "src/thesis/agents/dqn_targets.py",
    "src/thesis/agents/replay_buffer_v2.py",
    "src/thesis/agents/independent_dqn_v2.py",
    "src/thesis/envs/merge_env_candidate_v3.py",
    "src/thesis/rewards/pbrs_v2.py",
    "src/thesis/rewards/base_reward_v2.py",
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


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
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "torch", "pytest", "yaml"):
        modname = "yaml" if name == "yaml" else name
        try:
            m = __import__(modname)
            out[name] = getattr(m, "__version__", "installed")
        except Exception:
            out[name] = "not_installed"
    return out


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_ROOT / "configs" / "stage5a0.yaml")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.training.final_experiment_runtime import run_condition_suite, write_jsonl
    from thesis.training.final_lock_loader import FinalLockBlockedError, load_final_locks
    from thesis.training.final_v3_pipeline import ENVIRONMENT_CLASS

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
    exact_command = " ".join(
        [
            sys.executable,
            str(SCRIPT_PATH),
        ]
    )

    env_lock = REPO_ROOT / cfg["authoritative_locks"]["environment_lock_path"]
    comfort_lock = REPO_ROOT / cfg["authoritative_locks"]["comfort_lock_path"]
    env_before = _sha256(env_lock)
    comfort_before = _sha256(comfort_lock)
    log(f"environment lock sha before={env_before}")
    log(f"comfort lock sha before={comfort_before}")

    overall = "PASS"
    blocked = False
    integrity: dict[str, Any] = {
        "lock_hash_mismatch": 0,
        "physical_invariance_errors": 0,
        "decomposition_errors": 0,
        "telescoping_errors": 0,
        "determinism_errors": 0,
        "nan_inf_errors": 0,
        "v2_import_errors": 0,
    }
    suite: dict[str, Any] = {}

    try:
        if env_before != cfg["authoritative_locks"]["environment_lock_sha256"]:
            raise FinalLockBlockedError("environment lock hash mismatch")
        if comfort_before != cfg["authoritative_locks"]["comfort_lock_sha256"]:
            raise FinalLockBlockedError("comfort lock hash mismatch")
        bundle = load_final_locks(
            environment_lock_path=env_lock,
            comfort_lock_path=comfort_lock,
        )
    except FinalLockBlockedError as exc:
        log(f"BLOCKED: {exc}")
        overall = "BLOCKED"
        blocked = True
        integrity["lock_hash_mismatch"] = 1

    _write_yaml(
        dirs["artifacts"] / "resolved_config.yaml",
        {
            "run_id": run_id,
            "config_sha256": cfg_hash,
            "config": cfg,
            "source_hashes": source_hashes,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "environment_lock_sha256_before": env_before,
            "comfort_lock_sha256_before": comfort_before,
            "policy_training_started": False,
            "pilot_training_started": False,
            "sustained_training_invoked": False,
            "isolated_optimizer_updates_only": True,
            "pbrs_parameters_final": False,
            "integration_test_only": True,
        },
    )

    if not blocked:
        log("collecting integration traces (no sustained training)")
        suite = run_condition_suite(bundle)
        # Traces
        all_transitions: list[dict[str, Any]] = []
        reward_trace: list[dict[str, Any]] = []
        potential_trace: list[dict[str, Any]] = []
        replay_trace: list[dict[str, Any]] = []
        for cond, ep in suite["by_condition_episodes"].items():
            for row in ep["transitions"]:
                all_transitions.append(row)
                reward_trace.append(
                    {
                        "reward_condition": cond,
                        "policy_step": row["policy_step"],
                        "controller_id": row["controller_id"],
                        "base_reward": row["base_reward"],
                        "shaping_component": row["shaping_component"],
                        "learner_reward": row["learner_reward"],
                        "decomposition_error": row["decomposition_error"],
                    }
                )
                potential_trace.append(
                    {
                        "reward_condition": cond,
                        "policy_step": row["policy_step"],
                        "controller_id": row["controller_id"],
                        "experiences_t": row["experiences_t"],
                        "experiences_t1": row["experiences_t1"],
                        "raw_mean_t": row["raw_mean_t"],
                        "raw_mean_t1": row["raw_mean_t1"],
                        "actual_mean_t": row["actual_mean_t"],
                        "actual_mean_t1": row["actual_mean_t1"],
                        "raw_min_t": row["raw_min_t"],
                        "raw_min_t1": row["raw_min_t1"],
                        "actual_min_t": row["actual_min_t"],
                        "actual_min_t1": row["actual_min_t1"],
                        "mean_shaping_signal": row["mean_shaping_signal"],
                        "min_shaping_signal": row["min_shaping_signal"],
                        "scaled_mean_shaping": row["scaled_mean_shaping"],
                        "scaled_min_shaping": row["scaled_min_shaping"],
                    }
                )
            replay_trace.extend(ep["replay_rows"])
        for ep in (suite["early_exit"], suite["truncation"]):
            all_transitions.extend(ep["transitions"])
            replay_trace.extend(ep["replay_rows"])

        write_jsonl(dirs["raw"] / "integration_transition_trace.jsonl", all_transitions)
        write_jsonl(dirs["raw"] / "reward_condition_trace.jsonl", reward_trace)
        write_jsonl(dirs["raw"] / "potential_trace.jsonl", potential_trace)
        write_jsonl(dirs["raw"] / "replay_trace.jsonl", replay_trace)
        write_jsonl(dirs["raw"] / "isolated_update_trace.jsonl", suite["isolated_updates"])

        # Processed CSVs
        _write_csv(
            dirs["processed"] / "condition_decomposition.csv",
            [
                {
                    "reward_condition": r["reward_condition"],
                    "controller_id": r["controller_id"],
                    "policy_step": r["policy_step"],
                    "base_reward": r["base_reward"],
                    "shaping_component": r["shaping_component"],
                    "learner_reward": r["learner_reward"],
                    "decomposition_error": r["decomposition_error"],
                }
                for r in all_transitions
            ],
        )
        _write_csv(
            dirs["processed"] / "physical_invariance.csv",
            [suite["invariance"]],
        )
        term_rows = []
        for r in all_transitions:
            kind = (
                "controller_terminal"
                if r["controller_terminal"]
                else ("truncation" if r["truncated"] else "ongoing")
            )
            term_rows.append(
                {
                    "controller_id": r["controller_id"],
                    "reward_condition": r["reward_condition"],
                    "policy_step": r["policy_step"],
                    "kind": kind,
                    "terminated": r["terminated"],
                    "truncated": r["truncated"],
                    "controller_terminal": r["controller_terminal"],
                    "next_observation_is_none": r["next_observation"] is None,
                    "target": r["target"],
                    "learner_reward": r["learner_reward"],
                }
            )
        _write_csv(dirs["processed"] / "terminal_semantics.csv", term_rows)
        tele_rows = [
            {"scenario": k, **v} for k, v in suite["telescoping"].items()
        ]
        _write_csv(dirs["processed"] / "pbrs_telescoping.csv", tele_rows)
        _write_csv(dirs["processed"] / "dqn_update_summary.csv", suite["isolated_updates"])

        if suite["invariance"]["max_physical_diff"] != 0.0:
            integrity["physical_invariance_errors"] += 1
        if suite["invariance"]["max_base_reward_diff"] != 0.0:
            integrity["decomposition_errors"] += 1
        if any(
            float(v["mean_error"]) >= 1e-10 or float(v["min_error"]) >= 1e-10
            for v in suite["telescoping"].values()
        ):
            integrity["telescoping_errors"] += 1
        if any(not u["finite_loss"] for u in suite["isolated_updates"]):
            integrity["nan_inf_errors"] += 1
        max_decomp = max(
            (float(r["decomposition_error"]) for r in all_transitions), default=0.0
        )
        if max_decomp > 1e-12:
            integrity["decomposition_errors"] += 1

    log("running pytest regression suite (no sustained training loop)")
    targets = cfg.get("pytest_targets") or []
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
    (dirs["logs"] / "pytest.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    test_info = _parse_pytest(proc.stdout + "\n" + proc.stderr)
    log(f"pytest={test_info}")

    env_after = _sha256(env_lock)
    comfort_after = _sha256(comfort_lock)
    log(f"environment lock sha after={env_after}")
    log(f"comfort lock sha after={comfort_after}")
    if env_after != env_before or comfort_after != comfort_before:
        integrity["lock_hash_mismatch"] += 1
        overall = "FAIL"

    if blocked:
        overall = "BLOCKED"
    elif test_info["status"] != "PASS":
        overall = "FAIL"
    elif any(v > 0 for v in integrity.values()):
        overall = "FAIL"
    elif git_dirty:
        overall = "FAIL"
        log("FAIL: git_dirty=true (dissertation retention requires clean tree)")
    else:
        overall = "PASS"

    # Summary metrics
    n_phys = int(suite.get("invariance", {}).get("n_physical_transitions", 0))
    replay_by: dict[str, int] = {}
    ongoing = terminal = truncation = 0
    early_exit_cases = 0
    if suite:
        for r in suite.get("by_condition_episodes", {}).get("baseline", {}).get(
            "replay_rows", []
        ):
            key = f"{r['controller_id']}:{r['reward_condition']}"
            replay_by[key] = replay_by.get(key, 0) + 1
        for ep in list(suite.get("by_condition_episodes", {}).values()) + [
            suite.get("early_exit", {}),
            suite.get("truncation", {}),
        ]:
            for r in ep.get("transitions", []):
                if r.get("controller_terminal"):
                    terminal += 1
                elif r.get("truncated"):
                    truncation += 1
                else:
                    ongoing += 1
        a_exit = [
            r
            for r in suite.get("early_exit", {}).get("transitions", [])
            if r["controller_id"] == "A" and r["exit_event"]["A"] >= 1.0
        ]
        if a_exit:
            early_exit_cases = 1

    tele = suite.get("telescoping", {})
    mean_tele_err = max(
        (float(v.get("mean_error", 0.0)) for v in tele.values()), default=0.0
    )
    min_tele_err = max(
        (float(v.get("min_error", 0.0)) for v in tele.values()), default=0.0
    )
    updates = suite.get("isolated_updates", [])
    max_loss = max((float(u.get("loss", 0.0)) for u in updates), default=0.0)
    max_q = max((float(u.get("max_abs_q", 0.0)) for u in updates), default=0.0)
    fwd_counts = [int(u.get("target_network_forward_calls", 0)) for u in updates]

    _write_csv(
        dirs["processed"] / "integrity_summary.csv",
        [{**integrity, "overall": overall}],
    )

    summary = {
        "run_id": run_id,
        "overall": overall,
        "exact_command": exact_command,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "tests": test_info,
        "environment_lock_path": str(env_lock),
        "environment_lock_sha256_before": env_before,
        "environment_lock_sha256_after": env_after,
        "comfort_lock_path": str(comfort_lock),
        "comfort_lock_sha256_before": comfort_before,
        "comfort_lock_sha256_after": comfort_after,
        "final_environment_class": ENVIRONMENT_CLASS,
        "observation_dimension": 27,
        "dqn_algorithm": "vanilla_dqn_masked_target_max",
        "integration_test_architecture": True,
        "reward_conditions": ["baseline", "mean_pbrs", "min_pbrs"],
        "integration_lambda_mean": 0.2,
        "integration_lambda_min": 0.2,
        "pbrs_parameters_final": False,
        "integration_test_only": True,
        "n_physical_transitions": n_phys,
        "replay_rows_by_controller_condition": replay_by,
        "ongoing_row_count": ongoing,
        "terminal_row_count": terminal,
        "truncation_row_count": truncation,
        "early_exit_continuation_cases": early_exit_cases,
        "max_reward_decomposition_error": max(
            (
                float(r.get("decomposition_error", 0.0))
                for ep in suite.get("by_condition_episodes", {}).values()
                for r in ep.get("transitions", [])
            ),
            default=0.0,
        ),
        "max_physical_trace_diff": float(
            suite.get("invariance", {}).get("max_physical_diff", 0.0)
        ),
        "mean_telescoping_error": mean_tele_err,
        "min_telescoping_error": min_tele_err,
        "isolated_update_counts": len(updates),
        "max_loss": max_loss,
        "max_abs_q": max_q,
        "target_network_forward_counts": fwd_counts,
        "determinism_errors": integrity["determinism_errors"],
        "integrity": integrity,
        "environment_parameters_final": True,
        "comfort_parameters_final": True,
        "policy_training_started": False,
        "pilot_training_started": False,
        "sustained_training_invoked": False,
        "isolated_optimizer_updates_only": True,
        "source_hashes": source_hashes,
        "unresolved_limitations": [
            "PBRS lambda values are integration-test-only (0.2); not final.",
            "Network [32,32] is integration_test_architecture only.",
            "No final training-protocol or PBRS lock written in this stage.",
        ],
    }
    (dirs["reports"] / "stage5a0_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )

    report = f"""# Stage 5A-0 Report — Final V3 End-to-End Integration

## Overall: **{overall}**

- run_id: `{run_id}`
- git: `{git_commit}` dirty=`{git_dirty}`
- tests: `{test_info}`
- environment class: `{ENVIRONMENT_CLASS}`
- observation_dimension: 27
- algorithm: vanilla DQN (masked target max)
- integration_test_architecture: true
- reward conditions: baseline / mean_pbrs / min_pbrs
- integration lambdas: mean=0.2 min=0.2 (`pbrs_parameters_final=false`)
- environment lock: `{env_before}` (unchanged={env_before == env_after})
- comfort lock: `{comfort_before}` (unchanged={comfort_before == comfort_after})
- policy_training_started: false
- sustained_training_invoked: false
- isolated_optimizer_updates_only: true

## Integrity
```json
{json.dumps(integrity, indent=2)}
```

## Physical / PBRS
- max physical-trace diff: {summary['max_physical_trace_diff']}
- max decomposition error: {summary['max_reward_decomposition_error']}
- mean telescoping error: {mean_tele_err}
- min telescoping error: {min_tele_err}
- early-exit continuation cases: {early_exit_cases}

## Isolated DQN updates
- count: {len(updates)}
- max loss: {max_loss}
- max |Q|: {max_q}
- target-network forward counts: {fwd_counts}
"""
    (dirs["reports"] / "stage5a0_report.md").write_text(report, encoding="utf-8")

    manifest = {
        **summary,
        "utc_timestamp": _utc_now().isoformat(),
        "python_and_packages": _versions(),
        "configuration_sha256": cfg_hash,
        "output_paths": {k: str(v) for k, v in dirs.items()},
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "overall": overall}, indent=2), encoding="utf-8"
    )
    log(f"overall={overall}")
    return 0 if overall == "PASS" else (2 if overall == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
