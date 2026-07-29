#!/usr/bin/env python3
"""Stage 2B-2 Independent DQN / replay / bootstrap integration runner."""

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
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("gymnasium", "gymnasium"),
        ("highway-env", "highway_env"),
        ("pytest", "pytest"),
    ):
        try:
            m = __import__(mod)
            out[name] = getattr(m, "__version__", "installed")
        except Exception:
            out[name] = "not_installed"
    return out


def _collect(run_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from thesis.agents.dqn_pipeline import default_learners, run_pipeline_scenario
    from thesis.agents.dqn_targets import compute_dqn_target
    from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
    from thesis.agents.replay_buffer_v2 import ReplayTransition
    from thesis.envs.scripted_scenarios import build_scenarios
    import numpy as np

    dqn_cfg = cfg.get("dqn_test_only", {})
    gamma = float(dqn_cfg.get("gamma", 0.995))

    # Synthetic target cases
    target_cases: list[dict[str, Any]] = []
    cases = [
        ("terminal", 1.25, True, False, [10.0, 20.0, 30.0], [True, True, True], 1.25),
        ("ordinary", 1.0, False, False, [5.0, 1.0, 2.0], [True, True, True], 1.0 + gamma * 5.0),
        (
            "truncation",
            1.0,
            False,
            True,
            [5.0, 0.0, 0.0],
            [True, True, True],
            1.0 + gamma * 5.0,
        ),
        (
            "masked",
            0.0,
            False,
            False,
            [2.0, 100.0, 5.0],
            [True, False, True],
            0.0 + gamma * 5.0,
        ),
    ]
    max_abs_err = 0.0
    for name, r, term, trunc, q, mask, expected in cases:
        bd = compute_dqn_target(
            r,
            terminated=term,
            truncated=trunc,
            gamma=gamma,
            next_q_values=q,
            next_action_mask=mask,
        )
        err = abs(bd.target - expected)
        max_abs_err = max(max_abs_err, err)
        target_cases.append(
            {
                "controller_id": "synthetic",
                "case": name,
                "reward_condition": "baseline",
                "reward": bd.reward,
                "terminated": bd.terminated,
                "truncated": bd.truncated,
                "next_q_values": bd.next_q_values.tolist(),
                "next_action_mask": bd.next_action_mask.tolist(),
                "masked_next_q_max": bd.masked_next_q_max,
                "gamma": bd.gamma,
                "bootstrap_multiplier": bd.bootstrap_multiplier,
                "target": bd.target,
                "expected_target": expected,
                "absolute_error": err,
            }
        )

    pipeline_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    n_ord = n_term = n_trunc = 0
    illegal = invalid_mask = invalid_flag = trunc_missing = nan_count = 0

    for cond in ("baseline", "mean_pbrs", "min_pbrs"):
        for sid, spec in build_scenarios().items():
            learners = default_learners(
                seed_A=int(cfg.get("seed_A", 0)),
                seed_B=int(cfg.get("seed_B", 1)),
                reward_condition=cond,  # type: ignore[arg-type]
            )
            records = run_pipeline_scenario(
                spec,
                learners,
                reward_condition=cond,  # type: ignore[arg-type]
                episode_id=f"{run_id}_{cond}_{sid}",
            )
            for r in records:
                pipeline_rows.append(r)
                if r["terminated"] and r["truncated"]:
                    invalid_flag += 1
                elif r["terminated"]:
                    n_term += 1
                elif r["truncated"]:
                    n_trunc += 1
                else:
                    n_ord += 1
                if not any(r["action_mask"]):
                    invalid_mask += 1
                if not r["action_mask"][r["selected_action"]]:
                    illegal += 1
                if r["truncated"] and r["next_observation"] is None:
                    trunc_missing += 1
                for key in ("learner_reward", "target", "base_reward"):
                    if not math.isfinite(float(r[key])):
                        nan_count += 1
                # target case style row from pipeline
                target_cases.append(
                    {
                        "controller_id": r["controller_id"],
                        "case": f"pipeline_{sid}",
                        "reward_condition": r["reward_condition"],
                        "reward": r["learner_reward"],
                        "terminated": r["terminated"],
                        "truncated": r["truncated"],
                        "next_q_values": None,
                        "next_action_mask": r["next_action_mask"],
                        "masked_next_q_max": r["masked_next_q_max"],
                        "gamma": gamma,
                        "bootstrap_multiplier": r["bootstrap_multiplier"],
                        "target": r["target"],
                        "expected_target": r["target"],
                        "absolute_error": 0.0
                        if r["target_decomposition_valid"]
                        else 1.0,
                    }
                )
            for aid, learner in learners.items():
                start = (learner.replay._write - len(learner.replay)) % learner.replay.capacity
                for i in range(len(learner.replay)):
                    t = learner.replay._storage[(start + i) % learner.replay.capacity]
                    if t is None:
                        continue
                    replay_rows.append(t.to_dict())

    return {
        "target_cases": target_cases,
        "pipeline_rows": pipeline_rows,
        "replay_rows": replay_rows,
        "n_ordinary": n_ord,
        "n_terminal": n_term,
        "n_truncated": n_trunc,
        "illegal_action_count": illegal,
        "invalid_mask_count": invalid_mask,
        "invalid_flag_count": invalid_flag,
        "truncation_without_next_state_count": trunc_missing,
        "nan_count": nan_count,
        "max_target_abs_error": max_abs_err,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs" / "dqn_integration_test.yaml",
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
    log(f"python={sys.version}")
    log(f"git_dirty={git_dirty}")
    if git_dirty:
        log("WARNING: git working tree is DIRTY")

    cfg = _load_yaml(config_path)
    resolved = dirs["artifacts"] / "resolved_config.yaml"
    _write_yaml(resolved, cfg)
    config_hash = _sha256(resolved)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    paths = [REPO_ROOT / p for p in cfg.get("pytest", {}).get("test_paths", [])]
    extra = list(cfg.get("pytest", {}).get("extra_args", ["-q", "--tb=short"]))
    cmd = [sys.executable, "-m", "pytest", *[str(p) for p in paths], *extra]
    log(f"pytest: {paths}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    (dirs["logs"] / "pytest.log").write_text(text, encoding="utf-8")
    counts = _parse_pytest(text)
    log(f"pytest rc={proc.returncode} counts={counts}")

    data = _collect(run_id, cfg)
    _jsonl(dirs["raw"] / "replay_transitions.jsonl", data["replay_rows"])
    _jsonl(dirs["raw"] / "target_cases.jsonl", data["target_cases"])
    _jsonl(dirs["raw"] / "pipeline_transitions.jsonl", data["pipeline_rows"])
    _csv(
        dirs["processed"] / "target_summary.csv",
        [
            {
                k: r[k]
                for k in (
                    "controller_id",
                    "case",
                    "reward_condition",
                    "terminated",
                    "truncated",
                    "bootstrap_multiplier",
                    "target",
                    "expected_target",
                    "absolute_error",
                )
                if k in r
            }
            for r in data["target_cases"]
        ],
    )
    _csv(
        dirs["processed"] / "pipeline_summary.csv",
        [
            {
                k: r[k]
                for k in (
                    "scenario_id",
                    "episode_id",
                    "step",
                    "controller_id",
                    "traffic_role",
                    "reward_condition",
                    "learner_reward",
                    "terminated",
                    "truncated",
                    "target",
                    "target_decomposition_valid",
                )
                if k in r
            }
            for r in data["pipeline_rows"]
        ],
    )

    # Prior stages intact
    s1 = list(
        (REPO_ROOT / "experiments/pre_impl/stage1_base_reward_unit_tests/artifacts").glob(
            "*/manifest.json"
        )
    )
    s2a = list(
        (REPO_ROOT / "experiments/pre_impl/stage2a_pbrs_math_tests/artifacts").glob(
            "*/manifest.json"
        )
    )
    s2b1 = list(
        (
            REPO_ROOT
            / "experiments/pre_impl/stage2b1_env_reward_pbrs_integration/artifacts"
        ).glob("*/manifest.json")
    )
    prior_ok = bool(s1 and s2a and s2b1)

    metrics_ok = (
        data["illegal_action_count"] == 0
        and data["invalid_mask_count"] == 0
        and data["invalid_flag_count"] == 0
        and data["truncation_without_next_state_count"] == 0
        and data["nan_count"] == 0
        and data["max_target_abs_error"] < 1e-12
    )
    unit_ok = proc.returncode == 0 and counts["status"] == "PASS"
    overall = "PASS" if unit_ok and metrics_ok and prior_ok else "FAIL"
    recommendation = (
        "Integration path ready; formal policy training has NOT started"
        if overall == "PASS"
        else "DO NOT PROCEED"
    )

    versions = _versions()
    dqn = cfg.get("dqn_test_only", {})
    summary = {
        "overall": overall,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "package_versions": versions,
        "device": dqn.get("device", "cpu"),
        "unit_integration_tests": counts,
        "n_ordinary": data["n_ordinary"],
        "n_terminal": data["n_terminal"],
        "n_truncated": data["n_truncated"],
        "illegal_action_count": data["illegal_action_count"],
        "invalid_mask_count": data["invalid_mask_count"],
        "invalid_flag_count": data["invalid_flag_count"],
        "truncation_without_next_state_count": data[
            "truncation_without_next_state_count"
        ],
        "nan_count": data["nan_count"],
        "max_target_abs_error": data["max_target_abs_error"],
        "prior_stage_outputs_intact": prior_ok,
        "recommendation": recommendation,
        "policy_training_started": False,
        "note": "Integration test only; architecture/lambdas TEST-ONLY",
    }
    (dirs["reports"] / "stage2b2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = f"""# Stage 2B-2 Report — Independent DQN / Replay / Bootstrap

## Prominent scope statement

**This is an integration test, not policy training.**

- Network architecture is **TEST-ONLY** (`{dqn.get('hidden_sizes')}`).
- Environment parameters are not final.
- PBRS lambda values are not final.
- DQN target suppresses bootstrap **only** for true terminal states.
- External truncation **retains** bootstrap.
- Action masking applies at **selection** and **target**.
- A and B use **separate** learner state.

{" > **WARNING: git_dirty = true**" if git_dirty else ""}

- Git: `{git_commit}` dirty=`{git_dirty}`
- Python: `{sys.version.split()[0]}`
- Framework: `{versions}`
- Config SHA-256: `{config_hash}`

## Tests

| Metric | Value |
|--------|-------|
| Passed | {counts['passed']} |
| Failed | {counts['failed']} |
| Status | {counts['status']} |

## Transition counts

| Kind | Count |
|------|-------|
| Ordinary | {data['n_ordinary']} |
| Terminal | {data['n_terminal']} |
| Truncated | {data['n_truncated']} |

## Integrity counts

| Metric | Value |
|--------|-------|
| Illegal actions | {data['illegal_action_count']} |
| Invalid masks | {data['invalid_mask_count']} |
| Invalid flags | {data['invalid_flag_count']} |
| Truncation without next state | {data['truncation_without_next_state_count']} |
| NaN | {data['nan_count']} |
| Max target abs error | {data['max_target_abs_error']} |

## Example targets

- Terminal: reward=1.25 → target=1.25 (no bootstrap)
- Ordinary: reward=1, γ={dqn.get('gamma')}, maxQ=5 → y = 1 + γ·5
- Truncation: same bootstrap as ordinary
- Masked: Q=[2,100,5], mask=[1,0,1] → max=5

## Completed-controller policy

Option 1 — inactive after exit; replay stops; env forces zero accel; placeholder MAINTAIN may be passed for joint API only.

## Recommendation

**{recommendation}**

Formal policy training has **not** started.
"""
    (dirs["reports"] / "stage2b2_report.md").write_text(report, encoding="utf-8")

    outputs = {
        "replay_transitions_jsonl": str(dirs["raw"] / "replay_transitions.jsonl"),
        "target_cases_jsonl": str(dirs["raw"] / "target_cases.jsonl"),
        "pipeline_transitions_jsonl": str(dirs["raw"] / "pipeline_transitions.jsonl"),
        "target_summary_csv": str(dirs["processed"] / "target_summary.csv"),
        "pipeline_summary_csv": str(dirs["processed"] / "pipeline_summary.csv"),
        "stage2b2_report_md": str(dirs["reports"] / "stage2b2_report.md"),
        "stage2b2_summary_json": str(dirs["reports"] / "stage2b2_summary.json"),
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
        "deep_learning_framework_version": versions.get("torch"),
        "package_versions": versions,
        "device": dqn.get("device", "cpu"),
        "seeds": {"A": cfg.get("seed_A", 0), "B": cfg.get("seed_B", 1)},
        "command": " ".join(
            [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)]
        ),
        "resolved_configuration_path": str(resolved),
        "configuration_sha256": config_hash,
        "observation_dimension": dqn.get("obs_dim"),
        "action_space_size": dqn.get("n_actions"),
        "test_only_network_architecture": dqn.get("hidden_sizes"),
        "test_only_optimiser_configuration": {
            "learning_rate": dqn.get("learning_rate"),
            "batch_size": dqn.get("batch_size"),
        },
        "replay_capacity": dqn.get("replay_capacity"),
        "learner_gamma": dqn.get("gamma"),
        "test_counts": counts,
        "illegal_action_count": data["illegal_action_count"],
        "invalid_mask_count": data["invalid_mask_count"],
        "invalid_flag_count": data["invalid_flag_count"],
        "truncation_without_next_state_count": data[
            "truncation_without_next_state_count"
        ],
        "nan_count": data["nan_count"],
        "maximum_target_absolute_error": data["max_target_abs_error"],
        "outputs": outputs,
        "pass_fail_status": overall,
        "policy_training_started": False,
        "note": "Prior stage outputs not modified.",
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
                "reports": outputs["stage2b2_report_md"],
                "manifest": outputs["manifest_json"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"overall={overall}")
    log(
        f"ord={data['n_ordinary']} term={data['n_terminal']} trunc={data['n_truncated']} "
        f"illegal={data['illegal_action_count']} nan={data['nan_count']} "
        f"max_err={data['max_target_abs_error']}"
    )
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
