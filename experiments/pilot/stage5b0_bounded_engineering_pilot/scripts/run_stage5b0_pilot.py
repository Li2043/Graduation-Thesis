#!/usr/bin/env python3
"""Stage 5B-0 — bounded engineering pilot runner."""

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
    "src/thesis/training/pilot_config.py",
    "src/thesis/training/pilot_ic_schedule.py",
    "src/thesis/training/pilot_checkpoint.py",
    "src/thesis/training/pilot_evaluation.py",
    "src/thesis/training/pilot_training_loop.py",
    "src/thesis/training/pilot_resume.py",
    "src/thesis/training/final_v3_pipeline.py",
    "src/thesis/training/final_lock_loader.py",
    "src/thesis/agents/independent_dqn_v2.py",
    "src/thesis/agents/replay_buffer_v2.py",
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
        "checkpoints": EXP_ROOT / "checkpoints" / run_id,
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_ROOT / "configs" / "stage5b0.yaml")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.training.final_lock_loader import FinalLockBlockedError, load_final_locks
    from thesis.training.pilot_config import PilotConfig
    from thesis.training.pilot_resume import run_resume_equivalence
    from thesis.training.pilot_training_loop import PilotEngineeringError, PilotTrainer
    from thesis.training.pilot_checkpoint import load_checkpoint

    git_commit = _git(["rev-parse", "HEAD"]) or "unknown"
    git_dirty = bool(_git(["status", "--porcelain"]))
    run_id = _run_id(git_commit)
    dirs = _ensure(run_id)
    log_path = dirs["logs"] / "runner.log"

    def log(msg: str) -> None:
        line = f"[{_utc_now().isoformat()}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    cfg_yaml = _load_yaml(args.config)
    pilot_cfg = PilotConfig()
    pilot_cfg.validate()
    cfg_hash = pilot_cfg.sha256()
    source_hashes = {rel: _sha256(REPO_ROOT / rel) for rel in SOURCE_MODULES}

    log(f"run_id={run_id}")
    log(f"git_commit={git_commit} dirty={git_dirty}")
    log(f"pilot_config_hash={cfg_hash}")
    exact_command = f"{sys.executable} {SCRIPT_PATH}"

    env_lock = REPO_ROOT / cfg_yaml["authoritative_locks"]["environment_lock_path"]
    comfort_lock = REPO_ROOT / cfg_yaml["authoritative_locks"]["comfort_lock_path"]
    env_before = _sha256(env_lock)
    comfort_before = _sha256(comfort_lock)
    log(f"environment lock sha before={env_before}")
    log(f"comfort lock sha before={comfort_before}")

    overall = "PASS"
    blocked = False
    integrity = {
        "lock_hash_mismatch": 0,
        "engineering_failures": 0,
        "resume_errors": 0,
        "nan_inf_errors": 0,
        "illegal_action_errors": 0,
        "evaluation_mutation_errors": 0,
        "missing_checkpoint_errors": 0,
        "step_count_errors": 0,
    }

    resolved = {
        "run_id": run_id,
        "pilot_config": pilot_cfg.to_dict(),
        "pilot_config_sha256": cfg_hash,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "environment_lock_sha256_before": env_before,
        "comfort_lock_sha256_before": comfort_before,
        "environment_parameters_final": True,
        "comfort_parameters_final": True,
        "pbrs_parameters_final": False,
        "training_protocol_final": False,
        "pilot_training_started": True,
        "policy_training_started": True,
        "formal_training_started": False,
        "sustained_training_invoked": True,
        "source_hashes": source_hashes,
    }
    _write_yaml(dirs["artifacts"] / "resolved_pilot_config.yaml", resolved)
    (dirs["artifacts"] / "resolved_pilot_config.sha256").write_text(
        cfg_hash + "\n", encoding="utf-8"
    )

    try:
        if env_before != cfg_yaml["authoritative_locks"]["environment_lock_sha256"]:
            raise FinalLockBlockedError("environment lock hash mismatch")
        if comfort_before != cfg_yaml["authoritative_locks"]["comfort_lock_sha256"]:
            raise FinalLockBlockedError("comfort lock hash mismatch")
        bundle = load_final_locks(
            environment_lock_path=env_lock, comfort_lock_path=comfort_lock
        )
    except FinalLockBlockedError as exc:
        log(f"BLOCKED: {exc}")
        overall = "BLOCKED"
        blocked = True
        integrity["lock_hash_mismatch"] = 1

    run_rows: list[dict[str, Any]] = []
    update_summary: list[dict[str, Any]] = []
    checkpoint_summary: list[dict[str, Any]] = []
    eval_isolation_rows: list[dict[str, Any]] = []
    condition_path_rows: list[dict[str, Any]] = []
    behavioral_rows: list[dict[str, Any]] = []
    resume_result: dict[str, Any] = {}

    if not blocked:
        # Resume equivalence (separate from retained six runs)
        log("running resume-equivalence comparison (baseline/51001, 1000 steps)")
        try:
            resume_result = run_resume_equivalence(
                bundle,
                work_dir=dirs["checkpoints"] / "resume_equivalence",
                comparison_length=int(
                    cfg_yaml["resume_equivalence"]["comparison_length"]
                ),
                interruption_step=int(
                    cfg_yaml["resume_equivalence"]["interruption_step"]
                ),
                pilot_seed=int(cfg_yaml["resume_equivalence"]["pilot_seed"]),
            )
            _write_jsonl(
                dirs["raw"] / "resume_equivalence_trace.jsonl", [resume_result]
            )
            if not resume_result.get("passed"):
                integrity["resume_errors"] += 1
                log(f"FAIL resume equivalence: {resume_result}")
            else:
                log("resume equivalence PASS")
        except Exception as exc:  # noqa: BLE001
            integrity["resume_errors"] += 1
            integrity["engineering_failures"] += 1
            log(f"FAIL resume exception: {exc}")
            resume_result = {"passed": False, "error": str(exc)}

        # Six retained pilot runs
        for condition in pilot_cfg.conditions:
            for seed in pilot_cfg.pilot_seeds:
                log(f"starting pilot run condition={condition} seed={seed}")
                out_dir = dirs["raw"] / condition / str(seed)
                ckpt_dir = dirs["checkpoints"] / condition / str(seed)
                out_dir.mkdir(parents=True, exist_ok=True)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                try:
                    trainer = PilotTrainer(
                        bundle,
                        condition=condition,
                        pilot_seed=int(seed),
                        config=pilot_cfg,
                        checkpoint_dir=ckpt_dir,
                        write_traces=True,
                    )
                    diag = trainer.run()
                    # Reload final checkpoint
                    final_ckpt = ckpt_dir / "ckpt_step_05000.pt"
                    if not final_ckpt.is_file():
                        integrity["missing_checkpoint_errors"] += 1
                        raise PilotEngineeringError("missing final checkpoint")
                    payload = load_checkpoint(final_ckpt)
                    reloader = PilotTrainer(
                        bundle,
                        condition=condition,
                        pilot_seed=int(seed),
                        config=pilot_cfg,
                        checkpoint_dir=ckpt_dir / "reload_check",
                        write_traces=False,
                    )
                    reloader.import_checkpoint(payload)
                    if reloader.env_steps != 5000:
                        integrity["step_count_errors"] += 1

                    _write_jsonl(out_dir / "episode_trace.jsonl", diag.episode_trace)
                    _write_jsonl(
                        out_dir / "transition_trace.jsonl", diag.transition_trace
                    )
                    _write_jsonl(out_dir / "update_trace.jsonl", diag.update_trace)
                    _write_jsonl(
                        out_dir / "evaluation_trace.jsonl", diag.evaluation_trace
                    )
                    _write_jsonl(
                        out_dir / "checkpoint_trace.jsonl", diag.checkpoint_trace
                    )

                    if trainer.env_steps != 5000:
                        integrity["step_count_errors"] += 1
                    if diag.nan_inf_count:
                        integrity["nan_inf_errors"] += diag.nan_inf_count
                    if diag.illegal_action_count:
                        integrity["illegal_action_errors"] += diag.illegal_action_count
                    if any(e.get("mutation_any") for e in diag.evaluation_trace):
                        integrity["evaluation_mutation_errors"] += 1
                    if trainer.learners["A"]._update_count < 100:
                        integrity["engineering_failures"] += 1
                        log("FAIL: fewer than 100 updates for A")
                    if trainer.learners["B"]._update_count < 100:
                        integrity["engineering_failures"] += 1
                        log("FAIL: fewer than 100 updates for B")
                    if trainer.diag.target_syncs["A"] < 1 or trainer.diag.target_syncs["B"] < 1:
                        integrity["engineering_failures"] += 1
                        log("FAIL: missing target sync")

                    run_rows.append(
                        {
                            "condition": condition,
                            "pilot_seed": seed,
                            "env_steps": trainer.env_steps,
                            "episodes": trainer.episode_count,
                            "replay_A": len(trainer.learners["A"].replay),
                            "replay_B": len(trainer.learners["B"].replay),
                            "updates_A": trainer.learners["A"]._update_count,
                            "updates_B": trainer.learners["B"]._update_count,
                            "target_syncs_A": trainer.diag.target_syncs["A"],
                            "target_syncs_B": trainer.diag.target_syncs["B"],
                            "checkpoints": len(diag.checkpoint_trace),
                            "evaluations": len(diag.evaluation_trace),
                            "non_zero_shaping": diag.non_zero_shaping_count,
                            "max_loss": diag.max_abs_loss,
                            "max_abs_q": diag.max_abs_q,
                            "max_decomp_error": diag.max_decomp_error,
                            "completed": True,
                        }
                    )
                    update_summary.append(
                        {
                            "condition": condition,
                            "pilot_seed": seed,
                            "n_updates_logged": len(diag.update_trace),
                            "max_loss": diag.max_abs_loss,
                            "max_abs_q": diag.max_abs_q,
                            "max_abs_target": diag.max_abs_target,
                        }
                    )
                    for c in diag.checkpoint_trace:
                        checkpoint_summary.append(
                            {
                                "condition": condition,
                                "pilot_seed": seed,
                                **c,
                            }
                        )
                    for e in diag.evaluation_trace:
                        eval_isolation_rows.append(
                            {
                                "condition": condition,
                                "pilot_seed": seed,
                                **e,
                            }
                        )
                    condition_path_rows.append(
                        {
                            "condition": condition,
                            "pilot_seed": seed,
                            "non_zero_shaping_count": diag.non_zero_shaping_count,
                            "max_decomp_error": diag.max_decomp_error,
                        }
                    )
                    # Behavioural observations — NOT FOR FORMAL COMPARISON
                    successish = sum(
                        1
                        for ep in diag.episode_trace
                        if ep.get("term_reason") == "success"
                    )
                    collisions = sum(
                        1
                        for ep in diag.episode_trace
                        if ep.get("term_reason") == "collision"
                    )
                    behavioral_rows.append(
                        {
                            "NOTICE": "NOT FOR FORMAL CONDITION COMPARISON",
                            "condition": condition,
                            "pilot_seed": seed,
                            "episodes": trainer.episode_count,
                            "success_term_count_diagnostic": successish,
                            "collision_term_count_diagnostic": collisions,
                            "non_zero_shaping_count": diag.non_zero_shaping_count,
                        }
                    )
                    log(
                        f"completed {condition}/{seed}: steps={trainer.env_steps} "
                        f"updatesA={trainer.learners['A']._update_count} "
                        f"updatesB={trainer.learners['B']._update_count}"
                    )
                except Exception as exc:  # noqa: BLE001
                    integrity["engineering_failures"] += 1
                    log(f"ENGINEERING FAIL {condition}/{seed}: {exc}")
                    run_rows.append(
                        {
                            "condition": condition,
                            "pilot_seed": seed,
                            "completed": False,
                            "error": str(exc),
                        }
                    )

        # Cross-condition shaping checks
        base_nz = sum(
            r["non_zero_shaping"]
            for r in run_rows
            if r.get("condition") == "baseline" and r.get("completed")
        )
        mean_nz = sum(
            r["non_zero_shaping"]
            for r in run_rows
            if r.get("condition") == "mean_pbrs" and r.get("completed")
        )
        min_nz = sum(
            r["non_zero_shaping"]
            for r in run_rows
            if r.get("condition") == "min_pbrs" and r.get("completed")
        )
        if base_nz != 0:
            integrity["engineering_failures"] += 1
            log("FAIL: baseline non-zero shaping")
        if mean_nz < 1 or min_nz < 1:
            integrity["engineering_failures"] += 1
            log("FAIL: mean/min shaping never non-zero")

    log("running pytest suite")
    targets = cfg_yaml.get("pytest_targets") or []
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

    _write_csv(dirs["processed"] / "run_completion.csv", run_rows)
    _write_csv(
        dirs["processed"] / "training_integrity.csv", [{**integrity, "overall": ""}]
    )
    _write_csv(dirs["processed"] / "update_summary.csv", update_summary)
    _write_csv(dirs["processed"] / "checkpoint_summary.csv", checkpoint_summary)
    _write_csv(
        dirs["processed"] / "resume_equivalence.csv",
        [resume_result] if resume_result else [],
    )
    _write_csv(dirs["processed"] / "evaluation_isolation.csv", eval_isolation_rows)
    _write_csv(dirs["processed"] / "condition_path_summary.csv", condition_path_rows)
    _write_csv(
        dirs["processed"] / "pilot_behavioral_observations.csv", behavioral_rows
    )

    completed_runs = sum(1 for r in run_rows if r.get("completed"))
    if blocked:
        overall = "BLOCKED"
    elif test_info["status"] != "PASS":
        overall = "FAIL"
    elif any(v > 0 for v in integrity.values()):
        overall = "FAIL"
    elif completed_runs != 6:
        overall = "FAIL"
    elif git_dirty:
        overall = "FAIL"
        log("FAIL: git_dirty=true")
    else:
        overall = "PASS"

    # Patch integrity csv overall
    _write_csv(
        dirs["processed"] / "training_integrity.csv",
        [{**integrity, "overall": overall}],
    )

    total_steps = sum(int(r.get("env_steps", 0)) for r in run_rows if r.get("completed"))
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
        "pilot_config_hash": cfg_hash,
        "conditions": list(pilot_cfg.conditions),
        "seeds": list(pilot_cfg.pilot_seeds),
        "runs_completed": completed_runs,
        "environment_steps_per_run": 5000,
        "total_environment_steps": total_steps,
        "episodes_per_run": {
            f"{r['condition']}/{r['pilot_seed']}": r.get("episodes")
            for r in run_rows
            if r.get("completed")
        },
        "replay_sizes": {
            f"{r['condition']}/{r['pilot_seed']}": {
                "A": r.get("replay_A"),
                "B": r.get("replay_B"),
            }
            for r in run_rows
            if r.get("completed")
        },
        "optimiser_updates": {
            f"{r['condition']}/{r['pilot_seed']}": {
                "A": r.get("updates_A"),
                "B": r.get("updates_B"),
            }
            for r in run_rows
            if r.get("completed")
        },
        "target_syncs": {
            f"{r['condition']}/{r['pilot_seed']}": {
                "A": r.get("target_syncs_A"),
                "B": r.get("target_syncs_B"),
            }
            for r in run_rows
            if r.get("completed")
        },
        "checkpoint_counts": {
            f"{r['condition']}/{r['pilot_seed']}": r.get("checkpoints")
            for r in run_rows
            if r.get("completed")
        },
        "evaluation_counts": {
            f"{r['condition']}/{r['pilot_seed']}": r.get("evaluations")
            for r in run_rows
            if r.get("completed")
        },
        "resume_equivalence_errors": integrity["resume_errors"],
        "action_replay_integrity_counts": {
            "illegal_actions": integrity["illegal_action_errors"],
        },
        "nan_inf_counts": integrity["nan_inf_errors"],
        "maximum_loss": max((r.get("max_loss") or 0.0) for r in run_rows) if run_rows else 0.0,
        "maximum_absolute_q": max((r.get("max_abs_q") or 0.0) for r in run_rows) if run_rows else 0.0,
        "maximum_absolute_target": max(
            (u.get("max_abs_target") or 0.0) for u in update_summary
        )
        if update_summary
        else 0.0,
        "reward_decomposition_error": max(
            (r.get("max_decomp_error") or 0.0) for r in run_rows
        )
        if run_rows
        else 0.0,
        "evaluation_state_mutation_counts": integrity["evaluation_mutation_errors"],
        "baseline_nonzero_shaping_count": sum(
            r.get("non_zero_shaping", 0)
            for r in run_rows
            if r.get("condition") == "baseline" and r.get("completed")
        ),
        "mean_nonzero_shaping_count": sum(
            r.get("non_zero_shaping", 0)
            for r in run_rows
            if r.get("condition") == "mean_pbrs" and r.get("completed")
        ),
        "min_nonzero_shaping_count": sum(
            r.get("non_zero_shaping", 0)
            for r in run_rows
            if r.get("condition") == "min_pbrs" and r.get("completed")
        ),
        "integrity": integrity,
        "environment_parameters_final": True,
        "comfort_parameters_final": True,
        "pbrs_parameters_final": False,
        "training_protocol_final": False,
        "pilot_training_started": True,
        "policy_training_started": True,
        "formal_training_started": False,
        "sustained_training_invoked": True,
        "unresolved_engineering_limitations": [
            "Pilot hyperparameters are engineering-only; not a training-protocol lock.",
            "PBRS lambda=0.2 is pilot-only and not selected by performance.",
            "Network [64,64] is pilot-only; not architecture selection.",
        ],
        "behavioral_observations_notice": "NOT FOR FORMAL CONDITION COMPARISON",
    }
    (dirs["reports"] / "stage5b0_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    report = f"""# Stage 5B-0 Report — Bounded Engineering Pilot

## Overall: **{overall}**

- run_id: `{run_id}`
- git: `{git_commit}` dirty=`{git_dirty}`
- tests: `{test_info}`
- pilot_config_hash: `{cfg_hash}`
- runs completed: {completed_runs}/6
- total environment steps: {total_steps}
- resume equivalence errors: {integrity['resume_errors']}
- integrity: `{integrity}`

## Status flags

- environment_parameters_final = true
- comfort_parameters_final = true
- pbrs_parameters_final = false
- training_protocol_final = false
- pilot_training_started = true
- policy_training_started = true
- formal_training_started = false
- sustained_training_invoked = true

## Behavioural observations

**NOT FOR FORMAL CONDITION COMPARISON**

See `data/processed/{run_id}/pilot_behavioral_observations.csv`.
"""
    (dirs["reports"] / "stage5b0_report.md").write_text(report, encoding="utf-8")
    manifest = {
        **summary,
        "utc_timestamp": _utc_now().isoformat(),
        "python_and_packages": _versions(),
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
