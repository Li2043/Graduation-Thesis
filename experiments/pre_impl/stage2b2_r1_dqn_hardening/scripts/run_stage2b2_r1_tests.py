#!/usr/bin/env python3
"""Stage 2B-2R — strict mask + controller-terminal hardening test runner."""

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
    "src/thesis/agents/action_masking.py",
    "src/thesis/agents/dqn_targets.py",
    "src/thesis/agents/replay_buffer_v2.py",
    "src/thesis/agents/independent_dqn_v2.py",
    "src/thesis/agents/dqn_pipeline.py",
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
    keys = list(rows[0].keys())
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
    out: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
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
    parser.add_argument("--config", type=Path, default=EXP_ROOT / "configs" / "stage2b2_r1.yaml")
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

    env_lock = REPO_ROOT / cfg["authoritative_locks"]["environment_lock_path"]
    comfort_lock = REPO_ROOT / cfg["authoritative_locks"]["comfort_lock_path"]
    env_before = _sha256(env_lock)
    comfort_before = _sha256(comfort_lock)
    log(f"environment lock sha before={env_before}")
    log(f"comfort lock sha before={comfort_before}")
    if env_before != cfg["authoritative_locks"]["environment_lock_sha256"]:
        log("BLOCKED: environment lock hash mismatch")
        return 2
    if comfort_before != cfg["authoritative_locks"]["comfort_lock_sha256"]:
        log("BLOCKED: comfort lock hash mismatch")
        return 2

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
            "sustained_training_invoked": False,
        },
    )

    log("running pytest (no environment training loop)")
    targets = cfg.get("pytest_targets") or []
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
    (dirs["logs"] / "pytest.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    test_info = _parse_pytest(proc.stdout + "\n" + proc.stderr)
    log(f"pytest={test_info}")

    # Document mask / target cases exercised by the focused suite
    mask_cases = [
        {"case": "bool_mask", "accepted": True},
        {"case": "integer_0_1_mask", "accepted": True},
        {"case": "float_0_1_mask", "accepted": False},
        {"case": "float_0_5_mask", "accepted": False},
        {"case": "integer_2", "accepted": False},
        {"case": "negative", "accepted": False},
        {"case": "nan_float", "accepted": False},
        {"case": "wrong_length", "accepted": False},
        {"case": "multidimensional", "accepted": False},
        {"case": "all_false", "accepted": False},
    ]
    target_cases = [
        {"case": "controller_terminal_equals_reward", "bootstraps": False},
        {"case": "terminal_next_obs_none", "bootstraps": False},
        {"case": "terminal_next_mask_none", "bootstraps": False},
        {"case": "terminal_skips_target_forward", "bootstraps": False},
        {"case": "truncation_bootstraps", "bootstraps": True},
        {"case": "ongoing_requires_next", "bootstraps": True},
        {"case": "mixed_batch_bootstrap_only_forward", "bootstraps": "partial"},
        {"case": "vanilla_dqn_masked_max", "bootstraps": True},
    ]
    _write_csv(dirs["processed"] / "mask_validation_cases.csv", mask_cases)
    _write_csv(dirs["processed"] / "target_semantics_cases.csv", target_cases)

    env_after = _sha256(env_lock)
    comfort_after = _sha256(comfort_lock)
    log(f"environment lock sha after={env_after}")
    log(f"comfort lock sha after={comfort_after}")

    overall = "PASS" if test_info["status"] == "PASS" else "FAIL"
    if env_after != env_before or comfort_after != comfort_before:
        overall = "FAIL"
        log("FAIL: lock hash changed")
    if git_dirty:
        # Retention wants dirty=false at dissertation snapshot; warn but tests may still pass
        log("NOTE: git_dirty=true at runner start (commit before retained snapshot)")

    summary = {
        "run_id": run_id,
        "overall": overall,
        "exact_command": " ".join([sys.executable, str(SCRIPT_PATH)] + sys.argv[1:]),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "modified_source_files": list(SOURCE_MODULES),
        "source_hashes": source_hashes,
        "tests": test_info,
        "accepted_mask_cases": [c["case"] for c in mask_cases if c["accepted"]],
        "rejected_mask_cases": [c["case"] for c in mask_cases if not c["accepted"]],
        "terminal_target_cases": [c["case"] for c in target_cases if c["bootstraps"] is False],
        "truncation_target_cases": ["truncation_bootstraps"],
        "mixed_batch_cases": ["mixed_batch_bootstrap_only_forward"],
        "target_network_call_policy": "bootstrap_rows_only",
        "environment_lock_sha256_before": env_before,
        "environment_lock_sha256_after": env_after,
        "comfort_lock_sha256_before": comfort_before,
        "comfort_lock_sha256_after": comfort_after,
        "policy_training_started": False,
        "sustained_training_invoked": False,
        "algorithm": "vanilla_dqn_masked_target_max",
    }
    (dirs["reports"] / "stage2b2_r1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# Stage 2B-2R Report — Strict Action Mask & Terminal-Target Hardening",
        "",
        f"## Overall: **{overall}**",
        "",
        f"Git: `{git_commit}` dirty=`{git_dirty}`",
        f"Tests: `{test_info}`",
        f"Algorithm: vanilla DQN (masked target max)",
        f"Environment lock: `{env_before}` (unchanged={env_before == env_after})",
        f"Comfort lock: `{comfort_before}` (unchanged={comfort_before == comfort_after})",
        "policy_training_started=false",
        "",
        "## Modified sources",
        *[f"- `{p}` sha=`{source_hashes[p]}`" for p in SOURCE_MODULES],
        "",
    ]
    (dirs["reports"] / "stage2b2_r1_report.md").write_text("\n".join(report), encoding="utf-8")

    manifest = {
        **summary,
        "utc_timestamp": _utc_now().isoformat(),
        "python_and_packages": _versions(),
        "configuration_sha256": cfg_hash,
        "output_paths": {k: str(v) for k, v in dirs.items()},
    }
    (dirs["artifacts"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (EXP_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "overall": overall}, indent=2), encoding="utf-8"
    )
    log(f"overall={overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
