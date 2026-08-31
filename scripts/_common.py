"""Shared helpers for the migration bundle's orchestration scripts.

Path resolution is entirely relative to this file's location, so the
bundle works after being copied to any drive/folder (F:\\正式训练,
D:\\正式训练, C:\\thesis_formal_training, ...). Nothing here is allowed
to change scientific semantics -- it only locates files, launches the
EXISTING frozen training/eval scripts as subprocesses, and tracks
run state for crash recovery.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BUNDLE_ROOT / "project"
SB_SCRIPTS = PROJECT_ROOT / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
SB_SRC = PROJECT_ROOT / "src"
CHECKPOINTS_ROOT = BUNDLE_ROOT / "checkpoints"
SCENARIO_BANKS = BUNDLE_ROOT / "scenario_banks"
CONFIGS = BUNDLE_ROOT / "configs"
EXPERIMENT_RECORDS = BUNDLE_ROOT / "experiment_records"
LOGS = BUNDLE_ROOT / "logs"
OUTPUTS = BUNDLE_ROOT / "outputs"
VERIFICATION = BUNDLE_ROOT / "verification"

FROZEN_CONFIG_PATH = CONFIGS / "FROZEN_EXPERIMENT_CONFIG.json"
CHECKSUMS_PATH = BUNDLE_ROOT / "CHECKSUMS.sha256"
MANIFEST_PATH = BUNDLE_ROOT / "MIGRATION_MANIFEST.json"
RUN_STATE_DIR = OUTPUTS / "run_state"
NEEDS_DECISION_PATH = BUNDLE_ROOT / "NEEDS_USER_DECISION.md"


def load_frozen_config() -> dict:
    return json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))


def python_exe() -> str:
    """The venv python for THIS bundle (created by 00_SETUP), not
    whatever `python` happens to resolve to on PATH."""
    venv_py = BUNDLE_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: dict) -> None:
    """Write-to-temp-then-rename so a crash never leaves a partially
    written status/manifest file that a later script could mistake for
    valid state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_manifest_path(run_id: str) -> Path:
    return RUN_STATE_DIR / f"{run_id}.json"


def read_run_manifest(run_id: str) -> dict | None:
    p = run_manifest_path(run_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_run_manifest(run_id: str, **fields) -> None:
    p = run_manifest_path(run_id)
    existing = read_run_manifest(run_id) or {}
    existing.update(fields)
    existing["last_update_unix"] = time.time()
    write_json_atomic(p, existing)


def needs_user_decision(issue: str, evidence: str, options: list[str],
                         consequences: str, recommendation: str) -> None:
    """Write NEEDS_USER_DECISION.md and raise SystemExit(2) -- the
    convention every orchestration script uses when it hits a genuine
    scientific/judgment fork instead of a routine engineering step."""
    lines = [
        "# NEEDS_USER_DECISION",
        "",
        f"Written: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Issue", issue, "",
        "## Evidence", evidence, "",
        "## Options",
    ]
    for i, opt in enumerate(options, 1):
        lines.append(f"{i}. {opt}")
    lines += ["", "## Scientific consequences", consequences, "",
              "## Recommended conservative action", recommendation, ""]
    NEEDS_DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[STOP] Wrote {NEEDS_DECISION_PATH} -- a scientific decision is required, not an engineering one.")
    raise SystemExit(2)


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[int, Path] | None:
    """Latest COMPLETE checkpoint by step number in a
    seed_{seed}_{stage} directory. Never trusts a checkpoint whose
    mtime is within the last 5 seconds (possible in-progress write) --
    callers doing a live migration snapshot should prefer the
    second-latest in that edge case."""
    if not ckpt_dir.exists():
        return None
    best = None
    for f in ckpt_dir.glob("ckpt_step_*.pt"):
        try:
            step = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        if best is None or step > best[0]:
            best = (step, f)
    return best


def run_subprocess(cmd: list[str], *, log_file: Path, env_overrides: dict | None = None) -> subprocess.Popen:
    """Launch a training/eval script as its own OS process (process-level
    parallelism, per the migration's own rule -- never thread-level or
    algorithmic parallelism inside one run). Caller is responsible for
    tracking the returned Popen / its pid for status.py and crash
    recovery."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    log_fh = open(log_file, "a", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log_fh, stderr=subprocess.STDOUT, env=env)
