"""Input inventory and integrity for Stage 7A-0."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

FORMAL_BASELINE_SEEDS = list(range(61001, 61011))
FORMAL_CHECKPOINT_STEPS = [10_000, 25_000, 50_000, 75_000, 100_000]
ACTION_NAMES = {0: "maintain", 1: "accelerate", 2: "decelerate"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_paper_integrity(repo_root: Path) -> list[dict[str, Any]]:
    patterns = ["*.tex", "*.bib", "chapter*.md", "thesis*.md", "dissertation*.md", "*.docx"]
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pat in patterns:
        for path in sorted(repo_root.rglob(pat)):
            if not path.is_file():
                continue
            # skip diagnostic/experiment outputs and venv
            parts = set(path.parts)
            if parts & {"node_modules", ".venv", ".venv_stage2b1", "output", "releases"}:
                continue
            if path in seen:
                continue
            seen.add(path)
            rel = path.relative_to(repo_root).as_posix()
            rows.append(
                {
                    "path": rel,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_bytes(b"")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def inspect_weights(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = set(payload.keys()) if isinstance(payload, dict) else set()
    return {
        "contains_online_network": "A_online" in keys and "B_online" in keys,
        "contains_target_network": "A_target" in keys and "B_target" in keys,
        "contains_optimizer": any("optim" in k.lower() for k in keys),
        "contains_replay": any("replay" in k.lower() and k != "replay_seeds" for k in keys),
        "contains_rng_state": any("rng" in k.lower() for k in keys),
        "contains_schedule_state": any("schedule" in k.lower() or "ic_schedule" in k.lower() for k in keys),
        "resumable": False,  # published weights are not resumable
        "payload_keys": sorted(str(k) for k in keys),
    }


def build_checkpoint_inventory(
    *,
    stage6a_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    stage6a_root = Path(stage6a_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = stage6a_root / "aggregates" / "checkpoint_manifest.csv"
    pub_rows: list[dict[str, Any]] = []
    missing_full: list[dict[str, Any]] = []

    if manifest_csv.is_file():
        man = pd.read_csv(manifest_csv)
        base = man[man["condition"] == "baseline"].copy()
        for _, r in base.iterrows():
            seed = int(r["master_seed"])
            step = int(r["step"])
            name = str(r["checkpoint"])
            published = bool(r["published"])
            job = stage6a_root / "jobs" / f"baseline__{seed}"
            if name.startswith("ckpt_step_"):
                local_path = stage6a_root / "checkpoints" / f"baseline__{seed}" / name
                exists = local_path.is_file()
                row = {
                    "seed": seed,
                    "step": step,
                    "artifact": name,
                    "path": str(local_path.as_posix()) if exists else f"<missing>:{local_path.as_posix()}",
                    "exists": exists,
                    "published": published,
                    "size_bytes": int(local_path.stat().st_size) if exists else int(r["size_bytes"]),
                    "mtime_ns": int(local_path.stat().st_mtime_ns) if exists else None,
                    "sha256_recorded": str(r["sha256"]),
                    "sha256_actual": sha256_file(local_path) if exists else "",
                    "contains_online_network": True,
                    "contains_target_network": True,
                    "contains_optimizer": True,
                    "contains_replay": True,
                    "contains_rng_state": True,
                    "contains_schedule_state": True,
                    "resumable": exists,
                    "availability": "present" if exists else "missing_local_only",
                }
                if exists:
                    pub_rows.append(row)
                else:
                    missing_full.append(row)
            elif name == "final_online_target_weights.pt":
                path = job / name
                meta = inspect_weights(path) if path.is_file() else {}
                pub_rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "artifact": name,
                        "path": path.as_posix() if path.is_file() else f"<missing>:{path.as_posix()}",
                        "exists": path.is_file(),
                        "published": published,
                        "size_bytes": int(path.stat().st_size) if path.is_file() else 0,
                        "mtime_ns": int(path.stat().st_mtime_ns) if path.is_file() else None,
                        "sha256_recorded": str(r["sha256"]),
                        "sha256_actual": sha256_file(path) if path.is_file() else "",
                        **{
                            k: meta.get(k, False)
                            for k in (
                                "contains_online_network",
                                "contains_target_network",
                                "contains_optimizer",
                                "contains_replay",
                                "contains_rng_state",
                                "contains_schedule_state",
                                "resumable",
                            )
                        },
                        "availability": "present" if path.is_file() else "missing",
                    }
                )

    write_csv(out_dir / "checkpoint_integrity_before.csv", pub_rows + missing_full)

    locks = stage6a_root / "locks"
    inv = {
        "formal_baseline_condition_key": "baseline",
        "formal_master_seeds": FORMAL_BASELINE_SEEDS,
        "formal_checkpoint_steps": FORMAL_CHECKPOINT_STEPS,
        "planned_evaluation_episodes_per_checkpoint": 16,
        "validation_block_count": 8,
        "assignments_per_block": 2,
        "episode_time_limit_policy_steps": 400,
        "policy_interval_seconds": 0.2,
        "published_final_weights_count": sum(
            1 for r in pub_rows if r["artifact"] == "final_online_target_weights.pt" and r["exists"]
        ),
        "full_resumable_checkpoint_count": sum(1 for r in missing_full if r.get("exists")),
        "missing_full_checkpoint_count": len(missing_full),
        "continuation_probe_status": "BLOCKED" if missing_full and not any(r.get("exists") for r in missing_full) else "UNKNOWN",
        "continuation_block_reason": (
            "Published Stage 6A retains only final_online_target_weights.pt. "
            "Full ckpt_step_*.pt files were local_only_intermediate_or_replay_checkpoint "
            "and are absent from this results worktree; resume/continuation is impossible."
        ),
        "protocol_hash": (
            sha256_file(locks / "final_training_protocol.yaml")
            if (locks / "final_training_protocol.yaml").is_file()
            else None
        ),
        "environment_lock_hash": None,
        "comfort_lock_hash": None,
        "stage6a_root_logical": "formal-results-100k-complete/stage6a_20260730T094829Z_a89256db_44d5e647",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_checkpoints": pub_rows,
        "missing_full_checkpoints": missing_full,
    }
    # Fill lock hashes from repo if present via loader constants path later
    (out_dir / "input_inventory.json").write_text(
        json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return inv
