"""Verify formal_publish_manifest.json and lock hashes (read-only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thesis.analysis import (
    EXPECTED_COMFORT_LOCK,
    EXPECTED_ENV_LOCK,
    EXPECTED_PBRS_LOCK,
    EXPECTED_PROTOCOL_LOCK,
    EXPECTED_RUNNER_COMMIT,
)


class AnalysisBlockedError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    # Windows PowerShell exports may include UTF-8 BOM
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_publish_manifest(results_root: Path) -> dict[str, Any]:
    root = Path(results_root)
    manifest_path = root / "formal_publish_manifest.json"
    if not manifest_path.is_file():
        raise AnalysisBlockedError(f"missing publish manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    files = manifest.get("files") or []
    mismatches: list[str] = []
    verified = 0
    for entry in files:
        rel = entry["relative_path"]
        path = root / rel
        if not path.is_file():
            mismatches.append(f"missing:{rel}")
            continue
        size = path.stat().st_size
        if int(entry["size"]) != size:
            mismatches.append(f"size:{rel}:{size}!={entry['size']}")
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            mismatches.append(f"sha256:{rel}")
            continue
        verified += 1
    if mismatches:
        raise AnalysisBlockedError(
            f"publish manifest verification failed ({len(mismatches)}): "
            + "; ".join(mismatches[:10])
        )
    return {
        "n_files": len(files),
        "verified": verified,
        "formal_execution_id": manifest.get("formal_execution_id"),
        "protocol_hash": manifest.get("protocol_hash"),
        "git_runner_commit": manifest.get("git_runner_commit"),
        "overall": manifest.get("overall"),
        "completed_jobs": manifest.get("completed_jobs"),
        "failed_jobs": manifest.get("failed_jobs"),
    }


def verify_lock_hashes(results_root: Path, *, repo_root: Path) -> dict[str, str]:
    root = Path(results_root)
    locks = root / "locks"
    meta = _read_json(root / "execution_metadata.json")
    protocol_hash = sha256_file(locks / "final_training_protocol.yaml")
    pbrs_hash = sha256_file(locks / "final_pbrs_parameters.yaml")
    if protocol_hash != EXPECTED_PROTOCOL_LOCK:
        raise AnalysisBlockedError("training protocol lock hash mismatch")
    if pbrs_hash != EXPECTED_PBRS_LOCK:
        raise AnalysisBlockedError("PBRS lock hash mismatch")
    if str(meta.get("protocol_hash")) != EXPECTED_PROTOCOL_LOCK:
        raise AnalysisBlockedError("execution_metadata protocol_hash mismatch")
    if str(meta.get("pbrs_hash")) != EXPECTED_PBRS_LOCK:
        raise AnalysisBlockedError("execution_metadata pbrs_hash mismatch")
    if str(meta.get("runner_commit")) != EXPECTED_RUNNER_COMMIT:
        raise AnalysisBlockedError("runner_commit mismatch")

    # Authoritative locks in main repo (unchanged)
    env = (
        Path(repo_root)
        / "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
        / "20260730T003122Z_aee2d425/final_environment_lock.yaml"
    )
    comfort = (
        Path(repo_root)
        / "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/artifacts/"
        / "20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
    )
    env_h = sha256_file(env)
    comfort_h = sha256_file(comfort)
    if env_h != EXPECTED_ENV_LOCK:
        raise AnalysisBlockedError("environment lock hash mismatch")
    if comfort_h != EXPECTED_COMFORT_LOCK:
        raise AnalysisBlockedError("comfort lock hash mismatch")

    machine = root / "machine_dependency_manifest.json"
    if not machine.is_file():
        raise AnalysisBlockedError("missing machine_dependency_manifest.json")

    return {
        "environment_lock_sha256": env_h,
        "comfort_lock_sha256": comfort_h,
        "pbrs_lock_sha256": pbrs_hash,
        "training_protocol_sha256": protocol_hash,
        "runner_commit": str(meta["runner_commit"]),
        "formal_execution_id": str(meta["formal_execution_id"]),
        "machine_manifest_sha256": sha256_file(machine),
    }


__all__ = [
    "AnalysisBlockedError",
    "sha256_file",
    "verify_lock_hashes",
    "verify_publish_manifest",
]
