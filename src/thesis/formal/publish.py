"""Publish allowlist, oversized-file rejection, and publish manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

MAX_ORDINARY_GIT_FILE_MIB = 90
MAX_ORDINARY_GIT_FILE_BYTES = MAX_ORDINARY_GIT_FILE_MIB * 1024 * 1024

# Paths that must never enter ordinary Git publish trees.
FORBIDDEN_PUBLISH_SUFFIXES = (
    ".pt.tmp",
    ".venv",
)
FORBIDDEN_NAME_FRAGMENTS = (
    "replay_buffer",
    "full_replay",
    "transition_stream",
    "__pycache__",
    ".venv",
)

# Intermediate full checkpoints with replay are local-only.
LOCAL_ONLY_GLOBS = (
    "**/checkpoints/**/*.pt",
    "**/ckpt_step_*.pt",
)


PUBLISH_ALLOWED_SUFFIXES = (
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".jsonl.gz",
    ".csv",
    ".md",
    ".txt",
    ".log",
    ".log.gz",
    ".sha256",
    # Final network weights (small) may be published; full replay .pt excluded by path rules
    ".npz",
)


def is_local_only_checkpoint(path: Path) -> bool:
    p = str(path).replace("\\", "/").lower()
    if "/checkpoints/" in p and p.endswith(".pt"):
        return True
    if "ckpt_step_" in Path(path).name and path.suffix == ".pt":
        # Intermediate scheduled checkpoints with replay — local only
        return True
    return False


def is_publish_allowed(path: Path) -> bool:
    path = Path(path)
    name = path.name.lower()
    text = str(path).replace("\\", "/").lower()
    if is_local_only_checkpoint(path):
        # Final online/target network exports use distinct names
        if name.startswith("final_") and name.endswith((".npz", ".json")):
            return True
        return False
    for frag in FORBIDDEN_NAME_FRAGMENTS:
        if frag in text:
            return False
    if path.suffix.lower() in {".pt"} and "final_" not in name:
        return False
    # Allow listed analysis / report formats
    suf = path.suffix.lower()
    if suf in {".gz"}:
        # e.g. .jsonl.gz
        return any(str(path).lower().endswith(s) for s in PUBLISH_ALLOWED_SUFFIXES)
    if suf in PUBLISH_ALLOWED_SUFFIXES or path.name.endswith(".jsonl.gz") or path.name.endswith(".log.gz"):
        return True
    if suf in {".sha256"}:
        return True
    return False


def reject_oversized_git_files(paths: Iterable[Path], *, limit_bytes: int = MAX_ORDINARY_GIT_FILE_BYTES) -> list[str]:
    """Return list of violations; empty means OK."""
    violations: list[str] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > limit_bytes:
            violations.append(f"{path} is {size} bytes (> {limit_bytes})")
    return violations


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_publish_manifest(
    published_files: list[Path],
    *,
    root: Path,
    source_job: str,
    protocol_hash: str,
    runner_commit: str,
) -> dict[str, Any]:
    entries = []
    root = Path(root)
    for p in published_files:
        path = Path(p)
        rel = str(path.relative_to(root)).replace("\\", "/")
        entries.append(
            {
                "relative_path": rel,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_job": source_job,
            }
        )
    return {
        "protocol_hash": protocol_hash,
        "git_runner_commit": runner_commit,
        "files": entries,
        "n_files": len(entries),
    }


def write_publish_manifest(path: Path, manifest: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


__all__ = [
    "MAX_ORDINARY_GIT_FILE_BYTES",
    "MAX_ORDINARY_GIT_FILE_MIB",
    "build_publish_manifest",
    "is_local_only_checkpoint",
    "is_publish_allowed",
    "reject_oversized_git_files",
    "sha256_file",
    "write_publish_manifest",
]
