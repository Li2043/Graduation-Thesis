"""Manifest path and hash helpers for Stage 6B-H1 / H1.1 releases."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


_ABS_WIN = re.compile(r"^[A-Za-z]:[\\/]")
_ABS_POSIX = re.compile(r"^/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_absolute_path_string(value: str) -> bool:
    s = str(value)
    if _ABS_WIN.match(s) or s.startswith("\\\\"):
        return True
    if _ABS_POSIX.match(s) and not s.startswith("output/") and not s.startswith("experiments/"):
        # allow relative posix-like paths starting without drive
        if s.startswith("/Users/") or s.startswith("/home/"):
            return True
    return False


def to_h1_relative(path: Path, *, h1_root: Path) -> str:
    """Return path relative to experiments/formal/stage6b_h1/ using forward slashes."""
    path = Path(path).resolve()
    h1_root = Path(h1_root).resolve()
    rel = path.relative_to(h1_root)
    text = rel.as_posix()
    if ".." in Path(text).parts:
        raise ValueError(f"path escapes h1 root: {text}")
    return text


def verify_manifest_hashes(
    *,
    artifact_root: Path,
    manifest_path: Path,
    hash_key: str = "output_hashes",
) -> None:
    """Verify every registered relative path hash against files under artifact_root.

    ``artifact_root`` should be ``experiments/formal/stage6b_h1`` when hashes are
    stored relative to that directory.
    """
    artifact_root = Path(artifact_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes: Mapping[str, str] = payload.get(hash_key) or {}
    if not hashes:
        raise RuntimeError("manifest has empty output_hashes")
    seen: set[str] = set()
    for relative_path, expected_hash in hashes.items():
        rel = str(relative_path).replace("\\", "/")
        if rel in seen:
            raise RuntimeError(f"duplicate manifest path: {rel}")
        seen.add(rel)
        if is_absolute_path_string(rel):
            raise RuntimeError(f"absolute path in manifest: {rel}")
        if ".." in Path(rel).parts:
            raise RuntimeError(f"path escape in manifest: {rel}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)):
            raise RuntimeError(f"invalid sha256 for {rel}: {expected_hash}")
        file_path = (artifact_root / rel).resolve()
        try:
            file_path.relative_to(artifact_root)
        except ValueError as exc:
            raise RuntimeError(f"path escapes artifact root: {rel}") from exc
        if not file_path.is_file():
            raise FileNotFoundError(f"Manifest file is missing: {rel}")
        actual = sha256_file(file_path)
        if actual != expected_hash:
            raise RuntimeError(
                f"Hash mismatch for {rel}: expected={expected_hash}, actual={actual}"
            )


def collect_release_files(h1_root: Path) -> list[Path]:
    """Deterministic allowlisted formal files under stage6b_h1."""
    h1_root = Path(h1_root)
    patterns = [
        "output/data/**/*",
        "output/statistics/**/*",
        "output/diagnostics/**/*",
        "output/figures/**/*",
        "output/manifests/analysis_manifest.json",
        "output/manifests/acceptance_checks.json",
        "output/manifests/input_inventory.json",
        "output/manifests/h1_1_release_decision.json",
        "reports/stage6b_h1_execution_report.md",
        "reports/stage6b_h1_1_release_report.md",
        "reports/PAPER_CHANGES_REQUIRED_LATER.md",
        "reports/code_audit_before_changes.md",
        "reports/execution_vs_release_commit_diff.md",
        "analysis_requirements_h1.txt",
        "environment_snapshot.json",
        "pip_freeze.txt",
        "logs/stage6b_h1_runner.log",
        "logs/h1_1_preflight.txt",
    ]
    files: list[Path] = []
    for pattern in patterns:
        for p in sorted(h1_root.glob(pattern)):
            if not p.is_file():
                continue
            if p.name == "analysis_manifest.json" and "output/manifests" in p.as_posix():
                # hashed separately after all other files exist; skip during pre-hash collect
                continue
            if any(part in {"__pycache__", ".pytest_cache"} for part in p.parts):
                continue
            if p.suffix in {".pyc", ".tmp"}:
                continue
            files.append(p)
    # include analysis_manifest only when collecting for archive inventory after write
    return sorted(set(files), key=lambda x: x.as_posix())


def build_output_hashes(h1_root: Path, files: Iterable[Path]) -> dict[str, str]:
    h1_root = Path(h1_root).resolve()
    out: dict[str, str] = {}
    for path in files:
        rel = to_h1_relative(path, h1_root=h1_root)
        if rel.endswith("output/manifests/analysis_manifest.json"):
            continue
        out[rel] = sha256_file(path)
    return dict(sorted(out.items()))


__all__ = [
    "build_output_hashes",
    "collect_release_files",
    "is_absolute_path_string",
    "sha256_file",
    "to_h1_relative",
    "verify_manifest_hashes",
]
