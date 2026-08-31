"""Manifest helpers for Stage 7A-0."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

_ABS_WIN = re.compile(r"^[A-Za-z]:[\\/]")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_absolute_path_string(value: str) -> bool:
    s = str(value)
    return bool(_ABS_WIN.match(s) or s.startswith("\\\\") or s.startswith("/Users/") or s.startswith("/home/"))


def collect_hashes(root: Path, patterns: Iterable[str]) -> dict[str, str]:
    root = Path(root)
    out: dict[str, str] = {}
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file() and p.name != "baseline_diagnostic_manifest.json":
                rel = p.relative_to(root).as_posix()
                if ".." in Path(rel).parts:
                    raise RuntimeError(rel)
                out[rel] = sha256_file(p)
    return dict(sorted(out.items()))


def verify_manifest_hashes(*, artifact_root: Path, manifest_path: Path) -> None:
    artifact_root = Path(artifact_root).resolve()
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    hashes: Mapping[str, str] = payload.get("output_hashes") or {}
    seen: set[str] = set()
    for rel, expected in hashes.items():
        rel = str(rel).replace("\\", "/")
        if rel in seen:
            raise RuntimeError(f"duplicate {rel}")
        seen.add(rel)
        if is_absolute_path_string(rel) or ".." in Path(rel).parts:
            raise RuntimeError(f"bad path {rel}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            raise RuntimeError(f"bad hash {rel}")
        fp = (artifact_root / rel).resolve()
        fp.relative_to(artifact_root)
        if not fp.is_file():
            raise FileNotFoundError(rel)
        actual = sha256_file(fp)
        if actual != expected:
            raise RuntimeError(f"hash mismatch {rel}")
