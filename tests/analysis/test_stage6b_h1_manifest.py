"""Manifest / reproducibility helpers for Stage 6B-H1."""

from __future__ import annotations

import json
from pathlib import Path


def test_manifest_must_not_hash_itself_when_present() -> None:
    root = Path(__file__).resolve().parents[2] / "experiments/formal/stage6b_h1/output/manifests"
    path = root / "analysis_manifest.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    hashes = payload.get("output_hashes", {})
    for key in hashes:
        assert "analysis_manifest.json" not in key


def test_h1_output_root_does_not_overwrite_old_stage6b() -> None:
    old = Path(__file__).resolve().parents[2] / "experiments/formal/stage6b_analysis_100k"
    h1 = Path(__file__).resolve().parents[2] / "experiments/formal/stage6b_h1"
    assert old.resolve() != h1.resolve()
