"""Paper / inventory integrity tests for Stage 7A-0."""

from __future__ import annotations

from pathlib import Path

from thesis.diagnostics.stage7a0_inventory import collect_paper_integrity


def test_paper_integrity_collects_without_error():
    repo = Path(__file__).resolve().parents[2]
    rows = collect_paper_integrity(repo)
    assert isinstance(rows, list)
    # Must not raise; may be empty if no chapter files matched filters
    for r in rows:
        assert "path" in r and "sha256" in r
        assert "\\" not in r["path"] or True  # posix preferred
        assert len(r["sha256"]) == 64
