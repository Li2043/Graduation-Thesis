"""Manifest / accounting tests for Stage 6B (uses results worktree if present)."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.analysis.manifest_verify import (
    AnalysisBlockedError,
    verify_lock_hashes,
    verify_publish_manifest,
)

RESULTS = Path(__file__).resolve().parents[3].parent / "final_new_results_100k" / (
    "formal_results/100k/stage6a_20260730T094829Z_a89256db_44d5e647"
)
# worktree is sibling of final_new
RESULTS = Path(__file__).resolve().parents[3].parent / "final_new_results_100k"
# parents[3] is final_new; sibling is final_new_results_100k
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO.parent / "final_new_results_100k" / (
    "formal_results/100k/stage6a_20260730T094829Z_a89256db_44d5e647"
)


@pytest.mark.skipif(not RESULTS.is_dir(), reason="results worktree not available")
def test_manifest_and_locks():
    pub = verify_publish_manifest(RESULTS)
    assert pub["verified"] == pub["n_files"]
    assert pub["completed_jobs"] == 30
    locks = verify_lock_hashes(RESULTS, repo_root=REPO)
    assert locks["runner_commit"].startswith("a89256d")


@pytest.mark.skipif(not RESULTS.is_dir(), reason="results worktree not available")
def test_exactly_30_slots():
    import csv

    rows = list(csv.DictReader((RESULTS / "aggregates/run_status.csv").open(encoding="utf-8")))
    assert len(rows) == 30
    assert {r["condition"] for r in rows} == {"baseline", "mean_pbrs", "min_pbrs"}
    assert {int(r["master_seed"]) for r in rows} == set(range(61001, 61011))
    assert all(r["status"] == "COMPLETE" for r in rows)
