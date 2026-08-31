"""Paper integrity guard for Stage 7B-A1 Phase 1."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_pilot_does_not_live_under_paper_paths():
    pilot = REPO / "experiments" / "pilots" / "stage7b_a1_double_dqn"
    assert pilot.is_dir()
    assert "chapter" not in str(pilot)
    assert not list(pilot.rglob("*.tex"))
    assert not list(pilot.rglob("*.docx"))


def test_stage6_formal_runner_still_present():
    p = REPO / "experiments" / "formal" / "stage6a_formal_training" / "scripts" / "run_formal_job.py"
    assert p.is_file()
