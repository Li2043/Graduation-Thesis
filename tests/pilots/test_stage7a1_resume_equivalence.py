"""Tiny-budget resume equivalence for Stage 7A-1 FormalTrainer path."""

from __future__ import annotations

from thesis.pilots.stage7a1_resume import run_resume_equivalence


def test_resume_equivalence_tiny(tmp_path):
    report = run_resume_equivalence(
        work_dir=tmp_path / "resume",
        protocol_hash="test",
        master_seed=62001,
        interruption_step=30,
        comparison_step=60,
    )
    assert report["passed"] is True
    assert report["network_parameter_max_abs_diff"] == 0.0
    assert report["replay_row_mismatch"] == 0
    assert report["rng_state_mismatch"] == 0
    assert report["evaluation_outcome_mismatch"] == 0
