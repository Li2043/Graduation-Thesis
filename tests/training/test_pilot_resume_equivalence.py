"""Stage 5B-0 — resume equivalence (reduced length)."""

from __future__ import annotations

from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_resume import run_resume_equivalence


def test_uninterrupted_vs_resumed_equivalence(tmp_path):
    bundle = load_final_locks()
    result = run_resume_equivalence(
        bundle,
        work_dir=tmp_path / "resume",
        comparison_length=40,
        interruption_step=20,
        pilot_seed=51001,
    )
    assert result["passed"] is True
    assert result["action_mismatch_count"] == 0
    assert result["max_parameter_abs_diff"] <= 1e-12
    assert result["max_loss_diff"] <= 1e-12
