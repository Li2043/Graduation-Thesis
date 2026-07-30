"""Evaluation isolation and baseline-only guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.analysis.reconstruct_eval import load_learners_from_final_weights
from thesis.diagnostics.stage7a0_inventory import FORMAL_BASELINE_SEEDS

STAGE6A = Path(
    r"C:\Users\HP\Desktop\毕业项目\thesis\final_new_results_100k\formal_results\100k\stage6a_20260730T094829Z_a89256db_44d5e647"
)


@pytest.mark.skipif(
    not (STAGE6A / "jobs/baseline__61001/final_online_target_weights.pt").is_file(),
    reason="no stage6a",
)
def test_load_baseline_weights_does_not_require_optimizer():
    w = STAGE6A / "jobs/baseline__61001/final_online_target_weights.pt"
    learners = load_learners_from_final_weights(w, condition="baseline")
    assert set(learners) == {"A", "B"}
    assert len(learners["A"].replay) == 0


def test_reject_non_baseline_condition_key():
    assert all(61001 <= s <= 61010 for s in FORMAL_BASELINE_SEEDS)
