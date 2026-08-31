"""Competence gate and budget selection tests."""

from __future__ import annotations

from thesis.pilots.stage7a1_config import competence_gate_pass, select_stable_budget


def _row(
    step,
    mean_success=0.8,
    mean_collision=0.02,
    mean_truncation=0.1,
    seeds_ge=18,
    swap=0.8,
):
    return {
        "checkpoint_step": step,
        "mean_success": mean_success,
        "mean_collision": mean_collision,
        "mean_truncation": mean_truncation,
        "seeds_success_ge_0_75": seeds_ge,
        "swap_eligible_pair_proportion": swap,
    }


def test_gate_thresholds():
    ok = competence_gate_pass(_row(150000))
    assert ok["passed"] is True
    bad = competence_gate_pass(_row(150000, mean_success=0.7))
    assert bad["passed"] is False
    bad2 = competence_gate_pass(_row(150000, seeds_ge=15))
    assert bad2["passed"] is False


def test_budget_selection_earliest_consecutive():
    rows = [
        _row(100000, mean_success=0.7, seeds_ge=10),
        _row(150000),
        _row(200000),
        _row(250000),
        _row(300000),
    ]
    sel = select_stable_budget(rows)
    assert sel["stable_sufficient_budget"] == 150_000
    assert sel["confirmation_checkpoint"] == 200_000
    assert sel["status"] == "budget-responsive and competence-qualified"


def test_budget_selection_no_best_per_seed():
    # Even if 300K is best mean, earliest consecutive pair wins
    rows = [
        _row(100000),
        _row(150000),
        _row(200000, mean_success=0.95, seeds_ge=20),
        _row(250000, mean_success=0.94, seeds_ge=20),
        _row(300000, mean_success=0.99, seeds_ge=20),
    ]
    sel = select_stable_budget(rows)
    assert sel["stable_sufficient_budget"] == 100_000


def test_only_300k_promising():
    rows = [
        _row(100000, mean_success=0.4, seeds_ge=2),
        _row(150000, mean_success=0.5, seeds_ge=5),
        _row(200000, mean_success=0.6, seeds_ge=8),
        _row(250000, mean_success=0.7, seeds_ge=12),
        _row(300000),
    ]
    sel = select_stable_budget(rows)
    assert sel["stable_sufficient_budget"] is None
    assert "promising" in sel["status"]
