"""Budget selection alias module tests (same helpers)."""

from __future__ import annotations

from thesis.pilots.stage7a1_config import non_degradation, select_stable_budget


def test_non_degradation_limits():
    a = {
        "mean_success": 0.80,
        "mean_collision": 0.02,
        "mean_truncation": 0.10,
        "seeds_success_ge_0_75": 18,
    }
    b = {
        "mean_success": 0.76,
        "mean_collision": 0.03,
        "mean_truncation": 0.12,
        "seeds_success_ge_0_75": 17,
    }
    assert non_degradation(a, b) is True
    b2 = dict(b)
    b2["mean_success"] = 0.70
    assert non_degradation(a, b2) is False


def test_select_returns_dict_status():
    rows = [
        {
            "checkpoint_step": s,
            "mean_success": 0.4,
            "mean_collision": 0.1,
            "mean_truncation": 0.5,
            "seeds_success_ge_0_75": 2,
            "swap_eligible_pair_proportion": 0.2,
        }
        for s in (100000, 150000, 200000, 250000, 300000)
    ]
    sel = select_stable_budget(rows)
    assert "status" in sel
    assert sel["stable_sufficient_budget"] is None
