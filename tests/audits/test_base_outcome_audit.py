"""Unit tests for Stage 3A audit metrics and helpers."""

from __future__ import annotations

import pytest

from thesis.audits.audit_metrics import (
    braking_penalty_share,
    check_incentive_ordering,
    closed_cycle_progress_sum,
    discounted_return,
    normalised_order_gap,
    oscillation_ratio,
    undiscounted_return,
)
from thesis.audits.audit_scenarios import build_matched_blocks, build_all_audit_scenarios


def test_discounted_and_undiscounted_returns():
    r = [1.0, 1.0, 1.0]
    assert undiscounted_return(r) == pytest.approx(3.0)
    g = 0.995
    expected = 1.0 + g + g**2
    assert discounted_return(r, g) == pytest.approx(expected)


def test_closed_cycle_progress_sum_zero():
    deltas = [0.1, -0.1, 0.05, -0.05]
    assert closed_cycle_progress_sum(deltas) == pytest.approx(0.0, abs=1e-12)


def test_order_gap_calculation():
    og = normalised_order_gap(1.0, 1.02)
    assert og["order_gap"] == pytest.approx(0.02)
    assert og["normalised_order_gap"] == pytest.approx(0.02 / 1.01)


def test_oscillation_ratio():
    assert oscillation_ratio(0.01, 1.0) == pytest.approx(0.01)


def test_braking_penalty_share():
    assert braking_penalty_share(-0.05, 0.4, 0.6) == pytest.approx(0.05)


def test_incentive_ordering_reports_block_violations():
    bad = check_incentive_ordering(
        "block_x",
        g_safe_mainline=1.0,
        g_safe_ramp=1.0,
        g_slow_mainline=None,
        g_slow_ramp=None,
        g_stall_partial=2.0,  # stall better than safe → violation
        g_early_coll=-1.0,
        g_late_coll=-1.0,
    )
    assert bad.ok is False
    assert any("safe_not_above_stall" in v for v in bad.violations)


def test_fixture_only_excluded_from_primary_ranking_flags():
    scenarios = build_all_audit_scenarios()
    fixtures = [s for s in scenarios if s.fixture_only]
    assert fixtures
    assert all(s.primary_ranking is False for s in fixtures)


def test_eight_matched_blocks():
    blocks = build_matched_blocks()
    assert len(blocks) >= 8
    ids = [b.block_id for b in blocks]
    assert len(ids) == len(set(ids))


def test_non_finite_rejected():
    with pytest.raises(ValueError):
        discounted_return([float("nan")], 0.995)
