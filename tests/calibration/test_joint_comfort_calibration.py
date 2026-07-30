"""Tests for joint comfort calibration rules (Stage 3B-R1)."""

from __future__ import annotations

from thesis.calibration.joint_comfort_calibration import (
    A_COMFORT_GRID,
    A_HARD_GRID,
    ETA_GRID,
    build_complete_tuples,
    build_threshold_pairs,
    episode_braking_shares,
    select_feasible_tuple,
    valid_threshold_pair_r1,
)
from thesis.rewards.base_reward_v2 import compute_hard_braking_cost


def test_candidate_pair_gap_and_grid_sizes():
    ok, _ = valid_threshold_pair_r1(2.5, 3.0)
    assert not ok  # gap 0.5 < 1.0
    ok2, _ = valid_threshold_pair_r1(1.5, 3.0)
    assert ok2
    pairs = build_threshold_pairs()
    assert len(pairs) == 14
    assert len(A_COMFORT_GRID) == 3
    assert len(A_HARD_GRID) == 5
    assert len(ETA_GRID) == 19
    assert len(build_complete_tuples()) == 14 * 19


def test_h_thresholds_and_monotonicity():
    assert compute_hard_braking_cost(-1.0, 1.5, 3.5) == 0.0
    assert compute_hard_braking_cost(-1.5, 1.5, 3.5) == 0.0
    h1 = compute_hard_braking_cost(-2.5, 1.5, 3.5)
    h2 = compute_hard_braking_cost(-3.0, 1.5, 3.5)
    assert 0.0 < h1 < h2 <= 1.0
    assert compute_hard_braking_cost(-3.5, 1.5, 3.5) == 1.0
    assert compute_hard_braking_cost(-6.0, 1.5, 3.5) == 1.0


def test_braking_share_formula():
    rows = [
        {
            "controller_id": "A",
            "active_on_road": True,
            "fixture_flag": False,
            "finite": True,
            "invalid_term_trunc": False,
            "gamma": 1.0,
            "policy_level_acceleration": -3.5,
            "progress_component": 0.1,
            "exit_component": 0.0,
            "collision_component": 0.0,
        }
    ]
    sh = episode_braking_shares(rows, a_comfort=1.5, a_hard=3.5, eta_h=0.02)
    H = 1.0
    B = 0.02 * H
    D = abs(0.1) + 0.02 * H
    assert abs(sh["A"] - B / D) < 1e-12


def test_complete_tuple_evaluation_and_no_threshold_first_selection():
    # Selection operates on complete tuples; threshold-only rows are not selected.
    rows = [
        {
            "feasible": True,
            "eta_H": 0.02,
            "H_separation": 0.3,
            "median_nominal_share": 0.04,
            "a_comfort": 1.5,
            "a_hard": 4.0,
        },
        {
            "feasible": True,
            "eta_H": 0.015,
            "H_separation": 0.25,
            "median_nominal_share": 0.05,
            "a_comfort": 2.0,
            "a_hard": 3.5,
        },
        {
            "feasible": False,
            "eta_H": 0.01,
            "H_separation": 0.9,
            "median_nominal_share": 0.03,
            "a_comfort": 2.5,
            "a_hard": 6.0,
        },
    ]
    sel = select_feasible_tuple(rows)
    assert sel is not None
    assert sel["eta_H"] == 0.015  # smallest feasible eta, not threshold-first


def test_selection_tie_break_order():
    rows = [
        {
            "feasible": True,
            "eta_H": 0.02,
            "H_separation": 0.40,
            "median_nominal_share": 0.04,
            "a_comfort": 1.5,
            "a_hard": 3.5,
        },
        {
            "feasible": True,
            "eta_H": 0.02,
            "H_separation": 0.50,
            "median_nominal_share": 0.05,
            "a_comfort": 1.5,
            "a_hard": 4.0,
        },
    ]
    sel = select_feasible_tuple(rows)
    assert sel["H_separation"] == 0.50
