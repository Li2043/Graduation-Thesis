"""Tests for candidate selection rules (calibration-only; no reselection)."""

from __future__ import annotations

from thesis.certification.environment_candidate_selection import (
    calibration_feasible,
    validation_pass,
)
from thesis.certification.choice_state_scenarios import build_environment_candidates


def test_candidate_priority_order_preregistered():
    cands = build_environment_candidates()
    assert [c.candidate_id for c in cands] == [
        "G1-I1",
        "G1-I2",
        "G1-I3",
        "G2-I1",
        "G2-I2",
        "G2-I3",
        "G3-I1",
        "G3-I2",
        "G3-I3",
    ]
    assert [c.priority_rank for c in cands] == list(range(1, 10))


def test_calibration_feasible_thresholds():
    ok_eval = {
        "n_certified": 11,
        "certified_arrival_categories": ["mainline_lead", "ramp_lead", "near_simultaneous"],
        "background_relevance_rate": 0.8,
        "median_normalised_order_gap": 0.04,
        "maximum_normalised_order_gap": 0.09,
        "label_swap_max_error": 0.0,
    }
    ok, reasons = calibration_feasible(
        ok_eval,
        bg_safety={"spontaneous_collision_count": 0},
        integrity={
            "route_discontinuity_count": 0,
            "repeated_exit_count": 0,
            "invalid_flag_count": 0,
            "nan_inf_count": 0,
            "fixture_count": 0,
        },
    )
    assert ok and not reasons

    bad = dict(ok_eval)
    bad["n_certified"] = 10
    ok2, reasons2 = calibration_feasible(
        bad,
        bg_safety={"spontaneous_collision_count": 0},
        integrity={
            "route_discontinuity_count": 0,
            "repeated_exit_count": 0,
            "invalid_flag_count": 0,
            "nan_inf_count": 0,
            "fixture_count": 0,
        },
    )
    assert not ok2


def test_holdout_failure_does_not_imply_reselection_flag():
    # validation_pass is independent; selection logic never re-picks using validation
    eval_result = {
        "n_certified": 6,
        "background_relevance_rate": 0.9,
        "median_normalised_order_gap": 0.02,
        "maximum_normalised_order_gap": 0.05,
        "label_swap_max_error": 0.0,
        "certifications": [],
    }
    ok, reasons = validation_pass(
        eval_result,
        bg_safety={"spontaneous_collision_count": 0},
        integrity={
            "route_discontinuity_count": 0,
            "repeated_exit_count": 0,
            "invalid_flag_count": 0,
            "nan_inf_count": 0,
            "fixture_count": 0,
        },
    )
    assert not ok
    assert any("certified" in r for r in reasons)


def test_comfort_excluded_from_feasibility_keys():
    # Feasibility helpers accept only core metrics — no eta / H / comfort keys required
    ok_eval = {
        "n_certified": 11,
        "certified_arrival_categories": ["mainline_lead", "ramp_lead", "near_simultaneous"],
        "background_relevance_rate": 0.75,
        "median_normalised_order_gap": 0.05,
        "maximum_normalised_order_gap": 0.10,
        "label_swap_max_error": 0.0,
    }
    ok, _ = calibration_feasible(
        ok_eval,
        bg_safety={"spontaneous_collision_count": 0},
        integrity={
            "route_discontinuity_count": 0,
            "repeated_exit_count": 0,
            "invalid_flag_count": 0,
            "nan_inf_count": 0,
            "fixture_count": 0,
        },
    )
    assert ok
    assert "eta" not in ok_eval and "hard_brake" not in ok_eval
