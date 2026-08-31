"""Validation / reselection policy tests (Stage 3B-R1)."""

from __future__ import annotations

from thesis.calibration.comfort_validation import validate_selected_tuple
from thesis.calibration.joint_comfort_calibration import TraceBundle


def test_validation_failure_does_not_trigger_reselection():
    bundle = TraceBundle(
        integrity={
            "route_discontinuity_count": 0,
            "repeated_exit_count": 0,
            "invalid_flag_count": 0,
            "fixture_count": 0,
            "nan_inf_count": 0,
            "missing_substep_acceleration_count": 0,
        }
    )
    # Empty validation content => not enough usable blocks
    out = validate_selected_tuple(bundle, a_comfort=1.5, a_hard=3.5, eta_h=0.02)
    assert out["reselection_triggered"] is False
    assert out["selection_used_validation"] is False
    assert out["pass"] is False
