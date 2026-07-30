"""Unit tests for policy-level braking acceleration (Stage 3B-R1)."""

from __future__ import annotations

from thesis.calibration.policy_acceleration import braking_magnitude, policy_braking_acceleration


def test_policy_acceleration_uses_most_negative_substep():
    assert policy_braking_acceleration([0.0, -1.0, -3.0, 0.0]) == -3.0


def test_final_substep_zero_does_not_hide_earlier_brake():
    assert policy_braking_acceleration([-2.8, -2.5, -2.0, 0.0]) == -2.8


def test_inactive_none_substeps_excluded():
    assert policy_braking_acceleration([None, None, None, None]) == 0.0
    assert policy_braking_acceleration([None, -3.0, None, 0.0]) == -3.0


def test_braking_magnitude():
    assert braking_magnitude(-3.0) == 3.0
    assert braking_magnitude(1.0) == 0.0
