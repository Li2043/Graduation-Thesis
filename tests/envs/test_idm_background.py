"""Tests for deterministic bounded IDM background controller."""

from __future__ import annotations

import math

import pytest

from thesis.envs.final_environment_config import IDMProfile
from thesis.envs.idm_background import IDMState, idm_acceleration, idm_command, validate_idm_profile


def _profile(**kwargs) -> IDMProfile:
    base = dict(
        profile_id="I1",
        desired_speed=20.0,
        minimum_gap=2.0,
        desired_time_headway=1.5,
        maximum_acceleration=1.5,
        comfortable_deceleration=2.0,
        acceleration_exponent=4.0,
        priority_rank=1,
        maximum_emergency_deceleration=6.0,
    )
    base.update(kwargs)
    return IDMProfile(**base)


def test_idm_parameter_validation():
    validate_idm_profile(_profile())
    with pytest.raises(ValueError):
        validate_idm_profile(_profile(desired_speed=0.0))


def test_idm_finite_acceleration():
    a = idm_acceleration(_profile(), IDMState(speed=10.0, gap=20.0, leader_speed=10.0))
    assert math.isfinite(a)


def test_idm_no_leader_free_road():
    a = idm_acceleration(_profile(), IDMState(speed=10.0, gap=None, leader_speed=None))
    assert a > 0.0


def test_idm_response_to_closing_gap():
    free = idm_acceleration(_profile(), IDMState(speed=20.0, gap=None, leader_speed=None))
    close = idm_acceleration(_profile(), IDMState(speed=20.0, gap=5.0, leader_speed=10.0))
    assert close < free
    assert close < 0.0


def test_idm_bounded_for_tiny_positive_gap():
    cmd = idm_command(_profile(), IDMState(speed=20.0, gap=1e-6, leader_speed=0.0))
    assert -6.0 - 1e-12 <= cmd.commanded_acceleration <= 1.5 + 1e-12
    assert math.isfinite(cmd.commanded_acceleration)


def test_idm_emergency_bound_on_nonpositive_gap():
    cmd = idm_command(_profile(), IDMState(speed=15.0, gap=0.0, leader_speed=15.0))
    assert abs(cmd.commanded_acceleration - (-6.0)) < 1e-12
