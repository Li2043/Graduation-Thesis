"""Tests for exact stopping and SAT collision helpers."""

from __future__ import annotations

import math

from thesis.envs.vehicle_dynamics import (
    integrate_longitudinal,
    oriented_rectangles_collide,
    rectangles_overlap_sat,
    vehicle_rectangle_corners,
)


def test_exact_stopping_distance():
    v0 = 0.2
    a_cmd = -5.0
    dt = 0.05
    t_stop = v0 / (-a_cmd)
    assert t_stop < dt
    s_expected = 0.0 + v0 * t_stop + 0.5 * a_cmd * t_stop * t_stop
    s1, v1, a_real = integrate_longitudinal(
        route_position=0.0, speed=v0, acceleration=a_cmd, dt=dt
    )
    assert v1 == 0.0
    assert abs(s1 - s_expected) < 1e-12
    assert abs(a_real - (0.0 - v0) / dt) < 1e-12


def test_speed_bound_realised_acceleration():
    s1, v1, a_real = integrate_longitudinal(
        route_position=0.0, speed=29.95, acceleration=2.0, dt=0.05, v_max=30.0
    )
    assert abs(v1 - 30.0) < 1e-12
    assert abs(a_real - (30.0 - 29.95) / 0.05) < 1e-12
    assert a_real < 2.0


def test_no_stop_when_t_stop_exceeds_dt():
    s1, v1, a_real = integrate_longitudinal(
        route_position=10.0, speed=10.0, acceleration=-2.0, dt=0.05
    )
    assert v1 > 0
    assert abs(v1 - (10.0 - 2.0 * 0.05)) < 1e-12


def test_sat_detects_overlap_and_separation():
    c1 = vehicle_rectangle_corners(0.0, 0.0, 0.0, length=5.0, width=2.0)
    c2 = vehicle_rectangle_corners(1.0, 0.0, 0.0, length=5.0, width=2.0)
    assert rectangles_overlap_sat(c1, c2)
    c3 = vehicle_rectangle_corners(20.0, 0.0, 0.0, length=5.0, width=2.0)
    assert not rectangles_overlap_sat(c1, c3)


def test_oriented_merge_conflict_collision():
    # Two vehicles in merge conflict region with overlapping footprints
    assert oriented_rectangles_collide(
        x1=100.0,
        y1=0.0,
        heading1=0.0,
        x2=101.0,
        y2=0.2,
        heading2=0.1,
        length=5.0,
        width=2.0,
    )
    assert not oriented_rectangles_collide(
        x1=100.0,
        y1=0.0,
        heading1=0.0,
        x2=120.0,
        y2=0.0,
        heading2=0.0,
        length=5.0,
        width=2.0,
    )
