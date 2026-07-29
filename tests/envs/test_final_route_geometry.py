"""Tests for arc-length final route geometry."""

from __future__ import annotations

import math

from thesis.certification.choice_state_scenarios import GEOMETRY
from thesis.envs.final_route_geometry import build_final_route_geometry


def test_round_trip_mainline_and_ramp():
    geom = build_final_route_geometry(GEOMETRY[0])
    for role in ("mainline", "ramp"):
        for frac in (0.0, 0.2, 0.5, 0.8, 0.99):
            s = frac * geom.exit_route(role)
            pose = geom.pose(role, s)
            s_rec = geom.recover_route_position(role, pose.x, pose.y)
            assert abs(s_rec - s) < 0.05


def test_heading_and_position_continuity_at_join():
    geom = build_final_route_geometry(GEOMETRY[0])
    # Just before and at join on ramp
    s0 = geom.ramp_join_route - 1e-6
    s1 = geom.ramp_join_route
    p0 = geom.pose("ramp", s0)
    p1 = geom.pose("ramp", s1)
    assert abs(p0.x - p1.x) < 1e-3
    assert abs(p0.y - p1.y) < 1e-3
    assert abs(p0.heading - p1.heading) < 1e-3
    # Mainline at join
    pm = geom.pose("mainline", geom.join_x)
    assert abs(pm.y) < 1e-12
    assert abs(pm.heading) < 1e-12


def test_segment_labels():
    geom = build_final_route_geometry(GEOMETRY[0])
    assert geom.segment("mainline", 10.0) == "mainline_approach"
    assert geom.segment("mainline", 100.0) == "merge_conflict"
    assert geom.segment("mainline", 200.0) == "shared_mainline"
    assert geom.segment("ramp", 5.0) == "ramp_approach"
    assert geom.segment("ramp", geom.ramp_straight_length + 1.0) == "merge_connector"


def test_heading_continuity_max_jump():
    geom = build_final_route_geometry(GEOMETRY[0])
    samples = geom.heading_continuity_samples("ramp", n=500)
    max_jump = 0.0
    prev = samples[0][1]
    for s, h, _ in samples[1:]:
        # unwrap small
        dh = abs(h - prev)
        dh = min(dh, abs(dh - 2 * math.pi))
        max_jump = max(max_jump, dh)
        prev = h
    # Arc connector: Δθ ≈ Δs / R; with ~1.3 m samples and R=4, expect ~0.33 rad.
    assert max_jump < 0.40
