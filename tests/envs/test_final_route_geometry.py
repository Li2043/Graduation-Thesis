"""Tests for quintic merge centreline geometry (Stage 4A-0R2)."""

from __future__ import annotations

import math

from thesis.certification.choice_state_scenarios import GEOMETRY
from thesis.envs.final_route_geometry import (
    MAX_LATERAL_ACCEL_AT_20,
    build_final_route_geometry,
    quintic_q,
    quintic_qp,
    quintic_qpp,
)


def test_quintic_boundary_derivatives():
    for u in (0.0, 1.0):
        assert abs(float(quintic_qp(u))) < 1e-12
        assert abs(float(quintic_qpp(u))) < 1e-12
    assert abs(float(quintic_q(0.0))) < 1e-12
    assert abs(float(quintic_q(1.0)) - 1.0) < 1e-12


def test_ramp_parallel_offset_before_merge_start():
    geom = build_final_route_geometry(GEOMETRY[0])
    for s in (0.0, 20.0, geom.merge_start - 1.0):
        p = geom.pose("ramp", s)
        m = geom.pose("mainline", s)
        assert abs(p.y + geom.lateral_offset) < 1e-12
        assert abs(m.y) < 1e-12
        assert abs(p.heading) < 1e-12
        assert abs(p.x - s) < 1e-12
        assert p.segment == "ramp_approach"


def test_convergence_starts_at_merge_start_and_ends_at_merge_end():
    geom = build_final_route_geometry(GEOMETRY[0])
    p0 = geom.pose("ramp", geom.merge_start)
    assert abs(p0.y + geom.lateral_offset) < 1e-9
    assert p0.segment in ("ramp_approach", "merge_connector")
    # Just inside connector
    p_in = geom.pose("ramp", geom.merge_start + 1e-3)
    assert p_in.segment == "merge_connector"
    assert p_in.y > -geom.lateral_offset
    # At end of connector arc (== merge_end in world-x)
    p1 = geom.pose("ramp", geom.ramp_connector_end_route)
    assert abs(p1.y) < 1e-9
    assert abs(p1.x - geom.merge_end) < 1e-6


def test_heading_and_curvature_continuous_at_boundaries():
    geom = build_final_route_geometry(GEOMETRY[0])
    eps = 1e-6
    for s_b in (geom.merge_start, geom.ramp_connector_end_route):
        a = geom.pose("ramp", s_b - eps)
        b = geom.pose("ramp", s_b + eps)
        assert abs(a.heading - b.heading) < 1e-5
        assert abs(a.curvature - b.curvature) < 1e-4
        assert math.hypot(a.x - b.x, a.y - b.y) < 1e-4


def test_route_monotonic_and_connector_lut_monotonic():
    geom = build_final_route_geometry(GEOMETRY[0])
    prev = -1.0
    for i in range(200):
        s = geom.exit_route("ramp") * i / 199
        p = geom.pose("ramp", s)
        assert p.route_position + 1e-12 >= prev
        prev = p.route_position
    prev_s = -1.0
    for i in range(100):
        u = i / 99
        sc = geom.connector_arc_from_u(u)
        assert sc + 1e-12 >= prev_s
        prev_s = sc
        assert abs(geom.u_from_connector_arc(sc) - u) < 1e-5


def test_inverse_recovery_error_bound_all_geometries():
    for g in GEOMETRY:
        geom = build_final_route_geometry(g)
        for role in ("mainline", "ramp"):
            err = geom.max_route_recovery_error(role, n=1000)
            assert err <= 0.01, (g.geometry_id, role, err)


def test_g1_g2_g3_distinct_connector_lengths():
    g1, g2, g3 = [build_final_route_geometry(g) for g in GEOMETRY]
    assert g1.connector_world_length == 60.0
    assert g2.connector_world_length == 80.0
    assert g3.connector_world_length == 70.0
    assert g2.connector_arc_length > g1.connector_arc_length
    assert abs(g3.merge_start - 100.0) < 1e-12
    assert abs(g3.merge_end - 170.0) < 1e-12


def test_lateral_acceleration_plausibility_bound():
    for g in GEOMETRY:
        geom = build_final_route_geometry(g)
        d = geom.diagnostics()
        assert d["physically_feasible"]
        assert d["maximum_implied_lateral_acceleration_at_20"] <= MAX_LATERAL_ACCEL_AT_20 + 1e-12
