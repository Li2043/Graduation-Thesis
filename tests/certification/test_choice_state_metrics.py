"""Tests for Stage 4A choice-state metrics and macros."""

from __future__ import annotations

from thesis.certification.choice_state_metrics import (
    CellOutcome,
    background_meaningful,
    classify_exit_order,
    core_ordering_ok,
    go_go_problematic,
    no_unilateral_guarantee,
    normalised_order_gap,
    yield_yield_inefficient,
)
from thesis.certification.choice_state_scenarios import (
    GO_PROFILES,
    YIELD_PROFILES,
    build_ic_blocks,
    cell_kinds,
    expand_label_assignments,
    least_intervention_profile,
    macro_action_sequence,
)


def _cell(**kwargs) -> CellOutcome:
    base = dict(
        cell="GO_GO",
        success=True,
        collision=False,
        truncated=False,
        exit_order="mainline_first",
        G_team_core=1.0,
        G_A_core=0.5,
        G_B_core=0.5,
        episode_length=50,
        exit_time_mainline=40,
        exit_time_ramp=45,
        min_bumper_gap=3.0,
        min_ttc=2.0,
        min_accel=-2.0,
        max_accel=2.0,
        route_discontinuity=0,
        nan_count=0,
        invalid_flags=0,
        repeated_exit=0,
        fixture_count=0,
        bg_min_speed={"B_front": 18.0, "B_rear": 18.0},
        bg_max_brake={"B_front": 0.5, "B_rear": 0.5},
        bg_min_gap_to_learners=10.0,
    )
    base.update(kwargs)
    return CellOutcome(**base)


def test_exit_order_and_order_gap():
    assert classify_exit_order(10, 20) == "mainline_first"
    assert classify_exit_order(20, 10) == "ramp_first"
    og = normalised_order_gap(1.0, 1.05)
    assert og["normalised_order_gap"] < 0.1


def test_go_go_and_yield_yield_and_unilateral():
    ml = _cell(cell="GO_YIELD", G_team_core=1.5, episode_length=60)
    rp = _cell(cell="YIELD_GO", exit_order="ramp_first", G_team_core=1.4, episode_length=62)
    gg = _cell(cell="GO_GO", collision=True, success=False, G_team_core=-1.5)
    yy = _cell(cell="YIELD_YIELD", G_team_core=0.8, episode_length=90)
    assert go_go_problematic(gg, ml, rp)
    assert yield_yield_inefficient(yy, ml, rp)
    matrix = {
        "GO_GO": gg,
        "GO_YIELD": ml,
        "YIELD_GO": rp,
        "YIELD_YIELD": yy,
    }
    assert no_unilateral_guarantee(matrix)


def test_core_ordering_unsafe_gogo_does_not_require_yy_dominate_progress():
    """Near-miss GO/GO may keep high progress return; ordering still holds via YY < asymmetric."""
    ml = _cell(cell="GO_YIELD", G_team_core=1.53, episode_length=64)
    rp = _cell(cell="YIELD_GO", exit_order="ramp_first", G_team_core=1.52, episode_length=71)
    yy = _cell(cell="YIELD_YIELD", G_team_core=1.45, episode_length=73)
    gg_unsafe = _cell(
        cell="GO_GO",
        G_team_core=1.60,
        episode_length=44,
        min_bumper_gap=0.13,
        success=True,
        collision=False,
    )
    assert go_go_problematic(gg_unsafe, ml, rp)
    assert core_ordering_ok(ml, rp, yy, gg_unsafe)
    gg_safe = _cell(cell="GO_GO", G_team_core=1.60, episode_length=44, min_bumper_gap=3.0)
    assert not core_ordering_ok(ml, rp, yy, gg_safe)


def test_background_meaningful():
    a = _cell(bg_min_speed={"B_front": 10.0, "B_rear": 18.0})
    b = _cell(bg_min_speed={"B_front": 18.0, "B_rear": 18.0})
    assert background_meaningful(a, b)


def test_macro_generation_cells():
    assert cell_kinds("GO_YIELD") == ("GO", "YIELD")
    assert cell_kinds("YIELD_GO") == ("YIELD", "GO")
    go = least_intervention_profile(GO_PROFILES)
    yld = least_intervention_profile(YIELD_PROFILES)
    gy = macro_action_sequence("GO", "YIELD", go=go, yield_p=yld, total_steps=20, role_A="mainline")
    yg = macro_action_sequence("YIELD", "GO", go=go, yield_p=yld, total_steps=20, role_A="mainline")
    gg = macro_action_sequence("GO", "GO", go=go, yield_p=yld, total_steps=20, role_A="mainline")
    yy = macro_action_sequence("YIELD", "YIELD", go=go, yield_p=yld, total_steps=20, role_A="mainline")
    assert gy[0]["A"] == 1 and gy[0]["B"] == 2
    assert yg[0]["A"] == 2 and yg[0]["B"] == 1
    assert gg[0]["A"] == 1 and gg[0]["B"] == 1
    assert yy[0]["A"] == 2 and yy[0]["B"] == 2


def test_label_swap_physical_ic_identical():
    cal, _ = build_ic_blocks()
    a1, a2 = expand_label_assignments(cal[0])
    assert a1.spawn_route_mainline == a2.spawn_route_mainline
    assert a1.role_A != a2.role_A


def test_calibration_validation_separated():
    cal, val = build_ic_blocks()
    assert len(cal) == 12 and len(val) == 8
    assert {b.block_id for b in cal}.isdisjoint({b.block_id for b in val})
    assert all(b.block_set == "calibration" for b in cal)
    assert all(b.block_set == "validation" for b in val)
