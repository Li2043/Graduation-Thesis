"""Stage 4A-R1 holdout signature and lock-policy tests (real code paths)."""

from __future__ import annotations

from pathlib import Path

from thesis.certification.choice_state_scenarios import (
    GEOMETRY,
    build_environment_candidates,
    build_ic_blocks,
)
from thesis.certification.holdout_signatures import (
    assert_no_duplicate_holdout_for_geometries,
    audit_holdout_signatures_for_geometries,
    find_duplicate_signatures,
)
from thesis.certification.choice_state_scenarios import materialize_block_for_geometry
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.envs.final_route_geometry import build_final_route_geometry


def test_exactly_twelve_calibration_and_eight_validation_blocks():
    cal, val = build_ic_blocks()
    assert len(cal) == 12
    assert len(val) == 8


def test_validation_replacements_match_preregistered_specs():
    _, val = build_ic_blocks()
    by_id = {b.block_id: b for b in val}
    v1 = by_id["validation_001"]
    assert v1.spawn_speed_mainline == 20.0
    assert v1.spawn_speed_ramp == 20.0
    assert v1.delta_arrival == -0.4
    assert v1.background_time_headway == 1.8
    v6 = by_id["validation_006"]
    assert v6.spawn_speed_mainline == 16.0
    assert v6.spawn_speed_ramp == 18.0
    assert v6.delta_arrival == 0.0
    assert v6.background_time_headway == 1.2


def test_zero_duplicate_signatures_for_g1_g2_g3():
    cal, val = build_ic_blocks()
    audit = audit_holdout_signatures_for_geometries(cal, val, GEOMETRY)
    assert len(audit) == 3
    for row in audit:
        assert row["n_duplicate_signatures"] == 0
        assert row["pass"] is True
    assert_no_duplicate_holdout_for_geometries(cal, val, GEOMETRY)


def test_materialised_signature_audit_per_geometry_explicit():
    cal, val = build_ic_blocks()
    for geom in GEOMETRY:
        cm = [materialize_block_for_geometry(b, geom) for b in cal]
        vm = [materialize_block_for_geometry(b, geom) for b in val]
        assert find_duplicate_signatures(cm, vm) == []


def test_all_candidates_use_quintic_geometry():
    for cand in build_environment_candidates():
        geom = build_final_route_geometry(cand.geometry)
        d = geom.diagnostics()
        # Quintic connector: arc length slightly exceeds world-x length; heading ≪ π/2
        assert d["connector_arc_length"] > d["connector_world_x_length"]
        assert d["maximum_abs_heading"] < 0.2
        assert d["maximum_implied_lateral_acceleration_at_20"] <= 3.0 + 1e-12
        assert geom.max_route_recovery_error("ramp", n=500) <= 0.01
        assert abs(d["boundary_heading_jump_merge_start"]) < 1e-6
        assert abs(d["boundary_curvature_jump_merge_start"]) < 1e-6
        assert abs(d["boundary_heading_jump_merge_end"]) < 1e-6
        assert abs(d["boundary_curvature_jump_merge_end"]) < 1e-6


def test_observation_remains_27_dimensional():
    assert OBSERVATION_DIM == 27


def test_superseded_stage4a_lock_artifact_intact_and_not_overwritten():
    old = Path(
        "experiments/pre_impl/stage4a_environment_choice_state/artifacts/"
        "20260729T231946Z_c8d92bc3/final_environment_lock.sha256"
    )
    assert old.is_file()
    text = old.read_text(encoding="utf-8").strip()
    assert text.startswith("d5614d41d0c229db70b76973c55daa6905d7c5f07dc0781b81826b8891d76ded")
