"""Locality (finite local sensing range) amendment tests -- adopted into
the active tree 2026-08-17 (RANGE_STATUS=FROZEN, R=50m; see
LOCAL_SENSING_AMENDMENT.md). R values used in individual test fixtures
below (e.g. 20.0) are arbitrary, chosen purely to exercise the masking
mechanism in isolation from the frozen scientific value -- they do not
imply any change to the frozen R=50m sensing range used in actual
evaluation/training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from thesis.study_b.local_observation import (  # noqa: E402
    NEIGHBOUR_SLOTS,
    VehicleSnapshot,
    build_local_observation,
    build_neighbour_observations,
)

MERGE_START = 300.0


def _snap(vid: str, route_position: float, *, role: str = "ramp", speed: float = 20.0,
          target_speed: float = 20.0, active: bool = True) -> VehicleSnapshot:
    return VehicleSnapshot(
        vehicle_id=vid, role=role, speed=speed, route_position=route_position,
        acceleration=0.0, target_speed=target_speed, active=active,
    )


def _presence(rows: np.ndarray) -> list[float]:
    return [float(r[0]) for r in rows]


# ---- TEST A: all three neighbours inside range -> all three visible ----
def test_A_all_three_inside_range_all_visible():
    ego = _snap("ego", route_position=100.0)
    others = [_snap("a", 105.0), _snap("b", 110.0), _snap("c", 115.0)]
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=20.0)
    assert _presence(rows) == [1.0, 1.0, 1.0]


# ---- TEST B: two inside, one outside -> exactly two visible, third masked ----
def test_B_two_inside_one_outside():
    ego = _snap("ego", route_position=100.0)
    others = [_snap("a", 105.0), _snap("b", 110.0), _snap("far", 500.0)]
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=20.0)
    assert sorted(_presence(rows)) == [0.0, 1.0, 1.0]


# ---- TEST C: one inside, two outside -> exactly one visible ----
def test_C_one_inside_two_outside():
    ego = _snap("ego", route_position=100.0)
    others = [_snap("a", 105.0), _snap("far1", 500.0), _snap("far2", 600.0)]
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=20.0)
    assert sorted(_presence(rows)) == [0.0, 0.0, 1.0]


# ---- TEST D: all outside -> all neighbour slots masked ----
def test_D_all_outside_all_masked():
    ego = _snap("ego", route_position=100.0)
    others = [_snap("far1", 500.0), _snap("far2", 600.0), _snap("far3", 700.0)]
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=20.0)
    assert _presence(rows) == [0.0, 0.0, 0.0]


# ---- TEST E: boundary, distance exactly R -> inclusive, visible ----
def test_E_boundary_exactly_R_is_visible():
    ego = _snap("ego", route_position=100.0)
    R = 20.0
    others = [_snap("boundary", 100.0 + R)]  # |delta_d| == R exactly
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=R)
    assert _presence(rows)[0] == 1.0, "documented rule is inclusive: abs(delta_d) <= R must be visible"


# ---- TEST F: just outside boundary, R + epsilon -> invisible ----
def test_F_just_outside_boundary_invisible():
    ego = _snap("ego", route_position=100.0)
    R = 20.0
    others = [_snap("just_outside", 100.0 + R + 1e-6)]
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=R)
    assert _presence(rows)[0] == 0.0


# ---- TEST G: neighbour ordering among visible remains nearest-first, deterministic ----
def test_G_nearest_first_ordering_among_visible():
    ego = _snap("ego", route_position=100.0)
    others = [_snap("mid", 112.0), _snap("near", 103.0), _snap("far_in_range", 118.0)]
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=20.0)
    # delta_d_norm (index 1) should be monotonically non-decreasing in |delta_d| across slots
    abs_deltas = [abs(r[1]) for r in rows if r[0] == 1.0]
    assert abs_deltas == sorted(abs_deltas)


# ---- TEST H: target-speed privacy preserved under the amendment ----
def test_H_target_speed_privacy_preserved():
    ego = _snap("ego", route_position=100.0)
    other_a = _snap("a", 105.0, target_speed=18.0)
    other_b = _snap("a", 105.0, target_speed=30.0)  # same id/physical state, different hidden target_speed
    rows_a = build_neighbour_observations(ego, [other_a], merge_start=MERGE_START, local_sensing_range_m=20.0)
    rows_b = build_neighbour_observations(ego, [other_b], merge_start=MERGE_START, local_sensing_range_m=20.0)
    np.testing.assert_array_equal(rows_a, rows_b)


# ---- TEST I: no global fallback -- farther vehicles never fill remaining slots ----
def test_I_no_global_fallback_to_farther_vehicles():
    ego = _snap("ego", route_position=100.0)
    others = [_snap("near", 105.0), _snap("far1", 500.0), _snap("far2", 600.0)]
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=20.0)
    assert _presence(rows) == [1.0, 0.0, 0.0]
    # explicit: the two masked rows must be all-zero, not populated with far1/far2's data
    for row in rows[1:]:
        np.testing.assert_array_equal(row, np.zeros(4))


# ---- TEST J: N=4 locality property -- ego can observe FEWER than 3 neighbours ----
def test_J_n4_locality_property_fewer_than_three_visible():
    """With exactly 4 total vehicles (this project's frozen N), demonstrate
    that under the amendment an ego can see fewer than NEIGHBOUR_SLOTS=3
    others -- the property the OLD (unbounded) implementation could never
    exhibit at N=4 (see CURRENT_OBSERVATION_AUDIT.md sec 6)."""
    ego = _snap("V0", route_position=100.0)
    others = [_snap("V1", 105.0), _snap("V2", 500.0), _snap("V3", 600.0)]  # N=4 total
    assert len(others) == NEIGHBOUR_SLOTS  # sanity: this project's real N-1
    rows_old = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=None)
    rows_new = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=20.0)
    assert sum(_presence(rows_old)) == 3.0, "old behaviour: all N-1=3 always visible"
    assert sum(_presence(rows_new)) == 1.0, "new behaviour: only in-range vehicles visible, can be < 3"


# ---- TEST K: deterministic -- same inputs give identical observation ----
def test_K_deterministic_repeat_call():
    ego = _snap("ego", route_position=100.0)
    others = [_snap("a", 105.0), _snap("b", 110.0), _snap("c", 500.0)]
    obs1 = build_local_observation(ego, others, merge_start=MERGE_START, local_sensing_range_m=20.0)
    obs2 = build_local_observation(ego, others, merge_start=MERGE_START, local_sensing_range_m=20.0)
    np.testing.assert_array_equal(obs1, obs2)


# ---- Backward-compatibility sanity: default (None) is byte-identical to pre-amendment behaviour ----
def test_default_none_preserves_old_behaviour_exactly():
    ego = _snap("ego", route_position=100.0)
    others = [_snap("a", 105.0), _snap("b", 500.0), _snap("c", 600.0)]
    rows_default = build_neighbour_observations(ego, others, merge_start=MERGE_START)  # local_sensing_range_m omitted
    rows_explicit_none = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=None)
    np.testing.assert_array_equal(rows_default, rows_explicit_none)
    assert _presence(rows_default) == [1.0, 1.0, 1.0], "old behaviour: all active others visible regardless of distance"


# ---- TEST M: sanity check on the frozen scientific value R=50m itself ----
def test_M_frozen_r50_boundary_sanity():
    ego = _snap("ego", route_position=100.0)
    others = [_snap("a", 130.0), _snap("b", 160.0), _snap("c", 200.0)]  # 30/60/100m away
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START, local_sensing_range_m=50.0)
    assert _presence(rows) == [1.0, 0.0, 0.0], "at frozen R=50m: only the 30m neighbour is visible; 60m/100m are masked"
