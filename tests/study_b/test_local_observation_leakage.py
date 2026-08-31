from __future__ import annotations

import numpy as np
import pytest

from thesis.study_b.local_observation import (
    LOCAL_OBS_DIM,
    NEIGHBOUR_OBS_DIM,
    NEIGHBOUR_SLOTS,
    SELF_OBS_DIM,
    VehicleSnapshot,
    build_local_observation,
    build_neighbour_observations,
    build_self_observation,
    distance_to_merge,
)

MERGE_START = 200.0


def _ego() -> VehicleSnapshot:
    return VehicleSnapshot(
        vehicle_id="V0",
        role="ramp",
        speed=18.0,
        route_position=150.0,
        acceleration=0.0,
        target_speed=18.0,
        active=True,
    )


def _others(v1_target: float) -> list[VehicleSnapshot]:
    return [
        VehicleSnapshot(
            vehicle_id="V1",
            role="mainline",
            speed=20.0,
            route_position=160.0,
            acceleration=0.5,
            target_speed=v1_target,
            active=True,
        ),
        VehicleSnapshot(
            vehicle_id="V2",
            role="ramp",
            speed=18.0,
            route_position=90.0,
            acceleration=0.0,
            target_speed=18.0,
            active=True,
        ),
        VehicleSnapshot(
            vehicle_id="V3",
            role="mainline",
            speed=22.0,
            route_position=110.0,
            acceleration=0.0,
            target_speed=22.0,
            active=True,
        ),
    ]


def test_local_obs_dim_matches_constant():
    ego = _ego()
    obs = build_local_observation(ego, _others(18.0), merge_start=MERGE_START)
    assert obs.shape == (LOCAL_OBS_DIM,)
    assert LOCAL_OBS_DIM == SELF_OBS_DIM + NEIGHBOUR_SLOTS * NEIGHBOUR_OBS_DIM


def test_changing_neighbour_private_target_speed_does_not_change_ego_observation():
    """The core leakage test new_research_plan.md's Phase 0 checklist asks
    for: keep V1's external motion (speed, position) fixed, change ONLY its
    private v_target from 18 to 22, and confirm V0's observation is
    bit-identical."""
    ego = _ego()
    obs_18 = build_local_observation(ego, _others(v1_target=18.0), merge_start=MERGE_START)
    obs_22 = build_local_observation(ego, _others(v1_target=22.0), merge_start=MERGE_START)
    np.testing.assert_array_equal(obs_18, obs_22)


def test_ego_self_observation_does_reveal_its_own_target_speed():
    # Sanity companion to the leakage test: the restriction is specifically
    # about OTHER vehicles, not the ego's own state.
    ego_18 = VehicleSnapshot(
        vehicle_id="V0", role="ramp", speed=18.0, route_position=150.0,
        acceleration=0.0, target_speed=18.0, active=True,
    )
    ego_22 = VehicleSnapshot(
        vehicle_id="V0", role="ramp", speed=18.0, route_position=150.0,
        acceleration=0.0, target_speed=22.0, active=True,
    )
    self_obs_18 = build_self_observation(ego_18, merge_start=MERGE_START)
    self_obs_22 = build_self_observation(ego_22, merge_start=MERGE_START)
    assert not np.array_equal(self_obs_18, self_obs_22)


def test_neighbours_sorted_nearest_first():
    ego = _ego()
    rows = build_neighbour_observations(ego, _others(18.0), merge_start=MERGE_START)
    # ego at route_position=150 -> d_ego = 50. V1 at 160 -> d=40, |delta|=10.
    # V2 at 90 -> d=110, |delta|=60. V3 at 110 -> d=90, |delta|=40.
    # Expected order by |delta_d| ascending: V1 (10), V3 (40), V2 (60).
    assert rows.shape == (NEIGHBOUR_SLOTS, NEIGHBOUR_OBS_DIM)
    assert rows[0, 0] == pytest.approx(1.0)  # presence
    # V1 is nearest -> its lane_relation is -1.0 (mainline vs ego ramp)
    assert rows[0, 3] == pytest.approx(-1.0)
    # V2 (ramp, same lane as ego) should be last (farthest).
    assert rows[2, 3] == pytest.approx(1.0)


def test_inactive_neighbours_are_excluded_and_masked():
    ego = _ego()
    others = _others(18.0)
    others[0] = VehicleSnapshot(
        vehicle_id="V1", role="mainline", speed=20.0, route_position=160.0,
        acceleration=0.5, target_speed=18.0, active=False,
    )
    rows = build_neighbour_observations(ego, others, merge_start=MERGE_START)
    # Only 2 active others remain (V2, V3) -> last slot must be masked (presence=0).
    assert rows[2, 0] == pytest.approx(0.0)
    assert np.array_equal(rows[2], np.zeros(NEIGHBOUR_OBS_DIM))


def test_relative_distance_and_speed_normalization_clips_to_unit_range():
    ego = _ego()
    far_fast = [
        VehicleSnapshot(
            vehicle_id="V1", role="mainline", speed=40.0, route_position=1000.0,
            acceleration=0.0, target_speed=20.0, active=True,
        ),
    ]
    rows = build_neighbour_observations(ego, far_fast, merge_start=MERGE_START, k=1)
    assert -1.0 <= rows[0, 1] <= 1.0
    assert -1.0 <= rows[0, 2] <= 1.0


def test_distance_to_merge_positive_before_merge_start():
    assert distance_to_merge(150.0, merge_start=200.0) == pytest.approx(50.0)
    assert distance_to_merge(250.0, merge_start=200.0) == pytest.approx(-50.0)
