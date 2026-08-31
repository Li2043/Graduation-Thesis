from __future__ import annotations

import pytest

from thesis.study_b.utility import (
    EpisodeVehicleTrace,
    coordination_burden,
    episode_burdens,
    episode_utilities,
    generalized_gini_welfare,
    gini_coefficient,
    utility_range,
)


def test_gini_all_equal_is_zero():
    assert gini_coefficient([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_gini_all_zero_is_na_not_zero():
    assert gini_coefficient([0.0, 0.0, 0.0, 0.0]) is None


def test_gini_one_has_everything_is_max_for_n4():
    # Known closed form: max Gini at n=4 is (n-1)/n = 0.75.
    assert gini_coefficient([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.75)


def test_ggi_sorts_ascending_before_weighting():
    # new_research_plan.md's own prescribed unit-test vector.
    utilities = [0.1, 0.2, 0.8, 1.0]
    assert generalized_gini_welfare(utilities) == pytest.approx(0.36)


def test_ggi_sorts_unsorted_input_correctly():
    # Same multiset as above, given out of order -- must sort internally.
    utilities = [1.0, 0.1, 0.8, 0.2]
    assert generalized_gini_welfare(utilities) == pytest.approx(0.36)


def test_ggi_rejects_mismatched_weight_length():
    with pytest.raises(ValueError):
        generalized_gini_welfare([0.1, 0.2, 0.3], weights=(0.4, 0.3, 0.2, 0.1))


def test_coordination_burden_basic():
    burden = coordination_burden(
        dt=0.2,
        active_flags=[True, True, True],
        attainments=[1.0, 0.5, 1.0],
    )
    assert burden == pytest.approx(0.2 * 0.5)


def test_coordination_burden_ignores_inactive_steps():
    burden = coordination_burden(
        dt=0.2,
        active_flags=[True, False, True],
        attainments=[0.0, 0.0, 1.0],
    )
    # Middle step (attainment 0.0, huge deficit) is inactive -- must not count.
    assert burden == pytest.approx(0.2 * 1.0)


def test_utility_range():
    assert utility_range([0.2, 0.9, 0.5]) == pytest.approx(0.7)


def test_episode_utilities_and_burdens_from_traces():
    slow = EpisodeVehicleTrace(
        vehicle_id="V0",
        target_speed=18.0,
        speeds=[18.0, 18.0, 18.0],
        active_flags=[True, True, True],
        collided=False,
    )
    fast = EpisodeVehicleTrace(
        vehicle_id="V1",
        target_speed=22.0,
        speeds=[11.0, 22.0, 22.0],
        active_flags=[True, True, True],
        collided=False,
    )
    traces = {"V0": slow, "V1": fast}

    utilities = episode_utilities(traces)
    # slow vehicle stayed exactly at its own target the whole time -> U=1.0
    assert utilities["V0"] == pytest.approx(1.0)
    # fast vehicle: attainments = [0.5, 1.0, 1.0] -> mean = 0.8333...
    assert utilities["V1"] == pytest.approx(2.5 / 3.0)

    burdens = episode_burdens(traces, dt=0.2)
    assert burdens["V0"] == pytest.approx(0.0)
    assert burdens["V1"] == pytest.approx(0.2 * (0.5 + 0.0 + 0.0))


def test_hard_brake_count():
    trace = EpisodeVehicleTrace(
        vehicle_id="V0", target_speed=20.0,
        accelerations=[-3.5, -3.6, -1.0, 0.0, -3.5, 2.0],
    )
    assert trace.hard_brake_count() == 3  # -3.5, -3.6, -3.5 all <= -3.5


def test_episode_utilities_zero_for_collision():
    collided = EpisodeVehicleTrace(
        vehicle_id="V0",
        target_speed=20.0,
        speeds=[20.0, 20.0],
        active_flags=[True, True],
        collided=True,
    )
    utilities = episode_utilities({"V0": collided})
    assert utilities["V0"] == pytest.approx(0.0)
