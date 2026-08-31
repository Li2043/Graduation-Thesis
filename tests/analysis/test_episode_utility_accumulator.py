"""Unit tests for Stage 6B-H1 trajectory utility accumulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thesis.analysis.episode_utility_accumulator import (
    clip_speed_attainment,
    collect_active_state_attainment,
    finalise_episode_utilities,
    initialise_episode_utility_accumulator,
)


def test_trajectory_mean_not_final_state() -> None:
    acc = {"A": [0.2, 0.4, 1.0]}
    out = finalise_episode_utilities(accumulator=acc, collided_stakeholder_ids=[])
    assert out["A"] == pytest.approx(0.5333333333333333)
    assert out["A"] != pytest.approx(1.0)


def test_exit_absorbing_ones_not_included() -> None:
    # Real active samples only; post-exit absorbing 1.0 must not be appended.
    acc = {"A": [0.2, 0.4]}
    out = finalise_episode_utilities(accumulator=acc, collided_stakeholder_ids=[])
    assert out["A"] == pytest.approx(0.3)


def test_collision_override_to_zero() -> None:
    acc = {"A": [0.7, 0.8, 0.9]}
    out = finalise_episode_utilities(accumulator=acc, collided_stakeholder_ids={"A"})
    assert out["A"] == 0.0


def test_non_colliding_unaffected_by_other_collision() -> None:
    acc = {"A": [0.7, 0.8], "B": [0.2, 0.4]}
    out = finalise_episode_utilities(accumulator=acc, collided_stakeholder_ids={"A"})
    assert out["A"] == 0.0
    assert out["B"] == pytest.approx(0.3)


def test_collect_skips_inactive_and_completed() -> None:
    acc = initialise_episode_utility_accumulator(("A", "B"))
    vehicles = {
        "A": {"speed": 10.0, "target_speed": 20.0, "active_on_road": True, "completed": False},
        "B": {"speed": 20.0, "target_speed": 20.0, "active_on_road": False, "completed": True},
    }
    collect_active_state_attainment(
        vehicles=vehicles,
        stakeholder_ids=("A", "B"),
        accumulator=acc,
    )
    assert acc["A"] == [0.5]
    assert acc["B"] == []


def test_initial_state_sampling_before_first_step() -> None:
    """Mock: first collect happens before any step counter increment."""
    steps = {"n": 0}
    acc = initialise_episode_utility_accumulator(("A",))

    def fake_reset_vehicles():
        return {
            "A": {
                "speed": 10.0,
                "target_speed": 20.0,
                "active_on_road": True,
                "completed": False,
            }
        }

    # s0
    assert steps["n"] == 0
    collect_active_state_attainment(
        vehicles=fake_reset_vehicles(),
        stakeholder_ids=("A",),
        accumulator=acc,
    )
    steps["n"] += 1  # first env.step
    assert len(acc["A"]) == 1
    assert acc["A"][0] == pytest.approx(0.5)


def test_terminal_absorbing_not_sampled() -> None:
    acc = initialise_episode_utility_accumulator(("A",))
    active = {
        "A": {"speed": 10.0, "target_speed": 20.0, "active_on_road": True, "completed": False}
    }
    absorb = {
        "A": {"speed": 0.0, "target_speed": 20.0, "active_on_road": False, "completed": True}
    }
    collect_active_state_attainment(vehicles=active, stakeholder_ids=("A",), accumulator=acc)
    # After terminal: do not collect absorbing state
    terminated = True
    if not terminated:
        collect_active_state_attainment(vehicles=absorb, stakeholder_ids=("A",), accumulator=acc)
    out = finalise_episode_utilities(accumulator=acc, collided_stakeholder_ids=[])
    assert out["A"] == pytest.approx(0.5)


def test_truncation_does_not_count_as_collision() -> None:
    acc = {"A": [0.4, 0.6]}
    out = finalise_episode_utilities(accumulator=acc, collided_stakeholder_ids=[])
    assert out["A"] == pytest.approx(0.5)


def test_empty_samples_non_colliding_raises() -> None:
    with pytest.raises(RuntimeError, match="No valid active-state"):
        finalise_episode_utilities(accumulator={"A": []}, collided_stakeholder_ids=[])


@pytest.mark.parametrize("vt", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_target_speed_raises(vt: float) -> None:
    with pytest.raises(ValueError):
        clip_speed_attainment(10.0, vt)


def test_clip_below_zero_and_above_one() -> None:
    assert clip_speed_attainment(-1.0, 10.0) == 0.0
    assert clip_speed_attainment(20.0, 10.0) == 1.0


def test_float64_precision() -> None:
    samples = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    out = finalise_episode_utilities(accumulator={"A": samples}, collided_stakeholder_ids=[])
    expected = float(np.asarray(samples, dtype=np.float64).mean())
    assert out["A"] == expected
    assert math.isfinite(out["A"])


def test_stakeholder_order_independence() -> None:
    vehicles = {
        "A": {"speed": 10.0, "target_speed": 20.0, "active_on_road": True, "completed": False},
        "B": {"speed": 15.0, "target_speed": 20.0, "active_on_road": True, "completed": False},
    }
    acc1 = initialise_episode_utility_accumulator(("A", "B"))
    acc2 = initialise_episode_utility_accumulator(("B", "A"))
    collect_active_state_attainment(vehicles=vehicles, stakeholder_ids=("A", "B"), accumulator=acc1)
    collect_active_state_attainment(vehicles=vehicles, stakeholder_ids=("B", "A"), accumulator=acc2)
    u1 = finalise_episode_utilities(accumulator=acc1, collided_stakeholder_ids=[])
    u2 = finalise_episode_utilities(accumulator=acc2, collided_stakeholder_ids=[])
    assert u1 == u2
