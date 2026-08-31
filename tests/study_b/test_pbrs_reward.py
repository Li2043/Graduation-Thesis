from __future__ import annotations

import pytest

from thesis.study_b.pbrs_reward import (
    BASELINE,
    GAMMA,
    MEAN_PBRS,
    MIN_PBRS,
    PBRSRewardShaper,
    compute_potential,
    condition_by_name,
    experiences_from_step_info,
)


def test_condition_by_name_roundtrip():
    assert condition_by_name("baseline") is BASELINE
    assert condition_by_name("mean_pbrs") is MEAN_PBRS
    assert condition_by_name("min_pbrs") is MIN_PBRS


def test_condition_by_name_rejects_unknown():
    with pytest.raises(ValueError):
        condition_by_name("max_pbrs")


def test_baseline_potential_always_zero():
    assert compute_potential(BASELINE, [0.2, 0.9, 1.0, 0.5], terminated=False, truncated=False) == 0.0
    assert compute_potential(BASELINE, [], terminated=False, truncated=False) == 0.0


def test_potential_zeroed_at_true_terminal_not_at_truncation():
    experiences = [0.5, 0.5, 0.5, 0.5]
    assert compute_potential(MEAN_PBRS, experiences, terminated=True, truncated=False) == 0.0
    # Truncation preserves the raw potential.
    assert compute_potential(MEAN_PBRS, experiences, terminated=False, truncated=True) == pytest.approx(0.5)


def test_experiences_pins_completed_vehicle_at_one_regardless_of_last_attainment():
    attainments = {"V0": 0.3, "V1": 0.9}
    active = {"V0": False, "V1": True}  # V0 already completed
    experiences = experiences_from_step_info(attainments, active)
    assert sorted(experiences) == pytest.approx(sorted([1.0, 0.9]))


def test_shaper_baseline_shaping_always_zero():
    shaper = PBRSRewardShaper(BASELINE)
    shaper.reset(experiences=[0.5, 0.5, 0.5, 0.5])
    shaping = shaper.step(experiences_next=[0.9, 0.9, 0.9, 0.9], terminated=False, truncated=False)
    assert shaping == 0.0


def test_shaper_mean_pbrs_matches_hand_computed_value():
    shaper = PBRSRewardShaper(MEAN_PBRS, gamma=GAMMA)
    experiences_t = [0.4, 0.6, 0.8, 0.2]  # mean = 0.5
    experiences_t1 = [0.6, 0.6, 0.8, 0.4]  # mean = 0.6
    shaper.reset(experiences=experiences_t)
    shaping = shaper.step(experiences_next=experiences_t1, terminated=False, truncated=False)
    expected_F = GAMMA * 0.6 - 0.5
    assert shaping == pytest.approx(0.2 * expected_F)


def test_shaper_min_pbrs_matches_hand_computed_value():
    shaper = PBRSRewardShaper(MIN_PBRS, gamma=GAMMA)
    experiences_t = [0.4, 0.6, 0.8, 0.2]  # min = 0.2
    experiences_t1 = [0.6, 0.6, 0.8, 0.5]  # min = 0.5
    shaper.reset(experiences=experiences_t)
    shaping = shaper.step(experiences_next=experiences_t1, terminated=False, truncated=False)
    expected_F = GAMMA * 0.5 - 0.2
    assert shaping == pytest.approx(0.2 * expected_F)


def test_shaper_multi_step_sequence_accumulates_phi_prev_correctly():
    shaper = PBRSRewardShaper(MEAN_PBRS, gamma=GAMMA)
    shaper.reset(experiences=[0.5, 0.5, 0.5, 0.5])  # phi=0.5
    s1 = shaper.step(experiences_next=[0.6, 0.6, 0.6, 0.6], terminated=False, truncated=False)  # phi 0.5->0.6
    s2 = shaper.step(experiences_next=[0.7, 0.7, 0.7, 0.7], terminated=False, truncated=False)  # phi 0.6->0.7
    assert s1 == pytest.approx(0.2 * (GAMMA * 0.6 - 0.5))
    assert s2 == pytest.approx(0.2 * (GAMMA * 0.7 - 0.6))


def test_apply_per_vehicle_adds_same_shaping_to_each():
    shaper = PBRSRewardShaper(MEAN_PBRS)
    base = {"V0": 0.1, "V1": -0.05, "V2": 0.0, "V3": 0.2}
    shaped = shaper.apply_per_vehicle(base, shaping=0.03)
    assert shaped == {"V0": pytest.approx(0.13), "V1": pytest.approx(-0.02), "V2": pytest.approx(0.03), "V3": pytest.approx(0.23)}


def test_apply_team_averages_base_then_adds_shaping():
    shaper = PBRSRewardShaper(MEAN_PBRS)
    base = {"V0": 0.0, "V1": 1.0, "V2": 0.0, "V3": 1.0}  # mean = 0.5
    team_r = shaper.apply_team(base, shaping=0.1)
    assert team_r == pytest.approx(0.6)
