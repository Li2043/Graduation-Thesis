"""Stage 11 pilot (E30) -- unit tests for the stakeholder welfare functions
ported from chap02.tex's Stage 9 theoretical framework (see
stage11_welfare.py's module docstring for the full formula mapping).
"""

from __future__ import annotations

import pytest

from thesis.pilots.stage11_welfare import (
    episode_mobility_outcome,
    mean_welfare,
    min_welfare,
    stakeholder_experience,
    target_speed_attainment,
)

# --------------------------------------------------------- target_speed_attainment


def test_attainment_at_target_speed_is_one():
    assert target_speed_attainment(20.0, target_speed=20.0) == pytest.approx(1.0)


def test_attainment_below_target_speed_is_proportional():
    assert target_speed_attainment(10.0, target_speed=20.0) == pytest.approx(0.5)
    assert target_speed_attainment(0.0, target_speed=20.0) == pytest.approx(0.0)


def test_attainment_overspeed_is_capped_at_one():
    # chap02.tex is explicit: overspeed is never counted as extra welfare.
    assert target_speed_attainment(30.0, target_speed=20.0) == pytest.approx(1.0)


def test_attainment_rejects_non_positive_target_speed():
    with pytest.raises(ValueError):
        target_speed_attainment(10.0, target_speed=0.0)
    with pytest.raises(ValueError):
        target_speed_attainment(10.0, target_speed=-5.0)


# --------------------------------------------------------- stakeholder_experience


def test_experience_uses_raw_attainment_when_not_completed():
    assert stakeholder_experience(0.7, completed=False) == pytest.approx(0.7)


def test_experience_is_pinned_to_one_when_completed():
    # Absorbing status: even a vehicle that exited at low speed still scores 1.0.
    assert stakeholder_experience(0.3, completed=True) == pytest.approx(1.0)


# --------------------------------------------------------------- mean_welfare


def test_mean_welfare_averages_all_stakeholders():
    assert mean_welfare([1.0, 0.5]) == pytest.approx(0.75)
    assert mean_welfare([0.2, 0.4, 0.6, 0.8]) == pytest.approx(0.5)


def test_mean_welfare_rejects_empty():
    with pytest.raises(ValueError):
        mean_welfare([])


# ---------------------------------------------------------------- min_welfare


def test_min_welfare_is_worst_off_stakeholder():
    assert min_welfare([1.0, 0.5]) == pytest.approx(0.5)
    assert min_welfare([0.2, 0.9, 0.4]) == pytest.approx(0.2)


def test_min_welfare_disagrees_with_mean_on_the_chap02_example_vectors():
    # chap02.tex Eq. (example-vectors): E1=(1,1,1,0.2) has higher mean but lower min than E2=(.6,.6,.6,.6).
    e1 = [1.0, 1.0, 1.0, 0.2]
    e2 = [0.6, 0.6, 0.6, 0.6]
    assert mean_welfare(e1) > mean_welfare(e2)
    assert min_welfare(e1) < min_welfare(e2)


def test_min_welfare_rejects_empty():
    with pytest.raises(ValueError):
        min_welfare([])


# --------------------------------------------------------- episode_mobility_outcome


def test_episode_outcome_is_zero_on_collision_regardless_of_attainment():
    assert episode_mobility_outcome(collided=True, attainment_samples=[0.9, 0.95, 1.0]) == 0.0


def test_episode_outcome_is_time_average_when_no_collision():
    assert episode_mobility_outcome(collided=False, attainment_samples=[0.5, 1.0]) == pytest.approx(0.75)


def test_episode_outcome_is_zero_for_empty_samples_no_collision():
    # Degenerate case (e.g. instant termination before any on-road step recorded).
    assert episode_mobility_outcome(collided=False, attainment_samples=[]) == 0.0
