"""Stage 11 pilot (E30) v3 fix -- clawback/U_i must not reach back and erase
an already-completed vehicle's banked exit bonus or episode-level outcome.

Bug found in v2 (2026-08-06): collision detection checks every pair of
active vehicle ids regardless of completed status, so a vehicle that has
already exited (and keeps cruising at constant speed past the exit line)
can register a later, physically real collision (e.g. a trailing vehicle
catching up) in the same episode. Confirmed empirically in v2's own
trajectory data: one episode where the clawback penalty hit ~1.0 (the full
progress+exit ceiling) on a vehicle that had already banked its exit bonus.
See stage11_welfare.py's clawback_tracking_contribution/
episode_collided_for_vehicle docstrings and stage11_dyad_merge_runner.py's
module docstring for the full mechanism.
"""

from __future__ import annotations

import pytest

from thesis.pilots.stage11_welfare import (
    clawback_tracking_contribution,
    episode_collided_for_vehicle,
    episode_mobility_outcome,
)

# ----------------------------------------------------- clawback_tracking_contribution


def test_ordinary_progress_step_passes_through_unchanged():
    assert clawback_tracking_contribution(
        step_reward=0.012, exit_event=False, exit_reward_magnitude=0.6
    ) == pytest.approx(0.012)


def test_exit_step_excludes_the_exit_bonus():
    # step_reward on the exit step = small final progress delta + the exit bonus.
    step_reward = 0.008 + 0.6
    contribution = clawback_tracking_contribution(
        step_reward=step_reward, exit_event=True, exit_reward_magnitude=0.6
    )
    assert contribution == pytest.approx(0.008)  # exit bonus stripped out, tiny progress remains


def test_exit_step_with_larger_exit_magnitude():
    # Same logic must hold for the proposed higher exit reward (1.8), not just 0.6.
    step_reward = 0.005 + 1.8
    contribution = clawback_tracking_contribution(
        step_reward=step_reward, exit_event=True, exit_reward_magnitude=1.8
    )
    assert contribution == pytest.approx(0.005)


def test_banked_exit_bonus_is_never_reachable_by_a_later_clawback():
    """End-to-end of the bug scenario: accumulate ordinary progress, exit
    (banking the bonus, excluded from tracking), then simulate a LATER
    collision using whatever is left in the running total -- must be small,
    never anywhere near the full progress+exit ceiling."""
    cumulative = 0.0
    # A few ordinary progress steps before exit.
    for _ in range(5):
        cumulative += clawback_tracking_contribution(
            step_reward=0.02, exit_event=False, exit_reward_magnitude=0.6
        )
    # The exit step itself.
    cumulative += clawback_tracking_contribution(
        step_reward=0.01 + 0.6, exit_event=True, exit_reward_magnitude=0.6
    )
    # Vehicle is now completed; no further progress/exit reward accrues
    # (matches env.py: completed vehicles get delta=0, exit_event=False
    # forever after), so `cumulative` stays frozen at this point.
    assert cumulative == pytest.approx(0.11)  # 5*0.02 + 0.01, NOT +0.6
    # A later "collision" would use `cumulative` as the override -- far
    # below the ~1.0 (0.4 progress + 0.6 exit) ceiling a buggy tracker
    # would have produced.
    assert cumulative < 0.2


# --------------------------------------------------------- episode_collided_for_vehicle


def test_vehicle_still_in_progress_is_zeroed_by_episode_collision():
    assert episode_collided_for_vehicle(
        episode_collided=True, was_already_completed_before_collision=False
    ) is True


def test_already_completed_vehicle_is_not_zeroed_by_a_later_collision():
    assert episode_collided_for_vehicle(
        episode_collided=True, was_already_completed_before_collision=True
    ) is False


def test_no_collision_never_zeroes_anyone():
    assert episode_collided_for_vehicle(
        episode_collided=False, was_already_completed_before_collision=False
    ) is False
    assert episode_collided_for_vehicle(
        episode_collided=False, was_already_completed_before_collision=True
    ) is False


def test_fixed_flag_feeds_correctly_into_episode_mobility_outcome():
    """Integration of the two fixed pieces: an already-completed vehicle's
    U_i must reflect its own actual on-road attainment average, not 0,
    even when the episode as a whole ended in collision."""
    already_completed_flag = episode_collided_for_vehicle(
        episode_collided=True, was_already_completed_before_collision=True
    )
    outcome = episode_mobility_outcome(
        collided=already_completed_flag, attainment_samples=[0.8, 0.9, 1.0]
    )
    assert outcome == pytest.approx(0.9)  # NOT zeroed

    still_at_risk_flag = episode_collided_for_vehicle(
        episode_collided=True, was_already_completed_before_collision=False
    )
    outcome_at_risk = episode_mobility_outcome(
        collided=still_at_risk_flag, attainment_samples=[0.8, 0.9, 1.0]
    )
    assert outcome_at_risk == 0.0  # correctly zeroed -- this vehicle really did collide
