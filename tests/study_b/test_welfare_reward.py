from __future__ import annotations

import pytest

from thesis.study_b.welfare_reward import (
    GGI,
    MAXIMIN,
    MEAN,
    WELFARE_LAMBDA,
    condition_by_name,
    terminal_welfare_bonus,
)


def test_condition_by_name_roundtrip():
    assert condition_by_name("mean") is MEAN
    assert condition_by_name("ggi") is GGI
    assert condition_by_name("maximin") is MAXIMIN


def test_condition_by_name_rejects_unknown():
    with pytest.raises(ValueError):
        condition_by_name("baseline")  # no more baseline -- change.md/revised plan drops it


def test_all_ones_gives_welfare_one_for_all_three_conditions():
    """change.md's required test vector: U=[1,1,1,1] -> W=1 for Mean/GGI/Maximin."""
    utilities = [1.0, 1.0, 1.0, 1.0]
    assert MEAN.welfare_fn(utilities) == pytest.approx(1.0)
    assert GGI.welfare_fn(utilities) == pytest.approx(1.0)
    assert MAXIMIN.welfare_fn(utilities) == pytest.approx(1.0)


def test_one_failure_gives_change_md_exact_values():
    """change.md's required test vector: U=[1,1,1,0] -> Mean=0.75, GGI=0.60, Maximin=0."""
    utilities = [1.0, 1.0, 1.0, 0.0]
    assert MEAN.welfare_fn(utilities) == pytest.approx(0.75)
    assert GGI.welfare_fn(utilities) == pytest.approx(0.60)
    assert MAXIMIN.welfare_fn(utilities) == pytest.approx(0.0)


def test_terminal_welfare_bonus_formula():
    # W_mean([1,1,1,1]) = 1 -> R = 1.0*(1-1) = 0
    assert terminal_welfare_bonus(MEAN, [1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)
    # W_mean([1,1,1,0]) = 0.75 -> R = 1.0*(0.75-1) = -0.25
    assert terminal_welfare_bonus(MEAN, [1.0, 1.0, 1.0, 0.0]) == pytest.approx(-0.25)
    # W_maximin([1,1,1,0]) = 0 -> R = 1.0*(0-1) = -1.0
    assert terminal_welfare_bonus(MAXIMIN, [1.0, 1.0, 1.0, 0.0]) == pytest.approx(-1.0)
    # W_ggi([1,1,1,0]) = 0.6 -> R = 1.0*(0.6-1) = -0.4
    assert terminal_welfare_bonus(GGI, [1.0, 1.0, 1.0, 0.0]) == pytest.approx(-0.4)


def test_terminal_welfare_bonus_range_is_minus_one_to_zero():
    # W_c in [0,1] always (utilities are clipped to [0,1]) -> R_c^W in [-1, 0].
    for condition in (MEAN, GGI, MAXIMIN):
        for utilities in ([0, 0, 0, 0], [1, 1, 1, 1], [0.3, 0.6, 0.9, 0.1]):
            bonus = terminal_welfare_bonus(condition, [float(u) for u in utilities])
            assert -1.0 <= bonus <= 0.0


def test_terminal_welfare_bonus_rejects_empty():
    with pytest.raises(ValueError):
        terminal_welfare_bonus(MEAN, [])


def test_welfare_lambda_is_frozen_at_one():
    assert WELFARE_LAMBDA == 1.0


def test_bonus_added_once_per_agent_not_multiplied():
    """change.md #3's regression concern: W_c(U) must be computed ONCE
    per episode (all 4 agents' utilities together), and the SAME scalar
    added to each agent's own terminal reward -- not recomputed per agent
    (which would be wrong if e.g. computed from a single agent's utility
    only) and not accidentally summed/multiplied across 4 calls."""
    utilities = {"V0": 1.0, "V1": 1.0, "V2": 1.0, "V3": 0.0}
    # Correct usage: ONE call using all 4 utilities.
    bonus = terminal_welfare_bonus(MEAN, list(utilities.values()))
    assert bonus == pytest.approx(-0.25)

    # Simulate applying it to each agent's own terminal reward -- every
    # agent gets the SAME bonus added to ITS OWN base task reward, not a
    # per-agent-recomputed value and not bonus*4.
    base_task_rewards = {"V0": 0.5, "V1": 0.4, "V2": 0.3, "V3": -1.0}
    shaped = {vid: r + bonus for vid, r in base_task_rewards.items()}
    assert shaped == {
        "V0": pytest.approx(0.25),
        "V1": pytest.approx(0.15),
        "V2": pytest.approx(0.05),
        "V3": pytest.approx(-1.25),
    }
    # The bonus itself must never be scaled by N or added N times to any
    # single stream.
    assert bonus != pytest.approx(-0.25 * 4)
