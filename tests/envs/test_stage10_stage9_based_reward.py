"""Stage 11 pilot (E30) v4 -- Stage 9-based reward redesign env-level terms.

Covers the two new optional ``step()`` kwargs ported from Stage 9's formally
locked base reward (``thesis.rewards.base_reward_v2``): hard-braking cost
(``hard_braking_eta``) and active-time cost (``time_cost_per_step``). Both
default to ``None`` (disabled), so this file also asserts the v1-v7/Stage 11
v1-v3 call sites (omitting these kwargs) are byte-for-byte unaffected --
covered instead by the pre-existing ``test_stage10_symmetric_merge_env.py``
suite, which passes unmodified against this change.
"""

from __future__ import annotations

import pytest

from thesis.envs.stage10_symmetric_merge_env import (
    A_COMFORT,
    A_HARD,
    HARD_BRAKING_ETA,
    HighLevelAction,
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
    TIME_COST_PER_STEP,
)
from thesis.rewards.base_reward_v2 import compute_hard_braking_cost


def test_stage9_constants_match_locked_values():
    # See final_lock_loader.py's FinalLockBlockedError guard -- these four
    # numbers are hard-validated there as Stage 9's actual locked values.
    assert HARD_BRAKING_ETA == pytest.approx(0.015)
    assert TIME_COST_PER_STEP == pytest.approx(0.0005)
    assert A_COMFORT == pytest.approx(1.5)
    assert A_HARD == pytest.approx(3.5)


def test_hard_braking_cost_disabled_by_default():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1, n_vehicles=2))
    env.reset(seed=1)
    active = env.active_vehicle_ids
    actions = {vid: HighLevelAction.DECELERATE for vid in active}
    _, _reward, _term, _trunc, info = env.step(actions)
    assert info["hard_braking_eta_used"] == 0.0
    # cost is still reported (for diagnostics) even when the term is disabled.
    for vid in active:
        assert info["hard_braking_cost_used"][vid] == pytest.approx(0.0)


def test_hard_braking_cost_applies_when_decelerating_with_eta_enabled():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1, n_vehicles=2))
    env.reset(seed=1)
    active = env.active_vehicle_ids
    actions = {vid: HighLevelAction.DECELERATE for vid in active}
    _, reward, _term, _trunc, info = env.step(actions, hard_braking_eta=HARD_BRAKING_ETA)
    assert info["hard_braking_eta_used"] == pytest.approx(HARD_BRAKING_ETA)
    expected_h = compute_hard_braking_cost(-3.0, A_COMFORT, A_HARD)
    assert expected_h == pytest.approx(0.5625)
    for vid in active:
        assert info["hard_braking_cost_used"][vid] == pytest.approx(expected_h)
    # reward must be reduced by exactly eta * H relative to the no-cost case.
    env2 = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1, n_vehicles=2))
    env2.reset(seed=1)
    _, reward_no_cost, _t2, _tr2, _info2 = env2.step(actions)
    for vid in active:
        assert reward[vid] == pytest.approx(
            reward_no_cost[vid] - HARD_BRAKING_ETA * expected_h
        )


def test_hard_braking_cost_zero_when_accelerating_or_maintaining():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=2, n_vehicles=2))
    env.reset(seed=2)
    active = env.active_vehicle_ids
    for act in (HighLevelAction.ACCELERATE, HighLevelAction.MAINTAIN):
        env2 = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=2, n_vehicles=2))
        env2.reset(seed=2)
        actions = {vid: act for vid in env2.active_vehicle_ids}
        _, _reward, _term, _trunc, info = env2.step(actions, hard_braking_eta=HARD_BRAKING_ETA)
        for vid in env2.active_vehicle_ids:
            assert info["hard_braking_cost_used"][vid] == pytest.approx(0.0)


def test_time_cost_disabled_by_default():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1, n_vehicles=2))
    env.reset(seed=1)
    active = env.active_vehicle_ids
    actions = {vid: HighLevelAction.MAINTAIN for vid in active}
    _, _reward, _term, _trunc, info = env.step(actions)
    assert info["time_cost_per_step_used"] == 0.0
    assert all(info["time_cost_applied"].values())


def test_time_cost_charged_while_active_and_stops_once_exited():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1, n_vehicles=2, spawn_route_lead=0.0))
    env.reset(seed=1)
    active = env.active_vehicle_ids
    for vid in active:
        env._roles[vid] = "mainline"
        env._vehicles[vid].role = "mainline"
    env._vehicles[active[0]].route_position = 398.0
    env._vehicles[active[1]].route_position = 100.0
    env._vehicles[active[0]].speed = 18.0
    env._vehicles[active[1]].speed = 18.0

    _, reward, _term, _trunc, info = env.step(
        {vid: HighLevelAction.ACCELERATE for vid in active}, time_cost_per_step=TIME_COST_PER_STEP
    )
    exited_vid = next(vid for vid, v in info["exit_event"].items() if v)
    other_vid = next(vid for vid in active if vid != exited_vid)
    assert info["time_cost_per_step_used"] == pytest.approx(TIME_COST_PER_STEP)
    # Time cost is charged on the transition where the exit itself happens
    # (still active at the START of this transition) per Stage 9's
    # I_active(s_t) definition -- both vehicles were active before this step.
    assert info["time_cost_applied"][exited_vid] is True
    assert info["time_cost_applied"][other_vid] is True

    # Next step: the now-completed vehicle must stop accruing time cost.
    _, reward2, _term2, _trunc2, info2 = env.step(
        {vid: HighLevelAction.MAINTAIN for vid in active}, time_cost_per_step=TIME_COST_PER_STEP
    )
    assert info2["time_cost_applied"][exited_vid] is False
    assert info2["time_cost_applied"][other_vid] is True


def test_stage9_based_full_formula_matches_manual_computation():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=3, n_vehicles=2))
    env.reset(seed=3)
    active = env.active_vehicle_ids
    actions = {vid: HighLevelAction.DECELERATE for vid in active}
    _, reward, _term, _trunc, info = env.step(
        actions,
        collision_penalty_magnitude=1.0,
        ttc_penalty_weight=0.0,
        hard_braking_eta=HARD_BRAKING_ETA,
        time_cost_per_step=TIME_COST_PER_STEP,
    )
    expected_h = compute_hard_braking_cost(-3.0, A_COMFORT, A_HARD)
    for vid in active:
        # No collision, no exit expected from a fresh reset + one DECELERATE
        # step -- reward should equal progress - hard_braking - time_cost.
        assert not info["exit_event"][vid]
        assert not info["collision_penalty_applied"][vid]
        expected = (
            reward[vid]
            + HARD_BRAKING_ETA * expected_h
            + TIME_COST_PER_STEP * (1.0 if info["time_cost_applied"][vid] else 0.0)
        )
        # progress-only remainder must be non-negative (vehicles only move forward).
        assert expected >= -1e-9
