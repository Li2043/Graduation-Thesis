"""Stage 10 pilot v7 (E28) -- reward-MAGNITUDE curriculum (protocol S0.1
pilot v7): the collision-penalty magnitude and TTC-shaping weight now ramp
linearly over training instead of being static from step 0, directly
modelled on Alhazza (2026)'s DQN004 (continuous-ramp collision penalty) and
DQN006 (a second, independently-ramped shaping term) -- see
stage10_symmetric_merge_env.py's module docstring "Reward-magnitude
curriculum" section for the full rationale/citations.

Covers: the pure ``collision_penalty_at_step``/``ttc_weight_at_step`` ramp
functions (start/end/boundary values, linearity, post-ramp constancy), that
``Stage10SymmetricMergeEnv.step()``'s new optional overrides default to the
static v6 constants (so v1-v6 call sites are unaffected) and, when given,
actually replace the magnitude/weight used in the reward formula, the new
REWARD_VERSION_V7 tag, and the v7 seed guard.
"""

from __future__ import annotations

import pytest

from thesis.envs.stage10_symmetric_merge_env import (
    COLLISION_PENALTY_MAGNITUDE,
    REWARD_VERSION_V1_V5,
    REWARD_VERSION_V6,
    REWARD_VERSION_V7,
    TTC_PENALTY_WEIGHT,
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage10_role_phase_subpolicy_config import (
    COLLISION_PENALTY_CURRICULUM_END,
    COLLISION_PENALTY_CURRICULUM_START,
    MAX_STEPS_V4,
    PILOT_V6_SEEDS,
    PILOT_V7_SEEDS,
    REWARD_CURRICULUM_RAMP_STEPS,
    TTC_WEIGHT_CURRICULUM_END,
    TTC_WEIGHT_CURRICULUM_START,
    assert_stage10_pilot_guards,
    collision_penalty_at_step,
    ttc_weight_at_step,
)

# ---------------------------------------------------------------- constants


def test_curriculum_end_values_match_v6_static_targets():
    # The ramp must lock onto exactly the values v6 used statically -- the
    # curriculum changes WHEN full strength applies, not what "full strength" is.
    assert COLLISION_PENALTY_CURRICULUM_END == COLLISION_PENALTY_MAGNITUDE == 5.0
    assert TTC_WEIGHT_CURRICULUM_END == TTC_PENALTY_WEIGHT == 0.1
    assert REWARD_VERSION_V7 not in (REWARD_VERSION_V1_V5, REWARD_VERSION_V6)


def test_curriculum_start_values_are_lenient():
    assert COLLISION_PENALTY_CURRICULUM_START == 1.0  # v1-v5's own magnitude
    assert 0.0 < TTC_WEIGHT_CURRICULUM_START < TTC_WEIGHT_CURRICULUM_END


def test_ramp_completes_at_stage3_boundary():
    # 100,000 = Stage 1 (40,000) + Stage 2 (60,000) safety-valve budgets --
    # the ramp must finish exactly as Stage 3 (6 vehicles) begins.
    assert REWARD_CURRICULUM_RAMP_STEPS == 100_000


# ------------------------------------------------------- collision_penalty_at_step


def test_collision_penalty_at_step_zero_is_start_value():
    assert collision_penalty_at_step(0) == pytest.approx(COLLISION_PENALTY_CURRICULUM_START)


def test_collision_penalty_at_step_locks_at_end_value_once_ramp_completes():
    assert collision_penalty_at_step(REWARD_CURRICULUM_RAMP_STEPS) == pytest.approx(COLLISION_PENALTY_CURRICULUM_END)
    assert collision_penalty_at_step(REWARD_CURRICULUM_RAMP_STEPS + 50_000) == pytest.approx(COLLISION_PENALTY_CURRICULUM_END)
    assert collision_penalty_at_step(MAX_STEPS_V4) == pytest.approx(COLLISION_PENALTY_CURRICULUM_END)


def test_collision_penalty_at_step_is_linear_at_midpoint():
    mid = REWARD_CURRICULUM_RAMP_STEPS // 2
    expected = COLLISION_PENALTY_CURRICULUM_START + 0.5 * (
        COLLISION_PENALTY_CURRICULUM_END - COLLISION_PENALTY_CURRICULUM_START
    )
    assert collision_penalty_at_step(mid) == pytest.approx(expected)


def test_collision_penalty_at_step_is_monotonically_non_decreasing():
    prev = collision_penalty_at_step(0)
    for step in range(0, REWARD_CURRICULUM_RAMP_STEPS + 1, 5_000):
        cur = collision_penalty_at_step(step)
        assert cur >= prev - 1e-9
        prev = cur


# ------------------------------------------------------------ ttc_weight_at_step


def test_ttc_weight_at_step_zero_is_start_value():
    assert ttc_weight_at_step(0) == pytest.approx(TTC_WEIGHT_CURRICULUM_START)


def test_ttc_weight_at_step_locks_at_end_value_once_ramp_completes():
    assert ttc_weight_at_step(REWARD_CURRICULUM_RAMP_STEPS) == pytest.approx(TTC_WEIGHT_CURRICULUM_END)
    assert ttc_weight_at_step(MAX_STEPS_V4) == pytest.approx(TTC_WEIGHT_CURRICULUM_END)


def test_ttc_weight_at_step_is_linear_at_midpoint():
    mid = REWARD_CURRICULUM_RAMP_STEPS // 2
    expected = TTC_WEIGHT_CURRICULUM_START + 0.5 * (TTC_WEIGHT_CURRICULUM_END - TTC_WEIGHT_CURRICULUM_START)
    assert ttc_weight_at_step(mid) == pytest.approx(expected)


# --------------------------------------------------- env.step() override behaviour


def test_step_defaults_to_static_constants_when_overrides_omitted():
    """v1-v6 call sites (no kwargs passed) must be byte-for-byte unaffected."""
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=7, n_vehicles=2))
    env.reset(seed=7)
    active = env.active_vehicle_ids
    for vid in active:
        env._roles[vid] = "mainline"
        env._vehicles[vid].role = "mainline"
    env._vehicles[active[0]].route_position = 250.0
    env._vehicles[active[1]].route_position = 250.0
    env._vehicles[active[0]].speed = 15.0
    env._vehicles[active[1]].speed = 15.0
    actions = {vid: 0 for vid in active}
    _, _reward, _terminated, _truncated, info = env.step(actions)
    assert info["collision_penalty_magnitude_used"] == COLLISION_PENALTY_MAGNITUDE
    assert info["ttc_penalty_weight_used"] == TTC_PENALTY_WEIGHT


def test_step_override_replaces_collision_penalty_magnitude():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=8, n_vehicles=2))
    env.reset(seed=8)
    active = env.active_vehicle_ids
    for vid in active:
        env._roles[vid] = "mainline"
        env._vehicles[vid].role = "mainline"
    env._vehicles[active[0]].route_position = 250.0
    env._vehicles[active[1]].route_position = 250.0
    env._vehicles[active[0]].speed = 15.0
    env._vehicles[active[1]].speed = 15.0
    actions = {vid: 0 for vid in active}
    _, reward, terminated, _truncated, info = env.step(
        actions, collision_penalty_magnitude=1.0, ttc_penalty_weight=0.02
    )
    assert terminated is True
    assert info["collision_penalty_magnitude_used"] == 1.0
    assert info["ttc_penalty_weight_used"] == 0.02
    for vid in active:
        # dominated by -1.0 (the override), not the module's static -5.0 --
        # reward must be well ABOVE -5.0 to prove the override actually won.
        assert reward[vid] > -5.0
        assert reward[vid] < 0.0  # still net negative (collision happened)


def test_step_override_is_independent_per_call_no_persisted_state():
    """Passing an override on one call must not leak into the next call that omits it."""
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=9, n_vehicles=2, max_steps=50))
    env.reset(seed=9)
    active = env.active_vehicle_ids
    actions = {vid: 0 for vid in active}
    env.step(actions, collision_penalty_magnitude=1.0, ttc_penalty_weight=0.02)
    _, _reward, terminated, truncated, info = env.step(actions)
    if not (terminated or truncated):
        assert info["collision_penalty_magnitude_used"] == COLLISION_PENALTY_MAGNITUDE
        assert info["ttc_penalty_weight_used"] == TTC_PENALTY_WEIGHT


# --------------------------------------------------------------------- seeds


def test_v7_seeds_disjoint_from_v6_and_pass_guard():
    assert set(PILOT_V7_SEEDS).isdisjoint(PILOT_V6_SEEDS)
    for seed in PILOT_V7_SEEDS:
        assert_stage10_pilot_guards(master_seed=seed, max_steps=MAX_STEPS_V4)
