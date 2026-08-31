"""Stage 10 pilot v2 (E28) -- pre-training audit for the curriculum, LR
decay, post-merge zone lengthening, and trajectory logging additions.

Covers: env n_vehicles in {2,4}, curriculum vehicle-count switch via the
runner, LR schedule correctness incl. continuity across the step-40,000
boundary, new 150m post-merge zone geometry, and trajectory log well-
formedness -- without running the full 100K-step pilot.
"""

from __future__ import annotations

import pytest

from thesis.envs.stage10_symmetric_merge_env import (
    Stage10MergeEnvConfig,
    Stage10RouteGeometry,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage10_role_phase_subpolicy_config import (
    CURRICULUM_STAGE_A_STEPS,
    LEARNING_RATE_DECAY_STEPS,
    LEARNING_RATE_END,
    LEARNING_RATE_START,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    assert_stage10_pilot_guards,
    lr_at_step,
)


# --------------------------------------------------------------------- geometry
#
# NOTE (pilot v4): pilot v2 lengthened post-merge 100m->150m; pilot v3
# reverted this (the visitation imbalance it was meant to fix turned out to
# track policy competence, not geometry). Pilot v4 lengthens PRE-merge
# instead (150->200, a deliberately separate lever -- protocol S0.1's "easier
# baseline geometry" point). This test is updated in place again (compatible
# update, not a regression) to assert the CURRENT geometry -- see
# test_stage10_pilot_v4.py for the full v4 geometry test.
def test_post_merge_zone_reverted_to_100m_in_pilot_v3():
    g = Stage10RouteGeometry()
    assert g.merge_start == 200.0  # v1: 150 -> v2: unchanged -> v3: unchanged -> v4: 200 (pre-merge lever)
    assert g.merge_end == 300.0
    assert g.route_exit == 400.0  # v1: 350 -> v2: 400 -> v3: back to 350 -> v4: 400 again (different reason: pre, not post)
    pre_length = g.merge_start - g.route_start
    merging_length = g.merge_end - g.merge_start
    post_length = g.route_exit - g.merge_end
    assert pre_length == 200.0
    assert merging_length == 100.0
    assert post_length == 100.0  # v3's revert still holds -- post-merge itself untouched by v4


# ------------------------------------------------------------------ n_vehicles
def test_env_rejects_invalid_n_vehicles():
    with pytest.raises(ValueError):
        Stage10MergeEnvConfig(n_vehicles=3).validate()
    with pytest.raises(ValueError):
        Stage10MergeEnvConfig(n_vehicles=1).validate()


@pytest.mark.parametrize("seed", range(10))
def test_two_vehicle_mode_has_exactly_one_ramp_one_mainline(seed):
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=seed, n_vehicles=2))
    obs, info = env.reset(seed=seed)
    roles = info["roles"]
    assert len(roles) == 2
    assert sorted(roles.values()) == ["mainline", "ramp"]
    assert set(env.active_vehicle_ids) == set(roles.keys())
    assert set(obs.keys()) == set(roles.keys())


def test_two_vehicle_mode_step_and_collision_work():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=3, n_vehicles=2))
    env.reset(seed=3)
    active = env.active_vehicle_ids
    assert len(active) == 2
    actions = {vid: 0 for vid in active}  # MAINTAIN
    obs, reward, terminated, truncated, info = env.step(actions)
    assert set(obs.keys()) == set(active)
    assert set(reward.keys()) == set(active)
    assert set(info["zone_t"].keys()) == set(active)
    assert set(info["zone_t1"].keys()) == set(active)


def test_four_vehicle_mode_default_unchanged_from_pilot_v1():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=5))  # n_vehicles defaults to 4
    obs, info = env.reset(seed=5)
    assert len(info["roles"]) == 4
    assert sorted(info["roles"].values()) == ["mainline", "mainline", "ramp", "ramp"]
    assert len(env.active_vehicle_ids) == 4


# ------------------------------------------------------------------- LR decay
def test_lr_at_step_matches_stage8_arm2b_endpoints():
    assert lr_at_step(0) == pytest.approx(LEARNING_RATE_START)
    assert lr_at_step(0) == pytest.approx(0.0005)
    assert lr_at_step(LEARNING_RATE_DECAY_STEPS) == pytest.approx(LEARNING_RATE_END)
    assert lr_at_step(LEARNING_RATE_DECAY_STEPS) == pytest.approx(0.0001)
    assert lr_at_step(LEARNING_RATE_DECAY_STEPS + 50_000) == pytest.approx(LEARNING_RATE_END)  # clamped


def test_lr_at_step_monotonically_decreasing_mid_schedule():
    mid = lr_at_step(LEARNING_RATE_DECAY_STEPS // 2)
    assert LEARNING_RATE_END < mid < LEARNING_RATE_START


def test_lr_schedule_is_continuous_across_curriculum_boundary():
    """protocol requirement: LR decay must not reset/jump at the step-40,000
    curriculum transition -- it's purely a function of absolute step. A 2-step
    gap on this linear schedule naturally differs by ~2/DECAY_STEPS*(START-END)
    (~8e-9 here) -- that's the expected slope, not a discontinuity. A reset
    (e.g. jumping back to LEARNING_RATE_START) would differ by orders of
    magnitude more (~4e-4), which is what this bound actually needs to catch."""
    just_before = lr_at_step(CURRICULUM_STAGE_A_STEPS - 1)
    just_after = lr_at_step(CURRICULUM_STAGE_A_STEPS + 1)
    expected_slope_delta = 2.0 / LEARNING_RATE_DECAY_STEPS * (LEARNING_RATE_START - LEARNING_RATE_END)
    assert abs(just_before - just_after) == pytest.approx(expected_slope_delta, abs=1e-12)


# --------------------------------------------------------------- seed guards
def test_v1_and_v2_seeds_both_pass_the_guard_and_are_disjoint():
    assert set(PILOT_V1_SEEDS).isdisjoint(set(PILOT_V2_SEEDS))
    for seed in (*PILOT_V1_SEEDS, *PILOT_V2_SEEDS):
        assert_stage10_pilot_guards(master_seed=seed, max_steps=100_000)  # must not raise


def test_guard_rejects_seed_outside_both_pilot_blocks():
    # NOTE (pilot v4): 68009 (v3) and 68013 (v4) are now both legitimate
    # pilot-arm seeds, so neither serves as an "outside all pilot blocks"
    # example any more. 68017 was also drawn into PILOT_V4_SEEDS by the
    # 2026-08-05 4->8 seed extension -- 68021 is the first value genuinely
    # outside v1-v4 alike now.
    with pytest.raises(RuntimeError):
        assert_stage10_pilot_guards(master_seed=68021, max_steps=100_000)


# ------------------------------------------------ curriculum + logging (integration)
#
# NOTE (pilot v4): the runner's curriculum mechanism has changed twice since
# these tests were written -- v2's hard step-40,000 cutover, then v3's
# episode-level probability blend, and now v4's threshold-triggered 2->4->6
# staging (see stage10_role_phase_subpolicy_config.py's
# stage_index_for_advance and test_stage10_pilot_v4.py). run_pilot_training_job
# no longer accepts curriculum_pure_a_steps/curriculum_blend_end_steps at all
# -- the three integration tests that exercised v3's blend-specific windowing
# are removed here (not silently broken; their v4-equivalent coverage lives
# in test_stage10_pilot_v4.py, which tests the current mechanism directly
# rather than duplicating removed behaviour here).
