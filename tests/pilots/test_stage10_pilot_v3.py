"""Stage 10 pilot v3 (E28) -- pre-training audit for the probability-blend
curriculum, the peer-intent observation feature, and the post-merge zone
revert.

Covers: p_four_vehicle_at_step's ramp shape and boundary determinism,
n_vehicles_for_draw's threshold logic, the runner's empirical episode mix
inside the blend window, the peer-intent feature's correctness (incl. the
first-step neutral default), OBS_DIM==8 everywhere, and the reverted 100m
post-merge geometry -- without running the full 100K-step pilot.
"""

from __future__ import annotations

import pytest

from thesis.agents.independent_dqn_v2 import DQNConfig
from thesis.envs.stage10_symmetric_merge_env import (
    HighLevelAction,
    Stage10MergeEnvConfig,
    Stage10RouteGeometry,
    Stage10SymmetricMergeEnv,
    encode_peer_intent,
)
from thesis.pilots.stage10_role_phase_subpolicy_config import (
    OBS_DIM as CONFIG_OBS_DIM,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    PILOT_V3_SEEDS,
    assert_stage10_pilot_guards,
)
from thesis.pilots.stage10_role_phase_subpolicy_runner import build_dqn_config

# NOTE (pilot v4): the eight tests that used to live here --
# test_p_four_vehicle_zero_below_pure_a_and_one_at_and_above_blend_end,
# test_p_four_vehicle_linear_ramp_midpoint_and_monotonic,
# test_p_four_vehicle_rejects_invalid_window,
# test_p_four_vehicle_default_window_matches_frozen_constants,
# test_n_vehicles_for_draw_threshold_boundary,
# test_n_vehicles_for_draw_deterministic_outside_blend_window,
# test_runner_empirical_episode_starts_respect_blend_window_boundaries, and
# test_runner_zero_width_pure_a_window_still_deterministic_at_step_zero --
# exercised p_four_vehicle_at_step/n_vehicles_for_draw, which pilot v4 REMOVED
# (the probability-blend curriculum they implemented made results worse; see
# stage10_role_phase_subpolicy_config.py's "REMOVED in pilot v4" comment and
# protocol doc S0.1). Removed here too rather than left broken; the
# successor mechanism (threshold-triggered 2->4->6 staging) has its own full
# test coverage in test_stage10_pilot_v4.py, not duplicated here.


# ------------------------------------------------------------------- geometry
def test_post_merge_zone_reverted_to_100m():
    """NOTE (pilot v4): pre-merge was lengthened again (150->200) for a
    different reason (easier-baseline geometry, not visitation imbalance) --
    this test is updated to the current values; see
    test_stage10_pilot_v4.py's test_pre_merge_lengthened_others_unchanged for
    the full v4 geometry test."""
    g = Stage10RouteGeometry()
    assert g.merge_start == 200.0  # v1-v3: 150.0 -> v4: 200.0
    assert g.merge_end == 300.0
    assert g.route_exit == 400.0
    assert (g.route_exit - g.merge_end) == 100.0  # post-merge length -- v3's revert still holds
    assert (g.merge_start - g.route_start) == 200.0  # pre-merge length -- v4's own lever


# --------------------------------------------------------- OBS_DIM everywhere
def test_obs_dim_is_9_everywhere():
    """NOTE (pilot v4): OBS_DIM is 9 now (scene_vehicle_count added at index
    8); updated in place, see test_stage10_pilot_v4.py for the full test."""
    assert CONFIG_OBS_DIM == 9
    cfg: DQNConfig = build_dqn_config()
    assert cfg.obs_dim == 9


def test_reset_and_step_observations_have_9_dims():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1))
    obs, _ = env.reset(seed=1)
    for vid, o in obs.items():
        assert o.shape == (9,)
    actions = {vid: HighLevelAction.MAINTAIN for vid in env.active_vehicle_ids}
    obs2, *_ = env.step(actions)
    for vid, o in obs2.items():
        assert o.shape == (9,)


# --------------------------------------------------------------- peer intent
def test_encode_peer_intent_mapping():
    assert encode_peer_intent(HighLevelAction.ACCELERATE) == 1.0
    assert encode_peer_intent(HighLevelAction.DECELERATE) == -1.0
    assert encode_peer_intent(HighLevelAction.MAINTAIN) == 0.0


def test_peer_intent_defaults_to_neutral_on_first_observation():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=2, n_vehicles=2))
    obs, info = env.reset(seed=2)
    for vid in env.active_vehicle_ids:
        assert obs[vid][7] == 0.0  # peer_intent is the 8th (index 7) feature


def test_peer_intent_reflects_peers_most_recent_action_next_step():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=2, n_vehicles=2))
    obs, info = env.reset(seed=2)
    roles = info["roles"]
    ramp_id = next(vid for vid, r in roles.items() if r == "ramp")
    mainline_id = next(vid for vid, r in roles.items() if r == "mainline")

    # Force the two vehicles into the same lane-comparable band so they see
    # each other as peers regardless of the pre-merge lateral offset. Ramp's
    # world_y only reaches 0 at merge_end (300.0), linearly interpolating
    # from -4.0 across [merge_start=200, merge_end=300) -- pick a position
    # deep enough into the merging zone (frac=0.8 -> world_y=-0.8) to be
    # within the 1.5 lane-comparability threshold of mainline's world_y=0.
    # (NOTE pilot v4: these positions were 230.0/235.0 under the old
    # merge_start=150 geometry -- recomputed here for merge_start=200, or
    # this test would silently break: at the OLD numbers under the NEW
    # geometry, frac would be only 0.3 (world_y=-2.8), outside the
    # lane-comparability threshold, and the peer would go undetected.)
    env._vehicles[ramp_id].route_position = 280.0  # merging zone, frac=0.8, world_y=-0.8
    env._vehicles[mainline_id].route_position = 285.0

    actions = {ramp_id: HighLevelAction.ACCELERATE, mainline_id: HighLevelAction.DECELERATE}
    next_obs, *_ = env.step(actions)

    # From the mainline vehicle's perspective, its nearest peer (ramp) just
    # accelerated -> peer_intent should read +1.0 in the NEXT observation.
    assert next_obs[mainline_id][7] == pytest.approx(1.0)
    # From the ramp vehicle's perspective, its nearest peer (mainline) just
    # decelerated -> peer_intent should read -1.0.
    assert next_obs[ramp_id][7] == pytest.approx(-1.0)


def test_peer_intent_neutral_when_isolated_no_lane_comparable_peer():
    """Direct unit check of the no-peer branch (ahead=behind=None) -- an
    isolated vehicle (no lane-comparable neighbour, e.g. a ramp vehicle still
    in pre-merge with no other vehicle nearby in its own lane) must read the
    neutral default, not error or fall back to a stale value."""
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=9, n_vehicles=2))
    env.reset(seed=9)
    assert env._peer_intent(None, None) == 0.0


# --------------------------------------------------------------- seed guards
def test_v3_seeds_pass_the_guard_and_are_disjoint_from_v1_v2():
    assert set(PILOT_V3_SEEDS).isdisjoint(set(PILOT_V1_SEEDS))
    assert set(PILOT_V3_SEEDS).isdisjoint(set(PILOT_V2_SEEDS))
    for seed in PILOT_V3_SEEDS:
        assert_stage10_pilot_guards(master_seed=seed, max_steps=100_000)  # must not raise


def test_guard_rejects_seed_outside_all_three_pilot_blocks():
    # NOTE (pilot v4): 68013 is now a legitimate PILOT_V4_SEEDS member, and
    # 68017 was also drawn in by the 2026-08-05 4->8 seed extension --
    # 68021 is the first value genuinely outside v1-v4 alike now.
    with pytest.raises(RuntimeError):
        assert_stage10_pilot_guards(master_seed=68021, max_steps=100_000)
