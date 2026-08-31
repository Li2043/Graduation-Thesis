"""Stage 10 pilot (E28) pre-training audit -- geometry / routing-input smoke tests.

Covers protocol S8 items 1 (routing correctness incl. boundary values) and 3
(geometry smoke test, no impossible overlapping spawns).
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.envs.stage10_symmetric_merge_env import (
    OBS_DIM,
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
    world_y_for,
    zone_for_position,
)


def test_zone_boundaries_are_left_closed_and_deterministic():
    merge_start, merge_end = 150.0, 250.0
    assert zone_for_position(0.0, merge_start, merge_end) == "pre"
    assert zone_for_position(149.999, merge_start, merge_end) == "pre"
    assert zone_for_position(150.0, merge_start, merge_end) == "merging"  # exactly at merge_start
    assert zone_for_position(249.999, merge_start, merge_end) == "merging"
    assert zone_for_position(250.0, merge_start, merge_end) == "post"  # exactly at merge_end
    assert zone_for_position(350.0, merge_start, merge_end) == "post"


def test_world_y_mainline_is_always_zero():
    assert world_y_for("mainline", 0.0, 150.0, 250.0) == 0.0
    assert world_y_for("mainline", 999.0, 150.0, 250.0) == 0.0


def test_world_y_ramp_merges_linearly_across_merging_zone():
    assert world_y_for("ramp", 0.0, 150.0, 250.0) == -4.0
    assert world_y_for("ramp", 150.0, 150.0, 250.0) == -4.0
    assert world_y_for("ramp", 200.0, 150.0, 250.0) == pytest.approx(-2.0)
    assert world_y_for("ramp", 250.0, 150.0, 250.0) == 0.0
    assert world_y_for("ramp", 300.0, 150.0, 250.0) == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_reset_geometry_has_no_impossible_overlap(seed):
    # NOTE (pilot v4): VEHICLE_IDS was extended from 4 to 6 entries to
    # support the 6-vehicle curriculum stage; this test uses the DEFAULT
    # n_vehicles=4 config, so it must check against env.active_vehicle_ids
    # (the currently-active subset), not the full VEHICLE_IDS pool -- see
    # test_stage10_pilot_v4.py for dedicated 2/4/6-vehicle coverage.
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=seed))
    obs, info = env.reset(seed=seed)
    roles = info["roles"]
    active = env.active_vehicle_ids

    assert set(roles.keys()) == set(active)
    assert sorted(roles.values()) == ["mainline", "mainline", "ramp", "ramp"]

    for vid in active:
        assert obs[vid].shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs[vid]))

    for role in ("ramp", "mainline"):
        members = sorted(vid for vid in active if roles[vid] == role)
        assert len(members) == 2
        positions = [env._vehicles[vid].route_position for vid in members]
        assert positions[0] != positions[1], "lead/trail spawned at identical position"
        speeds = [env._vehicles[vid].speed for vid in members]
        assert all(v > 0 for v in speeds)


def test_step_runs_and_reports_zone_transitions():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=7))
    env.reset(seed=7)
    active = env.active_vehicle_ids
    actions = {vid: 0 for vid in active}  # MAINTAIN
    obs, reward, terminated, truncated, info = env.step(actions)
    assert set(obs.keys()) == set(active)
    assert set(reward.keys()) == set(active)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert set(info["zone_t"].keys()) == set(active)
    assert set(info["zone_t1"].keys()) == set(active)
    for vid in active:
        assert info["zone_t"][vid] in ("pre", "merging", "post")


def test_forced_collision_between_two_shared_lane_vehicles_is_detected():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=2))
    env.reset(seed=2)
    active = env.active_vehicle_ids  # ("V0","V1","V2","V3") at the default n_vehicles=4
    # Force a controlled scenario: two vehicles on the same role (both
    # already in the shared lane), placed at an identical position.
    for vid in active:
        env._roles[vid] = "mainline" if vid in ("V0", "V1") else "ramp"
        env._vehicles[vid].role = env._roles[vid]
    env._vehicles["V0"].route_position = 200.0
    env._vehicles["V1"].route_position = 200.0
    env._vehicles["V0"].speed = 15.0
    env._vehicles["V1"].speed = 15.0
    # Keep the other pair harmlessly far away in the pre-merge (laterally offset) zone.
    env._vehicles["V2"].route_position = 0.0
    env._vehicles["V3"].route_position = 5.0

    actions = {vid: 0 for vid in active}  # MAINTAIN -- both move identically
    obs, reward, terminated, truncated, info = env.step(actions)

    assert info["collision_event"] is True
    assert ("V0", "V1") in info["collision_pairs"]
    assert terminated is True
    assert truncated is False
    assert reward["V0"] < 0.0
    assert reward["V1"] < 0.0


def test_action_mask_always_all_legal():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=0))
    env.reset(seed=0)
    for vid in env.active_vehicle_ids:
        mask = env.action_mask(vid)
        assert mask.dtype == bool
        assert mask.shape == (3,)
        assert mask.all()


# --- Speed-asymmetry option (STAGE11_PROTOCOL.md Sec 12.4) ---------------


def test_target_speed_ramp_defaults_to_none_and_is_unvalidated_when_unset():
    cfg = Stage10MergeEnvConfig(n_vehicles=2)
    cfg.validate()  # must not raise
    assert cfg.target_speed_ramp is None
    assert cfg.spawn_speed_ramp is None


def test_target_speed_ramp_must_be_positive():
    cfg = Stage10MergeEnvConfig(n_vehicles=2, target_speed_ramp=0.0)
    with pytest.raises(ValueError, match="target_speed_ramp"):
        cfg.validate()
    cfg2 = Stage10MergeEnvConfig(n_vehicles=2, target_speed_ramp=-5.0)
    with pytest.raises(ValueError, match="target_speed_ramp"):
        cfg2.validate()


def test_spawn_speed_ramp_rejects_negative():
    cfg = Stage10MergeEnvConfig(n_vehicles=2, spawn_speed_ramp=-1.0)
    with pytest.raises(ValueError, match="spawn_speed_ramp"):
        cfg.validate()


def test_spawn_speed_ramp_zero_is_allowed():
    cfg = Stage10MergeEnvConfig(n_vehicles=2, spawn_speed_ramp=0.0)
    cfg.validate()  # must not raise -- a ramp vehicle starting from a stop is a valid scenario


def test_default_spawn_speed_is_identical_for_both_roles():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(n_vehicles=2, seed=0))
    _, info = env.reset(seed=0)
    roles = dict(info["roles"])
    speeds = {vid: env._vehicles[vid].speed for vid in roles}  # noqa: SLF001
    assert len(set(speeds.values())) == 1  # both roles spawn at the same speed by default


def test_spawn_speed_ramp_only_affects_ramp_role_vehicles():
    cfg = Stage10MergeEnvConfig(n_vehicles=2, seed=0, spawn_speed=18.0, spawn_speed_ramp=10.0)
    env = Stage10SymmetricMergeEnv(cfg)
    _, info = env.reset(seed=0)
    roles = dict(info["roles"])
    for vid, role in roles.items():
        speed = env._vehicles[vid].speed  # noqa: SLF001
        if role == "ramp":
            assert speed == pytest.approx(10.0)
        else:
            assert speed == pytest.approx(18.0)


def test_spawn_speed_ramp_generalises_to_four_and_six_vehicles():
    for n in (2, 4, 6):
        cfg = Stage10MergeEnvConfig(n_vehicles=n, seed=0, spawn_speed=18.0, spawn_speed_ramp=10.0)
        env = Stage10SymmetricMergeEnv(cfg)
        _, info = env.reset(seed=0)
        roles = dict(info["roles"])
        for vid, role in roles.items():
            speed = env._vehicles[vid].speed  # noqa: SLF001
            expected = 10.0 if role == "ramp" else 18.0
            assert speed == pytest.approx(expected), (n, vid, role)
