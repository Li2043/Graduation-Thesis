"""Tests for MergeEnvCandidateV3 physics substeps and kinematics."""

from __future__ import annotations

import math

import numpy as np

from thesis.certification.choice_state_scenarios import (
    build_environment_candidates,
    build_ic_blocks,
    materialize_block_for_geometry,
)
from thesis.envs.final_environment_config import TimingConfig, TargetSpeeds
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config
from thesis.envs.vehicle_dynamics import bumper_gap_along_x, time_to_collision
from thesis.rewards.base_reward_v2 import LEARNING_CONTROLLERS, STAKEHOLDER_SET


def _env(seed: int = 1001):
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    cfg = MergeEnvCandidateV3Config(candidate=cand, block=block, timing=TimingConfig())
    env = MergeEnvCandidateV3(cfg)
    env.reset(seed=seed)
    return env


def test_physics_substeps_equal_four():
    env = _env()
    _o, _r, _t, _tr, info = env.step({"A": 0, "B": 0})
    assert info["physics_substeps"] == 4
    assert len(info["substep_records"]) == 4


def test_deterministic_reset_and_trajectory():
    env1 = _env(42)
    env2 = _env(42)
    acts = [{"A": 1, "B": 2}, {"A": 0, "B": 0}, {"A": 1, "B": 1}]
    t1, t2 = [], []
    for a in acts:
        _o, _r, term, trunc, info = env1.step(a)
        t1.append((info["vehicles_t1"]["A"]["route_position"], info["vehicles_t1"]["A"]["speed"]))
        if term or trunc:
            break
    env2.reset(seed=42)
    for a in acts:
        _o, _r, term, trunc, info = env2.step(a)
        t2.append((info["vehicles_t1"]["A"]["route_position"], info["vehicles_t1"]["A"]["speed"]))
        if term or trunc:
            break
    assert t1 == t2


def test_route_continuity_across_substeps():
    env = _env()
    _o, _r, _t, _tr, info = env.step({"A": 1, "B": 1})
    prev = None
    for rec in info["substep_records"]:
        rp = rec["vehicles"]["A"]["route_position"]
        if prev is not None:
            assert rp + 1e-9 >= prev  # no reverse
            assert abs(rp - prev) < 5.0  # continuity within substep
        prev = rp


def test_collision_detection_inside_policy_interval():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    # Place learners overlapping on shared mainline
    block.spawn_route_mainline = cand.geometry.merge_start + 10.0
    block.spawn_route_ramp = cand.geometry.merge_start + 10.0
    block.spawn_speed_mainline = 20.0
    block.spawn_speed_ramp = 20.0
    env = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=block))
    env.reset(seed=1)
    # Force same world x by setting after reset
    env._vehicles["A"].role = "mainline"
    env._vehicles["B"].role = "mainline"
    env._vehicles["A"].route_position = cand.geometry.merge_start + 20.0
    env._vehicles["B"].route_position = cand.geometry.merge_start + 20.0
    env._sync(env._vehicles["A"])
    env._sync(env._vehicles["B"])
    pairs = env._detect_collisions()
    assert pairs


def test_bumper_gap_and_ttc():
    gap = bumper_gap_along_x(0.0, 10.0, vehicle_length=5.0)
    assert abs(gap - 5.0) < 1e-12
    assert time_to_collision(gap=10.0, v_rear=20.0, v_front=15.0) == 2.0
    assert time_to_collision(gap=10.0, v_rear=10.0, v_front=15.0) is None


def test_fixed_stakeholder_registry_and_roles():
    env = _env()
    assert set(STAKEHOLDER_SET) == {"A", "B", "B_front", "B_rear"}
    assert set(LEARNING_CONTROLLERS) == {"A", "B"}
    assert env.config.block.role_A in ("mainline", "ramp")
    assert env.config.block.role_A != env.config.block.role_B


def test_target_speed_invariance():
    env = _env()
    ts = TargetSpeeds().as_map()
    for k, v in ts.items():
        assert env._vehicles[k].target_speed == v
    env.step({"A": 1, "B": 2})
    for k, v in ts.items():
        assert env._vehicles[k].target_speed == v


def test_no_reverse_motion():
    env = _env()
    for _ in range(10):
        _o, _r, term, trunc, info = env.step({"A": 2, "B": 2})
        for aid in ("A", "B"):
            assert info["vehicles_t1"][aid]["speed"] >= -1e-12
        if term or trunc:
            break


def test_core_reward_excludes_hard_brake_term():
    env = _env()
    _o, rew, _t, _tr, info = env.step({"A": 2, "B": 2})
    for aid in ("A", "B"):
        c = info["components"][aid]
        assert set(c) >= {"progress_component", "exit_component", "collision_component", "core_reward"}
        assert "hard_brake" not in c
        assert math.isfinite(rew[aid])


def test_fixture_flag_false():
    env = _env()
    _o, _r, _t, _tr, info = env.step({"A": 0, "B": 0})
    assert info["fixture_only"] is False
