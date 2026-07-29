"""Regression tests for hardened MergeEnvCandidateV3."""

from __future__ import annotations

import math

import numpy as np

from thesis.certification.choice_state_scenarios import (
    build_environment_candidates,
    build_ic_blocks,
    materialize_block_for_geometry,
)
from thesis.envs.final_environment_config import TargetSpeeds, TimingConfig
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config
from thesis.rewards.base_reward_v2 import LEARNING_CONTROLLERS, STAKEHOLDER_SET


def _env(seed: int = 1001):
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    cfg = MergeEnvCandidateV3Config(candidate=cand, block=block, timing=TimingConfig())
    env = MergeEnvCandidateV3(cfg)
    env.reset(seed=seed)
    return env, cand


def test_physics_substeps_equal_four():
    env, _ = _env()
    _o, _r, _t, _tr, info = env.step({"A": 0, "B": 0})
    assert info["physics_substeps"] == 4
    assert len(info["substep_records"]) == 4


def test_deterministic_reset_and_trajectory():
    env1, _ = _env(42)
    env2, _ = _env(42)
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


def test_commanded_vs_realised_acceleration_logged():
    env, _ = _env()
    _o, _r, _t, _tr, info = env.step({"A": 1, "B": 2})
    va = info["vehicles_t1"]["A"]
    assert "commanded_acceleration" in va and "realised_acceleration" in va
    assert abs(va["commanded_acceleration"] - 2.0) < 1e-12
    assert math.isfinite(va["realised_acceleration"])


def test_merge_conflict_rectangle_collision_and_exact_pair():
    env, cand = _env()
    # Place both learners overlapping in merge_conflict on mainline corridor
    env._vehicles["A"].role = "mainline"
    env._vehicles["B"].role = "mainline"
    env._vehicles["A"].route_position = 0.5 * (cand.geometry.merge_start + cand.geometry.merge_end)
    env._vehicles["B"].route_position = env._vehicles["A"].route_position + 1.0
    env._vehicles["A"].active_on_road = True
    env._vehicles["B"].active_on_road = True
    env._sync(env._vehicles["A"])
    env._sync(env._vehicles["B"])
    pairs = env._detect_collisions()
    assert pairs == [("A", "B")]


def test_tunnelling_collision_through_env_step():
    """Overlap occurs mid-policy-step; coarse 0.20s endpoints would miss it."""
    from thesis.envs.final_environment_config import LearningDynamics

    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    cfg = MergeEnvCandidateV3Config(
        candidate=cand,
        block=block,
        timing=TimingConfig(),
        dynamics=LearningDynamics(v_max=80.0),
    )
    env = MergeEnvCandidateV3(cfg)
    env.reset(seed=1)
    x0 = cand.geometry.merge_end + 20.0
    env._vehicles["A"].role = "mainline"
    env._vehicles["B"].role = "mainline"
    env._vehicles["A"].route_position = x0 + 8.0
    env._vehicles["B"].route_position = x0
    env._vehicles["A"].speed = 0.0
    env._vehicles["B"].speed = 70.0
    env._vehicles["A"].active_on_road = True
    env._vehicles["B"].active_on_road = True
    env._sync(env._vehicles["A"])
    env._sync(env._vehicles["B"])
    assert not env._detect_collisions()
    # Coarse single 0.20s advance (no substeps): B travels ~14m → ahead by ~6m → no overlap
    b_end = x0 + 70.0 * 0.20
    a_end = x0 + 8.0
    assert abs(b_end - a_end) > 5.0
    _obs, _rew, term, _trunc, info = env.step({"A": 0, "B": 0})
    assert term
    assert info["term_reason"] == "collision"
    assert info["events"]["collision_pairs"] == [["A", "B"]]
    assert any(rec["collision_pairs"] for rec in info["substep_records"])


def test_immediate_substep_exit_removal():
    env, cand = _env()
    exit_s = cand.geometry.downstream_exit
    env._vehicles["A"].role = "mainline"
    env._vehicles["A"].route_position = exit_s - 0.5
    env._vehicles["A"].speed = 20.0
    env._vehicles["A"].active_on_road = True
    env._sync(env._vehicles["A"])
    # Keep B far away
    env._vehicles["B"].route_position = 10.0
    env._sync(env._vehicles["B"])
    _o, _r, _t, _tr, info = env.step({"A": 1, "B": 0})
    assert env._vehicles["A"].completed
    assert env._vehicles["A"].active_on_road is False
    assert env._vehicles["A"].physical_segment == "exited"
    assert info["exit_substep"]["A"] is not None
    assert env._exit_count["A"] == 1


def test_collision_over_exit_same_substep():
    env, cand = _env()
    # Overlapping near exit threshold: same substep would also cross exit,
    # but collision must take precedence (no exit credit / success).
    exit_s = cand.geometry.downstream_exit
    env._vehicles["A"].role = "mainline"
    env._vehicles["B"].role = "mainline"
    env._vehicles["A"].route_position = exit_s - 0.4
    env._vehicles["B"].route_position = exit_s - 0.2
    env._vehicles["A"].speed = 5.0
    env._vehicles["B"].speed = 5.0
    for aid in ("A", "B"):
        env._vehicles[aid].active_on_road = True
        env._vehicles[aid].completed = False
        env._sync(env._vehicles[aid])
    assert env._detect_collisions() == [("A", "B")]
    _o, _r, term, _tr, info = env.step({"A": 0, "B": 0})
    assert term and info["term_reason"] == "collision"
    assert info["events"]["collision_pairs"] == [["A", "B"]]
    # Collision precedence: exit events must not be awarded in that substep win path
    assert info["events"]["exit_event"]["A"] == 0.0
    assert info["events"]["exit_event"]["B"] == 0.0


def test_later_collision_cannot_involve_exited():
    env, cand = _env()
    exit_s = cand.geometry.downstream_exit
    env._vehicles["A"].role = "mainline"
    env._vehicles["A"].route_position = exit_s - 0.2
    env._vehicles["A"].speed = 30.0
    env._vehicles["B"].role = "mainline"
    env._vehicles["B"].route_position = exit_s - 40.0
    env._vehicles["B"].speed = 30.0
    for aid in ("A", "B"):
        env._vehicles[aid].active_on_road = True
        env._sync(env._vehicles[aid])
    # Step: A should exit in early substep; B must not collide with exited A
    for _ in range(5):
        _o, _r, term, trunc, info = env.step({"A": 1, "B": 1})
        if env._vehicles["A"].completed:
            assert env._vehicles["A"].active_on_road is False
            pairs = info["events"]["collision_pairs"]
            for p in pairs:
                assert "A" not in p
            break


def test_background_completion():
    env, cand = _env()
    exit_s = cand.geometry.downstream_exit
    env._vehicles["B_front"].route_position = exit_s - 0.5
    env._vehicles["B_front"].speed = 25.0
    env._vehicles["B_front"].active_on_road = True
    env._sync(env._vehicles["B_front"])
    _o, _r, _t, _tr, info = env.step({"A": 0, "B": 0})
    assert env._vehicles["B_front"].completed
    assert env._vehicles["B_front"].active_on_road is False
    assert env._vehicles["B_front"].physical_segment == "exited"
    assert "B_front" in STAKEHOLDER_SET
    assert info["events"]["exit_event_all"]["B_front"] == 1.0


def test_completed_excluded_from_idm_leader():
    env, _ = _env()
    env._vehicles["A"].role = "mainline"
    env._vehicles["B_front"].role = "mainline"
    env._vehicles["B_front"].route_position = 150.0
    env._vehicles["A"].route_position = 140.0
    env._sync(env._vehicles["A"])
    env._sync(env._vehicles["B_front"])
    lid, _ = env._leader_for("A")
    assert lid == "B_front"
    env._mark_exit("B_front", policy_step=1, substep=0)
    lid2, _ = env._leader_for("A")
    assert lid2 != "B_front"


def test_no_repeated_exits_invalid_flags_nan():
    env, cand = _env()
    exit_s = cand.geometry.downstream_exit
    env._vehicles["A"].route_position = exit_s - 0.3
    env._vehicles["A"].speed = 20.0
    env._sync(env._vehicles["A"])
    env.step({"A": 1, "B": 0})
    env.step({"A": 1, "B": 0})
    assert env._exit_count["A"] == 1
    obs, _r, term, trunc, info = env.step({"A": 0, "B": 0})
    assert not (term and trunc)
    assert info.get("nan_count", 0) == 0
    assert np.all(np.isfinite(obs["A"]))


def test_target_speed_invariance_and_registry():
    env, _ = _env()
    assert set(STAKEHOLDER_SET) == {"A", "B", "B_front", "B_rear"}
    assert set(LEARNING_CONTROLLERS) == {"A", "B"}
    ts = TargetSpeeds().as_map()
    for k, v in ts.items():
        assert env._vehicles[k].target_speed == v


def test_observation_dim():
    env, _ = _env()
    obs, _ = env.reset(seed=1)
    assert obs["A"].shape == (OBSERVATION_DIM,)
