"""Tests for Markovian final observations."""

from __future__ import annotations

import numpy as np

from thesis.certification.choice_state_scenarios import (
    build_environment_candidates,
    build_ic_blocks,
    materialize_block_for_geometry,
)
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config


def _env():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    env = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=block))
    env.reset(seed=block.seed)
    return env


def test_observation_dim_and_space_match():
    env = _env()
    obs, _ = env.reset(seed=1)
    for aid in ("A", "B"):
        assert obs[aid].shape == (OBSERVATION_DIM,)
        assert env.observation_space[aid].shape == (OBSERVATION_DIM,)
        assert np.all(np.isfinite(obs[aid]))


def test_peer_position_changes_observation():
    env = _env()
    obs0, _ = env.reset(seed=1)
    peer_role = env._vehicles["B"].role
    env._vehicles["B"].route_position += 8.0
    env._sync(env._vehicles["B"])
    obs1 = env._obs()
    assert not np.allclose(obs0["A"], obs1["A"])


def test_background_position_changes_observation():
    env = _env()
    obs0, _ = env.reset(seed=1)
    env._vehicles["B_front"].route_position += 12.0
    env._sync(env._vehicles["B_front"])
    obs1 = env._obs()
    assert not np.allclose(obs0["A"], obs1["A"])


def test_peer_speed_changes_observation():
    env = _env()
    obs0, _ = env.reset(seed=1)
    env._vehicles["B"].speed += 3.0
    obs1 = env._obs()
    assert not np.allclose(obs0["A"], obs1["A"])


def test_role_swap_permutes_observations():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    block.role_A = "mainline"
    block.role_B = "ramp"
    env1 = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=block))
    obs1, _ = env1.reset(seed=block.seed)

    block2 = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    block2.role_A = "ramp"
    block2.role_B = "mainline"
    env2 = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=block2))
    obs2, _ = env2.reset(seed=block.seed)

    # Controller A under swapped labels should match former B physical role features
    # Own role indicator is index 4
    assert obs1["A"][4] == 1.0 and obs1["B"][4] == -1.0
    assert obs2["A"][4] == -1.0 and obs2["B"][4] == 1.0
    # Peer relative geometry for A in env1 ≈ peer relative for B in env2 (same physical pair)
    assert np.allclose(obs1["A"][24:27], obs2["B"][24:27], atol=1e-5)


def test_completed_stakeholders_remain_represented():
    env = _env()
    env.reset(seed=1)
    env._vehicles["B_front"].completed = True
    env._vehicles["B_front"].active_on_road = False
    env._vehicles["B_front"].physical_segment = "exited"
    obs = env._obs()
    # B_front active flag at index 14
    assert obs["A"][14] == 0.0
    assert np.all(np.isfinite(obs["A"]))
