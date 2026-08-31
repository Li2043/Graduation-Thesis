"""Stage 4A-0R2 geometry correction integration tests."""

from __future__ import annotations

import numpy as np

from thesis.certification.choice_state_certification import certify_block
from thesis.certification.choice_state_scenarios import (
    GEOMETRY,
    build_environment_candidates,
    build_ic_blocks,
    materialize_block_for_geometry,
)
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.envs.final_route_geometry import MAX_LATERAL_ACCEL_AT_20, build_final_route_geometry
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config


def test_no_quarter_circle_semantics_remain():
    geom = build_final_route_geometry(GEOMETRY[0])
    # Old bug: ramp joined at merge_start with large heading (~π/2). Now heading≈0.
    p = geom.pose("ramp", geom.merge_start + 0.01)
    assert abs(p.heading) < 0.05
    assert p.segment == "merge_connector"
    # Must still be offset near start of connector
    assert p.y < -geom.lateral_offset + 0.5


def test_observation_27d_finite_under_corrected_geometry():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    env = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=block))
    obs, _ = env.reset(seed=block.seed)
    assert obs["A"].shape == (OBSERVATION_DIM,)
    assert np.all(np.isfinite(obs["A"]))
    env.step({"A": 1, "B": 0})
    obs2 = env._obs()
    assert np.all(np.isfinite(obs2["B"]))


def test_label_swap_physical_invariance():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][1], cand.geometry)
    result = certify_block(cand, block)
    assert result["label_swap_max_error"] <= 1e-12


def test_all_geometries_feasible_and_distinct():
    diags = [build_final_route_geometry(g).diagnostics() for g in GEOMETRY]
    assert diags[0]["connector_world_x_length"] < diags[1]["connector_world_x_length"]
    assert diags[2]["merge_start"] == 100.0
    for d in diags:
        assert d["maximum_implied_lateral_acceleration_at_20"] <= MAX_LATERAL_ACCEL_AT_20 + 1e-12
        assert d["physically_feasible"]


def test_no_dqn_or_optimiser_on_step():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    env = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=block))
    env.reset(seed=1)
    env.step({"A": 0, "B": 0})
    assert not hasattr(env, "optimizer")
    assert not hasattr(env, "optimiser")
    assert not hasattr(env, "learner")


def test_prior_stage4a0r_artifacts_intact():
    from pathlib import Path

    p = Path(
        "experiments/pre_impl/stage4a0r_v3_physics_hardening/artifacts/"
        "20260729T235855Z_8ab30c89/manifest.json"
    )
    assert p.is_file()
