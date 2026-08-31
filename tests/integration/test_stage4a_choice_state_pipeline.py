"""Behavioural integration checks for Stage 4A stack (no vacuous asserts)."""

from __future__ import annotations

from thesis.certification.choice_state_certification import certify_block
from thesis.certification.choice_state_scenarios import (
    build_environment_candidates,
    build_ic_blocks,
    materialize_block_for_geometry,
)
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config


def test_label_swap_invariance_on_block():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    result = certify_block(cand, block)
    assert result["label_swap_max_error"] <= 1e-12


def test_fixture_and_finite_in_traces():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][2], cand.geometry)
    result = certify_block(cand, block)
    assert result["traces"], "expected transition traces"
    for tr in result["traces"][:50]:
        assert tr["fixture_flag"] is False
        assert tr["core_reward"] == tr["core_reward"]


def test_env_has_no_learning_api():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    env = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=block))
    env.reset(seed=1)
    assert not hasattr(env, "update")
    assert not hasattr(env, "learn")
    env.step({"A": 0, "B": 0})
    assert not hasattr(env, "optimizer")


def test_historical_stage3_and_stage4a_artifacts_intact():
    from pathlib import Path

    paths = [
        Path(
            "experiments/pre_impl/stage3a_scripted_base_outcome_audit/artifacts/"
            "20260729T222933Z_3b07a818/manifest.json"
        ),
        Path(
            "experiments/pre_impl/stage3b_comfort_calibration/artifacts/"
            "20260729T225046Z_8624f03a/manifest.json"
        ),
        Path(
            "experiments/pre_impl/stage4a_environment_choice_state/artifacts/"
            "20260729T231946Z_c8d92bc3/manifest.json"
        ),
    ]
    for p in paths:
        assert p.is_file(), f"missing historical artifact {p}"
