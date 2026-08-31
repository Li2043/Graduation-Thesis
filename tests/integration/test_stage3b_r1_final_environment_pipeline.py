"""Stage 3B-R1 integration tests against the retained Stage 4A-R1 lock."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.calibration.final_environment_trace_loader import (
    EnvironmentLockError,
    load_final_environment_lock,
)
from thesis.calibration.joint_comfort_calibration import build_complete_tuples
from thesis.certification.holdout_signatures import find_duplicate_signatures
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config


def test_final_environment_lock_hash_and_g1i1():
    loaded = load_final_environment_lock()
    assert loaded.candidate.candidate_id == "G1-I1"
    assert loaded.lock_sha256.startswith("d2d82ac0")
    assert loaded.lock["observation_dimension"] == 27
    assert loaded.lock["physics_substeps_per_action"] == 4
    assert len(loaded.calibration_blocks) == 12
    assert len(loaded.validation_blocks) == 8


def test_calibration_validation_physically_disjoint():
    loaded = load_final_environment_lock()
    assert find_duplicate_signatures(loaded.calibration_blocks, loaded.validation_blocks) == []


def test_lock_hash_mismatch_blocked(tmp_path: Path):
    src = Path(
        "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
        "20260730T003122Z_aee2d425/final_environment_lock.yaml"
    )
    bad = tmp_path / "final_environment_lock.yaml"
    bad.write_text(src.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
    sha = tmp_path / "final_environment_lock.sha256"
    sha.write_text("0" * 64 + "  final_environment_lock.yaml\n", encoding="utf-8")
    with pytest.raises(EnvironmentLockError):
        load_final_environment_lock(bad, sha256_path=sha)


def test_no_dqn_on_env_step():
    loaded = load_final_environment_lock()
    env = MergeEnvCandidateV3(
        MergeEnvCandidateV3Config(candidate=loaded.candidate, block=loaded.calibration_blocks[0])
    )
    env.reset(seed=1)
    env.step({"A": 0, "B": 0})
    for name in ("update", "optimizer", "optimiser", "learner", "replay_buffer"):
        assert not hasattr(env, name)


def test_complete_tuple_grid_frozen():
    assert len(build_complete_tuples()) == 266


def test_prior_stage3b_failure_artifact_intact():
    p = Path(
        "experiments/pre_impl/stage3b_comfort_calibration/artifacts/"
        "20260729T225046Z_8624f03a/manifest.json"
    )
    assert p.is_file()


def test_stage4a_r1_lock_not_modified_by_presence():
    lock = Path(
        "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
        "20260730T003122Z_aee2d425/final_environment_lock.sha256"
    )
    text = lock.read_text(encoding="utf-8")
    assert text.startswith("d2d82ac02feb5bb2f5217f8e399972b91bd56cce343f60861954f66d7f70bf12")
