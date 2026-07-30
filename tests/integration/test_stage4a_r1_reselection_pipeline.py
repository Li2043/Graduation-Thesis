"""Stage 4A-R1 integration tests — real selection / lock / integrity paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thesis.certification.choice_state_certification import certify_block
from thesis.certification.choice_state_scenarios import (
    build_environment_candidates,
    build_ic_blocks,
    expand_label_assignments,
    materialize_block_for_geometry,
)
from thesis.certification.environment_candidate_selection import (
    build_final_environment_lock,
    select_environment_candidate,
)
from thesis.envs.final_environment_config import TimingConfig
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config


@pytest.fixture(scope="module")
def selection_result():
    return select_environment_candidate()


def test_four_physics_substeps_locked():
    t = TimingConfig()
    assert t.physics_dt == 0.05
    assert t.policy_interval == 0.20
    assert t.physics_substeps_per_action == 4


def test_observation_27d_ordering_unchanged():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    env = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=block))
    obs, _ = env.reset(seed=block.seed)
    assert obs["A"].shape == (OBSERVATION_DIM,)
    assert OBSERVATION_DIM == 27
    assert np.all(np.isfinite(obs["A"]))


def test_both_label_assignments_evaluated():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    assigns = expand_label_assignments(block)
    assert len(assigns) == 2
    assert {a.role_A for a in assigns} == {"mainline", "ramp"}
    result = certify_block(cand, block)
    assert result["label_swap_max_error"] <= 1e-12


def test_calibration_only_selection_and_lowest_priority_feasible(selection_result):
    result = selection_result
    assert result["selection_used_validation"] is False
    assert result["feasible_candidate_ids"], "expected at least one feasible candidate"
    cands = {c.candidate_id: c.priority_rank for c in build_environment_candidates()}
    selected_id = result["selected_candidate"]["candidate_id"]
    ordered = [
        c.candidate_id
        for c in sorted(build_environment_candidates(), key=lambda x: x.priority_rank)
    ]
    first_feasible = next(i for i in ordered if i in result["feasible_candidate_ids"])
    assert selected_id == first_feasible
    assert cands[selected_id] == min(cands[i] for i in result["feasible_candidate_ids"])


def test_validation_failure_does_not_trigger_reselection(selection_result):
    result = selection_result
    assert result["selection_used_validation"] is False
    if result.get("validation") and not result["validation"]["pass"]:
        assert any(
            f.get("note") == "holdout_failure_no_reselection" for f in result.get("failures", [])
        )
        assert result["environment_parameters_final"] is False


def test_no_comfort_component_in_ranking_keys(selection_result):
    for row in selection_result["candidates"]:
        assert "eta" not in row
        assert "comfort" not in row
        assert "hard_brake_penalty" not in row


def test_no_dqn_or_optimiser_update_occurs():
    cand = build_environment_candidates()[0]
    block = materialize_block_for_geometry(build_ic_blocks()[0][0], cand.geometry)
    env = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=block))
    env.reset(seed=1)
    env.step({"A": 1, "B": 0})
    for name in ("update", "learn", "optimizer", "optimiser", "learner", "replay_buffer"):
        assert not hasattr(env, name)


def test_old_lock_not_reused_for_new_lock_metadata():
    sel = build_environment_candidates()[0].to_dict()
    cal, val = build_ic_blocks()
    lock = build_final_environment_lock(
        selected=sel,
        calibration_blocks=[b.to_dict() for b in cal],
        validation_blocks=[b.to_dict() for b in val],
        git_commit="test",
        config_hashes={"stage4a_r1": "abc"},
    )
    assert lock["superseded_stage4a_run_id"] == "20260729T231946Z_c8d92bc3"
    assert lock["superseded_lock_sha256"].startswith("d5614d41")
    assert lock["observation_dimension"] == 27
    assert "quintic" in lock["route_geometry_version"]
    assert lock["environment_parameters_final"] is True
    assert lock["comfort_parameters_final"] is False
    assert lock["policy_training_started"] is False


def test_lock_written_only_after_full_pass_semantics(selection_result):
    result = selection_result
    if result["overall"] == "PASS":
        assert result["environment_parameters_final"] is True
    else:
        assert result["environment_parameters_final"] is False


def test_integrity_counts_zero_on_selected_candidate(selection_result):
    row = next(r for r in selection_result["candidates"] if r.get("selected"))
    for k in (
        "route_discontinuity_count",
        "repeated_exit_count",
        "invalid_flag_count",
        "nan_inf_count",
        "fixture_count",
    ):
        assert int(row[k]) == 0


def test_prior_hardening_artifacts_intact():
    paths = [
        Path(
            "experiments/pre_impl/stage4a_environment_choice_state/artifacts/"
            "20260729T231946Z_c8d92bc3/final_environment_lock.yaml"
        ),
        Path(
            "experiments/pre_impl/stage4a0r_v3_physics_hardening/artifacts/"
            "20260729T235855Z_8ab30c89/manifest.json"
        ),
        Path(
            "experiments/pre_impl/stage4a0r2_merge_geometry_correction/artifacts/"
            "20260730T000945Z_d0051f73/manifest.json"
        ),
    ]
    for p in paths:
        assert p.is_file(), f"missing historical artifact {p}"
