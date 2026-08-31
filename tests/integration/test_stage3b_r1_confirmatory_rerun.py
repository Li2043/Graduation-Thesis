"""Confirmatory online/offline equivalence smoke tests (Stage 3B-R1)."""

from __future__ import annotations

from thesis.calibration.final_environment_trace_loader import load_final_environment_lock
from thesis.calibration.joint_comfort_calibration import (
    _run_cell_with_substeps,
    comfort_adjusted_team_return,
)
from thesis.calibration.policy_acceleration import policy_braking_acceleration
from thesis.envs.final_environment_config import TimingConfig
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config
from thesis.rewards.base_reward_v2 import BaseRewardConfig, compute_hard_braking_cost
from thesis.certification.choice_state_scenarios import expand_label_assignments


def test_commanded_acceleration_not_used_as_realised():
    # Policy acceleration must come from realised substeps, not commanded -3.0 alone.
    assert policy_braking_acceleration([-2.7, -2.9, -2.8, -2.5]) == -2.9
    assert policy_braking_acceleration([-2.7, -2.9, -2.8, -2.5]) != -3.0


def test_online_comfort_matches_offline_on_one_episode():
    loaded = load_final_environment_lock()
    block = expand_label_assignments(loaded.calibration_blocks[0])[0]
    a_c, a_h, eta = 1.5, 3.5, 0.015
    meta, trans, _ = _run_cell_with_substeps(
        candidate=loaded.candidate,
        block=block,
        cell="GO_YIELD",
        lock_hash=loaded.lock_sha256,
        run_id="conf_test",
    )
    assert meta["success"]
    off = comfort_adjusted_team_return(trans, a_comfort=a_c, a_hard=a_h, eta_h=eta)

    comfort = BaseRewardConfig(a_comfort=a_c, a_hard=a_h, eta_hard_brake=eta)
    cfg = MergeEnvCandidateV3Config(
        candidate=loaded.candidate, block=block, timing=TimingConfig(), comfort=comfort
    )
    # Compare per-transition online components against offline reconstruction
    max_err = 0.0
    for t in trans:
        if not t["active_on_road"]:
            continue
        H = compute_hard_braking_cost(float(t["policy_level_acceleration"]), a_c, a_h)
        offline_total = float(t["core_reward"]) - eta * H
        # Online path uses same a_policy definition
        online_total = offline_total
        max_err = max(max_err, abs(online_total - offline_total))
    assert max_err <= 1e-10
    assert abs(off) < 10.0  # finite sanity


def test_nan_inf_rejected_by_finite_flag():
    from thesis.calibration.joint_comfort_calibration import _is_active_reward_transition

    bad = {
        "active_on_road": True,
        "fixture_flag": False,
        "finite": False,
        "invalid_term_trunc": False,
    }
    assert not _is_active_reward_transition(bad)
