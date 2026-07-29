"""Integration: confirmatory env rerun matches offline reconstruction."""

from __future__ import annotations

from pathlib import Path

from thesis.audits.audit_scenarios import build_block_scenarios, build_matched_blocks
from thesis.audits.base_outcome_audit import run_audit_scenario
from thesis.calibration.comfort_calibration import (
    confirmatory_scripted_rerun,
    reconstruct_scenario_returns,
)
from thesis.calibration.trace_loader import load_and_validate_stage3a_source
from thesis.rewards.base_reward_v2 import BaseRewardConfig


REPO = Path(__file__).resolve().parents[2]
STAGE3A_RUN = "20260729T222933Z_3b07a818"
STAGE3A_COMMIT = "3b07a81879e913a175bfd05f8c985fc095841d34"


def test_confirmatory_rerun_matches_offline_reconstruction():
    """Even if Stage 3B selection FAILs, reconstruction↔env agreement must hold."""
    _, transitions, _, _ = load_and_validate_stage3a_source(
        repo_root=REPO,
        stage3a_run_id=STAGE3A_RUN,
        expected_git_commit=STAGE3A_COMMIT,
    )
    a_c, a_h, eta = 2.0, 4.0, 0.02
    offline = reconstruct_scenario_returns(
        transitions, a_comfort=a_c, a_hard=a_h, eta=eta, gamma=0.995
    )
    conf = confirmatory_scripted_rerun(
        a_comfort=a_c, a_hard=a_h, eta=eta, gamma=0.995, offline_recon=offline
    )
    assert conf["max_abs_return_difference"] <= 1e-10
    assert conf["ok"] is True


def test_reward_params_do_not_change_scripted_kinematics():
    block = build_matched_blocks()[0]
    sc = next(s for s in build_block_scenarios(block) if s.scenario_id == "safe_mainline_first")
    sc.config.base_reward = BaseRewardConfig(
        eta_hard_brake=0.02, a_comfort=1.5, a_hard=4.0
    )
    o1 = run_audit_scenario(sc, run_id="kin1", gamma=0.995)
    sc.config.base_reward = BaseRewardConfig(
        eta_hard_brake=0.15, a_comfort=2.5, a_hard=7.0
    )
    o2 = run_audit_scenario(sc, run_id="kin2", gamma=0.995)
    assert o1.episode_length == o2.episode_length
    acc1 = [t["realised_acceleration"] for t in o1.transitions if t["controller_id"] == "A"]
    acc2 = [t["realised_acceleration"] for t in o2.transitions if t["controller_id"] == "A"]
    assert acc1 == acc2


def test_prior_stage3a_outputs_untouched():
    raw = (
        REPO
        / "experiments/pre_impl/stage3a_scripted_base_outcome_audit/data/raw"
        / STAGE3A_RUN
        / "transition_trace.jsonl"
    )
    assert raw.is_file()
    # size sanity — must remain the large frozen trace
    assert raw.stat().st_size > 1_000_000


def test_block_level_ordering_not_averaged_away():
    from thesis.audits.audit_metrics import check_incentive_ordering

    # One good block and one violation must surface as not-ok for the bad block
    bad = check_incentive_ordering(
        "block_bad",
        g_safe_mainline=1.0,
        g_safe_ramp=1.0,
        g_slow_mainline=0.9,
        g_slow_ramp=0.9,
        g_stall_partial=1.5,  # violates safe > stall
        g_early_coll=-1.0,
        g_late_coll=-1.0,
    )
    assert bad.ok is False
    assert any("safe_not_above_stall" in v for v in bad.violations)
