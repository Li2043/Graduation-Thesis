"""Unit tests for Stage 3B comfort calibration."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.calibration.calibration_metrics import (
    assert_eta_penalty_monotone,
    assert_h_monotonicities,
    compare_threshold_lex,
    h_from_accel,
    summarise_h,
    threshold_pair_feasible,
    valid_threshold_pair,
)
from thesis.calibration.comfort_calibration import (
    ThresholdMetrics,
    build_hard_braking_windows,
    evaluate_threshold_pair,
    reconstruct_scenario_returns,
    select_eta,
    select_threshold_pair,
)
from thesis.calibration.trace_loader import (
    filter_active_calibration_transitions,
    load_and_validate_stage3a_source,
    sha256_file,
)
from thesis.calibration.comfort_calibration import EtaMetrics


REPO = Path(__file__).resolve().parents[2]
STAGE3A_RUN = "20260729T222933Z_3b07a818"
STAGE3A_COMMIT = "3b07a81879e913a175bfd05f8c985fc095841d34"


def test_source_stage3a_hash_validation():
    man, tr, out, op = load_and_validate_stage3a_source(
        repo_root=REPO,
        stage3a_run_id=STAGE3A_RUN,
        expected_git_commit=STAGE3A_COMMIT,
    )
    assert man.summary_overall == "PASS"
    assert man.summary_git_dirty is False
    assert man.policy_training_started is False
    assert len(tr) > 0
    p = man.raw_dir / "transition_trace.jsonl"
    assert sha256_file(p) == man.file_hashes["transition_trace.jsonl"]


def test_active_transition_filtering_and_fixture_exclusion():
    _, tr, _, _ = load_and_validate_stage3a_source(
        repo_root=REPO,
        stage3a_run_id=STAGE3A_RUN,
        expected_git_commit=STAGE3A_COMMIT,
    )
    included, excluded, stats = filter_active_calibration_transitions(tr)
    assert stats.included > 0
    assert stats.excluded_by_reason.get("fixture_only", 0) > 0
    assert all(not r.get("fixture_only") for r in included)
    assert all(r["scenario_id"].startswith("safe") or r["scenario_id"].startswith("slow") or r["scenario_id"] == "hard_braking_safe" for r in included)
    assert "inactive_completed_controller" in stats.excluded_by_reason


def test_hard_window_construction():
    _, tr, _, _ = load_and_validate_stage3a_source(
        repo_root=REPO,
        stage3a_run_id=STAGE3A_RUN,
        expected_git_commit=STAGE3A_COMMIT,
    )
    windows, rows, failures = build_hard_braking_windows(tr)
    assert failures == []
    blocks = {w.block_id for w in windows}
    assert len(blocks) == 8
    assert all(r["delta_braking_magnitude"] >= 1.0 - 1e-12 for r in rows)


def test_h_boundary_and_quadratic():
    assert h_from_accel(-2.0, 2.0, 6.0) == 0.0
    assert h_from_accel(-1.0, 2.0, 6.0) == 0.0
    assert abs(h_from_accel(-6.0, 2.0, 6.0) - 1.0) <= 1e-12
    assert abs(h_from_accel(-8.0, 2.0, 6.0) - 1.0) <= 1e-12
    expected = ((4.0 - 2.0) / (6.0 - 2.0)) ** 2
    assert abs(h_from_accel(-4.0, 2.0, 6.0) - expected) <= 1e-12


def test_monotonicity_suite():
    assert_h_monotonicities()
    assert_eta_penalty_monotone(0.25, [0.02, 0.04, 0.06, 0.1])


def test_valid_threshold_pair_gate():
    ok, reasons = valid_threshold_pair(3.0, 4.0)
    assert not ok
    assert "a_hard_minus_a_comfort_lt_1.5" in reasons
    ok2, _ = valid_threshold_pair(2.0, 4.0)
    assert ok2


def test_threshold_feasibility_and_selection_deterministic():
    # Synthetic: nominal mostly H=0, hard window H=0.5
    included = []
    for i in range(100):
        included.append(
            {
                "block_id": "block_001",
                "scenario_id": "safe_mainline_first",
                "controller_id": "A",
                "step": i + 1,
                "realised_acceleration": 0.0 if i < 95 else -3.0,
                "fixture_only": False,
            }
        )
    window_rows = [
        {
            "block_id": "block_001",
            "controller_id": "A",
            "step": j + 1,
            "realised_acceleration_hard": -5.0,
            "realised_acceleration_nominal": 0.0,
            "delta_braking_magnitude": 5.0,
        }
        for j in range(10)
    ]
    # Pad other blocks with identical windows for block-failure rule
    for b in range(2, 9):
        bid = f"block_00{b}"
        for j in range(10):
            window_rows.append(
                {
                    "block_id": bid,
                    "controller_id": "A",
                    "step": j + 1,
                    "realised_acceleration_hard": -5.0,
                    "realised_acceleration_nominal": 0.0,
                    "delta_braking_magnitude": 5.0,
                }
            )
        for i in range(20):
            included.append(
                {
                    "block_id": bid,
                    "scenario_id": "safe_mainline_first",
                    "controller_id": "A",
                    "step": i + 1,
                    "realised_acceleration": 0.0,
                    "fixture_only": False,
                }
            )

    blocks = [f"block_00{i}" for i in range(1, 9)]
    t1 = evaluate_threshold_pair(
        a_comfort=2.0, a_hard=6.0, included=included, window_rows=window_rows, blocks=blocks
    )
    t2 = evaluate_threshold_pair(
        a_comfort=1.5, a_hard=6.0, included=included, window_rows=window_rows, blocks=blocks
    )
    selected = select_threshold_pair([t1, t2])
    assert selected is not None
    # Higher separation wins
    assert selected.a_comfort in {1.5, 2.0}
    selected2 = select_threshold_pair([t1, t2])
    assert selected2 is not None
    assert selected2.a_comfort == selected.a_comfort
    assert selected2.a_hard == selected.a_hard


def test_select_smallest_feasible_eta():
    cands = [
        EtaMetrics(2, 6, 0.08, True, [], 0.04, 0.05, 0.03, 0.01, 0.02, 0, [], True),
        EtaMetrics(2, 6, 0.04, True, [], 0.03, 0.04, 0.025, 0.01, 0.02, 0, [], True),
        EtaMetrics(2, 6, 0.02, False, ["x"], 0.01, 0.02, 0.01, 0.01, 0.02, 0, [], True),
    ]
    best = select_eta(cands)
    assert best is not None
    assert best.eta_hard_brake == 0.04


def test_no_feasible_eta_returns_none():
    cands = [
        EtaMetrics(2, 6, 0.02, False, ["bad"], 0.0, 0.0, 0.0, None, None, 1, [], False),
    ]
    assert select_eta(cands) is None


def test_exact_reward_reconstruction():
    rows = [
        {
            "block_id": "block_001",
            "scenario_id": "safe_mainline_first",
            "controller_id": "A",
            "step": 1,
            "fixture_only": False,
            "progress_component": 0.01,
            "exit_component": 0.0,
            "collision_component": 0.0,
            "realised_acceleration": -4.0,
        },
        {
            "block_id": "block_001",
            "scenario_id": "safe_mainline_first",
            "controller_id": "B",
            "step": 1,
            "fixture_only": False,
            "progress_component": 0.02,
            "exit_component": 0.6,
            "collision_component": 0.0,
            "realised_acceleration": 0.0,
        },
    ]
    out = reconstruct_scenario_returns(rows, a_comfort=2.0, a_hard=6.0, eta=0.1, gamma=0.995)
    r = out[("block_001", "safe_mainline_first")]
    h = h_from_accel(-4.0, 2.0, 6.0)
    expected_brake = -0.1 * h  # only A
    # discounted: step1 disc=1
    assert abs(r["G_hard_braking"] - expected_brake) <= 1e-12
    assert abs(r["G_team"] - (0.01 + 0.02 + 0.6 + expected_brake)) <= 1e-12


def test_fixture_excluded_from_reconstruction():
    rows = [
        {
            "block_id": "block_001",
            "scenario_id": "fixture_collision_A",
            "controller_id": "A",
            "step": 1,
            "fixture_only": True,
            "progress_component": 0.0,
            "exit_component": 0.0,
            "collision_component": -1.0,
            "realised_acceleration": 0.0,
        }
    ]
    out = reconstruct_scenario_returns(rows, a_comfort=2.0, a_hard=6.0, eta=0.1, gamma=0.995)
    assert out == {}


def test_source_files_unchanged_digest_stable():
    man, _, _, _ = load_and_validate_stage3a_source(
        repo_root=REPO,
        stage3a_run_id=STAGE3A_RUN,
        expected_git_commit=STAGE3A_COMMIT,
    )
    p = man.raw_dir / "transition_trace.jsonl"
    d1 = sha256_file(p)
    d2 = sha256_file(p)
    assert d1 == d2 == man.file_hashes["transition_trace.jsonl"]


def test_lex_compare_tolerance():
    assert compare_threshold_lex((0.2, -0.01, 2.0, 6.0), (0.2, -0.01, 2.0, 6.0)) == 0
    assert compare_threshold_lex((0.21, -0.01, 2.0, 6.0), (0.2, -0.01, 2.0, 6.0)) == 1


def test_summarise_h_empty_and_nonzero():
    empty = summarise_h([])
    assert empty.n == 0
    dist = summarise_h([0.0, 0.25, 1.0])
    assert abs(dist.nonzero_rate - 2 / 3) < 1e-12
    assert abs(dist.saturation_rate - 1 / 3) < 1e-12


def test_threshold_pair_feasible_helper():
    from thesis.calibration.calibration_metrics import HDistribution

    nom = HDistribution(100, 0.05, 0.01, 0.0, 0.02, 0.0)
    hard = HDistribution(10, 0.9, 0.3, 0.25, 0.5, 0.0)
    ok, reasons = threshold_pair_feasible(
        nominal=nom, hard_window=hard, separation_score=0.29, n_blocks_hard_detection_lt_0_70=0
    )
    assert ok and reasons == []
