"""Integration tests for Stage 3A scripted outcome rankings."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.audits.audit_scenarios import build_matched_blocks, build_block_scenarios
from thesis.audits.base_outcome_audit import (
    GAMMA,
    run_audit_scenario,
    run_label_swap_invariance_check,
)
from thesis.envs.merge_env_v2 import HighLevelAction


REPO = Path(__file__).resolve().parents[2]


def _find(block_id: str, scenario_id: str):
    block = next(b for b in build_matched_blocks() if b.block_id == block_id)
    return next(
        s for s in build_block_scenarios(block) if s.scenario_id == scenario_id
    )


def test_safe_exit_once_and_success_both():
    sc = _find("block_001", "safe_near_simultaneous")
    out = run_audit_scenario(sc, run_id="test")
    if out.term_reason == "success":
        assert out.exit_count_A == 1
        assert out.exit_count_B == 1
        assert out.terminated is True
        assert out.truncated is False


def test_truncation_separate():
    sc = _find("block_001", "stall_at_start")
    out = run_audit_scenario(sc, run_id="test")
    assert out.truncated is True
    assert out.terminated is False
    assert out.exit_count_A == 0
    assert out.exit_count_B == 0
    assert out.G_collision == pytest.approx(0.0, abs=1e-12)


def test_label_swap_invariance_tolerance():
    sc = _find("block_001", "stall_after_partial_progress")
    out = run_audit_scenario(sc, run_id="test")
    err = run_label_swap_invariance_check(out)
    assert err <= 1e-12


def test_fixture_only_not_primary():
    sc = _find("block_001", "fixture_collision_A")
    assert sc.fixture_only is True
    out = run_audit_scenario(sc, run_id="test")
    assert out.fixture_only is True
    assert out.primary_ranking is False


def test_collision_overrides_exit_on_fixture():
    sc = _find("block_001", "fixture_collision_A")
    out = run_audit_scenario(sc, run_id="test")
    assert out.terminated is True
    assert out.truncated is False
    # On collision transition, exit components should be zero
    last_step = max(t["step"] for t in out.transitions)
    for t in out.transitions:
        if t["step"] == last_step:
            assert t["exit_component"] == pytest.approx(0.0)
            assert t["collision_component"] == pytest.approx(-1.0)


def test_gamma_matches_dqn():
    assert GAMMA == pytest.approx(0.995)


def test_no_dqn_update_import_in_audit_module():
    import thesis.audits.base_outcome_audit as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "IndependentDQNLearner" not in src
    assert "optimiser.step" not in src
    assert ".update(" not in src


def test_prior_stage_outputs_untouched():
    s1 = REPO / "experiments/pre_impl/stage1_base_reward_unit_tests/artifacts"
    s2b2 = REPO / "experiments/pre_impl/stage2b2_dqn_replay_bootstrap/artifacts"
    assert any(s1.glob("*/manifest.json"))
    assert any(s2b2.glob("*/manifest.json"))
