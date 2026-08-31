"""Unit + smoke tests for analyze_reward_decomposition.py --
VDN_Conditional_Amendment_Protocol.md sec 7 (Diagnostic 2)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = _load_script("analyze_reward_decomposition")


def _synthetic_info(vid: str, *, exit_flag=False, collided=False, h_cost=0.0, time_active=True, ttc=0.0) -> dict:
    return {
        "exit_reward_magnitude_used": 0.6,
        "exit_event": {vid: exit_flag},
        "collision_penalty_used_per_vehicle": {vid: 1.0},
        "collision_penalty_applied": {vid: collided},
        "hard_braking_eta_used": 0.015,
        "hard_braking_cost_used": {vid: h_cost},
        "time_cost_per_step_used": 0.0005,
        "time_cost_applied": {vid: time_active},
        "ttc_penalty_weight_used": 0.0,
        "ttc_penalty": {vid: ttc},
    }


def test_decompose_step_reward_components_sum_to_total_on_plain_step():
    info = _synthetic_info("V0")
    base_reward = {"V0": 0.0132}  # arbitrary progress-only step
    parts = analyze.decompose_step_reward("V0", base_reward, info)
    assert parts["exit"] == 0.0
    assert parts["collision"] == 0.0
    assert parts["time_cost"] == pytest.approx(-0.0005)
    assert sum(v for k, v in parts.items() if k != "total") == pytest.approx(parts["total"])
    assert parts["total"] == pytest.approx(base_reward["V0"])


def test_decompose_step_reward_components_sum_to_total_on_exit_step():
    info = _synthetic_info("V0", exit_flag=True, time_active=False)
    base_reward = {"V0": 0.62}
    parts = analyze.decompose_step_reward("V0", base_reward, info)
    assert parts["exit"] == pytest.approx(0.6)
    assert parts["time_cost"] == 0.0
    assert sum(v for k, v in parts.items() if k != "total") == pytest.approx(parts["total"])


def test_decompose_step_reward_components_sum_to_total_on_collision_step():
    info = _synthetic_info("V0", collided=True, h_cost=0.3)
    base_reward = {"V0": -1.3045}
    parts = analyze.decompose_step_reward("V0", base_reward, info)
    assert parts["collision"] == pytest.approx(-1.0)
    assert parts["hard_brake"] == pytest.approx(-0.015 * 0.3)
    assert sum(v for k, v in parts.items() if k != "total") == pytest.approx(parts["total"])


def test_find_representative_scenarios_picks_first_of_each_outcome():
    rows = [
        {"scenario_id": "Q_00", "term_reason": "truncation"},
        {"scenario_id": "Q_01", "term_reason": "success"},
        {"scenario_id": "Q_02", "term_reason": "collision"},
        {"scenario_id": "Q_03", "term_reason": "success"},  # should NOT override Q_01
    ]
    picked = analyze.find_representative_scenarios(rows)
    assert picked == {"success": "Q_01", "timeout": "Q_00", "collision": "Q_02"}


def test_find_representative_scenarios_missing_outcome_is_none():
    rows = [{"scenario_id": "Q_00", "term_reason": "success"}]
    picked = analyze.find_representative_scenarios(rows)
    assert picked == {"success": "Q_00", "timeout": None, "collision": None}


def test_replay_episode_with_decomposition_end_to_end_smoke():
    checkpoint = (
        REPO_ROOT
        / "experiments/pilots/study_b_fairness_mappo/checkpoints/qualification_dqn_fallback_8seed/seed_900102/ckpt_step_800000.pt"
    )
    scenario_bank = REPO_ROOT / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    if not checkpoint.exists() or not scenario_bank.exists():
        pytest.skip("requires a real finished checkpoint + scenario bank on disk")

    sys.path.insert(0, str(SCRIPTS_DIR))
    from evaluate_policy import load_policy, run_eval  # noqa: E402

    from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
    from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

    scenarios = load_scenario_bank(scenario_bank)
    scenario_by_id = {s.scenario_id: s for s in scenarios}
    rows = run_eval(algorithm="dqn", checkpoint=checkpoint, scenario_bank=scenario_bank, episode_max_steps=200)
    picked = analyze.find_representative_scenarios(rows)
    assert picked["success"] is not None or picked["collision"] is not None  # this checkpoint has SOME outcomes

    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=200))
    select = load_policy(algorithm="dqn", checkpoint=checkpoint, env=env)

    any_key = next(k for k, v in picked.items() if v is not None)
    result = analyze.replay_episode_with_decomposition(
        select=select, env=env, scenario=scenario_by_id[picked[any_key]], condition_name="mean", episode_max_steps=200,
    )
    assert result["episode_length"] > 0
    assert isinstance(result["undiscounted_G"], float)
    assert isinstance(result["discounted_G"], float)
    assert set(result["component_totals"]) == {"progress", "exit", "collision", "hard_brake", "time_cost", "ttc"}
