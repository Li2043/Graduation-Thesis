"""Diagnostic_6_DQN_Pipeline_Verification_Protocol.md sec 5 (6A):
discounted reward ordering, using a genuine ORACLE success trajectory
(not just a learned-checkpoint one) -- the oracle controller reaches
100% completion (Diagnostic 3), so its trajectories are as close to a
noise-free ground truth as this study has. Verifies G_success^gamma is
comfortably larger than known collision/timeout trajectories from the
task-only Diagnostic 4 checkpoint, using the EXACT training gamma."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
sys.path.insert(0, str(REPO_SRC))
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_reward_decomposition import GAMMA, replay_episode_with_decomposition  # noqa: E402

from thesis.pilots.stage11_dyad_merge_pilot_config import GAMMA as ACTUAL_TRAINING_GAMMA  # noqa: E402
from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.oracle_controller import oracle_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

SCENARIO_BANK = REPO_SRC.parent / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"

# Real, previously-computed discounted returns from the task-only (welfare_lambda=0,
# include_time_cost=False) Diagnostic 4 checkpoint's own collision/timeout
# trajectories (analyze_reward_decomposition.py's addendum in
# DIAGNOSTIC_REPORT.md): collision G=-3.511, timeout G=-2.015.
KNOWN_LEARNED_COLLISION_DISCOUNTED_G = -3.511
KNOWN_LEARNED_TIMEOUT_DISCOUNTED_G = -2.015


def test_analyze_reward_decomposition_gamma_matches_actual_training_config():
    assert GAMMA == ACTUAL_TRAINING_GAMMA


@pytest.mark.parametrize("scenario_index", [0, 1, 2])
def test_oracle_success_discounted_return_exceeds_known_failure_trajectories(scenario_index):
    if not SCENARIO_BANK.exists():
        pytest.skip("requires the frozen Q scenario bank on disk")
    scenarios = load_scenario_bank(SCENARIO_BANK)
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=200, include_time_cost=False))
    scenario = scenarios[scenario_index]

    def select(obs):
        positions = {vid: env._env._vehicles[vid].route_position for vid in env.active_vehicle_ids}  # noqa: SLF001
        return oracle_actions(
            scenario=scenario, positions=positions, merge_start=200.0, merge_end=300.0,
            active_vehicle_ids={vid: True for vid in env.active_vehicle_ids},
        )

    result = replay_episode_with_decomposition(
        select=select, env=env, scenario=scenario, condition_name="mean", episode_max_steps=200,
    )
    assert result["term_reason"] == "success"
    assert result["discounted_G"] > KNOWN_LEARNED_TIMEOUT_DISCOUNTED_G
    assert result["discounted_G"] > KNOWN_LEARNED_COLLISION_DISCOUNTED_G
    assert result["undiscounted_G"] > 0.0
