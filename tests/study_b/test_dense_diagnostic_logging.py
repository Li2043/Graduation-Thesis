"""Tests for train_single_scenario_overfit.py's --dense-log-every feature
and stage10_shared_dqn.SharedDQNLearner.update()'s new per-update
diagnostics (td_error_mean_abs/td_error_max_abs/grad_norm) -- added to
investigate the 400K single-scenario overfit run's non-monotonic
trajectory (completion collapse coinciding with a checkpoint-level
mean_Q spike)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


overfit = _load_script("train_single_scenario_overfit")


def test_update_returns_td_error_and_grad_norm_diagnostics():
    from thesis.agents.independent_dqn_v2 import DQNConfig
    from thesis.agents.replay_buffer_v2 import ReplayTransition
    from thesis.agents.stage10_shared_dqn import SharedDQNLearner

    config = DQNConfig(obs_dim=4, n_actions=3, hidden_sizes=(8, 8), batch_size=4, replay_capacity=100)
    learner = SharedDQNLearner(config, seed=0)
    mask = np.array([True, True, True])
    for i in range(10):
        learner.store_transition(
            ReplayTransition(
                observation=np.array([float(i), 0.0, 0.0, 0.0]), action=i % 3, shaped_reward=0.1,
                next_observation=np.array([float(i) + 1, 0.0, 0.0, 0.0]), terminated=False, truncated=False,
                action_mask=mask, next_action_mask=mask, controller_terminal=False,
            )
        )
    result = learner.update()
    for key in ("loss", "td_error_mean_abs", "td_error_max_abs", "grad_norm"):
        assert key in result
        assert result[key] >= 0.0
        assert np.isfinite(result[key])


def test_dense_log_file_created_with_expected_fields(tmp_path):
    scenario_bank = Path(__file__).resolve().parents[2] / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    if not scenario_bank.exists():
        import pytest

        pytest.skip("requires the frozen Q scenario bank on disk")

    argv = [
        "--scenario-bank", str(scenario_bank), "--master-seed", "3",
        "--output-root", str(tmp_path / "out"), "--checkpoint-root", str(tmp_path / "ckpt"),
        "--max-steps", "300", "--episode-max-steps", "60", "--checkpoint-steps", "300",
        "--replay-warmup", "64", "--device", "cpu", "--dense-log-every", "20",
    ]
    rc = overfit.main(argv)
    assert rc == 0

    dense_log_path = tmp_path / "out" / "seed_3_dense_log.jsonl"
    assert dense_log_path.exists()
    lines = dense_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    row = json.loads(lines[0])
    for key in ("step", "update_count", "loss", "td_error_mean_abs", "td_error_max_abs", "grad_norm", "mean_Q", "mean_Q_spread"):
        assert key in row


def test_dense_log_disabled_by_default(tmp_path):
    scenario_bank = Path(__file__).resolve().parents[2] / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    if not scenario_bank.exists():
        import pytest

        pytest.skip("requires the frozen Q scenario bank on disk")

    argv = [
        "--scenario-bank", str(scenario_bank), "--master-seed", "4",
        "--output-root", str(tmp_path / "out"), "--checkpoint-root", str(tmp_path / "ckpt"),
        "--max-steps", "200", "--episode-max-steps", "60", "--checkpoint-steps", "200",
        "--replay-warmup", "64", "--device", "cpu",
    ]
    rc = overfit.main(argv)
    assert rc == 0
    assert not (tmp_path / "out" / "seed_4_dense_log.jsonl").exists()
