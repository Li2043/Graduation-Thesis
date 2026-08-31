"""Tests for audit_observation_scale.py -- VDN protocol Diagnostic 6J."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_script("audit_observation_scale")


def test_feature_statistics_detects_constant_feature():
    obs = np.zeros((10, len(audit.FEATURE_NAMES)))
    obs[:, 1] = np.linspace(0, 1, 10)  # only self_speed varies
    stats = audit.feature_statistics(obs)
    assert stats["self_role"]["std"] == 0.0
    assert stats["self_speed"]["std"] > 0.0


def test_collect_observations_oracle_and_random_end_to_end():
    scenario_bank = REPO_ROOT / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    if not scenario_bank.exists():
        import pytest

        pytest.skip("requires the frozen Q scenario bank on disk")
    obs_oracle = audit.collect_observations(policy="oracle", scenario_bank=scenario_bank, episode_max_steps=20)
    obs_random = audit.collect_observations(policy="random", scenario_bank=scenario_bank, episode_max_steps=20, seed=1)
    assert obs_oracle.shape[1] == len(audit.FEATURE_NAMES)
    assert obs_random.shape[1] == len(audit.FEATURE_NAMES)
    assert obs_oracle.shape[0] > 0
    assert obs_random.shape[0] > 0
    assert np.all(np.isfinite(obs_oracle))
    assert np.all(np.isfinite(obs_random))
