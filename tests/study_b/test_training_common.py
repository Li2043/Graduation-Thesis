from __future__ import annotations

import pytest

from thesis.study_b.scenario_generator import generate_scenario
from thesis.study_b.training_common import (
    StudyBEpisodeWindowStats,
    load_scenario_bank,
    save_scenario_bank,
)

ROLE_MEMBERS = {"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]}


def test_window_stats_basic_rates():
    stats = StudyBEpisodeWindowStats()
    stats.record_episode(
        term_reason="success",
        utilities={"V0": 1.0, "V1": 1.0, "V2": 1.0, "V3": 1.0},
        burdens={"V0": 0.0, "V1": 0.0, "V2": 0.0, "V3": 0.0},
    )
    stats.record_episode(
        term_reason="collision",
        utilities={"V0": 0.0, "V1": 0.0, "V2": 0.0, "V3": 0.0},
        burdens={"V0": 1.0, "V1": 1.0, "V2": 1.0, "V3": 1.0},
    )
    d = stats.as_dict()
    assert d["episodes"] == 2
    assert d["completion_rate"] == pytest.approx(0.5)
    assert d["collision_rate"] == pytest.approx(0.5)
    assert d["mean_U_mean"] == pytest.approx(0.5)
    # second episode was all-zero utility -> gini NA for that one, counted separately
    assert d["all_zero_utility_rate"] == pytest.approx(0.5)
    assert d["gini_mean"] == pytest.approx(0.0)  # only the first (all-equal) episode contributes


def test_window_stats_gini_mean_is_none_when_all_episodes_all_zero():
    stats = StudyBEpisodeWindowStats()
    stats.record_episode(
        term_reason="collision",
        utilities={"V0": 0.0, "V1": 0.0, "V2": 0.0, "V3": 0.0},
        burdens={"V0": 1.0, "V1": 1.0, "V2": 1.0, "V3": 1.0},
    )
    d = stats.as_dict()
    assert d["gini_mean"] is None


def test_window_stats_reset_clears_everything():
    stats = StudyBEpisodeWindowStats()
    stats.record_episode(
        term_reason="success",
        utilities={"V0": 1.0, "V1": 1.0, "V2": 1.0, "V3": 1.0},
        burdens={"V0": 0.0, "V1": 0.0, "V2": 0.0, "V3": 0.0},
    )
    stats.reset()
    d = stats.as_dict()
    assert d["episodes"] == 0
    assert d["completion_rate"] == 0.0


def test_scenario_bank_save_load_roundtrip(tmp_path):
    scenarios = [
        generate_scenario(scenario_id=f"s{i}", episode_seed=i, role_members=ROLE_MEMBERS)
        for i in range(5)
    ]
    path = tmp_path / "bank.json"
    save_scenario_bank(scenarios, path)
    loaded = load_scenario_bank(path)
    assert len(loaded) == 5
    for original, restored in zip(scenarios, loaded):
        assert original.scenario_id == restored.scenario_id
        assert original.vehicles == restored.vehicles
