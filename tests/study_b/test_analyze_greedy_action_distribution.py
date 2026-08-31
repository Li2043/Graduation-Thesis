"""Unit + smoke tests for analyze_greedy_action_distribution.py --
VDN_Conditional_Amendment_Protocol.md sec 6 (Diagnostic 1)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = _load_script("analyze_greedy_action_distribution")


def test_class_label_combines_role_and_speed_class():
    assert analyze.class_label("ramp", "fast") == "ramp-fast"
    assert analyze.class_label("mainline", "slow") == "mainline-slow"


def test_cross_class_similarity_identical_distributions_flagged():
    dist = {"MAINTAIN": 0.2, "ACCELERATE": 0.3, "DECELERATE": 0.5}
    distributions = {label: dict(dist) for label in ("ramp-fast", "ramp-slow", "mainline-fast", "mainline-slow")}
    result = analyze.cross_class_similarity(distributions)
    assert result["max_distance"] == 0.0
    assert result["near_identical_across_classes"] is True


def test_cross_class_similarity_very_different_distributions_not_flagged():
    distributions = {
        "ramp-fast": {"MAINTAIN": 0.0, "ACCELERATE": 1.0, "DECELERATE": 0.0},
        "ramp-slow": {"MAINTAIN": 0.0, "ACCELERATE": 0.0, "DECELERATE": 1.0},
        "mainline-fast": {"MAINTAIN": 1.0, "ACCELERATE": 0.0, "DECELERATE": 0.0},
        "mainline-slow": {"MAINTAIN": 0.5, "ACCELERATE": 0.5, "DECELERATE": 0.0},
    }
    result = analyze.cross_class_similarity(distributions)
    assert result["max_distance"] == 1.0
    assert result["near_identical_across_classes"] is False


def test_cross_class_similarity_threshold_is_configurable():
    distributions = {
        "ramp-fast": {"MAINTAIN": 0.5, "ACCELERATE": 0.3, "DECELERATE": 0.2},
        "ramp-slow": {"MAINTAIN": 0.55, "ACCELERATE": 0.25, "DECELERATE": 0.2},
        "mainline-fast": {"MAINTAIN": 0.5, "ACCELERATE": 0.3, "DECELERATE": 0.2},
        "mainline-slow": {"MAINTAIN": 0.5, "ACCELERATE": 0.3, "DECELERATE": 0.2},
    }
    loose = analyze.cross_class_similarity(distributions, threshold=0.10)
    strict = analyze.cross_class_similarity(distributions, threshold=0.01)
    assert loose["near_identical_across_classes"] is True
    assert strict["near_identical_across_classes"] is False


def test_tally_greedy_actions_end_to_end_smoke(tmp_path):
    checkpoint = (
        REPO_ROOT
        / "experiments/pilots/study_b_fairness_mappo/checkpoints/qualification_dqn_fallback_8seed/seed_900102/ckpt_step_800000.pt"
    )
    scenario_bank = REPO_ROOT / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    if not checkpoint.exists() or not scenario_bank.exists():
        import pytest

        pytest.skip("requires a real finished checkpoint + scenario bank on disk")

    report = analyze.tally_greedy_actions(
        algorithm="dqn", checkpoint=checkpoint, scenario_bank=scenario_bank, episode_max_steps=20,
    )
    assert set(report["distributions"]) == {"ramp-fast", "ramp-slow", "mainline-fast", "mainline-slow"}
    for dist in report["distributions"].values():
        assert set(dist) == {"MAINTAIN", "ACCELERATE", "DECELERATE"}
        assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert "near_identical_across_classes" in report["similarity"]
