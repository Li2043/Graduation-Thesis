"""Evaluator-oriented Stage 6B-H1 tests (no training side effects)."""

from __future__ import annotations

from thesis.analysis.episode_utility_accumulator import (
    collect_active_state_attainment,
    finalise_episode_utilities,
    initialise_episode_utility_accumulator,
)
from thesis.rewards.pbrs_v2 import STAKEHOLDER_ORDER


def test_utilities_in_unit_interval_for_synthetic_trajectory() -> None:
    acc = initialise_episode_utility_accumulator(STAKEHOLDER_ORDER)
    for speed in (5.0, 10.0, 15.0, 20.0):
        vehicles = {
            sid: {
                "speed": speed,
                "target_speed": 20.0,
                "active_on_road": True,
                "completed": False,
            }
            for sid in STAKEHOLDER_ORDER
        }
        collect_active_state_attainment(
            vehicles=vehicles,
            stakeholder_ids=STAKEHOLDER_ORDER,
            accumulator=acc,
        )
    utils = finalise_episode_utilities(accumulator=acc, collided_stakeholder_ids=[])
    assert set(utils) == set(STAKEHOLDER_ORDER)
    assert all(0.0 <= v <= 1.0 for v in utils.values())


def test_evaluation_guard_schema_present_in_reconstruct_module() -> None:
    src = open(
        __file__.replace(
            "tests\\analysis\\test_stage6b_h1_evaluator.py",
            "src\\thesis\\analysis\\reconstruct_eval.py",
        ).replace(
            "tests/analysis/test_stage6b_h1_evaluator.py",
            "src/thesis/analysis/reconstruct_eval.py",
        ),
        encoding="utf-8",
    ).read()
    assert "evaluation_guard" in src
    assert "trajectory_active_state_mean" in src or "util_acc" in src
    assert "finalise_episode_utilities" in src
    assert "compute_stakeholder_experiences" not in src or "Do NOT use final-state" in src
