"""Stage 7A-1 protocol lock tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from thesis.pilots.stage7a1_config import (
    CHECKPOINT_STEPS,
    FORBIDDEN_FORMAL_SEEDS,
    MAX_STEPS,
    PILOT_SEEDS,
)

PILOT_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "stage7a1_baseline_budget"
PROTOCOL = PILOT_ROOT / "configs" / "stage7a1_baseline_budget_protocol.yaml"


def test_protocol_frozen_fields():
    data = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert data["condition"] == "baseline"
    assert data["reward_shaping_enabled"] is False
    assert float(data["shaping_coefficient"]) == 0.0
    assert data["early_stopping"] is False
    assert data["best_checkpoint_selection"] is False
    assert data["statistical_unit"] == "training_seed"
    assert int(data["maximum_training_steps"]) == MAX_STEPS == 300_000
    assert tuple(data["master_seeds"]) == PILOT_SEEDS
    assert tuple(data["checkpoint_steps"]) == CHECKPOINT_STEPS
    assert data["allow_mean_pbrs"] is False
    assert data["allow_min_pbrs"] is False
    g = data["competence_gate"]
    assert g["min_seeds_success_ge_0_75"] == 16
    assert g["min_mean_success"] == 0.75
    assert g["max_mean_collision"] == 0.05
    assert g["max_mean_truncation"] == 0.15
    assert g["min_swap_eligible_pair_proportion"] == 0.75


def test_epsilon_not_stretched_for_300k():
    data = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert int(data["epsilon_decay_environment_steps"]) == 50_000
    assert float(data["epsilon_after_decay"]) == 0.10


def test_forbidden_seeds_listed():
    data = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert tuple(data["forbidden_formal_seeds"]) == FORBIDDEN_FORMAL_SEEDS
