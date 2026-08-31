"""Protocol lock tests for Stage 7B-A1."""

from __future__ import annotations

from pathlib import Path

import yaml

from thesis.pilots.stage7b_a1_config import (
    CHECKPOINT_STEPS,
    CONDITIONS,
    EXPECTED_EVAL_EPISODES,
    MAX_STEPS,
    PILOT_SEEDS,
    TRAINING_RUN_COUNT,
)

PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "pilots"
    / "stage7b_a1_double_dqn"
    / "configs"
    / "stage7b_a1_protocol.yaml"
)


def test_protocol_fields():
    data = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert data["stage"] == "stage7b_a1"
    assert tuple(data["conditions"]) == CONDITIONS
    assert tuple(data["master_seeds"]) == PILOT_SEEDS
    assert int(data["maximum_training_steps"]) == MAX_STEPS
    assert tuple(data["checkpoint_steps"]) == CHECKPOINT_STEPS
    assert data["reward_shaping_enabled"] is False
    assert data["early_stopping"] is False
    assert data["best_checkpoint_selection"] is False
    assert data["statistical_unit"] == "paired_training_seed"
    assert int(data["training_run_count"]) == TRAINING_RUN_COUNT == 40
    assert int(data["expected_evaluation_episode_count"]) == EXPECTED_EVAL_EPISODES
    g = data["competence_gate"]
    assert g["seeds_required_at_or_above_0_75"] == 16
    assert g["mean_success_minimum"] == 0.75
    lc = data["late_collapse"]
    assert lc["final_checkpoint"] == 300000
    assert lc["prior_checkpoints"] == [200000, 250000]
