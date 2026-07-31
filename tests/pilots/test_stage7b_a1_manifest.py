"""Manifest / count expectations."""

from __future__ import annotations

from thesis.pilots.stage7b_a1_config import (
    EXPECTED_EVAL_EPISODES,
    PILOT_SEEDS,
    TRAINING_RUN_COUNT,
)


def test_run_and_episode_counts():
    assert TRAINING_RUN_COUNT == 2 * len(PILOT_SEEDS) == 40
    assert EXPECTED_EVAL_EPISODES == 40 * 10 * 16 == 6400


def test_relative_paths_in_manifest_template():
    manifest = {
        "stage": "Stage 7B-A1",
        "conditions": ["vanilla_dqn", "double_dqn"],
        "master_seeds": list(PILOT_SEEDS),
        "output_hashes": {"output/statistics/example.csv": "abc"},
    }
    for v in manifest["output_hashes"]:
        assert not v.startswith("/")
        assert ":\\" not in v
