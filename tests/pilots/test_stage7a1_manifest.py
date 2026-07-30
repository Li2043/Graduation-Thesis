"""Manifest / integrity helpers for Stage 7A-1."""

from __future__ import annotations

from pathlib import Path

from thesis.pilots.stage7a1_config import MAX_STEPS, PILOT_SEEDS


def test_manifest_template_fields():
    # Structural expectations for the eventual manifest
    manifest = {
        "stage": "Stage 7A-1",
        "name": "Baseline-Only Unchanged-Budget Competence Pilot",
        "analysis_status": "exploratory",
        "condition": "baseline",
        "reward_shaping_enabled": False,
        "formal_experiment_modified": False,
        "paper_files_modified": False,
        "old_formal_seeds_reused": False,
        "master_seeds": list(PILOT_SEEDS),
        "maximum_training_steps": MAX_STEPS,
        "training_run_count": 20,
        "evaluation_episode_count": 3200,
        "statistical_unit": "training_seed",
    }
    assert manifest["master_seeds"][0] == 62001
    assert manifest["master_seeds"][-1] == 62020
    assert 61001 not in manifest["master_seeds"]
    # relative paths only in any path-like values
    for v in manifest.values():
        if isinstance(v, str) and (":\\" in v or v.startswith("/")):
            raise AssertionError(f"absolute path not allowed: {v}")


def test_pilot_output_root_isolation():
    root = Path("experiments/pilots/stage7a1_baseline_budget")
    assert "experiments/formal" not in str(root)
    assert "stage7a0" not in str(root)
