"""Stage 5A-0 — physical invariance across reward conditions."""

from __future__ import annotations

from thesis.training.final_experiment_runtime import (
    max_physical_diff,
    scripted_accelerate,
)
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.final_reward_conditions import IntegrationPBRSConfig
from thesis.training.final_v3_pipeline import run_final_v3_episode


def test_physical_traces_identical_across_conditions():
    bundle = load_final_locks()
    acts = scripted_accelerate(15)
    pcfg = IntegrationPBRSConfig()
    eps = {
        cond: run_final_v3_episode(
            bundle,
            reward_condition=cond,
            scripted_actions=acts,
            pbrs_config=pcfg,
            episode_id=f"inv_{cond}",
        )
        for cond in ("baseline", "mean_pbrs", "min_pbrs")
    }
    d1 = max_physical_diff(eps["baseline"]["transitions"], eps["mean_pbrs"]["transitions"])
    d2 = max_physical_diff(eps["baseline"]["transitions"], eps["min_pbrs"]["transitions"])
    assert d1 == 0.0
    assert d2 == 0.0
    # Episode length / termination equal
    assert eps["baseline"]["n_physical_transitions"] == eps["mean_pbrs"]["n_physical_transitions"]
    assert eps["baseline"]["transitions"][-1]["term_reason"] == eps["min_pbrs"]["transitions"][-1]["term_reason"]


def test_pipeline_uses_v3_not_v2():
    import ast
    from pathlib import Path

    path = Path("src/thesis/training/final_v3_pipeline.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported.add(alias.name)
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    assert "MergeEnvCandidateV3" in imported
    assert "MergeEnvV2" not in imported
    assert "MergeEnvConfig" not in imported
    assert "scripted_scenarios" not in imported

    bundle = load_final_locks()
    env = bundle.build_env(block_id="calibration_001")
    assert type(env).__name__ == "MergeEnvCandidateV3"
    obs, _ = env.reset()
    assert obs["A"].shape == (27,)
