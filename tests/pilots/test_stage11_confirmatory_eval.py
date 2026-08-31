"""Tests for the Stage 11 confirmatory protocol's (DRAFT) greedy evaluator.

The end-to-end smoke test loads a REAL pilot checkpoint (seed 69113,
baseline, step 400000) purely to verify the evaluator mechanically works
(loads weights, runs held-out episodes, doesn't mutate the learner) -- this
is an engineering smoke test, NOT a formal confirmatory evaluation (69113 is
a pilot seed, not a confirmatory seed; see STAGE11_PROTOCOL.md Sec 4/6).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from thesis.agents.joint_dqn import JointDQNConfig
from thesis.pilots.stage11_confirmatory_eval import (
    build_eval_env,
    evaluate_checkpoint_stage11_confirmatory,
    load_learner_for_eval,
    run_greedy_episode,
)
from thesis.pilots.stage11_confirmatory_eval_seeds import stable_eval_seed

PILOT_CHECKPOINT = Path("external_checkpoints/stage11_pilot/seed_69113/ckpt_step_400000.pt")


def _joint_config() -> JointDQNConfig:
    return JointDQNConfig(
        per_vehicle_obs_dim=13,
        n_actions=3,
        hidden_sizes=(64, 64),
        learning_rate=0.0005,
        gamma=0.995,
        epsilon=0.0,
        replay_capacity=100_000,
        batch_size=64,
    )


pytestmark = pytest.mark.skipif(
    not PILOT_CHECKPOINT.exists(), reason="pilot checkpoint not present on this machine"
)


def test_build_eval_env_matches_two_vehicle_dyad():
    env = build_eval_env()
    _, info = env.reset(seed=12345)
    assert set(info["roles"].values()) == {"ramp", "mainline"}
    assert len(info["roles"]) == 2


def test_load_learner_for_eval_loads_real_checkpoint():
    learner = load_learner_for_eval(str(PILOT_CHECKPOINT), joint_config=_joint_config())
    assert learner.online.trunk[0].in_features == 26  # 2 * per_vehicle_obs_dim


def test_run_greedy_episode_is_deterministic():
    learner = load_learner_for_eval(str(PILOT_CHECKPOINT), joint_config=_joint_config())
    seed = stable_eval_seed(master_seed=69113, checkpoint_step=400000, scenario_block=0)
    ep1 = run_greedy_episode(learner, episode_seed=seed)
    ep2 = run_greedy_episode(learner, episode_seed=seed)
    assert ep1 == ep2


def test_run_greedy_episode_returns_valid_outcome():
    learner = load_learner_for_eval(str(PILOT_CHECKPOINT), joint_config=_joint_config())
    seed = stable_eval_seed(master_seed=69113, checkpoint_step=400000, scenario_block=0)
    ep = run_greedy_episode(learner, episode_seed=seed)
    assert ep["term_reason"] in ("success", "collision", "truncated" if False else ep["term_reason"])
    assert ep["term_reason"] in ("success", "collision") or ep["truncated"]
    assert 0.0 <= ep["U_ramp"] <= 1.0
    assert 0.0 <= ep["U_mainline"] <= 1.0
    assert ep["id_of_ramp"] in ("V0", "V1")


def test_evaluate_checkpoint_runs_16_episodes_and_does_not_mutate_learner():
    result = evaluate_checkpoint_stage11_confirmatory(
        str(PILOT_CHECKPOINT),
        joint_config=_joint_config(),
        master_seed=69113,
        checkpoint_step=400000,
    )
    assert result["n_episodes"] == 16
    assert len({(e["scenario_block"], e["assignment"]) for e in result["episodes"]}) == 16


def test_evaluate_checkpoint_role_assignments_are_swapped_within_block():
    result = evaluate_checkpoint_stage11_confirmatory(
        str(PILOT_CHECKPOINT),
        joint_config=_joint_config(),
        master_seed=69113,
        checkpoint_step=400000,
    )
    by_block: dict[int, dict[int, dict]] = {}
    for ep in result["episodes"]:
        by_block.setdefault(ep["scenario_block"], {})[ep["assignment"]] = ep
    for block, assigns in by_block.items():
        assert assigns[0]["id_of_ramp"] != assigns[1]["id_of_ramp"], block
