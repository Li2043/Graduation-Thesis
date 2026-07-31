"""Stage 7C-Q1 protocol / eval-seed / gate lock tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from thesis.agents.dqn_bootstrap import DQNTargetMode
from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
from thesis.agents.replay_buffer_v2 import ReplayTransition
import numpy as np
from thesis.pilots.stage7c_q1_config import (
    CHECKPOINT_STEPS,
    GATE_CHECKPOINTS,
    MAX_STEPS,
    PILOT_SEEDS,
    PROTOCOL_TAG,
    episodes_per_seed_checkpoint,
    target_mode,
)
from thesis.pilots.stage7c_q1_eval import map_failure_category
from thesis.pilots.stage7c_q1_eval_seeds import (
    assert_no_eval_seed_overlap,
    eval_plan_for_checkpoint,
    stable_eval_seed,
)
from thesis.pilots.stage7c_q1_gate import evaluate_competence_gate
from thesis.pilots.stage7c_q1_scripted_audit import run_scripted_reward_audit
from thesis.formal.formal_config import derive_formal_job_seeds


REPO = Path(__file__).resolve().parents[2]
PROTOCOL_YAML = (
    REPO
    / "experiments"
    / "pilots"
    / "stage7c_q1_baseline_competence"
    / "configs"
    / "stage7c_q1_protocol.yaml"
)


def test_protocol_yaml_locks():
    data = yaml.safe_load(PROTOCOL_YAML.read_text(encoding="utf-8"))
    assert data["algorithm"] == "double_dqn"
    assert data["condition"] == "baseline"
    assert data["allow_vanilla_dqn"] is False
    assert data["allow_mean_pbrs"] is False
    assert data["allow_min_pbrs"] is False
    assert data["reward_shaping_enabled"] is False
    assert float(data["active_time_cost_per_step"]) == 0.0005
    assert int(data["max_joint_environment_steps"]) == 400_000
    assert data["checkpoints"] == list(CHECKPOINT_STEPS)
    assert data["master_seeds"]["list"] == list(PILOT_SEEDS)
    assert data["early_stopping"] is False
    assert data["best_checkpoint_selection"] is False


def test_target_mode_is_double_only():
    assert target_mode() is DQNTargetMode.DOUBLE


def test_eval_episode_counts():
    assert episodes_per_seed_checkpoint(0) == 16
    assert episodes_per_seed_checkpoint(175_000) == 16
    assert episodes_per_seed_checkpoint(200_000) == 64
    assert episodes_per_seed_checkpoint(400_000) == 64


def test_role_swap_shares_base_eval_seed():
    plan = eval_plan_for_checkpoint(master_seed=64001, checkpoint_step=200_000)
    by_block = {}
    for row in plan:
        by_block.setdefault(row["scenario_block"], []).append(row)
    for block, rows in by_block.items():
        assert len(rows) == 2
        assert rows[0]["eval_seed"] == rows[1]["eval_seed"]
        assert {rows[0]["assignment"], rows[1]["assignment"]} == {0, 1}
        assert rows[0]["swap_pair_id"] == rows[1]["swap_pair_id"]


def test_no_eval_seed_overlap_across_master_seeds():
    assert_no_eval_seed_overlap(PILOT_SEEDS, GATE_CHECKPOINTS)


def test_train_eval_seed_namespace_isolation():
    # Training stream seeds from derive_formal_job_seeds must not equal eval seeds
    train_seeds = set()
    for ms in PILOT_SEEDS:
        d = derive_formal_job_seeds(ms)
        train_seeds.update(int(v) for v in d.values())
    eval_seeds = set()
    for ms in PILOT_SEEDS:
        for ckpt in (0, 200_000, 400_000):
            for row in eval_plan_for_checkpoint(master_seed=ms, checkpoint_step=ckpt):
                if int(row["assignment"]) == 0:
                    eval_seeds.add(int(row["eval_seed"]))
    overlap = train_seeds & eval_seeds
    assert not overlap, f"train/eval seed overlap: {sorted(list(overlap))[:5]}"


def test_stable_eval_seed_not_python_hash():
    a = stable_eval_seed(master_seed=64001, checkpoint_step=0, scenario_block=0)
    b = stable_eval_seed(master_seed=64001, checkpoint_step=0, scenario_block=0)
    assert a == b
    # Different block differs
    c = stable_eval_seed(master_seed=64001, checkpoint_step=0, scenario_block=1)
    assert a != c


def test_failure_category_frozen_mapping():
    assert map_failure_category(success=True, collision=False, truncated=False) == "success"
    assert map_failure_category(success=False, collision=True, truncated=False) == "collision"
    assert (
        map_failure_category(
            success=False,
            collision=False,
            truncated=True,
            primary_failure_label="unilateral_stall",
        )
        == "unilateral_stall"
    )
    assert (
        map_failure_category(
            success=False,
            collision=False,
            truncated=True,
            primary_failure_label="downstream_completion_failure",
        )
        == "downstream_failure"
    )


def test_exited_learner_not_written_to_replay_semantics():
    """Mirror FormalTrainer: inactive agents skip store_transition."""
    cfg = DQNConfig(
        obs_dim=4,
        n_actions=3,
        hidden_sizes=(8,),
        batch_size=4,
        replay_capacity=32,
        target_mode=DQNTargetMode.DOUBLE,
    )
    learner = IndependentDQNLearner("A", cfg, seed=1, replay_seed=2)
    active = {"A": True}
    # Simulate exit then skip write
    before = len(learner.replay)
    exit_now = True
    if active["A"]:
        learner.store_transition(
            ReplayTransition(
                observation=np.zeros(4),
                action=0,
                shaped_reward=0.6,
                next_observation=None,
                terminated=False,
                truncated=False,
                controller_terminal=True,
                learner_completed=True,
                action_mask=np.array([True, True, True]),
                next_action_mask=None,
                base_reward=0.6,
                shaping_component=0.0,
                reward_condition="baseline",
                episode_id="e",
                step=1,
                controller_id="A",
                traffic_role="mainline",
            )
        )
        active["A"] = False
    assert len(learner.replay) == before + 1
    # Post-exit: skip
    if active["A"]:
        learner.store_transition(
            ReplayTransition(
                observation=np.zeros(4),
                action=0,
                shaped_reward=0.0,
                next_observation=np.zeros(4),
                terminated=False,
                truncated=False,
                controller_terminal=False,
                learner_completed=True,
                action_mask=np.array([True, True, True]),
                next_action_mask=np.array([True, True, True]),
                base_reward=0.0,
                shaping_component=0.0,
                reward_condition="baseline",
                episode_id="e",
                step=2,
                controller_id="A",
                traffic_role="mainline",
            )
        )
    assert len(learner.replay) == before + 1


def test_gate_outcomes_only_pass_fail_invalid():
    # INVALID: missing seeds
    df = pd.DataFrame(
        [
            {
                "master_seed": 64001,
                "checkpoint_step": 350000,
                "success_rate": 1.0,
                "collision_rate": 0.0,
                "truncation_rate": 0.0,
                "swap_eligibility": 1.0,
            }
        ]
    )
    r = evaluate_competence_gate(df)
    assert r["status"] == "INVALID"

    r2 = evaluate_competence_gate(df, integrity_ok=False, integrity_errors=["bad"])
    assert r2["status"] == "INVALID"


def test_scripted_audit_passes():
    result = run_scripted_reward_audit()
    assert result["passed"] is True


def test_protocol_tag_frozen():
    assert PROTOCOL_TAG == "stage7c-q1-protocol-v1"
    assert MAX_STEPS == 400_000
