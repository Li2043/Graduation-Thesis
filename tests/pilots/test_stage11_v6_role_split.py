"""Stage 11 pilot (E30) v6 -- role-split architecture (2 independent DQNs).

Covers: ``enable_role_split_v6`` requiring v5's reward flag, mutual
exclusivity/composition rules, the new v6 reward-version tag, the new
``ARCHITECTURE_ROLE_SPLIT`` label appearing in trajectory/manifest output,
each role's learner actually being a SEPARATE object with independently
growing replay buffers, and PILOT_V6_SEEDS being disjoint from every prior
version's seeds and every other stage's forbidden blocks.
"""

from __future__ import annotations

import json

import pytest

from thesis.pilots.stage11_dyad_merge_pilot_config import (
    ARCHITECTURE_ROLE_SPLIT,
    ARCHITECTURE_SHARED_PARAMETER,
    MAX_STEPS,
    PILOT_SEEDS,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    PILOT_V3_SEEDS,
    PILOT_V4_SEEDS,
    PILOT_V5_SEEDS,
    PILOT_V6_SEEDS,
    assert_stage11_pilot_guards,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V5_COLLISION_CURRICULUM_BIGGER_REPLAY,
    REWARD_VERSION_STAGE11_V6_ROLE_SPLIT,
    run_stage11_pilot_training_job,
)


# --------------------------------------------------------------- config constants


def test_v6_seeds_disjoint_from_v1_through_v5():
    for block in (PILOT_V1_SEEDS, PILOT_V2_SEEDS, PILOT_V3_SEEDS, PILOT_V4_SEEDS, PILOT_V5_SEEDS):
        assert set(PILOT_V6_SEEDS).isdisjoint(block)


def test_v6_seeds_pass_guard():
    for seed in PILOT_V6_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_v6():
    assert set(PILOT_V6_SEEDS) <= set(PILOT_SEEDS)


# --------------------------------------------------------------- flag composition rules


def test_role_split_without_v5_reward_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V6_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_role_split_v6=True,
            # enable_stage9_based_reward_v5 deliberately omitted (False)
        )


def test_role_split_with_v4_reward_instead_of_v5_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V6_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_role_split_v6=True,
            enable_stage9_based_reward=True,  # v4, not v5 -- still not allowed
        )


def test_v6_run_selects_the_v6_reward_version_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V6_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V6_ROLE_SPLIT
    assert manifest["architecture"] == ARCHITECTURE_ROLE_SPLIT


def test_v5_without_role_split_keeps_shared_architecture_label(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V6_SEEDS[1],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V5_COLLISION_CURRICULUM_BIGGER_REPLAY
    assert manifest["architecture"] == ARCHITECTURE_SHARED_PARAMETER


# --------------------------------------------------------------- learner separation


def test_v6_checkpoint_learner_payload_is_keyed_by_role(tmp_path):
    import torch

    checkpoint_root = tmp_path / "checkpoints"
    run_stage11_pilot_training_job(
        master_seed=PILOT_V6_SEEDS[2],
        output_root=tmp_path / "output",
        checkpoint_root=checkpoint_root,
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
    )
    payload = torch.load(
        checkpoint_root / f"seed_{PILOT_V6_SEEDS[2]}" / "ckpt_step_600.pt",
        weights_only=False,
    )
    assert set(payload["learner"].keys()) == {"ramp", "mainline"}
    for role_payload in payload["learner"].values():
        assert "online" in role_payload and "target" in role_payload


def test_v6_manifest_learner_by_role_reports_two_independently_sized_buffers(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V6_SEEDS[3],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=3000,
        strict=False,
        checkpoint_steps=(0, 3000),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
    )
    last = manifest["checkpoints"][-1]
    assert "learner_by_role" in last
    assert set(last["learner_by_role"].keys()) == {"ramp", "mainline"}
    # aggregate "learner" replay_size must equal the sum of both roles' (backward-compat shape check).
    per_role_sum = sum(v["replay_size"] for v in last["learner_by_role"].values())
    assert last["learner"]["replay_size"] == per_role_sum


def test_v6_trajectory_logs_role_split_architecture_label(tmp_path):
    output_root = tmp_path / "output"
    run_stage11_pilot_training_job(
        master_seed=PILOT_V6_SEEDS[4],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=2000,
        strict=False,
        checkpoint_steps=(0, 2000),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
    )
    traj_path = output_root / "trajectories" / f"seed_{PILOT_V6_SEEDS[4]}.jsonl"
    lines = [json.loads(l) for l in traj_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) > 0
    assert all(rec["architecture"] == ARCHITECTURE_ROLE_SPLIT for rec in lines)
