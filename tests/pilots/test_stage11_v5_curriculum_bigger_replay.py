"""Stage 11 pilot (E30) v5 -- collision-penalty curriculum (1.0->2.0) + bigger replay buffer.

Covers: the new ``collision_penalty_at_step_v5`` ramp (distinct start/end/
window from v1's generic 1.0->5.0 ramp), ``build_shared_dqn_config``'s new
``replay_capacity`` parameter, ``enable_stage9_based_reward_v5`` selecting
the ramped (not flat) collision penalty + the 100,000-capacity replay
buffer + the new v5 reward-version tag, mutual exclusivity with the other
two reward mechanisms, and PILOT_V5_SEEDS being disjoint from every prior
version's seeds and every other stage's forbidden blocks.
"""

from __future__ import annotations

import json

import pytest

from thesis.pilots.stage11_dyad_merge_pilot_config import (
    COLLISION_PENALTY_MAGNITUDE_V4,
    COLLISION_PENALTY_RAMP_END_V5,
    COLLISION_PENALTY_RAMP_START_V5,
    COLLISION_PENALTY_RAMP_STEPS_V5,
    MAX_STEPS,
    PILOT_SEEDS,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    PILOT_V3_SEEDS,
    PILOT_V4_SEEDS,
    PILOT_V5_SEEDS,
    REPLAY_CAPACITY,
    REPLAY_CAPACITY_V5,
    assert_stage11_pilot_guards,
    collision_penalty_at_step_v5,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V1,
    REWARD_VERSION_STAGE11_V4_STAGE9_BASED,
    REWARD_VERSION_STAGE11_V5_COLLISION_CURRICULUM_BIGGER_REPLAY,
    build_shared_dqn_config,
    run_stage11_pilot_training_job,
)


# --------------------------------------------------------------- config constants


def test_v5_ramp_constants():
    assert COLLISION_PENALTY_RAMP_START_V5 == pytest.approx(1.0)
    assert COLLISION_PENALTY_RAMP_END_V5 == pytest.approx(2.0)
    assert COLLISION_PENALTY_RAMP_STEPS_V5 == 40_000
    assert REPLAY_CAPACITY_V5 == 100_000


def test_v5_seeds_disjoint_from_v1_v2_v3_v4():
    assert set(PILOT_V5_SEEDS).isdisjoint(PILOT_V1_SEEDS)
    assert set(PILOT_V5_SEEDS).isdisjoint(PILOT_V2_SEEDS)
    assert set(PILOT_V5_SEEDS).isdisjoint(PILOT_V3_SEEDS)
    assert set(PILOT_V5_SEEDS).isdisjoint(PILOT_V4_SEEDS)


def test_v5_seeds_pass_guard():
    for seed in PILOT_V5_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_v5():
    assert set(PILOT_V5_SEEDS) <= set(PILOT_SEEDS)


# --------------------------------------------------------------- ramp function


def test_ramp_starts_at_1_0():
    assert collision_penalty_at_step_v5(0) == pytest.approx(1.0)


def test_ramp_reaches_2_0_at_ramp_steps_and_stays_there():
    assert collision_penalty_at_step_v5(COLLISION_PENALTY_RAMP_STEPS_V5) == pytest.approx(2.0)
    assert collision_penalty_at_step_v5(COLLISION_PENALTY_RAMP_STEPS_V5 + 20_000) == pytest.approx(2.0)
    assert collision_penalty_at_step_v5(MAX_STEPS) == pytest.approx(2.0)


def test_ramp_is_linear_and_monotonic_midway():
    half = COLLISION_PENALTY_RAMP_STEPS_V5 // 2
    mid_value = collision_penalty_at_step_v5(half)
    assert mid_value == pytest.approx(1.5, abs=1e-6)
    assert collision_penalty_at_step_v5(0) < mid_value < collision_penalty_at_step_v5(COLLISION_PENALTY_RAMP_STEPS_V5)


def test_ramp_never_exceeds_v4s_flat_value_below_start():
    # sanity: at step 0 the v5 ramp starts at the SAME value v4 used flat
    # throughout (1.0), not some other arbitrary starting point.
    assert collision_penalty_at_step_v5(0) == pytest.approx(COLLISION_PENALTY_MAGNITUDE_V4)


# --------------------------------------------------------------- build_shared_dqn_config


def test_build_shared_dqn_config_defaults_to_v1_v4_capacity():
    cfg = build_shared_dqn_config()
    assert cfg.replay_capacity == REPLAY_CAPACITY


def test_build_shared_dqn_config_accepts_v5_capacity_override():
    cfg = build_shared_dqn_config(replay_capacity=REPLAY_CAPACITY_V5)
    assert cfg.replay_capacity == REPLAY_CAPACITY_V5


# --------------------------------------------------------------- runner integration


def test_v5_and_v4_flags_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V5_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_stage9_based_reward=True,
        )


def test_v5_and_clawback_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V5_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_clawback_collision_penalty=True,
        )


def test_v5_run_selects_the_v5_reward_version_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V5_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V5_COLLISION_CURRICULUM_BIGGER_REPLAY


def test_default_call_site_still_selects_v1_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V5_SEEDS[1],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V1


def test_v5_trajectory_shows_ramped_not_flat_collision_penalty(tmp_path):
    """End-to-end: run past the ramp window and confirm the trajectory log's
    collision_penalty_magnitude actually increases from 1.0 towards 2.0 over
    training, rather than staying flat like v4."""
    output_root = tmp_path / "output"
    run_stage11_pilot_training_job(
        master_seed=PILOT_V5_SEEDS[2],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=3000,
        strict=False,
        checkpoint_steps=(0, 3000),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
    )
    traj_path = output_root / "trajectories" / f"seed_{PILOT_V5_SEEDS[2]}.jsonl"
    lines = [json.loads(l) for l in traj_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) > 0
    first_mag = lines[0]["collision_penalty_magnitude"]
    last_mag = lines[-1]["collision_penalty_magnitude"]
    assert first_mag == pytest.approx(1.0, abs=0.05)
    assert last_mag > first_mag  # still ramping within this short 3000-step smoke run
    assert last_mag <= 2.0 + 1e-9


def test_v5_run_actually_uses_the_bigger_buffer_end_to_end(tmp_path):
    """End-to-end proof the runner really constructs the learner with
    REPLAY_CAPACITY_V5, not just that build_shared_dqn_config CAN accept the
    override: run long enough to generate more than 20,000 transitions
    (v1-v4's capacity) and confirm replay_size keeps growing past that point
    -- with the old 20,000 cap this would plateau, proving the 100,000
    capacity is what's actually in effect during a real v5 run."""
    output_root = tmp_path / "output"
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V5_SEEDS[3],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=25_000,
        strict=False,
        checkpoint_steps=(0, 12_500, 25_000),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
    )
    sizes = [c["learner"]["replay_size"] for c in manifest["checkpoints"]]
    assert max(sizes) > REPLAY_CAPACITY  # proves it grew past the OLD 20,000 cap
    assert max(sizes) <= REPLAY_CAPACITY_V5
