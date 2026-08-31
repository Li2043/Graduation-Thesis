"""Stage 11 pilot (E30) -- config guards, reward-curriculum timing, env spawn
geometry, and an end-to-end short training smoke test.

Covers: the seed guard (disjoint from every prior stage's seed block,
including Stage 10's own 68001-68044), the reward-magnitude curriculum's
new (shorter) ramp window compared to Stage 10 v7, that
Stage10SymmetricMergeEnv accepts a 2-vehicle, spawn-from-0 configuration
unmodified, and a short end-to-end run of
``run_stage11_pilot_training_job`` to confirm the welfare/crossing-order
bookkeeping and trajectory logging actually execute without error.
"""

from __future__ import annotations

import json

import pytest

from thesis.envs.stage10_symmetric_merge_env import (
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage10_role_phase_subpolicy_config import (
    PILOT_SEEDS as STAGE10_PILOT_SEEDS,
)
from thesis.pilots.stage11_dyad_merge_pilot_config import (
    CHECKPOINT_STEPS,
    EPSILON_DECAY_STEPS,
    EPSILON_END,
    EPSILON_START,
    FORBIDDEN_STAGE10_SEEDS,
    MAX_STEPS,
    PILOT_SEEDS,
    PILOT_V1_SEEDS,
    REWARD_CURRICULUM_RAMP_STEPS,
    assert_stage11_pilot_guards,
    collision_penalty_at_step,
    epsilon_at_step,
    ttc_weight_at_step,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11,
    run_stage11_pilot_training_job,
)

# --------------------------------------------------------------------- seeds


def test_stage11_seeds_disjoint_from_stage10():
    assert set(PILOT_V1_SEEDS).isdisjoint(STAGE10_PILOT_SEEDS)
    assert set(PILOT_V1_SEEDS).isdisjoint(FORBIDDEN_STAGE10_SEEDS)


def test_guard_accepts_valid_seed_and_max_steps():
    for seed in PILOT_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_guard_rejects_seed_outside_pilot_block():
    # 69017 was one past v2's last seed (69016) when this test was written --
    # but 69017 is now a legitimate pilot v3 seed (clawback scoping fix +
    # higher exit reward, 2026-08-06), so it no longer demonstrates "outside
    # every reserved block". Then updated to 69025, 69033, 69041, 69049,
    # 69057, 69065, 69073, 69081, then 69113 (one past v12's last seed,
    # PILOT_V12_MIN_PBRS_SEEDS' 69112) -- which itself became a legitimate
    # seed when v12's PER+n-step round added PILOT_V12_BASELINE_V2_SEEDS
    # (69113-69120, 2026-08-09). Now updated to 69121 (one past THAT block's
    # last seed). Same fix pattern used at Stage 10's v4->v5->v6->v7
    # transitions and this file's own v1-v12(-round-2) transitions.
    with pytest.raises(RuntimeError):
        assert_stage11_pilot_guards(master_seed=69121, max_steps=MAX_STEPS)


def test_guard_rejects_stage10_seed():
    with pytest.raises(RuntimeError):
        assert_stage11_pilot_guards(master_seed=68037, max_steps=MAX_STEPS)


def test_guard_rejects_wrong_max_steps():
    with pytest.raises(RuntimeError):
        assert_stage11_pilot_guards(master_seed=PILOT_V1_SEEDS[0], max_steps=180_000)


# ------------------------------------------------------- reward curriculum timing


def test_reward_curriculum_ramp_is_shorter_than_stage10_v7s():
    # Stage 10 v7 used a 100,000-step ramp (sized for its 2+4-vehicle
    # curriculum stages); Stage 11 has no such stages, so its ramp must be
    # sized against its OWN (100,000-step) single-stage budget instead.
    assert REWARD_CURRICULUM_RAMP_STEPS == 40_000
    assert REWARD_CURRICULUM_RAMP_STEPS < 100_000


def test_collision_penalty_locks_at_end_value_before_budget_ends():
    # Ramp must complete well before MAX_STEPS, leaving a consolidation tail.
    assert collision_penalty_at_step(REWARD_CURRICULUM_RAMP_STEPS) == pytest.approx(5.0)
    assert collision_penalty_at_step(MAX_STEPS) == pytest.approx(5.0)
    assert collision_penalty_at_step(0) == pytest.approx(1.0)


def test_ttc_weight_locks_at_end_value_before_budget_ends():
    assert ttc_weight_at_step(REWARD_CURRICULUM_RAMP_STEPS) == pytest.approx(0.1)
    assert ttc_weight_at_step(0) == pytest.approx(0.02)


def test_epsilon_decays_over_80_percent_of_budget_not_20_percent():
    assert EPSILON_DECAY_STEPS == pytest.approx(0.8 * MAX_STEPS)
    assert epsilon_at_step(0) == pytest.approx(EPSILON_START)
    assert epsilon_at_step(EPSILON_DECAY_STEPS) == pytest.approx(EPSILON_END)


def test_checkpoint_steps_cover_full_budget():
    assert CHECKPOINT_STEPS[0] == 0
    assert CHECKPOINT_STEPS[-1] == MAX_STEPS


# ----------------------------------------------------------- env spawn geometry


def test_env_accepts_two_vehicle_spawn_from_route_start():
    env = Stage10SymmetricMergeEnv(
        Stage10MergeEnvConfig(seed=1, n_vehicles=2, spawn_route_lead=0.0, include_role_zone_features=True)
    )
    obs, info = env.reset(seed=1)
    assert set(env.active_vehicle_ids) == set(info["roles"].keys())
    assert len(env.active_vehicle_ids) == 2
    # Both vehicles must spawn near route_start (0), not 40m in.
    for vid in env.active_vehicle_ids:
        assert env._vehicles[vid].route_position == pytest.approx(0.0, abs=1.0)


def test_both_roles_spawn_at_identical_position_and_speed():
    """Documents the deliberately-preserved (not staggered) tie: both roles
    share the same jitter draw and spawn_speed, exactly as in Stage 10 --
    this is intentional (see module docstrings), not a bug to fix here."""
    env = Stage10SymmetricMergeEnv(
        Stage10MergeEnvConfig(seed=2, n_vehicles=2, spawn_route_lead=0.0, include_role_zone_features=True)
    )
    env.reset(seed=2)
    positions = [env._vehicles[vid].route_position for vid in env.active_vehicle_ids]
    speeds = [env._vehicles[vid].speed for vid in env.active_vehicle_ids]
    assert positions[0] == pytest.approx(positions[1])
    assert speeds[0] == pytest.approx(speeds[1])


# --------------------------------------------------------------- end-to-end smoke


def test_short_training_run_produces_manifest_and_trajectory(tmp_path):
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V1_SEEDS[0],
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=600,
        strict=False,  # short smoke run, not the frozen 100K budget
        checkpoint_steps=(0, 300, 600),
        episode_max_steps=50,
    )
    assert manifest["final_step"] == 600
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11
    assert manifest["condition"] == "baseline"
    assert len(manifest["checkpoints"]) == 3

    traj_path = output_root / "trajectories" / f"seed_{PILOT_V1_SEEDS[0]}.jsonl"
    assert traj_path.exists()
    lines = traj_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 600
    first = json.loads(lines[0])
    assert "welfare_mean" in first
    assert "welfare_min" in first
    assert first["welfare_min"] <= first["welfare_mean"]
    for v in first["vehicles"]:
        assert "attainment" in v
        assert "experience" in v
        assert 0.0 <= v["attainment"] <= 1.0

    # Window stats must include the new welfare/convention fields.
    last_ckpt = manifest["checkpoints"][-1]
    window = last_ckpt["window"]
    assert "mean_U_mean" in window
    assert "min_U_mean" in window
    assert "first_crosser_role_counts" in window
    assert "first_crosser_identity_counts" in window
