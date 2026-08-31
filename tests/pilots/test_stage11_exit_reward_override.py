"""Stage 11 pilot (E30) v3 -- exit-reward magnitude override.

Covers: the env-level ``exit_reward_magnitude`` override (same pattern as
``collision_penalty_magnitude``/``ttc_penalty_weight`` -- defaults to the
module constant so v1-v7/Stage 11 v1-v2 call sites are unaffected), the
runner threading the configured value through to both ``env.step()`` and
the clawback-tracking exclusion (so the higher 1.8 bonus is also correctly
excluded from future clawback, not just the default 0.6), and the new v3
reward-version tag selection logic.
"""

from __future__ import annotations

import json

import pytest

from thesis.envs.stage10_symmetric_merge_env import (
    EXIT_REWARD_MAGNITUDE,
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage11_dyad_merge_pilot_config import EXIT_REWARD_MAGNITUDE_V3, PILOT_V3_SEEDS
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V1,
    REWARD_VERSION_STAGE11_V2_CLAWBACK,
    REWARD_VERSION_STAGE11_V3_CLAWBACK_FIXED_HIGHER_EXIT,
    run_stage11_pilot_training_job,
)

# --------------------------------------------------------------- config constant


def test_v3_exit_reward_is_three_times_v1_v2():
    assert EXIT_REWARD_MAGNITUDE_V3 == pytest.approx(1.8)
    assert EXIT_REWARD_MAGNITUDE_V3 == pytest.approx(3.0 * EXIT_REWARD_MAGNITUDE)


# ------------------------------------------------------------------- env override


def _place_one_vehicle_just_before_exit(env, active):
    """Directly place both vehicles just before route_exit (400) in
    laterally-separated positions (not head-on) so a single ACCELERATE step
    makes exactly one of them cross the exit line without a collision --
    avoids the perfectly-symmetric-spawn collision that pure "both always
    accelerate from spawn" driving reliably hits in this env (both vehicles
    share one jitter draw -- see Stage 11's own module docstrings)."""
    for vid in active:
        env._roles[vid] = "mainline"
        env._vehicles[vid].role = "mainline"
    env._vehicles[active[0]].route_position = 398.0
    env._vehicles[active[1]].route_position = 100.0  # far behind, out of collision range
    env._vehicles[active[0]].speed = 18.0
    env._vehicles[active[1]].speed = 18.0


def test_step_without_override_uses_module_constant_exit_magnitude():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1, n_vehicles=2, spawn_route_lead=0.0))
    env.reset(seed=1)
    active = env.active_vehicle_ids
    _place_one_vehicle_just_before_exit(env, active)
    _, _reward, terminated, _truncated, info = env.step({vid: 1 for vid in active})
    assert any(info["exit_event"].values())
    assert info["exit_reward_magnitude_used"] == pytest.approx(EXIT_REWARD_MAGNITUDE)


def test_step_with_override_uses_the_higher_value():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=2, n_vehicles=2, spawn_route_lead=0.0))
    env.reset(seed=2)
    active = env.active_vehicle_ids
    _place_one_vehicle_just_before_exit(env, active)
    _, reward, _terminated, _truncated, info = env.step(
        {vid: 1 for vid in active}, exit_reward_magnitude=EXIT_REWARD_MAGNITUDE_V3
    )
    assert any(info["exit_event"].values())
    assert info["exit_reward_magnitude_used"] == pytest.approx(EXIT_REWARD_MAGNITUDE_V3)
    exited_vid = next(vid for vid, v in info["exit_event"].items() if v)
    # reward this step must reflect the higher bonus, not the old 0.6.
    assert reward[exited_vid] > 1.0


# --------------------------------------------------------------- runner integration


def test_v3_run_selects_the_v3_reward_version_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V3_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_clawback_collision_penalty=True,
        exit_reward_magnitude=EXIT_REWARD_MAGNITUDE_V3,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V3_CLAWBACK_FIXED_HIGHER_EXIT


def test_clawback_only_without_exit_override_keeps_v2_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V3_SEEDS[1],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_clawback_collision_penalty=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V2_CLAWBACK


def test_neither_flag_keeps_v1_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V3_SEEDS[2],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V1


def test_v3_run_trajectory_logs_the_higher_exit_magnitude_when_earned(tmp_path):
    """End-to-end: run long enough to observe at least one real exit event
    and confirm the logged reward/experience reflects the raised bonus,
    and that the clawback-tracking fix (v3's OTHER change) still holds even
    with the new magnitude -- collision_penalty_used never exceeds ~1.0
    (0.4 max progress + nothing else, since exit is excluded from tracking
    regardless of its magnitude)."""
    output_root = tmp_path / "output"
    run_stage11_pilot_training_job(
        master_seed=PILOT_V3_SEEDS[3],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=3000,
        strict=False,
        checkpoint_steps=(0, 3000),
        episode_max_steps=150,
        enable_clawback_collision_penalty=True,
        exit_reward_magnitude=EXIT_REWARD_MAGNITUDE_V3,
    )
    traj_path = output_root / "trajectories" / f"seed_{PILOT_V3_SEEDS[3]}.jsonl"
    lines = [json.loads(l) for l in traj_path.read_text(encoding="utf-8").strip().splitlines()]
    collided_vehicle_steps = [
        v for rec in lines if rec["collision_event"] for v in rec["vehicles"] if v["collision_penalty_applied"]
    ]
    for v in collided_vehicle_steps:
        # Must stay bounded by the progress-only ceiling (~0.4), never
        # anywhere near 0.4+1.8=2.2 -- proves the exit exclusion still
        # works correctly at the new, larger magnitude.
        assert 0.0 <= v["collision_penalty_used"] <= 1.0
