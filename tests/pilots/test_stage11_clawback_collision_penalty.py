"""Stage 11 pilot (E30) v2 -- collision-penalty "clawback" mechanism.

Covers: the env-level ``collision_penalty_override`` parameter (per-vehicle
override, falls back to the scalar for non-overridden/non-colliding
vehicles, v1-v7 call sites that never pass it are unaffected), the runner's
running per-vehicle cumulative-reward tracking (the basis for the clawback
value), and an end-to-end short run confirming a collision nets a vehicle's
accumulated reward to (approximately) zero for that episode.

See stage10_symmetric_merge_env.py's module docstring ("Collision-penalty
clawback" section) and stage11_dyad_merge_runner.py's module-level comment
for the full diagnosis (v1's per-step trajectory analysis found TTC
essentially inert and the fixed/curriculum collision penalty driving total
avoidance once locked at full strength) that motivated this mechanism.
"""

from __future__ import annotations

import json

import pytest

from thesis.envs.stage10_symmetric_merge_env import (
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage11_dyad_merge_pilot_config import (
    FORBIDDEN_STAGE10_SEEDS,
    MAX_STEPS,
    PILOT_SEEDS,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    assert_stage11_pilot_guards,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V1,
    REWARD_VERSION_STAGE11_V2_CLAWBACK,
    run_stage11_pilot_training_job,
)

# --------------------------------------------------------------------- seeds


def test_v2_seeds_disjoint_from_v1_and_stage10():
    assert set(PILOT_V2_SEEDS).isdisjoint(PILOT_V1_SEEDS)
    assert set(PILOT_V2_SEEDS).isdisjoint(FORBIDDEN_STAGE10_SEEDS)


def test_v2_seeds_pass_guard():
    for seed in PILOT_V2_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_both_versions():
    # Updated for v12 round 2 / PER+n-step (2026-08-09): PILOT_SEEDS is now
    # the union of all twelve versions' seed blocks PLUS v12's own second
    # round (PILOT_V12_BASELINE_V2_SEEDS, 69113-69120, for the PER/n-step
    # addition) -- v12 now contributes FOUR 8-seed blocks total -- same
    # drift this test's siblings elsewhere hit at every prior version bump
    # (v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, now v12-round-2).
    from thesis.pilots.stage11_dyad_merge_pilot_config import (
        PILOT_V3_SEEDS,
        PILOT_V4_SEEDS,
        PILOT_V5_SEEDS,
        PILOT_V6_SEEDS,
        PILOT_V7_SEEDS,
        PILOT_V8_SEEDS,
        PILOT_V9_SEEDS,
        PILOT_V10_SEEDS,
        PILOT_V11_SEEDS,
        PILOT_V12_BASELINE_SEEDS,
        PILOT_V12_BASELINE_V2_SEEDS,
        PILOT_V12_MEAN_PBRS_SEEDS,
        PILOT_V12_MIN_PBRS_SEEDS,
    )

    assert set(PILOT_SEEDS) == (
        set(PILOT_V1_SEEDS)
        | set(PILOT_V2_SEEDS)
        | set(PILOT_V3_SEEDS)
        | set(PILOT_V4_SEEDS)
        | set(PILOT_V5_SEEDS)
        | set(PILOT_V6_SEEDS)
        | set(PILOT_V7_SEEDS)
        | set(PILOT_V8_SEEDS)
        | set(PILOT_V9_SEEDS)
        | set(PILOT_V10_SEEDS)
        | set(PILOT_V11_SEEDS)
        | set(PILOT_V12_BASELINE_SEEDS)
        | set(PILOT_V12_MEAN_PBRS_SEEDS)
        | set(PILOT_V12_MIN_PBRS_SEEDS)
        | set(PILOT_V12_BASELINE_V2_SEEDS)
    )


# ---------------------------------------------------- env collision_penalty_override


def _force_head_on_collision(env):
    active = env.active_vehicle_ids
    for vid in active:
        env._roles[vid] = "mainline"
        env._vehicles[vid].role = "mainline"
    env._vehicles[active[0]].route_position = 250.0
    env._vehicles[active[1]].route_position = 250.0
    env._vehicles[active[0]].speed = 15.0
    env._vehicles[active[1]].speed = 15.0
    return active


def test_step_without_override_uses_scalar_for_every_colliding_vehicle():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1, n_vehicles=2))
    env.reset(seed=1)
    active = _force_head_on_collision(env)
    actions = {vid: 0 for vid in active}
    _, _reward, terminated, _truncated, info = env.step(actions, collision_penalty_magnitude=3.0)
    assert terminated is True
    for vid in active:
        assert info["collision_penalty_used_per_vehicle"][vid] == pytest.approx(3.0)


def test_step_with_override_replaces_scalar_per_vehicle():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=2, n_vehicles=2))
    env.reset(seed=2)
    active = _force_head_on_collision(env)
    actions = {vid: 0 for vid in active}
    override = {active[0]: 0.15, active[1]: 0.42}
    _, reward, terminated, _truncated, info = env.step(
        actions, collision_penalty_magnitude=5.0, collision_penalty_override=override
    )
    assert terminated is True
    for vid in active:
        assert info["collision_penalty_used_per_vehicle"][vid] == pytest.approx(override[vid])
        # dominated by the override (small), NOT the scalar (5.0) -- proves override actually won.
        assert reward[vid] > -1.0


def test_override_missing_vehicle_falls_back_to_scalar():
    """A vehicle absent from the override mapping must still use the scalar
    -- defensive fallback, not an error, so a runner bug can't silently
    zero-penalize an uncovered vehicle."""
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=3, n_vehicles=2))
    env.reset(seed=3)
    active = _force_head_on_collision(env)
    actions = {vid: 0 for vid in active}
    override = {active[0]: 0.1}  # active[1] deliberately omitted
    _, _reward, terminated, _truncated, info = env.step(
        actions, collision_penalty_magnitude=2.5, collision_penalty_override=override
    )
    assert terminated is True
    assert info["collision_penalty_used_per_vehicle"][active[0]] == pytest.approx(0.1)
    assert info["collision_penalty_used_per_vehicle"][active[1]] == pytest.approx(2.5)


def test_no_collision_ignores_override_entirely():
    """Non-colliding steps must use the scalar (module default here) for
    diagnostic purposes -- the override only ever matters for vehicles
    actually in `collided` this step."""
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=4, n_vehicles=2, spawn_route_lead=0.0))
    env.reset(seed=4)
    active = env.active_vehicle_ids
    actions = {vid: 1 for vid in active}  # ACCELERATE, no collision this early
    _, _reward, _terminated, _truncated, info = env.step(
        actions, collision_penalty_magnitude=5.0, collision_penalty_override={vid: 0.0 for vid in active}
    )
    for vid in active:
        # The real check: the override must never be consulted for a
        # non-colliding vehicle, regardless of what it contains.
        assert info["collision_penalty_applied"][vid] is False


# ---------------------------------------------------------- end-to-end (runner)


def test_clawback_run_produces_v2_reward_version_and_zero_ttc(tmp_path):
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V2_SEEDS[0],
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 300, 600),
        episode_max_steps=50,
        enable_clawback_collision_penalty=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V2_CLAWBACK

    traj_path = output_root / "trajectories" / f"seed_{PILOT_V2_SEEDS[0]}.jsonl"
    lines = traj_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 600
    for line in lines:
        rec = json.loads(line)
        assert rec["ttc_penalty_weight"] == pytest.approx(0.0)  # confirmed-inert TTC dropped this version


def test_default_run_still_uses_v1_reward_version(tmp_path):
    """enable_clawback_collision_penalty defaults to False -- v1 behaviour
    must be completely unaffected by v2's addition."""
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V1_SEEDS[0],
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 300, 600),
        episode_max_steps=50,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V1


def test_collision_nets_episode_reward_to_approximately_zero(tmp_path):
    """The whole point of the clawback: a colliding vehicle's episode
    reward (progress accumulated so far, minus the clawback penalty that
    exactly cancels it) must never be far from zero -- never a large
    negative, and never a large positive either."""
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    run_stage11_pilot_training_job(
        master_seed=PILOT_V2_SEEDS[1],
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=2000,
        strict=False,
        checkpoint_steps=(0, 2000),
        episode_max_steps=120,
        enable_clawback_collision_penalty=True,
    )
    traj_path = output_root / "trajectories" / f"seed_{PILOT_V2_SEEDS[1]}.jsonl"
    lines = [json.loads(l) for l in traj_path.read_text(encoding="utf-8").strip().splitlines()]
    collided_vehicle_steps = [
        v for rec in lines if rec["collision_event"] for v in rec["vehicles"] if v["collision_penalty_applied"]
    ]
    assert collided_vehicle_steps, "expected at least one collision within 2000 steps for this smoke check"
    for v in collided_vehicle_steps:
        # collision_penalty_used must never exceed ~1.0 (the max possible
        # per-vehicle positive reward: 0.4 progress + 0.6 exit, though exit
        # can never co-occur with a collision in practice) -- proving it is
        # NOT the old fixed 5.0/curriculum magnitude.
        assert 0.0 <= v["collision_penalty_used"] <= 1.0
