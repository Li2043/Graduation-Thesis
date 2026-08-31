"""Stage 11 pilot (E30) v4 -- Stage 9-based reward redesign, runner-level.

Covers: ``enable_stage9_based_reward`` selecting flat (non-curriculum)
collision penalty at Stage 9's proven 1.0 (not v6/v7's 5.0), TTC forced to
0.0, no clawback, plus the hard-braking/time-cost terms wired through to
``env.step()``; mutual exclusivity with ``enable_clawback_collision_penalty``;
the new v4 reward-version tag; and PILOT_V4_SEEDS being disjoint from every
prior version's seeds and every other stage's forbidden blocks.
"""

from __future__ import annotations

import json

import pytest

from thesis.pilots.stage11_dyad_merge_pilot_config import (
    COLLISION_PENALTY_MAGNITUDE_V4,
    EXIT_REWARD_MAGNITUDE_V4,
    HARD_BRAKING_ETA_V4,
    MAX_STEPS,
    PILOT_SEEDS,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    PILOT_V3_SEEDS,
    PILOT_V4_SEEDS,
    TIME_COST_PER_STEP_V4,
    TTC_PENALTY_WEIGHT_V4,
    assert_stage11_pilot_guards,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V1,
    REWARD_VERSION_STAGE11_V4_STAGE9_BASED,
    run_stage11_pilot_training_job,
)


def test_v4_constants_match_stage9_proven_values():
    assert COLLISION_PENALTY_MAGNITUDE_V4 == pytest.approx(1.0)
    assert TTC_PENALTY_WEIGHT_V4 == pytest.approx(0.0)
    assert EXIT_REWARD_MAGNITUDE_V4 == pytest.approx(0.6)
    assert HARD_BRAKING_ETA_V4 == pytest.approx(0.015)
    assert TIME_COST_PER_STEP_V4 == pytest.approx(0.0005)


def test_v4_seeds_disjoint_from_v1_v2_v3():
    assert set(PILOT_V4_SEEDS).isdisjoint(PILOT_V1_SEEDS)
    assert set(PILOT_V4_SEEDS).isdisjoint(PILOT_V2_SEEDS)
    assert set(PILOT_V4_SEEDS).isdisjoint(PILOT_V3_SEEDS)


def test_v4_seeds_pass_guard():
    for seed in PILOT_V4_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_v4():
    assert set(PILOT_V4_SEEDS) <= set(PILOT_SEEDS)


def test_enable_stage9_based_reward_and_clawback_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V4_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward=True,
            enable_clawback_collision_penalty=True,
        )


def test_v4_run_selects_the_v4_reward_version_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V4_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V4_STAGE9_BASED


def test_default_call_site_still_selects_v1_tag(tmp_path):
    """enable_stage9_based_reward defaults to False -- v1-v3 call sites
    (which never pass it) are unaffected."""
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V4_SEEDS[1],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V1


def test_v4_trajectory_logs_flat_collision_penalty_and_new_terms(tmp_path):
    """End-to-end: run long enough to see real steps and confirm the
    trajectory log reports the flat (non-curriculum) 1.0 collision magnitude
    throughout, TTC weight pinned at 0.0, and non-trivial hard-braking-eta/
    time-cost-per-step diagnostics on every logged step."""
    output_root = tmp_path / "output"
    run_stage11_pilot_training_job(
        master_seed=PILOT_V4_SEEDS[2],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=2000,
        strict=False,
        checkpoint_steps=(0, 2000),
        episode_max_steps=150,
        enable_stage9_based_reward=True,
    )
    traj_path = output_root / "trajectories" / f"seed_{PILOT_V4_SEEDS[2]}.jsonl"
    lines = [json.loads(l) for l in traj_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) > 0
    for rec in lines:
        assert rec["collision_penalty_magnitude"] == pytest.approx(COLLISION_PENALTY_MAGNITUDE_V4)
        assert rec["ttc_penalty_weight"] == pytest.approx(0.0)
        assert rec["hard_braking_eta"] == pytest.approx(HARD_BRAKING_ETA_V4)
        assert rec["time_cost_per_step"] == pytest.approx(TIME_COST_PER_STEP_V4)
        for v in rec["vehicles"]:
            assert "hard_braking_cost" in v
            assert "time_cost_applied" in v
            # collision penalty never exceeds Stage 9's flat 1.0 (no clawback
            # inflation) even on a colliding step.
            assert 0.0 <= v["collision_penalty_used"] <= 1.0


def test_v4_run_never_uses_the_v6v7_5point0_magnitude(tmp_path):
    output_root = tmp_path / "output"
    run_stage11_pilot_training_job(
        master_seed=PILOT_V4_SEEDS[3],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=1500,
        strict=False,
        checkpoint_steps=(0, 1500),
        episode_max_steps=150,
        enable_stage9_based_reward=True,
    )
    traj_path = output_root / "trajectories" / f"seed_{PILOT_V4_SEEDS[3]}.jsonl"
    lines = [json.loads(l) for l in traj_path.read_text(encoding="utf-8").strip().splitlines()]
    assert all(rec["collision_penalty_magnitude"] != pytest.approx(5.0) for rec in lines)
