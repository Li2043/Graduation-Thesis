"""Tests for the Stage 11 confirmatory protocol's (DRAFT) eval-seed module."""

from __future__ import annotations

import pytest

from thesis.pilots.stage11_confirmatory_config import (
    ALL_CONFIRMATORY_SEEDS,
    ASSIGNMENTS_PER_BLOCK,
    BASELINE_SEEDS,
    MEAN_PBRS_SEEDS,
    MIN_PBRS_SEEDS,
    N_VALIDATION_BLOCKS,
    STAGE11_RESERVED_SEED_BLOCK,
)
from thesis.pilots.stage11_confirmatory_eval_seeds import (
    assert_eval_seeds_disjoint_from_pilot_and_training,
    assert_no_eval_seed_overlap,
    eval_plan_for_checkpoint,
    find_swapped_assignment_seed,
    role_of_ramp,
    stable_eval_seed,
)


def test_confirmatory_seed_blocks_disjoint_from_each_other():
    assert set(BASELINE_SEEDS).isdisjoint(MEAN_PBRS_SEEDS)
    assert set(BASELINE_SEEDS).isdisjoint(MIN_PBRS_SEEDS)
    assert set(MEAN_PBRS_SEEDS).isdisjoint(MIN_PBRS_SEEDS)


def test_confirmatory_seed_blocks_each_have_eight_seeds():
    assert len(BASELINE_SEEDS) == 8
    assert len(MEAN_PBRS_SEEDS) == 8
    assert len(MIN_PBRS_SEEDS) == 8
    assert len(ALL_CONFIRMATORY_SEEDS) == 24


def test_confirmatory_seeds_disjoint_from_all_pilot_seeds():
    assert set(ALL_CONFIRMATORY_SEEDS).isdisjoint(STAGE11_RESERVED_SEED_BLOCK)


def test_stable_eval_seed_is_deterministic():
    a = stable_eval_seed(master_seed=69121, checkpoint_step=350000, scenario_block=0)
    b = stable_eval_seed(master_seed=69121, checkpoint_step=350000, scenario_block=0)
    assert a == b


def test_stable_eval_seed_differs_across_blocks():
    seeds = {
        stable_eval_seed(master_seed=69121, checkpoint_step=350000, scenario_block=b)
        for b in range(N_VALIDATION_BLOCKS)
    }
    assert len(seeds) == N_VALIDATION_BLOCKS


def test_role_of_ramp_is_deterministic():
    seed = stable_eval_seed(master_seed=69121, checkpoint_step=350000, scenario_block=0)
    assert role_of_ramp(seed) == role_of_ramp(seed)


def test_role_of_ramp_returns_v0_or_v1():
    seed = stable_eval_seed(master_seed=69121, checkpoint_step=350000, scenario_block=0)
    assert role_of_ramp(seed) in ("V0", "V1")


def test_find_swapped_assignment_seed_actually_swaps_role():
    base = stable_eval_seed(master_seed=69121, checkpoint_step=350000, scenario_block=0)
    swapped = find_swapped_assignment_seed(base)
    assert swapped != base
    assert role_of_ramp(swapped) != role_of_ramp(base)


def test_find_swapped_assignment_seed_deterministic():
    base = stable_eval_seed(master_seed=69121, checkpoint_step=350000, scenario_block=0)
    assert find_swapped_assignment_seed(base) == find_swapped_assignment_seed(base)


def test_eval_plan_has_expected_size_and_shape():
    plan = eval_plan_for_checkpoint(master_seed=69121, checkpoint_step=350000)
    assert len(plan) == N_VALIDATION_BLOCKS * ASSIGNMENTS_PER_BLOCK
    for block in range(N_VALIDATION_BLOCKS):
        rows = [r for r in plan if r["scenario_block"] == block]
        assert len(rows) == 2
        assignments = {r["assignment"] for r in rows}
        assert assignments == {0, 1}


def test_eval_plan_assignment_pair_has_swapped_roles():
    plan = eval_plan_for_checkpoint(master_seed=69121, checkpoint_step=350000)
    for block in range(N_VALIDATION_BLOCKS):
        rows = {r["assignment"]: r for r in plan if r["scenario_block"] == block}
        role0 = role_of_ramp(int(rows[0]["eval_seed"]))
        role1 = role_of_ramp(int(rows[1]["eval_seed"]))
        assert role0 != role1


def test_eval_plan_differs_across_checkpoints_for_same_seed():
    plan_a = eval_plan_for_checkpoint(master_seed=69121, checkpoint_step=350000)
    plan_b = eval_plan_for_checkpoint(master_seed=69121, checkpoint_step=375000)
    seeds_a = {r["eval_seed"] for r in plan_a}
    seeds_b = {r["eval_seed"] for r in plan_b}
    assert seeds_a.isdisjoint(seeds_b)


def test_assert_no_eval_seed_overlap_passes_for_disjoint_masters():
    assert_no_eval_seed_overlap(master_seeds=[69121, 69122], checkpoint_steps=[350000, 375000, 400000])


def test_assert_eval_seeds_disjoint_from_pilot_and_training_passes():
    assert_eval_seeds_disjoint_from_pilot_and_training(confirmatory_master_seeds=ALL_CONFIRMATORY_SEEDS)


def test_assert_eval_seeds_disjoint_from_pilot_and_training_catches_overlap():
    with pytest.raises(AssertionError):
        assert_eval_seeds_disjoint_from_pilot_and_training(
            confirmatory_master_seeds=[69100],  # inside the pilot block
        )
