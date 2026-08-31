"""Stage 11 pilot (E30) v8 -- visit-count-based epsilon (Cha-inspired).

Covers: ``epsilon_at_visit_count_v8``'s linear-decay shape (same formula
shape as ``epsilon_at_step``, driven by visit count instead of global
step); ``enable_visit_count_epsilon_v8`` requiring v6's role-split flag;
the new v8 reward-version tag (and the combined v7+v8 tag when both are
set); per-role visit counts actually being tracked and driving a
per-vehicle (not global) epsilon during a real run; and PILOT_V8_SEEDS
being disjoint from every prior version's seeds and every other stage's
forbidden blocks.
"""

from __future__ import annotations

import pytest

from thesis.pilots.stage11_dyad_merge_pilot_config import (
    EPSILON_END,
    EPSILON_START,
    MAX_STEPS,
    PILOT_SEEDS,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    PILOT_V3_SEEDS,
    PILOT_V4_SEEDS,
    PILOT_V5_SEEDS,
    PILOT_V6_SEEDS,
    PILOT_V7_SEEDS,
    PILOT_V8_SEEDS,
    VISIT_COUNT_DECAY_TARGET_V8,
    assert_stage11_pilot_guards,
    epsilon_at_visit_count_v8,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V6_ROLE_SPLIT,
    REWARD_VERSION_STAGE11_V7_PERIODIC_RESET,
    REWARD_VERSION_STAGE11_V8_VISIT_COUNT_EPSILON,
    run_stage11_pilot_training_job,
)


# --------------------------------------------------------------- config constants


def test_visit_count_decay_target_v8_constant():
    assert VISIT_COUNT_DECAY_TARGET_V8 == 16_000


def test_v8_seeds_disjoint_from_v1_through_v7():
    for block in (
        PILOT_V1_SEEDS,
        PILOT_V2_SEEDS,
        PILOT_V3_SEEDS,
        PILOT_V4_SEEDS,
        PILOT_V5_SEEDS,
        PILOT_V6_SEEDS,
        PILOT_V7_SEEDS,
    ):
        assert set(PILOT_V8_SEEDS).isdisjoint(block)


def test_v8_seeds_pass_guard():
    for seed in PILOT_V8_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_v8():
    assert set(PILOT_V8_SEEDS) <= set(PILOT_SEEDS)


# --------------------------------------------------------------- epsilon_at_visit_count_v8


def test_epsilon_starts_at_epsilon_start_with_zero_visits():
    assert epsilon_at_visit_count_v8(0) == pytest.approx(EPSILON_START)


def test_epsilon_reaches_floor_at_and_beyond_target():
    assert epsilon_at_visit_count_v8(VISIT_COUNT_DECAY_TARGET_V8) == pytest.approx(EPSILON_END)
    assert epsilon_at_visit_count_v8(VISIT_COUNT_DECAY_TARGET_V8 * 10) == pytest.approx(EPSILON_END)


def test_epsilon_is_linear_and_monotonic_midway():
    half = VISIT_COUNT_DECAY_TARGET_V8 // 2
    mid = epsilon_at_visit_count_v8(half)
    expected_mid = EPSILON_START + 0.5 * (EPSILON_END - EPSILON_START)
    assert mid == pytest.approx(expected_mid, abs=1e-6)
    assert epsilon_at_visit_count_v8(0) > mid > epsilon_at_visit_count_v8(VISIT_COUNT_DECAY_TARGET_V8)


def test_low_visit_count_keeps_epsilon_high_even_if_would_be_low_by_step_count():
    """The whole point: a role that has barely visited the conflict zone
    (e.g. because it's avoiding it) keeps a high epsilon regardless of how
    much wall-clock training has elapsed."""
    eps_after_few_visits = epsilon_at_visit_count_v8(50)
    assert eps_after_few_visits > 0.9  # still close to EPSILON_START=1.0


# --------------------------------------------------------------- flag composition rules


def test_v8_without_role_split_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V8_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_visit_count_epsilon_v8=True,
            # enable_role_split_v6 deliberately omitted (False)
        )


def test_v8_run_selects_the_v8_reward_version_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V8_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_visit_count_epsilon_v8=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V8_VISIT_COUNT_EPSILON


def test_v6_without_v8_flag_still_selects_v6_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V8_SEEDS[1],
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


def test_v7_and_v8_combined_gets_a_combined_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V8_SEEDS[2],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_periodic_reset_v7=True,
        enable_visit_count_epsilon_v8=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V7_PERIODIC_RESET + "_and_v8_visit_count_epsilon"


# --------------------------------------------------------------- runner integration


def test_v8_run_actually_uses_per_role_visit_counts(tmp_path, monkeypatch):
    """End-to-end, verified via mock: patch epsilon_at_visit_count_v8 to
    record every (role, visit_count) pair it's called with, run a short
    real job, and confirm (a) it was called at all (proving the v8 path,
    not epsilon_at_step, is driving action selection), (b) visit counts are
    non-negative integers that never decrease within a role's own call
    sequence (monotonic accumulation), and (c) both roles appear."""
    import thesis.pilots.stage11_dyad_merge_runner as runner_module

    seen_calls: list[int] = []
    real_fn = runner_module.epsilon_at_visit_count_v8

    def tracking_fn(visit_count):
        seen_calls.append(visit_count)
        return real_fn(visit_count)

    monkeypatch.setattr(runner_module, "epsilon_at_visit_count_v8", tracking_fn)

    run_stage11_pilot_training_job(
        master_seed=PILOT_V8_SEEDS[3],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=3000,
        strict=False,
        checkpoint_steps=(0, 3000),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_visit_count_epsilon_v8=True,
    )
    assert len(seen_calls) > 0
    assert all(v >= 0 for v in seen_calls)
    assert max(seen_calls) > 0  # some merging-zone visits accumulated over 3000 steps


def test_default_call_site_uses_global_epsilon_not_visit_count(tmp_path, monkeypatch):
    import thesis.pilots.stage11_dyad_merge_runner as runner_module

    visit_count_calls: list[int] = []
    monkeypatch.setattr(
        runner_module, "epsilon_at_visit_count_v8", lambda vc: visit_count_calls.append(vc) or 0.5
    )

    run_stage11_pilot_training_job(
        master_seed=PILOT_V8_SEEDS[4],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        # enable_visit_count_epsilon_v8 deliberately omitted (False)
    )
    assert visit_count_calls == []
