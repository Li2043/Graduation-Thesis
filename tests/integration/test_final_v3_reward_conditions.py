"""Stage 5A-0 — reward condition decomposition on final V3."""

from __future__ import annotations

import numpy as np

from thesis.training.final_experiment_runtime import scripted_accelerate
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.final_reward_conditions import IntegrationPBRSConfig
from thesis.training.final_v3_pipeline import run_final_v3_episode


def test_baseline_shaping_zero_and_decompositions():
    bundle = load_final_locks()
    pcfg = IntegrationPBRSConfig()
    base = run_final_v3_episode(
        bundle,
        reward_condition="baseline",
        scripted_actions=scripted_accelerate(8),
        pbrs_config=pcfg,
        episode_id="rew_base",
    )
    mean = run_final_v3_episode(
        bundle,
        reward_condition="mean_pbrs",
        scripted_actions=scripted_accelerate(8),
        pbrs_config=pcfg,
        episode_id="rew_mean",
    )
    mn = run_final_v3_episode(
        bundle,
        reward_condition="min_pbrs",
        scripted_actions=scripted_accelerate(8),
        pbrs_config=pcfg,
        episode_id="rew_min",
    )
    for r in base["transitions"]:
        assert r["shaping_component"] == 0.0
        assert r["learner_reward"] == r["base_reward"]
        assert r["decomposition_error"] <= 1e-12
    for a, b, c in zip(base["transitions"], mean["transitions"], mn["transitions"]):
        assert a["base_reward"] == b["base_reward"] == c["base_reward"]
        assert abs(b["learner_reward"] - (b["base_reward"] + b["shaping_component"])) < 1e-12
        assert abs(c["learner_reward"] - (c["base_reward"] + c["shaping_component"])) < 1e-12
        # Same physical transition → same shaping for A and B within each condition
    by_step_mean: dict[int, list] = {}
    for r in mean["transitions"]:
        by_step_mean.setdefault(r["policy_step"], []).append(r)
    for rows in by_step_mean.values():
        if len(rows) == 2:
            assert rows[0]["shaping_component"] == rows[1]["shaping_component"]


def test_lambda_zero_reproduces_baseline():
    bundle = load_final_locks()
    acts = scripted_accelerate(8)
    base = run_final_v3_episode(
        bundle, reward_condition="baseline", scripted_actions=acts, episode_id="z0"
    )
    zero = IntegrationPBRSConfig().with_lambda_zero()
    mean0 = run_final_v3_episode(
        bundle,
        reward_condition="mean_pbrs",
        scripted_actions=acts,
        pbrs_config=zero,
        episode_id="z1",
    )
    min0 = run_final_v3_episode(
        bundle,
        reward_condition="min_pbrs",
        scripted_actions=acts,
        pbrs_config=zero,
        episode_id="z2",
    )
    for a, b, c in zip(base["transitions"], mean0["transitions"], min0["transitions"]):
        assert a["learner_reward"] == b["learner_reward"] == c["learner_reward"]


def test_comfort_uses_most_negative_substep_accel():
    bundle = load_final_locks()
    ep = run_final_v3_episode(
        bundle,
        reward_condition="baseline",
        scripted_actions=[{"A": 2, "B": 2}] * 3,  # DECELERATE
        episode_id="comfort_accel",
    )
    # At least one hard-braking cost should be computed from policy-level min accel
    assert any(r["policy_level_acceleration"] <= 0.0 for r in ep["transitions"])
    for r in ep["transitions"]:
        # Reconstruction identity already checked; H >= 0
        assert r["hard_braking_cost"] >= -1e-15
