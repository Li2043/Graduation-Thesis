"""Stage 5A-0 — replay / controller-terminal semantics on V3."""

from __future__ import annotations

from thesis.training.final_experiment_runtime import (
    scripted_a_accel_b_maintain,
    scripted_accelerate,
)
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.final_reward_conditions import IntegrationPBRSConfig
from thesis.training.final_v3_pipeline import run_final_v3_episode


def test_safe_exit_controller_terminal_and_peer_continues():
    bundle = load_final_locks()
    ep = run_final_v3_episode(
        bundle,
        reward_condition="baseline",
        scripted_actions=scripted_a_accel_b_maintain(70),
        pbrs_config=IntegrationPBRSConfig(),
        episode_id="replay_exit",
    )
    a_rows = [r for r in ep["transitions"] if r["controller_id"] == "A"]
    b_rows = [r for r in ep["transitions"] if r["controller_id"] == "B"]
    exit_a = [r for r in a_rows if r["exit_event"]["A"] >= 1.0]
    assert exit_a
    assert exit_a[0]["controller_terminal"] is True
    assert exit_a[0]["next_observation"] is None
    assert exit_a[0]["next_action_mask"] is None
    # No later A rows after exit
    exit_step = exit_a[0]["policy_step"]
    assert all(r["policy_step"] <= exit_step for r in a_rows)
    # B continues after A exit
    assert any(r["policy_step"] > exit_step for r in b_rows)
    # Exited A contributes E=1 in peer potential at exit transition
    assert exit_a[0]["experiences_t1"]["A"] == 1.0
    peer_after = [r for r in b_rows if r["policy_step"] > exit_step]
    assert peer_after
    assert peer_after[0]["experiences_t"]["A"] == 1.0


def test_collision_and_success_targets_equal_reward():
    bundle = load_final_locks()
    coll = run_final_v3_episode(
        bundle,
        reward_condition="baseline",
        scripted_actions=scripted_accelerate(30),
        episode_id="replay_coll",
    )
    last = coll["transitions"][-1]
    assert last["terminated"] is True
    assert last["controller_terminal"] is True
    assert last["target"] == last["learner_reward"]
    assert last["bootstrap_multiplier"] == 0.0

    success = run_final_v3_episode(
        bundle,
        reward_condition="baseline",
        scripted_actions=scripted_a_accel_b_maintain(80),
        episode_id="replay_success",
    )
    term_rows = [r for r in success["transitions"] if r["terminated"]]
    assert term_rows
    for r in term_rows:
        assert r["controller_terminal"] is True
        assert r["target"] == r["learner_reward"]
        assert r["next_observation"] is None


def test_truncation_bootstraps_with_next_state():
    bundle = load_final_locks()
    ep = run_final_v3_episode(
        bundle,
        reward_condition="baseline",
        scripted_actions=scripted_accelerate(10),
        max_policy_steps=3,
        episode_id="replay_trunc",
    )
    trunc_rows = [r for r in ep["transitions"] if r["truncated"]]
    assert trunc_rows
    for r in trunc_rows:
        assert r["controller_terminal"] is False
        assert r["next_observation"] is not None
        assert r["next_action_mask"] is not None
        assert r["bootstrap_multiplier"] == 1.0
        assert len(r["observation"]) == 27


def test_strict_action_mask_and_obs_dim():
    bundle = load_final_locks()
    ep = run_final_v3_episode(
        bundle,
        reward_condition="baseline",
        scripted_actions=scripted_accelerate(3),
        episode_id="mask_obs",
    )
    for r in ep["transitions"]:
        assert r["observation_dim"] == 27
        assert len(r["action_mask"]) == 3
        assert all(isinstance(x, bool) for x in r["action_mask"])
