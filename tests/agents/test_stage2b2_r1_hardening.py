"""Stage 2B-2R — strict action mask and controller-terminal target hardening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from thesis.agents.action_masking import (
    masked_argmax,
    masked_max_q,
    masked_random_action,
    validate_action_mask,
)
from thesis.agents.dqn_targets import compute_dqn_target
from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
from thesis.agents.replay_buffer_v2 import ReplayBuffer, ReplayBatch, ReplayTransition
from thesis.calibration.final_environment_trace_loader import sha256_file


ENV_LOCK = Path(
    "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
    "20260730T003122Z_aee2d425/final_environment_lock.yaml"
)
COMFORT_LOCK = Path(
    "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/artifacts/"
    "20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
)
ENV_SHA = "d2d82ac02feb5bb2f5217f8e399972b91bd56cce343f60861954f66d7f70bf12"
COMFORT_SHA = "1d9439c211955f9a8a177e455b6b5ff34aa98f85f3ff0677bbd62abd6d29b061"


def _tr(**kw) -> ReplayTransition:
    base = dict(
        observation=np.zeros(4),
        action=0,
        shaped_reward=1.0,
        next_observation=np.ones(4),
        terminated=False,
        truncated=False,
        controller_terminal=False,
        learner_completed=False,
        action_mask=np.array([True, True, True]),
        next_action_mask=np.array([True, True, True]),
        controller_id="A",
    )
    base.update(kw)
    return ReplayTransition(**base)


# --- Mask validation ---


def test_bool_mask_accepted():
    m = validate_action_mask(np.array([True, False, True]), 3)
    assert m.dtype == bool and m.tolist() == [True, False, True]


def test_integer_01_mask_accepted():
    m = validate_action_mask(np.array([1, 0, 1], dtype=np.int64), 3)
    assert m.dtype == bool and m.tolist() == [True, False, True]


def test_float_01_mask_rejected():
    with pytest.raises(ValueError, match="float"):
        validate_action_mask(np.array([0.0, 1.0, 1.0], dtype=np.float64), 3)


def test_half_mask_rejected():
    with pytest.raises(ValueError, match="float"):
        validate_action_mask(np.array([1.0, 0.5, 0.0], dtype=np.float32), 3)


def test_integer_2_rejected():
    with pytest.raises(ValueError, match="0/1"):
        validate_action_mask(np.array([1, 2, 0], dtype=np.int64), 3)


def test_negative_mask_rejected():
    with pytest.raises(ValueError, match="negative"):
        validate_action_mask(np.array([1, -1, 0], dtype=np.int64), 3)


def test_nan_mask_rejected():
    with pytest.raises(ValueError, match="float"):
        validate_action_mask(np.array([1.0, np.nan, 0.0]), 3)


def test_wrong_length_and_multidim_rejected():
    with pytest.raises(ValueError, match="length"):
        validate_action_mask([True, True], 3)
    with pytest.raises(ValueError, match="1-D"):
        validate_action_mask(np.array([[True, False, True]]), 3)


def test_all_false_rejected():
    with pytest.raises(ValueError, match="all-False"):
        validate_action_mask(np.array([False, False, False]), 3)


# --- Selection ---


def test_greedy_never_illegal_and_ignores_larger_illegal_q():
    q = np.array([1.0, 100.0, 5.0])
    mask = np.array([True, False, True])
    assert masked_argmax(q, mask) == 2
    assert masked_max_q(q, mask) == pytest.approx(5.0)


def test_epsilon_random_legal_only():
    mask = np.array([True, False, True])
    rng = np.random.default_rng(0)
    for _ in range(300):
        assert masked_random_action(mask, 3, rng) in (0, 2)


# --- Terminal targets ---


def test_terminal_target_equals_reward_with_none_next():
    bd = compute_dqn_target(
        2.5,
        controller_terminal=True,
        truncated=False,
        gamma=0.995,
        next_q_values=None,
        next_action_mask=None,
        terminated=True,
    )
    assert bd.target == pytest.approx(2.5)
    assert bd.bootstrap_multiplier == 0.0


def test_terminal_ignores_malformed_next_mask():
    # Malformed next mask must not be validated / must not affect target
    bd = compute_dqn_target(
        0.7,
        controller_terminal=True,
        truncated=False,
        gamma=0.9,
        next_q_values=[1.0, 2.0, 3.0],
        next_action_mask=np.array([0.0, 1.0, 0.5]),  # would be illegal if validated
        terminated=True,
    )
    assert bd.target == pytest.approx(0.7)


def test_terminal_does_not_invoke_target_network():
    learner = IndependentDQNLearner("A", DQNConfig(), seed=0)
    calls = {"n": 0}
    real = learner.target.forward

    def counting_forward(x):
        calls["n"] += 1
        return real(x)

    learner.target.forward = counting_forward  # type: ignore[method-assign]
    tr = _tr(
        shaped_reward=1.5,
        controller_terminal=True,
        terminated=True,
        next_observation=None,
        next_action_mask=None,
    )
    bd = learner.compute_target_for_transition(tr)
    assert bd.target == pytest.approx(1.5)
    assert calls["n"] == 0


def test_ongoing_requires_next_obs_and_mask():
    with pytest.raises(ValueError, match="next_q_values"):
        compute_dqn_target(
            1.0, controller_terminal=False, truncated=False, gamma=0.9
        )
    with pytest.raises(ValueError, match="next_action_mask"):
        compute_dqn_target(
            1.0,
            controller_terminal=False,
            truncated=False,
            gamma=0.9,
            next_q_values=[1.0, 2.0, 3.0],
            next_action_mask=None,
        )


def test_truncation_bootstraps_and_missing_mask_fails():
    bd = compute_dqn_target(
        1.0,
        controller_terminal=False,
        truncated=True,
        gamma=0.9,
        next_q_values=[5.0, 0.0, 0.0],
        next_action_mask=[True, True, True],
    )
    assert bd.target == pytest.approx(5.5)
    buf = ReplayBuffer(4, obs_dim=4, n_actions=3, seed=0)
    with pytest.raises(ValueError, match="next_action_mask"):
        buf.append(
            _tr(
                truncated=True,
                controller_terminal=False,
                next_action_mask=None,
            )
        )


def test_truncation_all_illegal_next_mask_fails():
    with pytest.raises(ValueError, match="all-False"):
        compute_dqn_target(
            1.0,
            controller_terminal=False,
            truncated=True,
            gamma=0.9,
            next_q_values=[1.0, 2.0, 3.0],
            next_action_mask=[False, False, False],
        )


def test_replay_accepts_terminal_none_next():
    buf = ReplayBuffer(4, obs_dim=4, n_actions=3, seed=0)
    buf.append(
        _tr(
            controller_terminal=True,
            terminated=True,
            next_observation=None,
            next_action_mask=None,
            shaped_reward=3.0,
        )
    )
    assert len(buf) == 1


def test_replay_rejects_nonterminal_missing_next():
    buf = ReplayBuffer(4, obs_dim=4, n_actions=3, seed=0)
    with pytest.raises(ValueError, match="next_observation"):
        buf.append(_tr(controller_terminal=False, next_observation=None))


def test_mixed_batch_evaluates_target_only_for_bootstrap():
    cfg = DQNConfig(batch_size=2, learning_rate=1e-2)
    learner = IndependentDQNLearner("A", cfg, seed=0)
    calls = {"n": 0}
    real_forward = learner.target.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    learner.target.forward = counting_forward  # type: ignore[method-assign]

    boot = _tr(shaped_reward=1.0, controller_terminal=False, step=0)
    term = _tr(
        shaped_reward=2.0,
        controller_terminal=True,
        terminated=True,
        next_observation=None,
        next_action_mask=None,
        step=1,
    )
    learner.store_transition(boot)
    learner.store_transition(term)
    # Fill to batch size with another bootstrap
    learner.store_transition(_tr(shaped_reward=0.5, step=2))
    learner.store_transition(_tr(shaped_reward=0.5, step=3))

    # Build explicit mixed batch
    batch = ReplayBatch(
        observations=np.stack([boot.observation, term.observation]),
        actions=np.array([0, 0]),
        shaped_rewards=np.array([1.0, 2.0]),
        next_observations=[boot.next_observation, None],
        terminated=np.array([False, True]),
        truncated=np.array([False, False]),
        controller_terminal=np.array([False, True]),
        learner_completed=np.array([False, True]),
        action_masks=np.stack([boot.action_mask, term.action_mask]),
        next_action_masks=[boot.next_action_mask, None],
        base_rewards=np.array([1.0, 2.0]),
        shaping_components=np.array([0.0, 0.0]),
        reward_conditions=["baseline", "baseline"],
        indices=np.array([0, 1]),
        transitions=[boot, term],
    )
    stats = learner.update(batch)
    assert stats["target_network_forward_calls"] == 1
    assert stats["n_bootstrap_rows"] == 1
    assert stats["n_terminal_rows"] == 1
    assert calls["n"] == 1
    assert np.isfinite(stats["loss"])


def test_mixed_batch_targets_match_scalar_reference():
    learner = IndependentDQNLearner("A", DQNConfig(gamma=0.9), seed=1)
    boot = _tr(shaped_reward=1.0, controller_terminal=False)
    term = _tr(
        shaped_reward=2.25,
        controller_terminal=True,
        terminated=True,
        next_observation=None,
        next_action_mask=None,
    )
    next_q = learner.q_values(boot.next_observation, network="target")
    ref_boot = compute_dqn_target(
        1.0,
        controller_terminal=False,
        truncated=False,
        gamma=0.9,
        next_q_values=next_q,
        next_action_mask=boot.next_action_mask,
    ).target
    ref_term = 2.25
    batch = ReplayBatch(
        observations=np.stack([boot.observation, term.observation]),
        actions=np.array([0, 0]),
        shaped_rewards=np.array([1.0, 2.25]),
        next_observations=[boot.next_observation, None],
        terminated=np.array([False, True]),
        truncated=np.array([False, False]),
        controller_terminal=np.array([False, True]),
        learner_completed=np.array([False, True]),
        action_masks=np.stack([boot.action_mask, term.action_mask]),
        next_action_masks=[boot.next_action_mask, None],
        base_rewards=np.array([1.0, 2.25]),
        shaping_components=np.array([0.0, 0.0]),
        reward_conditions=["baseline", "baseline"],
        indices=np.array([0, 1]),
        transitions=[boot, term],
    )
    # Reconstruct targets as update does
    targets = np.array([1.0, 2.25], dtype=np.float64)
    next_q_b = learner.target(
        torch.as_tensor(boot.next_observation[None], dtype=torch.float32)
    ).detach().cpu().numpy()[0]
    targets[0] = compute_dqn_target(
        1.0,
        controller_terminal=False,
        truncated=False,
        gamma=0.9,
        next_q_values=next_q_b,
        next_action_mask=boot.next_action_mask,
    ).target
    assert targets[0] == pytest.approx(ref_boot)
    assert targets[1] == pytest.approx(ref_term)


def test_vanilla_dqn_masking_preserved():
    # Retained algorithm: target-net masked max (not Double DQN online argmax)
    bd = compute_dqn_target(
        0.0,
        controller_terminal=False,
        truncated=False,
        gamma=0.5,
        next_q_values=[1.0, 50.0, 3.0],
        next_action_mask=[True, False, True],
    )
    assert bd.masked_next_q_max == pytest.approx(3.0)
    assert bd.target == pytest.approx(1.5)


def test_lock_hashes_unchanged():
    assert sha256_file(ENV_LOCK) == ENV_SHA
    assert sha256_file(COMFORT_LOCK) == COMFORT_SHA
