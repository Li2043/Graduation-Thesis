"""Stage 2B-2R-H1 — DQN ingress and replay-semantics closure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thesis.agents.dqn_pipeline import build_transition_for_controller
from thesis.agents.dqn_targets import compute_dqn_targets_batch
from thesis.agents.replay_buffer_v2 import ReplayBuffer, ReplayTransition
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


def _info(*, exit_event: float = 0.0, completed: bool = False, step: int = 1):
    return {
        "diagnostics": {
            "per_agent": {
                "A": {
                    "base_total": 0.4,
                    "scaled_mean_shaping": 0.0,
                    "scaled_min_shaping": 0.0,
                }
            }
        },
        "vehicles_t": {"A": {"role": "mainline"}},
        "events": {"exit_event": {"A": exit_event}},
        "completion": {"A": completed},
        "step": step,
    }


def _tr(**kw) -> ReplayTransition:
    base = dict(
        observation=np.zeros(4),
        action=0,
        shaped_reward=0.1,
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


def test_float_mask_rejected_by_build_transition():
    with pytest.raises(ValueError, match="float"):
        build_transition_for_controller(
            controller_id="A",
            obs=np.zeros(4),
            next_obs=np.ones(4),
            action=0,
            action_mask=np.array([0.0, 1.0, 1.0], dtype=np.float64),
            next_action_mask=np.array([True, True, True]),
            terminated=False,
            truncated=False,
            info=_info(),
            reward_condition="baseline",
            episode_id="e0",
        )


def test_half_mask_cannot_be_hidden_by_pipeline_coercion():
    # Previously np.asarray(..., dtype=bool) would coerce 0.5 -> True.
    with pytest.raises(ValueError, match="float"):
        build_transition_for_controller(
            controller_id="A",
            obs=np.zeros(4),
            next_obs=np.ones(4),
            action=0,
            action_mask=np.array([1.0, 0.5, 0.0], dtype=np.float32),
            next_action_mask=np.array([True, True, True]),
            terminated=False,
            truncated=False,
            info=_info(),
            reward_condition="baseline",
            episode_id="e0",
        )


def test_malformed_terminal_next_mask_canonicalised_to_none():
    tr = _tr(
        terminated=True,
        controller_terminal=True,
        learner_completed=False,
        next_observation=np.array([9.0, 9.0, 9.0, 9.0]),
        next_action_mask=np.array([0.0, 1.0, 0.5], dtype=np.float64),
    )
    tr.validate(n_actions=3, obs_dim=4)
    assert tr.next_observation is None
    assert tr.next_action_mask is None


def test_terminated_without_controller_terminal_rejected():
    buf = ReplayBuffer(4, obs_dim=4, n_actions=3, seed=0)
    with pytest.raises(ValueError, match="terminated=True requires controller_terminal"):
        buf.append(
            _tr(
                terminated=True,
                controller_terminal=False,
                truncated=False,
            )
        )


def test_truncation_only_controller_terminal_rejected():
    buf = ReplayBuffer(4, obs_dim=4, n_actions=3, seed=0)
    with pytest.raises(ValueError, match="truncated-only controller_terminal"):
        buf.append(
            _tr(
                terminated=False,
                truncated=True,
                controller_terminal=True,
                learner_completed=False,
                next_observation=None,
                next_action_mask=None,
            )
        )


def test_learner_completed_requires_controller_terminal():
    buf = ReplayBuffer(4, obs_dim=4, n_actions=3, seed=0)
    with pytest.raises(ValueError, match="learner_completed=True requires"):
        buf.append(
            _tr(
                learner_completed=True,
                controller_terminal=False,
                terminated=False,
                truncated=False,
            )
        )


def test_malformed_bootstrap_indices_rejected():
    rewards = [1.0, 2.0, 3.0]
    cterm = [False, True, False]
    trunc = [False, False, True]
    nq = np.array([[5.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    nm = np.array([[True, True, True], [True, True, True]])

    with pytest.raises(ValueError, match="unique"):
        compute_dqn_targets_batch(
            rewards,
            cterm,
            trunc,
            0.9,
            nq,
            nm,
            terminated=[False, True, False],
            bootstrap_indices=[0, 0, 2],
        )

    with pytest.raises(ValueError, match="out of range"):
        compute_dqn_targets_batch(
            rewards,
            cterm,
            trunc,
            0.9,
            nq,
            nm,
            terminated=[False, True, False],
            bootstrap_indices=[0, 3],
        )

    with pytest.raises(ValueError, match="all and only non-terminal"):
        compute_dqn_targets_batch(
            rewards,
            cterm,
            trunc,
            0.9,
            nq,
            nm,
            terminated=[False, True, False],
            bootstrap_indices=[0],  # missing index 2
        )

    with pytest.raises(ValueError, match="all and only non-terminal"):
        compute_dqn_targets_batch(
            rewards,
            cterm,
            trunc,
            0.9,
            nq,
            nm,
            terminated=[False, True, False],
            bootstrap_indices=[0, 1, 2],  # includes terminal index 1
        )


def test_valid_bootstrap_indices_path_still_works():
    outs = compute_dqn_targets_batch(
        [1.0, 2.0, 3.0],
        [False, True, False],
        [False, False, True],
        0.9,
        np.array([[5.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[True, True, True], [True, True, True]]),
        terminated=[False, True, False],
        bootstrap_indices=[0, 2],
    )
    assert outs[0].target == pytest.approx(5.5)
    assert outs[1].target == pytest.approx(2.0)
    assert outs[2].target == pytest.approx(6.6)


def test_lock_hashes_unchanged():
    assert sha256_file(ENV_LOCK) == ENV_SHA
    assert sha256_file(COMFORT_LOCK) == COMFORT_SHA
