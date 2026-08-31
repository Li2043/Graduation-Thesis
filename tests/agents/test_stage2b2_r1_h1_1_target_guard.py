"""Stage 2B-2R-H1.1 — direct target terminal-invariant guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thesis.agents.dqn_targets import compute_dqn_target, compute_dqn_targets_batch
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


def test_scalar_rejects_terminated_without_controller_terminal():
    with pytest.raises(
        ValueError, match="terminated=True requires controller_terminal=True"
    ):
        compute_dqn_target(
            -1.0,
            controller_terminal=False,
            truncated=False,
            gamma=0.9,
            next_q_values=[1.0, 2.0, 3.0],
            next_action_mask=[True, True, True],
            terminated=True,
        )


def test_batch_rejects_terminated_non_controller_terminal_row():
    with pytest.raises(ValueError, match=r"offending row indices: \[1\]"):
        compute_dqn_targets_batch(
            rewards=[0.5, -1.0, 0.2],
            controller_terminal=[False, False, False],
            truncated=[False, False, False],
            gamma=0.9,
            next_q_values=np.ones((3, 3), dtype=np.float64),
            next_action_masks=np.ones((3, 3), dtype=bool),
            terminated=[False, True, False],
        )


def test_valid_collision_terminal_equals_reward():
    bd = compute_dqn_target(
        -1.0,
        controller_terminal=True,
        truncated=False,
        gamma=0.995,
        next_q_values=None,
        next_action_mask=None,
        terminated=True,
    )
    assert bd.target == pytest.approx(-1.0)
    assert bd.bootstrap_multiplier == 0.0
    assert bd.next_q_values is None
    assert bd.next_action_mask is None


def test_valid_success_terminal_equals_reward():
    bd = compute_dqn_target(
        0.6,
        controller_terminal=True,
        truncated=False,
        gamma=0.995,
        next_q_values=[9.0, 9.0, 9.0],  # ignored
        next_action_mask=np.array([0.0, 1.0, 0.5]),  # ignored / not validated
        terminated=True,
    )
    assert bd.target == pytest.approx(0.6)
    assert bd.bootstrap_multiplier == 0.0


def test_truncation_only_still_bootstraps():
    bd = compute_dqn_target(
        1.0,
        controller_terminal=False,
        truncated=True,
        gamma=0.9,
        next_q_values=[5.0, 0.0, 0.0],
        next_action_mask=[True, True, True],
        terminated=False,
    )
    assert bd.target == pytest.approx(5.5)
    assert bd.bootstrap_multiplier == 1.0


def test_lock_hashes_unchanged():
    assert sha256_file(ENV_LOCK) == ENV_SHA
    assert sha256_file(COMFORT_LOCK) == COMFORT_SHA
