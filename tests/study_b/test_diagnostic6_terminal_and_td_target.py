"""Diagnostic_6_DQN_Pipeline_Verification_Protocol.md secs 6B/6C/23
(Tests T1/T2/B1-B4): hand-verified 1-step TD target and terminal/done
handling, against ``compute_td_targets`` -- the pure function extracted
from ``SharedDQNLearner.update()`` specifically so this logic can be
checked against known values without mocking or training a real
network. ``compute_bootstrap_values`` (masked argmax/gather over network
outputs) is exercised separately below with tiny fixed-output mock
networks, matching the document's "next_Q=[0.1,0.7,0.4]" style examples
exactly."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from thesis.agents.dqn_bootstrap import DQNTargetMode, compute_bootstrap_values, compute_td_targets

GAMMA = 0.995  # exact training gamma, read from stage11_dyad_merge_pilot_config.GAMMA (see test below)


def test_gamma_constant_matches_actual_training_config():
    from thesis.pilots.stage11_dyad_merge_pilot_config import GAMMA as ACTUAL_GAMMA

    assert ACTUAL_GAMMA == GAMMA


# ------------------------------------------------------------------ T1/T2 (6C)

def test_t1_terminal_target_ignores_huge_next_q():
    """Document's exact T1: done=True, next_Q=[100,200,300] (huge, deliberate),
    reward=-1 -> expected target=-1 exactly. If the target leaks any
    bootstrap term, this would instead be close to -1 + gamma*300 ~ 298."""
    targets = compute_td_targets(
        shaped_rewards=[-1.0], controller_terminal=[True], bootstrap_indices=[], next_values=[], gamma=GAMMA,
    )
    assert targets[0] == pytest.approx(-1.0)


def test_t2_non_terminal_target_matches_hand_calculation():
    """Document's exact T2: done=False, reward=0.2, max_next_Q=0.7 ->
    expected target = 0.2 + gamma*0.7."""
    targets = compute_td_targets(
        shaped_rewards=[0.2], controller_terminal=[False], bootstrap_indices=[0], next_values=[0.7], gamma=GAMMA,
    )
    expected = 0.2 + GAMMA * 0.7
    assert targets[0] == pytest.approx(expected, abs=1e-6)


# ------------------------------------------------------------------ B1-B4 (6B)

def test_b1_completion_terminal_bootstrap_is_zero():
    targets = compute_td_targets(
        shaped_rewards=[0.6], controller_terminal=[True], bootstrap_indices=[], next_values=[], gamma=GAMMA,
    )
    # No bootstrap term at all was added -- target is exactly the reward.
    assert targets[0] == pytest.approx(0.6)


def test_b2_collision_terminal_bootstrap_is_zero():
    targets = compute_td_targets(
        shaped_rewards=[-1.0], controller_terminal=[True], bootstrap_indices=[], next_values=[], gamma=GAMMA,
    )
    assert targets[0] == pytest.approx(-1.0)


def test_b3_timeout_terminal_bootstrap_is_zero():
    # Timeout is truncated=True in the env, but MUST still be
    # controller_terminal=True for the learner (sec 6.3: "learning_done =
    # terminated or truncated"), which is exactly what heterogeneous_env's
    # callers already set (controller_terminal = terminated or exit_this_step,
    # with truncation-driven episode end folded into that same flag at the
    # call site -- see train_dqn_direct_welfare.py's episode_over handling).
    targets = compute_td_targets(
        shaped_rewards=[-0.5], controller_terminal=[True], bootstrap_indices=[], next_values=[], gamma=GAMMA,
    )
    assert targets[0] == pytest.approx(-0.5)


def test_b4_ordinary_transition_bootstrap_term_present():
    targets = compute_td_targets(
        shaped_rewards=[0.0], controller_terminal=[False], bootstrap_indices=[0], next_values=[1.0], gamma=GAMMA,
    )
    assert targets[0] == pytest.approx(GAMMA * 1.0)
    assert targets[0] != pytest.approx(0.0)  # bootstrap term must have actually been added


def test_mixed_batch_terminal_and_bootstrap_rows_independent():
    """Terminal and non-terminal rows in the SAME batch must not interfere --
    exercises the real shape compute_td_targets is called with (only
    bootstrap-index rows appear in next_values, in that index's own order)."""
    targets = compute_td_targets(
        shaped_rewards=[0.6, -1.0, 0.2, -0.5],
        controller_terminal=[True, True, False, True],
        bootstrap_indices=[2],
        next_values=[0.7],
        gamma=GAMMA,
    )
    np.testing.assert_allclose(targets, [0.6, -1.0, 0.2 + GAMMA * 0.7, -0.5])


def test_bootstrap_row_marked_controller_terminal_raises():
    """Caller-bug guard: a row listed in bootstrap_indices must never also
    be controller_terminal -- this would mean building the wrong branch
    for that row upstream."""
    with pytest.raises(ValueError):
        compute_td_targets(
            shaped_rewards=[0.2], controller_terminal=[True], bootstrap_indices=[0], next_values=[0.7], gamma=GAMMA,
        )


# ------------------------------------------------------------- compute_bootstrap_values

class _FixedQNetwork(nn.Module):
    """Always returns the same fixed Q-values regardless of input --
    lets compute_bootstrap_values be hand-verified against the document's
    exact 'next_Q=[0.1,0.7,0.4]'-style examples without training anything."""

    def __init__(self, fixed_q: list[float]):
        super().__init__()
        self._fixed_q = torch.tensor([fixed_q], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fixed_q.expand(x.shape[0], -1)


def test_compute_bootstrap_values_vanilla_takes_max_next_q():
    online = _FixedQNetwork([0.1, 0.7, 0.4])  # unused in VANILLA mode
    target = _FixedQNetwork([0.1, 0.7, 0.4])
    obs = torch.zeros((1, 4), dtype=torch.float32)
    mask = torch.tensor([[True, True, True]])
    values = compute_bootstrap_values(
        online_network=online, target_network=target, next_observations=obs,
        next_action_masks=mask, mode=DQNTargetMode.VANILLA,
    )
    assert values.item() == pytest.approx(0.7)


def test_compute_bootstrap_values_double_uses_online_argmax_target_value():
    """Double DQN: action SELECTED by online's argmax, but VALUE taken from
    target network at that action -- these differ here to make the
    distinction observable (online picks action 1, target's value at
    action 1 is deliberately different from target's own max)."""
    online = _FixedQNetwork([0.0, 5.0, 1.0])  # argmax -> action 1
    target = _FixedQNetwork([9.0, 2.0, 0.5])  # target's OWN max is action 0 (9.0), not action 1
    obs = torch.zeros((1, 4), dtype=torch.float32)
    mask = torch.tensor([[True, True, True]])
    values = compute_bootstrap_values(
        online_network=online, target_network=target, next_observations=obs,
        next_action_masks=mask, mode=DQNTargetMode.DOUBLE,
    )
    # Must be target's value AT action 1 (2.0), NOT target's own max (9.0).
    assert values.item() == pytest.approx(2.0)


def test_compute_bootstrap_values_respects_action_mask():
    online = _FixedQNetwork([10.0, 0.1, 0.2])  # argmax would be action 0 if legal
    target = _FixedQNetwork([10.0, 0.1, 0.2])
    obs = torch.zeros((1, 4), dtype=torch.float32)
    mask = torch.tensor([[False, True, True]])  # action 0 illegal
    values = compute_bootstrap_values(
        online_network=online, target_network=target, next_observations=obs,
        next_action_masks=mask, mode=DQNTargetMode.DOUBLE,
    )
    # Must pick among legal actions {1, 2} only -> action 2 (0.2) is the online argmax there.
    assert values.item() == pytest.approx(0.2)
