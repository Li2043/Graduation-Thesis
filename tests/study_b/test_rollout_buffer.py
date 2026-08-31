from __future__ import annotations

import numpy as np
import pytest

from thesis.study_b.rollout_buffer import RolloutBuffer


def _fill(buffer: RolloutBuffer, rewards, values, dones):
    for r, v, d in zip(rewards, values, dones):
        buffer.add(
            obs={"V0": np.zeros(1)},
            global_state=np.zeros(1),
            actions={"V0": 0},
            log_probs={"V0": 0.0},
            team_reward=r,
            value=v,
            done=d,
        )


def test_gae_reduces_to_monte_carlo_return_when_gamma_lambda_are_one():
    buffer = RolloutBuffer(agent_ids=("V0",))
    _fill(buffer, rewards=[1.0, 1.0, 1.0], values=[0.0, 0.0, 0.0], dones=[False, False, True])
    advantages, returns = buffer.compute_gae(last_value=0.0, gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(returns, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(advantages, [3.0, 2.0, 1.0])


def test_gae_single_step_matches_td_residual():
    buffer = RolloutBuffer(agent_ids=("V0",))
    _fill(buffer, rewards=[1.0], values=[0.5], dones=[False])
    advantages, returns = buffer.compute_gae(last_value=2.0, gamma=0.9, gae_lambda=0.95)
    expected_delta = 1.0 + 0.9 * 2.0 - 0.5
    assert advantages[0] == pytest.approx(expected_delta)
    assert returns[0] == pytest.approx(expected_delta + 0.5)


def test_gae_zeroes_bootstrap_across_done_boundary():
    buffer = RolloutBuffer(agent_ids=("V0",))
    _fill(buffer, rewards=[1.0, 5.0], values=[0.0, 0.0], dones=[True, False])
    advantages, returns = buffer.compute_gae(last_value=100.0, gamma=0.9, gae_lambda=0.9)
    # First transition is terminal -> its delta must not bootstrap through
    # next_value at all (mask=0), regardless of the large last_value passed.
    assert advantages[0] == pytest.approx(1.0)


def test_clear_empties_buffer():
    buffer = RolloutBuffer(agent_ids=("V0",))
    _fill(buffer, rewards=[1.0], values=[0.0], dones=[False])
    assert len(buffer) == 1
    buffer.clear()
    assert len(buffer) == 0
