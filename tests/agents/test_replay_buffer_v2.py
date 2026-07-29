"""Replay buffer schema and capacity tests."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.agents.replay_buffer_v2 import ReplayBuffer, ReplayTransition


def _tr(**kw) -> ReplayTransition:
    base = dict(
        observation=np.zeros(4),
        action=0,
        shaped_reward=0.1,
        next_observation=np.ones(4),
        terminated=False,
        truncated=False,
        action_mask=np.array([True, True, True]),
        next_action_mask=np.array([True, True, True]),
        base_reward=0.1,
        shaping_component=0.0,
        reward_condition="baseline",
        episode_id="e0",
        step=1,
        controller_id="A",
        traffic_role="mainline",
    )
    base.update(kw)
    return ReplayTransition(**base)


def test_07_illegal_stored_action():
    buf = ReplayBuffer(8, obs_dim=4, n_actions=3, seed=0)
    with pytest.raises(ValueError, match="illegal stored action"):
        buf.append(
            _tr(action=1, action_mask=np.array([True, False, True]))
        )


def test_08_replay_schema_retains_separate_flags():
    buf = ReplayBuffer(8, obs_dim=4, n_actions=3, seed=0)
    buf.append(_tr(terminated=True, truncated=False, shaped_reward=1.0))
    buf.append(_tr(terminated=False, truncated=True, shaped_reward=0.5, step=2))
    payload = buf.serialize()
    buf2 = ReplayBuffer.deserialize(payload)
    batch = buf2.sample(2)
    # Flags remain distinct
    assert set(zip(batch.terminated.tolist(), batch.truncated.tolist())) == {
        (True, False),
        (False, True),
    }
    assert not any(t and tr for t, tr in zip(batch.terminated, batch.truncated))


def test_09_invalid_simultaneous_flags():
    buf = ReplayBuffer(4, obs_dim=4, n_actions=3, seed=0)
    with pytest.raises(ValueError, match="simultaneously"):
        buf.append(_tr(terminated=True, truncated=True))


def test_23_replay_sampling_reproducibility():
    def fill(seed):
        b = ReplayBuffer(20, obs_dim=4, n_actions=3, seed=seed)
        for i in range(10):
            b.append(_tr(step=i, shaped_reward=float(i)))
        return b

    b1 = fill(123)
    b2 = fill(123)
    s1 = b1.sample(4).indices.tolist()
    s2 = b2.sample(4).indices.tolist()
    assert s1 == s2


def test_24_replay_capacity_circular_fifo():
    """Circular FIFO: when full, oldest entries are overwritten."""
    buf = ReplayBuffer(3, obs_dim=4, n_actions=3, seed=0)
    for i in range(5):
        buf.append(_tr(step=i, shaped_reward=float(i)))
    assert len(buf) == 3
    # Remaining should be steps 2,3,4 (oldest overwritten)
    payload = buf.serialize()
    steps = [t["step"] for t in __import__("json").loads(payload)["transitions"]]
    assert steps == [2, 3, 4]


def test_25_non_finite_obs_rejected():
    buf = ReplayBuffer(4, obs_dim=4, n_actions=3, seed=0)
    with pytest.raises(ValueError):
        buf.append(_tr(observation=np.array([1.0, np.nan, 0.0, 0.0])))
    with pytest.raises(ValueError):
        buf.append(_tr(shaped_reward=float("inf")))


def test_26_observation_dimension_validation():
    buf = ReplayBuffer(4, obs_dim=4, n_actions=3, seed=0)
    with pytest.raises(ValueError, match="dim"):
        buf.append(_tr(observation=np.zeros(3)))
