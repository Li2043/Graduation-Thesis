"""Diagnostic_6_DQN_Pipeline_Verification_Protocol.md sec 10 (6F, Test T8):
replay-buffer round-trip semantic-identity test. Inserts several
transitions with deliberately distinct episode/time/action/reward
values into a REAL ``ReplayBuffer`` and verifies every field stays
correctly paired after a full-buffer sample -- not just that shapes
match."""

from __future__ import annotations

import numpy as np

from thesis.agents.replay_buffer_v2 import ReplayBuffer, ReplayTransition

OBS_DIM = 4
N_ACTIONS = 3
ALL_LEGAL = np.array([True, True, True], dtype=bool)


def _make_transition(*, episode_id: str, step: int, action: int, reward: float, marker: float) -> ReplayTransition:
    """``marker`` is baked into every array field (observation, next_observation)
    at a unique, recognisable value so mismatched pairing is easy to detect."""
    return ReplayTransition(
        observation=np.array([marker, marker + 0.01, marker + 0.02, marker + 0.03]),
        action=action,
        shaped_reward=reward,
        next_observation=np.array([marker + 1.0, marker + 1.01, marker + 1.02, marker + 1.03]),
        terminated=False,
        truncated=False,
        action_mask=ALL_LEGAL,
        next_action_mask=ALL_LEGAL,
        controller_terminal=False,
        learner_completed=False,
        episode_id=episode_id,
        step=step,
    )


def test_replay_roundtrip_preserves_every_field_pairing():
    buf = ReplayBuffer(capacity=10, obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=0)
    specs = [
        {"episode_id": "ep7", "step": 31, "action": 2, "reward": 0.371, "marker": 31.0},
        {"episode_id": "ep7", "step": 32, "action": 0, "reward": -0.5, "marker": 32.0},
        {"episode_id": "ep8", "step": 0, "action": 1, "reward": 0.9, "marker": 100.0},
        {"episode_id": "ep8", "step": 1, "action": 2, "reward": -1.0, "marker": 101.0},
        {"episode_id": "ep9", "step": 5, "action": 1, "reward": 0.05, "marker": 500.0},
    ]
    for spec in specs:
        buf.append(_make_transition(**spec))

    batch = buf.sample(len(specs))  # full-buffer sample -- every transition returned exactly once

    by_marker = {spec["marker"]: spec for spec in specs}
    for row_idx in range(len(specs)):
        marker = float(batch.observations[row_idx][0])
        assert marker in by_marker, f"observation marker {marker} does not match any inserted transition"
        spec = by_marker.pop(marker)

        # obs <-> action
        assert int(batch.actions[row_idx]) == spec["action"]
        # obs <-> reward
        assert float(batch.shaped_rewards[row_idx]) == spec["reward"]
        # obs <-> next_obs (next_obs marker must be exactly marker+1.0, not some other row's)
        next_marker = float(batch.next_observations[row_idx][0])
        assert next_marker == spec["marker"] + 1.0
        # obs <-> episode_id / step (via the original transition object retained on the batch)
        transition = batch.transitions[row_idx]
        assert transition.episode_id == spec["episode_id"]
        assert transition.step == spec["step"]

    assert len(by_marker) == 0, f"some transitions were never returned: {by_marker}"


def test_replay_roundtrip_across_many_samples_never_cross_pairs():
    """Repeated sampling (with the buffer larger than any single sample)
    must never mix one transition's observation with another's action/
    reward/next_observation -- exercised across many draws to catch a
    rare indexing-order bug that a single sample might not surface."""
    buf = ReplayBuffer(capacity=50, obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=0)
    specs = [
        {"episode_id": f"ep{i}", "step": i, "action": i % 3, "reward": float(i) * 0.1, "marker": float(i) * 10.0}
        for i in range(20)
    ]
    for spec in specs:
        buf.append(_make_transition(**spec))

    by_marker = {spec["marker"]: spec for spec in specs}
    for _ in range(30):
        batch = buf.sample(8)
        for row_idx in range(8):
            marker = float(batch.observations[row_idx][0])
            spec = by_marker[marker]
            assert int(batch.actions[row_idx]) == spec["action"]
            assert float(batch.shaped_rewards[row_idx]) == spec["reward"]
            assert float(batch.next_observations[row_idx][0]) == spec["marker"] + 1.0
            assert batch.transitions[row_idx].episode_id == spec["episode_id"]
            assert batch.transitions[row_idx].step == spec["step"]


def test_replay_roundtrip_terminal_rows_have_no_next_observation():
    buf = ReplayBuffer(capacity=5, obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=0)
    terminal_transition = ReplayTransition(
        observation=np.array([1.0, 1.0, 1.0, 1.0]),
        action=1,
        shaped_reward=-1.0,
        next_observation=None,
        terminated=True,
        truncated=False,
        action_mask=ALL_LEGAL,
        next_action_mask=None,
        controller_terminal=True,
        learner_completed=False,
    )
    buf.append(terminal_transition)
    batch = buf.sample(1)
    assert batch.next_observations[0] is None
    assert bool(batch.controller_terminal[0]) is True
    assert 0 in batch.bootstrap_indices.tolist() or len(batch.bootstrap_indices) == 0
    assert 0 not in batch.bootstrap_indices.tolist()  # terminal row must NOT be a bootstrap row
