"""Stage 10 pilot (E28) pre-training audit -- routing, independence, and
cross-network bootstrap correctness (protocol S8 items 1, 2, 4).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from thesis.agents.dqn_bootstrap import DQNTargetMode, compute_bootstrap_values
from thesis.agents.independent_dqn_v2 import DQNConfig
from thesis.agents.replay_buffer_v2 import ReplayBatch, ReplayTransition
from thesis.agents.subpolicy_dqn import (
    ALL_SUBPOLICY_KEYS,
    SubPolicyManager,
    make_bootstrap_transition,
    subpolicy_key,
)


def _cfg(**overrides) -> DQNConfig:
    base = dict(
        obs_dim=3,
        n_actions=3,
        hidden_sizes=(8, 8),
        learning_rate=1e-3,
        gamma=0.5,
        epsilon=0.1,
        replay_capacity=100,
        batch_size=4,
        device="cpu",
        target_mode=DQNTargetMode.DOUBLE,
    )
    base.update(overrides)
    return DQNConfig(**base)


# --------------------------------------------------------------------- S8.1
def test_subpolicy_key_covers_all_six_role_zone_combinations():
    assert len(ALL_SUBPOLICY_KEYS) == 6
    assert len(set(ALL_SUBPOLICY_KEYS)) == 6
    assert subpolicy_key("ramp", "pre") == "ramp_pre"
    assert subpolicy_key("mainline", "merging") == "mainline_merging"
    assert subpolicy_key("mainline", "post") == "mainline_post"
    with pytest.raises(ValueError):
        subpolicy_key("ramp", "not_a_zone")
    with pytest.raises(ValueError):
        subpolicy_key("not_a_role", "pre")


def test_manager_routes_to_correct_distinct_learner_per_role_zone():
    manager = SubPolicyManager(_cfg(), seed=0)
    seen_ids = set()
    for role in ("ramp", "mainline"):
        for zone in ("pre", "merging", "post"):
            learner = manager.learner_for(role, zone)
            assert learner.policy_id == subpolicy_key(role, zone)
            assert id(learner) not in seen_ids
            seen_ids.add(id(learner))
    assert len(seen_ids) == 6
    # boundary-value routing consistency, mirrored from the env's own zone_for_position
    assert manager.route("ramp", "merging") == "ramp_merging"


# --------------------------------------------------------------------- S8.4
def test_six_learners_share_no_weights_or_buffers():
    manager = SubPolicyManager(_cfg(), seed=0)
    learners = list(manager.learners.values())
    for i, a in enumerate(learners):
        for b in learners[i + 1 :]:
            assert a.replay is not b.replay
            assert a.online is not b.online
            assert a.target is not b.target
            assert a.optimiser is not b.optimiser

    l0 = manager.learners["ramp_pre"]
    l1 = manager.learners["ramp_merging"]
    t = ReplayTransition(
        observation=np.zeros(3),
        action=0,
        shaped_reward=1.0,
        next_observation=None,
        terminated=True,
        truncated=False,
        action_mask=np.array([True, True, True]),
        next_action_mask=None,
        controller_terminal=True,
    )
    l0.store_transition(t)
    assert len(l0.replay) == 1
    assert len(l1.replay) == 0  # mutating one buffer must not affect siblings


# --------------------------------------------------------------------- S8.2
def test_cross_network_bootstrap_uses_incoming_policy_not_acting_policy():
    """A transition handed off at a zone boundary must bootstrap through the
    INCOMING sub-policy's own online/target networks, not the acting
    ("outgoing") sub-policy's -- protocol S2.4 / S8 item 2."""
    manager = SubPolicyManager(_cfg(gamma=0.5), seed=123)
    outgoing = manager.learners["ramp_pre"]
    incoming = manager.learners["ramp_merging"]

    probe_next_obs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mask = np.array([True, True, True])

    # Precondition: independently-seeded networks must actually differ on this
    # input, otherwise the test can't discriminate outgoing- vs incoming-network use.
    q_out = outgoing.q_values(probe_next_obs, network="target")
    q_in = incoming.q_values(probe_next_obs, network="target")
    assert not np.allclose(q_out, q_in), "precondition failed: networks coincide by chance"

    reward = 2.0
    transition = make_bootstrap_transition(
        observation=np.array([0.1, 0.2, 0.3]),
        action=1,
        shaped_reward=reward,
        next_observation=probe_next_obs,
        terminated=False,
        truncated=False,
        action_mask=mask,
        next_action_mask=mask,
        controller_terminal=False,
        bootstrap_policy_id="ramp_merging",  # crosses into the incoming sub-policy
    )
    transition.validate(n_actions=3, obs_dim=3)

    batch = ReplayBatch(
        observations=np.stack([transition.observation]),
        actions=np.array([transition.action]),
        shaped_rewards=np.array([transition.shaped_reward]),
        next_observations=[transition.next_observation],
        terminated=np.array([transition.terminated]),
        truncated=np.array([transition.truncated]),
        controller_terminal=np.array([transition.controller_terminal]),
        learner_completed=np.array([transition.learner_completed]),
        action_masks=np.stack([transition.action_mask]),
        next_action_masks=[transition.next_action_mask],
        base_rewards=np.array([0.0]),
        shaping_components=np.array([0.0]),
        reward_conditions=["baseline"],
        indices=np.array([0]),
        transitions=[transition],
    )

    others = {k: l for k, l in manager.learners.items() if k != "ramp_pre"}
    targets_arr, n_cross = outgoing.compute_targets(batch, others)
    assert n_cross == 1

    next_obs_t = torch.as_tensor(probe_next_obs, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    expected_from_incoming = compute_bootstrap_values(
        online_network=incoming.online,
        target_network=incoming.target,
        next_observations=next_obs_t,
        next_action_masks=mask_t,
        mode=DQNTargetMode.DOUBLE,
    ).item()
    expected_from_outgoing = compute_bootstrap_values(
        online_network=outgoing.online,
        target_network=outgoing.target,
        next_observations=next_obs_t,
        next_action_masks=mask_t,
        mode=DQNTargetMode.DOUBLE,
    ).item()
    assert not np.isclose(expected_from_incoming, expected_from_outgoing), (
        "precondition failed: incoming/outgoing bootstrap values coincide by chance"
    )

    assert targets_arr[0] == pytest.approx(reward + 0.5 * expected_from_incoming)
    assert targets_arr[0] != pytest.approx(reward + 0.5 * expected_from_outgoing)


def test_same_zone_transition_bootstraps_from_self_not_others():
    """The common case (no zone crossing): bootstrap_policy_id='' must use the
    acting learner's own networks, and must NOT be counted as cross-network."""
    manager = SubPolicyManager(_cfg(gamma=0.5), seed=7)
    learner = manager.learners["mainline_post"]
    probe_next_obs = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    mask = np.array([True, True, True])
    reward = -1.0
    transition = make_bootstrap_transition(
        observation=np.array([0.4, 0.5, 0.6]),
        action=2,
        shaped_reward=reward,
        next_observation=probe_next_obs,
        terminated=False,
        truncated=False,
        action_mask=mask,
        next_action_mask=mask,
        controller_terminal=False,
        bootstrap_policy_id="",  # no crossing -- same sub-policy
    )
    transition.validate(n_actions=3, obs_dim=3)
    batch = ReplayBatch(
        observations=np.stack([transition.observation]),
        actions=np.array([transition.action]),
        shaped_rewards=np.array([transition.shaped_reward]),
        next_observations=[transition.next_observation],
        terminated=np.array([transition.terminated]),
        truncated=np.array([transition.truncated]),
        controller_terminal=np.array([transition.controller_terminal]),
        learner_completed=np.array([transition.learner_completed]),
        action_masks=np.stack([transition.action_mask]),
        next_action_masks=[transition.next_action_mask],
        base_rewards=np.array([0.0]),
        shaping_components=np.array([0.0]),
        reward_conditions=["baseline"],
        indices=np.array([0]),
        transitions=[transition],
    )
    others = {k: l for k, l in manager.learners.items() if k != "mainline_post"}
    targets_arr, n_cross = learner.compute_targets(batch, others)
    assert n_cross == 0
    next_obs_t = torch.as_tensor(probe_next_obs, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    expected = compute_bootstrap_values(
        online_network=learner.online,
        target_network=learner.target,
        next_observations=next_obs_t,
        next_action_masks=mask_t,
        mode=DQNTargetMode.DOUBLE,
    ).item()
    assert targets_arr[0] == pytest.approx(reward + 0.5 * expected)


def test_terminal_transition_target_is_reward_only_no_bootstrap_lookup():
    manager = SubPolicyManager(_cfg(), seed=0)
    learner = manager.learners["ramp_post"]
    mask = np.array([True, True, True])
    transition = make_bootstrap_transition(
        observation=np.array([0.1, 0.1, 0.1]),
        action=0,
        shaped_reward=5.0,
        next_observation=None,
        terminated=True,
        truncated=False,
        action_mask=mask,
        next_action_mask=None,
        controller_terminal=True,
        bootstrap_policy_id="",
        learner_completed=True,
    )
    transition.validate(n_actions=3, obs_dim=3)
    batch = ReplayBatch(
        observations=np.stack([transition.observation]),
        actions=np.array([transition.action]),
        shaped_rewards=np.array([transition.shaped_reward]),
        next_observations=[None],
        terminated=np.array([True]),
        truncated=np.array([False]),
        controller_terminal=np.array([True]),
        learner_completed=np.array([True]),
        action_masks=np.stack([transition.action_mask]),
        next_action_masks=[None],
        base_rewards=np.array([0.0]),
        shaping_components=np.array([0.0]),
        reward_conditions=["baseline"],
        indices=np.array([0]),
        transitions=[transition],
    )
    targets_arr, n_cross = learner.compute_targets(batch, {})
    assert n_cross == 0
    assert targets_arr[0] == pytest.approx(5.0)


def test_update_runs_end_to_end_with_cross_network_row():
    """Smoke test: a real update() call (gradient step) succeeds with a
    cross-network row present, only the acting learner's parameters change."""
    manager = SubPolicyManager(_cfg(batch_size=1, gamma=0.9), seed=42)
    outgoing = manager.learners["mainline_pre"]
    incoming = manager.learners["mainline_merging"]
    mask = np.array([True, True, True])
    transition = make_bootstrap_transition(
        observation=np.array([0.2, 0.3, 0.4]),
        action=0,
        shaped_reward=1.0,
        next_observation=np.array([0.5, 0.5, 0.5]),
        terminated=False,
        truncated=False,
        action_mask=mask,
        next_action_mask=mask,
        controller_terminal=False,
        bootstrap_policy_id="mainline_merging",
    )
    outgoing.store_transition(transition)
    others = {k: l for k, l in manager.learners.items() if k != "mainline_pre"}
    before = torch.nn.utils.parameters_to_vector(incoming.online.parameters()).clone()
    stats = outgoing.update(others)
    after = torch.nn.utils.parameters_to_vector(incoming.online.parameters())
    assert torch.equal(before, after), "incoming learner's params must not change from outgoing's update()"
    assert stats["n_cross_network_rows"] == 1
    assert stats["n_bootstrap_rows"] == 1
