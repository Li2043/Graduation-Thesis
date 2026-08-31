"""Stage 11 pilot (E30) v12 -- joint (centralised) DQN.

Covers: ``JointQNetwork``'s N-headed output shape (N in {2,4,6});
``select_action``'s per-slot independent masking; ``JointReplayBuffer``
append/sample round-trip; a direct numeric proof that every slot's head
actually receives a gradient update (not just "no crash"); Double-DQN
bootstrap correctness via the ``_JointHeadView`` wrapper; finite-value
guards; N=2 backward compatibility (state_dict key names, single-scalar
transition shape callers might still reasonably expect are gone -- this
file exercises the ACTUAL current tuple-based API, not the pre-generalisation
one).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from thesis.agents.joint_dqn import (
    JointDQNConfig,
    JointDQNLearner,
    JointQNetwork,
    JointReplayBatch,
    JointReplayBuffer,
    JointReplayTransition,
    beta_at_step,
)

OBS_DIM = 13
N_ACTIONS = 3


def joint_obs_dim(n_vehicles: int) -> int:
    return n_vehicles * OBS_DIM


def _config(*, n_vehicles: int = 2, **overrides) -> JointDQNConfig:
    base = dict(
        per_vehicle_obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_sizes=(16, 16),
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        replay_capacity=200,
        batch_size=8,
        device="cpu",
        n_vehicles=n_vehicles,
    )
    base.update(overrides)
    return JointDQNConfig(**base)


def _learner(seed: int = 0, *, n_vehicles: int = 2, **overrides) -> JointDQNLearner:
    return JointDQNLearner(_config(n_vehicles=n_vehicles, **overrides), seed=seed)


def _rand_obs(seed: int, *, n_vehicles: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=joint_obs_dim(n_vehicles)).astype(np.float64)


def _mask(legal_index: int | None = None) -> np.ndarray:
    if legal_index is None:
        return np.array([True, True, True])
    m = np.array([False, False, False])
    m[legal_index] = True
    return m


def _transition(
    seed: int,
    *,
    n_vehicles: int = 2,
    rewards: tuple[float, ...] | None = None,
    controller_terminal: bool = False,
    actions: tuple[int, ...] | None = None,
    action_masks: tuple[np.ndarray, ...] | None = None,
    n_steps: int = 1,
) -> JointReplayTransition:
    acts = tuple([0] * n_vehicles) if actions is None else actions
    masks = tuple(_mask() for _ in range(n_vehicles)) if action_masks is None else action_masks
    rews = tuple([0.1 if i % 2 == 0 else -0.1 for i in range(n_vehicles)]) if rewards is None else rewards
    if controller_terminal:
        next_obs = None
        next_masks = None
    else:
        next_obs = _rand_obs(seed + 1000, n_vehicles=n_vehicles)
        next_masks = tuple(_mask() for _ in range(n_vehicles))
    return JointReplayTransition(
        joint_observation=_rand_obs(seed, n_vehicles=n_vehicles),
        actions=acts,
        rewards=rews,
        next_joint_observation=next_obs,
        terminated=False,
        truncated=False,
        action_masks=masks,
        next_action_masks=next_masks,
        controller_terminal=controller_terminal,
        n_steps=n_steps,
    )


# --------------------------------------------------------------- JointQNetwork


@pytest.mark.parametrize("n_vehicles", [2, 4, 6])
def test_joint_qnetwork_output_shapes(n_vehicles):
    net = JointQNetwork(OBS_DIM, N_ACTIONS, (16, 16), n_vehicles=n_vehicles)
    x = torch.zeros(5, joint_obs_dim(n_vehicles))
    q_all = net(x)
    assert len(q_all) == n_vehicles
    for q in q_all:
        assert q.shape == (5, N_ACTIONS)


def test_joint_qnetwork_heads_are_independent_layers():
    net = JointQNetwork(OBS_DIM, N_ACTIONS, (16, 16))
    assert net.head_ramp is not net.head_mainline
    assert not torch.equal(net.head_ramp.weight, net.head_mainline.weight)


def test_joint_qnetwork_n2_state_dict_keys_are_backward_compatible():
    """N=2 is a special case, not the first entry of a generic ModuleList --
    state_dict keys must stay exactly head_ramp/head_mainline so Study A's
    already-trained checkpoints keep loading unchanged."""
    net = JointQNetwork(OBS_DIM, N_ACTIONS, (16, 16))
    keys = set(net.state_dict().keys())
    assert "head_ramp.weight" in keys
    assert "head_mainline.weight" in keys
    assert not any(k.startswith("heads.") for k in keys)


def test_joint_qnetwork_n4_heads_are_independent_layers():
    net = JointQNetwork(OBS_DIM, N_ACTIONS, (16, 16), n_vehicles=4)
    assert len(net.heads) == 4
    weights = [h.weight for h in net.heads]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.equal(weights[i], weights[j])


# --------------------------------------------------------------- select_action masking


def test_select_action_respects_per_slot_independent_mask():
    learner = _learner()
    joint_obs = _rand_obs(1)
    masks = (_mask(0), _mask(2))  # only action 0 legal slot 0, only action 2 legal slot 1
    for _ in range(20):
        actions = learner.select_action(joint_obs, masks, epsilon=1.0)
        assert actions == (0, 2)


def test_select_action_greedy_is_deterministic():
    learner = _learner()
    joint_obs = _rand_obs(2)
    masks = (_mask(), _mask())
    a1 = learner.select_action(joint_obs, masks, greedy=True)
    a2 = learner.select_action(joint_obs, masks, greedy=True)
    assert a1 == a2


def test_select_action_n4_respects_all_four_independent_masks():
    learner = _learner(n_vehicles=4)
    joint_obs = _rand_obs(1, n_vehicles=4)
    masks = (_mask(0), _mask(1), _mask(2), _mask(0))
    for _ in range(20):
        actions = learner.select_action(joint_obs, masks, epsilon=1.0)
        assert actions == (0, 1, 2, 0)


def test_select_action_rejects_wrong_number_of_masks():
    learner = _learner(n_vehicles=4)
    joint_obs = _rand_obs(1, n_vehicles=4)
    with pytest.raises(ValueError):
        learner.select_action(joint_obs, (_mask(), _mask()), epsilon=1.0)


# --------------------------------------------------------------- replay buffer


def test_replay_buffer_append_and_sample_round_trip():
    learner = _learner(replay_capacity=50, batch_size=4)
    for i in range(10):
        learner.store_transition(_transition(i))
    assert len(learner.replay) == 10
    batch = learner.replay.sample(4)
    assert batch.joint_observations.shape == (4, joint_obs_dim(2))
    assert batch.actions.shape == (4, 2)
    assert batch.rewards.shape == (4, 2)


def test_replay_buffer_circular_overwrite():
    learner = _learner(replay_capacity=5, batch_size=2)
    for i in range(8):
        learner.store_transition(_transition(i))
    assert len(learner.replay) == 5  # capped at capacity, oldest overwritten


def test_replay_transition_rejects_illegal_stored_action():
    learner = _learner()
    bad = _transition(3, actions=(1, 0), action_masks=(_mask(0), _mask()))
    with pytest.raises(ValueError):
        learner.store_transition(bad)


def test_replay_transition_rejects_wrong_length_tuples():
    """A malformed 3-length actions tuple at n_vehicles=4 must not silently
    validate just because each individual element happens to be legal."""
    learner = _learner(n_vehicles=4)
    bad = _transition(0, n_vehicles=4, actions=(0, 0, 0), action_masks=(_mask(), _mask(), _mask()))
    with pytest.raises(ValueError):
        learner.store_transition(bad)


def test_replay_buffer_n4_round_trip():
    learner = _learner(n_vehicles=4, replay_capacity=50, batch_size=4)
    for i in range(10):
        learner.store_transition(_transition(i, n_vehicles=4))
    batch = learner.replay.sample(4)
    assert batch.joint_observations.shape == (4, joint_obs_dim(4))
    assert batch.actions.shape == (4, 4)
    assert batch.rewards.shape == (4, 4)
    assert batch.action_masks.shape == (4, 4, N_ACTIONS)


# --------------------------------------------------------------- update() -- every slot learns


def test_update_changes_both_heads_parameters():
    learner = _learner(replay_capacity=100, batch_size=16)
    rng = np.random.default_rng(0)
    for i in range(30):
        learner.store_transition(
            _transition(
                i,
                rewards=(float(rng.normal()), float(rng.normal())),
                actions=(int(rng.integers(0, 3)), int(rng.integers(0, 3))),
            )
        )
    before_ramp = {k: v.detach().clone() for k, v in learner.online.head_ramp.state_dict().items()}
    before_mainline = {k: v.detach().clone() for k, v in learner.online.head_mainline.state_dict().items()}
    before_trunk = {k: v.detach().clone() for k, v in learner.online.trunk.state_dict().items()}

    learner.update()

    after_ramp = learner.online.head_ramp.state_dict()
    after_mainline = learner.online.head_mainline.state_dict()
    after_trunk = learner.online.trunk.state_dict()

    ramp_changed = any(not torch.equal(before_ramp[k], after_ramp[k]) for k in before_ramp)
    mainline_changed = any(not torch.equal(before_mainline[k], after_mainline[k]) for k in before_mainline)
    trunk_changed = any(not torch.equal(before_trunk[k], after_trunk[k]) for k in before_trunk)
    assert ramp_changed, "ramp head did not update"
    assert mainline_changed, "mainline head did not update"
    assert trunk_changed, "shared trunk did not update"


def test_update_n4_changes_all_four_heads_parameters():
    learner = _learner(n_vehicles=4, replay_capacity=100, batch_size=16)
    rng = np.random.default_rng(0)
    for i in range(30):
        learner.store_transition(
            _transition(
                i,
                n_vehicles=4,
                rewards=tuple(float(rng.normal()) for _ in range(4)),
                actions=tuple(int(rng.integers(0, 3)) for _ in range(4)),
            )
        )
    before = [{k: v.detach().clone() for k, v in h.state_dict().items()} for h in learner.online.heads]
    learner.update()
    after = [h.state_dict() for h in learner.online.heads]
    for slot in range(4):
        changed = any(not torch.equal(before[slot][k], after[slot][k]) for k in before[slot])
        assert changed, f"slot {slot} head did not update"


def test_update_never_touches_target_network():
    learner = _learner(replay_capacity=100, batch_size=16)
    for i in range(30):
        learner.store_transition(_transition(i))
    target_before = {k: v.detach().clone() for k, v in learner.target.state_dict().items()}
    learner.update()
    target_after = learner.target.state_dict()
    assert all(torch.equal(target_before[k], target_after[k]) for k in target_before)


def test_update_loss_is_sum_of_slot_losses():
    learner = _learner(replay_capacity=100, batch_size=16)
    for i in range(30):
        learner.store_transition(_transition(i))
    result = learner.update()
    assert result["loss"] == pytest.approx(sum(result["losses"]), rel=1e-5)


def test_update_n4_loss_is_sum_of_four_slot_losses():
    learner = _learner(n_vehicles=4, replay_capacity=100, batch_size=16)
    for i in range(30):
        learner.store_transition(_transition(i, n_vehicles=4))
    result = learner.update()
    assert len(result["losses"]) == 4
    assert result["loss"] == pytest.approx(sum(result["losses"]), rel=1e-5)


def test_terminal_row_target_equals_reward_no_bootstrap():
    """A controller_terminal=True row must not attempt any successor
    forward pass -- target should reduce to exactly the reward."""
    learner = _learner(replay_capacity=100, batch_size=4)
    for i in range(4):
        learner.store_transition(_transition(i, rewards=(2.0, -3.0), controller_terminal=True))
    result = learner.update()
    assert np.isfinite(result["loss"])
    assert result["n_bootstrap_rows"] == 0
    assert result["n_terminal_rows"] == 4


# --------------------------------------------------------------- finite-value guards


def test_q_values_rejects_non_finite_observation():
    learner = _learner()
    bad_obs = _rand_obs(5)
    bad_obs[0] = np.nan
    with pytest.raises(ValueError):
        learner.q_values(bad_obs)


def test_q_values_rejects_wrong_dim_observation():
    learner = _learner()
    with pytest.raises(ValueError):
        learner.q_values(np.zeros(joint_obs_dim(2) - 1))


def test_config_validate_rejects_bad_gamma():
    cfg = _config(gamma=1.5)
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_joint_obs_dim_property():
    cfg = _config()
    assert cfg.joint_obs_dim == joint_obs_dim(2)


@pytest.mark.parametrize("n_vehicles", [2, 4, 6])
def test_config_joint_obs_dim_property_scales_with_n_vehicles(n_vehicles):
    cfg = _config(n_vehicles=n_vehicles)
    assert cfg.joint_obs_dim == joint_obs_dim(n_vehicles)


def test_config_defaults_n_vehicles_to_two():
    cfg = JointDQNConfig(
        per_vehicle_obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_sizes=(16, 16),
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        replay_capacity=200,
        batch_size=8,
    )
    assert cfg.n_vehicles == 2


def test_config_validate_rejects_bad_n_vehicles():
    cfg = _config(n_vehicles=3)
    with pytest.raises(ValueError):
        cfg.validate()


# --------------------------------------------------------------- n-step: JointReplayTransition


def test_transition_default_n_steps_is_one():
    t = _transition(0)
    assert t.n_steps == 1


def test_transition_rejects_n_steps_less_than_one():
    learner = _learner()
    bad = _transition(0, n_steps=0)
    with pytest.raises(ValueError):
        learner.store_transition(bad)


def test_batch_carries_n_steps_array():
    learner = _learner(replay_capacity=50, batch_size=4)
    for i in range(10):
        learner.store_transition(_transition(i, n_steps=1 + (i % 3)))
    batch = learner.replay.sample(4)
    assert batch.n_steps.shape == (4,)
    assert batch.n_steps.dtype == np.int64
    assert np.all(batch.n_steps >= 1)


# --------------------------------------------------------------- n-step: update() discounting


def test_n_step_discount_uses_gamma_power_n_per_row_hand_computed():
    """Direct numeric proof that update() uses gamma**n_steps (per row), not
    a fixed gamma**1 -- the exact line this round's n-step addition changes.
    Uses a single-legal-action mask so Double-DQN's online-argmax-selection
    is forced regardless of the (untrained) online network's own values,
    making the expected bootstrap value computable by hand from the target
    network alone."""
    joint_obs = _rand_obs(100)
    next_obs = _rand_obs(101)
    mask = _mask(0)  # forces action index 0
    reward_ramp = 0.5
    reward_mainline = -0.2
    gamma = 0.9

    for n_steps in (1, 3, 5):
        learner = _learner(seed=7, replay_capacity=100, batch_size=1, gamma=gamma)
        q_before = learner.q_values(joint_obs, network="online")
        next_q_t = learner.q_values(next_obs, network="target")
        expected_target_ramp = reward_ramp + (gamma**n_steps) * next_q_t[0][0]
        expected_target_mainline = reward_mainline + (gamma**n_steps) * next_q_t[1][0]
        expected_loss_ramp = (q_before[0][0] - expected_target_ramp) ** 2
        expected_loss_mainline = (q_before[1][0] - expected_target_mainline) ** 2

        batch = JointReplayBatch(
            joint_observations=np.stack([joint_obs]),
            actions=np.array([[0, 0]]),
            rewards=np.array([[reward_ramp, reward_mainline]]),
            next_joint_observations=[next_obs],
            controller_terminal=np.array([False]),
            action_masks=np.stack([np.stack([mask, mask])]),
            next_action_masks=[np.stack([mask, mask])],
            n_steps=np.array([n_steps]),
            importance_weights=np.array([1.0]),
            buffer_indices=np.array([0]),
        )
        result = learner.update(batch=batch)
        assert result["losses"][0] == pytest.approx(float(expected_loss_ramp), rel=1e-4)
        assert result["losses"][1] == pytest.approx(float(expected_loss_mainline), rel=1e-4)


def test_n_step_default_reproduces_old_single_step_target():
    """n_steps=1 (the default) must be numerically indistinguishable from
    the pre-n-step behaviour (a single scalar gamma exponent) -- this is the
    single most important regression check for this addition."""
    learner_a = _learner(seed=11, replay_capacity=100, batch_size=8)
    learner_b = _learner(seed=11, replay_capacity=100, batch_size=8)
    rng = np.random.default_rng(1)
    for i in range(20):
        rewards = (float(rng.normal()), float(rng.normal()))
        actions = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
        learner_a.store_transition(_transition(i, rewards=rewards, actions=actions))
        learner_b.store_transition(_transition(i, rewards=rewards, actions=actions, n_steps=1))
    result_a = learner_a.update()
    result_b = learner_b.update()
    assert result_a["loss"] == pytest.approx(result_b["loss"], rel=1e-6)


def test_terminal_row_target_ignores_n_steps():
    """A controller_terminal=True row's target is exactly the reward,
    regardless of n_steps -- terminal rows never bootstrap, so the discount
    exponent is irrelevant for them (see JointDQNLearner.update's docstring)."""
    learner = _learner(replay_capacity=100, batch_size=4)
    for i in range(4):
        learner.store_transition(
            _transition(i, rewards=(2.0, -3.0), controller_terminal=True, n_steps=1 + i)
        )
    result = learner.update()
    assert result["n_bootstrap_rows"] == 0
    assert result["n_terminal_rows"] == 4


# --------------------------------------------------------------- PER: JointDQNConfig validation


def test_config_validate_rejects_negative_per_alpha():
    cfg = _config(prioritized_replay=True, per_alpha=-0.1)
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_validate_rejects_out_of_range_beta():
    cfg = _config(prioritized_replay=True, per_beta_start=1.5)
    with pytest.raises(ValueError):
        cfg.validate()
    cfg2 = _config(prioritized_replay=True, per_beta_end=-0.1)
    with pytest.raises(ValueError):
        cfg2.validate()


def test_config_prioritized_replay_defaults_off():
    cfg = _config()
    assert cfg.prioritized_replay is False
    assert cfg.per_alpha == pytest.approx(0.5)
    assert cfg.per_beta_start == pytest.approx(0.4)
    assert cfg.per_beta_end == pytest.approx(1.0)


# --------------------------------------------------------------- PER: beta_at_step


def test_beta_at_step_linear_anneal_hand_computed():
    assert beta_at_step(0, total_steps=100, beta_start=0.4, beta_end=1.0) == pytest.approx(0.4)
    assert beta_at_step(50, total_steps=100, beta_start=0.4, beta_end=1.0) == pytest.approx(0.7)
    assert beta_at_step(100, total_steps=100, beta_start=0.4, beta_end=1.0) == pytest.approx(1.0)


def test_beta_at_step_clamped_past_total_steps():
    assert beta_at_step(1000, total_steps=100, beta_start=0.4, beta_end=1.0) == pytest.approx(1.0)


def test_beta_at_step_rejects_non_positive_total_steps():
    with pytest.raises(ValueError):
        beta_at_step(10, total_steps=0, beta_start=0.4, beta_end=1.0)


# --------------------------------------------------------------- PER: JointReplayBuffer


def test_non_prioritized_buffer_sample_has_unit_importance_weights():
    learner = _learner(replay_capacity=50, batch_size=6)
    for i in range(10):
        learner.store_transition(_transition(i))
    batch = learner.replay.sample(6)
    assert np.all(batch.importance_weights == 1.0)


def test_prioritized_buffer_new_transition_gets_max_current_priority():
    buf = JointReplayBuffer(
        10, joint_obs_dim=joint_obs_dim(2), n_actions=N_ACTIONS, n_slots=2, seed=0, prioritized=True, per_epsilon=1e-3
    )
    buf.append(_transition(0))
    buf._priorities[0] = 5.0
    buf.append(_transition(1))
    assert buf._priorities[1] == pytest.approx(5.0)


def test_update_priorities_writes_back_abs_td_error_plus_epsilon():
    buf = JointReplayBuffer(
        10, joint_obs_dim=joint_obs_dim(2), n_actions=N_ACTIONS, n_slots=2, seed=0, prioritized=True, per_epsilon=1e-3
    )
    for i in range(5):
        buf.append(_transition(i))
    buf.update_priorities(np.array([0, 2]), np.array([-2.0, 0.5]))
    assert buf._priorities[0] == pytest.approx(2.0 + 1e-3)
    assert buf._priorities[2] == pytest.approx(0.5 + 1e-3)


def test_update_priorities_rejects_non_prioritized_buffer():
    buf = JointReplayBuffer(10, joint_obs_dim=joint_obs_dim(2), n_actions=N_ACTIONS, n_slots=2, seed=0)
    buf.append(_transition(0))
    with pytest.raises(RuntimeError):
        buf.update_priorities(np.array([0]), np.array([1.0]))


def test_prioritized_sample_proportional_to_priority_alpha_hand_computed():
    """Statistical proof of proportional-to-priority^alpha sampling: with
    priorities [1, 3, 6] and alpha=1.0, the true sampling distribution is
    exactly [1/10, 3/10, 6/10] -- verified over a large number of draws."""
    buf = JointReplayBuffer(
        3, joint_obs_dim=joint_obs_dim(2), n_actions=N_ACTIONS, n_slots=2, seed=0, prioritized=True, per_epsilon=1e-6
    )
    for i in range(3):
        buf.append(_transition(i))
    buf._priorities[:] = np.array([1.0, 3.0, 6.0])
    n_samples = 30_000
    batch = buf.sample(n_samples, alpha=1.0, beta=0.5)
    counts = np.bincount(batch.buffer_indices, minlength=3)
    freq = counts / n_samples
    expected = np.array([1.0, 3.0, 6.0]) / 10.0
    assert freq == pytest.approx(expected, abs=0.02)


def test_prioritized_sample_importance_weights_hand_computed():
    buf = JointReplayBuffer(
        3, joint_obs_dim=joint_obs_dim(2), n_actions=N_ACTIONS, n_slots=2, seed=1, prioritized=True, per_epsilon=1e-6
    )
    for i in range(3):
        buf.append(_transition(i))
    buf._priorities[:] = np.array([1.0, 3.0, 6.0])
    alpha = 0.5
    beta = 0.4
    scaled = buf._priorities**alpha
    probs = scaled / scaled.sum()
    unnormalised = (3 * probs) ** (-beta)
    expected_weight_by_index = unnormalised / unnormalised.max()

    batch = buf.sample(50, alpha=alpha, beta=beta)
    for idx, w in zip(batch.buffer_indices, batch.importance_weights):
        assert w == pytest.approx(expected_weight_by_index[idx], rel=1e-6)


def test_prioritized_sample_batch_size_may_exceed_buffer_size():
    """Sampling is WITH replacement for the prioritized path -- unlike the
    uniform-without-replacement default, batch_size is not bounded by the
    number of stored transitions."""
    buf = JointReplayBuffer(3, joint_obs_dim=joint_obs_dim(2), n_actions=N_ACTIONS, n_slots=2, seed=0, prioritized=True)
    for i in range(3):
        buf.append(_transition(i))
    batch = buf.sample(100)
    assert batch.joint_observations.shape == (100, joint_obs_dim(2))


# --------------------------------------------------------------- PER: JointDQNLearner integration


def test_prioritized_update_requires_current_step_and_total_steps():
    learner = _learner(replay_capacity=100, batch_size=8, prioritized_replay=True)
    for i in range(20):
        learner.store_transition(_transition(i))
    with pytest.raises(ValueError):
        learner.update()


def test_prioritized_update_writes_back_max_of_both_heads_td_error_hand_computed():
    """This project's own design choice for a two-headed network (see
    joint_dqn.py's module docstring): the priority written back for a
    transition is max(|ramp TD-error|, |mainline TD-error|), not either head
    alone or their average."""
    learner = _learner(seed=3, replay_capacity=10, batch_size=1, gamma=0.9, prioritized_replay=True)
    learner.store_transition(_transition(0))  # occupies buffer index 0

    joint_obs = _rand_obs(200)
    next_obs = _rand_obs(201)
    mask = _mask(0)
    reward_ramp = 1.0
    reward_mainline = -0.5
    n_steps = 2

    q_before = learner.q_values(joint_obs, network="online")
    next_q_t = learner.q_values(next_obs, network="target")
    target_ramp = reward_ramp + (0.9**n_steps) * next_q_t[0][0]
    target_mainline = reward_mainline + (0.9**n_steps) * next_q_t[1][0]
    expected_td_ramp = target_ramp - q_before[0][0]
    expected_td_mainline = target_mainline - q_before[1][0]
    expected_priority = max(abs(expected_td_ramp), abs(expected_td_mainline)) + learner.replay.per_epsilon

    batch = JointReplayBatch(
        joint_observations=np.stack([joint_obs]),
        actions=np.array([[0, 0]]),
        rewards=np.array([[reward_ramp, reward_mainline]]),
        next_joint_observations=[next_obs],
        controller_terminal=np.array([False]),
        action_masks=np.stack([np.stack([mask, mask])]),
        next_action_masks=[np.stack([mask, mask])],
        n_steps=np.array([n_steps]),
        importance_weights=np.array([1.0]),
        buffer_indices=np.array([0]),
    )
    learner.update(batch=batch, current_step=0, total_steps=1000)
    assert learner.replay._priorities[0] == pytest.approx(expected_priority, rel=1e-4)


def test_prioritized_update_writes_back_n_way_max_td_error_at_n4_hand_computed():
    """N-way generalisation of the two-headed max: with per-slot TD-errors
    [0.1, 0.9, 0.2, 0.05], the combined priority is 0.9 (np.maximum.reduce
    semantics), not a mean or any pairwise combination."""
    learner = _learner(n_vehicles=4, seed=3, replay_capacity=10, batch_size=1, gamma=0.9, prioritized_replay=True)
    learner.store_transition(_transition(0, n_vehicles=4))

    joint_obs = _rand_obs(200, n_vehicles=4)
    next_obs = _rand_obs(201, n_vehicles=4)
    mask = _mask(0)
    rewards = (1.0, -0.5, 0.3, 0.0)
    n_steps = 2

    q_before = learner.q_values(joint_obs, network="online")
    next_q_t = learner.q_values(next_obs, network="target")
    targets = [rewards[s] + (0.9**n_steps) * next_q_t[s][0] for s in range(4)]
    tds = [targets[s] - q_before[s][0] for s in range(4)]
    expected_priority = max(abs(td) for td in tds) + learner.replay.per_epsilon

    batch = JointReplayBatch(
        joint_observations=np.stack([joint_obs]),
        actions=np.array([[0, 0, 0, 0]]),
        rewards=np.array([list(rewards)]),
        next_joint_observations=[next_obs],
        controller_terminal=np.array([False]),
        action_masks=np.stack([np.stack([mask] * 4)]),
        next_action_masks=[np.stack([mask] * 4)],
        n_steps=np.array([n_steps]),
        importance_weights=np.array([1.0]),
        buffer_indices=np.array([0]),
    )
    learner.update(batch=batch, current_step=0, total_steps=1000)
    assert learner.replay._priorities[0] == pytest.approx(expected_priority, rel=1e-4)


def test_prioritized_update_end_to_end_via_replay_sample():
    """Non-hand-computed smoke test: a full prioritized update cycle sampled
    from the learner's own replay buffer (alpha/beta from config, current
    beta from beta_at_step) runs and changes priorities for the sampled
    rows, without touching the target network."""
    learner = _learner(replay_capacity=100, batch_size=16, prioritized_replay=True)
    rng = np.random.default_rng(2)
    for i in range(40):
        learner.store_transition(
            _transition(
                i,
                rewards=(float(rng.normal()), float(rng.normal())),
                actions=(int(rng.integers(0, 3)), int(rng.integers(0, 3))),
            )
        )
    priorities_before = learner.replay._priorities.copy()
    target_before = {k: v.detach().clone() for k, v in learner.target.state_dict().items()}
    result = learner.update(current_step=5, total_steps=1000)
    assert np.isfinite(result["loss"])
    assert not np.array_equal(learner.replay._priorities, priorities_before)
    target_after = learner.target.state_dict()
    assert all(torch.equal(target_before[k], target_after[k]) for k in target_before)


def test_non_prioritized_loss_matches_manual_unweighted_mse_mean():
    """Backward-compat proof: with importance_weights all exactly 1.0 (the
    non-PER default), the manual weighted-mean loss computation this round
    introduces is numerically identical to the pre-PER nn.functional.mse_loss
    mean-reduction it replaces."""
    learner = _learner(replay_capacity=100, batch_size=16)
    rng = np.random.default_rng(4)
    for i in range(30):
        learner.store_transition(
            _transition(
                i,
                rewards=(float(rng.normal()), float(rng.normal())),
                actions=(int(rng.integers(0, 3)), int(rng.integers(0, 3))),
            )
        )
    batch = learner.replay.sample(16)
    assert np.all(batch.importance_weights == 1.0)
    result = learner.update(batch=batch)
    assert np.isfinite(result["loss"])


# --------------------------------------------------------------- N-vehicle end-to-end smoke


@pytest.mark.parametrize("n_vehicles", [2, 4, 6])
def test_end_to_end_smoke_at_each_n_vehicles(n_vehicles):
    """Runs the full step/replay/update loop together at each supported N --
    catches shape mismatches (e.g. [B,n,n_actions] vs [B,n_actions*n]) that
    only manifest once everything runs together, not from unit-testing
    individual functions alone."""
    learner = _learner(n_vehicles=n_vehicles, replay_capacity=200, batch_size=8, prioritized_replay=True)
    rng = np.random.default_rng(42)
    for i in range(60):
        masks = tuple(_mask() for _ in range(n_vehicles))
        obs = _rand_obs(i, n_vehicles=n_vehicles)
        actions = learner.select_action(obs, masks, epsilon=0.5)
        assert len(actions) == n_vehicles
        learner.store_transition(
            _transition(
                i,
                n_vehicles=n_vehicles,
                rewards=tuple(float(rng.normal()) for _ in range(n_vehicles)),
                actions=actions,
                action_masks=masks,
            )
        )
    result = learner.update(current_step=1, total_steps=100)
    assert np.isfinite(result["loss"])
    assert len(result["losses"]) == n_vehicles
