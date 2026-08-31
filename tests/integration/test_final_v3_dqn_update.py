"""Stage 5A-0 — isolated 27D Independent DQN updates."""

from __future__ import annotations

import numpy as np
import torch

from thesis.training.final_experiment_runtime import (
    collect_mixed_batch_transitions,
    isolated_dqn_update,
)
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.final_v3_pipeline import build_integration_learners


def test_mixed_batch_isolated_update_finite_and_bootstrap_only():
    bundle = load_final_locks()
    mixed = collect_mixed_batch_transitions(bundle, condition="baseline")
    assert any(t.controller_terminal for t in mixed)
    assert any((not t.controller_terminal) and (not t.truncated) for t in mixed)
    assert any(t.truncated and (not t.controller_terminal) for t in mixed)
    for t in mixed:
        assert t.observation.shape == (27,)
        if t.controller_terminal:
            assert t.next_observation is None
            assert t.next_action_mask is None

    learners = build_integration_learners(seed_A=7, seed_B=8)
    stats = isolated_dqn_update(learners["A"], mixed)
    assert stats["obs_shape"] == [len(mixed), 27]
    assert stats["q_shape"] == [len(mixed), 3]
    assert stats["finite_loss"] and stats["finite_q"]
    assert stats["online_param_changed"] is True
    assert stats["target_unchanged_without_sync"] is True
    assert stats["target_network_forward_calls"] == 1
    assert stats["n_bootstrap_rows"] >= 1
    assert stats["n_terminal_rows"] >= 1
    assert stats["policy_training_started"] is False
    assert stats["sustained_training_invoked"] is False


def test_isolated_update_determinism():
    bundle = load_final_locks()
    mixed = collect_mixed_batch_transitions(bundle, condition="mean_pbrs")
    l1 = build_integration_learners(reward_condition="mean_pbrs", seed_A=3, seed_B=4)["A"]
    l2 = build_integration_learners(reward_condition="mean_pbrs", seed_A=3, seed_B=4)["A"]
    s1 = isolated_dqn_update(l1, mixed)
    s2 = isolated_dqn_update(l2, mixed)
    assert s1["loss"] == s2["loss"]
    v1 = torch.nn.utils.parameters_to_vector(l1.online.parameters()).detach()
    v2 = torch.nn.utils.parameters_to_vector(l2.online.parameters()).detach()
    assert torch.allclose(v1, v2, atol=0.0, rtol=0.0)


def test_no_nan_in_update_tensors():
    bundle = load_final_locks()
    mixed = collect_mixed_batch_transitions(bundle, condition="min_pbrs")
    learner = build_integration_learners(reward_condition="min_pbrs", seed_A=1, seed_B=2)["B"]
    stats = isolated_dqn_update(learner, mixed)
    assert np.isfinite(stats["loss"])
    assert np.isfinite(stats["max_abs_q"])
