"""Terminal / truncation bootstrap semantics for Stage 7B-A1."""

from __future__ import annotations

import numpy as np

from thesis.agents.dqn_bootstrap import DQNTargetMode
from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
from thesis.agents.replay_buffer_v2 import ReplayTransition


def _learner(mode: DQNTargetMode) -> IndependentDQNLearner:
    cfg = DQNConfig(
        obs_dim=4,
        n_actions=3,
        hidden_sizes=(8,),
        batch_size=4,
        replay_capacity=32,
        target_mode=mode,
        gamma=0.5,
    )
    return IndependentDQNLearner("A", cfg, seed=3, replay_seed=4)


def test_terminal_rows_do_not_bootstrap():
    for mode in (DQNTargetMode.VANILLA, DQNTargetMode.DOUBLE):
        learner = _learner(mode)
        for i in range(8):
            learner.store_transition(
                ReplayTransition(
                    observation=np.zeros(4),
                    action=0,
                    shaped_reward=2.0,
                    next_observation=None,
                    terminated=True,
                    truncated=False,
                    controller_terminal=True,
                    learner_completed=True,
                    action_mask=np.array([True, True, True]),
                    next_action_mask=None,
                    base_reward=2.0,
                    shaping_component=0.0,
                    reward_condition="baseline",
                    episode_id="t",
                    step=i,
                    controller_id="A",
                    traffic_role="mainline",
                )
            )
        stats = learner.update()
        assert stats["n_bootstrap_rows"] == 0
        assert stats["n_terminal_rows"] == 4


def test_truncation_bootstraps():
    for mode in (DQNTargetMode.VANILLA, DQNTargetMode.DOUBLE):
        learner = _learner(mode)
        for i in range(8):
            learner.store_transition(
                ReplayTransition(
                    observation=np.zeros(4),
                    action=1,
                    shaped_reward=1.0,
                    next_observation=np.ones(4),
                    terminated=False,
                    truncated=True,
                    controller_terminal=False,
                    learner_completed=False,
                    action_mask=np.array([True, True, True]),
                    next_action_mask=np.array([True, True, True]),
                    base_reward=1.0,
                    shaping_component=0.0,
                    reward_condition="baseline",
                    episode_id="u",
                    step=i,
                    controller_id="A",
                    traffic_role="ramp",
                )
            )
        stats = learner.update()
        assert stats["n_bootstrap_rows"] == 4
