"""Explicit replay seed injection for formal runtime."""

from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
from thesis.formal.formal_config import derive_formal_job_seeds


def test_explicit_replay_seeds_not_learner_plus_17():
    seeds = derive_formal_job_seeds(61001)
    cfg = DQNConfig(obs_dim=27, n_actions=3, hidden_sizes=(64, 64))
    a = IndependentDQNLearner(
        "A", cfg, seed=seeds["learner_A_seed"], replay_seed=seeds["replay_A_seed"]
    )
    b = IndependentDQNLearner(
        "B", cfg, seed=seeds["learner_B_seed"], replay_seed=seeds["replay_B_seed"]
    )
    assert a.replay.seed == seeds["replay_A_seed"] == 61001 + 300_000
    assert b.replay.seed == seeds["replay_B_seed"] == 61001 + 400_000
    assert a.replay.seed != a.seed + 17
    assert b.replay.seed != b.seed + 17


def test_historical_default_still_learner_plus_17():
    cfg = DQNConfig(obs_dim=27, n_actions=3, hidden_sizes=(64, 64))
    learner = IndependentDQNLearner("A", cfg, seed=1000)
    assert learner.replay.seed == 1017
